import fs from 'node:fs';
import path from 'node:path';
import { SettingsManager } from './settings_manager.js';
import { LocalModelService } from './local_model_service.js';
import {
  buildTranslationGlossaryLocaleChain,
  buildTargetLanguageDescriptor as buildResolvedTargetLanguageDescriptor,
  getTranslationStopwords,
  isSimplifiedChineseTarget as isResolvedSimplifiedChineseTarget,
  isTraditionalChineseTarget as isResolvedTraditionalChineseTarget,
  normalizeLanguageKey,
  normalizeTargetLanguageEnglishName,
  normalizeTargetLanguageNativeName,
} from '../language/resolver.js';
import { getLocalModelInstallDir, LocalModelDefinition } from '../local_model_catalog.js';
import { OpenvinoRuntimeManager, type OpenvinoTranslateRuntimeDebug } from '../openvino_runtime_manager.js';
import { PathManager } from '../path_manager.js';
import {
  type CloudTranslateProvider,
  redactUrlSecrets,
  resolveCloudTranslateProvider,
  supportsCloudContextStrategy,
} from './cloud_translate_provider.js';
import { ChineseScriptNormalizer } from './chinese_script_normalizer.js';
import {
  getCloudTranslateAdapter,
  type CloudTranslateAdapterDeps,
} from './cloud_translate_adapter.js';
import {
  runCloudTranslationOrchestrator,
  type CloudTranslationOrchestratorDeps,
  type CloudTranslationOrchestratorInput,
  type CloudTranslationOrchestratorResult,
  type CloudTranslationBatchDebugInfo,
  type CloudTranslationProviderBatchConfig,
} from './llm/orchestrators/cloud_translation_orchestrator.js';
import { runCloudTranslationConnectionProbe } from './llm/orchestrators/cloud_translation_connection_probe.js';
import { inferLocalTranslateModelStrategy as inferLocalTranslateModelStrategyModule } from './local_llm/strategy.js';
import {
  buildLocalGenerationOptions as buildLocalGenerationOptionsModule,
  estimateLocalMaxNewTokens as estimateLocalMaxNewTokensModule,
} from './local_llm/generation.js';
import {
  resolveEffectiveLocalTranslationProfile,
  resolveLocalTranslationIntent,
} from './local_llm/effective_profile.js';
import {
  buildLocalJsonRepairPrompt as buildLocalJsonRepairPromptModule,
  buildLocalTranslationPrompt as buildLocalTranslationPromptModule,
} from './local_llm/prompting.js';
import { buildTranslateGemmaMessages, inferTranslateGemmaSourceLanguageCode } from './local_llm/translategemma.js';
import { resolveLocalOpenvinoProfile } from './local_llm/profiles.js';
import {
  runLocalTranslationOrchestrator,
  type RunLocalTranslationOrchestratorDeps,
} from './local_llm/orchestrators/local_translation_orchestrator.js';
import type { LocalTranslateModelStrategy, LocalTranslateStructuredMessage } from './local_llm/types.js';
import {
  isPlainTranslationProbeMode,
  resolveTranslationQualityMode,
  usesTemplateValidatedQualityChecks,
  type TranslationQualityMode,
} from './llm/orchestrators/translation_quality_policy.js';
import type { ResolvedCloudTranslateProvider } from './cloud_translate_provider.js';
import type { RunIssue } from '../../shared/run_monitor.js';
import type { ApiModelRequestOptions } from '../../src/types.js';

type TranslateProvider = CloudTranslateProvider | 'openvino-local';

type PromptTemplateId =
  | ''
  | 'subtitle_general'
  | 'subtitle_strict_alignment'
  | 'subtitle_concise_spoken'
  | 'subtitle_formal_precise'
  | 'subtitle_asr_recovery'
  | 'subtitle_technical_terms';

type TranslateProgressFn = (message: string) => void;

interface TranslateRequestOptions {
  text: string;
  targetLang: string;
  sourceLang?: string;
  glossary?: string;
  prompt?: string;
  promptTemplateId?: PromptTemplateId | string;
  key?: string;
  model?: string;
  modelOptions?: ApiModelRequestOptions;
  isConnectionTest?: boolean;
  lineSafeMode?: boolean;
  systemPromptOverride?: string;
  disableSystemPrompt?: boolean;
  jsonResponse?: boolean;
  signal?: AbortSignal;
}

interface TranslateProviderResult {
  text: string;
  meta: {
    endpointUrl: string;
    fallbackUsed: boolean;
    fallbackType: string | null;
    requestWarnings?: string[];
  };
}

export interface LocalTranslationBatchDebugInfo {
  source: 'local' | 'translategemma';
  mode: 'fixed_lines' | 'token_aware';
  batchCount: number;
  lineCounts: number[];
  charCounts: number[];
  promptTokens?: number[];
  estimatedOutputTokens?: number[];
  durationsMs?: number[];
  inputTokenBudget?: number | null;
  outputTokenBudget?: number | null;
  contextWindow?: number | null;
  safetyReserveTokens?: number | null;
  maxLines?: number | null;
  charBudget?: number | null;
  fallbackReason?: string | null;
  totalDurationMs?: number | null;
  maxDurationMs?: number | null;
}

export interface TranslationDebugInfo {
  requested: {
    sourceLang?: string;
    sourceLanguageDescriptor?: string | null;
    targetLang: string;
    targetLanguageDescriptor: string;
    lineCount: number;
    charCount: number;
    hasGlossary: boolean;
    effectiveGlossary: string | null;
    hasPrompt: boolean;
    promptTemplateId: string | null;
    jsonLineRepairEnabled: boolean;
    sourceHasStructuredPrefixes: boolean;
    sourceHasSpeakerTags: boolean;
    sourceSpeakerTaggedLineCount: number;
  };
  provider: {
    name: TranslateProvider;
    modelId?: string | null;
    model: string;
    endpoint: string;
    adapterKey?: string | null;
    profileId?: string | null;
    profileFamily?: string | null;
  };
  runtime?: OpenvinoTranslateRuntimeDebug | null;
  applied: {
    retryCount: number;
    fallback: boolean;
    fallbackType: string | null;
    translationQualityMode?: TranslationQualityMode;
    qualityRetryCount?: number;
    strictRetrySucceeded?: boolean;
    relaxedWholeRequest?: boolean;
    relaxedWholeRequestFallback?: boolean;
    cloudStrategy?: 'plain' | 'forced_alignment' | 'context_window' | 'provider_batch';
    cloudContextChunkCount?: number;
    cloudContextFallbackCount?: number;
    cloudBatching?: CloudTranslationBatchDebugInfo | null;
    localModelFamily?: string;
    localModelProfileId?: string | null;
    localPromptStyle?: string;
    localGenerationStyle?: string;
    localBypassChecks?: boolean;
    localPromptContract?: string;
    localAlignmentContract?: string;
    localRuntimeHintsApplied?: boolean;
    localBaselineConfidence?: string | null;
    localBaselineTaskFamily?: string | null;
    localFallbackBaseline?: boolean;
    localBatching?: LocalTranslationBatchDebugInfo | null;
    localBatchingMode?: string | null;
    localBatchCount?: number | null;
    localBatchLineCounts?: number[];
    localBatchPromptTokens?: number[];
  };
  quality?: {
    lineCountMatch: boolean;
    targetLanguageMatch: boolean;
    passThroughRisk: 'low' | 'medium' | 'high';
    repetitionRisk: 'low' | 'medium' | 'high';
    markerPreservation?: 'ok' | 'partial' | 'lost';
    strictRetryTriggered?: boolean;
  } | null;
  stats: {
    outputLineCount: number;
    outputCharCount: number;
  };
  timing?: {
    elapsedMs?: number | null;
    elapsedSec?: number | null;
    providerMs?: number | null;
    providerSec?: number | null;
    repairMs?: number | null;
    repairSec?: number | null;
    qualityRetryMs?: number | null;
    qualityRetrySec?: number | null;
  } | null;
  diagnostics?: {
    qualityIssueCodes?: string[];
    runtimeSource?: 'cloud' | 'local';
  } | null;
  warnings: string[];
  warningIssues?: RunIssue[];
  errors: {
    request: string | null;
  };
  errorIssues?: RunIssue[];
  artifacts?: {
    hasTimecodes: boolean;
    translationAssetNames?: string[];
  } | null;
}

export interface TranslationResult {
  translatedText: string;
  debug: TranslationDebugInfo;
}

interface LineSafeUnit {
  index: number;
  marker: string;
  prefix: string;
  content: string;
  speakerTag: string | null;
}

interface JsonLineRepairResult {
  text: string;
  missingCount: number;
  warnings?: string[];
}

interface CloudContextChunk {
  start: number;
  length: number;
}

interface CloudContextConfig {
  enabled: boolean;
  targetLines: number;
  minTargetLines: number;
  contextWindow: number;
  charBudget: number;
  maxSplitDepth: number;
}

interface LocalizedSystemPromptProfile {
  additionalInstructionsLabel: string;
  glossaryLabel: string;
  inputLabel: string;
  inputJsonLabel: string;
  baseIntro: string;
  baseRules: string[];
  lineSafeRules: string[];
  baseReturn: string;
  strictRetryLines: string[];
  jsonIntro: string;
  jsonRules: string[];
  jsonReturn: string;
  jsonNoExtras: string;
  cloudContextIntro: string;
  cloudContextRules: string[];
}

type TranslationQualityIssueCode =
  | 'empty_output'
  | 'repetition_loop'
  | 'adjacent_duplicate'
  | 'pass_through'
  | 'zh_tw_naturalization_needed'
  | 'target_lang_mismatch'
  | 'line_count_loss'
  | 'marker_loss';

interface SourceTextProfile {
  latinCount: number;
  hanCount: number;
  kanaCount: number;
  hangulCount: number;
  likelyChinese: boolean;
  likelyJapanese: boolean;
  likelyKorean: boolean;
  likelyEastAsian: boolean;
}

interface TranslationQualityAssessmentContext {
  expectedLineCount?: number;
  requireMarkers?: boolean;
  sourceProfile?: SourceTextProfile;
}

class ProviderHttpError extends Error {
  status: number;
  detail: string;
  retryAfterMs: number | null;

  constructor(prefix: string, status: number, detail: string, retryAfterMs: number | null = null) {
    super(`${prefix} (${status}): ${detail}`);
    this.status = status;
    this.detail = detail;
    this.retryAfterMs = Number.isFinite(retryAfterMs as number) ? Number(retryAfterMs) : null;
  }
}

export class TranslationService {
  private static readonly builtInGlossaryCache = new Map<string, string[]>();

  private static throwIfAborted(signal?: AbortSignal) {
    if (!signal?.aborted) return;
    throw new Error('Translation request aborted.');
  }

  private static getEnvNumber(name: string, fallback: number, min?: number, max?: number) {
    const raw = process.env[name];
    if (typeof raw !== 'string' || !raw.trim()) return fallback;
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) return fallback;
    if (typeof min === 'number' && parsed < min) return min;
    if (typeof max === 'number' && parsed > max) return max;
    return parsed;
  }

  private static getEnvBoolean(name: string, fallback = false) {
    const raw = String(process.env[name] || '').trim();
    if (!raw) return fallback;
    if (/^(1|true|yes|on)$/i.test(raw)) return true;
    if (/^(0|false|no|off)$/i.test(raw)) return false;
    return fallback;
  }

  private static sleep(ms: number, signal?: AbortSignal) {
    if (!signal) {
      return new Promise((resolve) => setTimeout(resolve, ms));
    }
    return new Promise<void>((resolve, reject) => {
      if (signal.aborted) {
        reject(new Error('Translation request aborted.'));
        return;
      }
      const timeout = setTimeout(() => {
        signal.removeEventListener('abort', onAbort);
        resolve();
      }, ms);
      const onAbort = () => {
        clearTimeout(timeout);
        signal.removeEventListener('abort', onAbort);
        reject(new Error('Translation request aborted.'));
      };
      signal.addEventListener('abort', onAbort, { once: true });
    });
  }

  private static parseRetryAfterMs(response: Response) {
    const raw = String(response.headers.get('retry-after') || '').trim();
    if (!raw) return null;

    const seconds = Number(raw);
    if (Number.isFinite(seconds) && seconds >= 0) {
      return Math.round(seconds * 1000);
    }

    const retryAt = Date.parse(raw);
    if (!Number.isFinite(retryAt)) return null;

    const delta = retryAt - Date.now();
    return delta > 0 ? delta : null;
  }

  private static normalizeTargetLanguage(targetLang: string) {
    return normalizeTargetLanguageEnglishName(targetLang);
  }

  private static normalizeTargetLanguageNative(targetLang: string) {
    return normalizeTargetLanguageNativeName(targetLang);
  }

  private static buildTargetLanguageDescriptor(targetLang: string) {
    return buildResolvedTargetLanguageDescriptor(targetLang);
  }

  private static buildSourceLanguageDescriptor(sourceLang?: string) {
    const normalized = normalizeLanguageKey(sourceLang);
    if (!normalized) return null;
    return buildResolvedTargetLanguageDescriptor(normalized);
  }

  private static getLanguageStopwords(targetLang: string) {
    return getTranslationStopwords(targetLang);
  }

  private static countStopwordHits(words: string[], stopwords: Set<string>) {
    let hits = 0;
    for (const word of words) {
      if (stopwords.has(word)) hits += 1;
    }
    return hits;
  }

  private static isTraditionalChineseTarget(targetLang: string) {
    return isResolvedTraditionalChineseTarget(targetLang);
  }

  private static isSimplifiedChineseTarget(targetLang: string) {
    return isResolvedSimplifiedChineseTarget(targetLang);
  }

  private static getTargetLanguageInstructionLines(targetLang: string) {
    const normalized = normalizeLanguageKey(targetLang);
    if (normalized === 'zh-tw') {
      return [
        'Use Traditional Chinese characters only. Never output Simplified Chinese characters.',
        'Prefer Taiwan Traditional Chinese wording.',
        'Do not leave translated lines in English unless the content is a proper noun, brand, or code identifier.',
        'Prefer terms such as: 程式碼, 軟體, 影片, 支援, 預設, 產生, 設定, 載入, 專案.',
      ];
    }
    if (normalized === 'zh-cn') {
      return [
        'Use Simplified Chinese characters only. Never output Traditional Chinese characters.',
        'Prefer Mainland Chinese wording.',
        'Do not leave translated lines in English unless the content is a proper noun, brand, or code identifier.',
      ];
    }
    if (normalized === 'ja' || normalized === 'jp') {
      return [
        'Use natural Japanese output.',
        'Do not leave translated lines in English unless the content is a proper noun, brand, or code identifier.',
      ];
    }
    if (normalized === 'ko' || normalized === 'kr') {
      return [
        'Use natural Korean output.',
        'Do not leave translated lines in English unless the content is a proper noun, brand, or code identifier.',
      ];
    }
    if (normalized === 'en' || normalized === 'english') {
      return [
        'Use natural English output.',
        'Do not leave translated lines in the original non-English language unless the content is a proper noun, brand, code identifier, or unavoidable quoted term.',
      ];
    }
    if (['fi', 'es', 'de', 'pt', 'it', 'fr', 'ru', 'pl', 'ar', 'nl', 'el', 'fa', 'hu'].includes(normalized)) {
      return [
        `Use natural ${this.normalizeTargetLanguage(targetLang)} output.`,
        'Do not keep the source line in English unless the content is a proper noun, brand, or code identifier.',
      ];
    }
    return [];
  }

  private static getTargetLanguageHardeningLines(targetLang: string, lineSafeMode = false, jsonMode = false) {
    const normalized = normalizeLanguageKey(targetLang);
    if (normalized === 'zh-tw') {
      return [
        '最終輸出不得保留日文、韓文、簡體中文或其他來源語言片段；專有名詞與品牌名稱除外。',
        '不要自行加入原文沒有的稱呼、敬語、親暱語氣、語助詞或情緒修飾，例如「親愛的」。',
        '若來源內容本身已是中文，請改寫為自然、流暢、台灣常用的繁體中文，而不是原樣照抄。',
        ...(lineSafeMode ? ['每行都必須完整翻譯，不得漏行，不得把多行併成一行。'] : []),
        ...(jsonMode ? ['每個 text 欄位都必須是自然繁體中文，不得殘留來源語言詞尾或片段。'] : []),
      ];
    }
    return [];
  }

  private static analyzeSourceTextProfile(text: string): SourceTextProfile {
    const plain = this.normalizeComparisonText(text).join(' ');
    const latinCount = (plain.match(/[A-Za-z]/g) || []).length;
    const hanCount = (plain.match(/[\p{Script=Han}]/gu) || []).length;
    const kanaCount = (plain.match(/[\p{Script=Hiragana}\p{Script=Katakana}]/gu) || []).length;
    const hangulCount = (plain.match(/[\p{Script=Hangul}]/gu) || []).length;
    const likelyJapanese = kanaCount >= 1;
    const likelyKorean = hangulCount >= 1;
    const likelyChinese = hanCount >= 2 && !likelyJapanese && !likelyKorean && hanCount >= Math.max(2, latinCount);
    return {
      latinCount,
      hanCount,
      kanaCount,
      hangulCount,
      likelyChinese,
      likelyJapanese,
      likelyKorean,
      likelyEastAsian: likelyChinese || likelyJapanese || likelyKorean,
    };
  }

  private static getSourceAwareTargetLanguageLines(
    targetLang: string,
    sourceText: string | undefined,
    options: { lineSafeMode?: boolean; jsonMode?: boolean } = {}
  ) {
    const normalized = normalizeLanguageKey(targetLang);
    if (normalized !== 'zh-tw') return [] as string[];

    const profile = this.analyzeSourceTextProfile(sourceText || '');
    if (!profile.likelyEastAsian) return [] as string[];

    const lines: string[] = [];

    if (profile.likelyJapanese) {
      lines.push(
        '\u4f86\u6e90\u770b\u8d77\u4f86\u662f\u65e5\u6587\u3002\u4e0d\u8981\u6b98\u7559\u65e5\u6587\u5047\u540d\u3001\u8a9e\u5c3e\u6216\u52a9\u8a5e\uff0c\u4f8b\u5982 \u300c\u3067\u3059\u300d\u3001\u300c\u307e\u3059\u300d\u3001\u300c\u306d\u300d\u3001\u300c\u3088\u300d\u3001\u300c\u304b\u300d\u3001\u300c\u306e\u300d\u3001\u300c\u3093\u3060\u300d\u3002'
      );
    }
    if (profile.likelyKorean) {
      lines.push(
        '\u4f86\u6e90\u770b\u8d77\u4f86\u662f\u97d3\u6587\u3002\u4e0d\u8981\u6b98\u7559\u97d3\u6587\u8a9e\u5c3e\u6216\u52a9\u8a5e\uff0c\u4f8b\u5982 \u300c\uc694\u300d\u3001\u300c\ub2c8\ub2e4\u300d\u3001\u300c\uc2b5\ub2c8\ub2e4\u300d\u3001\u300c\uc778\ub370\u300d\u3001\u300c\ub124\uc694\u300d\u3002'
      );
    }
    if (profile.likelyChinese) {
      lines.push(
        '\u4f86\u6e90\u770b\u8d77\u4f86\u5df2\u7d93\u662f\u4e2d\u6587\u3002\u8acb\u5c07\u5167\u5bb9\u6539\u5beb\u6210\u81ea\u7136\u7684\u53f0\u7063\u7e41\u9ad4\u4e2d\u6587\u5b57\u5e55\uff0c\u4e0d\u8981\u53ea\u662f\u539f\u6a23\u7167\u6284\u3002'
      );
    }
    lines.push(
      '\u4e0d\u8981\u91cd\u8907\u76f8\u9130\u7684\u5b57\u8a5e\u6216\u77ed\u8a9e\uff0c\u4f8b\u5982\u300c\u8981\u8981\u300d\u3001\u300c\u70ba\u4ec0\u9ebc\u70ba\u4ec0\u9ebc\u300d\u3002'
    );

    if (options.lineSafeMode) {
      lines.push(
        '\u6bcf\u4e00\u884c\u90fd\u8981\u5b8c\u6574\u7ffb\u6210\u7e41\u9ad4\u4e2d\u6587\uff0c\u4e0d\u8981\u5728\u4efb\u4f55\u4e00\u884c\u5c3e\u7aef\u6b98\u7559\u539f\u8a9e\u8a5e\u5c3e\u6216\u8a9e\u6c23\u8a5e\u3002'
      );
    }
    if (options.jsonMode) {
      lines.push(
        '\u6bcf\u500b text \u6b04\u4f4d\u90fd\u5fc5\u9808\u662f\u81ea\u7136\u7684\u53f0\u7063\u7e41\u9ad4\u4e2d\u6587\uff0c\u4e0d\u8981\u6b98\u7559\u65e5\u6587\u3001\u97d3\u6587\u6216\u751f\u786c\u7684\u4e2d\u5f0f\u5b57\u9762\u3002'
      );
    }

    return lines;
  }

  private static buildLocalTargetLanguageReinforcement(input: {
    targetLang: string;
    lineSafeMode?: boolean;
    strictMode?: boolean;
    jsonMode?: boolean;
  }) {
    const normalized = normalizeLanguageKey(input.targetLang);
    const rules = (() => {
      switch (normalized) {
        case 'zh-tw':
          return {
            onlyTarget: '\u8acb\u5168\u7a0b\u4ee5\u7e41\u9ad4\u4e2d\u6587\u8f38\u51fa\u6700\u7d42\u7ffb\u8b6f\u3002',
            avoidSource:
              '\u9664\u975e\u662f\u5c08\u6709\u540d\u8a5e\u3001\u54c1\u724c\u3001\u7a0b\u5f0f\u78bc\u8b58\u5225\u5b57\u6216\u7121\u6cd5\u907f\u514d\u7684\u5f15\u7528\uff0c\u4e0d\u8981\u4fdd\u7559\u539f\u8a9e\u3002',
            lineSafe:
              '\u82e5\u8f38\u5165\u5305\u542b [[L00001]] \u9019\u985e\u6a19\u8a18\uff0c\u8acb\u9010\u884c\u5c0d\u9f4a\uff0c\u4e26\u539f\u6a23\u4fdd\u7559\u6a19\u8a18\u3001\u6642\u9593\u78bc\u8207\u524d\u7db4\u3002',
            plain:
              '\u8acb\u53ea\u8f38\u51fa\u6700\u7d42\u8b6f\u6587\uff0c\u4e0d\u8981\u8f38\u51fa\u8aaa\u660e\u3001\u5206\u6790\u3001\u5099\u8a3b\u3001\u5099\u9078\u5167\u5bb9\u6216 <think> \u6a19\u7c64\u3002',
            strict:
              '\u9019\u662f\u56b4\u683c\u6a21\u5f0f\u3002\u82e5\u524d\u4e00\u6b21\u8f38\u51fa\u4ecd\u4fdd\u7559\u539f\u8a9e\uff0c\u9019\u6b21\u5fc5\u9808\u6539\u6210\u7e41\u9ad4\u4e2d\u6587\u3002',
            json:
              '\u53ea\u80fd\u56de\u50b3 JSON\uff0cid \u8207\u9805\u76ee\u6578\u91cf\u5fc5\u9808\u5b8c\u5168\u4e0d\u8b8a\uff0ctext \u5fc5\u9808\u662f\u7e41\u9ad4\u4e2d\u6587\u3002',
          };
        case 'zh-cn':
          return {
            onlyTarget: '\u8bf7\u5168\u7a0b\u4ee5\u7b80\u4f53\u4e2d\u6587\u8f93\u51fa\u6700\u7ec8\u7ffb\u8bd1\u3002',
            avoidSource:
              '\u9664\u975e\u662f\u4e13\u6709\u540d\u8bcd\u3001\u54c1\u724c\u3001\u4ee3\u7801\u6807\u8bc6\u6216\u65e0\u6cd5\u907f\u514d\u7684\u5f15\u7528\uff0c\u4e0d\u8981\u4fdd\u7559\u539f\u8bed\u3002',
            lineSafe:
              '\u82e5\u8f93\u5165\u5305\u542b [[L00001]] \u8fd9\u7c7b\u6807\u8bb0\uff0c\u8bf7\u9010\u884c\u5bf9\u9f50\uff0c\u5e76\u539f\u6837\u4fdd\u7559\u6807\u8bb0\u3001\u65f6\u95f4\u7801\u548c\u524d\u7f00\u3002',
            plain:
              '\u8bf7\u53ea\u8f93\u51fa\u6700\u7ec8\u8bd1\u6587\uff0c\u4e0d\u8981\u8f93\u51fa\u8bf4\u660e\u3001\u5206\u6790\u3001\u5907\u6ce8\u3001\u5907\u9009\u5185\u5bb9\u6216 <think> \u6807\u7b7e\u3002',
            strict:
              '\u8fd9\u662f\u4e25\u683c\u6a21\u5f0f\u3002\u82e5\u4e0a\u4e00\u6b21\u8f93\u51fa\u4ecd\u4fdd\u7559\u539f\u8bed\uff0c\u8fd9\u6b21\u5fc5\u987b\u6539\u6210\u7b80\u4f53\u4e2d\u6587\u3002',
            json:
              '\u53ea\u80fd\u8fd4\u56de JSON\uff0cid \u4e0e\u9879\u76ee\u6570\u91cf\u5fc5\u987b\u5b8c\u5168\u4e0d\u53d8\uff0ctext \u5fc5\u987b\u662f\u7b80\u4f53\u4e2d\u6587\u3002',
          };
        case 'ja':
        case 'jp':
          return {
            onlyTarget: '\u6700\u7d42\u51fa\u529b\u306f\u5fc5\u305a\u65e5\u672c\u8a9e\u3060\u3051\u3067\u8fd4\u3057\u3066\u304f\u3060\u3055\u3044\u3002',
            avoidSource:
              '\u56fa\u6709\u540d\u8a5e\u3001\u30d6\u30e9\u30f3\u30c9\u540d\u3001\u30b3\u30fc\u30c9\u8b58\u5225\u5b50\u3001\u307e\u305f\u306f\u907f\u3051\u3089\u308c\u306a\u3044\u5f15\u7528\u3092\u9664\u304d\u3001\u539f\u8a00\u8a9e\u3092\u6b8b\u3055\u306a\u3044\u3067\u304f\u3060\u3055\u3044\u3002',
            lineSafe:
              '\u5165\u529b\u306b [[L00001]] \u306e\u3088\u3046\u306a\u30de\u30fc\u30ab\u30fc\u304c\u3042\u308b\u5834\u5408\u306f\u3001\u5404\u884c\u3092\u5bfe\u5fdc\u3055\u305b\u3001\u30de\u30fc\u30ab\u30fc\u3001\u30bf\u30a4\u30e0\u30b3\u30fc\u30c9\u3001\u69cb\u9020\u5316\u3055\u308c\u305f\u63a5\u982d\u8f9e\u3092\u305d\u306e\u307e\u307e\u4fdd\u6301\u3057\u3066\u304f\u3060\u3055\u3044\u3002',
            plain:
              '\u6700\u7d42\u8a33\u3060\u3051\u3092\u51fa\u529b\u3057\u3001\u8aac\u660e\u3001\u5206\u6790\u3001\u6ce8\u8a18\u3001\u5019\u88dc\u3001<think> \u30bf\u30b0\u306f\u51fa\u529b\u3057\u306a\u3044\u3067\u304f\u3060\u3055\u3044\u3002',
            strict:
              '\u3053\u308c\u306f\u53b3\u683c\u30e2\u30fc\u30c9\u3067\u3059\u3002\u524d\u56de\u306e\u51fa\u529b\u304c\u539f\u8a00\u8a9e\u306e\u307e\u307e\u3060\u3063\u305f\u5834\u5408\u306f\u3001\u4eca\u56de\u306f\u5fc5\u305a\u65e5\u672c\u8a9e\u306b\u7ffb\u8a33\u3057\u3066\u304f\u3060\u3055\u3044\u3002',
            json:
              'JSON \u306e\u307f\u3092\u8fd4\u3057\u3001id \u3068\u9805\u76ee\u6570\u3092\u5909\u3048\u305a\u3001text \u306f\u5fc5\u305a\u65e5\u672c\u8a9e\u306b\u3057\u3066\u304f\u3060\u3055\u3044\u3002',
          };
        case 'ko':
        case 'kr':
          return {
            onlyTarget: '\ucd5c\uc885 \ucd9c\ub825\uc740 \ubc18\ub4dc\uc2dc \ud55c\uad6d\uc5b4\ub85c\ub9cc \ubc18\ud658\ud574 \uc8fc\uc138\uc694.',
            avoidSource:
              '\uace0\uc720\uba85\uc0ac, \ube0c\ub79c\ub4dc\uba85, \ucf54\ub4dc \uc2dd\ubcc4\uc790, \ub610\ub294 \ud53c\ud560 \uc218 \uc5c6\ub294 \uc778\uc6a9\uc744 \uc81c\uc678\ud558\uace0 \uc6d0\ubb38 \uc5b8\uc5b4\ub97c \ub0a8\uae30\uc9c0 \ub9c8\uc138\uc694.',
            lineSafe:
              '\uc785\ub825\uc5d0 [[L00001]] \uac19\uc740 \ub9c8\ucee4\uac00 \uc788\uc73c\uba74 \uac01 \uc904\uc744 \ub9de\ucdb0 \ubc88\uc5ed\ud558\uace0, \ub9c8\ucee4\u3001\uc2dc\uac04 \ucf54\ub4dc\u3001\uad6c\uc870\uc801 \uc811\ub450\uc5b4\ub97c \uadf8\ub300\ub85c \uc720\uc9c0\ud574 \uc8fc\uc138\uc694.',
            plain:
              '\ucd5c\uc885 \ubc88\uc5ed\ubb38\ub9cc \ucd9c\ub825\ud558\uace0 \uc124\uba85, \ubd84\uc11d, \uc8fc\uc11d, \ub300\uc548, <think> \ud0dc\uadf8\ub294 \ucd9c\ub825\ud558\uc9c0 \ub9c8\uc138\uc694.',
            strict:
              '\uc774\uac83\uc740 \uc5c4\uaca9 \ubaa8\ub4dc\uc785\ub2c8\ub2e4. \uc774\uc804 \ucd9c\ub825\uc774 \uc6d0\ubb38 \uc5b8\uc5b4\ub85c \ub0a8\uc544 \uc788\uc5c8\ub2e4\uba74 \uc774\ubc88\uc5d0\ub294 \ubc18\ub4dc\uc2dc \ud55c\uad6d\uc5b4\ub85c \ubc14\uafb8\uc138\uc694.',
            json:
              'JSON\ub9cc \ubc18\ud658\ud558\uace0 id\uc640 \ud56d\ubaa9 \uc218\ub97c \ubc14\uafb8\uc9c0 \ub9d0\uba70, text\ub294 \ubc18\ub4dc\uc2dc \ud55c\uad6d\uc5b4\uc5ec\uc57c \ud569\ub2c8\ub2e4.',
          };
        case 'en':
        case 'english':
          return {
            onlyTarget: 'Please output the final translation in natural English only.',
            avoidSource:
              'Do not leave the response in the original language unless it is a proper noun, brand, code identifier, or unavoidable quoted term.',
            lineSafe:
              'If the input contains markers like [[L00001]], keep every marker, timestamp, and structured prefix exactly unchanged while preserving one output line per input line.',
            plain:
              'Output only the final translation. Do not include explanations, analysis, notes, alternatives, or <think> tags.',
            strict:
              'This is strict mode. If the previous output stayed in the source language, translate it into English now.',
            json:
              'Return JSON only. Keep every id and item count unchanged, and make sure each text field is in English.',
          };
        default:
          return null;
      }
    })();

    if (!rules) return '';

    return [
      rules.onlyTarget,
      rules.avoidSource,
      input.jsonMode ? rules.json : input.lineSafeMode ? rules.lineSafe : rules.plain,
      input.strictMode ? rules.strict : '',
    ]
      .filter(Boolean)
      .join('\n');
  }

  private static getLocalizedSystemPromptProfile(targetLang: string): LocalizedSystemPromptProfile | null {
    const normalized = normalizeLanguageKey(targetLang);
    switch (normalized) {
      case 'zh-tw':
        return {
          additionalInstructionsLabel: '附加指示：',
          glossaryLabel: '術語表：',
          inputLabel: '輸入：',
          inputJsonLabel: '輸入 JSON：',
          baseIntro: '請將輸入內容翻譯成繁體中文。',
          baseRules: [
            '請只使用繁體中文，不要輸出簡體中文。',
            '優先使用台灣常用詞彙。',
            '除非是專有名詞、品牌或程式碼識別字，否則不要保留英文。',
            '請保持輸入與輸出的行數對齊。',
            '請保留時間戳記與結構化前綴。',
          ],
          lineSafeRules: [
            '如果每行都帶有像 [[L00001]] 的標記，請原樣保留。',
            '每個輸入行都必須對應一個輸出行。',
            '不要合併、拆分、重排、新增或刪除任何一行。',
            '像 <<SPEAKER:Speaker 1>> 這類語者標記只用來理解上下文，不要輸出。',
          ],
          baseReturn: '請只回傳翻譯後的純文字。',
          strictRetryLines: [
            '這次輸出必須完整翻譯成繁體中文。',
            '除非是專有名詞、品牌、程式碼識別字或無法避免的引用，否則不要保留原文。',
            '如果上一輪輸出仍停留在來源語言，這次必須改成繁體中文。',
            '請只回傳最終翻譯。',
          ],
          jsonIntro: '你是字幕翻譯引擎，請將內容翻譯成繁體中文。',
          jsonRules: [
            '輸入是 JSON，格式為 {"lines":[{"id":"...","text":"...","speaker":"..."}]}。',
            '只翻譯每個 item.text，item.id 必須完全保留不變。',
            'speaker 只用來理解上下文，不要把 speaker 資訊輸出到 text。',
            '不要合併、拆分、重排、新增或刪除任何項目。',
          ],
          jsonReturn: '只回傳 JSON：{"lines":[{"id":"L00001","text":"..."}]}',
          jsonNoExtras: '不要輸出 markdown、註解或額外欄位。',
          cloudContextIntro: '請只翻譯 TARGET 字幕行，並翻譯成繁體中文。',
          cloudContextRules: [
            '輸入中有兩種行：',
            '- [TRANSLATE_00001] 行必須翻譯。',
            '- [CONTEXT] 行只供理解，不可翻譯也不可回傳。',
            '請依原順序只回傳已翻譯的 TARGET 行。',
            '每個輸出行都必須保留原本的 [TRANSLATE_00001] 標籤。',
            '不要輸出 [CONTEXT] 行。',
            '不要加入說明、註解、markdown 或分析。',
            '像 <<SPEAKER:...>> 這類語者標記不要輸出。',
            '結構化前綴與時間戳記會在後處理還原，所以只需要翻譯文字內容。',
          ],
        };
      case 'zh-cn':
        return {
          additionalInstructionsLabel: '附加指示：',
          glossaryLabel: '术语表：',
          inputLabel: '输入：',
          inputJsonLabel: '输入 JSON：',
          baseIntro: '请将输入内容翻译成简体中文。',
          baseRules: [
            '请只使用简体中文，不要输出繁体中文。',
            '优先使用大陆常用词汇。',
            '除非是专有名词、品牌或代码标识符，否则不要保留英文。',
            '请保持输入与输出的行数对齐。',
            '请保留时间戳和结构化前缀。',
          ],
          lineSafeRules: [
            '如果每行都带有像 [[L00001]] 的标记，请原样保留。',
            '每个输入行都必须对应一个输出行。',
            '不要合并、拆分、重排、新增或删除任何一行。',
            '像 <<SPEAKER:Speaker 1>> 这类说话人标记只用于理解上下文，不要输出。',
          ],
          baseReturn: '请只返回翻译后的纯文本。',
          strictRetryLines: [
            '这次输出必须完整翻译成简体中文。',
            '除非是专有名词、品牌、代码标识符或无法避免的引用，否则不要保留原文。',
            '如果上一轮输出仍停留在源语言，这次必须改成简体中文。',
            '请只返回最终翻译。',
          ],
          jsonIntro: '你是字幕翻译引擎，请将内容翻译成简体中文。',
          jsonRules: [
            '输入是 JSON，格式为 {"lines":[{"id":"...","text":"...","speaker":"..."}]}。',
            '只翻译每个 item.text，item.id 必须完全保留不变。',
            'speaker 只用于理解上下文，不要把 speaker 信息输出到 text。',
            '不要合并、拆分、重排、新增或删除任何项目。',
          ],
          jsonReturn: '只返回 JSON：{"lines":[{"id":"L00001","text":"..."}]}',
          jsonNoExtras: '不要输出 markdown、注释或额外字段。',
          cloudContextIntro: '请只翻译 TARGET 字幕行，并翻译成简体中文。',
          cloudContextRules: [
            '输入中有两种行：',
            '- [TRANSLATE_00001] 行必须翻译。',
            '- [CONTEXT] 行只供理解，不可翻译也不可返回。',
            '请按原顺序只返回已翻译的 TARGET 行。',
            '每个输出行都必须保留原本的 [TRANSLATE_00001] 标签。',
            '不要输出 [CONTEXT] 行。',
            '不要加入说明、注释、markdown 或分析。',
            '像 <<SPEAKER:...>> 这类说话人标记不要输出。',
            '结构化前缀和时间戳会在后处理还原，所以只需要翻译文本内容。',
          ],
        };
      case 'en':
      case 'english':
        return {
          additionalInstructionsLabel: 'Additional instructions:',
          glossaryLabel: 'Glossary:',
          inputLabel: 'Input:',
          inputJsonLabel: 'Input JSON:',
          baseIntro: 'Translate the input into English.',
          baseRules: [
            'Use natural, fluent English.',
            'Do not leave translated lines in the original non-English language unless the content is a proper noun, brand, code identifier, or unavoidable quoted term.',
            'Keep the number of output lines aligned with the input.',
            'Keep timestamps and structured prefixes untouched.',
          ],
          lineSafeRules: [
            'If lines contain markers like [[L00001]], preserve every marker exactly.',
            'Keep one output line per input line.',
            'Do not merge, split, reorder, add, or remove lines.',
            'Tokens like <<SPEAKER:Speaker 1>> are context only and must not appear in the output.',
          ],
          baseReturn: 'Return only the final translated plain text.',
          strictRetryLines: [
            'This output must be fully translated into English.',
            'Do not leave the source sentence unchanged unless it is a proper noun, brand, code identifier, or unavoidable quoted term.',
            'If the previous output remained in the source language, translate it into English now.',
            'Return only the final translation.',
          ],
          jsonIntro: 'You are a subtitle translation engine. Translate the content into English.',
          jsonRules: [
            'Input is JSON in the form {"lines":[{"id":"...","text":"...","speaker":"..."}]}.',
            'Translate only each item.text and keep every item.id unchanged.',
            'Use speaker only as context and never include speaker metadata inside text.',
            'Do not merge, split, reorder, add, or remove items.',
          ],
          jsonReturn: 'Return JSON only: {"lines":[{"id":"L00001","text":"..."}]}',
          jsonNoExtras: 'Do not output markdown, commentary, or extra keys.',
          cloudContextIntro: 'Translate only the TARGET subtitle lines into English.',
          cloudContextRules: [
            'The input contains two kinds of lines:',
            '- [TRANSLATE_00001] lines must be translated.',
            '- [CONTEXT] lines are reference only and must never be translated or echoed.',
            'Return only translated TARGET lines in the same order.',
            'Each output line must keep the same [TRANSLATE_00001] label.',
            'Do not output [CONTEXT] lines.',
            'Do not add commentary, notes, markdown, or analysis.',
            'Do not output speaker metadata tokens such as <<SPEAKER:...>>.',
            'Structured prefixes and timestamps are restored separately, so translate only the text body.',
          ],
        };
      case 'fi':
        return {
          additionalInstructionsLabel: 'Lisäohjeet:',
          glossaryLabel: 'Sanasto:',
          inputLabel: 'Syöte:',
          inputJsonLabel: 'Syöte-JSON:',
          baseIntro: 'Käännä syöte suomeksi.',
          baseRules: [
            'Käytä luonnollista ja sujuvaa suomea.',
            'Älä jätä rivejä alkuperäiselle kielelle, ellei kyse ole erisnimestä, brändistä, kooditunnisteesta tai pakollisesta lainauksesta.',
            'Säilytä syötteen ja tulosteen rivimäärä samana.',
            'Säilytä aikakoodit ja rakenteelliset etuliitteet muuttumattomina.',
          ],
          lineSafeRules: [
            'Jos riveillä on merkkejä kuten [[L00001]], säilytä ne täsmälleen ennallaan.',
            'Tuota yksi tulosrivi jokaista syöteriviä kohden.',
            'Älä yhdistä, jaa, järjestä uudelleen, lisää tai poista rivejä.',
            'Tokenit kuten <<SPEAKER:Speaker 1>> ovat vain kontekstia eikä niitä saa tulostaa.',
          ],
          baseReturn: 'Palauta vain lopullinen käännetty raakateksti.',
          strictRetryLines: [
            'Tämän tulosteen on oltava kokonaan suomeksi.',
            'Älä jätä lähdelausetta ennalleen, ellei kyse ole erisnimestä, brändistä, kooditunnisteesta tai pakollisesta lainauksesta.',
            'Jos edellinen tuloste jäi lähdekielelle, käännä se nyt suomeksi.',
            'Palauta vain lopullinen käännös.',
          ],
          jsonIntro: 'Olet tekstitysten käännösmalli. Käännä sisältö suomeksi.',
          jsonRules: [
            'Syöte on JSON-muodossa {"lines":[{"id":"...","text":"...","speaker":"..."}]}.',
            'Käännä vain item.text ja säilytä jokainen item.id muuttumattomana.',
            'Käytä speaker-kenttää vain kontekstina äläkä lisää sitä text-kenttään.',
            'Älä yhdistä, jaa, järjestä uudelleen, lisää tai poista kohteita.',
          ],
          jsonReturn: 'Palauta vain JSON: {"lines":[{"id":"L00001","text":"..."}]}',
          jsonNoExtras: 'Älä tulosta markdownia, kommentteja tai ylimääräisiä kenttiä.',
          cloudContextIntro: 'Käännä vain TARGET-tekstitysrivit suomeksi.',
          cloudContextRules: [
            'Syötteessä on kahdenlaisia rivejä:',
            '- [TRANSLATE_00001] rivit on käännettävä.',
            '- [CONTEXT] rivit ovat vain taustatietoa eikä niitä saa kääntää tai palauttaa.',
            'Palauta vain käännetyt TARGET-rivit samassa järjestyksessä.',
            'Jokaisen rivin on säilytettävä sama [TRANSLATE_00001]-tunniste.',
            'Älä tulosta [CONTEXT]-rivejä.',
            'Älä lisää selityksiä, kommentteja, markdownia tai analyysia.',
            'Älä tulosta puhujatunnisteita kuten <<SPEAKER:...>>.',
            'Rakenteelliset etuliitteet ja aikakoodit palautetaan erikseen, joten käännä vain tekstisisältö.',
          ],
        };
      case 'es':
        return {
          additionalInstructionsLabel: 'Instrucciones adicionales:',
          glossaryLabel: 'Glosario:',
          inputLabel: 'Entrada:',
          inputJsonLabel: 'JSON de entrada:',
          baseIntro: 'Traduce la entrada al español.',
          baseRules: [
            'Usa un español natural y fluido.',
            'No dejes líneas en el idioma original salvo que sean nombres propios, marcas, identificadores de código o citas inevitables.',
            'Mantén alineado el número de líneas entre entrada y salida.',
            'Conserva intactos los códigos de tiempo y los prefijos estructurados.',
          ],
          lineSafeRules: [
            'Si las líneas incluyen marcadores como [[L00001]], consérvalos exactamente.',
            'Debe haber una línea de salida por cada línea de entrada.',
            'No fusiones, dividas, reordenes, añadas ni elimines líneas.',
            'Los tokens como <<SPEAKER:Speaker 1>> solo dan contexto y no deben aparecer en la salida.',
          ],
          baseReturn: 'Devuelve solo el texto traducido final.',
          strictRetryLines: [
            'Esta salida debe quedar completamente en español.',
            'No dejes la frase original sin traducir salvo que sea un nombre propio, una marca, un identificador de código o una cita inevitable.',
            'Si la salida anterior quedó en el idioma de origen, tradúcela ahora al español.',
            'Devuelve solo la traducción final.',
          ],
          jsonIntro: 'Eres un motor de traducción de subtítulos. Traduce el contenido al español.',
          jsonRules: [
            'La entrada es un JSON con la forma {"lines":[{"id":"...","text":"...","speaker":"..."}]}.',
            'Traduce solo item.text y conserva cada item.id sin cambios.',
            'Usa speaker solo como contexto y no lo incluyas dentro de text.',
            'No fusiones, dividas, reordenes, añadas ni elimines elementos.',
          ],
          jsonReturn: 'Devuelve solo JSON: {"lines":[{"id":"L00001","text":"..."}]}',
          jsonNoExtras: 'No añadas markdown, comentarios ni claves extra.',
          cloudContextIntro: 'Traduce solo las líneas TARGET de subtítulos al español.',
          cloudContextRules: [
            'La entrada contiene dos tipos de líneas:',
            '- Las líneas [TRANSLATE_00001] deben traducirse.',
            '- Las líneas [CONTEXT] son solo referencia y no deben traducirse ni devolverse.',
            'Devuelve únicamente las líneas TARGET traducidas en el mismo orden.',
            'Cada línea de salida debe conservar la misma etiqueta [TRANSLATE_00001].',
            'No devuelvas líneas [CONTEXT].',
            'No añadas comentarios, notas, markdown ni análisis.',
            'No incluyas tokens de hablante como <<SPEAKER:...>>.',
            'Los prefijos estructurados y los códigos de tiempo se restauran aparte, así que traduce solo el texto.',
          ],
        };
      case 'de':
        return {
          additionalInstructionsLabel: 'Zusaetzliche Anweisungen:',
          glossaryLabel: 'Glossar:',
          inputLabel: 'Eingabe:',
          inputJsonLabel: 'Eingabe-JSON:',
          baseIntro: 'Uebersetze die Eingabe ins Deutsche.',
          baseRules: [
            'Verwende natuerliches und fluessiges Deutsch.',
            'Lass Zeilen nicht in der Ausgangssprache, ausser es handelt sich um Eigennamen, Marken, Code-Bezeichner oder unvermeidbare Zitate.',
            'Die Anzahl der Ausgabezeilen muss mit der Eingabe uebereinstimmen.',
            'Zeitstempel und strukturierte Praefixe duerfen nicht veraendert werden.',
          ],
          lineSafeRules: [
            'Wenn Zeilen Marker wie [[L00001]] enthalten, muessen diese exakt erhalten bleiben.',
            'Es muss genau eine Ausgabezeile pro Eingabezeile geben.',
            'Zeilen nicht zusammenfuehren, aufteilen, umsortieren, hinzufuegen oder entfernen.',
            'Token wie <<SPEAKER:Speaker 1>> dienen nur als Kontext und duerfen nicht ausgegeben werden.',
          ],
          baseReturn: 'Gib nur den finalen uebersetzten Klartext zurueck.',
          strictRetryLines: [
            'Diese Ausgabe muss vollstaendig auf Deutsch sein.',
            'Lass den Ursprungssatz nicht unveraendert, ausser es handelt sich um Eigennamen, Marken, Code-Bezeichner oder unvermeidbare Zitate.',
            'Wenn die vorherige Ausgabe in der Ausgangssprache blieb, uebersetze sie jetzt ins Deutsche.',
            'Gib nur die finale Uebersetzung zurueck.',
          ],
          jsonIntro: 'Du bist eine Untertitel-Uebersetzungsengine. Uebersetze den Inhalt ins Deutsche.',
          jsonRules: [
            'Die Eingabe ist JSON im Format {"lines":[{"id":"...","text":"...","speaker":"..."}]}.',
            'Uebersetze nur item.text und lasse jede item.id unveraendert.',
            'Verwende speaker nur als Kontext und gib diese Information nicht in text aus.',
            'Elemente nicht zusammenfuehren, aufteilen, umsortieren, hinzufuegen oder entfernen.',
          ],
          jsonReturn: 'Gib nur JSON zurueck: {"lines":[{"id":"L00001","text":"..."}]}',
          jsonNoExtras: 'Kein Markdown, keine Kommentare und keine zusaetzlichen Schluessel ausgeben.',
          cloudContextIntro: 'Uebersetze nur die TARGET-Untertitelzeilen ins Deutsche.',
          cloudContextRules: [
            'Die Eingabe enthaelt zwei Arten von Zeilen:',
            '- [TRANSLATE_00001]-Zeilen muessen uebersetzt werden.',
            '- [CONTEXT]-Zeilen sind nur Referenz und duerfen weder uebersetzt noch ausgegeben werden.',
            'Gib nur die uebersetzten TARGET-Zeilen in derselben Reihenfolge zurueck.',
            'Jede Ausgabezeile muss dieselbe Kennzeichnung [TRANSLATE_00001] behalten.',
            'Keine [CONTEXT]-Zeilen ausgeben.',
            'Keine Kommentare, Hinweise, Markdown oder Analysen hinzufuegen.',
            'Sprecher-Token wie <<SPEAKER:...>> duerfen nicht ausgegeben werden.',
            'Strukturierte Praefixe und Zeitstempel werden separat wiederhergestellt, daher nur den Textinhalt uebersetzen.',
          ],
        };
      case 'pt':
        return {
          additionalInstructionsLabel: 'Instrucoes adicionais:',
          glossaryLabel: 'Glossario:',
          inputLabel: 'Entrada:',
          inputJsonLabel: 'JSON de entrada:',
          baseIntro: 'Traduza a entrada para portugues.',
          baseRules: [
            'Use um portugues natural e fluido.',
            'Nao deixe linhas no idioma original, exceto quando forem nomes proprios, marcas, identificadores de codigo ou citacoes inevitaveis.',
            'Mantenha alinhado o numero de linhas entre entrada e saida.',
            'Preserve os codigos de tempo e os prefixos estruturados.',
          ],
          lineSafeRules: [
            'Se houver marcadores como [[L00001]], preserve-os exatamente.',
            'Deve haver uma linha de saida para cada linha de entrada.',
            'Nao una, divida, reordene, adicione ou remova linhas.',
            'Tokens como <<SPEAKER:Speaker 1>> servem apenas como contexto e nao devem aparecer na saida.',
          ],
          baseReturn: 'Retorne apenas o texto final traduzido.',
          strictRetryLines: [
            'Esta saida deve ficar totalmente em portugues.',
            'Nao deixe a frase original sem traducao, exceto quando for nome proprio, marca, identificador de codigo ou citacao inevitavel.',
            'Se a saida anterior permaneceu no idioma de origem, traduza-a agora para portugues.',
            'Retorne apenas a traducao final.',
          ],
          jsonIntro: 'Voce e um mecanismo de traducao de legendas. Traduza o conteudo para portugues.',
          jsonRules: [
            'A entrada e um JSON no formato {"lines":[{"id":"...","text":"...","speaker":"..."}]}.',
            'Traduza apenas item.text e preserve cada item.id sem alteracao.',
            'Use speaker apenas como contexto e nao o inclua dentro de text.',
            'Nao una, divida, reordene, adicione ou remova itens.',
          ],
          jsonReturn: 'Retorne apenas JSON: {"lines":[{"id":"L00001","text":"..."}]}',
          jsonNoExtras: 'Nao adicione markdown, comentarios nem chaves extras.',
          cloudContextIntro: 'Traduza apenas as linhas TARGET de legenda para portugues.',
          cloudContextRules: [
            'A entrada contem dois tipos de linhas:',
            '- Linhas [TRANSLATE_00001] devem ser traduzidas.',
            '- Linhas [CONTEXT] servem apenas como referencia e nao devem ser traduzidas nem retornadas.',
            'Retorne apenas as linhas TARGET traduzidas na mesma ordem.',
            'Cada linha de saida deve manter o mesmo rotulo [TRANSLATE_00001].',
            'Nao retorne linhas [CONTEXT].',
            'Nao adicione comentarios, notas, markdown ou analise.',
            'Nao inclua tokens de falante como <<SPEAKER:...>>.',
            'Prefixos estruturados e codigos de tempo sao restaurados separadamente, portanto traduza apenas o texto.',
          ],
        };
      case 'it':
        return {
          additionalInstructionsLabel: 'Istruzioni aggiuntive:',
          glossaryLabel: 'Glossario:',
          inputLabel: 'Input:',
          inputJsonLabel: 'JSON di input:',
          baseIntro: 'Traduci l\'input in italiano.',
          baseRules: [
            'Usa un italiano naturale e scorrevole.',
            'Non lasciare righe nella lingua originale salvo che siano nomi propri, marchi, identificatori di codice o citazioni inevitabili.',
            'Mantieni allineato il numero di righe tra input e output.',
            'Conserva timestamp e prefissi strutturati senza modificarli.',
          ],
          lineSafeRules: [
            'Se le righe includono marcatori come [[L00001]], conservali esattamente.',
            'Deve esserci una riga di output per ogni riga di input.',
            'Non unire, dividere, riordinare, aggiungere o rimuovere righe.',
            'I token come <<SPEAKER:Speaker 1>> servono solo come contesto e non devono comparire nell\'output.',
          ],
          baseReturn: 'Restituisci solo il testo finale tradotto.',
          strictRetryLines: [
            'Questa uscita deve essere interamente in italiano.',
            'Non lasciare la frase originale invariata salvo che sia un nome proprio, un marchio, un identificatore di codice o una citazione inevitabile.',
            'Se l\'output precedente era rimasto nella lingua di origine, traducilo ora in italiano.',
            'Restituisci solo la traduzione finale.',
          ],
          jsonIntro: 'Sei un motore di traduzione per sottotitoli. Traduci il contenuto in italiano.',
          jsonRules: [
            'L\'input e un JSON con la forma {"lines":[{"id":"...","text":"...","speaker":"..."}]}.',
            'Traduci solo item.text e conserva ogni item.id senza modifiche.',
            'Usa speaker solo come contesto e non inserirlo dentro text.',
            'Non unire, dividere, riordinare, aggiungere o rimuovere elementi.',
          ],
          jsonReturn: 'Restituisci solo JSON: {"lines":[{"id":"L00001","text":"..."}]}',
          jsonNoExtras: 'Non aggiungere markdown, commenti o chiavi extra.',
          cloudContextIntro: 'Traduci solo le righe TARGET dei sottotitoli in italiano.',
          cloudContextRules: [
            'L\'input contiene due tipi di righe:',
            '- Le righe [TRANSLATE_00001] devono essere tradotte.',
            '- Le righe [CONTEXT] sono solo riferimento e non devono essere tradotte o restituite.',
            'Restituisci solo le righe TARGET tradotte nello stesso ordine.',
            'Ogni riga di output deve mantenere la stessa etichetta [TRANSLATE_00001].',
            'Non restituire righe [CONTEXT].',
            'Non aggiungere commenti, note, markdown o analisi.',
            'Non includere token del parlante come <<SPEAKER:...>>.',
            'Prefissi strutturati e timestamp vengono ripristinati separatamente, quindi traduci solo il testo.',
          ],
        };
      case 'fr':
        return {
          additionalInstructionsLabel: 'Instructions supplementaires :',
          glossaryLabel: 'Glossaire :',
          inputLabel: 'Entree :',
          inputJsonLabel: 'JSON d\'entree :',
          baseIntro: 'Traduisez l\'entree en francais.',
          baseRules: [
            'Utilisez un francais naturel et fluide.',
            'Ne laissez pas de lignes dans la langue source sauf s\'il s\'agit de noms propres, de marques, d\'identifiants de code ou de citations inevitables.',
            'Conservez le meme nombre de lignes entre l\'entree et la sortie.',
            'Conservez intacts les horodatages et les prefixes structures.',
          ],
          lineSafeRules: [
            'Si les lignes contiennent des marqueurs comme [[L00001]], conservez-les exactement.',
            'Il doit y avoir une ligne de sortie pour chaque ligne d\'entree.',
            'Ne fusionnez pas, ne scindez pas, ne reordonnez pas, n\'ajoutez pas et ne supprimez pas de lignes.',
            'Les jetons comme <<SPEAKER:Speaker 1>> ne servent qu\'au contexte et ne doivent pas apparaitre dans la sortie.',
          ],
          baseReturn: 'Retournez uniquement le texte final traduit.',
          strictRetryLines: [
            'Cette sortie doit etre entierement en francais.',
            'Ne laissez pas la phrase source intacte sauf s\'il s\'agit d\'un nom propre, d\'une marque, d\'un identifiant de code ou d\'une citation inevitable.',
            'Si la sortie precedente est restee dans la langue source, traduisez-la maintenant en francais.',
            'Retournez uniquement la traduction finale.',
          ],
          jsonIntro: 'Vous etes un moteur de traduction de sous-titres. Traduisez le contenu en francais.',
          jsonRules: [
            'L\'entree est un JSON de la forme {"lines":[{"id":"...","text":"...","speaker":"..."}]}.',
            'Traduisez uniquement item.text et conservez chaque item.id sans modification.',
            'Utilisez speaker uniquement comme contexte et ne l\'incluez pas dans text.',
            'Ne fusionnez pas, ne scindez pas, ne reordonnez pas, n\'ajoutez pas et ne supprimez pas d\'elements.',
          ],
          jsonReturn: 'Retournez uniquement le JSON : {"lines":[{"id":"L00001","text":"..."}]}',
          jsonNoExtras: 'N\'ajoutez ni markdown, ni commentaires, ni cles supplementaires.',
          cloudContextIntro: 'Traduisez uniquement les lignes TARGET de sous-titres en francais.',
          cloudContextRules: [
            'L\'entree contient deux types de lignes :',
            '- Les lignes [TRANSLATE_00001] doivent etre traduites.',
            '- Les lignes [CONTEXT] servent uniquement de reference et ne doivent ni etre traduites ni etre renvoyees.',
            'Retournez uniquement les lignes TARGET traduites dans le meme ordre.',
            'Chaque ligne de sortie doit conserver la meme etiquette [TRANSLATE_00001].',
            'Ne retournez pas de lignes [CONTEXT].',
            'N\'ajoutez ni commentaires, ni notes, ni markdown, ni analyse.',
            'N\'incluez pas de jetons de locuteur comme <<SPEAKER:...>>.',
            'Les prefixes structures et les horodatages sont restaures separement ; traduisez uniquement le texte.',
          ],
        };
      case 'ja':
      case 'jp':
        return {
          additionalInstructionsLabel: '追加指示：',
          glossaryLabel: '用語集：',
          inputLabel: '入力：',
          inputJsonLabel: '入力 JSON：',
          baseIntro: '入力を日本語に翻訳してください。',
          baseRules: [
            '自然で読みやすい日本語を使ってください。',
            '固有名詞、ブランド名、コード識別子、または避けられない引用を除き、原文の言語を残さないでください。',
            '入力と出力の行数を一致させてください。',
            'タイムスタンプと構造化プレフィックスはそのまま保持してください。',
          ],
          lineSafeRules: [
            '各行に [[L00001]] のようなマーカーがある場合は、そのまま保持してください。',
            '入力 1 行につき出力も 1 行にしてください。',
            '行の結合、分割、並べ替え、追加、削除をしないでください。',
            '<<SPEAKER:Speaker 1>> のようなトークンは文脈用であり、出力してはいけません。',
          ],
          baseReturn: '最終的な翻訳テキストだけを返してください。',
          strictRetryLines: [
            '今回の出力は必ず日本語に完全翻訳してください。',
            '固有名詞、ブランド名、コード識別子、または避けられない引用を除き、原文を残さないでください。',
            '前回の出力が元の言語のままだった場合は、今回は必ず日本語にしてください。',
            '最終翻訳だけを返してください。',
          ],
          jsonIntro: 'あなたは字幕翻訳エンジンです。内容を日本語に翻訳してください。',
          jsonRules: [
            '入力は {"lines":[{"id":"...","text":"...","speaker":"..."}]} 形式の JSON です。',
            '各 item.text だけを翻訳し、item.id は完全にそのまま保持してください。',
            'speaker は文脈理解にだけ使い、text に出力してはいけません。',
            '項目の結合、分割、並べ替え、追加、削除をしないでください。',
          ],
          jsonReturn: 'JSON のみを返してください：{"lines":[{"id":"L00001","text":"..."}]}',
          jsonNoExtras: 'markdown、コメント、追加キーは出力しないでください。',
          cloudContextIntro: 'TARGET 字幕行だけを日本語に翻訳してください。',
          cloudContextRules: [
            '入力には 2 種類の行があります。',
            '- [TRANSLATE_00001] 行は翻訳が必要です。',
            '- [CONTEXT] 行は参照用であり、翻訳も出力もしてはいけません。',
            '翻訳済みの TARGET 行だけを同じ順序で返してください。',
            '各出力行は同じ [TRANSLATE_00001] ラベルを保持してください。',
            '[CONTEXT] 行は出力しないでください。',
            '説明、注記、markdown、分析を追加しないでください。',
            '<<SPEAKER:...>> のような話者トークンは出力しないでください。',
            '構造化プレフィックスとタイムスタンプは後で復元されるため、本文だけを翻訳してください。',
          ],
        };
      case 'ko':
      case 'kr':
        return {
          additionalInstructionsLabel: '추가 지시:',
          glossaryLabel: '용어집:',
          inputLabel: '입력:',
          inputJsonLabel: '입력 JSON:',
          baseIntro: '입력을 한국어로 번역하세요.',
          baseRules: [
            '자연스럽고 읽기 쉬운 한국어를 사용하세요.',
            '고유명사, 브랜드명, 코드 식별자, 또는 피할 수 없는 인용을 제외하고 원문 언어를 남기지 마세요.',
            '입력과 출력의 줄 수를 맞추세요.',
            '타임스탬프와 구조화된 접두어는 그대로 유지하세요.',
          ],
          lineSafeRules: [
            '각 줄에 [[L00001]] 같은 마커가 있으면 그대로 유지하세요.',
            '입력 한 줄마다 출력도 한 줄이어야 합니다.',
            '줄을 합치거나 나누거나 재정렬하거나 추가하거나 삭제하지 마세요.',
            '<<SPEAKER:Speaker 1>> 같은 토큰은 문맥용이며 출력하면 안 됩니다.',
          ],
          baseReturn: '최종 번역 텍스트만 반환하세요.',
          strictRetryLines: [
            '이번 출력은 반드시 완전한 한국어 번역이어야 합니다.',
            '고유명사, 브랜드명, 코드 식별자, 또는 피할 수 없는 인용을 제외하고 원문을 남기지 마세요.',
            '이전 출력이 원문 언어로 남아 있었다면 이번에는 반드시 한국어로 바꾸세요.',
            '최종 번역만 반환하세요.',
          ],
          jsonIntro: '당신은 자막 번역 엔진입니다. 내용을 한국어로 번역하세요.',
          jsonRules: [
            '입력은 {"lines":[{"id":"...","text":"...","speaker":"..."}]} 형식의 JSON입니다.',
            '각 item.text만 번역하고 item.id는 완전히 그대로 유지하세요.',
            'speaker는 문맥 이해에만 사용하고 text에 출력하지 마세요.',
            '항목을 합치거나 나누거나 재정렬하거나 추가하거나 삭제하지 마세요.',
          ],
          jsonReturn: 'JSON만 반환하세요: {"lines":[{"id":"L00001","text":"..."}]}',
          jsonNoExtras: 'markdown, 주석, 추가 키를 출력하지 마세요.',
          cloudContextIntro: 'TARGET 자막 줄만 한국어로 번역하세요.',
          cloudContextRules: [
            '입력에는 두 종류의 줄이 있습니다.',
            '- [TRANSLATE_00001] 줄은 번역해야 합니다.',
            '- [CONTEXT] 줄은 참고용이며 번역하거나 출력하면 안 됩니다.',
            '번역된 TARGET 줄만 같은 순서로 반환하세요.',
            '각 출력 줄은 동일한 [TRANSLATE_00001] 라벨을 유지해야 합니다.',
            '[CONTEXT] 줄은 출력하지 마세요.',
            '설명, 주석, markdown, 분석을 추가하지 마세요.',
            '<<SPEAKER:...>> 같은 화자 토큰은 출력하지 마세요.',
            '구조화된 접두어와 타임스탬프는 나중에 복원되므로 본문 텍스트만 번역하세요.',
          ],
        };
      default:
        return null;
    }
  }

  private static normalizePromptTemplateId(promptTemplateId?: string): PromptTemplateId {
    if (
      promptTemplateId === 'subtitle_general' ||
      promptTemplateId === 'subtitle_strict_alignment' ||
      promptTemplateId === 'subtitle_concise_spoken' ||
      promptTemplateId === 'subtitle_formal_precise' ||
      promptTemplateId === 'subtitle_asr_recovery' ||
      promptTemplateId === 'subtitle_technical_terms'
    ) {
      return promptTemplateId;
    }
    return '';
  }

  private static resolveTranslationQualityModeForRequest(input: {
    promptTemplateId?: string;
    prompt?: string;
    enableJsonLineRepair?: boolean;
    modelStrategy?: LocalTranslateModelStrategy | null;
  }): TranslationQualityMode {
    if (input.modelStrategy) {
      return resolveLocalTranslationIntent({
        promptTemplateId: this.normalizePromptTemplateId(input.promptTemplateId),
        prompt: input.prompt,
        enableJsonLineRepair: input.enableJsonLineRepair,
        modelStrategy: input.modelStrategy,
      }).qualityMode;
    }
    return resolveTranslationQualityMode({
      promptTemplateId: this.normalizePromptTemplateId(input.promptTemplateId),
      prompt: input.prompt,
      enableJsonLineRepair: input.enableJsonLineRepair,
    });
  }

  private static getBuiltInGlossaryRootDir() {
    return PathManager.getBuiltInGlossaryRootPath();
  }

  private static parseGlossaryFileEntries(filePath: string) {
    if (!fs.existsSync(filePath)) return [];
    const raw = fs.readFileSync(filePath, 'utf8');
    return raw
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith('#') && line.includes('='));
  }

  private static mergeGlossaryEntries(entryGroups: string[][]) {
    const merged = new Map<string, string>();
    for (const group of entryGroups) {
      for (const entry of group) {
        const separatorIndex = entry.indexOf('=');
        if (separatorIndex <= 0) continue;
        const source = entry.slice(0, separatorIndex).trim();
        const target = entry.slice(separatorIndex + 1).trim();
        if (!source || !target) continue;
        merged.set(source.toLowerCase(), `${source}=${target}`);
      }
    }
    return Array.from(merged.values());
  }

  private static getBuiltInGlossaryEntries(targetLang: string, promptTemplateId?: string) {
    const localeChain = buildTranslationGlossaryLocaleChain(targetLang);
    const normalizedTemplateId = this.normalizePromptTemplateId(promptTemplateId);
    const cacheKey = `${localeChain.join('|')}:${normalizedTemplateId || 'base'}`;
    const cached = this.builtInGlossaryCache.get(cacheKey);
    if (cached) return cached;

    const glossaryRoot = this.getBuiltInGlossaryRootDir();
    const baseEntryGroups = localeChain.map((localeKey) =>
      this.parseGlossaryFileEntries(path.join(glossaryRoot, localeKey, 'base.txt'))
    );
    const templateEntryGroups =
      normalizedTemplateId
        ? localeChain.map((localeKey) =>
            this.parseGlossaryFileEntries(path.join(glossaryRoot, localeKey, `${normalizedTemplateId}.txt`))
          )
        : [];
    const merged = this.mergeGlossaryEntries([...baseEntryGroups, ...templateEntryGroups]);
    this.builtInGlossaryCache.set(cacheKey, merged);
    return merged;
  }

  private static getBuiltInGlossary(targetLang: string, promptTemplateId?: string) {
    return this.getBuiltInGlossaryEntries(targetLang, promptTemplateId).join('; ');
  }

  private static buildEffectiveGlossary(targetLang: string, glossary?: string, promptTemplateId?: string) {
    const builtIn = this.getBuiltInGlossary(targetLang, promptTemplateId);
    const custom = String(glossary || '').trim();
    return [builtIn, custom].filter(Boolean).join('; ');
  }

  private static normalizeDeepLTargetLanguage(targetLang: string) {
    const normalized = normalizeLanguageKey(targetLang);
    const map: Record<string, string> = {
      'zh-tw': 'ZH',
      zh: 'ZH',
      'zh-cn': 'ZH',
      en: 'EN',
      fi: 'FI',
      es: 'ES',
      de: 'DE',
      pt: 'PT',
      it: 'IT',
      fr: 'FR',
      ja: 'JA',
      jp: 'JA',
      ko: 'KO',
      kr: 'KO',
    };
    return map[normalized] || 'EN';
  }

  private static buildSystemPrompt(
    targetLang: string,
    glossary?: string,
    lineSafeMode = false,
    promptTemplateId?: string,
    sourceLang?: string
  ) {
    const effectiveGlossary = this.buildEffectiveGlossary(targetLang, glossary, promptTemplateId);
    const sourceLanguageDescriptor = this.buildSourceLanguageDescriptor(sourceLang);
    const localized = this.getLocalizedSystemPromptProfile(targetLang);
    if (localized) {
      return [
        localized.baseIntro,
        ...localized.baseRules,
        ...(sourceLanguageDescriptor ? [`Source language: ${sourceLanguageDescriptor}.`] : []),
        ...this.getTargetLanguageHardeningLines(targetLang, lineSafeMode, false),
        ...(lineSafeMode ? localized.lineSafeRules : []),
        localized.baseReturn,
        effectiveGlossary ? `${localized.glossaryLabel} ${effectiveGlossary}` : '',
      ]
        .filter(Boolean)
        .join('\n');
    }

    const lineSafeHints = lineSafeMode
      ? [
          'Each input line starts with a marker like [[L00001]].',
          'Keep every marker exactly unchanged and keep one output line per input line.',
          'Do not merge, split, reorder, add, or remove lines/markers.',
          'Context tokens like <<SPEAKER:Speaker 1>> are metadata only. Use them for context but never output them.',
        ]
      : [];

    return [
      `Translate the input into ${this.buildTargetLanguageDescriptor(targetLang)}.`,
      ...(sourceLanguageDescriptor ? [`The source language is ${sourceLanguageDescriptor}.`] : []),
      ...this.getTargetLanguageInstructionLines(targetLang),
      ...this.getTargetLanguageHardeningLines(targetLang, lineSafeMode, false),
      'Keep each input line aligned with the output line.',
      'Keep timestamps and structured prefixes untouched.',
      ...lineSafeHints,
      'Return only translated text.',
      effectiveGlossary ? `Glossary: ${effectiveGlossary}` : '',
    ]
      .filter(Boolean)
      .join('\n');
  }

  private static resolveSystemPrompt(options: TranslateRequestOptions) {
    if (options.disableSystemPrompt) return '';
    const basePrompt = this.buildSystemPrompt(
      options.targetLang,
      options.glossary,
      options.lineSafeMode,
      options.promptTemplateId,
      options.sourceLang
    );
    const custom = String(options.systemPromptOverride || '').trim();
    if (!custom) return basePrompt;
    const localized = this.getLocalizedSystemPromptProfile(options.targetLang);
    return [
      basePrompt,
      '',
      localized?.additionalInstructionsLabel || 'Additional instructions:',
      custom,
    ]
      .filter(Boolean)
      .join('\n');
  }

  private static getCloudContextConfig() {
    return {
      enabled: this.getEnvBoolean('TRANSLATE_CLOUD_CONTEXT_MODE', false),
      targetLines: Math.round(this.getEnvNumber('TRANSLATE_CLOUD_CONTEXT_TARGET_LINES', 24, 4, 120)),
      minTargetLines: Math.round(this.getEnvNumber('TRANSLATE_CLOUD_CONTEXT_MIN_TARGET_LINES', 6, 1, 60)),
      contextWindow: Math.round(this.getEnvNumber('TRANSLATE_CLOUD_CONTEXT_WINDOW', 2, 0, 8)),
      charBudget: Math.round(this.getEnvNumber('TRANSLATE_CLOUD_CONTEXT_CHAR_BUDGET', 2400, 200, 20000)),
      maxSplitDepth: Math.round(this.getEnvNumber('TRANSLATE_CLOUD_CONTEXT_MAX_SPLIT_DEPTH', 2, 0, 8)),
    } as CloudContextConfig;
  }

  private static isNvidiaHostedCloudProvider(resolvedProvider: ResolvedCloudTranslateProvider) {
    if (resolvedProvider.provider !== 'openai-compatible') return false;
    try {
      const host = new URL(resolvedProvider.endpointUrl).hostname.toLowerCase();
      return host === 'integrate.api.nvidia.com';
    } catch {
      return false;
    }
  }

  private static getNvidiaCloudBatchConfig(
    resolvedProvider: ResolvedCloudTranslateProvider
  ): CloudTranslationProviderBatchConfig | null {
    if (!this.isNvidiaHostedCloudProvider(resolvedProvider)) return null;
    if (!this.getEnvBoolean('TRANSLATE_NVIDIA_CLOUD_BATCHING', true)) return null;

    return {
      enabled: true,
      source: 'nvidia-hosted',
      targetLines: Math.round(this.getEnvNumber('TRANSLATE_NVIDIA_CLOUD_BATCH_SIZE', 24, 4, 120)),
      minTargetLines: Math.round(this.getEnvNumber('TRANSLATE_NVIDIA_CLOUD_MIN_BATCH_SIZE', 6, 1, 60)),
      charBudget: Math.round(this.getEnvNumber('TRANSLATE_NVIDIA_CLOUD_BATCH_CHAR_BUDGET', 2400, 200, 20000)),
      maxSplitDepth: Math.round(this.getEnvNumber('TRANSLATE_NVIDIA_CLOUD_MAX_SPLIT_DEPTH', 4, 0, 8)),
      maxOutputTokens: Math.round(this.getEnvNumber('TRANSLATE_NVIDIA_CLOUD_MAX_OUTPUT_TOKENS', 2048, 256, 16384)),
      timeoutMs: Math.round(this.getEnvNumber('TRANSLATE_NVIDIA_CLOUD_TIMEOUT_MS', 300000, 30000, 900000)),
      stream: this.getEnvBoolean('TRANSLATE_NVIDIA_CLOUD_STREAM', false),
    };
  }

  private static buildCloudContextSystemPrompt(input: {
    targetLang: string;
    sourceLang?: string;
    glossary?: string;
    promptTemplateId?: string;
    prompt?: string;
  }) {
    const effectiveGlossary = this.buildEffectiveGlossary(input.targetLang, input.glossary, input.promptTemplateId);
    const customPrompt = String(input.prompt || '').trim();
    const sourceLanguageDescriptor = this.buildSourceLanguageDescriptor(input.sourceLang);
    const localized = this.getLocalizedSystemPromptProfile(input.targetLang);
    if (localized) {
      return [
        localized.cloudContextIntro,
        ...(sourceLanguageDescriptor ? [`Source language: ${sourceLanguageDescriptor}.`] : []),
        ...localized.cloudContextRules,
        effectiveGlossary ? `${localized.glossaryLabel} ${effectiveGlossary}` : '',
        ...(customPrompt
          ? [
              localized.additionalInstructionsLabel,
              customPrompt,
            ]
          : []),
      ]
        .filter(Boolean)
        .join('\n');
    }

    return [
      `Translate only the TARGET subtitle lines into ${this.buildTargetLanguageDescriptor(input.targetLang)}.`,
      ...(sourceLanguageDescriptor ? [`The source language is ${sourceLanguageDescriptor}.`] : []),
      ...this.getTargetLanguageInstructionLines(input.targetLang),
      'Input contains two kinds of lines:',
      '- [TRANSLATE_00001] lines must be translated.',
      '- [CONTEXT] lines are reference only and must never be translated or echoed.',
      'Return only translated TARGET lines in the exact same order.',
      'Each output line must start with the same [TRANSLATE_00001] label followed by the translated text.',
      'Do not output [CONTEXT] lines.',
      'Do not add commentary, notes, markdown, or analysis.',
      'Keep speaker metadata tokens such as <<SPEAKER:...>> out of the output.',
      'Structured prefixes and timestamps are restored separately, so translate only the spoken text body.',
      effectiveGlossary ? `Glossary: ${effectiveGlossary}` : '',
      ...(customPrompt
        ? [
            'Additional instructions:',
            customPrompt,
          ]
        : []),
    ]
      .filter(Boolean)
      .join('\n');
  }

  private static buildJsonLineRepairPrompt(targetLang: string, glossary?: string, promptTemplateId?: string) {
    const effectiveGlossary = this.buildEffectiveGlossary(targetLang, glossary, promptTemplateId);
    const localized = this.getLocalizedSystemPromptProfile(targetLang);
    if (localized) {
      return [
        localized.jsonIntro,
        ...localized.jsonRules,
        ...this.getTargetLanguageHardeningLines(targetLang, false, true),
        localized.jsonReturn,
        localized.jsonNoExtras,
        effectiveGlossary ? `${localized.glossaryLabel} ${effectiveGlossary}` : '',
      ]
        .filter(Boolean)
        .join('\n');
    }
    return [
      `You are a subtitle translation engine. Translate into ${this.buildTargetLanguageDescriptor(targetLang)}.`,
      ...this.getTargetLanguageInstructionLines(targetLang),
      ...this.getTargetLanguageHardeningLines(targetLang, false, true),
      'Input is JSON with field "lines", each item has "id", "text", and optional "speaker".',
      'Translate only each item.text, keep item.id unchanged.',
      'Use optional speaker metadata only as context. Do not output speaker metadata inside item.text.',
      'Do not merge, split, reorder, add, or remove items.',
      'Return JSON only: {"lines":[{"id":"L00001","text":"..."}]}',
      'No markdown fences, no commentary, no extra keys.',
      effectiveGlossary ? `Glossary: ${effectiveGlossary}` : '',
    ]
      .filter(Boolean)
      .join('\n');
  }

  private static inferLocalTranslateModelStrategy(localModel: LocalModelDefinition): LocalTranslateModelStrategy {
    return inferLocalTranslateModelStrategyModule(localModel);
  }

  private static buildQwenChatMlPrompt(input: { systemPrompt: string; userText: string }) {
    const systemPrompt = String(input.systemPrompt || '').trim();
    const userText = String(input.userText || '').trim();
    return [
      '<|im_start|>system',
      systemPrompt,
      '<|im_end|>',
      '<|im_start|>user',
      userText,
      '<|im_end|>',
      '<|im_start|>assistant',
      '',
    ].join('\n');
  }

  private static buildQwen3NonThinkingChatPrompt(input: {
    systemPrompt: string;
    userText: string;
  }) {
    const systemPrompt = String(input.systemPrompt || '').trim();
    const userText = String(input.userText || '').trim();
    return [
      '<|im_start|>system',
      systemPrompt,
      '<|im_end|>',
      '<|im_start|>user',
      userText,
      '<|im_end|>',
      '<|im_start|>assistant',
      '<think>',
      '',
      '</think>',
      '',
    ].join('\n');
  }

  private static buildPhi4ChatPrompt(input: { systemPrompt: string; userText: string }) {
    const systemPrompt = String(input.systemPrompt || '').trim();
    const userText = String(input.userText || '').trim();
    return `<|system|>${systemPrompt}<|end|><|user|>${userText}<|end|><|assistant|>`;
  }

  private static buildDeepSeekR1DistillQwenPrompt(input: { userText: string }) {
    const userText = String(input.userText || '').trim();
    return `<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>${userText}<\uff5cAssistant\uff5c><think>\n`;
  }

  private static buildDeepSeekR1DistillQwenPlainPrompt(input: { userText: string }) {
    const userText = String(input.userText || '').trim();
    return `<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>${userText}<\uff5cAssistant\uff5c>`;
  }

  private static buildSeq2SeqTranslationPrompt(input: {
    text: string;
    targetLang: string;
    glossary?: string;
    lineSafeMode: boolean;
    promptOverride?: string;
  }) {
    const descriptor = this.buildTargetLanguageDescriptor(input.targetLang);
    const instructions = [
      `Translate the following text into ${descriptor}.`,
      input.lineSafeMode ? 'Preserve the original line breaks and return the same number of lines.' : 'Return only the translation.',
      'Do not explain the result.',
    ];
    if (String(input.glossary || '').trim()) {
      instructions.push(`Glossary:\n${String(input.glossary || '').trim()}`);
    }
    if (String(input.promptOverride || '').trim()) {
      instructions.push(String(input.promptOverride || '').trim());
    }
    return [...instructions, '', input.text].join('\n');
  }

  private static buildSeq2SeqJsonRepairPrompt(input: {
    payload: string;
    targetLang: string;
    glossary?: string;
  }) {
    const descriptor = this.buildTargetLanguageDescriptor(input.targetLang);
    const instructions = [
      `Translate every text field in the JSON payload into ${descriptor}.`,
      'Keep every id unchanged.',
      'Return valid JSON only.',
      'Use either {"lines":[{"id":"L00001","text":"..."}]} or a plain array of {"id","text"} objects.',
    ];
    if (String(input.glossary || '').trim()) {
      instructions.push(`Glossary:\n${String(input.glossary || '').trim()}`);
    }
    return [...instructions, '', 'Input JSON:', input.payload].join('\n');
  }

  private static buildLocalTranslationPrompt(input: {
    text: string;
    targetLang: string;
    sourceLang?: string;
    glossary?: string;
    lineSafeMode: boolean;
    modelStrategy: LocalTranslateModelStrategy;
    promptStyle?: LocalTranslateModelStrategy['promptStyle'];
    strictMode?: boolean;
    translationQualityMode?: TranslationQualityMode;
    promptOverride?: string;
    disableSystemPrompt?: boolean;
    promptTemplateId?: string;
    sourceText?: string;
  }) {
    if (input.disableSystemPrompt) {
      return input.text;
    }
    if (input.modelStrategy.family === 'seq2seq') {
      return this.buildSeq2SeqTranslationPrompt({
        text: input.text,
        targetLang: input.targetLang,
        glossary: input.glossary,
        lineSafeMode: input.lineSafeMode,
        promptOverride: input.promptOverride,
      });
    }
    const localized = this.getLocalizedSystemPromptProfile(input.targetLang);
    const systemPrompt = this.buildSystemPrompt(
      input.targetLang,
      input.glossary,
      input.lineSafeMode,
      input.promptTemplateId,
      input.sourceLang
    );
    return buildLocalTranslationPromptModule({
      text: input.text,
      systemPrompt,
      sourceAwarePrompt: this.getSourceAwareTargetLanguageLines(
        input.targetLang,
        input.sourceText ?? input.text,
        { lineSafeMode: input.lineSafeMode, jsonMode: false }
      ).join('\n'),
      customPrompt: String(input.promptOverride || '').trim(),
      translationQualityMode: input.translationQualityMode || (input.strictMode ? 'json_strict' : undefined),
      modelStrategy: input.modelStrategy,
      promptStyle: input.promptStyle,
      labels: localized,
      deepseekPromptBuilder: (userText) => this.buildDeepSeekR1DistillQwenPrompt({ userText }),
      deepseekPlainPromptBuilder: (userText) => this.buildDeepSeekR1DistillQwenPlainPrompt({ userText }),
    });
  }

  private static buildLocalTranslationMessages(input: {
    text: string;
    targetLang: string;
    sourceLang?: string;
    modelStrategy: LocalTranslateModelStrategy;
    promptStyle?: LocalTranslateModelStrategy['promptStyle'];
  }): LocalTranslateStructuredMessage[] | null {
    if (input.modelStrategy.family !== 'translategemma') return null;
    return buildTranslateGemmaMessages({
      text: input.text,
      sourceLang: input.sourceLang || inferTranslateGemmaSourceLanguageCode(input.text),
      targetLang: input.targetLang,
      promptStyle: input.promptStyle,
    });
  }

  private static buildLocalJsonRepairPrompt(input: {
    payload: string;
    targetLang: string;
    glossary?: string;
    modelStrategy: LocalTranslateModelStrategy;
    promptStyle?: LocalTranslateModelStrategy['promptStyle'];
    promptTemplateId?: string;
    sourceText?: string;
  }) {
    if (input.modelStrategy.family === 'seq2seq') {
      return this.buildSeq2SeqJsonRepairPrompt({
        payload: input.payload,
        targetLang: input.targetLang,
        glossary: input.glossary,
      });
    }
    const base = this.buildJsonLineRepairPrompt(input.targetLang, input.glossary, input.promptTemplateId);
    const localized = this.getLocalizedSystemPromptProfile(input.targetLang);
    return buildLocalJsonRepairPromptModule({
      payload: input.payload,
      basePrompt: base,
      sourceAwarePrompt: this.getSourceAwareTargetLanguageLines(
        input.targetLang,
        input.sourceText,
        { lineSafeMode: false, jsonMode: true }
      ).join('\n'),
      translationQualityMode: 'json_strict',
      modelStrategy: input.modelStrategy,
      promptStyle: input.promptStyle,
      labels: localized,
      deepseekPromptBuilder: (userText) => this.buildDeepSeekR1DistillQwenPrompt({ userText }),
      deepseekPlainPromptBuilder: (userText) => this.buildDeepSeekR1DistillQwenPlainPrompt({ userText }),
    });
  }

  private static buildLocalGenerationOptions(input: {
    localModel?: LocalModelDefinition;
    modelStrategy: LocalTranslateModelStrategy;
    jsonMode?: boolean;
    targetLang?: string;
    strictMode?: boolean;
    translationQualityMode?: TranslationQualityMode;
  }) {
    return buildLocalGenerationOptionsModule({
      localModel: input.localModel,
      modelStrategy: input.modelStrategy,
      jsonMode: input.jsonMode,
      targetLang: input.targetLang,
      strictMode: input.strictMode,
      translationQualityMode: input.translationQualityMode,
      getBoolean: (name, fallback) => this.getEnvBoolean(name, fallback),
      getNumber: (name, fallback, min, max) => this.getEnvNumber(name, fallback, min, max),
    });
  }

  private static estimateLocalMaxNewTokens(input: {
    text: string;
    lineCount: number;
    lineSafeMode: boolean;
    localModel?: LocalModelDefinition;
    jsonMode?: boolean;
    strictMode?: boolean;
  }) {
    return estimateLocalMaxNewTokensModule({
      ...input,
      getNumber: (name, fallback, min, max) => this.getEnvNumber(name, fallback, min, max),
    });
  }

  private static isFixedLineBatchingMode(rawMode: string) {
    const mode = String(rawMode || '').trim().toLowerCase();
    return mode === 'fixed_lines' || mode === 'fixed-line' || mode === 'line_char' || mode === 'line-char';
  }

  private static getLocalBatchingMode() {
    return String(process.env.TRANSLATE_LOCAL_BATCHING_MODE || '').trim().toLowerCase();
  }

  private static getLocalBatchCaps(input?: {
    localModel?: LocalModelDefinition;
    modelStrategy?: LocalTranslateModelStrategy;
    translationQualityMode?: TranslationQualityMode;
  }) {
    const qualityMode = input?.translationQualityMode || 'plain_probe';
    const batchingProfile =
      input?.modelStrategy
        ? resolveEffectiveLocalTranslationProfile({
            localModel: input.localModel,
            modelStrategy: input.modelStrategy,
            qualityMode,
          }).translationProfile.lineSafeBatching
        : null;
    const configuredMaxLines = Math.round(
      this.getEnvNumber('TRANSLATE_LOCAL_BATCH_SIZE', batchingProfile?.maxLines ?? 60, 4, 200)
    );
    const configuredCharBudget = Math.round(
      this.getEnvNumber('TRANSLATE_LOCAL_BATCH_CHAR_BUDGET', batchingProfile?.charBudget ?? 2800, 160, 40000)
    );
    const allowProfileCapOverride = ['1', 'true', 'yes', 'on'].includes(
      String(process.env.TRANSLATE_LOCAL_ALLOW_PROFILE_CAP_OVERRIDE || '').trim().toLowerCase()
    );
    const maxLines =
      batchingProfile?.maxLines && !allowProfileCapOverride
        ? Math.min(configuredMaxLines, batchingProfile.maxLines)
        : configuredMaxLines;
    const charBudget =
      batchingProfile?.charBudget && !allowProfileCapOverride
        ? Math.min(configuredCharBudget, batchingProfile.charBudget)
        : configuredCharBudget;
    return { qualityMode, batchingProfile, maxLines, charBudget };
  }

  private static splitLineSafeUnitsByLineAndChar(
    units: LineSafeUnit[],
    caps: {
      maxLines: number;
      charBudget: number;
    }
  ) {
    if (units.length <= 1) return units.length > 0 ? [units] : [];

    const batches: LineSafeUnit[][] = [];
    let current: LineSafeUnit[] = [];
    let currentChars = 0;

    for (const unit of units) {
      const estimatedChars = this.buildLineSafeInput([unit]).length + 1;
      const exceedsCurrent =
        current.length > 0 && (current.length >= caps.maxLines || currentChars + estimatedChars > caps.charBudget);
      if (exceedsCurrent) {
        batches.push(current);
        current = [];
        currentChars = 0;
      }
      current.push(unit);
      currentChars += estimatedChars;
    }

    if (current.length > 0) {
      batches.push(current);
    }

    return batches;
  }

  private static buildLocalBatchDebugInfo(input: {
    source: LocalTranslationBatchDebugInfo['source'];
    mode: LocalTranslationBatchDebugInfo['mode'];
    batches: LineSafeUnit[][];
    maxLines?: number | null;
    charBudget?: number | null;
    promptTokens?: number[];
    estimatedOutputTokens?: number[];
    inputTokenBudget?: number | null;
    outputTokenBudget?: number | null;
    contextWindow?: number | null;
    safetyReserveTokens?: number | null;
    fallbackReason?: string | null;
  }): LocalTranslationBatchDebugInfo {
    return {
      source: input.source,
      mode: input.mode,
      batchCount: input.batches.length,
      lineCounts: input.batches.map((batch) => batch.length),
      charCounts: input.batches.map((batch) => this.buildLineSafeInput(batch).length),
      promptTokens: input.promptTokens && input.promptTokens.length > 0 ? input.promptTokens : undefined,
      estimatedOutputTokens:
        input.estimatedOutputTokens && input.estimatedOutputTokens.length > 0 ? input.estimatedOutputTokens : undefined,
      inputTokenBudget: typeof input.inputTokenBudget === 'number' ? input.inputTokenBudget : null,
      outputTokenBudget: typeof input.outputTokenBudget === 'number' ? input.outputTokenBudget : null,
      contextWindow: typeof input.contextWindow === 'number' ? input.contextWindow : null,
      safetyReserveTokens: typeof input.safetyReserveTokens === 'number' ? input.safetyReserveTokens : null,
      maxLines: typeof input.maxLines === 'number' ? input.maxLines : null,
      charBudget: typeof input.charBudget === 'number' ? input.charBudget : null,
      fallbackReason: input.fallbackReason || null,
    };
  }

  private static async countLocalTranslationInputTokensBatch(input: {
    localModel: LocalModelDefinition;
    modelStrategy: LocalTranslateModelStrategy;
    sourceTexts: string[];
    targetLang: string;
    sourceLang?: string;
    glossary?: string;
    prompt?: string;
    promptTemplateId?: string;
    lineSafeMode: boolean;
    translationQualityMode?: TranslationQualityMode;
  }) {
    const modelPath = getLocalModelInstallDir(input.localModel);
    const profile = resolveEffectiveLocalTranslationProfile({
      localModel: input.localModel,
      modelStrategy: input.modelStrategy,
      qualityMode: input.translationQualityMode || 'plain_probe',
    }).translationProfile;
    const generation = this.buildLocalGenerationOptions({
      localModel: input.localModel,
      modelStrategy: input.modelStrategy,
      jsonMode: false,
      targetLang: input.targetLang,
      strictMode: false,
      translationQualityMode: input.translationQualityMode,
    });

    const entries = input.sourceTexts
      .map((rawText) => String(rawText || '').trim())
      .filter(Boolean)
      .map((sourceText) => {
        const messages = this.buildLocalTranslationMessages({
          text: sourceText,
          targetLang: input.targetLang,
          sourceLang: input.sourceLang,
          modelStrategy: input.modelStrategy,
          promptStyle: profile.effectivePromptStyle,
        });
        if (messages && messages.length > 0) {
          return {
            messages: messages as unknown as Array<Record<string, unknown>>,
            chatTemplateKwargs: generation.chatTemplateKwargs,
          };
        }

        const prompt = this.buildLocalTranslationPrompt({
          text: sourceText,
          sourceText,
          targetLang: input.targetLang,
          sourceLang: input.sourceLang,
          glossary: input.glossary,
          promptTemplateId: input.promptTemplateId,
          lineSafeMode: input.lineSafeMode,
          modelStrategy: input.modelStrategy,
          promptStyle: profile.effectivePromptStyle,
          strictMode: false,
          translationQualityMode: input.translationQualityMode,
          promptOverride: input.prompt,
          disableSystemPrompt: false,
        });
        return { prompt };
      });
    if (entries.length === 0) return [];

    const counts = await OpenvinoRuntimeManager.countHfInputTokens({
      modelPath,
      entries,
    });
    return counts.map((value) => Math.max(0, Math.floor(Number(value) || 0)));
  }

  private static estimateLocalBatchOutputTokens(input: {
    text: string;
    lineCount: number;
    localModel?: LocalModelDefinition;
  }) {
    const estimatedUnits = String(input.text || '').trim()
      ? Math.ceil(
          (String(input.text || '').match(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu) || [])
            .length +
            Math.max(
              0,
              String(input.text || '').replace(/\s+/g, ' ').length -
                (String(input.text || '').match(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu) || [])
                  .length
            ) /
              4
        )
      : 0;
    const base = Math.ceil(estimatedUnits * 1.25) + input.lineCount * 8;
    const cap = input.localModel?.runtimeHints?.maxOutputTokens;
    return Math.max(64, Math.min(cap || 8192, base));
  }

  private static async splitLineSafeUnitsForLocalTranslation(
    units: LineSafeUnit[],
    input?: {
      localModel?: LocalModelDefinition;
      modelStrategy?: LocalTranslateModelStrategy;
      targetLang?: string;
      sourceLang?: string;
      glossary?: string;
      prompt?: string;
      promptTemplateId?: string;
      translationQualityMode?: TranslationQualityMode;
      batchDebugSource?: LocalTranslationBatchDebugInfo['source'];
      onBatchPlan?: (plan: LocalTranslationBatchDebugInfo) => void;
    }
  ) {
    const caps = this.getLocalBatchCaps(input);
    const debugSource = input?.batchDebugSource || 'local';
    const emitBatchPlan = (plan: LocalTranslationBatchDebugInfo) => {
      input?.onBatchPlan?.(plan);
    };
    const fixedBatches = (fallbackReason?: string | null) => {
      const batches = this.splitLineSafeUnitsByLineAndChar(units, caps);
      emitBatchPlan(
        this.buildLocalBatchDebugInfo({
          source: debugSource,
          mode: 'fixed_lines',
          batches,
          maxLines: caps.maxLines,
          charBudget: caps.charBudget,
          fallbackReason,
        })
      );
      return batches;
    };

    if (units.length <= 1) return fixedBatches();

    const globalBatchingMode = this.getLocalBatchingMode();
    if (this.isFixedLineBatchingMode(globalBatchingMode)) {
      return fixedBatches();
    }

    const modelHints = input?.localModel?.runtimeHints;
    const hintBatching = modelHints?.batching;
    const envTokenBudget = Math.round(
      this.getEnvNumber(
        'TRANSLATE_LOCAL_INPUT_TOKEN_BUDGET',
        hintBatching?.inputTokenBudget || caps.batchingProfile?.tokenBudget || 0,
        0,
        1_000_000
      )
    );
    const contextWindow =
      typeof modelHints?.contextWindow === 'number' && Number.isFinite(modelHints.contextWindow)
        ? Math.floor(modelHints.contextWindow)
        : 0;
    const outputBudget =
      typeof hintBatching?.outputTokenBudget === 'number'
        ? Math.floor(hintBatching.outputTokenBudget)
        : typeof modelHints?.maxOutputTokens === 'number'
          ? Math.floor(modelHints.maxOutputTokens)
          : 0;
    const safetyReserve =
      typeof hintBatching?.safetyReserveTokens === 'number'
        ? Math.floor(hintBatching.safetyReserveTokens)
        : contextWindow > 0
          ? Math.max(128, Math.min(1024, Math.ceil(contextWindow * 0.08)))
          : 0;
    const derivedTokenBudget =
      envTokenBudget > 0
        ? envTokenBudget
        : contextWindow > 0 && outputBudget > 0
          ? Math.max(128, contextWindow - outputBudget - safetyReserve)
          : 0;
    const shouldUseTokenAware =
      globalBatchingMode === 'token_aware' ||
      globalBatchingMode === 'token-aware' ||
      (globalBatchingMode === '' || globalBatchingMode === 'auto'
        ? Boolean(input?.localModel && input?.modelStrategy && input?.targetLang && derivedTokenBudget > 0)
        : false);

    if (!shouldUseTokenAware || !input?.localModel || !input.modelStrategy || !input.targetLang || derivedTokenBudget <= 0) {
      return fixedBatches();
    }

    const batches: LineSafeUnit[][] = [];
    const selectedPromptTokens: number[] = [];
    const selectedEstimatedOutputTokens: number[] = [];
    let currentStart = 0;
    try {
      while (currentStart < units.length) {
        const candidateSourceTexts: string[] = [];
        const candidateLineCounts: number[] = [];
        const candidateCharCounts: number[] = [];
        const candidateOutputBudgets: number[] = [];

        let mergedText = '';
        let mergedChars = 0;
        for (let end = currentStart; end < units.length; end += 1) {
          const lineCount = end - currentStart + 1;
          if (lineCount > caps.maxLines) {
            break;
          }
          const candidateUnits = units.slice(currentStart, end + 1);
          mergedText = this.buildLineSafeInput(candidateUnits);
          mergedChars = mergedText.length + 1;
          if (lineCount > 1 && mergedChars > caps.charBudget) {
            break;
          }
          candidateSourceTexts.push(mergedText);
          candidateLineCounts.push(lineCount);
          candidateCharCounts.push(mergedChars);
          candidateOutputBudgets.push(
            this.estimateLocalBatchOutputTokens({
              text: this.buildSourceChunkText(candidateUnits),
              lineCount: candidateUnits.length,
              localModel: input.localModel,
            })
          );
          if (mergedChars > caps.charBudget) {
            break;
          }
        }

        const candidateTokenCounts = await this.countLocalTranslationInputTokensBatch({
          localModel: input.localModel,
          modelStrategy: input.modelStrategy,
          sourceTexts: candidateSourceTexts,
          targetLang: input.targetLang,
          sourceLang: input.sourceLang,
          glossary: input.glossary,
          prompt: input.prompt,
          promptTemplateId: input.promptTemplateId,
          lineSafeMode: true,
          translationQualityMode: input.translationQualityMode,
        });

        let bestRelativeEnd = -1;
        for (let index = 0; index < candidateSourceTexts.length; index += 1) {
          const lineCount = candidateLineCounts[index];
          const charCount = candidateCharCounts[index];
          const promptTokens = candidateTokenCounts[index] ?? Number.POSITIVE_INFINITY;
          const estimatedOutputTokens = candidateOutputBudgets[index] || outputBudget || 0;
          const totalTokens = promptTokens + estimatedOutputTokens + safetyReserve;
          const exceedsTokenBudget =
            promptTokens > derivedTokenBudget ||
            (contextWindow > 0 && totalTokens > contextWindow);
          if (lineCount > caps.maxLines || charCount > caps.charBudget || exceedsTokenBudget) {
            break;
          }
          bestRelativeEnd = index;
        }

        if (bestRelativeEnd < 0) {
          bestRelativeEnd = 0;
        }

        const absoluteEnd = currentStart + bestRelativeEnd;
        batches.push(units.slice(currentStart, absoluteEnd + 1));
        selectedPromptTokens.push(Math.max(0, Math.floor(Number(candidateTokenCounts[bestRelativeEnd]) || 0)));
        selectedEstimatedOutputTokens.push(Math.max(0, Math.floor(Number(candidateOutputBudgets[bestRelativeEnd]) || 0)));
        currentStart = absoluteEnd + 1;
      }
      emitBatchPlan(
        this.buildLocalBatchDebugInfo({
          source: debugSource,
          mode: 'token_aware',
          batches,
          promptTokens: selectedPromptTokens,
          estimatedOutputTokens: selectedEstimatedOutputTokens,
          inputTokenBudget: derivedTokenBudget,
          outputTokenBudget: outputBudget || null,
          contextWindow: contextWindow || null,
          safetyReserveTokens: safetyReserve || null,
          maxLines: caps.maxLines,
          charBudget: caps.charBudget,
        })
      );
      return batches;
    } catch {
      return fixedBatches('token_aware_count_failed');
    }
  }

  private static async splitLineSafeUnitsForTranslateGemma(
    units: LineSafeUnit[],
    input: {
      localModel: LocalModelDefinition;
      modelStrategy: LocalTranslateModelStrategy;
      targetLang: string;
      sourceLang?: string;
      glossary?: string;
      prompt?: string;
      promptTemplateId?: string;
      translationQualityMode?: TranslationQualityMode;
      onBatchPlan?: (plan: LocalTranslationBatchDebugInfo) => void;
    }
  ) {
    if (units.length <= 1) {
      const batches = units.length > 0 ? [units] : [];
      input.onBatchPlan?.(
        this.buildLocalBatchDebugInfo({
          source: 'translategemma',
          mode: 'fixed_lines',
          batches,
        })
      );
      return batches;
    }

    const batchingMode = String(process.env.TRANSLATEGEMMA_BATCHING_MODE || '')
      .trim()
      .toLowerCase();
    const useFixedLineBatching =
      batchingMode === 'fixed_lines' ||
      batchingMode === 'fixed-line' ||
      batchingMode === 'line_char' ||
      batchingMode === 'line-char';
    if (useFixedLineBatching || this.isFixedLineBatchingMode(this.getLocalBatchingMode())) {
      return await this.splitLineSafeUnitsForLocalTranslation(units, {
        localModel: input.localModel,
        modelStrategy: input.modelStrategy,
        targetLang: input.targetLang,
        sourceLang: input.sourceLang,
        glossary: input.glossary,
        prompt: input.prompt,
        promptTemplateId: input.promptTemplateId,
        translationQualityMode: input.translationQualityMode,
        batchDebugSource: 'translategemma',
        onBatchPlan: input.onBatchPlan,
      });
    }

    const qualityMode = input.translationQualityMode || 'plain_probe';
    const countLineSafePrompt = isPlainTranslationProbeMode(qualityMode);
    const batchingProfile = resolveEffectiveLocalTranslationProfile({
      localModel: input.localModel,
      modelStrategy: input.modelStrategy,
      qualityMode,
    }).translationProfile.lineSafeBatching;
    const configuredMaxLines = Math.round(
      this.getEnvNumber('TRANSLATE_LOCAL_BATCH_SIZE', batchingProfile?.maxLines ?? 24, 2, 200)
    );
    const configuredCharBudget = Math.round(
      this.getEnvNumber('TRANSLATE_LOCAL_BATCH_CHAR_BUDGET', batchingProfile?.charBudget ?? 1400, 120, 40000)
    );
    const tokenBudget = Math.round(
      this.getEnvNumber('TRANSLATEGEMMA_INPUT_TOKEN_BUDGET', batchingProfile?.tokenBudget ?? 960, 128, 4096)
    );
    const allowProfileCapOverride = ['1', 'true', 'yes', 'on'].includes(
      String(process.env.TRANSLATE_LOCAL_ALLOW_PROFILE_CAP_OVERRIDE || '').trim().toLowerCase()
    );
    const maxLines = batchingProfile?.maxLines && !allowProfileCapOverride
      ? Math.min(configuredMaxLines, batchingProfile.maxLines)
      : configuredMaxLines;
    const charBudget = batchingProfile?.charBudget && !allowProfileCapOverride
      ? Math.min(configuredCharBudget, batchingProfile.charBudget)
      : configuredCharBudget;

    const batches: LineSafeUnit[][] = [];
    const selectedPromptTokens: number[] = [];
    let currentStart = 0;
    while (currentStart < units.length) {
      const candidateSourceTexts: string[] = [];
      const candidateLineCounts: number[] = [];
      const candidateCharCounts: number[] = [];

      let mergedText = '';
      let mergedChars = 0;
      for (let end = currentStart; end < units.length; end += 1) {
        const lineCount = end - currentStart + 1;
        if (lineCount > maxLines) {
          break;
        }
        const candidateUnits = units.slice(currentStart, end + 1);
        mergedText = countLineSafePrompt
          ? this.buildLineSafeInput(candidateUnits)
          : candidateUnits.map((unit) => this.stripStructuredPrefix(unit.content).trim()).join('\n');
        mergedChars = mergedText.length + 1;
        if (lineCount > 1 && mergedChars > charBudget) {
          break;
        }
        candidateSourceTexts.push(mergedText);
        candidateLineCounts.push(lineCount);
        candidateCharCounts.push(mergedChars);
        if (mergedChars > charBudget) {
          break;
        }
      }

      const candidateTokenCounts = await this.countLocalTranslationInputTokensBatch({
        localModel: input.localModel,
        modelStrategy: input.modelStrategy,
        sourceTexts: candidateSourceTexts,
        targetLang: input.targetLang,
        sourceLang: input.sourceLang,
        glossary: input.glossary,
        prompt: countLineSafePrompt ? input.prompt : '',
        promptTemplateId: input.promptTemplateId,
        lineSafeMode: countLineSafePrompt,
        translationQualityMode: qualityMode,
      });

      let bestRelativeEnd = -1;
      for (let index = 0; index < candidateSourceTexts.length; index += 1) {
        const lineCount = candidateLineCounts[index];
        const charCount = candidateCharCounts[index];
        const tokenCount = candidateTokenCounts[index] ?? Number.POSITIVE_INFINITY;
        if (lineCount > maxLines || charCount > charBudget || tokenCount > tokenBudget) {
          break;
        }
        bestRelativeEnd = index;
      }

      if (bestRelativeEnd < 0) {
        bestRelativeEnd = 0;
      }

      const absoluteEnd = currentStart + bestRelativeEnd;
      batches.push(units.slice(currentStart, absoluteEnd + 1));
      selectedPromptTokens.push(Math.max(0, Math.floor(Number(candidateTokenCounts[bestRelativeEnd]) || 0)));
      currentStart = absoluteEnd + 1;
    }

    input.onBatchPlan?.(
      this.buildLocalBatchDebugInfo({
        source: 'translategemma',
        mode: 'token_aware',
        batches,
        promptTokens: selectedPromptTokens,
        inputTokenBudget: tokenBudget,
        maxLines,
        charBudget,
      })
    );

    return batches;
  }

  private static detectRepetitionLoop(text: string) {
    if (!text || text.length < 40) return false;
    for (let patternLength = 8; patternLength <= Math.floor(text.length / 2); patternLength += 1) {
      const head = text.slice(0, patternLength);
      const repeated = text.slice(patternLength, patternLength * 2);
      if (head && head === repeated) {
        return true;
      }
    }
    return false;
  }

  private static detectAdjacentDuplicate(text: string) {
    const plain = String(text || '');
    if (!plain.trim()) return false;

    const lines = plain.split('\n').map((line) => this.stripStructuredPrefix(this.stripLineMarker(line)).trim());
    const benignRepeatedTokens = new Set(['hello', 'hi', 'hey']);
    for (const line of lines) {
      if (!line) continue;
      const repeatedWordMatch = line.match(/\b([\p{L}\p{N}]{2,})\s+\1\b/iu);
      if (repeatedWordMatch) {
        const token = String(repeatedWordMatch[1] || '').toLowerCase();
        if (!benignRepeatedTokens.has(token)) {
          return true;
        }
      }

      const cjkNormalized = line
        .replace(/[^\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}\s]/gu, ' ')
        .replace(/\s+/g, ' ')
        .trim();
      const cjkCompact = cjkNormalized.replace(/\s+/g, '');
      if (
        cjkNormalized &&
        (/([\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]{2,10})\s+\1/u.test(cjkNormalized) ||
          /([\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]{2,10})\1/u.test(cjkCompact))
      ) {
        return true;
      }
    }

    return false;
  }

  private static parseLocalTranslatedText(raw: string, lineSafeMode = false) {
    let text = String(raw || '').trim();
    if (!text) return '';

    const fenced = text.match(/^```(?:text|markdown)?\s*([\s\S]*?)\s*```$/i);
    if (fenced?.[1]) {
      text = String(fenced[1] || '').trim();
    }

    text = text
      .replace(/<think>[\s\S]*?<\/think>/gi, '')
      .replace(/<\/?think>/gi, '')
      .replace(/<\|im_start\|>assistant/gi, '')
      .replace(/<\|im_start\|>user/gi, '')
      .replace(/<\|im_start\|>system/gi, '')
      .replace(/<\|im_end\|>/gi, '')
      .trim();

    text = text
      .split('\n')
      .filter((line) => !/^\s*(analysis|reasoning)\s*:/i.test(line))
      .join('\n')
      .trim();

    text = text.replace(/^(translation|output)\s*:\s*/i, '').trim();
    text = this.stripInjectedSpeakerContext(text);

    if (lineSafeMode) {
      const markerLines = text
        .split('\n')
        .map((line) => this.stripInjectedSpeakerContext(line.trim()))
        .filter((line) => /\[\[L\d{5}\]\]/.test(line));
      if (markerLines.length > 0) {
        return markerLines.join('\n');
      }
    }

    return text;
  }

  private static normalizeTraditionalChineseOutput(text: string) {
    return ChineseScriptNormalizer.normalizeToTaiwanTraditional(text);
  }

  private static normalizeTargetLanguageOutput(text: string, targetLang: string) {
    const normalized = normalizeLanguageKey(targetLang);
    if (normalized === 'zh-tw') {
      return this.normalizeTraditionalChineseOutput(text);
    }
    return String(text || '');
  }

  private static normalizeComparisonText(text: string) {
    return String(text || '')
      .split('\n')
      .map((line) => this.stripStructuredPrefix(this.stripLineMarker(this.stripInjectedSpeakerContext(line))))
      .map((line) => line.replace(/\s+/g, ' ').trim().toLowerCase())
      .filter(Boolean);
  }

  private static normalizeLooseComparisonText(text: string) {
    return this.normalizeComparisonText(text)
      .map((line) => line.replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\s+/g, ' ').trim())
      .filter(Boolean);
  }

  private static calculateLineTokenOverlap(a: string, b: string) {
    const tokensA = new Set(String(a || '').split(/\s+/).filter(Boolean));
    const tokensB = new Set(String(b || '').split(/\s+/).filter(Boolean));
    if (tokensA.size === 0 || tokensB.size === 0) return 0;
    let overlap = 0;
    for (const token of tokensA) {
      if (tokensB.has(token)) overlap += 1;
    }
    return overlap / Math.max(tokensA.size, tokensB.size);
  }

  private static isLikelyTargetLanguageMismatch(translatedText: string, targetLang: string) {
    const normalizedTarget = normalizeLanguageKey(targetLang);
    const plain = this.normalizeComparisonText(translatedText).join(' ');
    if (!plain) return false;

    const latinCount = (plain.match(/[A-Za-z]/g) || []).length;
    const hanCount = (plain.match(/[\p{Script=Han}]/gu) || []).length;
    const kanaCount = (plain.match(/[\p{Script=Hiragana}\p{Script=Katakana}]/gu) || []).length;
    const hangulCount = (plain.match(/[\p{Script=Hangul}]/gu) || []).length;
    const cyrillicCount = (plain.match(/[\p{Script=Cyrillic}]/gu) || []).length;
    const thaiCount = (plain.match(/[\p{Script=Thai}]/gu) || []).length;
    const arabicCount = (plain.match(/[\p{Script=Arabic}]/gu) || []).length;

    if (normalizedTarget === 'zh-tw' || normalizedTarget === 'zh-cn') {
      if (kanaCount >= 1 || hangulCount >= 1) {
        return true;
      }
      return hanCount < 2 && latinCount >= 4;
    }
    if (normalizedTarget === 'en' || normalizedTarget === 'english') {
      const nonLatinScriptCount = hanCount + kanaCount + hangulCount + cyrillicCount + thaiCount + arabicCount;
      const words = Array.from(
        plain
          .toLowerCase()
          .matchAll(/\b[a-z][a-z']*\b/g)
      ).map((match) => match[0]);
      const englishHits = this.countStopwordHits(words, this.getLanguageStopwords('en'));
      if (nonLatinScriptCount >= 4 && latinCount <= nonLatinScriptCount) {
        return true;
      }
      if (words.length >= 10 && englishHits === 0) {
        return true;
      }
      return false;
    }
    if (normalizedTarget === 'ja' || normalizedTarget === 'jp') {
      return hanCount + kanaCount < 2 && latinCount >= 4;
    }
    if (normalizedTarget === 'ko' || normalizedTarget === 'kr') {
      return hangulCount < 2 && latinCount >= 4;
    }

    if (['fi', 'es', 'de', 'pt', 'it', 'fr'].includes(normalizedTarget)) {
      const words = Array.from(
        plain
          .toLowerCase()
          .matchAll(/\b[\p{L}][\p{L}'’-]*\b/gu)
      ).map((match) => match[0]);
      if (words.length >= 6) {
        const englishHits = this.countStopwordHits(words, this.getLanguageStopwords('en'));
        const targetHits = this.countStopwordHits(words, this.getLanguageStopwords(normalizedTarget));
        const englishRatio = englishHits / words.length;
        if (
          englishHits >= 2 &&
          englishRatio >= 0.18 &&
          targetHits === 0
        ) {
          return true;
        }
        if (
          englishHits >= 4 &&
          englishHits >= targetHits + 3 &&
          targetHits <= 1
        ) {
          return true;
        }
      }
    }

    return false;
  }

  private static isLikelyPassThroughTranslation(sourceText: string, translatedText: string, targetLang: string) {
    const sourceLines = this.normalizeComparisonText(sourceText);
    const translatedLines = this.normalizeComparisonText(translatedText);
    if (sourceLines.length === 0 || translatedLines.length === 0) return false;
    if (sourceLines.length >= 2 && translatedLines.length < sourceLines.length) return true;

    const sourceProfile = this.analyzeSourceTextProfile(sourceText);
    const normalizedTarget = normalizeLanguageKey(targetLang);
    const sourceIsEastAsianNonChinese =
      sourceProfile.likelyJapanese ||
      sourceProfile.likelyKorean;
    const targetIsChinese =
      normalizedTarget === 'zh' ||
      normalizedTarget === 'zh-tw' ||
      normalizedTarget === 'zh-hant' ||
      normalizedTarget === 'zh-hk' ||
      normalizedTarget === 'zh-mo' ||
      normalizedTarget === 'zh-cn' ||
      normalizedTarget === 'zh-hans';

    const total = Math.min(sourceLines.length, translatedLines.length);
    let identicalCount = 0;
    let highOverlapCount = 0;
    let currentLeakRun = 0;
    let longestLeakRun = 0;
    const looseSourceLines = this.normalizeLooseComparisonText(sourceText);
    const looseTranslatedLines = this.normalizeLooseComparisonText(translatedText);
    for (let i = 0; i < total; i += 1) {
      const identical = sourceLines[i] === translatedLines[i];
      let highOverlap = this.calculateLineTokenOverlap(looseSourceLines[i] || '', looseTranslatedLines[i] || '') >= 0.92;
      if (sourceIsEastAsianNonChinese && targetIsChinese && highOverlap && !identical) {
        const translatedLine = translatedLines[i] || '';
        const sourceScriptLeaked =
          (sourceProfile.likelyJapanese && /[\p{Script=Hiragana}\p{Script=Katakana}]/u.test(translatedLine)) ||
          (sourceProfile.likelyKorean && /[\p{Script=Hangul}]/u.test(translatedLine));
        highOverlap = sourceScriptLeaked;
      }
      if (identical) identicalCount += 1;
      if (highOverlap) {
        highOverlapCount += 1;
      }
      if (identical || highOverlap) {
        currentLeakRun += 1;
        if (currentLeakRun > longestLeakRun) longestLeakRun = currentLeakRun;
      } else {
        currentLeakRun = 0;
      }
    }

    const identicalRatio = identicalCount / total;
    const overlapRatio = highOverlapCount / total;
    return (
      (total <= 4 && (identicalCount >= 1 || highOverlapCount >= 1)) ||
      (total <= 8 && (identicalCount >= 2 || highOverlapCount >= 2)) ||
      (total >= 2 && longestLeakRun >= 2) ||
      (identicalCount >= 1 && (identicalRatio >= 0.75 || (total >= 3 && identicalCount >= total - 1))) ||
      (highOverlapCount >= 1 && overlapRatio >= 0.85)
    );
  }

  private static needsZhTwNaturalization(sourceText: string, translatedText: string, context: TranslationQualityAssessmentContext = {}) {
    const profile = context.sourceProfile || this.analyzeSourceTextProfile(sourceText);
    if (!profile.likelyChinese) return false;

    const sourceLines = this.normalizeComparisonText(this.normalizeTraditionalChineseOutput(sourceText));
    const translatedLines = this.normalizeComparisonText(this.normalizeTraditionalChineseOutput(translatedText));
    if (sourceLines.length === 0 || translatedLines.length === 0) return false;

    const total = Math.min(sourceLines.length, translatedLines.length);
    let identicalCount = 0;
    for (let i = 0; i < total; i += 1) {
      if ((sourceLines[i] || '') === (translatedLines[i] || '')) {
        identicalCount += 1;
      }
    }

    const identicalRatio = total > 0 ? identicalCount / total : 0;
    return (
      (total <= 3 && identicalCount >= 1) ||
      (total >= 2 && identicalRatio >= 0.7) ||
      (identicalCount >= Math.max(2, total - 1))
    );
  }

  private static getTranslationQualityIssues(
    sourceText: string,
    translatedText: string,
    targetLang: string,
    context: TranslationQualityAssessmentContext = {}
  ) {
    const issues: TranslationQualityIssueCode[] = [];
    const normalizedSource = String(sourceText || '');
    const normalizedTranslated = String(translatedText || '');
    const trimmedTranslated = normalizedTranslated.trim();
    const sourceProfile = context.sourceProfile || this.analyzeSourceTextProfile(normalizedSource);

    if (!trimmedTranslated) {
      issues.push('empty_output');
    }
    if (this.detectRepetitionLoop(normalizedTranslated)) {
      issues.push('repetition_loop');
    }
    if (this.detectAdjacentDuplicate(normalizedTranslated) && !this.detectAdjacentDuplicate(normalizedSource)) {
      issues.push('adjacent_duplicate');
    }
    if (this.isLikelyPassThroughTranslation(normalizedSource, normalizedTranslated, targetLang)) {
      issues.push('pass_through');
    }
    if (
      normalizeLanguageKey(targetLang) === 'zh-tw' &&
      this.needsZhTwNaturalization(normalizedSource, normalizedTranslated, { ...context, sourceProfile })
    ) {
      issues.push('zh_tw_naturalization_needed');
    }
    if (this.isLikelyTargetLanguageMismatch(normalizedTranslated, targetLang)) {
      issues.push('target_lang_mismatch');
    }

    const expectedLineCount = Math.max(0, Math.round(Number(context.expectedLineCount || 0)));
    if (expectedLineCount > 1) {
      const outputLineCount = this.normalizeComparisonText(normalizedTranslated).length;
      if (outputLineCount < expectedLineCount) {
        issues.push('line_count_loss');
      }
    }

    if (context.requireMarkers && !/\[\[L\d{5}\]\]/.test(normalizedTranslated)) {
      issues.push('marker_loss');
    }

    return Array.from(new Set(issues));
  }

  private static addQualityIssueWarnings(issues: TranslationQualityIssueCode[], addWarning: (code: string) => void) {
    for (const issue of issues) {
      addWarning(`quality_issue_${issue}`);
    }
  }

  private static buildTranslationQualityContext(sourceText: string, requireMarkers = false): TranslationQualityAssessmentContext {
    const expectedLineCount = String(sourceText || '').split('\n').length;
    return {
      expectedLineCount: expectedLineCount > 1 ? expectedLineCount : undefined,
      requireMarkers,
    };
  }

  private static mergeUniqueWarnings(...warningGroups: Array<string[] | undefined>) {
    const merged: string[] = [];
    for (const group of warningGroups) {
      for (const warning of group || []) {
        if (warning && !merged.includes(warning)) {
          merged.push(warning);
        }
      }
    }
    return merged;
  }

  private static getPassThroughRisk(warnings: string[]) {
    if (warnings.includes('quality_issue_pass_through')) return 'high' as const;
    if (warnings.includes('quality_retry_triggered')) return 'medium' as const;
    return 'low' as const;
  }

  private static getRepetitionRisk(warnings: string[]) {
    if (warnings.includes('quality_issue_repetition_loop') || warnings.includes('quality_issue_adjacent_duplicate')) {
      return 'high' as const;
    }
    return 'low' as const;
  }

  private static getMarkerPreservation(
    sourceHasStructuredPrefixes: boolean,
    warnings: string[]
  ): 'ok' | 'partial' | 'lost' | undefined {
    if (!sourceHasStructuredPrefixes) return undefined;
    return warnings.includes('quality_issue_marker_loss') ? 'lost' : 'ok';
  }

  private static buildTranslationQualitySummary(input: {
    sourceLineCount: number;
    sourceHasStructuredPrefixes: boolean;
    output: string;
    warnings: string[];
    qualityRetryCount: number;
  }) {
    const outputLineCount = this.normalizeComparisonText(input.output).length;
    return {
      lineCountMatch: input.sourceLineCount <= 1 ? true : outputLineCount >= input.sourceLineCount,
      targetLanguageMatch: !input.warnings.includes('quality_issue_target_lang_mismatch'),
      passThroughRisk: this.getPassThroughRisk(input.warnings),
      repetitionRisk: this.getRepetitionRisk(input.warnings),
      markerPreservation: this.getMarkerPreservation(input.sourceHasStructuredPrefixes, input.warnings),
      strictRetryTriggered: input.qualityRetryCount > 0,
    };
  }

  private static buildTranslationWarningIssues(warnings: string[]): RunIssue[] {
    const qualityInfoCodes = new Set([
      'strict_retry_applied',
      'local_strict_retry_applied',
      'local_batch_translation_applied',
      'local_recursive_chunk_split_applied',
      'local_single_line_retry_applied',
      'translategemma_batch_translation_applied',
      'translategemma_recursive_chunk_split_applied',
      'translategemma_single_line_retry_applied',
      'residual_line_retry_applied',
      'line_safe_alignment_applied',
      'line_index_rebind_applied',
      'line_json_map_repair_applied',
      'line_json_map_pre_split_applied',
      'line_json_map_policy_split',
      'cloud_provider_batch_translation_applied',
      'cloud_provider_batch_split_applied',
    ]);
    const qualityWarningCodes = new Set([
      'quality_retry_triggered',
      'post_repair_quality_retry_triggered',
      'residual_line_retry_triggered',
      'line_json_map_partial_fallback',
      'line_json_map_policy_single_line_fallback',
      'line_json_map_policy_source_fallback',
      'line_json_map_repair_disabled',
      'line_alignment_repair_failed',
      'local_repetition_loop_detected',
      'cloud_context_parse_failed',
      'cloud_context_chunk_split',
      'cloud_context_split_depth_exhausted',
      'cloud_context_single_line_fallback',
      'quality_issue_target_lang_mismatch',
      'quality_issue_pass_through',
      'quality_issue_empty_output',
      'quality_issue_line_count_loss',
      'quality_issue_marker_loss',
      'quality_issue_repetition_loop',
      'quality_issue_adjacent_duplicate',
      'quality_issue_zh_tw_naturalization_needed',
    ]);
    const providerInfoCodes = new Set(['transient_retry_applied']);
    const providerWarningCodes = new Set(['provider_fallback_applied']);

    return warnings.map((code) => {
      if (qualityInfoCodes.has(code)) {
        return { code, severity: 'info', area: 'quality' } satisfies RunIssue;
      }
      if (qualityWarningCodes.has(code)) {
        return { code, severity: 'warning', area: 'quality' } satisfies RunIssue;
      }
      if (providerInfoCodes.has(code)) {
        return { code, severity: 'info', area: 'provider' } satisfies RunIssue;
      }
      if (providerWarningCodes.has(code)) {
        return { code, severity: 'warning', area: 'provider' } satisfies RunIssue;
      }
      switch (code) {
        case 'quality_issue_target_lang_mismatch':
        case 'quality_issue_pass_through':
        case 'quality_issue_empty_output':
        case 'quality_issue_line_count_loss':
        case 'quality_issue_marker_loss':
        case 'quality_issue_repetition_loop':
        case 'quality_issue_adjacent_duplicate':
        case 'quality_issue_zh_tw_naturalization_needed':
          return { code, severity: 'warning', area: 'quality' } satisfies RunIssue;
        default:
          return { code, severity: 'warning', area: 'provider' } satisfies RunIssue;
      }
    });
  }

  private static buildTranslationErrorIssues(errors: { request: string | null }): RunIssue[] {
    if (!errors.request) return [];
    return [
      {
        code: 'translation.request.failed',
        severity: 'error',
        area: 'provider',
        technicalMessage: errors.request,
      },
    ];
  }

  private static shouldRetryForTranslationQuality(
    sourceText: string,
    translatedText: string,
    targetLang: string,
    context: TranslationQualityAssessmentContext = {}
  ) {
    return this.getTranslationQualityIssues(sourceText, translatedText, targetLang, context).length > 0;
  }

  private static buildStrictRetryIssueLines(
    targetLang: string,
    issues: TranslationQualityIssueCode[],
    context: TranslationQualityAssessmentContext = {}
  ) {
    if (issues.length === 0) return '';

    const issueSet = new Set(issues);
    const descriptor = this.buildTargetLanguageDescriptor(targetLang);
    const lines = [
      `The previous output had quality problems. Fix them in this retry and return only the corrected translation into ${descriptor}.`,
    ];

    if (issueSet.has('empty_output')) {
      lines.push('- The previous output was empty or missing usable translated text.');
    }
    if (issueSet.has('repetition_loop')) {
      lines.push('- The previous output repeated itself. Do not repeat phrases, lines, or blocks.');
    }
    if (issueSet.has('adjacent_duplicate')) {
      lines.push('- The previous output duplicated adjacent words or short phrases. Remove accidental duplicates such as repeated words.');
    }
    if (issueSet.has('pass_through')) {
      lines.push('- The previous output kept the source text unchanged or too close to the source. Fully translate it.');
    }
    if (issueSet.has('zh_tw_naturalization_needed')) {
      lines.push('- The previous output stayed too close to the original Chinese wording. Rewrite it into natural Taiwan Traditional Chinese subtitle wording instead of copying the source text.');
    }
    if (issueSet.has('target_lang_mismatch')) {
      lines.push(`- The previous output was not clearly in ${descriptor}. Make every translated line stay in the target language.`);
    }
    if (issueSet.has('line_count_loss') && context.expectedLineCount && context.expectedLineCount > 1) {
      lines.push(`- Keep exactly ${context.expectedLineCount} output lines. Do not drop, merge, or collapse lines.`);
    }
    if (issueSet.has('marker_loss')) {
      lines.push('- Preserve every line marker like [[L00001]] exactly as given.');
    }

    lines.push('Fix every problem above in this retry.');
    return lines.join('\n');
  }

  private static buildStrictRetryInstruction(
    targetLang: string,
    issues: TranslationQualityIssueCode[] = [],
    context: TranslationQualityAssessmentContext = {}
  ) {
    const localized = this.getLocalizedSystemPromptProfile(targetLang);
    const issueLines = this.buildStrictRetryIssueLines(targetLang, issues, context);
    if (localized) {
      return [
        localized.strictRetryLines.join('\n'),
        this.getTargetLanguageHardeningLines(targetLang, Boolean(context.requireMarkers), false).join('\n'),
        issueLines,
      ]
        .filter(Boolean)
        .join('\n');
    }
    return [
      `The output must be fully translated into ${this.buildTargetLanguageDescriptor(targetLang)}.`,
      'Do not leave the source sentence unchanged unless it is a proper noun, brand, code identifier, or unavoidable quoted term.',
      'If the previous output remained in the source language, translate it now.',
      'Return only the final translation.',
      ...this.getTargetLanguageHardeningLines(targetLang, Boolean(context.requireMarkers), false),
      issueLines,
    ].join('\n');
  }

  private static extractSpeakerTag(prefix: string) {
    const tokens = Array.from(String(prefix || '').matchAll(/\[([^\]]+)\]/g))
      .map((match) => String(match[1] || '').trim())
      .filter(Boolean);
    if (tokens.length === 0) return null;

    const speakerToken = [...tokens]
      .reverse()
      .find((token) => !/^\d{2}:\d{2}:\d{2}(?:[.,]\d{1,3})?$/i.test(token));
    return speakerToken || null;
  }

  private static buildInjectedSpeakerContext(speakerTag: string | null) {
    if (!speakerTag) return '';
    const safe = String(speakerTag).replace(/[<>]/g, '').trim();
    return safe ? `<<SPEAKER:${safe}>>` : '';
  }

  private static stripInjectedSpeakerContext(text: string) {
    return String(text || '').replace(/<<SPEAKER:[^>]+>>\s*/gi, '').trim();
  }

  private static buildSourceChunkText(units: LineSafeUnit[]) {
    return units
      .map((unit) => {
        const body = this.stripStructuredPrefix(unit.content);
        return unit.prefix ? `${unit.prefix}${body}` : body;
      })
      .join('\n');
  }

  private static splitLineSafeUnits(units: LineSafeUnit[]) {
    if (units.length <= 1) return [units];
    const midpoint = Math.ceil(units.length / 2);
    return [units.slice(0, midpoint), units.slice(midpoint)];
  }

  private static buildLocalTranslationOrchestratorDeps(): RunLocalTranslationOrchestratorDeps {
    return {
      throwIfAborted: this.throwIfAborted.bind(this),
      buildLineSafeUnits: this.buildLineSafeUnits.bind(this),
      buildSourceChunkText: this.buildSourceChunkText.bind(this),
      buildLineSafeInput: this.buildLineSafeInput.bind(this),
      splitLineSafeUnits: this.splitLineSafeUnits.bind(this),
      splitLineSafeUnitsForLocalTranslation: this.splitLineSafeUnitsForLocalTranslation.bind(this),
      splitLineSafeUnitsForTranslateGemma: this.splitLineSafeUnitsForTranslateGemma.bind(this),
      stripStructuredPrefix: this.stripStructuredPrefix.bind(this),
      stripInjectedSpeakerContext: this.stripInjectedSpeakerContext.bind(this),
      parseLocalTranslatedText: this.parseLocalTranslatedText.bind(this),
      parseLineSafeOutput: this.parseLineSafeOutput.bind(this),
      rebindByLineIndex: this.rebindByLineIndex.bind(this),
      normalizeTargetLanguageOutput: this.normalizeTargetLanguageOutput.bind(this),
      getTranslationQualityIssues: this.getTranslationQualityIssues.bind(this),
      addQualityIssueWarnings: this.addQualityIssueWarnings.bind(this),
      buildStrictRetryInstruction: this.buildStrictRetryInstruction.bind(this),
      buildTargetLanguageDescriptor: this.buildTargetLanguageDescriptor.bind(this),
      estimateLocalMaxNewTokens: this.estimateLocalMaxNewTokens.bind(this),
      buildLocalTranslationPrompt: this.buildLocalTranslationPrompt.bind(this),
      buildLocalTranslationMessages: this.buildLocalTranslationMessages.bind(this),
      buildLocalGenerationOptions: this.buildLocalGenerationOptions.bind(this),
      repairLineAlignmentWithLocalJsonMap: this.repairLineAlignmentWithLocalJsonMap.bind(this),
      isLikelyPassThroughTranslation: this.isLikelyPassThroughTranslation.bind(this),
      isPlainTranslationProbeMode: isPlainTranslationProbeMode,
      getTranslateRuntimeDebug: (modelId) => OpenvinoRuntimeManager.getTranslateRuntimeDebug(modelId),
    };
  }

  private static async translateWithLocalModel(
    input: {
      text: string;
      targetLang: string;
      sourceLang?: string;
      glossary?: string;
      prompt?: string;
      promptTemplateId?: string;
      enableJsonLineRepair?: boolean;
      signal?: AbortSignal;
    },
    localModel: LocalModelDefinition,
    onProgress?: TranslateProgressFn
  ): Promise<TranslationResult> {
    this.throwIfAborted(input.signal);
    const modelStrategy = this.inferLocalTranslateModelStrategy(localModel);
    const localProfile = resolveLocalOpenvinoProfile(localModel);
    const enableJsonLineRepair = input.enableJsonLineRepair !== false;
    const normalizedPromptTemplateId = this.normalizePromptTemplateId(input.promptTemplateId);
    const effectiveGlossary = this.buildEffectiveGlossary(input.targetLang, input.glossary, normalizedPromptTemplateId) || null;
    const effectiveLocalProfile = resolveEffectiveLocalTranslationProfile({
      localModel,
      modelStrategy,
      promptTemplateId: normalizedPromptTemplateId,
      prompt: input.prompt,
      enableJsonLineRepair,
    });
    const translationQualityMode = effectiveLocalProfile.intent.qualityMode;
    const localTranslationProfile = effectiveLocalProfile.translationProfile;
    const residualRetryLimit = Math.round(this.getEnvNumber('TRANSLATE_LOCAL_RESIDUAL_RETRY_LIMIT', 8, 0, 200));
    const jsonRepairMaxLinesBeforeSplit = Math.round(
      this.getEnvNumber('TRANSLATE_LOCAL_JSON_REPAIR_MAX_LINES_BEFORE_SPLIT', 32, 0, 400)
    );
    // Debug contract is preserved in the extracted orchestrator:
    // sourceHasSpeakerTags: sourceSpeakerTaggedLineCount > 0,

    return runLocalTranslationOrchestrator(
      {
        input: {
          ...input,
          effectiveGlossary,
          promptTemplateId: normalizedPromptTemplateId,
          enableJsonLineRepair,
        },
        localModel,
        modelStrategy,
        localProfile,
        localTranslationProfile,
        effectiveLocalProfile,
        translationQualityMode,
        residualRetryLimit,
        jsonRepairMaxLinesBeforeSplit,
      },
      this.buildLocalTranslationOrchestratorDeps(),
      onProgress
    );
  }

  private static buildLineSafeUnits(sourceText: string): LineSafeUnit[] {
    return String(sourceText || '').split('\n').map((line, index) => {
      const marker = `[[L${String(index + 1).padStart(5, '0')}]]`;
      const structured = line.match(/^((?:\[[^\]]+\]\s*)+)(.*)$/);
      const prefix = structured ? structured[1] : '';
      const content = structured ? structured[2] : line;
      const speakerTag = this.extractSpeakerTag(prefix);
      return {
        index: index + 1,
        marker,
        prefix,
        content,
        speakerTag,
      };
    });
  }

  private static buildLineSafeInput(units: LineSafeUnit[]) {
    return units
      .map((unit) => {
        const body = String(unit.content || '').trim();
        const speakerContext = this.buildInjectedSpeakerContext(unit.speakerTag);
        const contextPrefix = speakerContext ? `${speakerContext} ` : '';
        return body ? `${unit.marker} ${contextPrefix}${body}`.trim() : `${unit.marker} ${contextPrefix}`.trim();
      })
      .join('\n');
  }

  private static buildCloudContextChunks(units: LineSafeUnit[], targetLines: number, charBudget: number) {
    const chunks: CloudContextChunk[] = [];
    let offset = 0;

    while (offset < units.length) {
      let length = 0;
      let usedChars = 0;

      while (offset + length < units.length && length < targetLines) {
        const nextUnit = units[offset + length];
        const nextChars = this.stripStructuredPrefix(nextUnit.content).length;
        if (length > 0 && usedChars + nextChars > charBudget) break;
        usedChars += nextChars;
        length += 1;
      }

      if (length <= 0) length = 1;
      chunks.push({ start: offset, length });
      offset += length;
    }

    return chunks;
  }

  private static buildCloudContextInput(units: LineSafeUnit[], start: number, length: number, contextWindow: number) {
    const chunkStart = Math.max(0, start);
    const chunkEnd = Math.min(units.length, chunkStart + Math.max(1, length));
    const windowStart = Math.max(0, chunkStart - Math.max(0, contextWindow));
    const windowEnd = Math.min(units.length, chunkEnd + Math.max(0, contextWindow));
    const targetUnits = units.slice(chunkStart, chunkEnd);

    const text = units
      .slice(windowStart, windowEnd)
      .map((unit) => {
        const body = this.stripStructuredPrefix(unit.content);
        const speakerContext = this.buildInjectedSpeakerContext(unit.speakerTag);
        const content = [speakerContext, body].filter(Boolean).join(' ').trim();
        const label =
          unit.index >= chunkStart + 1 && unit.index <= chunkEnd ? `[TRANSLATE_${String(unit.index).padStart(5, '0')}]` : '[CONTEXT]';
        return content ? `${label} ${content}` : label;
      })
      .join('\n');

    return { text, targetUnits };
  }

  private static parseCloudContextOutput(output: string, units: LineSafeUnit[]) {
    const translatedMap = new Map<number, string>();

    String(output || '')
      .split('\n')
      .forEach((line) => {
        const matched = String(line).match(/\[TRANSLATE_(\d{5})\]\s*(.*)$/i);
        if (!matched) return;
        const index = Number(matched[1]);
        if (!Number.isFinite(index) || translatedMap.has(index)) return;
        translatedMap.set(index, this.stripInjectedSpeakerContext(String(matched[2] || '').trim()));
      });

    if (units.some((unit) => !translatedMap.has(unit.index))) {
      return null;
    }

    return units
      .map((unit) => {
        const translated = translatedMap.get(unit.index) || '';
        return unit.prefix ? `${unit.prefix}${translated}` : translated;
      })
      .join('\n');
  }

  private static parseLineSafeOutput(output: string, units: LineSafeUnit[]) {
    const markerMap = new Map<number, string>();

    String(output || '')
      .split('\n')
      .forEach((line) => {
        const matched = line.match(/\[\[L(\d{5})\]\]\s*(.*)$/);
        if (!matched) return;
        const index = Number(matched[1]);
        const content = this.stripInjectedSpeakerContext(String(matched[2] || '').trim());
        if (!Number.isFinite(index)) return;
        markerMap.set(index, content);
      });

    const missing = units.some((unit) => !markerMap.has(unit.index));
    if (missing) return null;

    return units
      .map((unit) => {
        const translated = markerMap.get(unit.index) || '';
        return unit.prefix ? `${unit.prefix}${translated}` : translated;
      })
      .join('\n');
  }

  private static normalizeLineId(raw: unknown) {
    if (typeof raw === 'number' && Number.isFinite(raw)) {
      return Math.floor(raw);
    }
    const text = String(raw || '').trim();
    if (!text) return null;
    const matched = text.match(/L?(\d{1,9})/i);
    if (!matched) return null;
    const value = Number(matched[1]);
    return Number.isFinite(value) ? value : null;
  }

  private static extractJsonPayload(raw: string) {
    let text = String(raw || '').trim();
    if (!text) return '';

    const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
    if (fenced?.[1]) {
      text = fenced[1].trim();
    }

    const firstObj = text.indexOf('{');
    const lastObj = text.lastIndexOf('}');
    if (firstObj >= 0 && lastObj > firstObj) {
      return text.slice(firstObj, lastObj + 1);
    }

    const firstArr = text.indexOf('[');
    const lastArr = text.lastIndexOf(']');
    if (firstArr >= 0 && lastArr > firstArr) {
      return text.slice(firstArr, lastArr + 1);
    }

    return text;
  }

  private static parseJsonLineRepairOutput(output: string, units: LineSafeUnit[]): JsonLineRepairResult | null {
    const payload = this.extractJsonPayload(output);
    if (!payload) return null;

    let parsed: any;
    try {
      parsed = JSON.parse(payload);
    } catch {
      return null;
    }

    const rows: Array<{ id: unknown; text: unknown }> = Array.isArray(parsed)
      ? parsed
      : Array.isArray(parsed?.lines)
        ? parsed.lines
        : Array.isArray(parsed?.items)
          ? parsed.items
          : [];

    const mapped = new Map<number, string>();

    if (rows.length > 0) {
      rows.forEach((row) => {
        const id = this.normalizeLineId((row as any)?.id);
        const text = String((row as any)?.text ?? '').trim();
        const cleaned = this.stripInjectedSpeakerContext(text);
        if (id && !mapped.has(id)) {
          mapped.set(id, cleaned);
        }
      });
    } else if (parsed && typeof parsed === 'object') {
      Object.entries(parsed).forEach(([key, value]) => {
        const id = this.normalizeLineId(key);
        if (!id || mapped.has(id)) return;
        mapped.set(id, this.stripInjectedSpeakerContext(String(value ?? '').trim()));
      });
    }

    if (mapped.size === 0) return null;

    let missingCount = 0;
    const text = units
      .map((unit) => {
        const translated = mapped.get(unit.index);
        if (translated == null) {
          missingCount += 1;
        }
        const value = translated ?? this.stripStructuredPrefix(unit.content);
        return unit.prefix ? `${unit.prefix}${value}` : value;
      })
      .join('\n');

    return { text, missingCount };
  }

  private static stripStructuredPrefix(line: string) {
    const matched = String(line || '').match(/^((?:\[[^\]]+\]\s*)+)(.*)$/);
    return matched ? String(matched[2] || '').trim() : String(line || '').trim();
  }

  private static stripLineMarker(line: string) {
    return String(line || '').replace(/\[\[L\d{5}\]\]\s*/g, '').trim();
  }

  private static rebindByLineIndex(units: LineSafeUnit[], translatedText: string) {
    const translatedLines = String(translatedText || '').split('\n');
    if (translatedLines.length !== units.length) return null;

    return units
      .map((unit, idx) => {
        const raw = translatedLines[idx] || '';
        const withoutMarker = this.stripLineMarker(raw);
        const content = this.stripStructuredPrefix(this.stripInjectedSpeakerContext(withoutMarker));
        return unit.prefix ? `${unit.prefix}${content}` : content;
      })
      .join('\n');
  }

  private static extractErrorMessage(rawText: string, fallback: string) {
    const text = String(rawText || '').trim();
    if (!text) return fallback;

    try {
      const parsed = JSON.parse(text);
      const nested = parsed?.error;
      const nestedMessage =
        typeof nested === 'string'
          ? nested
          : nested?.message || parsed?.message || parsed?.detail || parsed?.error_description;
      if (typeof nestedMessage === 'string' && nestedMessage.trim()) {
        return nestedMessage.trim();
      }
    } catch {
      // fall through
    }

    return text;
  }

  private static async fetchWithTimeout(url: string, init: RequestInit, timeoutMs = 120000, signal?: AbortSignal) {
    const controller = new AbortController();
    const combinedSignal =
      signal && typeof AbortSignal.any === 'function'
        ? AbortSignal.any([signal, controller.signal])
        : signal || controller.signal;
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      return await fetch(url, {
        ...init,
        signal: combinedSignal,
        redirect: 'error',
      });
    } finally {
      clearTimeout(timeout);
    }
  }

  private static parseOpenAiLikeContent(content: any) {
    if (typeof content === 'string') return content;
    if (!Array.isArray(content)) return '';
    return content
      .map((part: any) => {
        if (typeof part === 'string') return part;
        if (typeof part?.text === 'string') return part.text;
        return '';
      })
      .join('');
  }

  private static parseGeminiContent(data: any) {
    const parts = data?.candidates?.[0]?.content?.parts;
    if (!Array.isArray(parts)) return '';
    return parts.map((part: any) => (typeof part?.text === 'string' ? part.text : '')).join('');
  }

  private static parseAnthropicContent(data: any) {
    const blocks = data?.content;
    if (!Array.isArray(blocks)) return '';
    return blocks
      .map((b: any) => (b?.type === 'text' && typeof b?.text === 'string' ? b.text : ''))
      .join('');
  }

  private static parseOllamaContent(data: any) {
    if (typeof data?.response === 'string') return data.response;
    if (typeof data?.message?.content === 'string') return data.message.content;
    return '';
  }

  private static parseResponsesContent(data: any) {
    if (typeof data?.output_text === 'string') return data.output_text;
    if (Array.isArray(data?.output)) {
      return data.output
        .flatMap((item: any) => Array.isArray(item?.content) ? item.content : [])
        .map((part: any) => (part?.type?.includes('text') && typeof part?.text === 'string' ? part.text : ''))
        .join('');
    }
    return '';
  }

  private static hasAnthropicEnvelope(data: any) {
    return Array.isArray(data?.content) || (typeof data?.id === 'string' && data?.type === 'message');
  }

  private static hasGeminiEnvelope(data: any) {
    return Array.isArray(data?.candidates) || typeof data?.promptFeedback === 'object' || typeof data?.usageMetadata === 'object';
  }

  private static hasOllamaEnvelope(data: any) {
    return (
      typeof data?.done === 'boolean' ||
      typeof data?.model === 'string' ||
      typeof data?.response === 'string' ||
      typeof data?.message === 'object'
    );
  }

  private static hasOpenAiChatEnvelope(data: any) {
    return Array.isArray(data?.choices) || typeof data?.id === 'string';
  }

  private static hasResponsesEnvelope(data: any) {
    return Array.isArray(data?.output) || typeof data?.output_text === 'string' || typeof data?.id === 'string';
  }

  private static isProviderHttpError(error: unknown): error is ProviderHttpError {
    return error instanceof ProviderHttpError;
  }

  private static isRetryableStatus(status: number) {
    return status === 408 || status === 409 || status === 425 || status === 429 || status >= 500;
  }

  private static isRetryableError(error: unknown) {
    if (error instanceof ProviderHttpError) {
      return this.isRetryableStatus(error.status);
    }
    const message = error instanceof Error ? error.message : String(error || '');
    return /network|fetch|timeout|abort|econnreset|socket|temporarily/i.test(message);
  }

  private static isProviderContentFilterError(error: unknown) {
    if (!(error instanceof ProviderHttpError)) return false;
    if (![400, 403].includes(error.status)) return false;
    const detail = String(error.detail || error.message || '').toLowerCase();
    return /content management policy|content filter|content filtering|prompt triggering|response was filtered|safety/i.test(detail);
  }

  private static getCloudTranslationRetryConfig() {
    return {
      maxRetries: Math.round(this.getEnvNumber('TRANSLATE_MAX_RETRIES', 2, 0, 5)),
      baseRetryMs: Math.round(this.getEnvNumber('TRANSLATE_RETRY_BASE_MS', 800, 100, 10000)),
      rateLimitRetryMs: Math.round(this.getEnvNumber('TRANSLATE_RATE_LIMIT_RETRY_MS', 12000, 1000, 120000)),
    };
  }

  private static getRemoteTranslateBatchSize() {
    return Math.round(this.getEnvNumber('TRANSLATE_REMOTE_BATCH_SIZE', 60, 10, 200));
  }

  private static shouldFallbackToResponses(error: unknown) {
    if (!(error instanceof ProviderHttpError)) return false;
    if ([404, 405, 501].includes(error.status)) return true;
    const detail = String(error.detail || '').toLowerCase();
    return detail.includes('chat') && (detail.includes('unsupported') || detail.includes('not found'));
  }

  private static getOpenAiResponsesEndpoint(chatEndpoint: string) {
    const parsed = new URL(chatEndpoint);
    const lower = parsed.pathname.toLowerCase();
    if (lower.includes('/chat/completions')) {
      parsed.pathname = parsed.pathname.replace(/\/chat\/completions$/i, '/responses');
      return parsed.toString();
    }
    parsed.pathname = parsed.pathname.replace(/\/+$/, '');
    parsed.pathname = `${parsed.pathname}/v1/responses`.replace(/\/{2,}/g, '/');
    return parsed.toString();
  }

  private static getOllamaFallbackEndpoint(currentEndpoint: string, fallbackTarget: 'chat' | 'generate') {
    const parsed = new URL(currentEndpoint);
    const targetPath = fallbackTarget === 'chat' ? '/api/chat' : '/api/generate';
    if (parsed.pathname.toLowerCase().endsWith('/api/chat') || parsed.pathname.toLowerCase().endsWith('/api/generate')) {
      parsed.pathname = parsed.pathname.replace(/\/api\/chat$/i, targetPath).replace(/\/api\/generate$/i, targetPath);
      return parsed.toString();
    }
    parsed.pathname = `${parsed.pathname.replace(/\/+$/, '')}${targetPath}`.replace(/\/{2,}/g, '/');
    return parsed.toString();
  }

  private static async requestTranslationByProvider(
    provider: CloudTranslateProvider,
    endpointUrl: string,
    options: TranslateRequestOptions,
    onProgress?: TranslateProgressFn
  ): Promise<TranslateProviderResult> {
    const adapter = getCloudTranslateAdapter(provider);
    const deps: CloudTranslateAdapterDeps = {
      throwIfAborted: this.throwIfAborted.bind(this),
      fetchWithTimeout: this.fetchWithTimeout.bind(this),
      parseRetryAfterMs: this.parseRetryAfterMs.bind(this),
      extractErrorMessage: this.extractErrorMessage.bind(this),
      resolveSystemPrompt: this.resolveSystemPrompt.bind(this),
      normalizeDeepLTargetLanguage: this.normalizeDeepLTargetLanguage.bind(this),
      parseOpenAiLikeContent: this.parseOpenAiLikeContent.bind(this),
      parseGeminiContent: this.parseGeminiContent.bind(this),
      parseAnthropicContent: this.parseAnthropicContent.bind(this),
      parseOllamaContent: this.parseOllamaContent.bind(this),
      parseResponsesContent: this.parseResponsesContent.bind(this),
      hasAnthropicEnvelope: this.hasAnthropicEnvelope.bind(this),
      hasGeminiEnvelope: this.hasGeminiEnvelope.bind(this),
      hasOllamaEnvelope: this.hasOllamaEnvelope.bind(this),
      hasOpenAiChatEnvelope: this.hasOpenAiChatEnvelope.bind(this),
      hasResponsesEnvelope: this.hasResponsesEnvelope.bind(this),
      shouldFallbackToResponses: this.shouldFallbackToResponses.bind(this),
      getOpenAiResponsesEndpoint: this.getOpenAiResponsesEndpoint.bind(this),
      getOllamaFallbackEndpoint: this.getOllamaFallbackEndpoint.bind(this),
      makeProviderHttpError: (prefix, status, detail, retryAfterMs) =>
        new ProviderHttpError(prefix, status, detail, retryAfterMs),
      isProviderHttpError: this.isProviderHttpError.bind(this),
    };
    return adapter.request(endpointUrl, options, deps, onProgress);
  }

  private static async repairLineAlignmentWithJsonMap(
    provider: CloudTranslateProvider,
    endpointUrl: string,
    units: LineSafeUnit[],
    options: {
      targetLang: string;
      sourceLang?: string;
      glossary?: string;
      promptTemplateId?: string;
      key?: string;
      model?: string;
      modelOptions?: ApiModelRequestOptions;
    },
    onProgress?: TranslateProgressFn,
    signal?: AbortSignal
  ): Promise<JsonLineRepairResult | null> {
    if (units.length === 0) return null;

    const batchSize = Math.round(this.getEnvNumber('TRANSLATE_JSON_REPAIR_BATCH_SIZE', 80, 10, 200));
    const totalBatches = Math.max(1, Math.ceil(units.length / batchSize));
    const mergedRows: string[] = [];
    let totalMissing = 0;
    const warnings: string[] = [];
    const addWarning = (code: string) => {
      if (!warnings.includes(code)) warnings.push(code);
    };
    const { maxRetries, baseRetryMs, rateLimitRetryMs } = this.getCloudTranslationRetryConfig();

    const requestWithRetry = async (requestOptions: TranslateRequestOptions) => {
      let attempt = 0;
      while (true) {
        this.throwIfAborted(signal);
        try {
          return await this.requestTranslationByProvider(
            provider,
            endpointUrl,
            requestOptions,
            onProgress
          );
        } catch (error) {
          if (!this.isRetryableError(error) || attempt >= maxRetries) {
            throw error;
          }
          attempt += 1;
          addWarning('transient_retry_applied');
          const retryDelay =
            error instanceof ProviderHttpError && error.status === 429
              ? Math.max(error.retryAfterMs || 0, rateLimitRetryMs * attempt)
              : baseRetryMs * attempt;
          const status = error instanceof ProviderHttpError ? error.status : 'network';
          onProgress?.(`Retrying translation request (${attempt}/${maxRetries}) after transient error (${status})...`);
          await this.sleep(retryDelay, signal);
        }
      }
    };

    const translateSingleJsonFallback = async (unit: LineSafeUnit): Promise<JsonLineRepairResult> => {
      try {
        const result = await requestWithRetry({
          text: this.stripStructuredPrefix(unit.content),
          targetLang: options.targetLang,
          sourceLang: options.sourceLang,
          glossary: options.glossary,
          promptTemplateId: options.promptTemplateId,
          key: options.key,
          model: options.model,
          modelOptions: options.modelOptions,
          lineSafeMode: false,
          signal,
        });
        const normalized = this.normalizeTargetLanguageOutput(result.text, options.targetLang).replace(/\s*\n+\s*/g, ' ').trim();
        addWarning('line_json_map_policy_single_line_fallback');
        return {
          text: unit.prefix ? `${unit.prefix}${normalized}` : normalized,
          missingCount: 0,
        };
      } catch (singleError) {
        if (!this.isProviderContentFilterError(singleError)) {
          throw singleError;
        }
        addWarning('line_json_map_policy_source_fallback');
        const fallback = this.stripStructuredPrefix(unit.content);
        return {
          text: unit.prefix ? `${unit.prefix}${fallback}` : fallback,
          missingCount: 1,
        };
      }
    };

    const processBatch = async (
      batchUnits: LineSafeUnit[],
      batchIndex: number,
      batchCount: number
    ): Promise<JsonLineRepairResult | null> => {
      onProgress?.(`Repairing subtitle alignment with strict JSON mapping (${batchIndex}/${batchCount})...`);

      const payload = JSON.stringify({
        lines: batchUnits.map((unit) => ({
          id: `L${String(unit.index).padStart(5, '0')}`,
          text: this.stripStructuredPrefix(unit.content),
          speaker: unit.speakerTag || undefined,
        })),
      });

      let result: TranslateProviderResult;
      try {
        result = await requestWithRetry({
          text: payload,
          targetLang: options.targetLang,
          sourceLang: options.sourceLang,
          glossary: options.glossary,
          promptTemplateId: options.promptTemplateId,
          key: options.key,
          model: options.model,
          modelOptions: options.modelOptions,
          lineSafeMode: false,
          systemPromptOverride: this.buildJsonLineRepairPrompt(options.targetLang, options.glossary, options.promptTemplateId),
          jsonResponse: true,
          signal,
        });
      } catch (error) {
        if (!this.isProviderContentFilterError(error)) {
          throw error;
        }
        if (batchUnits.length > 1) {
          addWarning('line_json_map_policy_split');
          const midpoint = Math.ceil(batchUnits.length / 2);
          const left = await processBatch(batchUnits.slice(0, midpoint), batchIndex * 2 - 1, batchCount * 2);
          const right = await processBatch(batchUnits.slice(midpoint), batchIndex * 2, batchCount * 2);
          if (!left || !right) return null;
          return {
            text: [left.text, right.text].filter(Boolean).join('\n'),
            missingCount: left.missingCount + right.missingCount,
          };
        }
        return translateSingleJsonFallback(batchUnits[0]);
      }

      const repaired = this.parseJsonLineRepairOutput(result.text, batchUnits)
        ?? (() => {
          const markerRestored = this.parseLineSafeOutput(result.text, batchUnits);
          if (markerRestored == null) return null;
          return {
            text: markerRestored,
            missingCount: 0,
          } as JsonLineRepairResult;
        })();

      return repaired;
    };

    for (let offset = 0; offset < units.length; offset += batchSize) {
      this.throwIfAborted(signal);
      const batchUnits = units.slice(offset, offset + batchSize);
      const batchIndex = Math.floor(offset / batchSize) + 1;
      const repaired = await processBatch(batchUnits, batchIndex, totalBatches);

      if (!repaired) {
        return null;
      }

      totalMissing += repaired.missingCount;
      mergedRows.push(...repaired.text.split('\n'));
    }

    return {
      text: mergedRows.join('\n'),
      missingCount: totalMissing,
      warnings,
    };
  }

  private static async repairLineAlignmentWithLocalJsonMap(
    units: LineSafeUnit[],
    localModel: LocalModelDefinition,
    options: {
      targetLang: string;
      glossary?: string;
      promptTemplateId?: string;
      modelStrategy: LocalTranslateModelStrategy;
    },
    onProgress?: TranslateProgressFn,
    signal?: AbortSignal
  ): Promise<JsonLineRepairResult | null> {
    if (units.length === 0) return null;
    const resolvedProfile = resolveEffectiveLocalTranslationProfile({
      localModel,
      modelStrategy: options.modelStrategy,
      qualityMode: 'json_strict',
    }).translationProfile;

    const batchSize = Math.round(this.getEnvNumber('TRANSLATE_JSON_REPAIR_BATCH_SIZE', 80, 10, 200));
    const totalBatches = Math.max(1, Math.ceil(units.length / batchSize));
    const mergedRows: string[] = [];
    let totalMissing = 0;

    for (let offset = 0; offset < units.length; offset += batchSize) {
      this.throwIfAborted(signal);
      const batchUnits = units.slice(offset, offset + batchSize);
      const batchIndex = Math.floor(offset / batchSize) + 1;
      onProgress?.(`Repairing subtitle alignment with strict JSON mapping (${batchIndex}/${totalBatches})...`);

      const payload = JSON.stringify({
        lines: batchUnits.map((unit) => ({
          id: `L${String(unit.index).padStart(5, '0')}`,
          text: this.stripStructuredPrefix(unit.content),
          speaker: unit.speakerTag || undefined,
        })),
      });

      const prompt = this.buildLocalJsonRepairPrompt({
        payload,
        sourceText: batchUnits.map((unit) => this.stripStructuredPrefix(unit.content)).join('\n'),
        targetLang: options.targetLang,
        glossary: options.glossary,
        promptTemplateId: options.promptTemplateId,
        modelStrategy: options.modelStrategy,
        promptStyle: resolvedProfile.effectivePromptStyle,
      });

      const raw = await OpenvinoRuntimeManager.translateWithLocalModel({
        modelId: localModel.id,
        modelPath: getLocalModelInstallDir(localModel),
        prompt,
        maxNewTokens: this.estimateLocalMaxNewTokens({
          text: payload,
          lineCount: batchUnits.length,
          lineSafeMode: false,
          localModel,
          jsonMode: true,
          strictMode: true,
        }),
        generation: this.buildLocalGenerationOptions({
          localModel,
          modelStrategy: options.modelStrategy,
          jsonMode: true,
          targetLang: options.targetLang,
          strictMode: true,
          translationQualityMode: 'json_strict',
        }),
      });

      const normalized = this.parseLocalTranslatedText(raw, false);
      const repaired = this.parseJsonLineRepairOutput(normalized, batchUnits)
        ?? (() => {
          const markerRestored = this.parseLineSafeOutput(normalized, batchUnits);
          if (markerRestored == null) return null;
          return {
            text: markerRestored,
            missingCount: 0,
          } as JsonLineRepairResult;
        })();

      if (!repaired) {
        return null;
      }

      totalMissing += repaired.missingCount;
      mergedRows.push(...repaired.text.split('\n'));
    }

    return {
      text: mergedRows.join('\n'),
      missingCount: totalMissing,
    };
  }

  private static mapConnectionError(error: unknown) {
    const message = error instanceof Error ? error.message : String(error || '');

    if (error instanceof ProviderHttpError) {
      if (error.status === 401 || error.status === 403) return `Authentication failed (${error.status}). ${error.detail}`;
      if (error.status === 404) return `Endpoint not found (404). Please verify API URL path. ${error.detail}`;
      if (error.status === 405) return `Method not allowed (405). Please verify this endpoint supports POST. ${error.detail}`;
      if (error.status === 429) return `Rate limited (429). Please retry later. ${error.detail}`;
      if (error.status >= 500) return `Provider server error (${error.status}). ${error.detail}`;
      return `Request failed (${error.status}). ${error.detail}`;
    }

    if (/abort|timeout/i.test(message)) return 'Connection timeout.';
    return message || 'Connection failed.';
  }

  private static buildCloudTranslationOrchestratorInput(
    input: {
      text: string;
      targetLang: string;
      sourceLang?: string;
      glossary?: string;
      prompt?: string;
      promptTemplateId?: string;
      enableJsonLineRepair: boolean;
      isConnectionTest?: boolean;
      signal?: AbortSignal;
    },
    resolvedProvider: ResolvedCloudTranslateProvider,
    model: { key?: string; model?: string; options?: ApiModelRequestOptions }
  ): CloudTranslationOrchestratorInput {
    const qualityMode = this.resolveTranslationQualityModeForRequest({
      promptTemplateId: input.promptTemplateId,
      prompt: input.prompt,
      enableJsonLineRepair: input.enableJsonLineRepair,
    });
    return {
      text: input.text,
      targetLang: input.targetLang,
      sourceLang: input.sourceLang,
      glossary: input.glossary,
      prompt: input.prompt,
      promptTemplateId: input.promptTemplateId,
      enableJsonLineRepair: input.enableJsonLineRepair,
      qualityMode,
      supportsContextMode: supportsCloudContextStrategy(resolvedProvider.provider),
      isConnectionTest: input.isConnectionTest,
      providerRequest: {
        provider: resolvedProvider.provider,
        endpointUrl: resolvedProvider.endpointUrl,
        key: model.key,
        model: resolvedProvider.effectiveModel || model.model,
        modelOptions: model.options,
      },
      providerBatching: this.getNvidiaCloudBatchConfig(resolvedProvider),
      signal: input.signal,
    };
  }

  private static buildCloudTranslationOrchestratorDeps(): CloudTranslationOrchestratorDeps {
    return {
      getCloudContextConfig: this.getCloudContextConfig.bind(this),
      getRemoteBatchSize: this.getRemoteTranslateBatchSize.bind(this),
      getRetryConfig: this.getCloudTranslationRetryConfig.bind(this),
      throwIfAborted: this.throwIfAborted.bind(this),
      sleep: this.sleep.bind(this),
      mapConnectionError: this.mapConnectionError.bind(this),
      isRetryableError: this.isRetryableError.bind(this),
      isProviderHttpError: this.isProviderHttpError.bind(this),
      isProviderContentFilterError: this.isProviderContentFilterError.bind(this),
      requestTranslationByProvider: this.requestTranslationByProvider.bind(this),
      buildLineSafeUnits: this.buildLineSafeUnits.bind(this),
      stripStructuredPrefix: this.stripStructuredPrefix.bind(this),
      buildLineSafeInput: this.buildLineSafeInput.bind(this),
      parseLineSafeOutput: this.parseLineSafeOutput.bind(this),
      normalizeTargetLanguageOutput: this.normalizeTargetLanguageOutput.bind(this),
      repairLineAlignmentWithJsonMap: this.repairLineAlignmentWithJsonMap.bind(this),
      rebindByLineIndex: this.rebindByLineIndex.bind(this),
      buildCloudContextInput: this.buildCloudContextInput.bind(this),
      buildCloudContextSystemPrompt: this.buildCloudContextSystemPrompt.bind(this),
      parseCloudContextOutput: this.parseCloudContextOutput.bind(this),
      buildCloudContextChunks: this.buildCloudContextChunks.bind(this),
    };
  }

  private static buildCloudTranslationDebugInfo(input: {
    configuredModelId?: string;
    sourceText: string;
    sourceLang?: string;
    targetLang: string;
    glossary?: string;
    prompt?: string;
    promptTemplateId?: string;
    enableJsonLineRepair: boolean;
    isConnectionTest?: boolean;
    resolvedProvider: ResolvedCloudTranslateProvider;
    orchestrated: CloudTranslationOrchestratorResult;
    qualityRetryCount: number;
    strictRetrySucceeded: boolean;
    output: string;
    elapsedMs: number;
    qualityRetryMs: number | null;
  }): TranslationDebugInfo {
    const qualityMode = this.resolveTranslationQualityModeForRequest({
      promptTemplateId: input.promptTemplateId,
      prompt: input.prompt,
      enableJsonLineRepair: input.enableJsonLineRepair,
    });
    const warnings = input.orchestrated.warnings;
    const errors = {
      request: input.orchestrated.lastError,
    };
    return {
      requested: {
        sourceLang: input.sourceLang ? String(input.sourceLang) : undefined,
        sourceLanguageDescriptor: this.buildSourceLanguageDescriptor(input.sourceLang),
        targetLang: input.targetLang,
        targetLanguageDescriptor: this.buildTargetLanguageDescriptor(input.targetLang),
        lineCount: input.orchestrated.sourceLineCount,
        charCount: input.sourceText.length,
        hasGlossary: Boolean(input.glossary && input.glossary.trim()),
        effectiveGlossary: this.buildEffectiveGlossary(input.targetLang, input.glossary, input.promptTemplateId) || null,
        hasPrompt: Boolean(String(input.prompt || '').trim()),
        promptTemplateId: input.promptTemplateId ? String(input.promptTemplateId) : null,
        jsonLineRepairEnabled: input.enableJsonLineRepair,
        sourceHasStructuredPrefixes: input.orchestrated.sourceHasStructuredPrefixes,
        sourceHasSpeakerTags: input.orchestrated.sourceSpeakerTaggedLineCount > 0,
        sourceSpeakerTaggedLineCount: input.orchestrated.sourceSpeakerTaggedLineCount,
      },
      provider: {
        name: input.resolvedProvider.provider,
        modelId: input.configuredModelId ? String(input.configuredModelId) : null,
        model: input.resolvedProvider.effectiveModel,
        endpoint: redactUrlSecrets(input.orchestrated.endpointUsed),
        adapterKey: input.resolvedProvider.adapterKey,
        profileId: input.resolvedProvider.profileId,
        profileFamily: input.resolvedProvider.profileFamily,
      },
      applied: {
        retryCount: input.orchestrated.retryCount,
        fallback: input.orchestrated.fallbackUsed,
        fallbackType: input.orchestrated.fallbackType,
        translationQualityMode: qualityMode,
        qualityRetryCount: input.qualityRetryCount,
        strictRetrySucceeded: input.strictRetrySucceeded,
        relaxedWholeRequest: input.orchestrated.relaxedWholeRequestApplied,
        relaxedWholeRequestFallback: input.orchestrated.relaxedWholeRequestFallback,
        cloudStrategy: input.orchestrated.cloudStrategy,
        cloudContextChunkCount: input.orchestrated.cloudContextChunkCount,
        cloudContextFallbackCount: input.orchestrated.cloudContextFallbackCount,
        cloudBatching: input.orchestrated.cloudBatching,
      },
      quality: this.buildTranslationQualitySummary({
        sourceLineCount: input.orchestrated.sourceLineCount,
        sourceHasStructuredPrefixes: input.orchestrated.sourceHasStructuredPrefixes,
        output: input.output,
        warnings,
        qualityRetryCount: input.qualityRetryCount,
      }),
      stats: {
        outputLineCount: input.output.split('\n').length,
        outputCharCount: input.output.length,
      },
      timing: {
        elapsedMs: input.elapsedMs,
        elapsedSec: Number((input.elapsedMs / 1000).toFixed(3)),
        providerMs: null,
        providerSec: null,
        repairMs: null,
        repairSec: null,
        qualityRetryMs: input.qualityRetryMs,
        qualityRetrySec:
          typeof input.qualityRetryMs === 'number' ? Number((input.qualityRetryMs / 1000).toFixed(3)) : null,
      },
      diagnostics: {
        qualityIssueCodes: warnings.filter((warning) => warning.startsWith('quality_issue_')),
        runtimeSource: 'cloud',
      },
      warnings,
      warningIssues: this.buildTranslationWarningIssues(warnings),
      errors,
      errorIssues: this.buildTranslationErrorIssues(errors),
      artifacts: {
        hasTimecodes: false,
      },
    };
  }

  private static async maybeRetryCloudTranslationForQuality(
    input: {
      text: string;
      targetLang: string;
      sourceLang?: string;
      glossary?: string;
      prompt?: string;
      promptTemplateId?: string;
      enableJsonLineRepair: boolean;
      isConnectionTest?: boolean;
      signal?: AbortSignal;
    },
    resolvedProvider: ResolvedCloudTranslateProvider,
    model: { key?: string; model?: string; options?: ApiModelRequestOptions },
    orchestrated: CloudTranslationOrchestratorResult,
    onProgress?: TranslateProgressFn
  ): Promise<{
    orchestrated: CloudTranslationOrchestratorResult;
    qualityRetryCount: number;
    strictRetrySucceeded: boolean;
    qualityRetryMs: number | null;
  }> {
    const qualityMode = this.resolveTranslationQualityModeForRequest({
      promptTemplateId: input.promptTemplateId,
      prompt: input.prompt,
      enableJsonLineRepair: input.enableJsonLineRepair,
    });
    if (!usesTemplateValidatedQualityChecks(qualityMode) || input.isConnectionTest) {
      return {
        orchestrated,
        qualityRetryCount: 0,
        strictRetrySucceeded: false,
        qualityRetryMs: null,
      };
    }

    const addWarning = (warnings: string[], code: string) => {
      if (!warnings.includes(code)) warnings.push(code);
    };
    const qualityContext = this.buildTranslationQualityContext(input.text);
    const initialWarnings = [...(orchestrated.warnings || [])];
    const initialIssues = this.getTranslationQualityIssues(
      input.text,
      orchestrated.output,
      input.targetLang,
      qualityContext
    );
    this.addQualityIssueWarnings(initialIssues, (code) => addWarning(initialWarnings, code));
    if (initialIssues.length === 0) {
      return {
        orchestrated: {
          ...orchestrated,
          warnings: initialWarnings,
        },
        qualityRetryCount: 0,
        strictRetrySucceeded: false,
        qualityRetryMs: null,
      };
    }

    addWarning(initialWarnings, 'quality_retry_triggered');
    onProgress?.('Detected untranslated, repetitive, or target-language-mismatched output, retrying with stricter prompt...');
    const retryStart = Date.now();

    try {
      onProgress?.(`Retrying translation provider (${resolvedProvider.provider}) with stricter prompt...`);
      const strictPrompt = [input.prompt, this.buildStrictRetryInstruction(input.targetLang, initialIssues, qualityContext)]
        .filter(Boolean)
        .join('\n\n');
      const strictOrchestrated = await runCloudTranslationOrchestrator(
        this.buildCloudTranslationOrchestratorInput(
          {
            ...input,
            prompt: strictPrompt,
          },
          resolvedProvider,
          model
        ),
        this.buildCloudTranslationOrchestratorDeps(),
        onProgress
      );
      const strictOutput = this.normalizeTargetLanguageOutput(strictOrchestrated.output, input.targetLang);
      const strictWarnings = [...(strictOrchestrated.warnings || [])];
      const strictIssues = this.getTranslationQualityIssues(
        input.text,
        strictOutput,
        input.targetLang,
        qualityContext
      );
      this.addQualityIssueWarnings(strictIssues, (code) => addWarning(strictWarnings, code));
      if (strictIssues.length > 0) {
        return {
          orchestrated: {
            ...orchestrated,
            warnings: this.mergeUniqueWarnings(initialWarnings, strictWarnings),
          },
          qualityRetryCount: 1,
          strictRetrySucceeded: false,
          qualityRetryMs: Date.now() - retryStart,
        };
      }

      addWarning(strictWarnings, 'strict_retry_applied');
      return {
        orchestrated: {
          ...strictOrchestrated,
          output: strictOutput,
          warnings: this.mergeUniqueWarnings(initialWarnings, strictWarnings),
          retryCount: orchestrated.retryCount + strictOrchestrated.retryCount,
          fallbackUsed: orchestrated.fallbackUsed || strictOrchestrated.fallbackUsed,
          fallbackType: strictOrchestrated.fallbackType || orchestrated.fallbackType,
          cloudContextChunkCount: orchestrated.cloudContextChunkCount + strictOrchestrated.cloudContextChunkCount,
          cloudContextFallbackCount: orchestrated.cloudContextFallbackCount + strictOrchestrated.cloudContextFallbackCount,
          cloudBatching: strictOrchestrated.cloudBatching || orchestrated.cloudBatching,
          relaxedWholeRequestApplied:
            orchestrated.relaxedWholeRequestApplied || strictOrchestrated.relaxedWholeRequestApplied,
          relaxedWholeRequestFallback:
            orchestrated.relaxedWholeRequestFallback || strictOrchestrated.relaxedWholeRequestFallback,
        },
        qualityRetryCount: 1,
        strictRetrySucceeded: true,
        qualityRetryMs: Date.now() - retryStart,
      };
    } catch {
      return {
        orchestrated: {
          ...orchestrated,
          warnings: initialWarnings,
        },
        qualityRetryCount: 1,
        strictRetrySucceeded: false,
        qualityRetryMs: Date.now() - retryStart,
      };
    }
  }

  private static async translateWithCloudModel(
    input: {
      text: string;
      targetLang: string;
      sourceLang?: string;
      glossary?: string;
      prompt?: string;
      promptTemplateId?: string;
      modelId?: string;
      enableJsonLineRepair: boolean;
      isConnectionTest?: boolean;
      signal?: AbortSignal;
    },
    resolvedProvider: ResolvedCloudTranslateProvider,
    model: { key?: string; model?: string; options?: ApiModelRequestOptions },
    onProgress?: TranslateProgressFn
  ): Promise<TranslationResult> {
    const startedAt = Date.now();
    const initialOrchestrated = await runCloudTranslationOrchestrator(
      this.buildCloudTranslationOrchestratorInput(input, resolvedProvider, model),
      this.buildCloudTranslationOrchestratorDeps(),
      onProgress
    );
    const qualityRetried = await this.maybeRetryCloudTranslationForQuality(
      input,
      resolvedProvider,
      model,
      initialOrchestrated,
      onProgress
    );
    const orchestrated = qualityRetried.orchestrated;

    const output = orchestrated.output;
    onProgress?.('Translation completed.');

    return {
      translatedText: output,
      debug: this.buildCloudTranslationDebugInfo({
        configuredModelId: input.modelId,
        sourceText: input.text,
        sourceLang: input.sourceLang,
        targetLang: input.targetLang,
        glossary: input.glossary,
        prompt: input.prompt,
        promptTemplateId: input.promptTemplateId,
        enableJsonLineRepair: input.enableJsonLineRepair,
        isConnectionTest: input.isConnectionTest,
        resolvedProvider,
        orchestrated,
        qualityRetryCount: qualityRetried.qualityRetryCount,
        strictRetrySucceeded: qualityRetried.strictRetrySucceeded,
        output,
        elapsedMs: Date.now() - startedAt,
        qualityRetryMs: qualityRetried.qualityRetryMs,
      }),
    };
  }

  static async testConnection(input: {
    url: string;
    key?: string;
    model?: string;
    name?: string;
    options?: ApiModelRequestOptions;
  }) {
    return runCloudTranslationConnectionProbe(input, {
      buildOrchestratorInput: this.buildCloudTranslationOrchestratorInput.bind(this),
      buildOrchestratorDeps: this.buildCloudTranslationOrchestratorDeps.bind(this),
      mapConnectionError: this.mapConnectionError.bind(this),
    });
  }

  static async translateTextDetailed(
    input: {
      text: string;
      targetLang: string;
      sourceLang?: string;
      glossary?: string;
      prompt?: string;
      promptTemplateId?: string;
      modelId?: string;
      enableJsonLineRepair?: boolean;
      signal?: AbortSignal;
    },
    onProgress?: TranslateProgressFn
  ): Promise<TranslationResult> {
    this.throwIfAborted(input.signal);
    const { text, targetLang, glossary, prompt, promptTemplateId, modelId } = input;
    const enableJsonLineRepair = input.enableJsonLineRepair !== false;
    if (!text || !text.trim()) throw new Error('Source text is required');

    onProgress?.('Loading translation model configuration...');

    const settings = await SettingsManager.getSettings({ mask: false });
    const localModel = await LocalModelService.resolveLocalModelForRequest('translate', modelId, settings);
    if (localModel) {
      return this.translateWithLocalModel(
        {
          text,
          targetLang,
          sourceLang: input.sourceLang,
          glossary,
          prompt,
          promptTemplateId,
          enableJsonLineRepair,
          signal: input.signal,
        },
        localModel,
        onProgress
      );
    }

    const models = settings.translateModels || [];
    const model = models.find((m: any) => m.id === modelId) || models[0];
    if (!model) throw new Error('No translation model configured. Please configure one in Settings.');
    if (!model.url) throw new Error('Translation model URL is missing.');

    const resolvedProvider = resolveCloudTranslateProvider({
      url: model.url,
      modelName: model.name,
      model: model.model,
      apiKey: model.key,
    });
    return this.translateWithCloudModel(
      {
        text,
        targetLang,
        sourceLang: input.sourceLang,
        glossary,
        prompt,
        promptTemplateId,
        enableJsonLineRepair,
        signal: input.signal,
      },
      resolvedProvider,
      {
        key: model.key,
        model: model.model,
        options: model.options,
      },
      onProgress
    );
  }

  static async translateText(input: {
    text: string;
    targetLang: string;
    sourceLang?: string;
    glossary?: string;
    prompt?: string;
    promptTemplateId?: string;
    modelId?: string;
    enableJsonLineRepair?: boolean;
  }) {
    const result = await this.translateTextDetailed(input);
    return result.translatedText;
  }
}
