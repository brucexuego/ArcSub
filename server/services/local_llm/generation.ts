import type { LocalModelDefinition } from '../../local_model_catalog.js';
import type { TranslationQualityMode } from '../llm/orchestrators/translation_quality_policy.js';
import { getLocalOpenvinoOfficialBaselineForRepoId } from './openvino_official_baselines.js';
import { resolveLocalTranslationProfile } from './model_profiles.js';
import type { LocalTranslateGenerationOptions, LocalTranslateModelStrategy } from './types.js';

interface EnvReaders {
  getBoolean: (name: string, fallback: boolean) => boolean;
  getNumber: (name: string, fallback: number, min?: number, max?: number) => number;
}

interface BuildLocalGenerationOptionsInput extends EnvReaders {
  localModel?: LocalModelDefinition | null;
  modelStrategy: LocalTranslateModelStrategy;
  jsonMode?: boolean;
  targetLang?: string;
  strictMode?: boolean;
  translationQualityMode?: TranslationQualityMode;
}

interface EstimateLocalMaxNewTokensInput {
  text: string;
  lineCount: number;
  lineSafeMode: boolean;
  jsonMode?: boolean;
  strictMode?: boolean;
  getNumber: (name: string, fallback: number, min?: number, max?: number) => number;
}

function getPromptLookupOptions(input: EnvReaders) {
  const promptLookupEnabled = input.getBoolean('OPENVINO_LOCAL_TRANSLATE_PROMPT_LOOKUP', false);
  if (!promptLookupEnabled) return null;

  return {
    numAssistantTokens: Math.round(input.getNumber('OPENVINO_LOCAL_TRANSLATE_PROMPT_LOOKUP_ASSISTANT_TOKENS', 3, 1, 32)),
    maxNgramSize: Math.round(input.getNumber('OPENVINO_LOCAL_TRANSLATE_PROMPT_LOOKUP_MAX_NGRAM_SIZE', 3, 1, 16)),
    assistantConfidenceThreshold: Number(
      input.getNumber('OPENVINO_LOCAL_TRANSLATE_PROMPT_LOOKUP_CONFIDENCE', 0, 0, 1).toFixed(3)
    ),
  } satisfies LocalTranslateGenerationOptions;
}

function getOfficialGenerationDefaults(localModel?: LocalModelDefinition | null) {
  if (!localModel?.repoId) return null;
  return getLocalOpenvinoOfficialBaselineForRepoId(localModel.repoId)?.officialGeneration || null;
}

export function buildLocalGenerationOptions(input: BuildLocalGenerationOptionsInput): LocalTranslateGenerationOptions {
  const promptLookupOptions = getPromptLookupOptions(input);
  const officialDefaults = getOfficialGenerationDefaults(input.localModel);
  const qualityMode = input.translationQualityMode || (input.jsonMode ? 'json_strict' : 'plain_probe');
  const resolvedProfile = resolveLocalTranslationProfile(input.localModel, input.modelStrategy, qualityMode);
  const familyDefaults = resolvedProfile.qualityModeDefaults;

  if (input.modelStrategy.family === 'translategemma') {
    return {
      doSample: familyDefaults.doSample ?? false,
      temperature: familyDefaults.temperature ?? 0,
      topP: familyDefaults.topP ?? 1,
      topK: familyDefaults.topK ?? 1,
      minP: familyDefaults.minP,
      repetitionPenalty: familyDefaults.repetitionPenalty,
      frequencyPenalty: familyDefaults.frequencyPenalty,
      noRepeatNgramSize: familyDefaults.noRepeatNgramSize,
      applyChatTemplate: true,
    };
  }

  if (input.modelStrategy.generationStyle === 'qwen3') {
    const strictLikeMode = input.jsonMode || input.strictMode || qualityMode !== 'plain_probe';
    return {
      doSample: familyDefaults.doSample ?? officialDefaults?.doSample ?? true,
      temperature: Number(
        input.getNumber(
          strictLikeMode
            ? 'OPENVINO_LOCAL_TRANSLATE_QWEN3_STRICT_TEMPERATURE'
            : 'OPENVINO_LOCAL_TRANSLATE_QWEN3_TEMPERATURE',
          strictLikeMode
            ? familyDefaults.temperature ?? 0
            : familyDefaults.temperature ?? officialDefaults?.temperature ?? 0.6,
          0,
          2
        ).toFixed(3)
      ),
      topP: Number(
        input
          .getNumber('OPENVINO_LOCAL_TRANSLATE_QWEN3_TOP_P', familyDefaults.topP ?? officialDefaults?.topP ?? 0.95, 0, 1)
          .toFixed(3)
      ),
      topK: Math.round(
        input.getNumber('OPENVINO_LOCAL_TRANSLATE_QWEN3_TOP_K', familyDefaults.topK ?? officialDefaults?.topK ?? 20, 1, 200)
      ),
      minP: Number(input.getNumber('OPENVINO_LOCAL_TRANSLATE_QWEN3_MIN_P', familyDefaults.minP ?? 0, 0, 1).toFixed(3)),
      presencePenalty: Number(
        input.getNumber('OPENVINO_LOCAL_TRANSLATE_QWEN3_PRESENCE_PENALTY', familyDefaults.presencePenalty ?? 0, 0, 2).toFixed(3)
      ),
      repetitionPenalty: familyDefaults.repetitionPenalty,
      frequencyPenalty: familyDefaults.frequencyPenalty,
      noRepeatNgramSize: familyDefaults.noRepeatNgramSize,
      applyChatTemplate: false,
      ...(promptLookupOptions || {}),
    };
  }

  if (input.modelStrategy.generationStyle === 'deepseek_r1') {
    return {
      doSample: familyDefaults.doSample ?? true,
      temperature: Number(
        input.getNumber(
          input.strictMode ? 'OPENVINO_LOCAL_TRANSLATE_DEEPSEEK_R1_STRICT_TEMPERATURE' : 'OPENVINO_LOCAL_TRANSLATE_DEEPSEEK_R1_TEMPERATURE',
          input.strictMode
            ? familyDefaults.temperature ?? 0
            : familyDefaults.temperature ?? 0.6,
          0,
          2
        ).toFixed(3)
      ),
      topP: Number(input.getNumber('OPENVINO_LOCAL_TRANSLATE_DEEPSEEK_R1_TOP_P', familyDefaults.topP ?? 0.95, 0, 1).toFixed(3)),
      topK: Math.round(input.getNumber('OPENVINO_LOCAL_TRANSLATE_DEEPSEEK_R1_TOP_K', familyDefaults.topK ?? 50, 1, 200)),
      minP: Number(input.getNumber('OPENVINO_LOCAL_TRANSLATE_DEEPSEEK_R1_MIN_P', familyDefaults.minP ?? 0, 0, 1).toFixed(3)),
      repetitionPenalty: familyDefaults.repetitionPenalty,
      noRepeatNgramSize: familyDefaults.noRepeatNgramSize,
      applyChatTemplate: false,
      ...(promptLookupOptions || {}),
    };
  }

  if (input.modelStrategy.generationStyle !== 'qwen') {
    return {
      doSample: familyDefaults.doSample ?? false,
      temperature: familyDefaults.temperature ?? 0.2,
      topP: familyDefaults.topP ?? 0.95,
      topK: familyDefaults.topK,
      minP: familyDefaults.minP,
      repetitionPenalty: familyDefaults.repetitionPenalty,
      frequencyPenalty: familyDefaults.frequencyPenalty,
      noRepeatNgramSize: familyDefaults.noRepeatNgramSize,
      applyChatTemplate: false,
      ...(promptLookupOptions || {}),
    };
  }

  if (input.jsonMode) {
    return {
      doSample: false,
      temperature: 0,
      topP: 1,
      topK: 1,
      minP: 0,
      repetitionPenalty: familyDefaults.repetitionPenalty,
      frequencyPenalty: familyDefaults.frequencyPenalty,
      noRepeatNgramSize: familyDefaults.noRepeatNgramSize,
      applyChatTemplate: false,
      ...(promptLookupOptions || {}),
    };
  }

  const normalizedTarget = String(input.targetLang || '').trim().toLowerCase();
  const enableSampling = input.getBoolean('OPENVINO_LOCAL_TRANSLATE_ENABLE_SAMPLING', false);
  const qwenBaselineRepetitionPenalty = familyDefaults.repetitionPenalty ?? officialDefaults?.repetitionPenalty ?? 1.05;

  if (
    input.strictMode ||
    qualityMode !== 'plain_probe' ||
    !enableSampling ||
    normalizedTarget === 'zh-tw' ||
    normalizedTarget === 'zh-cn' ||
    normalizedTarget === 'en' ||
    normalizedTarget === 'english'
  ) {
    return {
      doSample: familyDefaults.doSample ?? false,
      temperature: familyDefaults.temperature ?? 0,
      topP: familyDefaults.topP ?? 1,
      topK: familyDefaults.topK ?? 1,
      minP: familyDefaults.minP ?? 0,
      repetitionPenalty: qwenBaselineRepetitionPenalty,
      frequencyPenalty: familyDefaults.frequencyPenalty,
      noRepeatNgramSize: familyDefaults.noRepeatNgramSize,
      applyChatTemplate: false,
      ...(promptLookupOptions || {}),
    };
  }

  return {
    doSample: familyDefaults.doSample ?? officialDefaults?.doSample ?? true,
    temperature: familyDefaults.temperature ?? officialDefaults?.temperature ?? 0.7,
    topP: familyDefaults.topP ?? officialDefaults?.topP ?? 0.8,
    topK: familyDefaults.topK ?? officialDefaults?.topK ?? 20,
    minP: familyDefaults.minP ?? 0,
    repetitionPenalty: qwenBaselineRepetitionPenalty,
    frequencyPenalty: familyDefaults.frequencyPenalty,
    noRepeatNgramSize: familyDefaults.noRepeatNgramSize,
    applyChatTemplate: false,
    ...(promptLookupOptions || {}),
  };
}

export function estimateTokenLikeCount(text: string) {
  const normalized = String(text || '').trim();
  if (!normalized) return 0;

  const compact = normalized.replace(/\s+/g, ' ');
  const cjkChars = (
    compact.match(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu) || []
  ).length;
  const remainingChars = Math.max(0, compact.length - cjkChars);
  return cjkChars + Math.ceil(remainingChars / 4);
}

export function estimateLocalMaxNewTokens(input: EstimateLocalMaxNewTokensInput) {
  const estimatedUnits = estimateTokenLikeCount(input.text);
  const baseBudget = input.jsonMode
    ? Math.ceil(estimatedUnits * 1.35) + input.lineCount * 10
    : Math.ceil(estimatedUnits * 1.15) + input.lineCount * (input.lineSafeMode ? 6 : 2) + (input.strictMode ? 24 : 0);
  const minTokens = input.jsonMode ? 128 : input.lineSafeMode ? 96 : 64;
  const maxTokens = Math.round(
    input.getNumber(
      input.jsonMode ? 'OPENVINO_LOCAL_TRANSLATE_JSON_MAX_NEW_TOKENS' : 'OPENVINO_LOCAL_TRANSLATE_MAX_NEW_TOKENS',
      input.jsonMode ? 3072 : 2048,
      64,
      8192
    )
  );
  return Math.max(minTokens, Math.min(maxTokens, baseBudget));
}
