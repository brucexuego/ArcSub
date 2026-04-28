import fs from 'fs-extra';
import https from 'https';
import path from 'path';
import { createHash } from 'crypto';
import { SettingsManager, AppSettings } from './settings_manager.js';
import {
  buildLocalModelId,
  BUILTIN_LOCAL_MODELS,
  HuggingFaceModelMetadata,
  inferLocalModelDefinitionFromHf,
  inferLocalModelDefinitionFromInstalledDir,
  LocalModelDefinition,
  LocalModelSelection,
  LocalModelType,
  LocalModelRuntimeHints,
  sanitizeLocalModelDefinition,
  getLocalModelInstallDir,
  getDefaultLocalModelSelection,
} from '../local_model_catalog.js';
import { OpenvinoRuntimeManager } from '../openvino_runtime_manager.js';

type LocalModelInstallPhase =
  | 'queued'
  | 'downloading'
  | 'converting'
  | 'verifying'
  | 'persisting'
  | 'completed'
  | 'failed';

interface InstallState {
  modelId: string;
  type: LocalModelType;
  repoId: string;
  name: string;
  installing: boolean;
  phase: LocalModelInstallPhase;
  startedAt: number;
  updatedAt: number;
  completedAt?: number;
  message?: string | null;
  error?: string | null;
}

interface LocalModelOverviewEntry {
  id: string;
  type: LocalModelType;
  name: string;
  repoId: string;
  sourceFormat: LocalModelDefinition['sourceFormat'];
  conversionMethod: LocalModelDefinition['conversionMethod'];
  runtimeLayout: LocalModelDefinition['runtimeLayout'];
  runtime: LocalModelDefinition['runtime'];
  runtimeHints?: LocalModelRuntimeHints;
  selected: boolean;
  installError?: string | null;
}

interface LocalModelMetadataInspection {
  repoId: string;
  canonicalRepoId: string;
  type: LocalModelType;
  inferredModel: LocalModelDefinition;
  runtimeHints?: LocalModelRuntimeHints;
  metadataPath?: string | null;
}

export class LocalModelService {
  private static readonly HF_BASE = 'https://huggingface.co';
  private static readonly OVERVIEW_CACHE_MS = 1500;
  private static readonly INSTALL_STATUS_TTL_MS = 10 * 60 * 1000;
  private static installTasks = new Map<string, Promise<void>>();
  private static installStates = new Map<string, InstallState>();
  private static installErrors = new Map<string, string>();
  private static overviewCache: { value: Awaited<ReturnType<typeof LocalModelService['getCatalogView']>>; at: number } | null = null;

  private static invalidateOverviewCache() {
    this.overviewCache = null;
  }

  private static getInstallStateSnapshot(modelId: string) {
    const state = this.installStates.get(modelId);
    return state ? { ...state } : null;
  }

  private static updateInstallState(modelId: string, patch: Partial<InstallState>) {
    const previous = this.installStates.get(modelId);
    if (!previous) return null;
    const next: InstallState = {
      ...previous,
      ...patch,
      updatedAt: Date.now(),
    };
    this.installStates.set(modelId, next);
    this.invalidateOverviewCache();
    return { ...next };
  }

  private static pruneInstallStates() {
    const now = Date.now();
    let changed = false;
    for (const [modelId, state] of this.installStates.entries()) {
      if (state.installing) continue;
      const referenceTime = state.updatedAt || state.completedAt || state.startedAt;
      if (now - referenceTime <= this.INSTALL_STATUS_TTL_MS) continue;
      this.installStates.delete(modelId);
      changed = true;
    }
    if (changed) {
      this.invalidateOverviewCache();
    }
  }

  private static getInstallStatusList() {
    this.pruneInstallStates();
    return Array.from(this.installStates.values())
      .map((state) => ({ ...state }))
      .sort((a, b) => b.updatedAt - a.updatedAt);
  }

  private static getArtifactInstallPhase(model: LocalModelDefinition): LocalModelInstallPhase {
    return model.installMode === 'hf-direct' ? 'downloading' : 'converting';
  }

  private static async buildOverview() {
    const settings = await SettingsManager.getSettings({ mask: false });
    return this.getCatalogView(settings);
  }

  private static getHfToken() {
    const token = String(process.env.HF_TOKEN || '').trim();
    return token || null;
  }

  private static shouldAttachHfAuth(rawUrl: string) {
    try {
      const hostname = new URL(rawUrl).hostname.toLowerCase();
      return hostname === 'huggingface.co' || hostname.endsWith('.huggingface.co');
    } catch {
      return false;
    }
  }

  private static buildDownloadHeaders(rawUrl: string) {
    const headers: Record<string, string> = {
      'User-Agent': 'ArcSub-LocalModelInstaller/2.0',
      Accept: 'application/json, text/plain, */*',
    };
    const token = this.getHfToken();
    if (token && this.shouldAttachHfAuth(rawUrl)) {
      headers.Authorization = `Bearer ${token}`;
    }
    return headers;
  }

  private static normalizeRepoIdInput(rawValue: string) {
    const raw = String(rawValue || '').trim();
    if (!raw) {
      throw new Error('Hugging Face model id is required.');
    }

    if (/^https?:\/\//i.test(raw)) {
      let url: URL;
      try {
        url = new URL(raw);
      } catch {
        throw new Error('Invalid Hugging Face model URL.');
      }
      const hostname = url.hostname.toLowerCase();
      if (hostname !== 'huggingface.co' && !hostname.endsWith('.huggingface.co')) {
        throw new Error('Only Hugging Face model URLs are supported.');
      }

      const parts = url.pathname
        .split('/')
        .map((part) => String(part || '').trim())
        .filter(Boolean);

      const filtered = parts[0] === 'models' ? parts.slice(1) : parts;
      if (filtered.length < 2) {
        throw new Error('Invalid Hugging Face model URL.');
      }
      return `${filtered[0]}/${filtered[1]}`;
    }

    const normalized = raw.replace(/^\/+|\/+$/g, '');
    const parts = normalized.split('/').filter(Boolean);
    if (parts.length !== 2) {
      throw new Error('Hugging Face model id must be in the form owner/model-name.');
    }
    return `${parts[0]}/${parts[1]}`;
  }

  private static getHfApiUrl(repoId: string) {
    const [owner, name] = repoId.split('/');
    return `${this.HF_BASE}/api/models/${encodeURIComponent(owner)}/${encodeURIComponent(name)}`;
  }

  private static getHfFileUrl(repoId: string, fileName: string) {
    const encodedPath = fileName
      .split('/')
      .map((segment) => encodeURIComponent(segment))
      .join('/');
    return `${this.HF_BASE}/${repoId}/resolve/main/${encodedPath}`;
  }

  private static async fetchJsonWithRedirect(url: string): Promise<{ status: number; body: any }> {
    return new Promise((resolve, reject) => {
      const request = https.get(url, { headers: this.buildDownloadHeaders(url) }, (response) => {
        const status = response.statusCode || 0;
        if ([301, 302, 303, 307, 308].includes(status)) {
          const redirectUrl = response.headers.location;
          response.resume();
          if (!redirectUrl) {
            reject(new Error(`Request redirect without location: ${url}`));
            return;
          }
          const nextUrl = new URL(redirectUrl, url).toString();
          this.fetchJsonWithRedirect(nextUrl).then(resolve).catch(reject);
          return;
        }

        let raw = '';
        response.setEncoding('utf8');
        response.on('data', (chunk) => {
          raw += chunk;
        });
        response.on('end', () => {
          if (status < 200 || status >= 300) {
            const errorCode = String(response.headers['x-error-code'] || '').trim();
            const errorMessage = String(response.headers['x-error-message'] || '').trim();
            if (
              (status === 401 || status === 403) &&
              (errorCode.toLowerCase() === 'gatedrepo' ||
                /gated|restricted|authenticated|access to model/i.test(errorMessage))
            ) {
              reject(
                new Error(
                  `Unable to access Hugging Face model metadata (${status}). ` +
                    `This repository requires approval/token access. Please accept the model license and set HF_TOKEN in .env.`
                )
              );
              return;
            }
            if (status === 404) {
              reject(new Error(`Hugging Face model not found: ${url}`));
              return;
            }
            reject(new Error(`Failed to fetch Hugging Face model metadata (${status}).`));
            return;
          }

          try {
            resolve({
              status,
              body: raw ? JSON.parse(raw) : {},
            });
          } catch (error: any) {
            reject(new Error(`Failed to parse Hugging Face model metadata: ${String(error?.message || error)}`));
          }
        });
      });

      request.on('error', (error) => {
        reject(error);
      });
      request.setTimeout(30000, () => {
        request.destroy(new Error(`Metadata request timeout for ${url}`));
      });
    });
  }

  private static async fetchHfModelMetadata(repoId: string): Promise<HuggingFaceModelMetadata> {
    const { body } = await this.fetchJsonWithRedirect(this.getHfApiUrl(repoId));
    return body || {};
  }

  private static async fetchTextWithRedirect(url: string): Promise<{ status: number; body: string }> {
    return new Promise((resolve, reject) => {
      const request = https.get(url, { headers: this.buildDownloadHeaders(url) }, (response) => {
        const status = response.statusCode || 0;
        if ([301, 302, 303, 307, 308].includes(status)) {
          const redirectUrl = response.headers.location;
          response.resume();
          if (!redirectUrl) {
            reject(new Error(`Request redirect without location: ${url}`));
            return;
          }
          const nextUrl = new URL(redirectUrl, url).toString();
          this.fetchTextWithRedirect(nextUrl).then(resolve).catch(reject);
          return;
        }

        let raw = '';
        response.setEncoding('utf8');
        response.on('data', (chunk) => {
          raw += chunk;
        });
        response.on('end', () => {
          if (status < 200 || status >= 300) {
            if (status === 404) {
              reject(new Error(`Hugging Face file not found: ${url}`));
              return;
            }
            reject(new Error(`Failed to fetch Hugging Face file (${status}): ${url}`));
            return;
          }
          resolve({ status, body: raw });
        });
      });

      request.on('error', (error) => {
        reject(error);
      });
      request.setTimeout(30000, () => {
        request.destroy(new Error(`Request timeout for ${url}`));
      });
    });
  }

  private static async fetchOptionalHfJson(repoId: string, fileName: string) {
    try {
      const { body } = await this.fetchJsonWithRedirect(this.getHfFileUrl(repoId, fileName));
      return body && typeof body === 'object' ? body : null;
    } catch (error: any) {
      if (/not found|404/i.test(String(error?.message || error || ''))) return null;
      return null;
    }
  }

  private static async fetchOptionalHfText(repoId: string, fileName: string) {
    try {
      const { body } = await this.fetchTextWithRedirect(this.getHfFileUrl(repoId, fileName));
      return String(body || '');
    } catch (error: any) {
      if (/not found|404/i.test(String(error?.message || error || ''))) return '';
      return '';
    }
  }

  private static tryReadLocalJson(modelDir: string, fileName: string) {
    try {
      const filePath = path.join(modelDir, fileName);
      if (!fs.existsSync(filePath)) return null;
      return JSON.parse(fs.readFileSync(filePath, 'utf8'));
    } catch {
      return null;
    }
  }

  private static tryReadLocalText(modelDir: string, fileName: string) {
    try {
      const filePath = path.join(modelDir, fileName);
      if (!fs.existsSync(filePath)) return '';
      return fs.readFileSync(filePath, 'utf8');
    } catch {
      return '';
    }
  }

  private static parseModelCardFrontMatter(readme: string) {
    const text = String(readme || '');
    const match = text.match(/^---\s*\r?\n([\s\S]*?)\r?\n---\s*(?:\r?\n|$)/);
    if (!match) return {};
    const lines = match[1].split(/\r?\n/);
    const data: Record<string, any> = {};
    let currentKey = '';
    for (const rawLine of lines) {
      const line = rawLine.replace(/\t/g, '  ');
      const keyMatch = line.match(/^([A-Za-z0-9_.-]+)\s*:\s*(.*)$/);
      if (keyMatch) {
        currentKey = keyMatch[1];
        const rawValue = keyMatch[2].trim();
        if (!rawValue) {
          data[currentKey] = [];
        } else {
          data[currentKey] = rawValue.replace(/^['"]|['"]$/g, '');
        }
        continue;
      }
      const itemMatch = line.match(/^\s*-\s*(.+)$/);
      if (itemMatch && currentKey) {
        if (!Array.isArray(data[currentKey])) data[currentKey] = [];
        data[currentKey].push(itemMatch[1].trim().replace(/^['"]|['"]$/g, ''));
      }
    }
    return data;
  }

  private static summarizeReadme(readme: string) {
    const withoutFrontMatter = String(readme || '').replace(/^---\s*\r?\n[\s\S]*?\r?\n---\s*/, '');
    const cleaned = withoutFrontMatter
      .replace(/```[\s\S]*?```/g, ' ')
      .replace(/!\[[^\]]*]\([^)]+\)/g, ' ')
      .replace(/\[[^\]]+]\([^)]+\)/g, (match) => match.replace(/^\[|\]\([^)]+\)$/g, ''))
      .replace(/[#>*_`|<>{}\[\]]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
    return cleaned ? cleaned.slice(0, 900) : null;
  }

  private static numberFromConfig(
    source: Record<string, any> | null | undefined,
    keys: string[],
    min: number,
    max: number,
    integer = true
  ) {
    if (!source || typeof source !== 'object') return undefined;
    for (const key of keys) {
      const value = Number(source[key]);
      if (Number.isFinite(value) && value >= min && value <= max) {
        return integer ? Math.round(value) : value;
      }
    }
    return undefined;
  }

  private static boolFromConfig(source: Record<string, any> | null | undefined, keys: string[]) {
    if (!source || typeof source !== 'object') return undefined;
    for (const key of keys) {
      const value = source[key];
      if (typeof value === 'boolean') return value;
    }
    return undefined;
  }

  private static deriveRuntimeHints(input: {
    type: LocalModelType;
    repoId: string;
    metadata: HuggingFaceModelMetadata & Record<string, any>;
    readme?: string;
    config?: Record<string, any> | null;
    generationConfig?: Record<string, any> | null;
    preprocessorConfig?: Record<string, any> | null;
    tokenizerConfig?: Record<string, any> | null;
    chatTemplate?: string;
  }): LocalModelRuntimeHints | undefined {
    const cardData =
      input.metadata?.cardData && typeof input.metadata.cardData === 'object'
        ? input.metadata.cardData
        : this.parseModelCardFrontMatter(input.readme || '');
    const config = input.config || (input.metadata?.config && typeof input.metadata.config === 'object' ? input.metadata.config : null);
    const generationConfig = input.generationConfig || null;
    const preprocessorConfig = input.preprocessorConfig || null;
    const tokenizerConfig = input.tokenizerConfig || null;
    const evidence: NonNullable<LocalModelRuntimeHints['evidence']> = [];

    const addEvidence = (
      source: string,
      key: string,
      value: string | number | boolean | null,
      confidence: 'high' | 'medium' | 'low' = 'high'
    ) => {
      evidence.push({ source, key, value, confidence });
    };

    const modelCard: NonNullable<LocalModelRuntimeHints['modelCard']> = {
      license:
        typeof (cardData as any)?.license === 'string'
          ? (cardData as any).license
          : typeof input.metadata?.license === 'string'
            ? input.metadata.license
            : null,
      baseModel:
        (cardData as any)?.base_model ||
        (cardData as any)?.baseModel ||
        (Array.isArray(input.metadata?.tags)
          ? input.metadata.tags
              .map((tag: unknown) => String(tag || ''))
              .find((tag: string) => tag.startsWith('base_model:'))
              ?.slice('base_model:'.length) || null
          : null),
      pipelineTag:
        typeof input.metadata?.pipeline_tag === 'string'
          ? input.metadata.pipeline_tag
          : typeof (cardData as any)?.pipeline_tag === 'string'
            ? (cardData as any).pipeline_tag
            : null,
      libraryName:
        typeof input.metadata?.library_name === 'string'
          ? input.metadata.library_name
          : typeof (cardData as any)?.library_name === 'string'
            ? (cardData as any).library_name
            : null,
      summary: this.summarizeReadme(input.readme || ''),
    };

    if (input.type === 'asr') {
      const taskRaw = String(generationConfig?.task || (config as any)?.task || '').trim().toLowerCase();
      const task = taskRaw === 'translate' ? 'translate' : taskRaw === 'transcribe' ? 'transcribe' : undefined;
      const returnTimestamps = this.boolFromConfig(generationConfig, ['return_timestamps', 'returnTimestamps']);
      const wordTimestamps = this.boolFromConfig(generationConfig, [
        'return_token_timestamps',
        'return_word_timestamps',
        'word_timestamps',
      ]);
      const chunkLengthSec = this.numberFromConfig(
        preprocessorConfig,
        ['chunk_length', 'chunk_length_s', 'chunk_length_sec'],
        1,
        3600
      );
      const samplingRate = this.numberFromConfig(preprocessorConfig, ['sampling_rate', 'sample_rate'], 1, 384000, true);
      const maxTargetPositions =
        this.numberFromConfig(config, ['max_target_positions', 'max_source_positions'], 1, 1_000_000, true) ||
        this.numberFromConfig(generationConfig, ['max_length'], 1, 1_000_000, true);

      if (task) addEvidence('generation_config.json', 'task', task);
      if (returnTimestamps != null) addEvidence('generation_config.json', 'return_timestamps', returnTimestamps);
      if (wordTimestamps != null) addEvidence('generation_config.json', 'word_timestamps', wordTimestamps);
      if (chunkLengthSec != null) addEvidence('preprocessor_config.json', 'chunkLengthSec', chunkLengthSec);
      if (samplingRate != null) addEvidence('preprocessor_config.json', 'samplingRate', samplingRate);
      if (maxTargetPositions != null) addEvidence('config.json', 'maxTargetPositions', maxTargetPositions);

      const asr: NonNullable<LocalModelRuntimeHints['asr']> = {
        ...(task ? { task } : {}),
        ...(returnTimestamps != null ? { returnTimestamps } : {}),
        ...(wordTimestamps != null ? { wordTimestamps } : {}),
        ...(chunkLengthSec != null ? { chunkLengthSec } : {}),
        ...(samplingRate != null ? { samplingRate } : {}),
        ...(maxTargetPositions != null ? { maxTargetPositions } : {}),
        confidence: evidence.length > 0 ? 'medium' : 'low',
      };
      const hints: LocalModelRuntimeHints = {
        inspectedAt: new Date().toISOString(),
        hfSha: typeof input.metadata?.sha === 'string' ? input.metadata.sha : undefined,
        modelCard,
        ...(Object.keys(asr).length > 1 ? { asr } : {}),
        ...(evidence.length > 0 ? { evidence } : {}),
      };
      return Object.keys(hints).length > 0 ? hints : undefined;
    }

    if (input.type !== 'translate') return undefined;

    const textConfig = (config as any)?.text_config && typeof (config as any).text_config === 'object'
      ? (config as any).text_config
      : null;
    const rootConfigContext = this.numberFromConfig(
      config,
      ['max_position_embeddings', 'n_positions', 'seq_length', 'max_sequence_length', 'context_length', 'max_seq_len'],
      128,
      10_000_000
    );
    const textConfigContext = this.numberFromConfig(
      textConfig,
      ['max_position_embeddings', 'n_positions', 'seq_length', 'max_sequence_length', 'context_length', 'max_seq_len'],
      128,
      10_000_000
    );
    const configContext = this.numberFromConfig(
      config,
      ['max_position_embeddings', 'n_positions', 'seq_length', 'max_sequence_length', 'context_length', 'max_seq_len'],
      128,
      10_000_000
    ) || textConfigContext;
    const tokenizerContext = this.numberFromConfig(tokenizerConfig, ['model_max_length', 'max_len'], 128, 10_000_000);
    const contextWindow = configContext || tokenizerContext;
    if (rootConfigContext) {
      addEvidence('config.json', 'contextWindow', rootConfigContext);
    } else if (textConfigContext) {
      addEvidence('config.json:text_config', 'contextWindow', textConfigContext);
    }
    if (!configContext && tokenizerContext) addEvidence('tokenizer_config.json', 'model_max_length', tokenizerContext);

    const maxNewTokens = this.numberFromConfig(generationConfig, ['max_new_tokens'], 1, 1_000_000);
    const maxLengthTokens = this.numberFromConfig(generationConfig, ['max_length'], 1, 10_000_000);
    const maxOutputTokens =
      maxNewTokens ||
      (maxLengthTokens && (!contextWindow || maxLengthTokens < Math.floor(contextWindow * 0.9))
        ? maxLengthTokens
        : undefined);
    if (maxNewTokens) addEvidence('generation_config.json', 'max_new_tokens', maxNewTokens);
    if (!maxNewTokens && maxOutputTokens) addEvidence('generation_config.json', 'max_length', maxOutputTokens);
    if (!maxNewTokens && maxLengthTokens && contextWindow && maxLengthTokens >= Math.floor(contextWindow * 0.9)) {
      addEvidence('generation_config.json', 'max_length_context_window', maxLengthTokens, 'medium');
    }

    const generation: NonNullable<LocalModelRuntimeHints['generation']> = {};
    const doSample = this.boolFromConfig(generationConfig, ['do_sample']);
    if (doSample != null) {
      generation.doSample = doSample;
      addEvidence('generation_config.json', 'do_sample', doSample);
    }
    const generationMap: Array<[keyof typeof generation, string, string[], number, number]> = [
      ['temperature', 'temperature', ['temperature'], 0, 2],
      ['topP', 'top_p', ['top_p', 'topP'], 0, 1],
      ['topK', 'top_k', ['top_k', 'topK'], 0, 1000],
      ['minP', 'min_p', ['min_p', 'minP'], 0, 1],
      ['repetitionPenalty', 'repetition_penalty', ['repetition_penalty', 'repetitionPenalty'], 0, 10],
      ['presencePenalty', 'presence_penalty', ['presence_penalty', 'presencePenalty'], -2, 2],
      ['frequencyPenalty', 'frequency_penalty', ['frequency_penalty', 'frequencyPenalty'], -2, 2],
      ['noRepeatNgramSize', 'no_repeat_ngram_size', ['no_repeat_ngram_size', 'noRepeatNgramSize'], 0, 64],
    ];
    for (const [targetKey, evidenceKey, keys, min, max] of generationMap) {
      const integerValue = targetKey === 'topK' || targetKey === 'noRepeatNgramSize';
      const value = this.numberFromConfig(generationConfig, keys, min, max, integerValue);
      if (value != null) {
        (generation as any)[targetKey] = value;
        addEvidence('generation_config.json', evidenceKey, value);
      }
    }

    const templateText =
      String(input.chatTemplate || '') ||
      String(tokenizerConfig?.chat_template || '') ||
      String((cardData as any)?.chat_template || '');
    const templateHash = templateText
      ? createHash('sha1').update(templateText).digest('hex').slice(0, 16)
      : null;
    const repoSignature = `${input.repoId} ${templateText}`.toLowerCase();
    const supportsThinking =
      /enable_thinking|\/no_think|<think>|thinking/i.test(templateText) ||
      /qwen3|deepseek-r1/i.test(input.repoId);
    const defaultEnableThinking = supportsThinking ? false : undefined;
    if (templateText) addEvidence(tokenizerConfig?.chat_template ? 'tokenizer_config.json' : 'chat_template', 'chat_template', templateHash);

    const outputBudget = maxOutputTokens || (contextWindow ? Math.max(256, Math.min(2048, Math.floor(contextWindow * 0.25))) : undefined);
    const safetyReserve = contextWindow ? Math.max(128, Math.min(1024, Math.ceil(contextWindow * 0.08))) : undefined;
    const inputTokenBudget =
      contextWindow && outputBudget && safetyReserve
        ? Math.max(128, contextWindow - outputBudget - safetyReserve)
        : undefined;
    if (inputTokenBudget) addEvidence('derived', 'inputTokenBudget', inputTokenBudget, 'medium');

    const hints: LocalModelRuntimeHints = {
      inspectedAt: new Date().toISOString(),
      hfSha: typeof input.metadata?.sha === 'string' ? input.metadata.sha : undefined,
      modelCard,
      ...(contextWindow ? { contextWindow } : {}),
      ...(inputTokenBudget ? { maxInputTokens: inputTokenBudget } : {}),
      ...(maxOutputTokens ? { maxOutputTokens } : {}),
      ...(Object.keys(generation).length > 0 ? { generation } : {}),
      chatTemplate: {
        available: Boolean(templateText),
        supportsThinking,
        ...(defaultEnableThinking != null ? { defaultEnableThinking } : {}),
        templateSource: tokenizerConfig?.chat_template ? 'tokenizer_config.json' : templateText ? 'chat_template' : null,
        templateHash,
        ...(supportsThinking ? { kwargs: { enable_thinking: false } } : {}),
      },
      ...(inputTokenBudget
        ? {
            batching: {
              mode: 'token_aware',
              inputTokenBudget,
              outputTokenBudget: outputBudget,
              safetyReserveTokens: safetyReserve,
              confidence: contextWindow ? 'medium' : 'low',
            },
          }
        : {}),
      ...(evidence.length > 0 ? { evidence } : {}),
    };

    if (/deepseek-r1/i.test(repoSignature) && hints.chatTemplate?.kwargs) {
      hints.chatTemplate.kwargs = { ...hints.chatTemplate.kwargs, enable_thinking: false };
    }
    return hints;
  }

  private static async inspectModelMetadata(
    type: LocalModelType,
    repoIdOrUrl: string,
    options: { modelDir?: string; inferredModel?: LocalModelDefinition; remoteMetadataOptional?: boolean } = {}
  ): Promise<LocalModelMetadataInspection> {
    const normalizedRepoId = this.normalizeRepoIdInput(repoIdOrUrl);
    let remoteMetadataError: string | null = null;
    let metadata: HuggingFaceModelMetadata & Record<string, any>;
    try {
      metadata = await this.fetchHfModelMetadata(normalizedRepoId) as HuggingFaceModelMetadata & Record<string, any>;
    } catch (error: any) {
      if (!options.remoteMetadataOptional || !options.inferredModel) {
        throw error;
      }
      remoteMetadataError = String(error?.message || error || 'Failed to fetch Hugging Face model metadata.');
      metadata = {
        id: normalizedRepoId,
        tags: [],
      } as HuggingFaceModelMetadata & Record<string, any>;
    }
    const canonicalRepoId = String(metadata?.id || normalizedRepoId).trim() || normalizedRepoId;
    const inferredModel = options.inferredModel || inferLocalModelDefinitionFromHf(type, canonicalRepoId, metadata);
    const shouldFetchRemoteFiles = !remoteMetadataError;
    const [
      readme,
      remoteConfig,
      remoteGenerationConfig,
      remotePreprocessorConfig,
      remoteTokenizerConfig,
      remoteChatTemplate,
    ] = shouldFetchRemoteFiles
      ? await Promise.all([
          this.fetchOptionalHfText(canonicalRepoId, 'README.md'),
          this.fetchOptionalHfJson(canonicalRepoId, 'config.json'),
          this.fetchOptionalHfJson(canonicalRepoId, 'generation_config.json'),
          this.fetchOptionalHfJson(canonicalRepoId, 'preprocessor_config.json'),
          this.fetchOptionalHfJson(canonicalRepoId, 'tokenizer_config.json'),
          this.fetchOptionalHfText(canonicalRepoId, 'chat_template.jinja'),
        ])
      : ['', null, null, null, null, ''];

    const modelDir = options.modelDir || '';
    const localConfig = modelDir ? this.tryReadLocalJson(modelDir, 'config.json') : null;
    const localGenerationConfig = modelDir ? this.tryReadLocalJson(modelDir, 'generation_config.json') : null;
    const localPreprocessorConfig = modelDir ? this.tryReadLocalJson(modelDir, 'preprocessor_config.json') : null;
    const localTokenizerConfig = modelDir ? this.tryReadLocalJson(modelDir, 'tokenizer_config.json') : null;
    const localChatTemplate = modelDir ? this.tryReadLocalText(modelDir, 'chat_template.jinja') : '';
    const runtimeHints = this.deriveRuntimeHints({
      type,
      repoId: canonicalRepoId,
      metadata,
      readme,
      config: localConfig || remoteConfig,
      generationConfig: localGenerationConfig || remoteGenerationConfig,
      preprocessorConfig: localPreprocessorConfig || remotePreprocessorConfig,
      tokenizerConfig: localTokenizerConfig || remoteTokenizerConfig,
      chatTemplate: localChatTemplate || remoteChatTemplate,
    });

    const payload = {
      inspectedAt: new Date().toISOString(),
      repoId: canonicalRepoId,
      type,
      apiMetadata: metadata,
      remoteMetadataError,
      readme,
      files: {
        config: localConfig || remoteConfig,
        generationConfig: localGenerationConfig || remoteGenerationConfig,
        preprocessorConfig: localPreprocessorConfig || remotePreprocessorConfig,
        tokenizerConfig: localTokenizerConfig || remoteTokenizerConfig,
        chatTemplate: localChatTemplate || remoteChatTemplate || '',
      },
      runtimeHints,
    };
    let metadataPath: string | null = null;
    if (modelDir) {
      metadataPath = path.join(modelDir, 'arcsub_model_metadata.json');
      await fs.writeJson(metadataPath, payload, { spaces: 2 });
    }

    return {
      repoId: normalizedRepoId,
      canonicalRepoId,
      type,
      inferredModel: {
        ...inferredModel,
        ...(runtimeHints ? { runtimeHints } : {}),
      },
      runtimeHints,
      metadataPath,
    };
  }

  private static async downloadWithRedirect(url: string, destination: string): Promise<void> {
    await fs.ensureDir(path.dirname(destination));
    const tempPath = `${destination}.part`;

    await new Promise<void>((resolve, reject) => {
      const request = https.get(url, { headers: this.buildDownloadHeaders(url) }, (response) => {
        const status = response.statusCode || 0;
        if ([301, 302, 303, 307, 308].includes(status)) {
          const redirectUrl = response.headers.location;
          response.resume();
          if (!redirectUrl) {
            reject(new Error(`Download redirect without location: ${url}`));
            return;
          }
          const nextUrl = new URL(redirectUrl, url).toString();
          this.downloadWithRedirect(nextUrl, destination).then(resolve).catch(reject);
          return;
        }

        if (status !== 200) {
          const errorCode = String(response.headers['x-error-code'] || '').trim();
          const errorMessage = String(response.headers['x-error-message'] || '').trim();
          response.resume();
          if (
            (status === 401 || status === 403) &&
            (errorCode.toLowerCase() === 'gatedrepo' ||
              /gated|restricted|authenticated|access to model/i.test(errorMessage))
          ) {
            reject(
              new Error(
                `Failed to download model file (${status}): ${url}. ` +
                  `Hugging Face reports gated access (${errorCode || 'GatedRepo'}). ` +
                  `Please request/accept access on the model page and set HF_TOKEN in .env.`
              )
            );
            return;
          }
          reject(new Error(`Failed to download model file (${status}): ${url}`));
          return;
        }

        const writer = fs.createWriteStream(tempPath);
        response.pipe(writer);
        writer.on('finish', async () => {
          try {
            await fs.move(tempPath, destination, { overwrite: true });
            resolve();
          } catch (error) {
            await fs.remove(tempPath).catch(() => {});
            reject(error);
          }
        });
        writer.on('error', async (error) => {
          await fs.remove(tempPath).catch(() => {});
          reject(error);
        });
      });

      request.on('error', async (error) => {
        await fs.remove(tempPath).catch(() => {});
        reject(error);
      });
      request.setTimeout(120000, () => {
        request.destroy(new Error(`Download timeout for ${url}`));
      });
    });
  }

  private static async ensureModelFiles(model: LocalModelDefinition) {
    const modelDir = getLocalModelInstallDir(model);
    await fs.ensureDir(modelDir);
    const downloadRepoId = String(model.downloadRepoId || model.repoId).trim() || model.repoId;

    for (const fileName of model.requiredFiles) {
      const destination = path.join(modelDir, fileName);
      if (await fs.pathExists(destination)) continue;
      const fileUrl = this.getHfFileUrl(downloadRepoId, fileName);
      await this.downloadWithRedirect(fileUrl, destination);
    }
  }

  private static async ensureConvertedQwen3AsrModel(model: LocalModelDefinition) {
    const modelDir = getLocalModelInstallDir(model);
    const checks = await Promise.all(
      model.requiredFiles.map((fileName) => fs.pathExists(path.join(modelDir, fileName)))
    );
    if (checks.every(Boolean)) {
      return;
    }

    await fs.remove(modelDir);
    await fs.ensureDir(modelDir);
    await OpenvinoRuntimeManager.convertOfficialQwenAsrModel({
      repoId: model.repoId,
      outputDir: modelDir,
    });
  }

  private static async ensureAutoConvertedModel(model: LocalModelDefinition) {
    const modelDir = getLocalModelInstallDir(model);
    const probed = inferLocalModelDefinitionFromInstalledDir(model.type, model.repoId, modelDir, {
      id: model.id,
      displayName: model.displayName,
      localSubdir: model.localSubdir,
      downloadRepoId: model.downloadRepoId,
      installMode: model.installMode,
      sourceFormat: model.sourceFormat,
      conversionMethod: model.conversionMethod,
      source: model.source,
    });
    if (probed && (await this.isModelInstalled(probed))) {
      return;
    }

    await fs.remove(modelDir);
    await fs.ensureDir(modelDir);
    await OpenvinoRuntimeManager.convertHuggingFaceModel({
      repoId: String(model.downloadRepoId || model.repoId).trim() || model.repoId,
      outputDir: modelDir,
      type: model.type,
      sourceFormat: model.sourceFormat,
      conversionMethod: model.conversionMethod,
      runtimeLayout: model.runtimeLayout,
    });
  }

  private static async ensureModelInstalledArtifacts(model: LocalModelDefinition) {
    if (model.installMode === 'hf-qwen3-asr-convert') {
      await this.ensureConvertedQwen3AsrModel(model);
      return;
    }
    if (model.installMode === 'hf-auto-convert') {
      await this.ensureAutoConvertedModel(model);
      return;
    }
    await this.ensureModelFiles(model);
  }

  static async isModelInstalled(model: LocalModelDefinition) {
    const modelDir = getLocalModelInstallDir(model);
    const checks = await Promise.all(
      model.requiredFiles.map((fileName) => fs.pathExists(path.join(modelDir, fileName)))
    );
    return checks.every(Boolean);
  }

  private static dedupeModels(models: LocalModelDefinition[]) {
    const byId = new Map<string, LocalModelDefinition>();
    for (const candidate of models) {
      const normalized = sanitizeLocalModelDefinition(candidate);
      if (!normalized) continue;
      const existing = byId.get(normalized.id);
      if (!existing || (existing.source !== 'builtin' && normalized.source === 'builtin')) {
        byId.set(normalized.id, normalized);
      }
    }
    return Array.from(byId.values());
  }

  private static async getRegisteredModels(settings: AppSettings | null | undefined) {
    const configured = Array.isArray(settings?.localModels?.installed)
      ? settings!.localModels.installed
          .map((item) => sanitizeLocalModelDefinition(item))
          .filter(Boolean) as LocalModelDefinition[]
      : [];

    const merged: LocalModelDefinition[] = [...configured];
    const builtinInstallStates = await Promise.all(
      BUILTIN_LOCAL_MODELS.map(async (builtin) => ({
        builtin,
        installed: await this.isModelInstalled(builtin),
      }))
    );
    for (const { builtin, installed } of builtinInstallStates) {
      if (merged.some((item) => item.id === builtin.id)) continue;
      if (!installed) continue;
      merged.push({ ...builtin, requiredFiles: [...builtin.requiredFiles] });
    }

    return this.dedupeModels(merged);
  }

  private static async filterInstalledModels(models: LocalModelDefinition[]) {
    const checks = await Promise.all(
      models.map(async (model) => ({
        model,
        installed: await this.isModelInstalled(model),
      }))
    );
    return checks.filter((item) => item.installed).map((item) => item.model);
  }

  private static normalizeSelection(
    settings: AppSettings | null | undefined,
    availableModels: LocalModelDefinition[]
  ) {
    const defaults = getDefaultLocalModelSelection();
    const current = settings?.localModels || defaults;
    const asrModels = availableModels.filter((model) => model.type === 'asr');
    const translateModels = availableModels.filter((model) => model.type === 'translate');

    return {
      asrSelectedId:
        typeof current.asrSelectedId === 'string' && asrModels.some((model) => model.id === current.asrSelectedId)
          ? current.asrSelectedId
          : asrModels[0]?.id || defaults.asrSelectedId,
      translateSelectedId:
        typeof current.translateSelectedId === 'string' &&
        translateModels.some((model) => model.id === current.translateSelectedId)
          ? current.translateSelectedId
          : translateModels[0]?.id || defaults.translateSelectedId,
    };
  }

  private static buildLocalModelState(
    settings: AppSettings | null | undefined,
    installedModels: LocalModelDefinition[]
  ): LocalModelSelection {
    const selection = this.normalizeSelection(settings, installedModels);
    return {
      ...selection,
      installed: installedModels.map((model) => ({
        ...model,
        requiredFiles: [...model.requiredFiles],
      })),
    };
  }

  private static sortOverviewEntries(
    items: LocalModelOverviewEntry[],
    selection: { asrSelectedId: string; translateSelectedId: string }
  ) {
    const score = (item: LocalModelOverviewEntry) => {
      if (item.type === 'asr') return item.id === selection.asrSelectedId ? 0 : 1;
      return item.id === selection.translateSelectedId ? 0 : 1;
    };

    return [...items].sort((a, b) => {
      const scoreDelta = score(a) - score(b);
      if (scoreDelta !== 0) return scoreDelta;
      if (a.type !== b.type) return a.type.localeCompare(b.type);
      return a.name.localeCompare(b.name, undefined, { sensitivity: 'base' });
    });
  }

  private static async getCatalogView(settings: AppSettings) {
    const registered = await this.getRegisteredModels(settings);
    const installedModels = await this.filterInstalledModels(registered);

    const selection = this.normalizeSelection(settings, installedModels);
    const entries = installedModels.map((model) => ({
      id: model.id,
      type: model.type,
      name: model.displayName,
      repoId: model.repoId,
      sourceFormat: model.sourceFormat,
      conversionMethod: model.conversionMethod,
      runtimeLayout: model.runtimeLayout,
      runtime: model.runtime,
      runtimeHints: model.runtimeHints,
      selected: model.type === 'asr' ? selection.asrSelectedId === model.id : selection.translateSelectedId === model.id,
      installError: this.installErrors.get(model.id) || null,
    }));

    return {
      catalog: this.sortOverviewEntries(entries, selection),
      selection,
      installs: this.getInstallStatusList(),
    };
  }

  private static async persistInstalledModels(
    settings: AppSettings,
    installedModels: LocalModelDefinition[],
    selectionOverrides?: Partial<Pick<LocalModelSelection, 'asrSelectedId' | 'translateSelectedId'>>
  ) {
    const localModels = this.buildLocalModelState(settings, this.dedupeModels(installedModels));
    if (selectionOverrides?.asrSelectedId) {
      localModels.asrSelectedId = selectionOverrides.asrSelectedId;
    }
    if (selectionOverrides?.translateSelectedId) {
      localModels.translateSelectedId = selectionOverrides.translateSelectedId;
    }
    await SettingsManager.updateSettings({ localModels });
  }

  private static async resolveRegisteredModel(settings: AppSettings, modelId: string) {
    const registered = await this.getRegisteredModels(settings);
    return registered.find((model) => model.id === modelId) || null;
  }

  private static async resolveInstalledModels(settings: AppSettings) {
    const registered = await this.getRegisteredModels(settings);
    return this.filterInstalledModels(registered);
  }

  static async getLocalModelsOverview(options: { forceFresh?: boolean } = {}) {
    if (!options.forceFresh && this.overviewCache && Date.now() - this.overviewCache.at < this.OVERVIEW_CACHE_MS) {
      return this.overviewCache.value;
    }

    const overview = await this.buildOverview();
    this.overviewCache = { value: overview, at: Date.now() };
    return overview;
  }

  static async selectModel(type: LocalModelType, modelId: string) {
    const settings = await SettingsManager.getSettings({ mask: false });
    const installedModels = await this.resolveInstalledModels(settings);
    const model = installedModels.find((item) => item.id === modelId);
    if (!model || model.type !== type) {
      throw new Error('Invalid local model selection.');
    }

    const currentState = this.buildLocalModelState(settings, installedModels);
    const nextState =
      type === 'asr'
        ? { ...currentState, asrSelectedId: modelId }
        : { ...currentState, translateSelectedId: modelId };

    await SettingsManager.updateSettings({ localModels: nextState });
    this.invalidateOverviewCache();
    return this.getLocalModelsOverview({ forceFresh: true });
  }

  static async inspectModel(type: LocalModelType, repoIdOrUrl: string) {
    const inspection = await this.inspectModelMetadata(type, repoIdOrUrl);
    return {
      repoId: inspection.canonicalRepoId,
      type,
      inferredModel: {
        id: inspection.inferredModel.id,
        name: inspection.inferredModel.displayName,
        repoId: inspection.inferredModel.repoId,
        sourceFormat: inspection.inferredModel.sourceFormat,
        conversionMethod: inspection.inferredModel.conversionMethod,
        runtimeLayout: inspection.inferredModel.runtimeLayout,
        runtime: inspection.inferredModel.runtime,
      },
      runtimeHints: inspection.runtimeHints || null,
    };
  }

  private static async runInstallModelTask(modelId: string, model: LocalModelDefinition) {
    try {
      this.updateInstallState(modelId, {
        phase: this.getArtifactInstallPhase(model),
        message: null,
        error: null,
      });
      await this.ensureModelInstalledArtifacts(model);

      this.updateInstallState(modelId, {
        phase: 'verifying',
        message: null,
      });
      const postInstallInspection = await this.inspectModelMetadata(model.type, model.repoId, {
        modelDir: getLocalModelInstallDir(model),
        inferredModel: model,
        remoteMetadataOptional: true,
      });
      const finalizedModel =
        inferLocalModelDefinitionFromInstalledDir(model.type, model.repoId, getLocalModelInstallDir(model), {
          id: model.id,
          displayName: model.displayName,
          localSubdir: model.localSubdir,
          downloadRepoId: model.downloadRepoId,
          installMode: model.installMode,
          sourceFormat: model.sourceFormat,
          conversionMethod: model.conversionMethod,
          source: model.source,
          runtimeHints: postInstallInspection.runtimeHints || model.runtimeHints,
        }) || null;
      if (!finalizedModel) {
        throw new Error(
          'Model conversion completed, but the resulting OpenVINO files do not match a supported ArcSub runtime layout.'
        );
      }

      this.updateInstallState(modelId, {
        phase: 'persisting',
        message: null,
      });
      const latestSettings = await SettingsManager.getSettings({ mask: false });
      const registered = await this.getRegisteredModels(latestSettings);
      const nextInstalled = this.dedupeModels([
        ...registered.filter((item) => item.id !== finalizedModel.id),
        finalizedModel,
      ]);
      await this.persistInstalledModels(latestSettings, nextInstalled, {
        asrSelectedId: finalizedModel.type === 'asr' ? finalizedModel.id : undefined,
        translateSelectedId: finalizedModel.type === 'translate' ? finalizedModel.id : undefined,
      });

      this.installErrors.delete(modelId);
      this.updateInstallState(modelId, {
        installing: false,
        phase: 'completed',
        completedAt: Date.now(),
        message: null,
        error: null,
      });
    } catch (error: any) {
      const message = String(error?.message || error || 'Model installation failed.');
      this.installErrors.set(modelId, message);
      this.updateInstallState(modelId, {
        installing: false,
        phase: 'failed',
        completedAt: Date.now(),
        message,
        error: message,
      });
      throw new Error(message);
    }
  }

  static async startInstallModel(type: LocalModelType, repoIdOrUrl: string) {
    const inspection = await this.inspectModelMetadata(type, repoIdOrUrl);
    const canonicalRepoId = inspection.canonicalRepoId;
    const inferredModel = inspection.inferredModel;
    const modelId = inferredModel.id || buildLocalModelId(type, canonicalRepoId);

    const existingTask = this.installTasks.get(modelId);
    if (existingTask) {
      this.invalidateOverviewCache();
      return {
        ...(await this.getLocalModelsOverview({ forceFresh: true })),
        install: this.getInstallStateSnapshot(modelId),
      };
    }

    const settings = await SettingsManager.getSettings({ mask: false });
    const registered = await this.getRegisteredModels(settings);
    const existing = registered.find((model) => model.id === modelId);
    const model: LocalModelDefinition = existing
      ? {
          ...existing,
          ...inferredModel,
          runtimeHints: inferredModel.runtimeHints || existing.runtimeHints,
          source: existing.source === 'builtin' ? 'builtin' : inferredModel.source,
        }
      : inferredModel;

    const latestTask = this.installTasks.get(modelId);
    if (latestTask) {
      this.invalidateOverviewCache();
      return {
        ...(await this.getLocalModelsOverview({ forceFresh: true })),
        install: this.getInstallStateSnapshot(modelId),
      };
    }

    const startedAt = Date.now();
    this.installErrors.delete(modelId);
    this.installStates.set(modelId, {
      modelId,
      type: model.type,
      repoId: model.repoId,
      name: model.displayName,
      installing: true,
      phase: 'queued',
      startedAt,
      updatedAt: startedAt,
      message: null,
      error: null,
    });
    this.invalidateOverviewCache();

    const installTask = this.runInstallModelTask(modelId, model);
    this.installTasks.set(modelId, installTask);
    void installTask.catch(() => undefined);
    void installTask
      .finally(() => {
        if (this.installTasks.get(modelId) === installTask) {
          this.installTasks.delete(modelId);
        }
        this.invalidateOverviewCache();
      })
      .catch(() => undefined);

    return {
      ...(await this.getLocalModelsOverview({ forceFresh: true })),
      install: this.getInstallStateSnapshot(modelId),
    };
  }

  static async installModel(type: LocalModelType, repoIdOrUrl: string) {
    const started = await this.startInstallModel(type, repoIdOrUrl);
    const modelId = started.install?.modelId || null;
    const installTask = modelId ? this.installTasks.get(modelId) : null;
    if (installTask) {
      await installTask;
    }
    const state = modelId ? this.getInstallStateSnapshot(modelId) : null;
    if (state?.phase === 'failed' && state.error) {
      throw new Error(state.error);
    }
    return this.getLocalModelsOverview({ forceFresh: true });
  }

  static async removeModel(modelId: string) {
    const settings = await SettingsManager.getSettings({ mask: false });
    const model = await this.resolveRegisteredModel(settings, modelId);
    if (!model) {
      throw new Error('Local model not found.');
    }

    if (this.installStates.get(modelId)?.installing || this.installTasks.has(modelId)) {
      throw new Error('Model installation is still in progress. Please wait before removing it.');
    }

    try {
      if (model.type === 'asr') {
        await OpenvinoRuntimeManager.releaseAsrRuntime();
      } else {
        await OpenvinoRuntimeManager.releaseTranslateRuntime();
      }
    } catch (error: any) {
      throw new Error(
        `Failed to release local ${model.type} runtime before remove: ${String(
          error?.message || error || 'unknown error'
        )}`
      );
    }

    const modelDir = getLocalModelInstallDir(model);
    await fs.remove(modelDir);
    this.installErrors.delete(modelId);

    const registered = await this.getRegisteredModels(settings);
    const nextInstalled = registered.filter((item) => item.id !== modelId);
    await this.persistInstalledModels(settings, nextInstalled);

    this.invalidateOverviewCache();
    return this.getLocalModelsOverview({ forceFresh: true });
  }

  static async getRuntimeModels() {
    const settings = await SettingsManager.getSettings();
    const installedLocalModels = await this.resolveInstalledModels(settings as any);
    const selection = this.normalizeSelection(settings as any, installedLocalModels);

    const asrModels = Array.isArray(settings.asrModels) ? [...settings.asrModels] : [];
    const translateModels = Array.isArray(settings.translateModels) ? [...settings.translateModels] : [];

    const localAsrModels = installedLocalModels
      .filter((model) => model.type === 'asr')
      .sort((a, b) => {
        const aSelected = a.id === selection.asrSelectedId ? 0 : 1;
        const bSelected = b.id === selection.asrSelectedId ? 0 : 1;
        if (aSelected !== bSelected) return aSelected - bSelected;
        return a.displayName.localeCompare(b.displayName, undefined, { sensitivity: 'base' });
      });

    for (const local of localAsrModels) {
      if (asrModels.some((model: any) => model?.id === local.id)) continue;
      asrModels.push({
        id: local.id,
        name: `${local.displayName} (Local)`,
        url: 'local://openvino/asr',
        key: '',
        model: local.repoId,
        isLocal: true,
        provider: 'local-openvino',
      });
    }

    const localTranslateModels = installedLocalModels
      .filter((model) => model.type === 'translate')
      .sort((a, b) => {
        const aSelected = a.id === selection.translateSelectedId ? 0 : 1;
        const bSelected = b.id === selection.translateSelectedId ? 0 : 1;
        if (aSelected !== bSelected) return aSelected - bSelected;
        return a.displayName.localeCompare(b.displayName, undefined, { sensitivity: 'base' });
      });

    for (const local of localTranslateModels) {
      if (translateModels.some((model: any) => model?.id === local.id)) continue;
      translateModels.push({
        id: local.id,
        name: `${local.displayName} (OpenVINO Local)`,
        url: 'local://openvino/translate',
        key: '',
        model: local.repoId,
        isLocal: true,
        provider: 'local-openvino',
      });
    }

    return {
      asrModels,
      translateModels,
      localSelection: selection,
    };
  }

  static async getOpenvinoStatus() {
    return OpenvinoRuntimeManager.getOpenvinoRuntimeStatus();
  }

  static async resolveLocalModelForRequest(
    type: LocalModelType,
    modelId: string | undefined,
    settings?: AppSettings
  ) {
    if (!modelId) return null;

    const resolvedSettings = settings || (await SettingsManager.getSettings({ mask: false }));
    const localModel = await this.resolveRegisteredModel(resolvedSettings, modelId);
    if (!localModel) {
      if (String(modelId).startsWith('local_')) {
        throw new Error('Invalid local model for this request.');
      }
      return null;
    }

    if (localModel.type !== type) {
      throw new Error('Invalid local model for this request.');
    }

    if (!(await this.isModelInstalled(localModel))) {
      throw new Error(
        type === 'asr'
          ? 'Local ASR model is not installed. Please install it in Settings first.'
          : 'Local translation model is not installed. Please install it in Settings first.'
      );
    }

    return localModel;
  }

  static async releaseLocalRuntimes(target: 'asr' | 'translate' | 'all') {
    console.log(
      `[${new Date().toISOString()}] [LocalModelService] releaseLocalRuntimes start ${JSON.stringify({ target })}`
    );
    const errors: string[] = [];
    if (target === 'asr' || target === 'all') {
      try {
        await OpenvinoRuntimeManager.releaseAsrRuntime();
      } catch (error: any) {
        errors.push(String(error?.message || error || 'Failed to release local ASR runtime.'));
      }
    }
    if (target === 'translate' || target === 'all') {
      try {
        await OpenvinoRuntimeManager.releaseTranslateRuntime();
      } catch (error: any) {
        errors.push(String(error?.message || error || 'Failed to release local translation runtime.'));
      }
    }

    const result = {
      success: errors.length === 0,
      released: {
        asr: target === 'asr' || target === 'all',
        translate: target === 'translate' || target === 'all',
      },
      errors,
    };

    console.log(
      `[${new Date().toISOString()}] [LocalModelService] releaseLocalRuntimes done ${JSON.stringify({
        target,
        success: result.success,
        released: result.released,
        errorCount: result.errors.length,
      })}`
    );

    return result;
  }

  static async preloadLocalRuntime(target: 'asr' | 'translate', modelId: string) {
    const settings = await SettingsManager.getSettings({ mask: false });
    const localModel = await this.resolveLocalModelForRequest(target, modelId, settings);
    if (!localModel) {
      throw new Error('Only installed local models can be preloaded.');
    }

    if (target === 'asr') {
      throw new Error('ASR preload is not implemented for this route.');
    }

    const modelPath = getLocalModelInstallDir(localModel);
    const result = await OpenvinoRuntimeManager.preloadTranslateRuntime({
      modelId: localModel.id,
      modelPath,
    });

    return {
      success: true,
      target,
      modelId: localModel.id,
      runtimeDebug: result.runtimeDebug,
    };
  }

  static listModelsByType(type: LocalModelType) {
    return BUILTIN_LOCAL_MODELS.filter((model) => model.type === type);
  }
}
