import type { ApiModelRequestOptions, ApiModelTranslationBatchingOptions } from '../../../../src/types.js';
import type {
  CloudTranslateBatchingProfile,
  CloudTranslateBatchingSource,
  ResolvedCloudTranslateProvider,
} from '../types.js';

type EnvNumberReader = (name: string, fallback: number, min?: number, max?: number) => number;
type EnvBooleanReader = (name: string, fallback?: boolean) => boolean;

function toLimit(value: unknown, fallback: number, min: number, max: number) {
  const parsed = Number(value);
  const next = Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
  return Math.max(min, Math.min(max, Math.round(next)));
}

function toBoolean(value: unknown, fallback: boolean) {
  if (typeof value === 'boolean') return value;
  if (value == null) return fallback;
  const raw = String(value).trim();
  if (/^(1|true|yes|on)$/i.test(raw)) return true;
  if (/^(0|false|no|off)$/i.test(raw)) return false;
  return fallback;
}

function getHost(endpointUrl: string) {
  try {
    return new URL(endpointUrl).hostname.toLowerCase();
  } catch {
    return '';
  }
}

function getOptionBatching(options?: ApiModelRequestOptions): ApiModelTranslationBatchingOptions | null {
  const value = options?.translation?.batching;
  return value && typeof value === 'object' ? value : null;
}

function buildProfile(input: {
  source: CloudTranslateBatchingSource;
  defaults: CloudTranslateBatchingProfile;
  options?: ApiModelTranslationBatchingOptions | null;
}) {
  const options = input.options || null;
  const source: CloudTranslateBatchingSource = options ? 'model-options' : input.source;
  return {
    enabled: toBoolean(options?.enabled, input.defaults.enabled),
    source,
    targetLines: toLimit(options?.targetLines, input.defaults.targetLines, 1, 240),
    minTargetLines: toLimit(options?.minTargetLines, input.defaults.minTargetLines, 1, 120),
    charBudget: toLimit(options?.charBudget, input.defaults.charBudget, 120, 50000),
    maxSplitDepth: toLimit(options?.maxSplitDepth, input.defaults.maxSplitDepth, 0, 12),
    maxOutputTokens: toLimit(options?.maxOutputTokens, input.defaults.maxOutputTokens, 128, 65536),
    timeoutMs: toLimit(options?.timeoutMs, input.defaults.timeoutMs, 30000, 900000),
    stream: toBoolean(options?.stream, input.defaults.stream),
  } satisfies CloudTranslateBatchingProfile;
}

export function resolveCloudTranslateBatchingProfile(input: {
  resolvedProvider: ResolvedCloudTranslateProvider;
  modelOptions?: ApiModelRequestOptions;
  readEnvNumber: EnvNumberReader;
  readEnvBoolean: EnvBooleanReader;
  requestTimeoutMs: number;
}): CloudTranslateBatchingProfile | null {
  const optionBatching = getOptionBatching(input.modelOptions);
  const host = getHost(input.resolvedProvider.endpointUrl);
  const provider = input.resolvedProvider.provider;

  if (optionBatching?.enabled === false) return null;

  const fallbackProfile = {
    enabled: false,
    source: 'provider-profile' as const,
    targetLines: 24,
    minTargetLines: 6,
    charBudget: 2400,
    maxSplitDepth: 3,
    maxOutputTokens: 2048,
    timeoutMs: input.requestTimeoutMs,
    stream: false,
  } satisfies CloudTranslateBatchingProfile;

  if (provider === 'openai-compatible' && host === 'integrate.api.nvidia.com') {
    if (!input.readEnvBoolean('TRANSLATE_NVIDIA_CLOUD_BATCHING', true) && !optionBatching) return null;
    return buildProfile({
      source: 'nvidia-hosted',
      defaults: {
        ...fallbackProfile,
        enabled: true,
        source: 'nvidia-hosted',
        targetLines: Math.round(input.readEnvNumber('TRANSLATE_NVIDIA_CLOUD_BATCH_SIZE', 24, 4, 120)),
        minTargetLines: Math.round(input.readEnvNumber('TRANSLATE_NVIDIA_CLOUD_MIN_BATCH_SIZE', 6, 1, 60)),
        charBudget: Math.round(input.readEnvNumber('TRANSLATE_NVIDIA_CLOUD_BATCH_CHAR_BUDGET', 2400, 200, 20000)),
        maxSplitDepth: Math.round(input.readEnvNumber('TRANSLATE_NVIDIA_CLOUD_MAX_SPLIT_DEPTH', 4, 0, 8)),
        maxOutputTokens: Math.round(input.readEnvNumber('TRANSLATE_NVIDIA_CLOUD_MAX_OUTPUT_TOKENS', 2048, 256, 16384)),
        timeoutMs: Math.round(input.readEnvNumber('TRANSLATE_NVIDIA_CLOUD_TIMEOUT_MS', input.requestTimeoutMs, 30000, 900000)),
        stream: input.readEnvBoolean('TRANSLATE_NVIDIA_CLOUD_STREAM', false),
      },
      options: optionBatching,
    });
  }

  if (provider === 'openai-compatible' && host === 'models.github.ai') {
    return buildProfile({
      source: 'github-models',
      defaults: {
        ...fallbackProfile,
        enabled: optionBatching?.enabled === true,
        source: 'github-models',
        targetLines: 24,
        minTargetLines: 4,
        charBudget: 2200,
        maxSplitDepth: 3,
        maxOutputTokens: 2048,
      },
      options: optionBatching,
    });
  }

  if (provider === 'gemini-native') {
    return buildProfile({
      source: 'gemini-native',
      defaults: {
        ...fallbackProfile,
        enabled: optionBatching?.enabled === true,
        source: 'gemini-native',
        targetLines: 48,
        minTargetLines: 8,
        charBudget: 6000,
        maxSplitDepth: 2,
        maxOutputTokens: 4096,
        stream: true,
      },
      options: optionBatching,
    });
  }

  if (optionBatching) {
    return buildProfile({
      source: provider === 'openai-compatible' ? 'openai-compatible' : 'provider-profile',
      defaults: {
        ...fallbackProfile,
        enabled: true,
        source: provider === 'openai-compatible' ? 'openai-compatible' : 'provider-profile',
      },
      options: optionBatching,
    });
  }

  return null;
}
