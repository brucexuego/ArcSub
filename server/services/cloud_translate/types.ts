import type { ApiModelRequestOptions } from '../../../src/types.js';
import type { LlmAdapterKey } from '../llm/canonical/llm_capabilities.js';

export type CloudTranslateProvider =
  | 'deepl'
  | 'anthropic'
  | 'gemini-native'
  | 'mistral-chat'
  | 'cohere-chat'
  | 'xai-chat'
  | 'ollama-chat'
  | 'ollama-generate'
  | 'openai-compatible';

export type CloudTranslateRuntimeAdapterKey =
  | LlmAdapterKey
  | 'deepl-translate'
  | 'ollama-chat'
  | 'ollama-generate';

export type CloudTranslateExecutionMode =
  | 'auto'
  | 'cloud_context'
  | 'cloud_strict'
  | 'cloud_relaxed'
  | 'provider_native';

export interface CloudTranslateProviderCapabilities {
  supportsLargeContext: boolean;
  supportsJsonObject: boolean;
  supportsJsonSchema: boolean;
  supportsStreaming: boolean;
  supportsReasoningConfig: boolean;
  supportsTopK: boolean;
  translationNative: boolean;
  defaultExecutionMode: CloudTranslateExecutionMode;
  profileId?: string | null;
  profileFamily?: string | null;
  nativeProviderLabel?: string | null;
}

export interface CloudTranslateQuotaProfile {
  rpm?: number | null;
  tpm?: number | null;
  rpd?: number | null;
  maxConcurrency?: number | null;
  safetyReserveTokens?: number | null;
  tokenEstimator?: 'chars_heuristic' | 'gemini_like' | 'openai_like';
  profileId?: string | null;
}

export type CloudTranslateBatchingSource =
  | 'provider-profile'
  | 'model-options'
  | 'nvidia-hosted'
  | 'github-models'
  | 'gemini-native'
  | 'openai-compatible';

export interface CloudTranslateBatchingProfile {
  enabled: boolean;
  source: CloudTranslateBatchingSource;
  targetLines: number;
  minTargetLines: number;
  charBudget: number;
  maxSplitDepth: number;
  maxOutputTokens: number;
  timeoutMs: number;
  stream: boolean;
}

export interface CloudTranslateProviderMatchInput {
  rawUrl: string;
  parsedUrl: URL;
  hostname: string;
  pathname: string;
  modelName: string;
}

export interface CloudTranslateProviderDefinition {
  provider: CloudTranslateProvider;
  defaultModel: string;
  adapterKey: CloudTranslateRuntimeAdapterKey;
  capabilities: CloudTranslateProviderCapabilities;
  detect(input: CloudTranslateProviderMatchInput): boolean;
  buildEndpointUrl(rawUrl: string, model?: string, apiKey?: string): string;
  getQuotaProfile?(input: {
    model?: string;
    options?: ApiModelRequestOptions;
  }): CloudTranslateQuotaProfile | null;
}

export interface ResolveCloudTranslateProviderInput {
  url: string;
  modelName?: string;
  model?: string;
  apiKey?: string;
  options?: ApiModelRequestOptions;
}

export interface ResolvedCloudTranslateProvider {
  provider: CloudTranslateProvider;
  endpointUrl: string;
  effectiveModel: string;
  adapterKey: CloudTranslateRuntimeAdapterKey;
  profileId: string | null;
  profileFamily: string | null;
  capabilities: CloudTranslateProviderCapabilities;
  quotaProfile: CloudTranslateQuotaProfile | null;
  defaultExecutionMode: CloudTranslateExecutionMode;
}
