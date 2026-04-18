import type { LlmAdapterKey } from '../canonical/llm_capabilities.js';
import type { CanonicalReasoningConfig, CanonicalSamplingConfig } from '../canonical/llm_types.js';

export interface LlmModelProfile {
  id: string;
  family: string;
  adapterKey: LlmAdapterKey;
  defaultModel?: string;
  matchPrefixes?: string[];
  matchPatterns?: RegExp[];
  supportsJsonSchema?: boolean;
  supportsReasoning?: boolean;
  supportsTools?: boolean;
  defaultSampling?: CanonicalSamplingConfig;
  defaultReasoning?: CanonicalReasoningConfig;
  warnings?: string[];
}

export function matchesLlmModelProfile(profile: LlmModelProfile, modelName: string) {
  const normalized = String(modelName || '').trim().toLowerCase();
  if (!normalized) return false;

  if (Array.isArray(profile.matchPrefixes) && profile.matchPrefixes.some((prefix) => normalized.startsWith(prefix.toLowerCase()))) {
    return true;
  }

  if (Array.isArray(profile.matchPatterns) && profile.matchPatterns.some((pattern) => pattern.test(normalized))) {
    return true;
  }

  return false;
}
