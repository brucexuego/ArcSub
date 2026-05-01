import type { ApiModelQuotaOptions, ApiModelRequestOptions } from '../../../../src/types.js';
import type { CloudTranslateProvider, CloudTranslateQuotaProfile } from '../types.js';

function toLimit(value: unknown) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? Math.round(parsed) : null;
}

function fromOptions(options?: ApiModelRequestOptions): ApiModelQuotaOptions | null {
  return options?.translation?.quota || options?.quota || null;
}

export function resolveCloudTranslateQuotaProfile(input: {
  provider: CloudTranslateProvider;
  model?: string;
  options?: ApiModelRequestOptions;
  defaults?: CloudTranslateQuotaProfile | null;
}): CloudTranslateQuotaProfile | null {
  const override = fromOptions(input.options);
  const defaults = input.defaults || null;
  const merged: CloudTranslateQuotaProfile = {
    ...(defaults || {}),
    ...(override
      ? {
          rpm: toLimit(override.rpm),
          tpm: toLimit(override.tpm),
          rpd: toLimit(override.rpd),
          maxConcurrency: toLimit(override.maxConcurrency),
        }
      : {}),
  };

  const hasAnyLimit =
    toLimit(merged.rpm) ||
    toLimit(merged.tpm) ||
    toLimit(merged.rpd) ||
    toLimit(merged.maxConcurrency);
  if (!hasAnyLimit) return null;

  return {
    ...merged,
    rpm: toLimit(merged.rpm),
    tpm: toLimit(merged.tpm),
    rpd: toLimit(merged.rpd),
    maxConcurrency: toLimit(merged.maxConcurrency),
    safetyReserveTokens: toLimit(merged.safetyReserveTokens) || null,
    tokenEstimator:
      merged.tokenEstimator ||
      (input.provider === 'gemini-native'
        ? 'gemini_like'
        : input.provider === 'openai-compatible'
          ? 'openai_like'
          : 'chars_heuristic'),
    profileId: merged.profileId || `${input.provider}:${String(input.model || '').trim() || 'default'}`,
  };
}
