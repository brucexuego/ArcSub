import type { ApiModelRequestOptions } from '../../../src/types.js';
import type {
  CloudTranslateExecutionMode,
  ResolvedCloudTranslateProvider,
} from './types.js';

export interface CloudTranslateExecutionPlan {
  executionMode: CloudTranslateExecutionMode;
  enableJsonLineRepair: boolean;
  supportsContextMode: boolean;
  warnings: string[];
}

function normalizeMode(value: unknown): CloudTranslateExecutionMode {
  const raw = String(value || '').trim().toLowerCase();
  if (
    raw === 'auto' ||
    raw === 'cloud_context' ||
    raw === 'cloud_strict' ||
    raw === 'cloud_relaxed' ||
    raw === 'provider_native'
  ) {
    return raw;
  }
  return 'auto';
}

export function buildCloudTranslateExecutionPlan(input: {
  resolvedProvider: ResolvedCloudTranslateProvider;
  modelOptions?: ApiModelRequestOptions;
  requestedJsonLineRepair?: boolean;
  hasPromptTemplate?: boolean;
  hasCustomPrompt?: boolean;
  isConnectionTest?: boolean;
}): CloudTranslateExecutionPlan {
  const requestedMode = normalizeMode(input.modelOptions?.translation?.executionMode);
  const defaultMode = input.resolvedProvider.defaultExecutionMode || 'auto';
  const executionMode = requestedMode === 'auto' ? defaultMode : requestedMode;
  const manualStrict = input.requestedJsonLineRepair === true;
  const warnings: string[] = [];

  let enableJsonLineRepair = manualStrict || executionMode === 'cloud_strict';
  if (input.resolvedProvider.capabilities.translationNative && executionMode !== 'cloud_strict') {
    enableJsonLineRepair = manualStrict;
  }
  if (input.isConnectionTest) {
    enableJsonLineRepair = false;
  }

  const supportsContextMode =
    input.resolvedProvider.capabilities.supportsLargeContext &&
    (executionMode === 'cloud_context' || executionMode === 'cloud_relaxed' || executionMode === 'auto');

  if (manualStrict && input.resolvedProvider.quotaProfile) {
    warnings.push('cloud_quota_strict_mode_may_increase_calls');
  }

  return {
    executionMode,
    enableJsonLineRepair,
    supportsContextMode,
    warnings,
  };
}
