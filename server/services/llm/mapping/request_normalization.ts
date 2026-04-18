import { getCanonicalProviderCapabilities, type LlmAdapterKey } from '../canonical/llm_capabilities.js';
import type {
  CanonicalLlmRequest,
  CanonicalReasoningConfig,
  CanonicalSamplingConfig,
  CanonicalStructuredOutput,
  CanonicalToolConfig,
} from '../canonical/llm_types.js';
import type { LlmModelProfile } from '../profiles/base.js';
import { llmModelProfileRegistry } from '../profiles/registry.js';

export interface NormalizeCanonicalLlmRequestInput {
  request: CanonicalLlmRequest;
  adapterKey: LlmAdapterKey;
  modelName?: string;
}

export interface NormalizedCanonicalLlmRequestResult {
  request: CanonicalLlmRequest;
  profile: LlmModelProfile | null;
  warnings: string[];
}

function dedupeWarnings(values: string[]) {
  return Array.from(new Set(values.filter(Boolean)));
}

function mergeSampling(
  defaults: CanonicalSamplingConfig | undefined,
  request: CanonicalSamplingConfig | undefined
): CanonicalSamplingConfig | undefined {
  const merged: CanonicalSamplingConfig = {
    ...(defaults || {}),
    ...(request || {}),
  };
  return Object.keys(merged).length > 0 ? merged : undefined;
}

function mergeReasoning(
  defaults: CanonicalReasoningConfig | undefined,
  request: CanonicalReasoningConfig | undefined
): CanonicalReasoningConfig | undefined {
  const merged: CanonicalReasoningConfig = {
    ...(defaults || {}),
    ...(request || {}),
  };
  return Object.keys(merged).length > 0 ? merged : undefined;
}

function normalizeStructuredOutput(
  structuredOutput: CanonicalStructuredOutput | undefined,
  input: NormalizeCanonicalLlmRequestInput,
  profile: LlmModelProfile | null,
  warnings: string[]
) {
  if (!structuredOutput) return structuredOutput;

  const capabilities = getCanonicalProviderCapabilities(input.adapterKey);
  const profileSupportsJsonSchema = profile?.supportsJsonSchema;

  if (structuredOutput.mode === 'json_schema') {
    if (!capabilities.supportsJsonSchema || profileSupportsJsonSchema === false) {
      if (capabilities.supportsJsonObject) {
        warnings.push('structured_output_json_schema_downgraded_to_json_object');
        return { mode: 'json_object' } satisfies CanonicalStructuredOutput;
      }
      warnings.push('structured_output_json_schema_downgraded_to_text');
      return { mode: 'text' } satisfies CanonicalStructuredOutput;
    }
    return structuredOutput;
  }

  if (structuredOutput.mode === 'json_object' && !capabilities.supportsJsonObject) {
    warnings.push('structured_output_json_object_downgraded_to_text');
    return { mode: 'text' } satisfies CanonicalStructuredOutput;
  }

  return structuredOutput;
}

function normalizeSampling(
  sampling: CanonicalSamplingConfig | undefined,
  input: NormalizeCanonicalLlmRequestInput,
  warnings: string[]
) {
  if (!sampling) return undefined;
  const capabilities = getCanonicalProviderCapabilities(input.adapterKey);
  const normalized = { ...sampling };

  if (!capabilities.supportsTopK && normalized.topK != null) {
    if (input.request.sampling?.topK != null) warnings.push('sampling_topk_removed');
    delete normalized.topK;
  }
  if (!capabilities.supportsTopP && normalized.topP != null) {
    if (input.request.sampling?.topP != null) warnings.push('sampling_topp_removed');
    delete normalized.topP;
  }
  if (!capabilities.supportsSeed && normalized.seed != null) {
    if (input.request.sampling?.seed != null) warnings.push('sampling_seed_removed');
    delete normalized.seed;
  }

  return Object.keys(normalized).length > 0 ? normalized : undefined;
}

function normalizeReasoning(
  reasoning: CanonicalReasoningConfig | undefined,
  input: NormalizeCanonicalLlmRequestInput,
  profile: LlmModelProfile | null,
  warnings: string[]
) {
  if (!reasoning) return undefined;
  const capabilities = getCanonicalProviderCapabilities(input.adapterKey);
  if (!capabilities.supportsReasoningConfig || profile?.supportsReasoning === false) {
    if (input.request.reasoning) warnings.push('reasoning_removed_unsupported');
    return undefined;
  }
  return reasoning;
}

function normalizeTooling(
  tooling: CanonicalToolConfig | undefined,
  input: NormalizeCanonicalLlmRequestInput,
  profile: LlmModelProfile | null,
  warnings: string[]
) {
  if (!tooling) return undefined;
  const capabilities = getCanonicalProviderCapabilities(input.adapterKey);
  if (!capabilities.supportsTools || profile?.supportsTools === false) {
    if (input.request.tooling) warnings.push('tooling_removed_unsupported');
    return undefined;
  }

  const normalized: CanonicalToolConfig = { ...tooling };
  if (normalized.strict && !capabilities.supportsStrictToolSchema) {
    warnings.push('tooling_strict_disabled');
    normalized.strict = false;
  }
  if (normalized.allowParallel && !capabilities.supportsParallelToolCalls) {
    warnings.push('tooling_parallel_disabled');
    normalized.allowParallel = false;
  }

  return normalized;
}

export function normalizeCanonicalLlmRequest(input: NormalizeCanonicalLlmRequestInput): NormalizedCanonicalLlmRequestResult {
  const profile =
    llmModelProfileRegistry.match(String(input.request.model || '').trim() || String(input.modelName || '').trim()) || null;

  const warnings: string[] = [];
  const normalizedSampling = normalizeSampling(
    mergeSampling(profile?.defaultSampling, input.request.sampling),
    input,
    warnings
  );
  const normalizedReasoning = normalizeReasoning(
    mergeReasoning(profile?.defaultReasoning, input.request.reasoning),
    input,
    profile,
    warnings
  );
  const normalizedTooling = normalizeTooling(input.request.tooling, input, profile, warnings);
  const normalizedStructuredOutput = normalizeStructuredOutput(input.request.structuredOutput, input, profile, warnings);
  const capabilities = getCanonicalProviderCapabilities(input.adapterKey);

  const nextRequest: CanonicalLlmRequest = {
    ...input.request,
    model: String(input.request.model || '').trim() || String(profile?.defaultModel || '').trim(),
    structuredOutput: normalizedStructuredOutput,
    sampling: normalizedSampling,
    reasoning: normalizedReasoning,
    tooling: normalizedTooling,
  };

  if (!capabilities.supportsConversationReference && nextRequest.conversationRef) {
    if (input.request.conversationRef) warnings.push('conversation_ref_removed');
    delete nextRequest.conversationRef;
  }

  return {
    request: nextRequest,
    profile,
    warnings: dedupeWarnings(warnings),
  };
}
