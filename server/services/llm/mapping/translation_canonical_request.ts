import type { LlmAdapterKey } from '../canonical/llm_capabilities.js';
import type { CanonicalLlmRequest, CanonicalMessage, CanonicalStructuredOutput } from '../canonical/llm_types.js';
import { normalizeCanonicalLlmRequest } from './request_normalization.js';
import type { LlmModelProfile } from '../profiles/base.js';
import { resolveProviderTranslationProfile } from '../profiles/translation_profiles.js';

export interface BuildCanonicalTranslationRequestInput {
  adapterKey: LlmAdapterKey;
  providerFamily?: string;
  model?: string;
  text: string;
  sourceLang?: string;
  targetLang: string;
  systemPrompt?: string;
  jsonResponse?: boolean;
  isConnectionTest?: boolean;
  promptTemplateId?: string;
  glossary?: string;
  samplingOverrides?: {
    temperature?: number | null;
    topP?: number | null;
    topK?: number | null;
    maxOutputTokens?: number | null;
  };
  providerHints?: Record<string, unknown>;
}

export interface BuildCanonicalTranslationRequestResult {
  request: CanonicalLlmRequest;
  profile: LlmModelProfile | null;
  warnings: string[];
}

function buildUserMessage(text: string): CanonicalMessage {
  return {
    role: 'user',
    parts: [{ type: 'text', text }],
  };
}

function buildStructuredOutput(jsonResponse: boolean | undefined): CanonicalStructuredOutput | undefined {
  return jsonResponse ? { mode: 'json_object' } : { mode: 'text' };
}

function toFiniteNumber(value: unknown): number | undefined {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

export function buildCanonicalTranslationRequest(input: BuildCanonicalTranslationRequestInput): BuildCanonicalTranslationRequestResult {
  const providerTranslationProfile = resolveProviderTranslationProfile({
    providerFamily: input.providerFamily,
    runtimeFamily: input.adapterKey,
    modelName: input.model,
  });
  const samplingOverrides = input.samplingOverrides || {};
  const overrideTemperature = toFiniteNumber(samplingOverrides.temperature);
  const overrideTopP = toFiniteNumber(samplingOverrides.topP);
  const overrideTopK = toFiniteNumber(samplingOverrides.topK);
  const overrideMaxOutputTokens = toFiniteNumber(samplingOverrides.maxOutputTokens);
  const baseRequest: CanonicalLlmRequest = {
    model: String(input.model || '').trim(),
    instructions: String(input.systemPrompt || '').trim() || undefined,
    messages: [buildUserMessage(input.text)],
    structuredOutput: buildStructuredOutput(input.jsonResponse),
    sampling: {
      temperature: overrideTemperature ?? providerTranslationProfile?.arcsubDefaults.temperature ?? 0.2,
      topP: overrideTopP ?? providerTranslationProfile?.arcsubDefaults.topP,
      topK: overrideTopK ?? providerTranslationProfile?.arcsubDefaults.topK,
      maxOutputTokens: input.isConnectionTest ? 32 : overrideMaxOutputTokens,
    },
    providerHints: input.providerHints,
    metadata: {
      targetLang: String(input.targetLang || '').trim(),
      sourceLang: String(input.sourceLang || '').trim(),
      promptTemplateId: String(input.promptTemplateId || '').trim(),
      hasGlossary: input.glossary && input.glossary.trim() ? '1' : '0',
      connectionTest: input.isConnectionTest ? '1' : '0',
      providerTranslationProfileId: String(providerTranslationProfile?.id || '').trim(),
    },
  };

  return normalizeCanonicalLlmRequest({
    request: baseRequest,
    adapterKey: input.adapterKey,
    modelName: input.model,
  });
}
