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
  targetLang: string;
  systemPrompt?: string;
  jsonResponse?: boolean;
  isConnectionTest?: boolean;
  promptTemplateId?: string;
  glossary?: string;
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

export function buildCanonicalTranslationRequest(input: BuildCanonicalTranslationRequestInput): BuildCanonicalTranslationRequestResult {
  const providerTranslationProfile = resolveProviderTranslationProfile({
    providerFamily: input.providerFamily,
    runtimeFamily: input.adapterKey,
    modelName: input.model,
  });
  const baseRequest: CanonicalLlmRequest = {
    model: String(input.model || '').trim(),
    instructions: String(input.systemPrompt || '').trim() || undefined,
    messages: [buildUserMessage(input.text)],
    structuredOutput: buildStructuredOutput(input.jsonResponse),
    sampling: {
      temperature: providerTranslationProfile?.arcsubDefaults.temperature ?? 0.2,
      topP: providerTranslationProfile?.arcsubDefaults.topP,
      topK: providerTranslationProfile?.arcsubDefaults.topK,
      maxOutputTokens: input.isConnectionTest ? 32 : undefined,
    },
    metadata: {
      targetLang: String(input.targetLang || '').trim(),
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
