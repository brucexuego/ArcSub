import { getCanonicalProviderCapabilities } from '../canonical/llm_capabilities.js';
import type { CanonicalLlmResponse } from '../canonical/llm_types.js';
import {
  buildGeminiContents,
  buildGeminiSystemInstruction,
  getCanonicalInstructions,
  getCanonicalUserText,
  wantsJsonObject,
  wantsJsonSchema,
} from '../mapping/provider_payloads.js';
import type { LlmAdapter, ProviderHttpResponse } from './base.js';

function parseGeminiContent(data: any) {
  const parts = data?.candidates?.[0]?.content?.parts;
  if (!Array.isArray(parts)) return '';
  return parts
    .map((part: any) => (!part?.thought && typeof part?.text === 'string' ? part.text : ''))
    .join('');
}

function stripModelPrefix(model: string | undefined) {
  return String(model || '').replace(/^models\//i, '').trim();
}

function isGemma3Model(model: string | undefined) {
  return /^gemma-3[-_]/i.test(stripModelPrefix(model));
}

function buildGeminiSingleUserContents(input: Parameters<LlmAdapter['buildRequest']>[0]) {
  const instructions = getCanonicalInstructions(input);
  const userText = getCanonicalUserText(input);
  return [
    {
      role: 'user',
      parts: [{
        text: [
          instructions,
          instructions && userText ? 'Input:' : '',
          userText,
        ].filter(Boolean).join('\n\n'),
      }],
    },
  ];
}

export const geminiNativeAdapter: LlmAdapter = {
  key: 'gemini-native',
  capabilities: getCanonicalProviderCapabilities('gemini-native'),
  buildRequest(input, context) {
    const endpoint = new URL(context.endpointUrl);
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (context.apiKey && !endpoint.searchParams.get('key')) {
      headers['x-goog-api-key'] = context.apiKey;
    }

    const omitDeveloperInstruction = isGemma3Model(input.model);
    const body: Record<string, any> = {
      contents: omitDeveloperInstruction ? buildGeminiSingleUserContents(input) : buildGeminiContents(input),
      generationConfig: {
        temperature: input.sampling?.temperature ?? 0.2,
        maxOutputTokens: input.sampling?.maxOutputTokens,
        ...(input.sampling?.topP != null ? { topP: input.sampling.topP } : {}),
        ...(input.sampling?.topK != null ? { topK: input.sampling.topK } : {}),
        ...(input.sampling?.seed != null ? { seed: input.sampling.seed } : {}),
        ...(input.sampling?.frequencyPenalty != null ? { frequencyPenalty: input.sampling.frequencyPenalty } : {}),
        ...(input.sampling?.presencePenalty != null ? { presencePenalty: input.sampling.presencePenalty } : {}),
        ...(Array.isArray(input.sampling?.stop) && input.sampling?.stop.length > 0
          ? { stopSequences: input.sampling.stop }
          : {}),
        ...(wantsJsonObject(input) ? { responseMimeType: 'application/json' } : {}),
      },
    };
    const jsonSchema = wantsJsonSchema(input);
    if (jsonSchema) {
      body.generationConfig.responseMimeType = 'application/json';
      body.generationConfig.responseSchema = jsonSchema.schema;
    }

    const systemInstruction = omitDeveloperInstruction ? null : buildGeminiSystemInstruction(input);
    if (systemInstruction) {
      body.systemInstruction = systemInstruction;
    }

    return {
      method: 'POST' as const,
      url: endpoint.toString(),
      headers,
      body: JSON.stringify(body),
      timeoutMs: 120000,
    };
  },
  parseResponse(response: ProviderHttpResponse): CanonicalLlmResponse {
    const data: any = response.body;
    return {
      providerFamily: 'gemini-native',
      model: '',
      outputText: parseGeminiContent(data).trim() || undefined,
      finishReason: typeof data?.candidates?.[0]?.finishReason === 'string' ? data.candidates[0].finishReason : undefined,
      usage: data?.usageMetadata
        ? {
            inputTokens: Number.isFinite(Number(data.usageMetadata?.promptTokenCount))
              ? Number(data.usageMetadata.promptTokenCount)
              : undefined,
            outputTokens: Number.isFinite(Number(data.usageMetadata?.candidatesTokenCount))
              ? Number(data.usageMetadata.candidatesTokenCount)
              : undefined,
          }
        : undefined,
      rawProviderMeta: {
        promptFeedback: data?.promptFeedback,
      },
    };
  },
};
