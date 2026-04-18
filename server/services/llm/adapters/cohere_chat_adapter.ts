import { getCanonicalProviderCapabilities } from '../canonical/llm_capabilities.js';
import type { CanonicalLlmResponse } from '../canonical/llm_types.js';
import { buildOpenAiChatMessages, wantsJsonObject, wantsJsonSchema } from '../mapping/provider_payloads.js';
import type { LlmAdapter, ProviderHttpResponse } from './base.js';

function parseCohereText(data: any) {
  const parts = data?.message?.content;
  if (!Array.isArray(parts)) return '';
  return parts
    .map((part: any) => (part?.type === 'text' && typeof part?.text === 'string' ? part.text : ''))
    .join('');
}

export const cohereChatAdapter: LlmAdapter = {
  key: 'cohere-chat',
  capabilities: getCanonicalProviderCapabilities('cohere-chat'),
  buildRequest(input, context) {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (context.apiKey) headers.Authorization = `Bearer ${context.apiKey}`;

    const body: Record<string, any> = {
      model: context.modelOverride || input.model,
      messages: buildOpenAiChatMessages(input),
      temperature: input.sampling?.temperature ?? 0.2,
    };
    if (input.sampling?.maxOutputTokens != null) {
      body.max_tokens = input.sampling.maxOutputTokens;
    }
    if (input.sampling?.topP != null) {
      body.p = input.sampling.topP;
    }
    if (input.sampling?.topK != null) {
      body.k = input.sampling.topK;
    }
    if (input.sampling?.seed != null) {
      body.seed = input.sampling.seed;
    }
    if (input.sampling?.frequencyPenalty != null) {
      body.frequency_penalty = input.sampling.frequencyPenalty;
    }
    if (input.sampling?.presencePenalty != null) {
      body.presence_penalty = input.sampling.presencePenalty;
    }
    if (Array.isArray(input.sampling?.stop) && input.sampling?.stop.length > 0) {
      body.stop_sequences = input.sampling.stop;
    }
    if (wantsJsonObject(input)) {
      body.response_format = { type: 'json_object' };
    }
    const jsonSchema = wantsJsonSchema(input);
    if (jsonSchema) {
      body.response_format = {
        type: 'json_schema',
        json_schema: {
          schema: jsonSchema.schema,
          name: jsonSchema.name || 'structured_output',
        },
      };
    }
    if (input.reasoning && input.reasoning.mode !== 'off') {
      body.thinking = {
        type: 'enabled',
        ...(input.reasoning.budgetTokens != null ? { budget_tokens: input.reasoning.budgetTokens } : {}),
      };
    }

    return {
      method: 'POST' as const,
      url: context.endpointUrl,
      headers,
      body: JSON.stringify(body),
      timeoutMs: 120000,
    };
  },
  parseResponse(response: ProviderHttpResponse): CanonicalLlmResponse {
    const data: any = response.body;
    return {
      providerFamily: 'cohere-chat',
      model: String(data?.model || ''),
      outputText: parseCohereText(data).trim() || undefined,
      finishReason: typeof data?.finish_reason === 'string' ? data.finish_reason : undefined,
      usage: data?.usage
        ? {
            inputTokens: Number.isFinite(Number(data.usage?.tokens?.input_tokens))
              ? Number(data.usage.tokens.input_tokens)
              : undefined,
            outputTokens: Number.isFinite(Number(data.usage?.tokens?.output_tokens))
              ? Number(data.usage.tokens.output_tokens)
              : undefined,
          }
        : undefined,
      responseRef: typeof data?.id === 'string' ? data.id : undefined,
      rawProviderMeta: {
        billedUnits: data?.usage?.billed_units,
      },
    };
  },
};
