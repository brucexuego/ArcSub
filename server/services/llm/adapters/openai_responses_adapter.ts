import { getCanonicalProviderCapabilities } from '../canonical/llm_capabilities.js';
import type { CanonicalLlmResponse } from '../canonical/llm_types.js';
import { getCanonicalInstructions, getCanonicalUserText, wantsJsonObject, wantsJsonSchema } from '../mapping/provider_payloads.js';
import type { LlmAdapter, LlmAdapterContext, ProviderHttpResponse } from './base.js';

function parseResponsesContent(data: any) {
  if (typeof data?.output_text === 'string') return data.output_text;
  if (Array.isArray(data?.output)) {
    return data.output
      .flatMap((item: any) => (Array.isArray(item?.content) ? item.content : []))
      .map((part: any) => (part?.type?.includes('text') && typeof part?.text === 'string' ? part.text : ''))
      .join('');
  }
  return '';
}

export const openAiResponsesAdapter: LlmAdapter = {
  key: 'openai-responses',
  capabilities: getCanonicalProviderCapabilities('openai-responses'),
  buildRequest(input, context) {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (context.apiKey) headers.Authorization = `Bearer ${context.apiKey}`;

    const body: Record<string, any> = {
      model: context.modelOverride || input.model,
      input: getCanonicalUserText(input),
      temperature: input.sampling?.temperature ?? 0.2,
    };
    if (input.sampling?.maxOutputTokens != null) {
      body.max_output_tokens = input.sampling.maxOutputTokens;
    }
    if (input.sampling?.topP != null) {
      body.top_p = input.sampling.topP;
    }
    if (input.sampling?.seed != null) {
      body.seed = input.sampling.seed;
    }
    const instructions = getCanonicalInstructions(input);
    if (instructions) {
      body.instructions = instructions;
    }
    if (input.conversationRef) {
      body.previous_response_id = input.conversationRef;
    }
    if (wantsJsonObject(input)) {
      body.text = { format: { type: 'json_object' } };
    }
    const jsonSchema = wantsJsonSchema(input);
    if (jsonSchema) {
      body.text = {
        format: {
          type: 'json_schema',
          name: jsonSchema.name || 'structured_output',
          schema: jsonSchema.schema,
          strict: jsonSchema.strict !== false,
        },
      };
    }
    if (input.reasoning && input.reasoning.mode !== 'off') {
      body.reasoning = {
        effort: input.reasoning.effort || 'medium',
        ...(input.reasoning.budgetTokens != null ? { max_output_tokens: input.reasoning.budgetTokens } : {}),
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
      providerFamily: 'openai-responses',
      model: String(data?.model || ''),
      outputText: parseResponsesContent(data).trim() || undefined,
      finishReason: undefined,
      usage: data?.usage
        ? {
            inputTokens: Number.isFinite(Number(data.usage?.input_tokens)) ? Number(data.usage.input_tokens) : undefined,
            outputTokens: Number.isFinite(Number(data.usage?.output_tokens)) ? Number(data.usage.output_tokens) : undefined,
            reasoningTokens: Number.isFinite(Number(data.usage?.reasoning_tokens)) ? Number(data.usage.reasoning_tokens) : undefined,
          }
        : undefined,
      responseRef: typeof data?.id === 'string' ? data.id : undefined,
      rawProviderMeta: {
        object: data?.object,
      },
    };
  },
};
