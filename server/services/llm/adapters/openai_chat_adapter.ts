import { getCanonicalProviderCapabilities } from '../canonical/llm_capabilities.js';
import type { CanonicalLlmRequest, CanonicalLlmResponse } from '../canonical/llm_types.js';
import { buildOpenAiChatMessages, wantsJsonObject, wantsJsonSchema } from '../mapping/provider_payloads.js';
import type { LlmAdapter, LlmAdapterContext, ProviderHttpResponse } from './base.js';

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function deepMerge(base: Record<string, unknown>, override: Record<string, unknown>): Record<string, unknown> {
  const merged: Record<string, unknown> = { ...base };
  for (const [key, overrideValue] of Object.entries(override)) {
    const baseValue = merged[key];
    if (isPlainObject(baseValue) && isPlainObject(overrideValue)) {
      merged[key] = deepMerge(baseValue, overrideValue);
      continue;
    }
    merged[key] = overrideValue;
  }
  return merged;
}

function parseOpenAiLikeContent(content: any) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .map((part: any) => {
      if (typeof part === 'string') return part;
      if (typeof part?.text === 'string') return part.text;
      return '';
    })
    .join('');
}

function parseOpenAiChatResponse(response: ProviderHttpResponse, providerFamily: string): CanonicalLlmResponse {
  const data: any = response.body;
  return {
    providerFamily,
    model: String(data?.model || ''),
    outputText: parseOpenAiLikeContent(data?.choices?.[0]?.message?.content).trim() || undefined,
    finishReason: typeof data?.choices?.[0]?.finish_reason === 'string' ? data.choices[0].finish_reason : undefined,
    usage: data?.usage
      ? {
          inputTokens: Number.isFinite(Number(data.usage?.prompt_tokens)) ? Number(data.usage.prompt_tokens) : undefined,
          outputTokens: Number.isFinite(Number(data.usage?.completion_tokens)) ? Number(data.usage.completion_tokens) : undefined,
        }
      : undefined,
    responseRef: typeof data?.id === 'string' ? data.id : undefined,
    rawProviderMeta: {
      object: data?.object,
    },
  };
}

function buildOpenAiChatRequest(input: CanonicalLlmRequest, context: LlmAdapterContext) {
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
    body.top_p = input.sampling.topP;
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
    body.stop = input.sampling.stop;
  }
  if (wantsJsonObject(input)) {
    body.response_format = { type: 'json_object' };
  }
  const jsonSchema = wantsJsonSchema(input);
  if (jsonSchema) {
    body.response_format = {
      type: 'json_schema',
      json_schema: {
        name: jsonSchema.name || 'structured_output',
        schema: jsonSchema.schema,
        strict: jsonSchema.strict !== false,
      },
    };
  }

  const requestHeaders = isPlainObject(input.providerHints?.requestHeaders)
    ? (input.providerHints?.requestHeaders as Record<string, unknown>)
    : null;
  if (requestHeaders) {
    for (const [key, value] of Object.entries(requestHeaders)) {
      if (typeof value === 'string' && key.trim()) {
        headers[key] = value;
      }
    }
  }

  const requestBody = isPlainObject(input.providerHints?.requestBody)
    ? (input.providerHints?.requestBody as Record<string, unknown>)
    : null;
  const finalBody = requestBody ? deepMerge(body, requestBody) : body;

  return {
    method: 'POST' as const,
    url: context.endpointUrl,
    headers,
    body: JSON.stringify(finalBody),
    timeoutMs: 120000,
  };
}

export const openAiChatAdapter: LlmAdapter = {
  key: 'openai-chat',
  capabilities: getCanonicalProviderCapabilities('openai-chat'),
  buildRequest(input, context) {
    return buildOpenAiChatRequest(input, context);
  },
  parseResponse(response) {
    return parseOpenAiChatResponse(response, 'openai-chat');
  },
};

export const openAiCompatibleChatAdapter: LlmAdapter = {
  key: 'openai-compatible-chat',
  capabilities: getCanonicalProviderCapabilities('openai-compatible-chat'),
  buildRequest(input, context) {
    return buildOpenAiChatRequest(input, context);
  },
  parseResponse(response) {
    return parseOpenAiChatResponse(response, 'openai-compatible-chat');
  },
};
