import { getCanonicalProviderCapabilities } from '../canonical/llm_capabilities.js';
import type { CanonicalLlmRequest, CanonicalLlmResponse } from '../canonical/llm_types.js';
import { buildOpenAiChatMessages, getCanonicalUserText, wantsJsonObject, wantsJsonSchema } from '../mapping/provider_payloads.js';
import { buildTranslateGemmaMessages } from '../../local_llm/translategemma.js';
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

function isGitHubModelsEndpoint(endpointUrl: string) {
  try {
    return new URL(endpointUrl).hostname.toLowerCase() === 'models.github.ai';
  } catch {
    return false;
  }
}

function getEndpointHostname(endpointUrl: string) {
  try {
    return new URL(endpointUrl).hostname.toLowerCase().replace(/^\[|\]$/g, '');
  } catch {
    return '';
  }
}

function isNvidiaEndpoint(endpointUrl: string) {
  return getEndpointHostname(endpointUrl) === 'integrate.api.nvidia.com';
}

function isXAiEndpoint(endpointUrl: string) {
  const hostname = getEndpointHostname(endpointUrl);
  return hostname === 'api.x.ai' || hostname.endsWith('.x.ai');
}

function isPrivateOrLocalEndpoint(endpointUrl: string) {
  const hostname = getEndpointHostname(endpointUrl);
  return (
    hostname === 'localhost' ||
    hostname === '127.0.0.1' ||
    hostname === '::1' ||
    hostname.endsWith('.local') ||
    /^10\./.test(hostname) ||
    /^192\.168\./.test(hostname) ||
    /^172\.(1[6-9]|2\d|3[0-1])\./.test(hostname)
  );
}

function isQwenThinkingFamily(model: string) {
  const normalized = String(model || '').toLowerCase();
  return (
    /\bqwen[-_/. ]?3(?:\.\d+)?\b/.test(normalized) ||
    normalized.includes('qwen3') ||
    normalized.includes('qwen-3')
  );
}

function isGemmaThinkingFamily(model: string) {
  const normalized = String(model || '').toLowerCase();
  return (
    /\bgemma[-_/. ]?4\b/.test(normalized) ||
    normalized.includes('gemma4') ||
    normalized.includes('gemma-4')
  );
}

function isTranslateGemmaVllmModel(model: string) {
  const normalized = String(model || '').toLowerCase();
  return normalized.includes('translategemma') && normalized.includes('vllm');
}

function shouldDisableThinkingByDefault(input: CanonicalLlmRequest, context: LlmAdapterContext) {
  const model = String(context.modelOverride || input.model || '');
  if (!isQwenThinkingFamily(model) && !isGemmaThinkingFamily(model)) return false;
  const hints = `${model} ${context.endpointUrl} ${input.metadata?.providerTranslationProfileId || ''}`.toLowerCase();
  return isPrivateOrLocalEndpoint(context.endpointUrl) || hints.includes('vllm');
}

function shouldDisableReasoningEffortByDefault(input: CanonicalLlmRequest, context: LlmAdapterContext) {
  if (input.reasoning && input.reasoning.mode !== 'off') return false;
  const model = String(context.modelOverride || input.model || '').toLowerCase();
  if (isNvidiaEndpoint(context.endpointUrl) && model.includes('deepseek-v4-pro')) return true;
  if (isXAiEndpoint(context.endpointUrl) && (model.includes('grok-4') || model.includes('reasoning'))) return true;
  return model.includes('grok-4-fast-reasoning');
}

function buildDefaultBody(input: CanonicalLlmRequest, context: LlmAdapterContext) {
  let defaults: Record<string, unknown> = {};
  if (shouldDisableThinkingByDefault(input, context)) {
    defaults = deepMerge(defaults, { chat_template_kwargs: { enable_thinking: false } });
  }
  if (shouldDisableReasoningEffortByDefault(input, context)) {
    defaults = deepMerge(defaults, { reasoning_effort: 'none' });
  }
  return Object.keys(defaults).length > 0 ? defaults : null;
}

function stripReasoningBlocks(raw: string) {
  let text = String(raw || '');
  const lower = text.toLowerCase();
  const firstOpen = lower.indexOf('<think>');
  const lastClose = lower.lastIndexOf('</think>');
  if (firstOpen >= 0 && lastClose >= firstOpen) {
    text = `${text.slice(0, firstOpen)}${text.slice(lastClose + '</think>'.length)}`;
  }
  return text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
}

function hasHeader(headers: Record<string, string>, target: string) {
  const targetLower = target.toLowerCase();
  return Object.keys(headers).some((key) => key.toLowerCase() === targetLower);
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
  const message = data?.choices?.[0]?.message;
  const reasoningText = parseOpenAiLikeContent(message?.reasoning ?? message?.reasoning_content);
  const outputText = stripReasoningBlocks(parseOpenAiLikeContent(message?.content)).trim();
  return {
    providerFamily,
    model: String(data?.model || ''),
    outputText: outputText || undefined,
    finishReason: typeof data?.choices?.[0]?.finish_reason === 'string' ? data.choices[0].finish_reason : undefined,
    usage: data?.usage
      ? {
          inputTokens: Number.isFinite(Number(data.usage?.prompt_tokens)) ? Number(data.usage.prompt_tokens) : undefined,
          outputTokens: Number.isFinite(Number(data.usage?.completion_tokens)) ? Number(data.usage.completion_tokens) : undefined,
          reasoningTokens: Number.isFinite(Number(data.usage?.completion_tokens_details?.reasoning_tokens))
            ? Number(data.usage.completion_tokens_details.reasoning_tokens)
            : undefined,
        }
      : undefined,
    reasoningSummary: reasoningText.trim() || undefined,
    responseRef: typeof data?.id === 'string' ? data.id : undefined,
    rawProviderMeta: {
      object: data?.object,
      hasReasoning: Boolean(reasoningText.trim()),
    },
  };
}

function buildOpenAiChatRequest(input: CanonicalLlmRequest, context: LlmAdapterContext) {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (context.apiKey) headers.Authorization = `Bearer ${context.apiKey}`;
  const model = context.modelOverride || input.model;
  const translateGemmaMessages = isTranslateGemmaVllmModel(model)
    ? buildTranslateGemmaMessages({
        text: getCanonicalUserText(input),
        sourceLang: input.metadata?.sourceLang,
        targetLang: input.metadata?.targetLang || 'en',
        promptStyle: 'translategemma_vllm',
      })
    : null;

  const body: Record<string, any> = {
    model,
    messages: translateGemmaMessages || buildOpenAiChatMessages(input),
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
  const defaultBody = buildDefaultBody(input, context);
  const bodyWithDefaults = defaultBody ? deepMerge(body, defaultBody) : body;
  const finalBody = requestBody ? deepMerge(bodyWithDefaults, requestBody) : bodyWithDefaults;
  if (isGitHubModelsEndpoint(context.endpointUrl)) {
    if (!hasHeader(headers, 'X-GitHub-Api-Version')) {
      headers['X-GitHub-Api-Version'] = '2026-03-10';
    }
    if (!hasHeader(headers, 'Accept') && finalBody.stream !== true) {
      headers.Accept = 'application/vnd.github+json';
    }
  }

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
