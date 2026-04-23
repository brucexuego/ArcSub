import type { CloudTranslateProvider } from './cloud_translate_provider.js';
import { ensureBuiltinLlmAdaptersRegistered } from './llm/adapters/builtin.js';
import { llmAdapterRegistry } from './llm/adapters/registry.js';
import type { LlmAdapterKey } from './llm/canonical/llm_capabilities.js';
import { buildCanonicalTranslationRequest } from './llm/mapping/translation_canonical_request.js';
import { deepLCloudTranslateAdapter } from './cloud_translate_deepl_adapter.js';
import type { ApiModelRequestOptions } from '../../src/types.js';
import {
  ollamaChatCloudTranslateAdapter,
  ollamaGenerateCloudTranslateAdapter,
} from './cloud_translate_ollama_adapters.js';

export interface CloudTranslateAdapterRequestOptions {
  text: string;
  targetLang: string;
  sourceLang?: string;
  glossary?: string;
  prompt?: string;
  promptTemplateId?: string;
  key?: string;
  model?: string;
  modelOptions?: ApiModelRequestOptions;
  isConnectionTest?: boolean;
  lineSafeMode?: boolean;
  systemPromptOverride?: string;
  disableSystemPrompt?: boolean;
  jsonResponse?: boolean;
  signal?: AbortSignal;
}

export interface CloudTranslateProviderResult {
  text: string;
  meta: {
    endpointUrl: string;
    fallbackUsed: boolean;
    fallbackType: string | null;
    requestWarnings?: string[];
  };
}

export interface CloudTranslateProviderError extends Error {
  status: number;
  detail: string;
  retryAfterMs: number | null;
}

export interface CloudTranslateAdapterDeps {
  throwIfAborted(signal?: AbortSignal): void;
  fetchWithTimeout(url: string, init: RequestInit, timeoutMs?: number, signal?: AbortSignal): Promise<Response>;
  parseRetryAfterMs(response: Response): number | null;
  extractErrorMessage(rawText: string, fallback: string): string;
  resolveSystemPrompt(options: CloudTranslateAdapterRequestOptions): string;
  normalizeDeepLTargetLanguage(targetLang: string): string;
  parseOpenAiLikeContent(content: unknown): string;
  parseGeminiContent(data: unknown): string;
  parseAnthropicContent(data: unknown): string;
  parseOllamaContent(data: unknown): string;
  parseResponsesContent(data: unknown): string;
  hasAnthropicEnvelope(data: unknown): boolean;
  hasGeminiEnvelope(data: unknown): boolean;
  hasOllamaEnvelope(data: unknown): boolean;
  hasOpenAiChatEnvelope(data: unknown): boolean;
  hasResponsesEnvelope(data: unknown): boolean;
  shouldFallbackToResponses(error: unknown): boolean;
  getOpenAiResponsesEndpoint(chatEndpoint: string): string;
  getOllamaFallbackEndpoint(currentEndpoint: string, fallbackTarget: 'chat' | 'generate'): string;
  makeProviderHttpError(
    prefix: string,
    status: number,
    detail: string,
    retryAfterMs?: number | null
  ): CloudTranslateProviderError;
  isProviderHttpError(error: unknown): error is CloudTranslateProviderError;
}

export interface CloudTranslateAdapter {
  provider: CloudTranslateProvider;
  request(
    endpointUrl: string,
    options: CloudTranslateAdapterRequestOptions,
    deps: CloudTranslateAdapterDeps,
    onProgress?: (message: string) => void
  ): Promise<CloudTranslateProviderResult>;
}

ensureBuiltinLlmAdaptersRegistered();

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function toFiniteNumber(value: unknown): number | undefined {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function parseJsonSafe(raw: string | undefined) {
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

type ParsedOpenAiChatEventStreamResult =
  | { kind: 'completion'; body: Record<string, unknown> }
  | { kind: 'error'; message: string; status?: number }
  | null;

function extractSseErrorFrame(parsed: any): { message: string; status?: number } | null {
  if (!parsed || typeof parsed !== 'object') return null;

  const topLevelError = parsed.error;
  if (typeof topLevelError === 'string' && topLevelError.trim()) {
    return { message: topLevelError.trim() };
  }

  if (isPlainObject(topLevelError)) {
    const messageCandidate =
      topLevelError.message ||
      topLevelError.detail ||
      topLevelError.error_description ||
      topLevelError.error ||
      parsed.message;
    const statusCandidate =
      topLevelError.status ?? topLevelError.http_status ?? topLevelError.status_code ?? topLevelError.code;
    const parsedStatus = Number(statusCandidate);
    const status =
      Number.isFinite(parsedStatus) && parsedStatus >= 100 && parsedStatus <= 599 ? Math.round(parsedStatus) : undefined;
    if (typeof messageCandidate === 'string' && messageCandidate.trim()) {
      return {
        message: messageCandidate.trim(),
        status,
      };
    }
  }

  const objectType = String(parsed.object || parsed.type || '').toLowerCase();
  if (
    objectType.includes('error') &&
    typeof parsed.message === 'string' &&
    parsed.message.trim()
  ) {
    const parsedStatus = Number(parsed.status ?? parsed.code ?? 0);
    const status =
      Number.isFinite(parsedStatus) && parsedStatus >= 100 && parsedStatus <= 599 ? Math.round(parsedStatus) : undefined;
    return {
      message: parsed.message.trim(),
      status,
    };
  }

  return null;
}

function parseOpenAiChatEventStream(rawText: string): ParsedOpenAiChatEventStreamResult {
  const chunks = String(rawText || '').split(/\r?\n/);
  let id = '';
  let model = '';
  let created = 0;
  let finishReason: string | undefined;
  let messageContent = '';
  let reasoningContent = '';
  let usage: Record<string, unknown> | undefined;
  let sawChunk = false;

  for (const line of chunks) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;

    const payload = trimmed.slice('data:'.length).trim();
    if (!payload) continue;
    if (payload === '[DONE]') break;

    const parsed = parseJsonSafe(payload);
    if (!parsed || typeof parsed !== 'object') continue;

    const sseError = extractSseErrorFrame(parsed);
    if (sseError) {
      return {
        kind: 'error',
        message: sseError.message,
        status: sseError.status,
      };
    }

    if (!id && typeof (parsed as any).id === 'string') id = (parsed as any).id;
    if (!model && typeof (parsed as any).model === 'string') model = (parsed as any).model;
    if (!created) {
      const parsedCreated = Number((parsed as any).created);
      if (Number.isFinite(parsedCreated) && parsedCreated > 0) created = parsedCreated;
    }
    if (isPlainObject((parsed as any).usage)) {
      usage = (parsed as any).usage;
    }

    const choices = Array.isArray((parsed as any).choices) ? (parsed as any).choices : [];
    let frameHasChoicePayload = false;
    for (const choice of choices) {
      const delta = (choice as any)?.delta;
      if (delta && typeof delta === 'object') {
        frameHasChoicePayload = true;
      }
      if (typeof delta?.content === 'string') {
        messageContent += delta.content;
      } else if (Array.isArray(delta?.content)) {
        messageContent += delta.content
          .map((part: any) => (typeof part?.text === 'string' ? part.text : typeof part === 'string' ? part : ''))
          .join('');
      }
      if (typeof delta?.reasoning_content === 'string') {
        reasoningContent += delta.reasoning_content;
      }
      if (!finishReason && typeof (choice as any)?.finish_reason === 'string' && (choice as any).finish_reason) {
        finishReason = (choice as any).finish_reason;
      }
      if (typeof (choice as any)?.finish_reason === 'string' && (choice as any).finish_reason) {
        frameHasChoicePayload = true;
      }
    }

    if (frameHasChoicePayload) {
      sawChunk = true;
    }
  }

  if (!sawChunk) return null;

  return {
    kind: 'completion',
    body: {
      id: id || undefined,
      object: 'chat.completion',
      created: created || Math.floor(Date.now() / 1000),
      model: model || '',
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: messageContent,
            reasoning_content: reasoningContent || undefined,
          },
          finish_reason: finishReason || 'stop',
        },
      ],
      usage,
    },
  };
}

async function executeLlmAdapterRequest(input: {
  adapterKey: LlmAdapterKey;
  providerFamily: CloudTranslateProvider;
  endpointUrl: string;
  options: CloudTranslateAdapterRequestOptions;
  deps: CloudTranslateAdapterDeps;
  errorPrefix: string;
}) {
  const adapter = llmAdapterRegistry.require(input.adapterKey);
  const canonical = buildCanonicalTranslationRequest({
    adapterKey: input.adapterKey,
    providerFamily: input.providerFamily,
    model: input.options.model,
    text: input.options.text,
    sourceLang: input.options.sourceLang,
    targetLang: input.options.targetLang,
    systemPrompt: input.deps.resolveSystemPrompt(input.options),
    jsonResponse: input.options.jsonResponse,
    isConnectionTest: input.options.isConnectionTest,
    promptTemplateId: input.options.promptTemplateId,
    glossary: input.options.glossary,
    samplingOverrides: {
      temperature: input.options.modelOptions?.sampling?.temperature,
      topP: input.options.modelOptions?.sampling?.topP,
      maxOutputTokens: input.options.modelOptions?.sampling?.maxOutputTokens,
    },
    providerHints: {
      requestHeaders: input.options.modelOptions?.headers,
      requestBody: input.options.modelOptions?.body,
    },
  });
  const canonicalRequest = canonical.request;

  const providerRequest = adapter.buildRequest(canonicalRequest, {
    endpointUrl: input.endpointUrl,
    apiKey: input.options.key,
    modelOverride: canonicalRequest.model,
  });

  const requestBody =
    typeof providerRequest.body === 'string'
      ? providerRequest.body
      : providerRequest.body == null
        ? undefined
        : JSON.stringify(providerRequest.body);

  const parsedRequestBody =
    typeof requestBody === 'string' ? parseJsonSafe(requestBody) : null;
  const isStreamingRequest = Boolean(
    parsedRequestBody &&
      typeof parsedRequestBody === 'object' &&
      !Array.isArray(parsedRequestBody) &&
      (parsedRequestBody as any).stream === true
  );
  const requestHeaders: Record<string, string> = { ...(providerRequest.headers || {}) };
  const hasAcceptHeader = Object.keys(requestHeaders).some((key) => key.toLowerCase() === 'accept');
  if (isStreamingRequest && !hasAcceptHeader) {
    requestHeaders.Accept = 'text/event-stream';
  }
  const timeoutOverride = toFiniteNumber(input.options.modelOptions?.timeoutMs);
  const effectiveTimeoutMs =
    timeoutOverride && timeoutOverride > 0 ? timeoutOverride : providerRequest.timeoutMs ?? 120000;

  const response = await input.deps.fetchWithTimeout(
    providerRequest.url,
    {
      method: providerRequest.method,
      headers: requestHeaders,
      body: requestBody,
    },
    effectiveTimeoutMs,
    input.options.signal
  );
  const rawText = await response.text();
  if (!response.ok) {
    throw input.deps.makeProviderHttpError(
      input.errorPrefix,
      response.status,
      input.deps.extractErrorMessage(rawText, response.statusText),
      input.deps.parseRetryAfterMs(response)
    );
  }

  let body: unknown = null;
  const contentType = String(response.headers.get('content-type') || '').toLowerCase();
  const streamResult =
    isStreamingRequest || contentType.includes('text/event-stream')
      ? parseOpenAiChatEventStream(rawText)
    : null;
  if (streamResult?.kind === 'error') {
    throw input.deps.makeProviderHttpError(
      input.errorPrefix,
      streamResult.status ?? 502,
      streamResult.message,
      input.deps.parseRetryAfterMs(response)
    );
  }
  if (streamResult?.kind === 'completion') {
    body = streamResult.body;
  } else {
    try {
      body = JSON.parse(rawText || '{}');
    } catch {
      body = rawText;
    }
  }

  const parsed = adapter.parseResponse(
    {
      status: response.status,
      headers: Object.fromEntries(response.headers.entries()),
      body,
    },
    {
      endpointUrl: input.endpointUrl,
      apiKey: input.options.key,
      modelOverride: canonicalRequest.model,
    }
  );

  return {
    parsed,
    body,
    warnings: canonical.warnings,
  };
}

async function requestAnthropic(
  endpointUrl: string,
  options: CloudTranslateAdapterRequestOptions,
  deps: CloudTranslateAdapterDeps
) {
  const { parsed, body, warnings } = await executeLlmAdapterRequest({
    adapterKey: 'anthropic-messages',
    providerFamily: 'anthropic',
    endpointUrl,
    options: {
      ...options,
      model: String(options.model || '').trim() || 'claude-3-5-sonnet-latest',
    },
    deps,
    errorPrefix: 'Anthropic API error',
  });
  const translated = String(parsed.outputText || '').trim();
  if (!translated) {
    if (options.isConnectionTest && deps.hasAnthropicEnvelope(body)) {
      return { text: '__connection_ok__', warnings };
    }
    throw new Error('Anthropic API returned empty content.');
  }
  return { text: translated, warnings };
}

async function requestGeminiNative(
  endpointUrl: string,
  options: CloudTranslateAdapterRequestOptions,
  deps: CloudTranslateAdapterDeps
) {
  const send = async (jsonResponse: boolean) =>
    executeLlmAdapterRequest({
      adapterKey: 'gemini-native',
      providerFamily: 'gemini-native',
      endpointUrl,
      options: {
        ...options,
        model: String(options.model || '').trim() || 'gemini-2.5-flash',
        jsonResponse,
      },
      deps,
      errorPrefix: 'Gemini API error',
    });

  let result;
  try {
    result = await send(Boolean(options.jsonResponse));
  } catch (error) {
    if (!options.jsonResponse || !deps.isProviderHttpError(error)) throw error;
    const detail = String(error.detail || '').toLowerCase();
    if (![400, 404, 422].includes(error.status) || !/responsemime|json|schema|unsupported|invalid/.test(detail)) {
      throw error;
    }
    result = await send(false);
  }

  const translated = String(result.parsed.outputText || '').trim();
  if (!translated) {
    if (options.isConnectionTest && deps.hasGeminiEnvelope(result.body)) {
      return { text: '__connection_ok__', warnings: result.warnings };
    }
    throw new Error('Gemini API returned empty content.');
  }
  return { text: translated, warnings: result.warnings };
}

async function requestOpenAiChat(
  endpointUrl: string,
  options: CloudTranslateAdapterRequestOptions,
  deps: CloudTranslateAdapterDeps
) {
  const send = async (jsonResponse: boolean) =>
    executeLlmAdapterRequest({
      adapterKey: 'openai-compatible-chat',
      providerFamily: 'openai-compatible',
      endpointUrl,
      options: {
        ...options,
        model: String(options.model || '').trim() || 'gpt-4o-mini',
        jsonResponse,
      },
      deps,
      errorPrefix: 'Translation API error',
    });

  let result;
  try {
    result = await send(Boolean(options.jsonResponse));
  } catch (error) {
    if (!options.jsonResponse || !deps.isProviderHttpError(error)) throw error;
    const detail = String(error.detail || '').toLowerCase();
    if (![400, 404, 422].includes(error.status) || !/response_format|json|schema|unsupported|invalid/.test(detail)) {
      throw error;
    }
    result = await send(false);
  }

  const translated = String(result.parsed.outputText || '').trim();
  if (!translated) {
    if (options.isConnectionTest && deps.hasOpenAiChatEnvelope(result.body)) {
      return { text: '__connection_ok__', warnings: result.warnings };
    }
    throw new Error('Translation API returned empty content.');
  }
  return { text: translated, warnings: result.warnings };
}

async function requestOpenAiResponses(
  endpointUrl: string,
  options: CloudTranslateAdapterRequestOptions,
  deps: CloudTranslateAdapterDeps
) {
  const { parsed, body, warnings } = await executeLlmAdapterRequest({
    adapterKey: 'openai-responses',
    providerFamily: 'openai-compatible',
    endpointUrl,
    options: {
      ...options,
      model: String(options.model || '').trim() || 'gpt-4o-mini',
    },
    deps,
    errorPrefix: 'Responses API error',
  });
  const translated = String(parsed.outputText || '').trim();
  if (!translated) {
    if (options.isConnectionTest && deps.hasResponsesEnvelope(body)) {
      return { text: '__connection_ok__', warnings };
    }
    throw new Error('Responses API returned empty content.');
  }
  return { text: translated, warnings };
}

async function requestMistralChat(
  endpointUrl: string,
  options: CloudTranslateAdapterRequestOptions,
  deps: CloudTranslateAdapterDeps
) {
  const send = async (jsonResponse: boolean) =>
    executeLlmAdapterRequest({
      adapterKey: 'mistral-chat',
      providerFamily: 'mistral-chat',
      endpointUrl,
      options: {
        ...options,
        model: String(options.model || '').trim() || 'mistral-small-latest',
        jsonResponse,
      },
      deps,
      errorPrefix: 'Mistral API error',
    });

  let result;
  try {
    result = await send(Boolean(options.jsonResponse));
  } catch (error) {
    if (!options.jsonResponse || !deps.isProviderHttpError(error)) throw error;
    const detail = String(error.detail || '').toLowerCase();
    if (![400, 404, 422].includes(error.status) || !/response_format|json|schema|unsupported|invalid/.test(detail)) {
      throw error;
    }
    result = await send(false);
  }

  const translated = String(result.parsed.outputText || '').trim();
  if (!translated) {
    if (options.isConnectionTest && deps.hasOpenAiChatEnvelope(result.body)) {
      return { text: '__connection_ok__', warnings: result.warnings };
    }
    throw new Error('Mistral API returned empty content.');
  }
  return { text: translated, warnings: result.warnings };
}

async function requestCohereChat(
  endpointUrl: string,
  options: CloudTranslateAdapterRequestOptions,
  deps: CloudTranslateAdapterDeps
) {
  const send = async (jsonResponse: boolean) =>
    executeLlmAdapterRequest({
      adapterKey: 'cohere-chat',
      providerFamily: 'cohere-chat',
      endpointUrl,
      options: {
        ...options,
        model: String(options.model || '').trim() || 'command-a-03-2025',
        jsonResponse,
      },
      deps,
      errorPrefix: 'Cohere API error',
    });

  let result;
  try {
    result = await send(Boolean(options.jsonResponse));
  } catch (error) {
    if (!options.jsonResponse || !deps.isProviderHttpError(error)) throw error;
    const detail = String(error.detail || '').toLowerCase();
    if (![400, 404, 422].includes(error.status) || !/response_format|json|schema|unsupported|invalid/.test(detail)) {
      throw error;
    }
    result = await send(false);
  }

  const translated = String(result.parsed.outputText || '').trim();
  if (!translated) {
    if (options.isConnectionTest && typeof (result.body as any)?.id === 'string') {
      return { text: '__connection_ok__', warnings: result.warnings };
    }
    throw new Error('Cohere API returned empty content.');
  }
  return { text: translated, warnings: result.warnings };
}

async function requestXaiChat(
  endpointUrl: string,
  options: CloudTranslateAdapterRequestOptions,
  deps: CloudTranslateAdapterDeps
) {
  const send = async (jsonResponse: boolean) =>
    executeLlmAdapterRequest({
      adapterKey: 'xai-chat',
      providerFamily: 'xai-chat',
      endpointUrl,
      options: {
        ...options,
        model: String(options.model || '').trim() || 'grok-4-fast-reasoning',
        jsonResponse,
      },
      deps,
      errorPrefix: 'xAI API error',
    });

  let result;
  try {
    result = await send(Boolean(options.jsonResponse));
  } catch (error) {
    if (!options.jsonResponse || !deps.isProviderHttpError(error)) throw error;
    const detail = String(error.detail || '').toLowerCase();
    if (![400, 404, 422].includes(error.status) || !/response_format|json|schema|unsupported|invalid/.test(detail)) {
      throw error;
    }
    result = await send(false);
  }

  const translated = String(result.parsed.outputText || '').trim();
  if (!translated) {
    if (options.isConnectionTest && deps.hasOpenAiChatEnvelope(result.body)) {
      return { text: '__connection_ok__', warnings: result.warnings };
    }
    throw new Error('xAI API returned empty content.');
  }
  return { text: translated, warnings: result.warnings };
}

const cloudTranslateAdapters: Record<CloudTranslateProvider, CloudTranslateAdapter> = {
  deepl: deepLCloudTranslateAdapter,
  anthropic: {
    provider: 'anthropic',
    async request(endpointUrl, options, deps) {
      deps.throwIfAborted(options.signal);
      const result = await requestAnthropic(endpointUrl, options, deps);
      return { text: result.text, meta: { endpointUrl, fallbackUsed: false, fallbackType: null, requestWarnings: result.warnings } };
    },
  },
  'gemini-native': {
    provider: 'gemini-native',
    async request(endpointUrl, options, deps) {
      deps.throwIfAborted(options.signal);
      const result = await requestGeminiNative(endpointUrl, options, deps);
      return { text: result.text, meta: { endpointUrl, fallbackUsed: false, fallbackType: null, requestWarnings: result.warnings } };
    },
  },
  'mistral-chat': {
    provider: 'mistral-chat',
    async request(endpointUrl, options, deps) {
      deps.throwIfAborted(options.signal);
      const result = await requestMistralChat(endpointUrl, options, deps);
      return { text: result.text, meta: { endpointUrl, fallbackUsed: false, fallbackType: null, requestWarnings: result.warnings } };
    },
  },
  'cohere-chat': {
    provider: 'cohere-chat',
    async request(endpointUrl, options, deps) {
      deps.throwIfAborted(options.signal);
      const result = await requestCohereChat(endpointUrl, options, deps);
      return { text: result.text, meta: { endpointUrl, fallbackUsed: false, fallbackType: null, requestWarnings: result.warnings } };
    },
  },
  'xai-chat': {
    provider: 'xai-chat',
    async request(endpointUrl, options, deps) {
      deps.throwIfAborted(options.signal);
      const result = await requestXaiChat(endpointUrl, options, deps);
      return { text: result.text, meta: { endpointUrl, fallbackUsed: false, fallbackType: null, requestWarnings: result.warnings } };
    },
  },
  'ollama-chat': ollamaChatCloudTranslateAdapter,
  'ollama-generate': ollamaGenerateCloudTranslateAdapter,
  'openai-compatible': {
    provider: 'openai-compatible',
    async request(endpointUrl, options, deps, onProgress) {
      deps.throwIfAborted(options.signal);
      try {
        const result = await requestOpenAiChat(endpointUrl, options, deps);
        return { text: result.text, meta: { endpointUrl, fallbackUsed: false, fallbackType: null, requestWarnings: result.warnings } };
      } catch (error) {
        if (!deps.shouldFallbackToResponses(error)) throw error;
        const fallbackUrl = deps.getOpenAiResponsesEndpoint(endpointUrl);
        onProgress?.('Chat completions unavailable, retrying with Responses API fallback...');
        const result = await requestOpenAiResponses(fallbackUrl, options, deps);
        return {
          text: result.text,
          meta: {
            endpointUrl: fallbackUrl,
            fallbackUsed: true,
            fallbackType: 'openai_responses',
            requestWarnings: result.warnings,
          },
        };
      }
    },
  },
};

export function getCloudTranslateAdapter(provider: CloudTranslateProvider) {
  return cloudTranslateAdapters[provider];
}
