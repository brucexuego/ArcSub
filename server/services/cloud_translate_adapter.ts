import type { CloudTranslateProvider } from './cloud_translate_provider.js';
import { GoogleGenAI } from '@google/genai';
import { ensureBuiltinLlmAdaptersRegistered } from './llm/adapters/builtin.js';
import { llmAdapterRegistry } from './llm/adapters/registry.js';
import type { LlmAdapterKey } from './llm/canonical/llm_capabilities.js';
import { buildCanonicalTranslationRequest } from './llm/mapping/translation_canonical_request.js';
import {
  buildGeminiContents,
  buildGeminiSystemInstruction,
  getCanonicalInstructions,
  getCanonicalUserText,
  wantsJsonObject,
  wantsJsonSchema,
} from './llm/mapping/provider_payloads.js';
import { extractCloudTranslateErrorMessage, extractErrorStatus } from './cloud_translate/errors.js';
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
    responseHeaders?: Record<string, string>;
  };
}

export interface CloudTranslateProviderError extends Error {
  status: number;
  detail: string;
  retryAfterMs: number | null;
}

export interface CloudTranslateAdapterDeps {
  throwIfAborted(signal?: AbortSignal): void;
  resolveRequestTimeoutMs(providerTimeoutMs?: number, overrideTimeoutMs?: number): number;
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

function deepMerge(base: Record<string, any>, override: Record<string, any>): Record<string, any> {
  const merged: Record<string, any> = { ...base };
  for (const [key, overrideValue] of Object.entries(override)) {
    const baseValue = merged[key];
    if (isPlainObject(baseValue) && isPlainObject(overrideValue)) {
      merged[key] = deepMerge(baseValue as Record<string, any>, overrideValue as Record<string, any>);
      continue;
    }
    merged[key] = overrideValue;
  }
  return merged;
}

function getProviderBodyOptions(options: CloudTranslateAdapterRequestOptions) {
  return isPlainObject(options.modelOptions?.body)
    ? (options.modelOptions?.body as Record<string, any>)
    : {};
}

function shouldUseGeminiSdk(endpointUrl: string) {
  try {
    return new URL(endpointUrl).hostname.toLowerCase() === 'generativelanguage.googleapis.com';
  } catch {
    return false;
  }
}

function getGeminiApiVersion(endpointUrl: string) {
  try {
    const parts = new URL(endpointUrl).pathname.split('/').filter(Boolean);
    const version = parts.find((part) => /^v\d+(?:alpha|beta)?$/i.test(part));
    return version || 'v1beta';
  } catch {
    return 'v1beta';
  }
}

function getGeminiBaseUrl(endpointUrl: string) {
  try {
    const parsed = new URL(endpointUrl);
    return `${parsed.protocol}//${parsed.host}`;
  } catch {
    return undefined;
  }
}

function collectGeminiText(response: any) {
  if (typeof response?.text === 'string') return response.text;
  const parts = response?.candidates?.[0]?.content?.parts;
  if (!Array.isArray(parts)) return '';
  return parts
    .map((part: any) => (!part?.thought && typeof part?.text === 'string' ? part.text : ''))
    .join('');
}

function stripModelPrefix(model: string) {
  return String(model || '').replace(/^models\//i, '').trim();
}

function isGemma3Model(model: string | undefined | null) {
  return /^gemma-3[-_]/i.test(stripModelPrefix(String(model || '')));
}

function shouldUseGeminiSingleUserPrompt(
  request: ReturnType<typeof buildCanonicalTranslationRequest>['request'],
  geminiOptions: Record<string, any>
) {
  const promptMode = String(geminiOptions.promptMode || geminiOptions.prompt_mode || '').trim().toLowerCase();
  if (['system', 'system_instruction', 'split_system'].includes(promptMode)) return false;
  if (['single_user', 'single_user_prompt', 'user_prompt'].includes(promptMode)) return true;

  return Boolean(getCanonicalInstructions(request));
}

function buildGeminiSingleUserContents(
  request: ReturnType<typeof buildCanonicalTranslationRequest>['request'],
  promptOverride?: string
) {
  const instructions = getCanonicalInstructions(request);
  const userText = getCanonicalUserText(request);
  const text = promptOverride
    || [
      instructions,
      instructions && userText ? 'Input:' : '',
      userText,
    ].filter(Boolean).join('\n\n');

  return [
    {
      role: 'user',
      parts: [{ text }],
    },
  ];
}

function buildGeminiDefaultSystemInstruction() {
  return {
    parts: [{ text: 'You are a top-tier subtitle translation engine.' }],
  };
}

function buildGeminiConciseTranslationPrompt(options: CloudTranslateAdapterRequestOptions) {
  const targetLang = String(options.targetLang || '').trim() || 'the requested target language';
  const sourceLang = String(options.sourceLang || '').trim();
  const glossary = String(options.glossary || '').trim();
  const additional = String(options.prompt || options.systemPromptOverride || '').trim();
  const targetDescriptor =
    targetLang.toLowerCase().includes('zh-tw') || /traditional chinese|繁體|繁中/i.test(targetLang)
      ? `Traditional Chinese for Taiwan (${targetLang})`
      : targetLang;
  const rules = [
    `Translate this subtitle text${sourceLang ? ` from ${sourceLang}` : ''} into ${targetDescriptor}. Return only the translation.`,
    options.lineSafeMode
      ? 'If a line starts with a marker such as [[L00001]], keep every marker exactly unchanged and keep one output line per input line.'
      : 'Keep timestamps and line breaks.',
    glossary ? `Glossary / terminology: ${glossary}` : '',
    additional ? `Additional requirements: ${additional}` : '',
    String(options.text || ''),
  ];
  return rules.filter(Boolean).join('\n');
}

function shouldUseGeminiConcisePrompt(options: CloudTranslateAdapterRequestOptions, geminiOptions: Record<string, any>) {
  const promptMode = String(geminiOptions.promptMode || geminiOptions.prompt_mode || '').trim().toLowerCase();
  if (['canonical_user', 'full_user_prompt'].includes(promptMode)) return false;
  if (['concise', 'provider_native'].includes(promptMode)) return true;
  return !options.jsonResponse;
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
      topK: input.options.modelOptions?.sampling?.topK,
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
  const effectiveTimeoutMs = input.deps.resolveRequestTimeoutMs(providerRequest.timeoutMs, timeoutOverride);

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
    responseHeaders: Object.fromEntries(response.headers.entries()),
  };
}

async function requestAnthropic(
  endpointUrl: string,
  options: CloudTranslateAdapterRequestOptions,
  deps: CloudTranslateAdapterDeps
) {
  const { parsed, body, warnings, responseHeaders } = await executeLlmAdapterRequest({
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
      return { text: '__connection_ok__', warnings, responseHeaders };
    }
    throw new Error('Anthropic API returned empty content.');
  }
  return { text: translated, warnings, responseHeaders };
}

async function requestGeminiNativeSdk(
  endpointUrl: string,
  options: CloudTranslateAdapterRequestOptions,
  deps: CloudTranslateAdapterDeps
) {
  const bodyOptions = getProviderBodyOptions(options);
  const geminiOptions = isPlainObject(bodyOptions.gemini) ? (bodyOptions.gemini as Record<string, any>) : {};
  const canonical = buildCanonicalTranslationRequest({
    adapterKey: 'gemini-native',
    providerFamily: 'gemini-native',
    model: String(options.model || '').trim() || 'gemini-2.5-flash',
    text: options.text,
    sourceLang: options.sourceLang,
    targetLang: options.targetLang,
    systemPrompt: deps.resolveSystemPrompt(options),
    jsonResponse: options.jsonResponse,
    isConnectionTest: options.isConnectionTest,
    promptTemplateId: options.promptTemplateId,
    glossary: options.glossary,
    samplingOverrides: {
      temperature: options.modelOptions?.sampling?.temperature,
      topP: options.modelOptions?.sampling?.topP,
      topK: options.modelOptions?.sampling?.topK,
      maxOutputTokens: options.modelOptions?.sampling?.maxOutputTokens,
    },
  });
  const request = canonical.request;
  const requestModel = stripModelPrefix(request.model || options.model || 'gemini-2.5-flash');
  const omitDeveloperInstruction = isGemma3Model(requestModel);
  const generationConfig: Record<string, any> = {
    temperature: request.sampling?.temperature ?? 0.2,
    ...(request.sampling?.maxOutputTokens != null ? { maxOutputTokens: request.sampling.maxOutputTokens } : {}),
    ...(request.sampling?.topP != null ? { topP: request.sampling.topP } : {}),
    ...(request.sampling?.topK != null ? { topK: request.sampling.topK } : {}),
    ...(request.sampling?.seed != null ? { seed: request.sampling.seed } : {}),
    ...(request.sampling?.frequencyPenalty != null ? { frequencyPenalty: request.sampling.frequencyPenalty } : {}),
    ...(request.sampling?.presencePenalty != null ? { presencePenalty: request.sampling.presencePenalty } : {}),
    ...(Array.isArray(request.sampling?.stop) && request.sampling.stop.length > 0
      ? { stopSequences: request.sampling.stop }
      : {}),
    ...(wantsJsonObject(request) ? { responseMimeType: 'application/json' } : {}),
  };
  const jsonSchema = wantsJsonSchema(request);
  if (jsonSchema) {
    generationConfig.responseMimeType = 'application/json';
    generationConfig.responseSchema = jsonSchema.schema;
  }

  const configOverrides = [
    isPlainObject(bodyOptions.generationConfig) ? bodyOptions.generationConfig as Record<string, any> : null,
    isPlainObject(bodyOptions.config) ? bodyOptions.config as Record<string, any> : null,
    isPlainObject(geminiOptions.config) ? geminiOptions.config as Record<string, any> : null,
  ].filter(Boolean) as Array<Record<string, any>>;
  let config = configOverrides.reduce((merged, override) => deepMerge(merged, override), generationConfig);
  const singleUserPrompt = omitDeveloperInstruction || shouldUseGeminiSingleUserPrompt(request, geminiOptions);
  if (singleUserPrompt) {
    if (omitDeveloperInstruction) {
      delete config.systemInstruction;
    } else if (!config.systemInstruction) {
      config.systemInstruction = buildGeminiDefaultSystemInstruction();
    }
  } else {
    const systemInstruction = buildGeminiSystemInstruction(request);
    if (systemInstruction) {
      config.systemInstruction = systemInstruction;
    }
  }
  if (isPlainObject(bodyOptions.thinkingConfig) && !config.thinkingConfig) {
    config.thinkingConfig = bodyOptions.thinkingConfig;
  }
  if (isPlainObject(geminiOptions.thinkingConfig)) {
    config.thinkingConfig = geminiOptions.thinkingConfig;
  }
  if (Array.isArray(bodyOptions.safetySettings) && !config.safetySettings) {
    config.safetySettings = bodyOptions.safetySettings;
  }
  if (Array.isArray(geminiOptions.safetySettings)) {
    config.safetySettings = geminiOptions.safetySettings;
  }

  const timeoutOverride = toFiniteNumber(options.modelOptions?.timeoutMs);
  const effectiveTimeoutMs = deps.resolveRequestTimeoutMs(120000, timeoutOverride);
  const requestHeaders = isPlainObject(options.modelOptions?.headers)
    ? (options.modelOptions?.headers as Record<string, string>)
    : {};
  const ai = new GoogleGenAI({
    apiKey: options.key,
    apiVersion: getGeminiApiVersion(endpointUrl),
    httpOptions: {
      baseUrl: getGeminiBaseUrl(endpointUrl),
      timeout: effectiveTimeoutMs,
      headers: requestHeaders,
    },
  });
  const params = {
    model: requestModel,
    contents: singleUserPrompt
      ? buildGeminiSingleUserContents(
          request,
          shouldUseGeminiConcisePrompt(options, geminiOptions)
            ? buildGeminiConciseTranslationPrompt(options)
            : undefined
        )
      : buildGeminiContents(request),
    config: {
      ...config,
      abortSignal: options.signal,
    },
  };
  const streamByDefault = !options.isConnectionTest && !options.jsonResponse;
  const stream =
    bodyOptions.stream === true ||
    geminiOptions.stream === true ||
    (streamByDefault && bodyOptions.stream !== false && geminiOptions.stream !== false);

  try {
    if (stream) {
      let text = '';
      const response = await ai.models.generateContentStream(params as any);
      for await (const chunk of response as any) {
        deps.throwIfAborted(options.signal);
        text += collectGeminiText(chunk);
      }
      return {
        text,
        body: { candidates: [{ content: { parts: [{ text }] } }] },
        warnings: canonical.warnings,
      };
    }

    const response = await ai.models.generateContent(params as any);
    return {
      text: collectGeminiText(response),
      body: response,
      warnings: canonical.warnings,
    };
  } catch (error: any) {
    if (options.signal?.aborted) throw error;
    const raw = parseJsonSafe(String(error?.message || '')) || error;
    const status = extractErrorStatus(raw) || extractErrorStatus(error) || 502;
    throw deps.makeProviderHttpError(
      'Gemini API error',
      status,
      extractCloudTranslateErrorMessage(raw, String(error?.message || error || 'Gemini API request failed.')),
      null
    );
  }
}

async function requestGeminiNative(
  endpointUrl: string,
  options: CloudTranslateAdapterRequestOptions,
  deps: CloudTranslateAdapterDeps
) {
  const useSdk = shouldUseGeminiSdk(endpointUrl);
  const send = async (jsonResponse: boolean) =>
    useSdk
      ? requestGeminiNativeSdk(
          endpointUrl,
          {
            ...options,
            model: String(options.model || '').trim() || 'gemini-2.5-flash',
            jsonResponse,
          },
          deps
        )
      : executeLlmAdapterRequest({
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

  const translated = String((result as any).parsed?.outputText || (result as any).text || '').trim();
  if (!translated) {
    if (options.isConnectionTest && deps.hasGeminiEnvelope((result as any).body)) {
      return { text: '__connection_ok__', warnings: result.warnings, responseHeaders: result.responseHeaders };
    }
    throw new Error('Gemini API returned empty content.');
  }
  return { text: translated, warnings: result.warnings, responseHeaders: result.responseHeaders };
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
      return { text: '__connection_ok__', warnings: result.warnings, responseHeaders: result.responseHeaders };
    }
    throw new Error('Translation API returned empty content.');
  }
  return { text: translated, warnings: result.warnings, responseHeaders: result.responseHeaders };
}

async function requestOpenAiResponses(
  endpointUrl: string,
  options: CloudTranslateAdapterRequestOptions,
  deps: CloudTranslateAdapterDeps
) {
  const { parsed, body, warnings, responseHeaders } = await executeLlmAdapterRequest({
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
      return { text: '__connection_ok__', warnings, responseHeaders };
    }
    throw new Error('Responses API returned empty content.');
  }
  return { text: translated, warnings, responseHeaders };
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
      return { text: '__connection_ok__', warnings: result.warnings, responseHeaders: result.responseHeaders };
    }
    throw new Error('Mistral API returned empty content.');
  }
  return { text: translated, warnings: result.warnings, responseHeaders: result.responseHeaders };
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
      return { text: '__connection_ok__', warnings: result.warnings, responseHeaders: result.responseHeaders };
    }
    throw new Error('Cohere API returned empty content.');
  }
  return { text: translated, warnings: result.warnings, responseHeaders: result.responseHeaders };
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
      return { text: '__connection_ok__', warnings: result.warnings, responseHeaders: result.responseHeaders };
    }
    throw new Error('xAI API returned empty content.');
  }
  return { text: translated, warnings: result.warnings, responseHeaders: result.responseHeaders };
}

const cloudTranslateAdapters: Record<CloudTranslateProvider, CloudTranslateAdapter> = {
  deepl: deepLCloudTranslateAdapter,
  anthropic: {
    provider: 'anthropic',
    async request(endpointUrl, options, deps) {
      deps.throwIfAborted(options.signal);
      const result = await requestAnthropic(endpointUrl, options, deps);
      return { text: result.text, meta: { endpointUrl, fallbackUsed: false, fallbackType: null, requestWarnings: result.warnings, responseHeaders: result.responseHeaders } };
    },
  },
  'gemini-native': {
    provider: 'gemini-native',
    async request(endpointUrl, options, deps) {
      deps.throwIfAborted(options.signal);
      const result = await requestGeminiNative(endpointUrl, options, deps);
      return { text: result.text, meta: { endpointUrl, fallbackUsed: false, fallbackType: null, requestWarnings: result.warnings, responseHeaders: result.responseHeaders } };
    },
  },
  'mistral-chat': {
    provider: 'mistral-chat',
    async request(endpointUrl, options, deps) {
      deps.throwIfAborted(options.signal);
      const result = await requestMistralChat(endpointUrl, options, deps);
      return { text: result.text, meta: { endpointUrl, fallbackUsed: false, fallbackType: null, requestWarnings: result.warnings, responseHeaders: result.responseHeaders } };
    },
  },
  'cohere-chat': {
    provider: 'cohere-chat',
    async request(endpointUrl, options, deps) {
      deps.throwIfAborted(options.signal);
      const result = await requestCohereChat(endpointUrl, options, deps);
      return { text: result.text, meta: { endpointUrl, fallbackUsed: false, fallbackType: null, requestWarnings: result.warnings, responseHeaders: result.responseHeaders } };
    },
  },
  'xai-chat': {
    provider: 'xai-chat',
    async request(endpointUrl, options, deps) {
      deps.throwIfAborted(options.signal);
      const result = await requestXaiChat(endpointUrl, options, deps);
      return { text: result.text, meta: { endpointUrl, fallbackUsed: false, fallbackType: null, requestWarnings: result.warnings, responseHeaders: result.responseHeaders } };
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
        return { text: result.text, meta: { endpointUrl, fallbackUsed: false, fallbackType: null, requestWarnings: result.warnings, responseHeaders: result.responseHeaders } };
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
            responseHeaders: result.responseHeaders,
          },
        };
      }
    },
  },
};

export function getCloudTranslateAdapter(provider: CloudTranslateProvider) {
  return cloudTranslateAdapters[provider];
}
