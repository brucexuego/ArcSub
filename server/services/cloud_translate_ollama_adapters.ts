import { buildCanonicalTranslationRequest } from './llm/mapping/translation_canonical_request.js';
import { buildOllamaPrompt, buildOpenAiChatMessages } from './llm/mapping/provider_payloads.js';
import type {
  CloudTranslateAdapter,
  CloudTranslateAdapterDeps,
  CloudTranslateAdapterRequestOptions,
} from './cloud_translate_adapter.js';

function isPlainObject(value: unknown): value is Record<string, any> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function deepMerge(base: Record<string, any>, override: Record<string, any>): Record<string, any> {
  const merged: Record<string, any> = { ...base };
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

function getOllamaBodyOptions(options: CloudTranslateAdapterRequestOptions) {
  return isPlainObject(options.modelOptions?.body)
    ? (options.modelOptions?.body as Record<string, any>)
    : {};
}

function buildOllamaGenerateOptions(request: ReturnType<typeof buildCanonicalTranslationRequest>['request']) {
  const generationOptions: Record<string, number> = {
    temperature: request.sampling?.temperature ?? 0.2,
  };
  if (request.sampling?.topP != null) generationOptions.top_p = request.sampling.topP;
  if (request.sampling?.topK != null) generationOptions.top_k = request.sampling.topK;
  if (request.sampling?.maxOutputTokens != null) generationOptions.num_predict = request.sampling.maxOutputTokens;
  return generationOptions;
}

async function requestOllamaChat(
  endpointUrl: string,
  options: CloudTranslateAdapterRequestOptions,
  deps: CloudTranslateAdapterDeps
) {
  const bodyOptions = getOllamaBodyOptions(options);
  const canonical = buildCanonicalTranslationRequest({
    adapterKey: 'openai-compatible-chat',
    model: String(options.model || '').trim() || 'qwen2.5:7b',
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

  const body = deepMerge({
    model: request.model,
    stream: false,
    think: false,
    options: buildOllamaGenerateOptions(request),
    messages: buildOpenAiChatMessages(request),
  }, bodyOptions);

  const response = await deps.fetchWithTimeout(
    endpointUrl,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    },
    deps.resolveRequestTimeoutMs(120000, options.modelOptions?.timeoutMs ?? undefined),
    options.signal
  );
  const rawText = await response.text();
  if (!response.ok) {
    throw deps.makeProviderHttpError(
      'Ollama API error',
      response.status,
      deps.extractErrorMessage(rawText, response.statusText),
      deps.parseRetryAfterMs(response)
    );
  }

  const data = JSON.parse(rawText || '{}');
  const translated = deps.parseOllamaContent(data).trim();
  if (!translated) {
    if (options.isConnectionTest && deps.hasOllamaEnvelope(data)) {
      return { text: '__connection_ok__', warnings: canonical.warnings };
    }
    throw new Error('Ollama API returned empty content.');
  }
  return { text: translated, warnings: canonical.warnings };
}

async function requestOllamaGenerate(
  endpointUrl: string,
  options: CloudTranslateAdapterRequestOptions,
  deps: CloudTranslateAdapterDeps
) {
  const bodyOptions = getOllamaBodyOptions(options);
  const canonical = buildCanonicalTranslationRequest({
    adapterKey: 'openai-compatible-chat',
    model: String(options.model || '').trim() || 'qwen2.5:7b',
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
  const body = deepMerge({
    model: request.model,
    prompt: buildOllamaPrompt(request),
    stream: false,
    think: false,
    options: buildOllamaGenerateOptions(request),
  }, bodyOptions);

  const response = await deps.fetchWithTimeout(
    endpointUrl,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    },
    deps.resolveRequestTimeoutMs(120000, options.modelOptions?.timeoutMs ?? undefined),
    options.signal
  );
  const rawText = await response.text();
  if (!response.ok) {
    throw deps.makeProviderHttpError(
      'Ollama API error',
      response.status,
      deps.extractErrorMessage(rawText, response.statusText),
      deps.parseRetryAfterMs(response)
    );
  }

  const data = JSON.parse(rawText || '{}');
  const translated = deps.parseOllamaContent(data).trim();
  if (!translated) {
    if (options.isConnectionTest && deps.hasOllamaEnvelope(data)) {
      return { text: '__connection_ok__', warnings: canonical.warnings };
    }
    throw new Error('Ollama API returned empty content.');
  }
  return { text: translated, warnings: canonical.warnings };
}

export const ollamaChatCloudTranslateAdapter: CloudTranslateAdapter = {
  provider: 'ollama-chat',
  async request(endpointUrl, options, deps, onProgress) {
    deps.throwIfAborted(options.signal);
    try {
      const result = await requestOllamaChat(endpointUrl, options, deps);
      return { text: result.text, meta: { endpointUrl, fallbackUsed: false, fallbackType: null, requestWarnings: result.warnings } };
    } catch (error) {
      if (!deps.isProviderHttpError(error) || ![404, 405].includes(error.status)) throw error;
      const fallbackUrl = deps.getOllamaFallbackEndpoint(endpointUrl, 'generate');
      onProgress?.('Ollama /api/chat unavailable, retrying with /api/generate fallback...');
      const result = await requestOllamaGenerate(fallbackUrl, options, deps);
      return {
        text: result.text,
        meta: { endpointUrl: fallbackUrl, fallbackUsed: true, fallbackType: 'ollama_generate', requestWarnings: result.warnings },
      };
    }
  },
};

export const ollamaGenerateCloudTranslateAdapter: CloudTranslateAdapter = {
  provider: 'ollama-generate',
  async request(endpointUrl, options, deps, onProgress) {
    deps.throwIfAborted(options.signal);
    try {
      const result = await requestOllamaGenerate(endpointUrl, options, deps);
      return { text: result.text, meta: { endpointUrl, fallbackUsed: false, fallbackType: null, requestWarnings: result.warnings } };
    } catch (error) {
      if (!deps.isProviderHttpError(error) || ![404, 405].includes(error.status)) throw error;
      const fallbackUrl = deps.getOllamaFallbackEndpoint(endpointUrl, 'chat');
      onProgress?.('Ollama /api/generate unavailable, retrying with /api/chat fallback...');
      const result = await requestOllamaChat(fallbackUrl, options, deps);
      return {
        text: result.text,
        meta: { endpointUrl: fallbackUrl, fallbackUsed: true, fallbackType: 'ollama_chat', requestWarnings: result.warnings },
      };
    }
  },
};
