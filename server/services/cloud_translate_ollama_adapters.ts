import { buildCanonicalTranslationRequest } from './llm/mapping/translation_canonical_request.js';
import { buildOllamaPrompt, buildOpenAiChatMessages } from './llm/mapping/provider_payloads.js';
import type {
  CloudTranslateAdapter,
  CloudTranslateAdapterDeps,
  CloudTranslateAdapterRequestOptions,
} from './cloud_translate_adapter.js';

async function requestOllamaChat(
  endpointUrl: string,
  options: CloudTranslateAdapterRequestOptions,
  deps: CloudTranslateAdapterDeps
) {
  const canonical = buildCanonicalTranslationRequest({
    adapterKey: 'openai-compatible-chat',
    model: String(options.model || '').trim() || 'qwen2.5:7b',
    text: options.text,
    targetLang: options.targetLang,
    systemPrompt: deps.resolveSystemPrompt(options),
    jsonResponse: options.jsonResponse,
    isConnectionTest: options.isConnectionTest,
    promptTemplateId: options.promptTemplateId,
    glossary: options.glossary,
  });
  const request = canonical.request;

  const body = {
    model: request.model,
    stream: false,
    options: { temperature: request.sampling?.temperature ?? 0.2 },
    messages: buildOpenAiChatMessages(request),
  };

  const response = await deps.fetchWithTimeout(
    endpointUrl,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    },
    120000,
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
  const canonical = buildCanonicalTranslationRequest({
    adapterKey: 'openai-compatible-chat',
    model: String(options.model || '').trim() || 'qwen2.5:7b',
    text: options.text,
    targetLang: options.targetLang,
    systemPrompt: deps.resolveSystemPrompt(options),
    jsonResponse: options.jsonResponse,
    isConnectionTest: options.isConnectionTest,
    promptTemplateId: options.promptTemplateId,
    glossary: options.glossary,
  });
  const request = canonical.request;
  const body = {
    model: request.model,
    prompt: buildOllamaPrompt(request),
    stream: false,
    options: { temperature: request.sampling?.temperature ?? 0.2 },
  };

  const response = await deps.fetchWithTimeout(
    endpointUrl,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    },
    120000,
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
