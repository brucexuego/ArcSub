import type {
  CloudTranslateAdapter,
  CloudTranslateAdapterDeps,
  CloudTranslateAdapterRequestOptions,
} from './cloud_translate_adapter.js';

async function requestDeepL(
  endpointUrl: string,
  options: CloudTranslateAdapterRequestOptions,
  deps: CloudTranslateAdapterDeps
) {
  const payload = new URLSearchParams();
  payload.set('text', options.text);
  payload.set('target_lang', deps.normalizeDeepLTargetLanguage(options.targetLang));
  if (options.key) payload.set('auth_key', options.key);
  if (options.glossary) payload.set('context', options.glossary.slice(0, 4000));

  const response = await deps.fetchWithTimeout(
    endpointUrl,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: payload.toString(),
    },
    deps.resolveRequestTimeoutMs(30000, options.modelOptions?.timeoutMs ?? undefined),
    options.signal
  );

  const rawText = await response.text();
  if (!response.ok) {
    throw deps.makeProviderHttpError(
      'DeepL API error',
      response.status,
      deps.extractErrorMessage(rawText, response.statusText),
      deps.parseRetryAfterMs(response)
    );
  }

  const data = JSON.parse(rawText || '{}');
  const translated = data?.translations?.[0]?.text;
  if (typeof translated !== 'string' || !translated.trim()) {
    if (options.isConnectionTest && Array.isArray(data?.translations)) {
      return '__connection_ok__';
    }
    throw new Error('DeepL returned empty translation.');
  }
  return translated;
}

export const deepLCloudTranslateAdapter: CloudTranslateAdapter = {
  provider: 'deepl',
  async request(endpointUrl, options, deps) {
    deps.throwIfAborted(options.signal);
    const text = await requestDeepL(endpointUrl, options, deps);
    return { text, meta: { endpointUrl, fallbackUsed: false, fallbackType: null } };
  },
};
