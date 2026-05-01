import { resolveCloudTranslateQuotaProfile } from '../profiles/quotas.js';
import type { CloudTranslateProviderDefinition } from '../types.js';
import {
  createCloudTranslateProviderDefinition,
  createDefaultCapabilities,
  ensureEndpointPath,
  parseCloudTranslateUrl,
} from './shared.js';

export const googleGeminiTranslateProvider: CloudTranslateProviderDefinition = createCloudTranslateProviderDefinition({
  provider: 'gemini-native',
  defaultModel: 'gemini-2.5-flash',
  adapterKey: 'gemini-native',
  capabilities: createDefaultCapabilities({
    provider: 'gemini-native',
    profileId: 'google-gemini-translate',
    profileFamily: 'google-gemini',
    supportsStreaming: true,
    supportsReasoningConfig: true,
    supportsTopK: true,
    defaultExecutionMode: 'cloud_relaxed',
  }),
  detect(input) {
    return input.hostname === 'generativelanguage.googleapis.com' ||
      input.pathname.includes(':generatecontent') ||
      input.modelName.includes('gemini') ||
      input.modelName.includes('gemma');
  },
  buildEndpointUrl(rawUrl, model, apiKey) {
    const parsed = parseCloudTranslateUrl(rawUrl);
    if (parsed.pathname.toLowerCase().includes(':generatecontent')) {
      const next = new URL(parsed.toString());
      if (apiKey && !next.searchParams.get('key')) next.searchParams.set('key', apiKey);
      return next.toString();
    }
    const targetModel = String(model || '').trim() || 'gemini-2.5-flash';
    const next = ensureEndpointPath(parsed, `/v1beta/models/${encodeURIComponent(targetModel)}:generateContent`);
    if (apiKey && !next.searchParams.get('key')) next.searchParams.set('key', apiKey);
    return next.toString();
  },
  getQuotaProfile(input) {
    return resolveCloudTranslateQuotaProfile({
      provider: 'gemini-native',
      model: input.model,
      options: input.options,
      defaults: {
        tokenEstimator: 'gemini_like',
        safetyReserveTokens: 256,
      },
    });
  },
});
