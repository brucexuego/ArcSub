import type { CloudTranslateProviderDefinition } from '../types.js';
import {
  createCloudTranslateProviderDefinition,
  createDefaultCapabilities,
  ensureEndpointPath,
  parseCloudTranslateUrl,
} from './shared.js';

export const deeplTranslateProvider: CloudTranslateProviderDefinition = createCloudTranslateProviderDefinition({
  provider: 'deepl',
  defaultModel: 'deepl-v2',
  adapterKey: 'deepl-translate',
  capabilities: createDefaultCapabilities({
    provider: 'deepl',
    profileId: 'deepl-translate',
    profileFamily: 'deepl',
    supportsJsonObject: false,
    supportsJsonSchema: false,
    translationNative: true,
    defaultExecutionMode: 'provider_native',
  }),
  detect(input) {
    return input.hostname.includes('deepl.com') || input.modelName.includes('deepl');
  },
  buildEndpointUrl(rawUrl) {
    const parsed = parseCloudTranslateUrl(rawUrl);
    if (parsed.pathname.toLowerCase().includes('/translate')) return parsed.toString();
    return ensureEndpointPath(parsed, '/v2/translate').toString();
  },
});

export const ollamaChatTranslateProvider: CloudTranslateProviderDefinition = createCloudTranslateProviderDefinition({
  provider: 'ollama-chat',
  defaultModel: 'qwen2.5:7b',
  adapterKey: 'ollama-chat',
  capabilities: createDefaultCapabilities({
    provider: 'ollama-chat',
    profileId: 'ollama-chat',
    profileFamily: 'ollama',
    supportsJsonSchema: false,
    defaultExecutionMode: 'cloud_strict',
  }),
  detect(input) {
    return input.pathname.includes('/api/chat') || input.hostname.includes('ollama') || input.modelName.includes('ollama');
  },
  buildEndpointUrl(rawUrl) {
    const parsed = parseCloudTranslateUrl(rawUrl);
    if (parsed.pathname.toLowerCase().endsWith('/api/chat')) return parsed.toString();
    return ensureEndpointPath(parsed, '/api/chat').toString();
  },
});

export const ollamaGenerateTranslateProvider: CloudTranslateProviderDefinition = createCloudTranslateProviderDefinition({
  provider: 'ollama-generate',
  defaultModel: 'qwen2.5:7b',
  adapterKey: 'ollama-generate',
  capabilities: createDefaultCapabilities({
    provider: 'ollama-generate',
    profileId: 'ollama-generate',
    profileFamily: 'ollama',
    supportsJsonSchema: false,
    defaultExecutionMode: 'cloud_strict',
  }),
  detect(input) {
    return input.pathname.includes('/api/generate');
  },
  buildEndpointUrl(rawUrl) {
    return parseCloudTranslateUrl(rawUrl).toString();
  },
});
