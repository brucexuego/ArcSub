import type {
  CloudTranslateProvider,
  CloudTranslateProviderDefinition,
  CloudTranslateProviderMatchInput,
} from './types.js';
import {
  anthropicTranslateProvider,
  cohereTranslateProvider,
  mistralTranslateProvider,
  xaiTranslateProvider,
} from './providers/chat_providers.js';
import { googleGeminiTranslateProvider } from './providers/google_gemini.js';
import {
  deeplTranslateProvider,
  ollamaChatTranslateProvider,
  ollamaGenerateTranslateProvider,
} from './providers/local_compat.js';
import { openAiCompatibleTranslateProvider } from './providers/openai_compatible.js';

const PROVIDERS: CloudTranslateProviderDefinition[] = [
  deeplTranslateProvider,
  anthropicTranslateProvider,
  googleGeminiTranslateProvider,
  mistralTranslateProvider,
  cohereTranslateProvider,
  xaiTranslateProvider,
  ollamaGenerateTranslateProvider,
  ollamaChatTranslateProvider,
  openAiCompatibleTranslateProvider,
];

const PROVIDER_BY_KEY = new Map<CloudTranslateProvider, CloudTranslateProviderDefinition>(
  PROVIDERS.map((provider) => [provider.provider, provider])
);

export function listCloudTranslateProviders() {
  return [...PROVIDERS];
}

export function getCloudTranslateProviderDefinition(provider: CloudTranslateProvider) {
  return PROVIDER_BY_KEY.get(provider) || null;
}

export function requireCloudTranslateProviderDefinition(provider: CloudTranslateProvider) {
  const definition = getCloudTranslateProviderDefinition(provider);
  if (!definition) {
    throw new Error(`Cloud translation provider is not registered: ${provider}`);
  }
  return definition;
}

export function matchCloudTranslateProvider(input: CloudTranslateProviderMatchInput): CloudTranslateProviderDefinition {
  return PROVIDERS.find((provider) => provider.detect(input)) || openAiCompatibleTranslateProvider;
}
