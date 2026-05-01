import { resolveCloudTranslateQuotaProfile } from '../profiles/quotas.js';
import type { CloudTranslateProviderDefinition } from '../types.js';
import {
  createCloudTranslateProviderDefinition,
  createDefaultCapabilities,
  ensureEndpointPath,
  parseCloudTranslateUrl,
} from './shared.js';

export const anthropicTranslateProvider: CloudTranslateProviderDefinition = createCloudTranslateProviderDefinition({
  provider: 'anthropic',
  defaultModel: 'claude-3-5-sonnet-latest',
  adapterKey: 'anthropic-messages',
  capabilities: createDefaultCapabilities({
    provider: 'anthropic',
    profileId: 'anthropic-messages',
    profileFamily: 'anthropic',
    supportsReasoningConfig: true,
    defaultExecutionMode: 'cloud_relaxed',
  }),
  detect(input) {
    return input.hostname.includes('anthropic.com') || input.pathname.includes('/v1/messages') ||
      input.modelName.includes('claude') || input.modelName.includes('anthropic');
  },
  buildEndpointUrl(rawUrl) {
    const parsed = parseCloudTranslateUrl(rawUrl);
    if (parsed.pathname.toLowerCase().endsWith('/v1/messages')) return parsed.toString();
    return ensureEndpointPath(parsed, '/v1/messages').toString();
  },
  getQuotaProfile(input) {
    return resolveCloudTranslateQuotaProfile({ provider: 'anthropic', model: input.model, options: input.options });
  },
});

export const mistralTranslateProvider: CloudTranslateProviderDefinition = createCloudTranslateProviderDefinition({
  provider: 'mistral-chat',
  defaultModel: 'mistral-small-latest',
  adapterKey: 'mistral-chat',
  capabilities: createDefaultCapabilities({
    provider: 'mistral-chat',
    profileId: 'mistral-chat',
    profileFamily: 'mistral',
    defaultExecutionMode: 'cloud_relaxed',
  }),
  detect(input) {
    return input.hostname.includes('mistral.ai') || input.modelName.includes('mistral');
  },
  buildEndpointUrl(rawUrl) {
    const parsed = parseCloudTranslateUrl(rawUrl);
    if (parsed.pathname.toLowerCase().includes('/chat/completions')) return parsed.toString();
    return ensureEndpointPath(parsed, '/v1/chat/completions').toString();
  },
  getQuotaProfile(input) {
    return resolveCloudTranslateQuotaProfile({ provider: 'mistral-chat', model: input.model, options: input.options });
  },
});

export const cohereTranslateProvider: CloudTranslateProviderDefinition = createCloudTranslateProviderDefinition({
  provider: 'cohere-chat',
  defaultModel: 'command-a-03-2025',
  adapterKey: 'cohere-chat',
  capabilities: createDefaultCapabilities({
    provider: 'cohere-chat',
    profileId: 'cohere-chat',
    profileFamily: 'cohere',
    supportsTopK: true,
    translationNative: true,
    defaultExecutionMode: 'provider_native',
  }),
  detect(input) {
    return input.hostname.includes('cohere.com') || input.pathname.includes('/v2/chat') ||
      input.modelName.includes('cohere') || input.modelName.includes('command-r') || input.modelName.includes('command a');
  },
  buildEndpointUrl(rawUrl) {
    const parsed = parseCloudTranslateUrl(rawUrl);
    if (parsed.pathname.toLowerCase().endsWith('/v2/chat')) return parsed.toString();
    return ensureEndpointPath(parsed, '/v2/chat').toString();
  },
  getQuotaProfile(input) {
    return resolveCloudTranslateQuotaProfile({ provider: 'cohere-chat', model: input.model, options: input.options });
  },
});

export const xaiTranslateProvider: CloudTranslateProviderDefinition = createCloudTranslateProviderDefinition({
  provider: 'xai-chat',
  defaultModel: 'grok-4-fast-reasoning',
  adapterKey: 'xai-chat',
  capabilities: createDefaultCapabilities({
    provider: 'xai-chat',
    profileId: 'xai-chat',
    profileFamily: 'xai',
    supportsReasoningConfig: true,
    defaultExecutionMode: 'cloud_relaxed',
  }),
  detect(input) {
    return input.hostname.includes('x.ai') || input.modelName.includes('xai') || input.modelName.includes('grok');
  },
  buildEndpointUrl(rawUrl) {
    const parsed = parseCloudTranslateUrl(rawUrl);
    if (parsed.pathname.toLowerCase().includes('/chat/completions')) return parsed.toString();
    return ensureEndpointPath(parsed, '/v1/chat/completions').toString();
  },
  getQuotaProfile(input) {
    return resolveCloudTranslateQuotaProfile({ provider: 'xai-chat', model: input.model, options: input.options });
  },
});
