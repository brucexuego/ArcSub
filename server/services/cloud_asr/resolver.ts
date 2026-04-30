import type { CloudAsrProvider, ResolveCloudAsrProviderInput, ResolvedCloudAsrProvider } from './types.js';
import {
  matchCloudAsrProvider,
  requireCloudAsrProviderDefinition,
} from './registry.js';

function parseUrl(url: string) {
  return new URL(String(url || '').trim());
}

export function detectCloudAsrProvider(url: string, modelName?: string): CloudAsrProvider {
  const parsedUrl = parseUrl(url);
  const provider = matchCloudAsrProvider({
    rawUrl: String(url || ''),
    parsedUrl,
    hostname: parsedUrl.hostname.toLowerCase(),
    pathname: parsedUrl.pathname.toLowerCase(),
    modelName: String(modelName || '').toLowerCase(),
  });
  return provider.provider;
}

export function buildCloudAsrEndpointUrl(provider: CloudAsrProvider, rawUrl: string, model?: string) {
  return requireCloudAsrProviderDefinition(provider).buildEndpointUrl(rawUrl, model);
}

export function getDefaultCloudAsrModel(provider: CloudAsrProvider) {
  return requireCloudAsrProviderDefinition(provider).defaultModel;
}

export function buildCloudAsrRequestHeaders(provider: CloudAsrProvider, key?: string): Record<string, string> {
  return requireCloudAsrProviderDefinition(provider).buildHeaders(key);
}

function isGithubPhi4MultimodalModel(model: string) {
  const normalized = String(model || '').trim().toLowerCase();
  return normalized.includes('phi-4-multimodal-instruct') ||
    /(^|[/:\s_-])phi[-_\s]?4(?:[-_\s]multimodal)?($|[/:\s_-])/i.test(normalized);
}

function resolveCloudAsrProfileId(provider: CloudAsrProvider, effectiveModel: string, fallback: string | null | undefined) {
  if (provider === 'github-models') {
    return isGithubPhi4MultimodalModel(effectiveModel) ? 'github-phi4-multimodal' : 'github-models';
  }
  return fallback || null;
}

export function resolveCloudAsrProvider(input: ResolveCloudAsrProviderInput): ResolvedCloudAsrProvider {
  const provider = detectCloudAsrProvider(input.url, input.modelName);
  const definition = requireCloudAsrProviderDefinition(provider);
  const rawModel = String(input.model || '').trim() || definition.defaultModel;
  const effectiveModel = provider === 'github-models'
    ? rawModel.toLowerCase()
    : rawModel;
  const capabilities = definition.capabilities;
  const profileId = resolveCloudAsrProfileId(provider, effectiveModel, capabilities.profileId);

  return {
    provider,
    providerKey: provider,
    endpointUrl: definition.buildEndpointUrl(input.url, effectiveModel),
    effectiveModel,
    defaultModel: definition.defaultModel,
    capabilities,
    profileId,
    profileFamily: capabilities.profileFamily || null,
  };
}
