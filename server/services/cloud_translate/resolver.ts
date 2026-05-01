import { llmModelProfileRegistry } from '../llm/profiles/registry.js';
import {
  matchCloudTranslateProvider,
  requireCloudTranslateProviderDefinition,
} from './registry.js';
import type {
  CloudTranslateProvider,
  ResolveCloudTranslateProviderInput,
  ResolvedCloudTranslateProvider,
} from './types.js';

function parseUrl(url: string) {
  return new URL(String(url || '').trim());
}

function matchProfile(modelName?: string, model?: string) {
  return llmModelProfileRegistry.match(String(model || '').trim() || String(modelName || '').trim());
}

export function redactUrlSecrets(rawUrl: string) {
  try {
    const parsed = new URL(rawUrl);
    ['key', 'api_key', 'token', 'auth_key'].forEach((k) => parsed.searchParams.delete(k));
    return parsed.toString();
  } catch {
    return rawUrl;
  }
}

export function detectCloudTranslateProvider(url: string, modelName?: string): CloudTranslateProvider {
  const parsedUrl = parseUrl(url);
  const provider = matchCloudTranslateProvider({
    rawUrl: String(url || ''),
    parsedUrl,
    hostname: parsedUrl.hostname.toLowerCase(),
    pathname: parsedUrl.pathname.toLowerCase(),
    modelName: String(modelName || '').toLowerCase(),
  });
  return provider.provider;
}

export function getDefaultCloudTranslateModel(provider: CloudTranslateProvider) {
  return requireCloudTranslateProviderDefinition(provider).defaultModel;
}

export function getCloudTranslateRuntimeAdapterKey(provider: CloudTranslateProvider) {
  return requireCloudTranslateProviderDefinition(provider).adapterKey;
}

export function getEffectiveCloudTranslateModel(
  provider: CloudTranslateProvider,
  input: { modelName?: string; model?: string }
) {
  const explicitModel = String(input.model || '').trim();
  if (explicitModel) return explicitModel;

  const matchedProfile = matchProfile(input.modelName, input.model);
  const profileDefaultModel = String(matchedProfile?.defaultModel || '').trim();
  if (profileDefaultModel) return profileDefaultModel;

  return getDefaultCloudTranslateModel(provider);
}

export function supportsCloudContextStrategy(provider: CloudTranslateProvider) {
  return requireCloudTranslateProviderDefinition(provider).capabilities.supportsLargeContext;
}

export function buildCloudTranslateEndpointUrl(
  provider: CloudTranslateProvider,
  rawUrl: string,
  model: string | undefined,
  apiKey: string | undefined
) {
  return requireCloudTranslateProviderDefinition(provider).buildEndpointUrl(rawUrl, model, apiKey);
}

export function resolveCloudTranslateProvider(input: ResolveCloudTranslateProviderInput): ResolvedCloudTranslateProvider {
  const provider = detectCloudTranslateProvider(input.url, input.modelName);
  const definition = requireCloudTranslateProviderDefinition(provider);
  const effectiveModel = getEffectiveCloudTranslateModel(provider, input);
  const matchedProfile = matchProfile(input.modelName, input.model) || matchProfile(input.modelName, effectiveModel);
  const quotaProfile = definition.getQuotaProfile?.({
    model: effectiveModel,
    options: input.options,
  }) || null;

  return {
    provider,
    endpointUrl: definition.buildEndpointUrl(input.url, effectiveModel, input.apiKey),
    effectiveModel,
    adapterKey: definition.adapterKey,
    profileId: matchedProfile?.id || definition.capabilities.profileId || null,
    profileFamily: matchedProfile?.family || definition.capabilities.profileFamily || null,
    capabilities: definition.capabilities,
    quotaProfile,
    defaultExecutionMode: definition.capabilities.defaultExecutionMode,
  };
}
