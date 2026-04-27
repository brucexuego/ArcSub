import type { LlmAdapterKey } from './llm/canonical/llm_capabilities.js';
import { llmModelProfileRegistry } from './llm/profiles/registry.js';

export type CloudTranslateProvider =
  | 'deepl'
  | 'anthropic'
  | 'gemini-native'
  | 'mistral-chat'
  | 'cohere-chat'
  | 'xai-chat'
  | 'ollama-chat'
  | 'ollama-generate'
  | 'openai-compatible';

export type CloudTranslateRuntimeAdapterKey =
  | LlmAdapterKey
  | 'deepl-translate'
  | 'ollama-chat'
  | 'ollama-generate';

export interface ResolveCloudTranslateProviderInput {
  url: string;
  modelName?: string;
  model?: string;
  apiKey?: string;
}

export interface ResolvedCloudTranslateProvider {
  provider: CloudTranslateProvider;
  endpointUrl: string;
  effectiveModel: string;
  adapterKey: CloudTranslateRuntimeAdapterKey;
  profileId: string | null;
  profileFamily: string | null;
}

function matchProfile(modelName?: string, model?: string) {
  return llmModelProfileRegistry.match(String(model || '').trim() || String(modelName || '').trim());
}

function parseUrl(url: string) {
  return new URL(String(url || '').trim());
}

function isGitHubModelsHost(parsed: URL) {
  return parsed.hostname.toLowerCase() === 'models.github.ai';
}

function buildGitHubModelsChatEndpoint(parsed: URL) {
  const next = new URL(parsed.toString());
  const parts = next.pathname
    .split('/')
    .map((part) => part.trim())
    .filter(Boolean);
  const orgIndex = parts.findIndex((part) => part.toLowerCase() === 'orgs');
  const org = orgIndex >= 0 ? parts[orgIndex + 1] : '';

  next.pathname = org
    ? `/orgs/${encodeURIComponent(org)}/inference/chat/completions`
    : '/inference/chat/completions';
  return next.toString();
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
  const parsed = parseUrl(url);
  const host = parsed.hostname.toLowerCase();
  const pathname = parsed.pathname.toLowerCase();
  const named = String(modelName || '').toLowerCase();

  if (host.includes('deepl.com')) return 'deepl';
  if (host.includes('anthropic.com') || pathname.includes('/v1/messages')) return 'anthropic';
  if (host.includes('mistral.ai')) return 'mistral-chat';
  if (host.includes('cohere.com') || pathname.includes('/v2/chat')) return 'cohere-chat';
  if (host.includes('x.ai')) return 'xai-chat';
  if (isGitHubModelsHost(parsed)) return 'openai-compatible';

  if (host.includes('generativelanguage.googleapis.com')) {
    if (pathname.includes('/openai/')) return 'openai-compatible';
    return 'gemini-native';
  }

  if (pathname.includes('/api/generate')) return 'ollama-generate';
  if (pathname.includes('/api/chat')) return 'ollama-chat';
  if (host.includes('ollama')) return 'ollama-chat';

  if (named.includes('anthropic') || named.includes('claude')) return 'anthropic';
  if (named.includes('gemini')) return 'gemini-native';
  if (named.includes('mistral')) return 'mistral-chat';
  if (named.includes('cohere') || named.includes('command-r') || named.includes('command a')) return 'cohere-chat';
  if (named.includes('xai') || named.includes('grok')) return 'xai-chat';
  if (named.includes('ollama')) return 'ollama-chat';
  if (named.includes('deepl')) return 'deepl';

  return 'openai-compatible';
}

export function getCloudTranslateRuntimeAdapterKey(
  provider: CloudTranslateProvider,
  modelName?: string,
  model?: string
): CloudTranslateRuntimeAdapterKey {
  if (provider === 'deepl') return 'deepl-translate';
  if (provider === 'anthropic') return 'anthropic-messages';
  if (provider === 'gemini-native') return 'gemini-native';
  if (provider === 'mistral-chat') return 'mistral-chat';
  if (provider === 'cohere-chat') return 'cohere-chat';
  if (provider === 'xai-chat') return 'xai-chat';
  if (provider === 'ollama-chat') return 'ollama-chat';
  if (provider === 'ollama-generate') return 'ollama-generate';

  return 'openai-compatible-chat';
}

export function getDefaultCloudTranslateModel(provider: CloudTranslateProvider) {
  if (provider === 'anthropic') return 'claude-3-5-sonnet-latest';
  if (provider === 'gemini-native') return 'gemini-2.5-flash';
  if (provider === 'mistral-chat') return 'mistral-small-latest';
  if (provider === 'cohere-chat') return 'command-a-03-2025';
  if (provider === 'xai-chat') return 'grok-4-fast-reasoning';
  if (provider === 'ollama-chat' || provider === 'ollama-generate') return 'qwen2.5:7b';
  if (provider === 'deepl') return 'deepl-v2';
  return 'gpt-4o-mini';
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
  return (
    provider === 'openai-compatible' ||
    provider === 'anthropic' ||
    provider === 'gemini-native' ||
    provider === 'mistral-chat' ||
    provider === 'cohere-chat' ||
    provider === 'xai-chat'
  );
}

export function buildCloudTranslateEndpointUrl(
  provider: CloudTranslateProvider,
  rawUrl: string,
  model: string | undefined,
  apiKey: string | undefined
) {
  const parsed = parseUrl(rawUrl);
  const path = parsed.pathname.toLowerCase();
  const trimmedModel = String(model || '').trim();

  if (provider === 'openai-compatible' && isGitHubModelsHost(parsed)) {
    return buildGitHubModelsChatEndpoint(parsed);
  }

  const ensureEndpointPath = (targetPath: string) => {
    const next = new URL(parsed.toString());
    const current = (next.pathname || '').replace(/\/+$/, '');
    const target = `/${targetPath.replace(/^\/+/, '')}`.replace(/\/{2,}/g, '/');
    const currentLower = current.toLowerCase();
    const targetLower = target.toLowerCase();

    if (!current || current === '/') {
      next.pathname = target;
      return next;
    }
    if (currentLower === targetLower || currentLower.endsWith(targetLower)) {
      next.pathname = current;
      return next;
    }
    if (currentLower === '/v1' && targetLower.startsWith('/v1/')) {
      next.pathname = target;
      return next;
    }
    if (currentLower === '/v2' && targetLower.startsWith('/v2/')) {
      next.pathname = target;
      return next;
    }

    next.pathname = `${current}/${target.slice(1)}`.replace(/\/{2,}/g, '/');
    return next;
  };

  if (provider === 'deepl') {
    if (path.includes('/translate')) return parsed.toString();
    return ensureEndpointPath('/v2/translate').toString();
  }

  if (provider === 'anthropic') {
    if (path.endsWith('/v1/messages')) return parsed.toString();
    return ensureEndpointPath('/v1/messages').toString();
  }

  if (provider === 'cohere-chat') {
    if (path.endsWith('/v2/chat')) return parsed.toString();
    return ensureEndpointPath('/v2/chat').toString();
  }

  if (provider === 'ollama-generate') return parsed.toString();

  if (provider === 'ollama-chat') {
    if (path.endsWith('/api/chat')) return parsed.toString();
    return ensureEndpointPath('/api/chat').toString();
  }

  if (provider === 'gemini-native') {
    if (path.includes(':generatecontent')) {
      const next = new URL(parsed.toString());
      if (apiKey && !next.searchParams.get('key')) next.searchParams.set('key', apiKey);
      return next.toString();
    }

    const targetModel = trimmedModel || getDefaultCloudTranslateModel('gemini-native');
    const next = ensureEndpointPath(`/v1beta/models/${encodeURIComponent(targetModel)}:generateContent`);
    if (apiKey && !next.searchParams.get('key')) next.searchParams.set('key', apiKey);
    return next.toString();
  }

  if (provider === 'mistral-chat' || provider === 'xai-chat') {
    if (path.includes('/chat/completions')) return parsed.toString();
    return ensureEndpointPath('/v1/chat/completions').toString();
  }

  if (path.includes('/chat/completions')) return parsed.toString();
  return ensureEndpointPath('/v1/chat/completions').toString();
}

export function resolveCloudTranslateProvider(input: ResolveCloudTranslateProviderInput): ResolvedCloudTranslateProvider {
  const provider = detectCloudTranslateProvider(input.url, input.modelName);
  const effectiveModel = getEffectiveCloudTranslateModel(provider, input);
  const matchedProfile = matchProfile(input.modelName, input.model) || matchProfile(input.modelName, effectiveModel);
  return {
    provider,
    endpointUrl: buildCloudTranslateEndpointUrl(provider, input.url, effectiveModel, input.apiKey),
    effectiveModel,
    adapterKey: getCloudTranslateRuntimeAdapterKey(provider, input.modelName, input.model),
    profileId: matchedProfile?.id || null,
    profileFamily: matchedProfile?.family || null,
  };
}
