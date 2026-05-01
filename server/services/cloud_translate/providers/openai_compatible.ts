import { resolveCloudTranslateQuotaProfile } from '../profiles/quotas.js';
import type { CloudTranslateProviderDefinition } from '../types.js';
import {
  createCloudTranslateProviderDefinition,
  createDefaultCapabilities,
  ensureEndpointPath,
  parseCloudTranslateUrl,
} from './shared.js';

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

export const openAiCompatibleTranslateProvider: CloudTranslateProviderDefinition =
  createCloudTranslateProviderDefinition({
    provider: 'openai-compatible',
    defaultModel: 'gpt-4o-mini',
    adapterKey: 'openai-compatible-chat',
    capabilities: createDefaultCapabilities({
      provider: 'openai-compatible',
      profileId: 'openai-compatible-chat',
      profileFamily: 'openai-compatible',
      supportsJsonSchema: false,
      supportsStreaming: true,
      defaultExecutionMode: 'cloud_relaxed',
    }),
    detect(input) {
      const host = input.hostname;
      const pathname = input.pathname;
      if (host === 'models.github.ai') return true;
      if (host === 'integrate.api.nvidia.com') return true;
      if (host.includes('generativelanguage.googleapis.com') && pathname.includes('/openai/')) return true;
      return false;
    },
    buildEndpointUrl(rawUrl) {
      const parsed = parseCloudTranslateUrl(rawUrl);
      if (isGitHubModelsHost(parsed)) return buildGitHubModelsChatEndpoint(parsed);
      if (parsed.pathname.toLowerCase().includes('/chat/completions')) return parsed.toString();
      return ensureEndpointPath(parsed, '/v1/chat/completions').toString();
    },
    getQuotaProfile(input) {
      return resolveCloudTranslateQuotaProfile({
        provider: 'openai-compatible',
        model: input.model,
        options: input.options,
        defaults: {
          tokenEstimator: 'openai_like',
          safetyReserveTokens: 256,
        },
      });
    },
  });
