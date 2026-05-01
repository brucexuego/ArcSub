import type {
  CloudTranslateProvider,
  CloudTranslateProviderDefinition,
} from '../types.js';

export function parseCloudTranslateUrl(url: string) {
  return new URL(String(url || '').trim());
}

export function ensureEndpointPath(parsed: URL, targetPath: string) {
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
  if (currentLower === '/v1beta' && targetLower.startsWith('/v1beta/')) {
    next.pathname = target;
    return next;
  }
  if (
    (currentLower === '/v1/models' || currentLower === '/v1beta/models') &&
    (targetLower.startsWith('/v1/models/') || targetLower.startsWith('/v1beta/models/'))
  ) {
    next.pathname = target;
    return next;
  }
  if (currentLower === '/v2' && targetLower.startsWith('/v2/')) {
    next.pathname = target;
    return next;
  }

  next.pathname = `${current}/${target.slice(1)}`.replace(/\/{2,}/g, '/');
  return next;
}

export function createCloudTranslateProviderDefinition(input: CloudTranslateProviderDefinition) {
  return input;
}

export function createDefaultCapabilities(input: {
  provider: CloudTranslateProvider;
  profileId?: string;
  profileFamily?: string;
  supportsLargeContext?: boolean;
  supportsJsonObject?: boolean;
  supportsJsonSchema?: boolean;
  supportsStreaming?: boolean;
  supportsReasoningConfig?: boolean;
  supportsTopK?: boolean;
  translationNative?: boolean;
  defaultExecutionMode?: CloudTranslateProviderDefinition['capabilities']['defaultExecutionMode'];
}): CloudTranslateProviderDefinition['capabilities'] {
  return {
    supportsLargeContext: input.supportsLargeContext !== false,
    supportsJsonObject: input.supportsJsonObject !== false,
    supportsJsonSchema: input.supportsJsonSchema !== false,
    supportsStreaming: input.supportsStreaming === true,
    supportsReasoningConfig: input.supportsReasoningConfig === true,
    supportsTopK: input.supportsTopK === true,
    translationNative: input.translationNative === true,
    defaultExecutionMode: input.defaultExecutionMode || 'auto',
    profileId: input.profileId || input.provider,
    profileFamily: input.profileFamily || input.provider,
    nativeProviderLabel: input.provider,
  };
}
