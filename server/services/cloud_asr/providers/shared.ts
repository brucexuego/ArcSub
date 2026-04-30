import type { CloudAsrProvider, CloudAsrProviderDefinition } from '../types.js';
import { getCloudAsrProviderCapabilities } from '../profiles/capabilities.js';

export function parseCloudAsrUrl(url: string) {
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

  next.pathname = `${current}/${target.slice(1)}`.replace(/\/{2,}/g, '/');
  return next;
}

export function buildBearerHeaders(key?: string): Record<string, string> {
  const trimmed = String(key || '').trim();
  return trimmed ? { Authorization: `Bearer ${trimmed}` } : {};
}

export function createCloudAsrProviderDefinition(input: Omit<CloudAsrProviderDefinition, 'capabilities'> & {
  provider: CloudAsrProvider;
  capabilities?: CloudAsrProviderDefinition['capabilities'];
}): CloudAsrProviderDefinition {
  return {
    ...input,
    capabilities: input.capabilities || getCloudAsrProviderCapabilities(input.provider),
  };
}
