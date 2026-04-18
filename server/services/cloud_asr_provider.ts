export type CloudAsrProvider = 'openai-whisper' | 'whispercpp-inference';

export interface ResolveCloudAsrProviderInput {
  url: string;
  modelName?: string;
  model?: string;
}

export interface ResolvedCloudAsrProvider {
  provider: CloudAsrProvider;
  endpointUrl: string;
  effectiveModel: string;
}

function parseUrl(url: string) {
  return new URL(String(url || '').trim());
}

export function detectCloudAsrProvider(url: string, modelName?: string): CloudAsrProvider {
  const parsed = parseUrl(url);
  const pathname = parsed.pathname.toLowerCase();
  const named = String(modelName || '').toLowerCase();

  if (pathname.includes('/inference')) return 'whispercpp-inference';
  if (named.includes('whisper.cpp') || named.includes('whispercpp')) return 'whispercpp-inference';
  return 'openai-whisper';
}

export function buildCloudAsrEndpointUrl(provider: CloudAsrProvider, rawUrl: string) {
  const parsed = parseUrl(rawUrl);
  const path = parsed.pathname.toLowerCase();

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

    next.pathname = `${current}/${target.slice(1)}`.replace(/\/{2,}/g, '/');
    return next;
  };

  if (provider === 'whispercpp-inference') {
    if (path.endsWith('/inference')) return parsed.toString();
    return ensureEndpointPath('/inference').toString();
  }

  if (path.includes('/audio/transcriptions')) return parsed.toString();
  return ensureEndpointPath('/v1/audio/transcriptions').toString();
}

export function getDefaultCloudAsrModel(provider: CloudAsrProvider) {
  if (provider === 'whispercpp-inference') return 'whispercpp';
  return 'whisper-1';
}

export function resolveCloudAsrProvider(input: ResolveCloudAsrProviderInput): ResolvedCloudAsrProvider {
  const provider = detectCloudAsrProvider(input.url, input.modelName);
  return {
    provider,
    endpointUrl: buildCloudAsrEndpointUrl(provider, input.url),
    effectiveModel: String(input.model || '').trim() || getDefaultCloudAsrModel(provider),
  };
}
