export type CloudAsrProvider =
  | 'openai-whisper'
  | 'whispercpp-inference'
  | 'elevenlabs-scribe'
  | 'github-models-phi4-multimodal'
  | 'google-cloud-chirp3'
  | 'google-gemini-audio';

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
  const hostname = parsed.hostname.toLowerCase();
  const pathname = parsed.pathname.toLowerCase();
  const named = String(modelName || '').toLowerCase();

  if (
    /(^|\.)elevenlabs\.io$/.test(hostname) ||
    pathname.includes('/speech-to-text') ||
    named.includes('elevenlabs') ||
    named.includes('scribe')
  ) {
    return 'elevenlabs-scribe';
  }
  if (
    hostname === 'models.github.ai' ||
    pathname.includes('/inference/chat/completions') ||
    (named.includes('github') && (named.includes('phi-4') || named.includes('phi4')))
  ) {
    return 'github-models-phi4-multimodal';
  }
  if (
    hostname === 'speech.googleapis.com' ||
    hostname.endsWith('-speech.googleapis.com') ||
    pathname.includes('/recognizers/') ||
    (named.includes('google') && named.includes('chirp'))
  ) {
    return 'google-cloud-chirp3';
  }
  if (
    hostname === 'generativelanguage.googleapis.com' ||
    pathname.includes(':generatecontent') ||
    (named.includes('gemini') && (named.includes('asr') || named.includes('audio') || named.includes('transcri')))
  ) {
    return 'google-gemini-audio';
  }
  if (pathname.includes('/inference')) return 'whispercpp-inference';
  if (named.includes('whisper.cpp') || named.includes('whispercpp')) return 'whispercpp-inference';
  return 'openai-whisper';
}

export function buildCloudAsrEndpointUrl(provider: CloudAsrProvider, rawUrl: string, model?: string) {
  const parsed = parseUrl(rawUrl);
  const path = parsed.pathname.toLowerCase();
  const trimmedModel = String(model || '').trim();

  const buildGitHubModelsChatEndpoint = () => {
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
  };

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
    if (currentLower === '/v1beta' && targetLower.startsWith('/v1beta/')) {
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
  if (provider === 'elevenlabs-scribe') {
    if (path.includes('/speech-to-text')) return parsed.toString();
    return ensureEndpointPath('/v1/speech-to-text').toString();
  }
  if (provider === 'github-models-phi4-multimodal') {
    if (path.includes('/inference/chat/completions')) return parsed.toString();
    return buildGitHubModelsChatEndpoint();
  }
  if (provider === 'google-cloud-chirp3') {
    if (path.includes('/recognizers/') && path.endsWith(':recognize')) return parsed.toString();
    if (path.includes('/recognizers/')) {
      const next = new URL(parsed.toString());
      next.pathname = `${next.pathname.replace(/\/+$/, '')}:recognize`;
      return next.toString();
    }
    return parsed.toString();
  }
  if (provider === 'google-gemini-audio') {
    if (path.includes(':generatecontent')) return parsed.toString();
    const targetModel = trimmedModel || getDefaultCloudAsrModel('google-gemini-audio');
    return ensureEndpointPath(`/v1beta/models/${encodeURIComponent(targetModel)}:generateContent`).toString();
  }

  if (path.includes('/audio/transcriptions')) return parsed.toString();
  return ensureEndpointPath('/v1/audio/transcriptions').toString();
}

export function getDefaultCloudAsrModel(provider: CloudAsrProvider) {
  if (provider === 'google-gemini-audio') return 'gemini-2.5-flash';
  if (provider === 'google-cloud-chirp3') return 'chirp_3';
  if (provider === 'github-models-phi4-multimodal') return 'microsoft/Phi-4-multimodal-instruct';
  if (provider === 'elevenlabs-scribe') return 'scribe_v2';
  if (provider === 'whispercpp-inference') return 'whispercpp';
  return 'whisper-1';
}

export function resolveCloudAsrProvider(input: ResolveCloudAsrProviderInput): ResolvedCloudAsrProvider {
  const provider = detectCloudAsrProvider(input.url, input.modelName);
  const effectiveModel = String(input.model || '').trim() || getDefaultCloudAsrModel(provider);
  return {
    provider,
    endpointUrl: buildCloudAsrEndpointUrl(provider, input.url, effectiveModel),
    effectiveModel,
  };
}
