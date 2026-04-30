import type { CloudAsrChunkingPolicy, CloudAsrProvider, ResolvedCloudAsrProvider } from '../types.js';

export interface CloudAsrChunkingEnv {
  getNumber(name: string, fallback: number, min?: number, max?: number): number;
  getString(name: string, fallback: string): string;
  sanitizeAudioBitrate(value: string, fallback?: string): string;
}

type CloudAsrChunkingPolicyInput = CloudAsrProvider | Pick<ResolvedCloudAsrProvider, 'provider' | 'effectiveModel' | 'profileId'>;

function normalizeChunkingInput(input: CloudAsrChunkingPolicyInput) {
  if (typeof input === 'string') {
    return {
      provider: input,
      effectiveModel: '',
      profileId: null,
    };
  }
  return {
    provider: input.provider,
    effectiveModel: String(input.effectiveModel || '').trim().toLowerCase(),
    profileId: input.profileId || null,
  };
}

function matchesGithubPhi4Multimodal(input: ReturnType<typeof normalizeChunkingInput>) {
  const model = input.effectiveModel;
  return input.profileId === 'github-phi4-multimodal' ||
    /(^|[/:\s_-])phi[-_\s]?4(?:[-_\s]multimodal)?($|[/:\s_-])/i.test(model) ||
    model.includes('phi-4-multimodal-instruct');
}

export function resolveCloudAsrChunkingPolicy(
  input: CloudAsrChunkingPolicyInput,
  env: CloudAsrChunkingEnv
): CloudAsrChunkingPolicy | null {
  const normalized = normalizeChunkingInput(input);
  const { provider } = normalized;

  if (provider === 'github-models' && matchesGithubPhi4Multimodal(normalized)) {
    const initialChunkSec = Math.round(env.getNumber('ASR_PHI4_FILE_LIMIT_CHUNK_SEC', 5, 2, 60));
    const minChunkSec = Math.round(env.getNumber('ASR_PHI4_FILE_LIMIT_MIN_CHUNK_SEC', 2, 1, 30));
    return {
      initialChunkSec,
      minChunkSec: Math.min(initialChunkSec, minChunkSec),
      maxChunks: Math.round(env.getNumber('ASR_PHI4_FILE_LIMIT_MAX_CHUNKS', 2000, 1, 10000)),
      audioFormat: 'mp3',
      audioBitrate: env.sanitizeAudioBitrate(env.getString('ASR_PHI4_AUDIO_BITRATE', '16k'), '16k'),
      reason: 'github_phi4_8000_token_limit',
      profileId: 'github-phi4-multimodal',
      modelId: normalized.effectiveModel || null,
    };
  }

  if (provider === 'deepgram') {
    const initialChunkSec = Math.round(env.getNumber('ASR_DEEPGRAM_FILE_LIMIT_CHUNK_SEC', 480, 30, 1800));
    const minChunkSec = Math.round(env.getNumber('ASR_DEEPGRAM_FILE_LIMIT_MIN_CHUNK_SEC', 60, 10, 600));
    return {
      initialChunkSec,
      minChunkSec: Math.min(initialChunkSec, minChunkSec),
      maxChunks: Math.round(env.getNumber('ASR_DEEPGRAM_FILE_LIMIT_MAX_CHUNKS', 200, 1, 2000)),
      audioFormat: 'mp3',
      audioBitrate: env.sanitizeAudioBitrate(env.getString('ASR_DEEPGRAM_AUDIO_BITRATE', '32k'), '32k'),
      reason: 'deepgram_upload_or_processing_limit',
      profileId: 'deepgram-listen',
      modelId: normalized.effectiveModel || null,
    };
  }

  if (provider === 'gladia') {
    const initialChunkSec = Math.round(env.getNumber('ASR_GLADIA_FILE_LIMIT_CHUNK_SEC', 3600, 600, 8100));
    const minChunkSec = Math.round(env.getNumber('ASR_GLADIA_FILE_LIMIT_MIN_CHUNK_SEC', 600, 60, 1800));
    return {
      initialChunkSec,
      minChunkSec: Math.min(initialChunkSec, minChunkSec),
      maxChunks: Math.round(env.getNumber('ASR_GLADIA_FILE_LIMIT_MAX_CHUNKS', 200, 1, 2000)),
      audioFormat: 'mp3',
      audioBitrate: env.sanitizeAudioBitrate(env.getString('ASR_GLADIA_AUDIO_BITRATE', '32k'), '32k'),
      reason: 'gladia_upload_or_duration_limit',
      profileId: 'gladia-pre-recorded',
      modelId: normalized.effectiveModel || null,
    };
  }

  return null;
}
