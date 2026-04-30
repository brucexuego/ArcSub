import type { CloudAsrProvider } from '../types.js';

export interface CloudAsrTimeoutEnv {
  getNumber(name: string, fallback: number, min?: number, max?: number): number;
}

export function resolveCloudAsrRequestTimeoutMs(
  provider: CloudAsrProvider,
  options: { audioDurationSec?: number | null },
  env: CloudAsrTimeoutEnv
) {
  const sharedTimeoutMs = Math.round(env.getNumber('ASR_CLOUD_REQUEST_TIMEOUT_MS', 120000, 15000, 1800000));

  if (provider === 'deepgram') {
    const deepgramFloorMs = Math.round(env.getNumber('ASR_DEEPGRAM_REQUEST_TIMEOUT_MS', 600000, 60000, 1200000));
    return Math.max(sharedTimeoutMs, deepgramFloorMs);
  }

  if (provider === 'gladia') {
    const gladiaFloorMs = Math.round(env.getNumber('ASR_GLADIA_REQUEST_TIMEOUT_MS', 900000, 60000, 7200000));
    return Math.max(sharedTimeoutMs, gladiaFloorMs);
  }

  if (provider !== 'elevenlabs') {
    return sharedTimeoutMs;
  }

  const audioDurationSec = Number(options.audioDurationSec);
  const estimatedTimeoutMs = Number.isFinite(audioDurationSec) && audioDurationSec > 0
    ? Math.ceil(audioDurationSec * 1000 * 2.2 + 120000)
    : 1800000;
  const elevenLabsFloorMs = Math.round(env.getNumber('ASR_ELEVENLABS_REQUEST_TIMEOUT_MS', 1800000, 60000, 7200000));
  return Math.round(Math.max(sharedTimeoutMs, elevenLabsFloorMs, Math.min(estimatedTimeoutMs, 7200000)));
}
