import type { LocalModelDefinition } from '../../local_model_catalog.js';

export interface LocalAsrModelProfile {
  id: string;
  repoIds: string[];
  notes?: string[];
}

const LOCAL_ASR_MODEL_PROFILES: LocalAsrModelProfile[] = [
  {
    id: 'local-whisper-large-v3-int8',
    repoIds: ['OpenVINO/whisper-large-v3-int8-ov'],
  },
  {
    id: 'local-whisper-large-v3-int4',
    repoIds: ['OpenVINO/whisper-large-v3-int4-ov'],
  },
  {
    id: 'local-whisper-medium-int8',
    repoIds: ['OpenVINO/whisper-medium-int8-ov'],
  },
  {
    id: 'local-whisper-medium-int4',
    repoIds: ['OpenVINO/whisper-medium-int4-ov'],
  },
  {
    id: 'local-whisper-medium-fp16',
    repoIds: ['OpenVINO/whisper-medium-fp16-ov'],
  },
  {
    id: 'local-qwen3-asr-0-6b',
    repoIds: ['Qwen/Qwen3-ASR-0.6B'],
  },
  {
    id: 'local-qwen3-asr-1-7b',
    repoIds: ['Qwen/Qwen3-ASR-1.7B'],
  },
  {
    id: 'local-breeze-asr-25',
    repoIds: ['MediaTek-Research/Breeze-ASR-25'],
    notes: ['Breeze ASR 25 is fine-tuned from Whisper large-v2; uses the local Whisper OpenVINO runtime path.'],
  },
  {
    id: 'local-reazon-japanese-wav2vec2-large-rs35kh',
    repoIds: ['reazon-research/japanese-wav2vec2-large-rs35kh'],
    notes: [
      'Japanese CTC ASR model fine-tuned on ReazonSpeech; uses the OpenVINO CTC ASR runtime path after export.',
      'This is an ASR transcription model, not the default Japanese forced-alignment profile.',
    ],
  },
  {
    id: 'local-cohere-transcribe-03-2026',
    repoIds: ['CohereLabs/cohere-transcribe-03-2026'],
    notes: [
      'Cohere Transcribe is a gated 2B ASR model with 14 supported languages; ArcSub exports it to a Cohere-specific OpenVINO INT8 runtime.',
      'It requires an explicit source language and does not provide native timestamps or diarization.',
    ],
  },
];

const MODEL_PROFILE_BY_REPO_ID = new Map<string, LocalAsrModelProfile>();
for (const profile of LOCAL_ASR_MODEL_PROFILES) {
  for (const repoId of profile.repoIds) {
    MODEL_PROFILE_BY_REPO_ID.set(String(repoId || '').trim().toLowerCase(), profile);
  }
}

function normalizeRepoId(repoId: string | null | undefined) {
  return String(repoId || '').trim().toLowerCase();
}

function slugifyRepoId(repoId: string | null | undefined) {
  return normalizeRepoId(repoId)
    .replace(/^(openvino|qwen)\//, '$1-')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function buildAutoLocalAsrModelProfile(localModel?: LocalModelDefinition | null): LocalAsrModelProfile | null {
  if (!localModel?.repoId) return null;
  return {
    id: `auto-${slugifyRepoId(localModel.repoId)}`,
    repoIds: [localModel.repoId],
    notes: ['Auto-generated exact-model ASR profile identity. Uses runtime/profile defaults until model-specific ASR overrides are added.'],
  };
}

export function listLocalAsrModelProfiles() {
  return [...LOCAL_ASR_MODEL_PROFILES];
}

export function getLocalAsrModelProfile(localModel?: LocalModelDefinition | null) {
  if (!localModel?.repoId) return null;
  return MODEL_PROFILE_BY_REPO_ID.get(normalizeRepoId(localModel.repoId)) || buildAutoLocalAsrModelProfile(localModel);
}
