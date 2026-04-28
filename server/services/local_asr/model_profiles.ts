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
    id: 'local-breeze-asr-26',
    repoIds: ['MediaTek-Research/Breeze-ASR-26'],
    notes: ['Breeze Taigi ASR fine-tuned from Whisper large-v2; uses the local Whisper OpenVINO runtime path.'],
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
