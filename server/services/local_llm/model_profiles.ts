import type { LocalModelDefinition } from '../../local_model_catalog.js';
import type { TranslationQualityMode } from '../llm/orchestrators/translation_quality_policy.js';
import { getLocalTranslationFamilyProfile, type LocalTranslationFamilyProfile, type LocalTranslationLineSafeBatchingProfile } from './translation_profiles.js';
import type { LocalTranslateGenerationOptions, LocalTranslateModelStrategy, LocalTranslatePromptStyle } from './types.js';

export interface LocalTranslationModelProfile {
  id: string;
  repoIds: string[];
  promptStyleOverrides?: Partial<Record<TranslationQualityMode, LocalTranslatePromptStyle>>;
  qualityModeDefaults?: Partial<Record<TranslationQualityMode, Partial<LocalTranslateGenerationOptions>>>;
  lineSafeBatching?: Partial<Record<TranslationQualityMode, LocalTranslationLineSafeBatchingProfile>>;
  notes?: string[];
}

export interface LocalTranslationResolvedProfile {
  familyProfile: LocalTranslationFamilyProfile;
  modelProfile: LocalTranslationModelProfile | null;
  effectivePromptStyle: LocalTranslatePromptStyle;
  qualityModeDefaults: Partial<LocalTranslateGenerationOptions>;
  lineSafeBatching: LocalTranslationLineSafeBatchingProfile | null;
}

const LOCAL_TRANSLATION_MODEL_PROFILES: LocalTranslationModelProfile[] = [
  {
    id: 'local-phi-2',
    repoIds: [
      'OpenVINO/phi-2-fp16-ov',
      'OpenVINO/phi-2-int4-ov',
      'OpenVINO/phi-2-int8-ov',
    ],
    promptStyleOverrides: {
      plain_probe: 'phi4_chat',
      template_validated: 'phi4_chat',
      json_strict: 'phi4_chat',
    },
    notes: ['Phi-2 models use a dedicated exact-model profile so later tuning does not affect Phi-3.x or Phi-4 variants.'],
  },
  {
    id: 'local-phi-3-mini',
    repoIds: [
      'OpenVINO/Phi-3-mini-4k-instruct-fp16-ov',
      'OpenVINO/Phi-3-mini-4k-instruct-int4-ov',
      'OpenVINO/Phi-3-mini-4k-instruct-int8-ov',
      'OpenVINO/Phi-3-mini-128k-instruct-fp16-ov',
      'OpenVINO/Phi-3-mini-128k-instruct-int4-ov',
      'OpenVINO/Phi-3-mini-128k-instruct-int8-ov',
    ],
    promptStyleOverrides: {
      plain_probe: 'phi4_chat',
      template_validated: 'phi4_chat',
      json_strict: 'phi4_chat',
    },
    lineSafeBatching: {
      template_validated: {
        maxLines: 12,
        charBudget: 720,
      },
      json_strict: {
        maxLines: 8,
        charBudget: 480,
      },
    },
    notes: ['Phi-3 mini variants share the exact same model-level prompt identity and conservative batching envelope.'],
  },
  {
    id: 'local-phi-3-medium',
    repoIds: [
      'OpenVINO/Phi-3-medium-4k-instruct-fp16-ov',
      'OpenVINO/Phi-3-medium-4k-instruct-int4-ov',
      'OpenVINO/Phi-3-medium-4k-instruct-int8-ov',
    ],
    promptStyleOverrides: {
      plain_probe: 'phi4_chat',
      template_validated: 'phi4_chat',
      json_strict: 'phi4_chat',
    },
    lineSafeBatching: {
      template_validated: {
        maxLines: 14,
        charBudget: 800,
      },
      json_strict: {
        maxLines: 10,
        charBudget: 560,
      },
    },
    notes: ['Phi-3 medium keeps its own exact-model profile so future batching changes stay model-scoped.'],
  },
  {
    id: 'local-phi-3-5-mini',
    repoIds: [
      'OpenVINO/Phi-3.5-mini-instruct-fp16-ov',
      'OpenVINO/Phi-3.5-mini-instruct-int4-ov',
      'OpenVINO/Phi-3.5-mini-instruct-int8-ov',
    ],
    promptStyleOverrides: {
      plain_probe: 'phi4_chat',
      template_validated: 'phi4_chat',
      json_strict: 'phi4_chat',
    },
    lineSafeBatching: {
      template_validated: {
        maxLines: 12,
        charBudget: 720,
      },
      json_strict: {
        maxLines: 8,
        charBudget: 480,
      },
    },
    notes: ['Phi-3.5 mini exact-model profile keeps Phi chat routing explicit instead of relying on generic fallback detection.'],
  },
  {
    id: 'local-phi-3-5-vision',
    repoIds: [
      'OpenVINO/Phi-3.5-vision-instruct-fp16-ov',
      'OpenVINO/Phi-3.5-vision-instruct-int4-ov',
      'OpenVINO/Phi-3.5-vision-instruct-int8-ov',
    ],
    promptStyleOverrides: {
      plain_probe: 'phi4_chat',
      template_validated: 'phi4_chat',
      json_strict: 'phi4_chat',
    },
    lineSafeBatching: {
      template_validated: {
        maxLines: 10,
        charBudget: 640,
      },
      json_strict: {
        maxLines: 8,
        charBudget: 480,
      },
    },
    notes: ['Phi-3.5 vision exact-model profile reserves a separate tuning lane from text-only Phi variants.'],
  },
  {
    id: 'local-phi-4',
    repoIds: [
      'OpenVINO/phi-4-fp16-ov',
      'OpenVINO/phi-4-int4-ov',
      'OpenVINO/phi-4-int8-ov',
    ],
    promptStyleOverrides: {
      plain_probe: 'phi4_chat',
      template_validated: 'phi4_chat',
      json_strict: 'phi4_chat',
    },
    notes: ['Phi-4 exact-model profile isolates future Phi-4 tuning from Phi-4-mini and earlier Phi variants.'],
  },
  {
    id: 'local-phi-4-mini',
    repoIds: [
      'OpenVINO/Phi-4-mini-instruct-fp16-ov',
      'OpenVINO/Phi-4-mini-instruct-int4-ov',
      'OpenVINO/Phi-4-mini-instruct-int8-ov',
    ],
    promptStyleOverrides: {
      plain_probe: 'phi4_chat',
      template_validated: 'phi4_chat',
      json_strict: 'phi4_chat',
    },
    lineSafeBatching: {
      template_validated: {
        maxLines: 12,
        charBudget: 720,
      },
      json_strict: {
        maxLines: 8,
        charBudget: 480,
      },
    },
    notes: ['Phi-4 mini exact-model profile keeps the currently stable mini variant independently tunable.'],
  },
  {
    id: 'local-qwen3-1-7b-int8',
    repoIds: ['OpenVINO/Qwen3-1.7B-int8-ov'],
    lineSafeBatching: {
      template_validated: {
        maxLines: 4,
        charBudget: 260,
      },
      json_strict: {
        maxLines: 3,
        charBudget: 220,
      },
    },
    notes: ['Qwen3 1.7B needs tighter validated/json batching than larger Qwen2.5 models, but it is less fragile than Qwen3 4B.'],
  },
  {
    id: 'local-qwen3-8b-int4',
    repoIds: ['OpenVINO/Qwen3-8B-int4-ov'],
    lineSafeBatching: {
      template_validated: {
        maxLines: 4,
        charBudget: 260,
      },
      json_strict: {
        maxLines: 3,
        charBudget: 220,
      },
    },
    notes: ['Qwen3 8B keeps its own validated/json batching so future tuning does not affect other Qwen3 variants.'],
  },
  {
    id: 'local-qwen2-5-1-5b-int4',
    repoIds: ['OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov'],
    qualityModeDefaults: {
      template_validated: {
        repetitionPenalty: 1.15,
        noRepeatNgramSize: 3,
      },
      json_strict: {
        repetitionPenalty: 1.15,
        noRepeatNgramSize: 3,
      },
    },
    lineSafeBatching: {
      template_validated: {
        maxLines: 6,
        charBudget: 360,
      },
      json_strict: {
        maxLines: 4,
        charBudget: 260,
      },
    },
    notes: ['Qwen2.5 1.5B needs tighter validated/json batching plus stronger repetition suppression to reduce marker loss and pass-through on subtitle batches.'],
  },
  {
    id: 'local-qwen2-5-7b-int4',
    repoIds: ['OpenVINO/Qwen2.5-7B-Instruct-int4-ov'],
    lineSafeBatching: {
      template_validated: {
        maxLines: 12,
        charBudget: 640,
      },
      json_strict: {
        maxLines: 8,
        charBudget: 480,
      },
    },
    notes: ['Qwen2.5 7B can keep slightly larger validated/json batches than the 1.5B model without collapsing line structure.'],
  },
  {
    id: 'local-qwen3-4b-int8',
    repoIds: ['OpenVINO/Qwen3-4B-int8-ov'],
    qualityModeDefaults: {
      template_validated: {
        noRepeatNgramSize: 3,
      },
      json_strict: {
        noRepeatNgramSize: 3,
      },
    },
    lineSafeBatching: {
      template_validated: {
        maxLines: 2,
        charBudget: 160,
      },
      json_strict: {
        maxLines: 1,
        charBudget: 120,
      },
    },
    notes: ['Qwen3 4B needs aggressive micro-batching and no-repeat control for subtitle translation to avoid line collapse and pass-through.'],
  },
  {
    id: 'local-translategemma-4b-it',
    repoIds: ['google/translategemma-4b-it'],
    promptStyleOverrides: {
      plain_probe: 'translategemma_google',
      template_validated: 'translategemma_google',
      json_strict: 'translategemma_google',
    },
    lineSafeBatching: {
      plain_probe: {
        maxLines: 4,
        charBudget: 320,
        tokenBudget: 960,
      },
      template_validated: {
        maxLines: 4,
        charBudget: 320,
        tokenBudget: 960,
      },
      json_strict: {
        maxLines: 2,
        charBudget: 160,
        tokenBudget: 640,
      },
    },
    notes: ['TranslateGemma uses structured translation messages instead of ArcSub generic system-prompt wrappers.', 'Subtitle translation should stay in small contiguous batches so line coverage can be validated externally.'],
  },
  {
    id: 'local-vllm-translategemma-4b-it',
    repoIds: ['Infomaniak-AI/vllm-translategemma-4b-it'],
    promptStyleOverrides: {
      plain_probe: 'translategemma_vllm',
      template_validated: 'translategemma_vllm',
      json_strict: 'translategemma_vllm',
    },
    lineSafeBatching: {
      plain_probe: {
        maxLines: 4,
        charBudget: 320,
        tokenBudget: 960,
      },
      template_validated: {
        maxLines: 4,
        charBudget: 320,
        tokenBudget: 960,
      },
      json_strict: {
        maxLines: 2,
        charBudget: 160,
        tokenBudget: 640,
      },
    },
    notes: ['The vLLM-optimized TranslateGemma recipe uses the delimiter-style chat payload documented by vLLM recipes.', 'Subtitle translation should stay in small contiguous batches so line coverage can be validated externally.'],
  },
  {
    id: 'local-gemma-3-4b-it',
    repoIds: ['OpenVINO/gemma-3-4b-it-int4-ov', 'OpenVINO/gemma-3-4b-it-int8-ov'],
    qualityModeDefaults: {
      template_validated: {
        topP: 0.75,
        repetitionPenalty: 1.08,
        noRepeatNgramSize: 3,
      },
      json_strict: {
        repetitionPenalty: 1.08,
        noRepeatNgramSize: 3,
      },
    },
    lineSafeBatching: {
      template_validated: {
        maxLines: 6,
        charBudget: 360,
      },
      json_strict: {
        maxLines: 5,
        charBudget: 340,
      },
    },
    notes: ['Gemma 3 4B responds best to its dedicated chat template plus tighter subtitle batches and mild repetition suppression.'],
  },
  {
    id: 'local-deepseek-r1-distill-qwen-7b-int4',
    repoIds: ['OpenVINO/DeepSeek-R1-Distill-Qwen-7B-int4-ov'],
    promptStyleOverrides: {
      template_validated: 'deepseek_r1_plain',
      json_strict: 'deepseek_r1_plain',
    },
    qualityModeDefaults: {
      template_validated: {
        repetitionPenalty: 1.12,
        noRepeatNgramSize: 3,
      },
      json_strict: {
        repetitionPenalty: 1.12,
        noRepeatNgramSize: 3,
      },
    },
    lineSafeBatching: {
      template_validated: {
        maxLines: 2,
        charBudget: 180,
      },
      json_strict: {
        maxLines: 2,
        charBudget: 150,
      },
    },
    notes: ['DeepSeek 7B int4 keeps validated/json modes on native non-thinking prompt wrappers plus per-model micro-batching.'],
  },
];

const MODEL_PROFILE_BY_REPO_ID = new Map<string, LocalTranslationModelProfile>();
for (const profile of LOCAL_TRANSLATION_MODEL_PROFILES) {
  for (const repoId of profile.repoIds) {
    MODEL_PROFILE_BY_REPO_ID.set(String(repoId || '').trim().toLowerCase(), profile);
  }
}

function normalizeRepoId(repoId: string | null | undefined) {
  return String(repoId || '').trim().toLowerCase();
}

function slugifyRepoId(repoId: string | null | undefined) {
  return normalizeRepoId(repoId)
    .replace(/^openvino\//, 'openvino-')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function buildAutoLocalTranslationModelProfile(localModel?: LocalModelDefinition | null): LocalTranslationModelProfile | null {
  if (!localModel?.repoId) return null;
  return {
    id: `auto-${slugifyRepoId(localModel.repoId)}`,
    repoIds: [localModel.repoId],
    notes: ['Auto-generated exact-model profile identity. Uses family defaults until model-specific overrides are added.'],
  };
}

function resolveEffectivePromptStyle(
  modelStrategy: LocalTranslateModelStrategy,
  modelProfile: LocalTranslationModelProfile | null,
  qualityMode: TranslationQualityMode
) {
  return modelProfile?.promptStyleOverrides?.[qualityMode] || modelStrategy.promptStyle;
}

function mergeGenerationDefaults(
  familyDefaults: Partial<LocalTranslateGenerationOptions>,
  modelDefaults?: Partial<LocalTranslateGenerationOptions>
) {
  return {
    ...familyDefaults,
    ...(modelDefaults || {}),
  };
}

function mergeLineSafeBatching(
  familyDefaults?: LocalTranslationLineSafeBatchingProfile,
  modelDefaults?: LocalTranslationLineSafeBatchingProfile
) {
  if (!familyDefaults && !modelDefaults) return null;
  return {
    ...(familyDefaults || {}),
    ...(modelDefaults || {}),
  } satisfies LocalTranslationLineSafeBatchingProfile;
}

export function listLocalTranslationModelProfiles() {
  return [...LOCAL_TRANSLATION_MODEL_PROFILES];
}

export function getLocalTranslationModelProfile(localModel?: LocalModelDefinition | null) {
  if (!localModel?.repoId) return null;
  return MODEL_PROFILE_BY_REPO_ID.get(normalizeRepoId(localModel.repoId)) || buildAutoLocalTranslationModelProfile(localModel);
}

export function resolveLocalTranslationProfile(
  localModel: LocalModelDefinition | null | undefined,
  modelStrategy: LocalTranslateModelStrategy,
  qualityMode: TranslationQualityMode
): LocalTranslationResolvedProfile {
  const familyProfile = getLocalTranslationFamilyProfile(modelStrategy.family);
  const modelProfile = getLocalTranslationModelProfile(localModel);
  return {
    familyProfile,
    modelProfile,
    effectivePromptStyle: resolveEffectivePromptStyle(modelStrategy, modelProfile, qualityMode),
    qualityModeDefaults: mergeGenerationDefaults(
      familyProfile.qualityModeDefaults[qualityMode] || {},
      modelProfile?.qualityModeDefaults?.[qualityMode]
    ),
    lineSafeBatching: mergeLineSafeBatching(
      familyProfile.lineSafeBatching?.[qualityMode],
      modelProfile?.lineSafeBatching?.[qualityMode]
    ),
  };
}
