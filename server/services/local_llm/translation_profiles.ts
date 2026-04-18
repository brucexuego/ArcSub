import type { TranslationQualityMode } from '../llm/orchestrators/translation_quality_policy.js';
import type { LocalTranslateGenerationOptions, LocalTranslateModelStrategy } from './types.js';

type LocalTranslateFamily = LocalTranslateModelStrategy['family'];

export interface LocalTranslationLineSafeBatchingProfile {
  maxLines?: number;
  charBudget?: number;
}

export interface LocalTranslationFamilyProfile {
  id: string;
  family: LocalTranslateFamily;
  qualityModeDefaults: Record<TranslationQualityMode, Partial<LocalTranslateGenerationOptions>>;
  lineSafeBatching?: Partial<Record<TranslationQualityMode, LocalTranslationLineSafeBatchingProfile>>;
  notes?: string[];
}

const LOCAL_TRANSLATION_FAMILY_PROFILES: LocalTranslationFamilyProfile[] = [
  {
    id: 'local-generic',
    family: 'generic',
    qualityModeDefaults: {
      plain_probe: {
        doSample: false,
        temperature: 0.2,
        topP: 0.95,
      },
      template_validated: {
        doSample: false,
        temperature: 0.1,
        topP: 0.9,
      },
      json_strict: {
        doSample: false,
        temperature: 0,
        topP: 1,
        topK: 1,
        minP: 0,
      },
    },
    lineSafeBatching: {
      plain_probe: {
        maxLines: 24,
        charBudget: 1200,
      },
      template_validated: {
        maxLines: 18,
        charBudget: 900,
      },
      json_strict: {
        maxLines: 12,
        charBudget: 640,
      },
    },
    notes: ['Generic local profile favors deterministic translation output over chat-style variety.'],
  },
  {
    id: 'local-seq2seq',
    family: 'seq2seq',
    qualityModeDefaults: {
      plain_probe: {
        doSample: false,
        temperature: 0,
        topP: 1,
      },
      template_validated: {
        doSample: false,
        temperature: 0,
        topP: 1,
      },
      json_strict: {
        doSample: false,
        temperature: 0,
        topP: 1,
        topK: 1,
      },
    },
    lineSafeBatching: {
      plain_probe: {
        maxLines: 8,
        charBudget: 360,
      },
      template_validated: {
        maxLines: 6,
        charBudget: 280,
      },
      json_strict: {
        maxLines: 4,
        charBudget: 220,
      },
    },
    notes: ['Seq2Seq encoder-decoder models need compact instruction prompts and smaller batches than causal chat models.'],
  },
  {
    id: 'local-qwen2-5',
    family: 'qwen2_5',
    qualityModeDefaults: {
      plain_probe: {
        doSample: true,
        temperature: 0.7,
        topP: 0.8,
        topK: 20,
      },
      template_validated: {
        doSample: false,
        temperature: 0,
        topP: 1,
        topK: 1,
        minP: 0,
      },
      json_strict: {
        doSample: false,
        temperature: 0,
        topP: 1,
        topK: 1,
        minP: 0,
      },
    },
    lineSafeBatching: {
      plain_probe: {
        maxLines: 18,
        charBudget: 1000,
      },
      template_validated: {
        maxLines: 10,
        charBudget: 560,
      },
      json_strict: {
        maxLines: 8,
        charBudget: 480,
      },
    },
    notes: ['Qwen2.5 uses official OpenVINO chat defaults for plain probe and deterministic overrides for validated modes.'],
  },
  {
    id: 'local-qwen3',
    family: 'qwen3',
    qualityModeDefaults: {
      plain_probe: {
        doSample: true,
        temperature: 0.6,
        topP: 0.95,
        topK: 20,
      },
      template_validated: {
        doSample: false,
        temperature: 0,
        topP: 1,
        topK: 1,
        minP: 0,
        presencePenalty: 0,
      },
      json_strict: {
        doSample: false,
        temperature: 0,
        topP: 1,
        topK: 1,
        minP: 0,
        presencePenalty: 0,
      },
    },
    lineSafeBatching: {
      plain_probe: {
        maxLines: 12,
        charBudget: 720,
      },
      template_validated: {
        maxLines: 6,
        charBudget: 360,
      },
      json_strict: {
        maxLines: 4,
        charBudget: 240,
      },
    },
    notes: ['Qwen3 plain probe keeps official sampling; validated modes switch to deterministic output to reduce line collapse and pass-through.'],
  },
  {
    id: 'local-deepseek-r1-distill-qwen',
    family: 'deepseek_r1_distill_qwen',
    qualityModeDefaults: {
      plain_probe: {
        doSample: true,
        temperature: 0.6,
        topP: 0.95,
        topK: 50,
      },
      template_validated: {
        doSample: false,
        temperature: 0,
        topP: 1,
        topK: 1,
      },
      json_strict: {
        doSample: false,
        temperature: 0,
        topP: 1,
        topK: 1,
      },
    },
    lineSafeBatching: {
      plain_probe: {
        maxLines: 10,
        charBudget: 720,
      },
      template_validated: {
        maxLines: 6,
        charBudget: 360,
      },
      json_strict: {
        maxLines: 4,
        charBudget: 240,
      },
    },
    notes: ['DeepSeek R1 distill profile disables reasoning-style variance in validated translation modes.'],
  },
  {
    id: 'local-phi4',
    family: 'phi4',
    qualityModeDefaults: {
      plain_probe: {
        doSample: false,
        temperature: 0.2,
        topP: 0.95,
      },
      template_validated: {
        doSample: false,
        temperature: 0.1,
        topP: 0.9,
      },
      json_strict: {
        doSample: false,
        temperature: 0,
        topP: 1,
        topK: 1,
      },
    },
    lineSafeBatching: {
      plain_probe: {
        maxLines: 20,
        charBudget: 1000,
      },
      template_validated: {
        maxLines: 12,
        charBudget: 720,
      },
      json_strict: {
        maxLines: 8,
        charBudget: 480,
      },
    },
    notes: ['Phi family stays conservative across all translation modes.'],
  },
  {
    id: 'local-gemma3',
    family: 'gemma3',
    qualityModeDefaults: {
      plain_probe: {
        doSample: false,
        temperature: 0.2,
        topP: 0.9,
      },
      template_validated: {
        doSample: false,
        temperature: 0.1,
        topP: 0.85,
      },
      json_strict: {
        doSample: false,
        temperature: 0,
        topP: 1,
        topK: 1,
      },
    },
    lineSafeBatching: {
      plain_probe: {
        maxLines: 16,
        charBudget: 900,
      },
      template_validated: {
        maxLines: 10,
        charBudget: 600,
      },
      json_strict: {
        maxLines: 8,
        charBudget: 480,
      },
    },
    notes: ['Gemma 3 is treated as a conservative VLM-capable local translator, not a high-variance chat model.'],
  },
];

const PROFILE_BY_FAMILY = new Map(LOCAL_TRANSLATION_FAMILY_PROFILES.map((profile) => [profile.family, profile]));

export function listLocalTranslationFamilyProfiles() {
  return [...LOCAL_TRANSLATION_FAMILY_PROFILES];
}

export function getLocalTranslationFamilyProfile(family: LocalTranslateFamily) {
  return PROFILE_BY_FAMILY.get(family) || PROFILE_BY_FAMILY.get('generic')!;
}
