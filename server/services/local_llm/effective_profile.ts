import type { LocalModelDefinition } from '../../local_model_catalog.js';
import {
  resolveTranslationQualityIntent,
  type TranslationQualityIntent,
  type TranslationQualityMode,
} from '../llm/orchestrators/translation_quality_policy.js';
import { resolveLocalTranslationProfile, type LocalTranslationResolvedProfile } from './model_profiles.js';
import type { LocalTranslateModelStrategy } from './types.js';

export interface LocalTranslationIntentInput {
  promptTemplateId?: string | null;
  prompt?: string | null;
  enableJsonLineRepair?: boolean;
  modelStrategy?: LocalTranslateModelStrategy | null;
  qualityMode?: TranslationQualityMode;
}

export interface EffectiveLocalTranslationProfile {
  intent: TranslationQualityIntent;
  translationProfile: LocalTranslationResolvedProfile;
  sourcePrecedence: string[];
  runtimeHintsApplied: boolean;
}

function overrideIntentQualityMode(
  intent: TranslationQualityIntent,
  qualityMode: TranslationQualityMode
): TranslationQualityIntent {
  return {
    ...intent,
    qualityMode,
    minimalQualityChecks: qualityMode === 'plain_probe',
  };
}

export function resolveLocalTranslationIntent(input: LocalTranslationIntentInput): TranslationQualityIntent {
  const baseIntent = resolveTranslationQualityIntent({
    promptTemplateId: input.promptTemplateId,
    prompt: input.prompt,
    enableJsonLineRepair: input.enableJsonLineRepair,
  });

  if (input.qualityMode) {
    return overrideIntentQualityMode(baseIntent, input.qualityMode);
  }

  if (input.modelStrategy?.family === 'translategemma') {
    if (baseIntent.usesTemplatePrompt || baseIntent.usesJsonRepair) {
      return overrideIntentQualityMode(baseIntent, 'template_validated');
    }
    return overrideIntentQualityMode(baseIntent, 'plain_probe');
  }

  return baseIntent;
}

export function resolveEffectiveLocalTranslationProfile(input: {
  localModel?: LocalModelDefinition | null;
  modelStrategy: LocalTranslateModelStrategy;
  promptTemplateId?: string | null;
  prompt?: string | null;
  enableJsonLineRepair?: boolean;
  qualityMode?: TranslationQualityMode;
}): EffectiveLocalTranslationProfile {
  const intent = resolveLocalTranslationIntent({
    promptTemplateId: input.promptTemplateId,
    prompt: input.prompt,
    enableJsonLineRepair: input.enableJsonLineRepair,
    modelStrategy: input.modelStrategy,
    qualityMode: input.qualityMode,
  });
  const translationProfile = resolveLocalTranslationProfile(input.localModel, input.modelStrategy, intent.qualityMode);

  return {
    intent,
    translationProfile,
    sourcePrecedence: ['family_defaults', 'exact_profile', 'hf_runtime_hints', 'env_or_ui_override'],
    runtimeHintsApplied: Boolean(input.localModel?.runtimeHints),
  };
}
