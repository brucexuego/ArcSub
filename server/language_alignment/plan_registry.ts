import { LanguageAlignmentRegistry } from './registry.js';
import { AlignmentModelProfileRegistry } from './model_profiles.js';
import type { AlignmentModelProfile } from './model_profiles.js';
import type {
  ForcedAlignmentLanguageModule,
  ForcedAlignmentPlanConfig,
  ForcedAlignmentSelectionContext,
} from './shared/types.js';

export type ResolvedAlignmentPlan = {
  language: string | null;
  sampleText: string;
  variant: string | null;
  module: ForcedAlignmentLanguageModule | null;
  config: ForcedAlignmentPlanConfig;
  profile: AlignmentModelProfile;
  mode: 'native' | 'ctc';
};

export class AlignmentPlanRegistry {
  static resolve(context: ForcedAlignmentSelectionContext): ResolvedAlignmentPlan | null {
    const sampleText = String(context.sampleText || '').trim();
    const nativeProfile = AlignmentModelProfileRegistry.resolveNativeOverride(context);
    if (nativeProfile) {
      return {
        language: String(
          context.providerMeta?.forcedAlignment?.language || context.language || ''
        ).trim() || null,
        sampleText,
        variant: null,
        module: null,
        config: {
          profileId: nativeProfile.id,
          clipPaddingSec: 0,
          minOverallConfidence: 0,
          minOverallAlignedRatio: 0,
          progressLabel: `${nativeProfile.modelId} native forced alignment`,
        },
        profile: nativeProfile,
        mode: 'native',
      };
    }

    const languageModule = LanguageAlignmentRegistry.getForcedAlignmentModule(context.language, sampleText);
    if (!languageModule) {
      return null;
    }

    const config = languageModule.getPlanConfig(context.language, sampleText, context);
    const profile = AlignmentModelProfileRegistry.require(config.profileId);
    return {
      language: String(context.language || '').trim().toLowerCase() || null,
      sampleText,
      variant: languageModule.resolveVariant?.(context.language, sampleText) || null,
      module: languageModule,
      config,
      profile,
      mode: profile.kind === 'native' ? 'native' : 'ctc',
    };
  }
}

