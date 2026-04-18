import type { LocalModelDefinition } from '../../local_model_catalog.js';
import { resolveLocalOpenvinoProfile } from '../local_llm/profiles.js';
import { getLocalAsrModelProfile, type LocalAsrModelProfile } from './model_profiles.js';

export interface LocalResolvedAsrProfile {
  baseProfile: ReturnType<typeof resolveLocalOpenvinoProfile>;
  modelProfile: LocalAsrModelProfile | null;
}

export function resolveLocalAsrProfile(localModel: LocalModelDefinition) {
  return {
    baseProfile: resolveLocalOpenvinoProfile(localModel),
    modelProfile: getLocalAsrModelProfile(localModel),
  } satisfies LocalResolvedAsrProfile;
}
