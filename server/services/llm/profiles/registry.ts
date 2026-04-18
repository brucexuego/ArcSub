import type { LlmModelProfile } from './base.js';
import { matchesLlmModelProfile } from './base.js';
import { PRESET_LLM_MODEL_PROFILES } from './preset_profiles.js';

export class LlmModelProfileRegistry {
  private readonly profiles: LlmModelProfile[] = [];

  register(profile: LlmModelProfile) {
    this.profiles.push(profile);
    return profile;
  }

  registerMany(profiles: LlmModelProfile[]) {
    profiles.forEach((profile) => this.register(profile));
    return profiles;
  }

  list() {
    return [...this.profiles];
  }

  match(modelName: string) {
    return this.profiles.find((profile) => matchesLlmModelProfile(profile, modelName)) || null;
  }
}

export const llmModelProfileRegistry = new LlmModelProfileRegistry();
llmModelProfileRegistry.registerMany(PRESET_LLM_MODEL_PROFILES);
