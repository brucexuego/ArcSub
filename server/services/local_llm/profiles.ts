import type { LocalModelDefinition } from '../../local_model_catalog.js';
import { getLocalOpenvinoOfficialBaselineForRepoId, hasExactLocalOpenvinoOfficialBaselineForRepoId } from './openvino_official_baselines.js';
import { inferLocalTranslateModelStrategy } from './strategy.js';
import type { LocalOpenvinoResolvedProfile } from './types.js';

function inferProfileFamily(localModel: LocalModelDefinition) {
  if (localModel.type === 'asr') {
    const repoId = String(localModel.repoId || '').toLowerCase();
    if (localModel.runtime === 'openvino-whisper-node') return 'whisper';
    if (localModel.runtime === 'openvino-ctc-asr') return 'ctc_asr';
    if (localModel.runtime === 'openvino-qwen3-asr') return 'qwen3_asr';
    if (localModel.runtime === 'openvino-cohere-asr') return 'cohere_asr';
    if (localModel.runtime === 'hf-transformers-asr') return 'hf_transformers_asr';
    if (repoId.includes('whisper')) return 'whisper';
    if (repoId.includes('qwen3-asr')) return 'qwen3_asr';
    return 'local-asr';
  }

  const strategy = inferLocalTranslateModelStrategy(localModel);
  return strategy.family;
}

export function resolveLocalOpenvinoProfile(localModel: LocalModelDefinition): LocalOpenvinoResolvedProfile {
  const baseline = getLocalOpenvinoOfficialBaselineForRepoId(localModel.repoId);
  return {
    baseline,
    profileId: baseline.id,
    profileFamily: inferProfileFamily(localModel),
    usedFallbackBaseline: !hasExactLocalOpenvinoOfficialBaselineForRepoId(localModel.repoId),
  };
}
