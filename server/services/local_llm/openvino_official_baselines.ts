import type { LocalOpenvinoOfficialBaseline } from './types.js';

type RuntimeFamily = LocalOpenvinoOfficialBaseline['runtimeFamily'];
type TaskFamily = LocalOpenvinoOfficialBaseline['taskFamily'];

type RepoSeed = {
  repoId: string;
  runtimeFamily: RuntimeFamily;
  taskFamily: TaskFamily;
};

const CUSTOM_REPO_SEEDS: RepoSeed[] = [
  {
    repoId: 'MediaTek-Research/Breeze-ASR-26',
    runtimeFamily: 'openvino-whisper-node',
    taskFamily: 'asr',
  },
];

const LLM_REPO_IDS = [
  'OpenVINO/codegen25-7b-multi-fp16-ov',
  'OpenVINO/codegen-6B-multi-fp16-ov',
  'OpenVINO/codegen-6B-multi-int4-ov',
  'OpenVINO/codegen-6B-multi-int8-ov',
  'OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-fp16-ov',
  'OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov',
  'OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int8-ov',
  'OpenVINO/DeepSeek-R1-Distill-Qwen-14B-fp16-ov',
  'OpenVINO/DeepSeek-R1-Distill-Qwen-14B-int4-ov',
  'OpenVINO/DeepSeek-R1-Distill-Qwen-14B-int8-ov',
  'OpenVINO/DeepSeek-R1-Distill-Qwen-7B-fp16-ov',
  'OpenVINO/DeepSeek-R1-Distill-Qwen-7B-int4-ov',
  'OpenVINO/DeepSeek-R1-Distill-Qwen-7B-int8-ov',
  'OpenVINO/dolly-v2-12b-int8-ov',
  'OpenVINO/dolly-v2-3b-fp16-ov',
  'OpenVINO/dolly-v2-3b-int4-ov',
  'OpenVINO/dolly-v2-3b-int8-ov',
  'OpenVINO/dolly-v2-7b-fp16-ov',
  'OpenVINO/falcon-7b-instruct-fp16-ov',
  'OpenVINO/falcon-7b-instruct-int4-ov',
  'OpenVINO/falcon-7b-instruct-int8-ov',
  'OpenVINO/gemma-2-9b-it-fp16-ov',
  'OpenVINO/gemma-2-9b-it-int4-ov',
  'OpenVINO/gemma-2-9b-it-int8-ov',
  'OpenVINO/gemma-2b-it-fp16-ov',
  'OpenVINO/gemma-2b-it-int4-ov',
  'OpenVINO/gemma-2b-it-int8-ov',
  'OpenVINO/gemma-7b-fp16-ov',
  'OpenVINO/gemma-7b-int4-ov',
  'OpenVINO/gemma-7b-int8-ov',
  'OpenVINO/gemma-7b-it-fp16-ov',
  'OpenVINO/gemma-7b-it-int4-ov',
  'OpenVINO/gemma-7b-it-int8-ov',
  'OpenVINO/gpt-j-6b-fp16-ov',
  'OpenVINO/gpt-j-6b-int4-ov',
  'OpenVINO/gpt-j-6b-int8-ov',
  'OpenVINO/gpt-neox-20b-fp16-ov',
  'OpenVINO/gpt-neox-20b-int8-ov',
  'OpenVINO/gpt-oss-20b-int4-ov',
  'OpenVINO/gpt-oss-20b-int8-ov',
  'OpenVINO/LFM2.5-350M-fp16-ov',
  'OpenVINO/LFM2.5-350M-int8-ov',
  'OpenVINO/mistral-7b-instruct-v0.1-fp16-ov',
  'OpenVINO/mistral-7b-instruct-v0.1-int4-ov',
  'OpenVINO/mistral-7b-instruct-v0.1-int8-ov',
  'OpenVINO/Mistral-7B-Instruct-v0.2-fp16-ov',
  'OpenVINO/Mistral-7B-Instruct-v0.2-int4-ov',
  'OpenVINO/Mistral-7B-Instruct-v0.2-int8-ov',
  'OpenVINO/mixtral-8x7b-instruct-v0.1-int4-ov',
  'OpenVINO/Mixtral-8x7B-Instruct-v0.1-int8-ov',
  'OpenVINO/neural-chat-7b-v3-3-fp16-ov',
  'OpenVINO/neural-chat-7b-v3-3-int4-ov',
  'OpenVINO/neural-chat-7b-v3-3-int8-ov',
  'OpenVINO/persimmon-8b-chat-fp16-ov',
  'OpenVINO/persimmon-8b-chat-int4-ov',
  'OpenVINO/persimmon-8b-chat-int8-ov',
  'OpenVINO/phi-2-fp16-ov',
  'OpenVINO/phi-2-int4-ov',
  'OpenVINO/phi-2-int8-ov',
  'OpenVINO/Phi-3.5-mini-instruct-fp16-ov',
  'OpenVINO/Phi-3.5-mini-instruct-int4-ov',
  'OpenVINO/Phi-3.5-mini-instruct-int8-ov',
  'OpenVINO/Phi-3-medium-4k-instruct-fp16-ov',
  'OpenVINO/Phi-3-medium-4k-instruct-int4-ov',
  'OpenVINO/Phi-3-medium-4k-instruct-int8-ov',
  'OpenVINO/Phi-3-mini-128k-instruct-fp16-ov',
  'OpenVINO/Phi-3-mini-128k-instruct-int4-ov',
  'OpenVINO/Phi-3-mini-128k-instruct-int8-ov',
  'OpenVINO/Phi-3-mini-4k-instruct-fp16-ov',
  'OpenVINO/Phi-3-mini-4k-instruct-int4-ov',
  'OpenVINO/Phi-3-mini-4k-instruct-int8-ov',
  'OpenVINO/phi-4-fp16-ov',
  'OpenVINO/phi-4-int4-ov',
  'OpenVINO/phi-4-int8-ov',
  'OpenVINO/Phi-4-mini-instruct-fp16-ov',
  'OpenVINO/Phi-4-mini-instruct-int4-ov',
  'OpenVINO/Phi-4-mini-instruct-int8-ov',
  'OpenVINO/pythia-12b-fp16-ov',
  'OpenVINO/pythia-12b-int8-ov',
  'OpenVINO/pythia-1b-int4-ov',
  'OpenVINO/pythia-6.9b-fp16-ov',
  'OpenVINO/pythia-6.9b-int4-ov',
  'OpenVINO/pythia-6.9b-int8-ov',
  'OpenVINO/Qwen2.5-1.5B-Instruct-fp16-ov',
  'OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov',
  'OpenVINO/Qwen2.5-1.5B-Instruct-int8-ov',
  'OpenVINO/Qwen2.5-14B-Instruct-fp16-ov',
  'OpenVINO/Qwen2.5-14B-Instruct-int4-ov',
  'OpenVINO/Qwen2.5-14B-Instruct-int8-ov',
  'OpenVINO/Qwen2.5-7B-Instruct-fp16-ov',
  'OpenVINO/Qwen2.5-7B-Instruct-int4-ov',
  'OpenVINO/Qwen2.5-7B-Instruct-int8-ov',
  'OpenVINO/Qwen2-0.5B-fp16-ov',
  'OpenVINO/Qwen2-0.5B-Instruct-fp16-ov',
  'OpenVINO/Qwen2-0.5B-Instruct-int4-ov',
  'OpenVINO/Qwen2-0.5B-Instruct-int8-ov',
  'OpenVINO/Qwen2-0.5B-int4-ov',
  'OpenVINO/Qwen2-0.5B-int8-ov',
  'OpenVINO/Qwen2-1.5B-fp16-ov',
  'OpenVINO/Qwen2-1.5B-Instruct-fp16-ov',
  'OpenVINO/Qwen2-1.5B-Instruct-int4-ov',
  'OpenVINO/Qwen2-1.5B-Instruct-int8-ov',
  'OpenVINO/Qwen2-1.5B-int4-ov',
  'OpenVINO/Qwen2-1.5B-int8-ov',
  'OpenVINO/Qwen2-7B-Instruct-fp16-ov',
  'OpenVINO/Qwen2-7B-Instruct-int4-ov',
  'OpenVINO/Qwen2-7B-Instruct-int8-ov',
  'OpenVINO/Qwen3-0.6B-fp16-ov',
  'OpenVINO/Qwen3-0.6B-int4-ov',
  'OpenVINO/Qwen3-0.6B-int8-ov',
  'OpenVINO/Qwen3-1.7B-fp16-ov',
  'OpenVINO/Qwen3-1.7B-int4-ov',
  'OpenVINO/Qwen3-1.7B-int8-ov',
  'OpenVINO/Qwen3-14B-fp16-ov',
  'OpenVINO/Qwen3-14B-int4-ov',
  'OpenVINO/Qwen3-14B-int8-ov',
  'OpenVINO/Qwen3-30B-A3B-Instruct-2507-int4-ov',
  'OpenVINO/Qwen3-30B-A3B-Instruct-2507-int8-ov',
  'OpenVINO/Qwen3-4B-fp16-ov',
  'OpenVINO/Qwen3-4B-int4-ov',
  'OpenVINO/Qwen3-4B-int8-ov',
  'OpenVINO/Qwen3-8B-fp16-ov',
  'OpenVINO/Qwen3-8B-int4-ov',
  'OpenVINO/Qwen3-8B-int8-ov',
  'OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov',
  'OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int8-ov',
  'OpenVINO/RedPajama-INCITE-Chat-3B-v1-fp16-ov',
  'OpenVINO/RedPajama-INCITE-Chat-3B-v1-int4-ov',
  'OpenVINO/RedPajama-INCITE-Chat-3B-v1-int8-ov',
  'OpenVINO/RedPajama-INCITE-Instruct-3B-v1-fp16-ov',
  'OpenVINO/RedPajama-INCITE-Instruct-3B-v1-int4-ov',
  'OpenVINO/RedPajama-INCITE-Instruct-3B-v1-int8-ov',
  'OpenVINO/starcoder2-15b-fp16-ov',
  'OpenVINO/starcoder2-15b-int4-ov',
  'OpenVINO/starcoder2-15b-int8-ov',
  'OpenVINO/starcoder2-7b-fp16-ov',
  'OpenVINO/starcoder2-7b-int4-ov',
  'OpenVINO/starcoder2-7b-int8-ov',
  'OpenVINO/TinyLlama-1.1B-Chat-v1.0-fp16-ov',
  'OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov',
  'OpenVINO/TinyLlama-1.1B-Chat-v1.0-int8-ov',
  'OpenVINO/zephyr-7b-beta-fp16-ov',
  'OpenVINO/zephyr-7b-beta-int4-ov',
  'OpenVINO/zephyr-7b-beta-int8-ov',
] as const;

const VLM_REPO_IDS = [
  'OpenVINO/gemma-3-12b-it-fp16-ov',
  'OpenVINO/gemma-3-12b-it-int4-ov',
  'OpenVINO/gemma-3-12b-it-int8-ov',
  'OpenVINO/gemma-3-4b-it-fp16-ov',
  'OpenVINO/gemma-3-4b-it-int4-ov',
  'OpenVINO/gemma-3-4b-it-int8-ov',
  'OpenVINO/InternVL2-1B-fp16-ov',
  'OpenVINO/InternVL2-1B-int4-ov',
  'OpenVINO/InternVL2-1B-int8-ov',
  'OpenVINO/InternVL2-2B-fp16-ov',
  'OpenVINO/InternVL2-2B-int4-ov',
  'OpenVINO/InternVL2-2B-int8-ov',
  'OpenVINO/Phi-3.5-vision-instruct-fp16-ov',
  'OpenVINO/Phi-3.5-vision-instruct-int4-ov',
  'OpenVINO/Phi-3.5-vision-instruct-int8-ov',
] as const;

const ASR_REPO_IDS = [
  'OpenVINO/distil-whisper-large-v2-fp16-ov',
  'OpenVINO/distil-whisper-large-v2-int4-ov',
  'OpenVINO/distil-whisper-large-v2-int8-ov',
  'OpenVINO/distil-whisper-large-v3-fp16-ov',
  'OpenVINO/distil-whisper-large-v3-int4-ov',
  'OpenVINO/distil-whisper-large-v3-int8-ov',
  'OpenVINO/whisper-base-fp16-ov',
  'OpenVINO/whisper-base-int4-ov',
  'OpenVINO/whisper-base-int8-ov',
  'OpenVINO/whisper-large-v3-fp16-ov',
  'OpenVINO/whisper-large-v3-int4-ov',
  'OpenVINO/whisper-large-v3-int8-ov',
  'OpenVINO/whisper-medium.en-fp16-ov',
  'OpenVINO/whisper-medium.en-int4-ov',
  'OpenVINO/whisper-medium.en-int8-ov',
  'OpenVINO/whisper-medium-fp16-ov',
  'OpenVINO/whisper-medium-int4-ov',
  'OpenVINO/whisper-medium-int8-ov',
  'OpenVINO/whisper-tiny-fp16-ov',
  'OpenVINO/whisper-tiny-int4-ov',
  'OpenVINO/whisper-tiny-int8-ov',
] as const;

const COLLECTION_SEEDS: RepoSeed[] = [
  ...CUSTOM_REPO_SEEDS,
  ...LLM_REPO_IDS.map((repoId) => ({ repoId, runtimeFamily: 'openvino-llm-node' as const, taskFamily: 'translate' as const })),
  ...VLM_REPO_IDS.map((repoId) => ({ repoId, runtimeFamily: 'openvino-llm-node' as const, taskFamily: 'vlm' as const })),
  ...ASR_REPO_IDS.map((repoId) => ({ repoId, runtimeFamily: 'openvino-whisper-node' as const, taskFamily: 'asr' as const })),
];

function normalizeRepoId(value: string) {
  return String(value || '').trim().toLowerCase();
}

function slugifyRepoId(repoId: string) {
  return normalizeRepoId(repoId).replace(/^openvino\//, 'openvino-').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
}

function buildModelLinks(repoId: string, includeConfig = false, includePreprocessor = false) {
  const links = [`https://huggingface.co/${repoId}`];
  if (includeConfig) {
    links.push(`https://huggingface.co/${repoId}/raw/main/generation_config.json`);
  }
  if (includePreprocessor) {
    links.push(`https://huggingface.co/${repoId}/raw/main/preprocessor_config.json`);
  }
  return links;
}

function createExactOrFamilyBaseline(seed: RepoSeed): LocalOpenvinoOfficialBaseline {
  const normalized = normalizeRepoId(seed.repoId);
  const base: LocalOpenvinoOfficialBaseline = {
    id: slugifyRepoId(seed.repoId),
    repoId: seed.repoId,
    runtimeFamily: seed.runtimeFamily,
    taskFamily: seed.taskFamily,
    baselineConfidence: 'partial_public',
    sourceLinks: buildModelLinks(seed.repoId),
    notes: ['Collection baseline only; verify model-specific generation_config when this model is first tuned.'],
  };

  if (/openvino\/qwen3-/i.test(normalized)) {
    return {
      ...base,
      baselineConfidence: 'full_public',
      sourceLinks: buildModelLinks(seed.repoId, true),
      officialGeneration: {
        doSample: true,
        temperature: 0.6,
        topK: 20,
        topP: 0.95,
      },
      notes: ['Qwen3 family baseline from OpenVINO public configs.', 'Disable thinking in direct-answer translation flows.'],
    };
  }

  if (/openvino\/qwen2\.5-1\.5b-instruct-/i.test(normalized)) {
    return {
      ...base,
      baselineConfidence: 'full_public',
      sourceLinks: buildModelLinks(seed.repoId, true),
      officialGeneration: {
        doSample: true,
        temperature: 0.7,
        topK: 20,
        topP: 0.8,
        repetitionPenalty: 1.1,
      },
      notes: ['Qwen2.5 1.5B baseline from OpenVINO public configs.'],
    };
  }

  if (/openvino\/qwen2\.5-.*instruct-/i.test(normalized)) {
    return {
      ...base,
      baselineConfidence: 'full_public',
      sourceLinks: buildModelLinks(seed.repoId, true),
      officialGeneration: {
        doSample: true,
        temperature: 0.7,
        topK: 20,
        topP: 0.8,
        repetitionPenalty: 1.05,
      },
      notes: ['Qwen2.5 instruct-family baseline from OpenVINO public configs.'],
    };
  }

  if (/openvino\/gemma-3-/i.test(normalized)) {
    return {
      ...base,
      taskFamily: 'vlm',
      baselineConfidence: 'gated_partial',
      officialGeneration: {
        maxNewTokensExample: 100,
      },
      notes: ['Gemma 3 is treated as VLM-first; do not assume text-only Qwen defaults.', 'License-gated repo, so only partial public baseline is stored.'],
    };
  }

  if (/openvino\/(distil-)?whisper-/i.test(normalized)) {
    return {
      ...base,
      baselineConfidence: 'full_public',
      sourceLinks: buildModelLinks(seed.repoId, true, true),
      officialAsr: {
        task: 'transcribe',
        returnTimestamps: false,
        chunkLengthSec: 30,
        samplingRate: 16000,
        maxTargetPositions: 448,
      },
      notes: ['Whisper-family baseline from OpenVINO public ASR configs.'],
    };
  }

  if (/mediatek-research\/breeze-asr-26/i.test(normalized)) {
    return {
      ...base,
      baselineConfidence: 'partial_public',
      sourceLinks: buildModelLinks(seed.repoId, true, true),
      officialAsr: {
        task: 'transcribe',
        returnTimestamps: false,
        chunkLengthSec: 30,
        samplingRate: 16000,
        maxTargetPositions: 448,
      },
      notes: [
        'Breeze-ASR-26 is a public Whisper-large-v2 fine-tune and follows the local Whisper OpenVINO runtime path.',
        'Model-specific transcription quality should be validated on Taigi audio rather than generic multilingual samples.',
      ],
    };
  }

  if (/qwen\/qwen3-asr-/i.test(normalized)) {
    return {
      ...base,
      runtimeFamily: 'openvino-qwen3-asr',
      baselineConfidence: 'partial_public',
      officialAsr: {
        task: 'transcribe',
        returnTimestamps: true,
        samplingRate: 16000,
      },
      notes: [
        'Qwen3-ASR fallback baseline: runtime family is inferred from ArcSub local integration.',
        'Verify helper/runtime-specific generation details when this model is first tuned.',
      ],
    };
  }

  return base;
}

const OPENVINO_OFFICIAL_BASELINES: LocalOpenvinoOfficialBaseline[] = COLLECTION_SEEDS.map(createExactOrFamilyBaseline);
const BASELINE_BY_ID = new Map(OPENVINO_OFFICIAL_BASELINES.map((item) => [item.id.toLowerCase(), item]));
const BASELINE_BY_REPO = new Map(OPENVINO_OFFICIAL_BASELINES.map((item) => [normalizeRepoId(item.repoId), item]));

function inferFallbackTaskFamily(repoId: string): TaskFamily {
  const normalized = normalizeRepoId(repoId);
  if (/qwen\/qwen3-asr-/.test(normalized)) return 'asr';
  if (/whisper/.test(normalized)) return 'asr';
  if (/gemma-3|internvl|vision/.test(normalized)) return 'vlm';
  return 'translate';
}

function inferFallbackRuntimeFamily(repoId: string): RuntimeFamily {
  const normalized = normalizeRepoId(repoId);
  if (/qwen\/qwen3-asr-/.test(normalized)) return 'openvino-qwen3-asr';
  if (/wav2vec2|hubert|wavlm|unispeech|sew|data2vec-audio/.test(normalized)) return 'openvino-ctc-asr';
  if (/(^|\/)(flan-|mt5|umt5|t5|bart|mbart|marian|pegasus|m2m100|m2m_100|fsmt|prophetnet)/.test(normalized)) {
    return 'openvino-seq2seq-translate';
  }
  return inferFallbackTaskFamily(repoId) === 'asr' ? 'openvino-whisper-node' : 'openvino-llm-node';
}

function buildFallbackBaseline(repoId: string): LocalOpenvinoOfficialBaseline {
  const taskFamily = inferFallbackTaskFamily(repoId);
  const runtimeFamily = inferFallbackRuntimeFamily(repoId);
  const normalized = normalizeRepoId(repoId);
  const fallback = createExactOrFamilyBaseline({ repoId, runtimeFamily, taskFamily });
  const fallbackNotes = [
    ...(fallback.notes || []),
    'ArcSub fallback baseline: repo not explicitly listed in current collection registry; verify model-specific configs when first used.',
  ];

  if (taskFamily === 'translate' && !fallback.officialGeneration && !/qwen|gemma-3/.test(normalized)) {
    fallback.officialGeneration = {
      doSample: true,
      temperature: 0.7,
      topK: 20,
      topP: 0.8,
    };
    fallbackNotes.push('Generic translation fallback uses ArcSub default OpenVINO text-generation seed values.');
  }

  if (taskFamily === 'vlm' && !fallback.officialGeneration) {
    fallback.officialGeneration = { maxNewTokensExample: 100 };
    fallbackNotes.push('Generic VLM fallback uses conservative max_new_tokens example value 100.');
  }

  if (taskFamily === 'asr' && !fallback.officialAsr) {
    fallback.officialAsr = {
      task: 'transcribe',
      returnTimestamps: false,
      chunkLengthSec: 30,
      samplingRate: 16000,
      maxTargetPositions: 448,
    };
    fallbackNotes.push('Generic ASR fallback uses Whisper-family transcription defaults.');
  }

  return {
    ...fallback,
    baselineConfidence: fallback.baselineConfidence === 'full_public' ? 'partial_public' : fallback.baselineConfidence,
    notes: fallbackNotes,
  };
}

export function listLocalOpenvinoOfficialBaselines() {
  return [...OPENVINO_OFFICIAL_BASELINES];
}

export function getLocalOpenvinoOfficialBaselineById(id: string) {
  const normalizedId = String(id || '').trim().toLowerCase();
  return BASELINE_BY_ID.get(normalizedId) || null;
}

export function hasExactLocalOpenvinoOfficialBaselineForRepoId(repoId: string) {
  return BASELINE_BY_REPO.has(normalizeRepoId(repoId));
}

export function getLocalOpenvinoOfficialBaselineForRepoId(repoId: string) {
  const normalizedRepoId = normalizeRepoId(repoId);
  return BASELINE_BY_REPO.get(normalizedRepoId) || buildFallbackBaseline(repoId);
}
