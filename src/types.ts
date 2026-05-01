import type { ProjectStatus } from './project_status';

export interface Project {
  id: string;
  name: string;
  videoTitle?: string;
  status: ProjectStatus;
  lastUpdated: string;
  thumbnail?: string;
  duration?: string;
  resolution?: string;
  videoUrl?: string;
  audioUrl?: string;
  mediaSourceType?: 'online' | 'upload';
  notes?: string;
  originalSubtitles?: string;
  translatedSubtitles?: string;
  transcriptionSourceLanguage?: string | null;
  translationMetadata?: {
    lastModelId?: string | null;
    lastProviderModel?: string | null;
    lastProviderName?: string | null;
    lastSourceType?: 'transcription' | 'project';
    lastSourceLang?: string | null;
    lastTargetLang?: string | null;
    lastTranslatedAt?: string | null;
  };
  videoMetadata?: any;
}

export type MaterialCategory = 'video' | 'audio' | 'subtitle' | 'other';

export interface Material {
  name: string;
  category: MaterialCategory;
  size: string;
  date: string;
}

export interface ResourceStats {
  cpu: number;
  ram: number;
  gpu: number;
  vram: number;
  ramUsedGB?: number;
  ramTotalGB?: number;
  vramUsedGB?: number;
  vramTotalGB?: number;
  platform?: string;
  cpuModel?: string;
  gpuModel?: string;
  accelerators?: AcceleratorStats[];
}

export interface AcceleratorStats {
  id: string;
  kind: 'gpu' | 'npu';
  vendor: 'intel';
  model: string;
  utilization: number;
  vramUsedGB?: number;
  vramTotalGB?: number;
  luid?: string;
  memorySource?: 'dedicated' | 'shared';
  engineTypes?: string[];
  physIndex?: number;
  taskManagerIndex?: number;
}

export interface ApiConfig {
  id: string;
  name: string;
  url: string;
  key: string;
  model?: string;
  options?: ApiModelRequestOptions;
  isLocal?: boolean;
  provider?: 'cloud' | 'local-openvino';
}

export interface ApiModelSamplingOptions {
  temperature?: number | null;
  topP?: number | null;
  topK?: number | null;
  maxOutputTokens?: number | null;
}

export interface ApiModelQuotaOptions {
  rpm?: number | null;
  tpm?: number | null;
  rpd?: number | null;
  maxConcurrency?: number | null;
}

export interface ApiModelTranslationBatchingOptions {
  enabled?: boolean | null;
  targetLines?: number | null;
  minTargetLines?: number | null;
  charBudget?: number | null;
  maxSplitDepth?: number | null;
  maxOutputTokens?: number | null;
  timeoutMs?: number | null;
  stream?: boolean | null;
}

export interface ApiModelTranslationOptions {
  executionMode?: 'auto' | 'cloud_context' | 'cloud_strict' | 'cloud_relaxed' | 'provider_native';
  quota?: ApiModelQuotaOptions;
  batching?: ApiModelTranslationBatchingOptions;
  contextWindow?: number | null;
  targetLines?: number | null;
  charBudget?: number | null;
}

export interface ApiModelRequestOptions {
  sampling?: ApiModelSamplingOptions;
  headers?: Record<string, string>;
  body?: Record<string, unknown>;
  quota?: ApiModelQuotaOptions;
  translation?: ApiModelTranslationOptions;
  timeoutMs?: number | null;
}

export interface LocalModelSelection {
  asrSelectedId: string;
  translateSelectedId: string;
  installed?: Array<{
    id: string;
    type: 'asr' | 'translate';
    displayName: string;
    repoId: string;
    downloadRepoId?: string;
    localSubdir: string;
    requiredFiles: string[];
    runtime:
      | 'openvino-whisper-node'
      | 'openvino-ctc-asr'
      | 'openvino-qwen3-asr'
      | 'openvino-cohere-asr'
      | 'hf-transformers-asr'
      | 'openvino-seq2seq-translate'
      | 'openvino-llm-node';
    runtimeLayout?:
      | 'asr-whisper'
      | 'asr-ctc'
      | 'asr-qwen3-official'
      | 'asr-cohere-ov'
      | 'asr-hf-transformers'
      | 'translate-llm'
      | 'translate-seq2seq'
      | 'translate-vlm';
    installMode?: 'hf-direct' | 'hf-qwen3-asr-convert' | 'hf-auto-convert';
    sourceFormat?:
      | 'openvino-ir'
      | 'onnx'
      | 'tensorflow'
      | 'tensorflow-lite'
      | 'paddle'
      | 'pytorch'
      | 'jax-flax'
      | 'keras'
      | 'gguf'
      | 'unknown';
    conversionMethod?:
      | 'direct-download'
      | 'openvino-convert-model'
      | 'optimum-export-openvino'
      | 'openvino-ctc-asr-export'
      | 'openvino-qwen3-asr-export'
      | 'openvino-cohere-asr-export'
      | 'unsupported';
    device: 'AUTO';
    source: 'builtin' | 'custom';
  }>;
}
