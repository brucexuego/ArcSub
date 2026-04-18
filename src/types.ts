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
  isLocal?: boolean;
  provider?: 'cloud' | 'local-openvino';
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
      | 'openvino-seq2seq-translate'
      | 'openvino-llm-node';
    runtimeLayout?:
      | 'asr-whisper'
      | 'asr-ctc'
      | 'asr-qwen3-official'
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
      | 'unsupported';
    device: 'AUTO';
    source: 'builtin' | 'custom';
  }>;
}
