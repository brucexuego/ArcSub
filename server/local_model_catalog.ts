import fs from 'fs';
import path from 'path';
import { PathManager } from './path_manager.js';

export type LocalModelType = 'asr' | 'translate';
export type LocalModelRuntime =
  | 'openvino-whisper-node'
  | 'openvino-ctc-asr'
  | 'openvino-qwen3-asr'
  | 'openvino-seq2seq-translate'
  | 'openvino-llm-node';
export type LocalModelInstallMode = 'hf-direct' | 'hf-qwen3-asr-convert' | 'hf-auto-convert';
export type LocalModelSourceFormat =
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
export type LocalModelConversionMethod =
  | 'direct-download'
  | 'openvino-convert-model'
  | 'optimum-export-openvino'
  | 'openvino-ctc-asr-export'
  | 'openvino-qwen3-asr-export'
  | 'unsupported';
export type LocalModelRuntimeLayout =
  | 'asr-whisper'
  | 'asr-ctc'
  | 'asr-qwen3-official'
  | 'translate-llm'
  | 'translate-seq2seq'
  | 'translate-vlm';

export interface LocalModelRuntimeHintEvidence {
  source: string;
  key?: string;
  value?: string | number | boolean | null;
  confidence?: 'high' | 'medium' | 'low';
}

export interface LocalModelRuntimeHints {
  inspectedAt?: string;
  hfSha?: string;
  modelCard?: {
    license?: string | null;
    baseModel?: string | string[] | null;
    pipelineTag?: string | null;
    libraryName?: string | null;
    summary?: string | null;
  };
  contextWindow?: number;
  maxInputTokens?: number;
  maxOutputTokens?: number;
  generation?: {
    doSample?: boolean;
    temperature?: number;
    topP?: number;
    topK?: number;
    minP?: number;
    repetitionPenalty?: number;
    presencePenalty?: number;
    frequencyPenalty?: number;
    noRepeatNgramSize?: number;
  };
  chatTemplate?: {
    available?: boolean;
    supportsThinking?: boolean;
    defaultEnableThinking?: boolean;
    templateSource?: string | null;
    templateHash?: string | null;
    kwargs?: Record<string, unknown>;
  };
  batching?: {
    mode?: 'token_aware' | 'fixed_lines';
    inputTokenBudget?: number;
    outputTokenBudget?: number;
    safetyReserveTokens?: number;
    maxLines?: number;
    charBudget?: number;
    confidence?: 'high' | 'medium' | 'low';
  };
  asr?: {
    task?: 'transcribe' | 'translate';
    returnTimestamps?: boolean;
    wordTimestamps?: boolean;
    chunkLengthSec?: number;
    samplingRate?: number;
    maxTargetPositions?: number;
    confidence?: 'high' | 'medium' | 'low';
  };
  evidence?: LocalModelRuntimeHintEvidence[];
}

export interface LocalModelDefinition {
  id: string;
  type: LocalModelType;
  displayName: string;
  repoId: string;
  downloadRepoId?: string;
  localSubdir: string;
  requiredFiles: string[];
  runtime: LocalModelRuntime;
  runtimeLayout: LocalModelRuntimeLayout;
  installMode: LocalModelInstallMode;
  sourceFormat: LocalModelSourceFormat;
  conversionMethod: LocalModelConversionMethod;
  device: 'AUTO';
  source: 'builtin' | 'custom';
  runtimeHints?: LocalModelRuntimeHints;
}

export interface LocalModelSelection {
  asrSelectedId: string;
  translateSelectedId: string;
  installed: LocalModelDefinition[];
}

export interface HuggingFaceModelMetadata {
  id?: string;
  pipeline_tag?: string | null;
  library_name?: string | null;
  tags?: string[];
  siblings?: Array<{ rfilename?: string }>;
  config?: {
    model_type?: string | null;
    architectures?: string[] | null;
  } | null;
}

const ASR_REQUIRED_FILES = [
  'config.json',
  'generation_config.json',
  'preprocessor_config.json',
  'openvino_encoder_model.xml',
  'openvino_encoder_model.bin',
  'openvino_decoder_model.xml',
  'openvino_decoder_model.bin',
  'openvino_tokenizer.xml',
  'openvino_tokenizer.bin',
  'openvino_detokenizer.xml',
  'openvino_detokenizer.bin',
  'tokenizer.json',
  'tokenizer_config.json',
];

const ASR_OPTIONAL_FILES = [
  'special_tokens_map.json',
  'added_tokens.json',
  'normalizer.json',
  'vocab.json',
  'merges.txt',
  'openvino_config.json',
];

const CTC_ASR_BASE_REQUIRED_FILES = [
  'config.json',
  'preprocessor_config.json',
  'tokenizer_config.json',
  'openvino_model.xml',
  'openvino_model.bin',
];

const CTC_ASR_TOKENIZER_FILES = ['tokenizer.json', 'vocab.json', 'vocab.txt'];

const CTC_ASR_OPTIONAL_FILES = [
  'special_tokens_map.json',
  'added_tokens.json',
  'tokenizer.json',
  'vocab.json',
  'vocab.txt',
];

const QWEN3_ASR_OFFICIAL_CONVERTED_BASE_REQUIRED_FILES = [
  'config.json',
  'preprocessor_config.json',
  'tokenizer_config.json',
  'merges.txt',
  'vocab.json',
  'thinker/openvino_thinker_language_model.xml',
  'thinker/openvino_thinker_language_model.bin',
  'thinker/openvino_thinker_audio_model.xml',
  'thinker/openvino_thinker_audio_model.bin',
  'thinker/openvino_thinker_audio_encoder_model.xml',
  'thinker/openvino_thinker_audio_encoder_model.bin',
  'thinker/openvino_thinker_embedding_model.xml',
  'thinker/openvino_thinker_embedding_model.bin',
];

const QWEN3_ASR_OFFICIAL_CONVERTED_TEMPLATE_FILES = ['chat_template.jinja', 'chat_template.json'];

const QWEN3_ASR_OFFICIAL_CONVERTED_OPTIONAL_FILES = [
  'generation_config.json',
  'added_tokens.json',
  'special_tokens_map.json',
  'tokenizer.json',
];

const TRANSLATE_SHARED_REQUIRED_FILES = [
  'config.json',
  'generation_config.json',
  'openvino_tokenizer.xml',
  'openvino_tokenizer.bin',
  'openvino_detokenizer.xml',
  'openvino_detokenizer.bin',
  'tokenizer.json',
  'tokenizer_config.json',
];

const TRANSLATE_LLM_REQUIRED_FILES = [
  'openvino_model.xml',
  'openvino_model.bin',
];

const TRANSLATE_SEQ2SEQ_REQUIRED_FILES = [
  'openvino_encoder_model.xml',
  'openvino_encoder_model.bin',
  'openvino_decoder_model.xml',
  'openvino_decoder_model.bin',
];

const TRANSLATE_VLM_REQUIRED_FILES = [
  'openvino_language_model.xml',
  'openvino_language_model.bin',
  'openvino_text_embeddings_model.xml',
  'openvino_text_embeddings_model.bin',
  'openvino_vision_embeddings_model.xml',
  'openvino_vision_embeddings_model.bin',
];

const TRANSLATE_OPTIONAL_FILES = [
  'special_tokens_map.json',
  'added_tokens.json',
  'vocab.json',
  'merges.txt',
  'chat_template.jinja',
  'openvino_config.json',
  'preprocessor_config.json',
];

function normalizeFileList(files: string[]) {
  const unique = new Set<string>();
  for (const file of files) {
    const normalized = String(file || '').trim();
    if (!normalized) continue;
    unique.add(normalized);
  }
  return Array.from(unique);
}

function pickExistingFiles(fileSet: Set<string>, orderedFiles: string[]) {
  return orderedFiles.filter((file) => fileSet.has(file));
}

function hasFile(fileSet: Set<string>, pattern: RegExp | string) {
  if (typeof pattern === 'string') {
    return fileSet.has(pattern);
  }
  for (const file of fileSet) {
    if (pattern.test(file)) return true;
  }
  return false;
}

function collectSiblingFiles(metadata: HuggingFaceModelMetadata) {
  return Array.isArray(metadata?.siblings)
    ? metadata.siblings
        .map((item) => String(item?.rfilename || '').trim())
        .filter(Boolean)
    : [];
}

function normalizeTagSet(metadata: HuggingFaceModelMetadata) {
  return new Set((Array.isArray(metadata?.tags) ? metadata.tags : []).map((tag) => String(tag || '').trim().toLowerCase()));
}

function extractBaseModelRepoId(metadata: HuggingFaceModelMetadata) {
  const tags = Array.isArray(metadata?.tags) ? metadata.tags : [];
  const normalized = tags
    .map((tag) => String(tag || '').trim())
    .filter(Boolean);
  const direct = normalized.find((tag) => /^base_model:[^:][^/]*\/[^:]+$/i.test(tag));
  if (direct) {
    return direct.slice('base_model:'.length).trim();
  }
  const quantized = normalized.find((tag) => /^base_model:quantized:[^:][^/]*\/.+$/i.test(tag));
  if (quantized) {
    return quantized.slice('base_model:quantized:'.length).trim();
  }
  return '';
}

function inferWhisperBaseModelRepoId(repoId: string, metadata: HuggingFaceModelMetadata) {
  const directBase = extractBaseModelRepoId(metadata);
  if (/^openai\/whisper-/i.test(directBase)) {
    return directBase;
  }

  const candidates = [
    String(repoId || '').trim(),
    ...collectSiblingFiles(metadata),
  ];
  const patterns: Array<{ regex: RegExp; base: string }> = [
    { regex: /whisper[-_.]tiny[-_.]en(?:\.|$)/i, base: 'openai/whisper-tiny.en' },
    { regex: /whisper[-_.]base[-_.]en(?:\.|$)/i, base: 'openai/whisper-base.en' },
    { regex: /whisper[-_.]small[-_.]en(?:\.|$)/i, base: 'openai/whisper-small.en' },
    { regex: /whisper[-_.]medium[-_.]en(?:\.|$)/i, base: 'openai/whisper-medium.en' },
    { regex: /whisper[-_.]turbo(?:\.|$)/i, base: 'openai/whisper-large-v3-turbo' },
    { regex: /whisper[-_.]large[-_.]v3(?:\.|$)/i, base: 'openai/whisper-large-v3' },
    { regex: /whisper[-_.]large[-_.]v2(?:\.|$)/i, base: 'openai/whisper-large-v2' },
    { regex: /whisper[-_.]large(?:\.|$)/i, base: 'openai/whisper-large' },
    { regex: /whisper[-_.]medium(?:\.|$)/i, base: 'openai/whisper-medium' },
    { regex: /whisper[-_.]small(?:\.|$)/i, base: 'openai/whisper-small' },
    { regex: /whisper[-_.]base(?:\.|$)/i, base: 'openai/whisper-base' },
    { regex: /whisper[-_.]tiny(?:\.|$)/i, base: 'openai/whisper-tiny' },
  ];

  for (const candidate of candidates) {
    for (const pattern of patterns) {
      if (pattern.regex.test(candidate)) {
        return pattern.base;
      }
    }
  }

  return '';
}

function formatRepoDisplayToken(token: string) {
  const raw = String(token || '').trim();
  if (!raw) return '';
  const normalized = raw.toLowerCase();
  const exact: Record<string, string> = {
    ai: 'AI',
    api: 'API',
    asr: 'ASR',
    ctc: 'CTC',
    fp16: 'FP16',
    fp32: 'FP32',
    gguf: 'GGUF',
    hf: 'HF',
    int4: 'INT4',
    int8: 'INT8',
    it: 'IT',
    llm: 'LLM',
    ov: 'OV',
    vlm: 'VLM',
    vllm: 'vLLM',
    openvino: 'OpenVINO',
    qwen: 'Qwen',
    qwen2: 'Qwen2',
    qwen3: 'Qwen3',
    gemma: 'Gemma',
    translategemma: 'TranslateGemma',
    deepseek: 'DeepSeek',
    distil: 'Distil',
    whisper: 'Whisper',
    phi: 'Phi',
  };
  if (exact[normalized]) return exact[normalized];
  if (/^\d+b$/i.test(raw)) return raw.toUpperCase();
  if (/^v\d+(?:\.\d+)?$/i.test(raw)) return `v${raw.slice(1)}`;
  if (/^r\d+$/i.test(raw)) return raw.toUpperCase();
  if (/^\d+(?:\.\d+)?$/.test(raw)) return raw;
  return raw.charAt(0).toUpperCase() + raw.slice(1);
}

function formatRepoDisplaySegment(segment: string, options: { dropTrailingOv?: boolean } = {}) {
  const tokens = String(segment || '')
    .replace(/[_]+/g, '-')
    .split(/[-\s]+/)
    .map((token) => token.trim())
    .filter(Boolean);
  if (options.dropTrailingOv && tokens[tokens.length - 1]?.toLowerCase() === 'ov') {
    tokens.pop();
  }
  return tokens.map(formatRepoDisplayToken).filter(Boolean).join(' ');
}

function buildDefaultDisplayName(repoId: string) {
  const normalized = String(repoId || '').trim().replace(/^\/+|\/+$/g, '');
  if (!normalized) return '';
  const parts = normalized.split('/').filter(Boolean);
  if (parts.length < 2) {
    return formatRepoDisplaySegment(normalized, { dropTrailingOv: true }) || normalized;
  }
  const owner = formatRepoDisplaySegment(parts[0]);
  const name = formatRepoDisplaySegment(parts.slice(1).join(' '), { dropTrailingOv: true });
  return [owner, name].filter(Boolean).join(' ') || normalized;
}

function isOfficialQwen3AsrRepo(repoId: string) {
  const normalized = String(repoId || '').trim().toLowerCase();
  return /^qwen\/qwen3-asr-[^/]+$/.test(normalized);
}

function getOfficialQwen3AsrConvertedRequiredFiles(fileSet?: Set<string>) {
  const templateFile =
    fileSet && fileSet.has('chat_template.jinja')
      ? 'chat_template.jinja'
      : fileSet && fileSet.has('chat_template.json')
        ? 'chat_template.json'
        : 'chat_template.jinja';
  return [...QWEN3_ASR_OFFICIAL_CONVERTED_BASE_REQUIRED_FILES, templateFile];
}

function getCtcAsrRequiredFiles(fileSet?: Set<string>) {
  const tokenizerFile =
    fileSet && fileSet.has('tokenizer.json')
      ? 'tokenizer.json'
      : fileSet && fileSet.has('vocab.json')
        ? 'vocab.json'
        : fileSet && fileSet.has('vocab.txt')
          ? 'vocab.txt'
          : 'vocab.json';
  return [...CTC_ASR_BASE_REQUIRED_FILES, tokenizerFile];
}

function getRequiredFilesForRuntimeLayout(layout: LocalModelRuntimeLayout, fileSet?: Set<string>) {
  switch (layout) {
    case 'asr-whisper':
      return fileSet ? pickExistingFiles(fileSet, [...ASR_REQUIRED_FILES, ...ASR_OPTIONAL_FILES]) : [...ASR_REQUIRED_FILES];
    case 'asr-ctc':
      return fileSet
        ? pickExistingFiles(fileSet, [...getCtcAsrRequiredFiles(fileSet), ...CTC_ASR_OPTIONAL_FILES])
        : getCtcAsrRequiredFiles();
    case 'asr-qwen3-official':
      return fileSet
        ? pickExistingFiles(fileSet, [...getOfficialQwen3AsrConvertedRequiredFiles(fileSet), ...QWEN3_ASR_OFFICIAL_CONVERTED_OPTIONAL_FILES])
        : getOfficialQwen3AsrConvertedRequiredFiles();
    case 'translate-vlm':
      return fileSet
        ? pickExistingFiles(
            fileSet,
            [...TRANSLATE_SHARED_REQUIRED_FILES, ...TRANSLATE_VLM_REQUIRED_FILES, ...TRANSLATE_OPTIONAL_FILES]
          )
        : [...TRANSLATE_SHARED_REQUIRED_FILES, ...TRANSLATE_VLM_REQUIRED_FILES];
    case 'translate-seq2seq':
      return fileSet
        ? pickExistingFiles(
            fileSet,
            [...TRANSLATE_SHARED_REQUIRED_FILES, ...TRANSLATE_SEQ2SEQ_REQUIRED_FILES, ...TRANSLATE_OPTIONAL_FILES]
          )
        : [...TRANSLATE_SHARED_REQUIRED_FILES, ...TRANSLATE_SEQ2SEQ_REQUIRED_FILES];
    case 'translate-llm':
    default:
      return fileSet
        ? pickExistingFiles(
            fileSet,
            [...TRANSLATE_SHARED_REQUIRED_FILES, ...TRANSLATE_LLM_REQUIRED_FILES, ...TRANSLATE_OPTIONAL_FILES]
          )
        : [...TRANSLATE_SHARED_REQUIRED_FILES, ...TRANSLATE_LLM_REQUIRED_FILES];
  }
}

function inferRuntimeLayoutFromArtifacts(type: LocalModelType, fileSet: Set<string>): LocalModelRuntimeLayout | null {
  if (type === 'asr') {
    const hasOfficialTemplateFile = QWEN3_ASR_OFFICIAL_CONVERTED_TEMPLATE_FILES.some((file) => fileSet.has(file));
    const hasOfficialQwenLayout =
      hasOfficialTemplateFile &&
      QWEN3_ASR_OFFICIAL_CONVERTED_BASE_REQUIRED_FILES.every((file) => fileSet.has(file));
    if (hasOfficialQwenLayout) {
      return 'asr-qwen3-official';
    }

    const hasCtcTokenizerFile = CTC_ASR_TOKENIZER_FILES.some((file) => fileSet.has(file));
    const hasCtcLayout = hasCtcTokenizerFile && CTC_ASR_BASE_REQUIRED_FILES.every((file) => fileSet.has(file));
    if (hasCtcLayout) {
      return 'asr-ctc';
    }

    if (ASR_REQUIRED_FILES.every((file) => fileSet.has(file))) {
      return 'asr-whisper';
    }

    return null;
  }

  const hasShared = TRANSLATE_SHARED_REQUIRED_FILES.every((file) => fileSet.has(file));
  if (!hasShared) return null;
  if (TRANSLATE_VLM_REQUIRED_FILES.every((file) => fileSet.has(file))) {
    return 'translate-vlm';
  }
  if (TRANSLATE_SEQ2SEQ_REQUIRED_FILES.every((file) => fileSet.has(file))) {
    return 'translate-seq2seq';
  }
  if (TRANSLATE_LLM_REQUIRED_FILES.every((file) => fileSet.has(file))) {
    return 'translate-llm';
  }
  return null;
}

function inferRuntimeFromLayout(layout: LocalModelRuntimeLayout | null): LocalModelRuntime | null {
  switch (layout) {
    case 'asr-whisper':
      return 'openvino-whisper-node';
    case 'asr-ctc':
      return 'openvino-ctc-asr';
    case 'asr-qwen3-official':
      return 'openvino-qwen3-asr';
    case 'translate-llm':
      return 'openvino-llm-node';
    case 'translate-seq2seq':
      return 'openvino-seq2seq-translate';
    case 'translate-vlm':
      return 'openvino-llm-node';
    default:
      return null;
  }
}

function normalizeRuntimeValue(input: any, runtimeLayout: LocalModelRuntimeLayout | null): LocalModelRuntime | null {
  if (input.runtime === 'openvino-qwen3-asr') {
    return 'openvino-qwen3-asr';
  }
  if (
    input.runtime === 'openvino-whisper-node' ||
    input.runtime === 'openvino-ctc-asr' ||
    input.runtime === 'openvino-seq2seq-translate' ||
    input.runtime === 'openvino-llm-node'
  ) {
    return input.runtime;
  }
  return inferRuntimeFromLayout(runtimeLayout);
}

function inferRuntimeLayoutFromStoredDefinition(input: any): LocalModelRuntimeLayout | null {
  const fileSet = new Set<string>(
    Array.isArray(input?.requiredFiles)
      ? input.requiredFiles.map((file: any) => String(file || '').trim()).filter(Boolean)
      : []
  );
  const type = input?.type === 'asr' || input?.type === 'translate' ? input.type : null;
  if (!type || fileSet.size === 0) return null;
  return inferRuntimeLayoutFromArtifacts(type, fileSet);
}

function inferDirectSourceFormatFromArtifacts(fileSet: Set<string>) {
  const hasOpenvinoArtifacts =
    hasFile(fileSet, /^openvino_.*\.(xml|bin)$/i) ||
    hasFile(fileSet, /^thinker\/openvino_.*\.(xml|bin)$/i) ||
    hasFile(fileSet, /^.*\.xml$/i);
  return hasOpenvinoArtifacts ? 'openvino-ir' : 'unknown';
}

function inferSourceFormatFromStoredDefinition(input: any, fileSet: Set<string>): LocalModelSourceFormat {
  const runtimeLayout = inferRuntimeLayoutFromStoredDefinition(input);
  if (runtimeLayout) return 'openvino-ir';
  return inferDirectSourceFormatFromArtifacts(fileSet);
}

function inferConversionMethodFromStoredDefinition(
  input: any,
  sourceFormat: LocalModelSourceFormat,
  runtimeLayout: LocalModelRuntimeLayout | null
): LocalModelConversionMethod {
  if (input?.conversionMethod === 'direct-download' ||
      input?.conversionMethod === 'openvino-convert-model' ||
      input?.conversionMethod === 'optimum-export-openvino' ||
      input?.conversionMethod === 'openvino-ctc-asr-export' ||
      input?.conversionMethod === 'openvino-qwen3-asr-export' ||
      input?.conversionMethod === 'unsupported') {
    return input.conversionMethod;
  }

  if (input?.installMode === 'hf-qwen3-asr-convert') {
    return 'openvino-qwen3-asr-export';
  }
  if (input?.installMode === 'hf-auto-convert') {
    if (
      runtimeLayout === 'translate-seq2seq' &&
      (sourceFormat === 'onnx' || sourceFormat === 'tensorflow-lite' || sourceFormat === 'paddle')
    ) {
      return 'unsupported';
    }
    return sourceFormat === 'pytorch' || sourceFormat === 'jax-flax' || sourceFormat === 'keras'
      ? 'optimum-export-openvino'
      : 'openvino-convert-model';
  }
  if (sourceFormat === 'openvino-ir') {
    return 'direct-download';
  }
  return 'unsupported';
}

function detectSourceFormatFromHfMetadata(metadata: HuggingFaceModelMetadata): LocalModelSourceFormat {
  const allFiles = collectSiblingFiles(metadata);
  const fileSet = new Set(allFiles);
  const tags = normalizeTagSet(metadata);
  const libraryName = String(metadata?.library_name || '').trim().toLowerCase();

  if (
    tags.has('openvino') ||
    hasFile(fileSet, /^openvino_.*\.(xml|bin)$/i) ||
    hasFile(fileSet, /^thinker\/openvino_.*\.(xml|bin)$/i)
  ) {
    return 'openvino-ir';
  }
  if (tags.has('gguf') || hasFile(fileSet, /\.gguf$/i)) {
    return 'gguf';
  }
  if (hasFile(fileSet, /\.onnx$/i)) {
    return 'onnx';
  }
  if (hasFile(fileSet, /\.tflite$/i)) {
    return 'tensorflow-lite';
  }
  if (hasFile(fileSet, /\.pdmodel$/i) || hasFile(fileSet, /\.pdiparams$/i)) {
    return 'paddle';
  }
  if (libraryName === 'keras' || hasFile(fileSet, /\.keras$/i)) {
    return 'keras';
  }
  if (
    hasFile(fileSet, /(^|\/)saved_model\.pb$/i) ||
    hasFile(fileSet, /\.meta$/i) ||
    hasFile(fileSet, /\.index$/i) ||
    hasFile(fileSet, /(^|\/)variables\//i)
  ) {
    return 'tensorflow';
  }
  if (hasFile(fileSet, /\.pt2$/i) || hasFile(fileSet, /(?:script|traced|exported).*\.(pt|pth)$/i)) {
    return 'pytorch';
  }
  if (hasFile(fileSet, /(^|\/)flax_model\.msgpack$/i) || tags.has('jax') || tags.has('flax')) {
    return 'jax-flax';
  }
  const hasPyTorchWeights =
    hasFile(fileSet, /(^|\/)(model\.safetensors|pytorch_model\.bin)$/i) ||
    hasFile(fileSet, /\.(pt2|pt|pth)$/i) ||
    tags.has('pytorch') ||
    tags.has('safetensors');
  const hasTensorFlowWeights =
    hasFile(fileSet, /(^|\/)tf_model\.h5$/i) ||
    hasFile(fileSet, /(^|\/)saved_model\.pb$/i) ||
    hasFile(fileSet, /\.meta$/i) ||
    hasFile(fileSet, /\.index$/i) ||
    hasFile(fileSet, /(^|\/)variables\//i) ||
    tags.has('tf') ||
    tags.has('tensorflow');
  if (
    hasPyTorchWeights
  ) {
    return 'pytorch';
  }
  if (hasTensorFlowWeights) {
    if (libraryName === 'transformers' && hasPyTorchWeights) {
      return 'pytorch';
    }
    return 'tensorflow';
  }
  if (hasFile(fileSet, /\.h5$/i)) {
    return libraryName === 'keras' ? 'keras' : 'tensorflow';
  }
  if (hasFile(fileSet, /\.pb$/i)) {
    return 'tensorflow';
  }
  return 'unknown';
}

function guessRuntimeLayoutFromMetadata(
  type: LocalModelType,
  repoId: string,
  metadata: HuggingFaceModelMetadata
): LocalModelRuntimeLayout {
  if (type === 'asr') {
    if (isOfficialQwen3AsrRepo(repoId)) {
      return 'asr-qwen3-official';
    }
    const modelType = String(metadata?.config?.model_type || '').trim().toLowerCase();
    const architectures = Array.isArray(metadata?.config?.architectures)
      ? metadata!.config!.architectures!.map((item) => String(item || '').trim().toLowerCase())
      : [];
    const ctcLikeArchitecture = architectures.some((item) => /ctc/.test(item));
    if (/wav2vec2|hubert|wavlm|unispeech|sew|data2vec-audio/.test(modelType) || ctcLikeArchitecture) {
      return 'asr-ctc';
    }
    return 'asr-whisper';
  }

  const pipelineTag = String(metadata?.pipeline_tag || '').trim().toLowerCase();
  const modelType = String(metadata?.config?.model_type || '').trim().toLowerCase();
  const tags = normalizeTagSet(metadata);

  if (
    pipelineTag === 'image-text-to-text' ||
    tags.has('image-text-to-text') ||
    /(?:vision|vl|internvl|llava|paligemma)/i.test(modelType)
  ) {
    return 'translate-vlm';
  }

  const architectures = Array.isArray(metadata?.config?.architectures)
    ? metadata!.config!.architectures!.map((item) => String(item || '').trim().toLowerCase())
    : [];
  const seq2seqLikeArchitecture = architectures.some((item) =>
    /(conditionalgeneration|forconditionalgeneration|forconditionalgeneration|fortranslation|forseq2seq)/.test(item)
  );
  if (
    pipelineTag === 'text2text-generation' ||
    pipelineTag === 'translation' ||
    /(?:^|[-_])(t5|mt5|umt5|bart|mbart|marian|pegasus|m2m100|m2m_100|fsmt|prophetnet)(?:$|[-_])/i.test(modelType) ||
    seq2seqLikeArchitecture
  ) {
    return 'translate-seq2seq';
  }

  return 'translate-llm';
}

function isSupportedTranslatePipelineTag(pipelineTag: string) {
  if (!pipelineTag) return true;
  return (
    pipelineTag === 'text-generation' ||
    pipelineTag === 'text2text-generation' ||
    pipelineTag === 'translation' ||
    pipelineTag === 'conversational' ||
    pipelineTag === 'image-text-to-text'
  );
}

function inferConversionMethodFromHfMetadata(
  type: LocalModelType,
  repoId: string,
  metadata: HuggingFaceModelMetadata,
  sourceFormat: LocalModelSourceFormat
): LocalModelConversionMethod {
  const runtimeLayout = guessRuntimeLayoutFromMetadata(type, repoId, metadata);
  if (sourceFormat === 'openvino-ir') {
    return 'direct-download';
  }
  if (type === 'asr' && isOfficialQwen3AsrRepo(repoId)) {
    return 'openvino-qwen3-asr-export';
  }
  if (sourceFormat === 'gguf') {
    return 'unsupported';
  }

  const libraryName = String(metadata?.library_name || '').trim().toLowerCase();
  const fileSet = new Set(collectSiblingFiles(metadata));
  const looksLikeHfCheckpoint =
    libraryName === 'transformers' ||
    libraryName === 'diffusers' ||
    hasFile(fileSet, /(^|\/)(model\.safetensors|pytorch_model\.bin|flax_model\.msgpack|tf_model\.h5)$/i);

  const baseModelRepoId =
    runtimeLayout === 'asr-whisper'
      ? inferWhisperBaseModelRepoId(repoId, metadata)
      : extractBaseModelRepoId(metadata);

  if (
    type === 'asr' &&
    runtimeLayout === 'asr-whisper' &&
    (sourceFormat === 'onnx' || sourceFormat === 'tensorflow-lite') &&
    baseModelRepoId &&
    /whisper/i.test(baseModelRepoId)
  ) {
    return 'optimum-export-openvino';
  }

  if (
    type === 'asr' &&
    runtimeLayout === 'asr-ctc' &&
    looksLikeHfCheckpoint &&
    (sourceFormat === 'pytorch' || sourceFormat === 'tensorflow' || sourceFormat === 'keras' || sourceFormat === 'jax-flax')
  ) {
    return 'openvino-ctc-asr-export';
  }

  if (
    type === 'translate' &&
    runtimeLayout === 'translate-seq2seq' &&
    looksLikeHfCheckpoint &&
    (sourceFormat === 'pytorch' || sourceFormat === 'tensorflow' || sourceFormat === 'keras' || sourceFormat === 'jax-flax')
  ) {
    return 'optimum-export-openvino';
  }

  if (looksLikeHfCheckpoint && (sourceFormat === 'pytorch' || sourceFormat === 'jax-flax' || sourceFormat === 'keras')) {
    return 'optimum-export-openvino';
  }
  if (looksLikeHfCheckpoint && sourceFormat === 'tensorflow') {
    return 'optimum-export-openvino';
  }
  if (
    runtimeLayout === 'translate-seq2seq' &&
    (sourceFormat === 'onnx' || sourceFormat === 'tensorflow-lite' || sourceFormat === 'paddle')
  ) {
    return 'unsupported';
  }

  if (
    sourceFormat === 'onnx' ||
    sourceFormat === 'tensorflow' ||
    sourceFormat === 'tensorflow-lite' ||
    sourceFormat === 'paddle'
  ) {
    return 'openvino-convert-model';
  }
  return 'unsupported';
}

function createBuiltInModel(
  definition: Omit<LocalModelDefinition, 'source' | 'runtimeLayout' | 'sourceFormat' | 'conversionMethod'> & {
    runtimeLayout?: LocalModelRuntimeLayout;
    sourceFormat?: LocalModelSourceFormat;
    conversionMethod?: LocalModelConversionMethod;
  }
): LocalModelDefinition {
  const runtimeLayout =
    definition.runtimeLayout ||
    inferRuntimeLayoutFromArtifacts(definition.type, new Set(definition.requiredFiles)) ||
    (definition.type === 'asr' ? 'asr-whisper' : 'translate-llm');
  return {
    ...definition,
    runtimeLayout,
    sourceFormat: definition.sourceFormat || 'openvino-ir',
    conversionMethod: definition.conversionMethod || 'direct-download',
    source: 'builtin',
  };
}

function buildDefinitionFromArtifactFiles(
  type: LocalModelType,
  repoId: string,
  fileSet: Set<string>,
  overrides?: Partial<
    Pick<
      LocalModelDefinition,
      | 'id'
      | 'displayName'
      | 'localSubdir'
      | 'downloadRepoId'
      | 'installMode'
      | 'sourceFormat'
      | 'conversionMethod'
      | 'source'
      | 'runtimeHints'
    >
  >
): LocalModelDefinition | null {
  const runtimeLayout = inferRuntimeLayoutFromArtifacts(type, fileSet);
  const runtime = inferRuntimeFromLayout(runtimeLayout);
  if (!runtimeLayout || !runtime) return null;

  const requiredFiles = getRequiredFilesForRuntimeLayout(runtimeLayout, fileSet);
  if (requiredFiles.length === 0) return null;

  return {
    id: overrides?.id || buildLocalModelId(type, repoId),
    type,
    displayName: overrides?.displayName || buildDefaultDisplayName(repoId),
    repoId,
    downloadRepoId: overrides?.downloadRepoId || undefined,
    localSubdir: overrides?.localSubdir || buildLocalModelSubdir(repoId),
    requiredFiles,
    runtime,
    runtimeLayout,
    installMode: overrides?.installMode || 'hf-direct',
    sourceFormat: overrides?.sourceFormat || inferDirectSourceFormatFromArtifacts(fileSet),
    conversionMethod: overrides?.conversionMethod || 'direct-download',
    device: 'AUTO',
    source: overrides?.source || 'custom',
    ...(overrides?.runtimeHints ? { runtimeHints: overrides.runtimeHints } : {}),
  };
}

function listRelativeFilesRecursively(rootDir: string) {
  const collected: string[] = [];

  const visit = (currentDir: string, relativePrefix = '') => {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });
    for (const entry of entries) {
      const nextRelative = relativePrefix ? `${relativePrefix}/${entry.name}` : entry.name;
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        visit(fullPath, nextRelative);
      } else if (entry.isFile()) {
        collected.push(nextRelative.replace(/\\/g, '/'));
      }
    }
  };

  visit(rootDir);
  return collected;
}

export const BUILTIN_LOCAL_MODELS: LocalModelDefinition[] = [
  createBuiltInModel({
    id: 'local_asr_openvino_whisper_large_v3_int8_ov',
    type: 'asr',
    displayName: 'OpenVINO Whisper Large v3 INT8',
    repoId: 'OpenVINO/whisper-large-v3-int8-ov',
    localSubdir: 'openvino/whisper-large-v3-int8-ov',
    requiredFiles: [
      'config.json',
      'generation_config.json',
      'preprocessor_config.json',
      'openvino_encoder_model.xml',
      'openvino_encoder_model.bin',
      'openvino_decoder_model.xml',
      'openvino_decoder_model.bin',
      'openvino_tokenizer.xml',
      'openvino_tokenizer.bin',
      'openvino_detokenizer.xml',
      'openvino_detokenizer.bin',
      'tokenizer.json',
      'tokenizer_config.json',
      'special_tokens_map.json',
      'added_tokens.json',
      'normalizer.json',
      'vocab.json',
      'merges.txt',
      'openvino_config.json',
    ],
    runtime: 'openvino-whisper-node',
    installMode: 'hf-direct',
    device: 'AUTO',
  }),
  createBuiltInModel({
    id: 'local_asr_openvino_whisper_large_v3_int4_ov',
    type: 'asr',
    displayName: 'OpenVINO Whisper Large v3 INT4',
    repoId: 'OpenVINO/whisper-large-v3-int4-ov',
    localSubdir: 'openvino/whisper-large-v3-int4-ov',
    requiredFiles: [
      'config.json',
      'generation_config.json',
      'preprocessor_config.json',
      'openvino_encoder_model.xml',
      'openvino_encoder_model.bin',
      'openvino_decoder_model.xml',
      'openvino_decoder_model.bin',
      'openvino_tokenizer.xml',
      'openvino_tokenizer.bin',
      'openvino_detokenizer.xml',
      'openvino_detokenizer.bin',
      'tokenizer.json',
      'tokenizer_config.json',
      'special_tokens_map.json',
      'added_tokens.json',
      'normalizer.json',
      'vocab.json',
      'merges.txt',
      'openvino_config.json',
    ],
    runtime: 'openvino-whisper-node',
    installMode: 'hf-direct',
    device: 'AUTO',
  }),
  createBuiltInModel({
    id: 'local_translate_openvino_qwen3_8b_int4_ov',
    type: 'translate',
    displayName: 'OpenVINO Qwen3 8B INT4',
    repoId: 'OpenVINO/Qwen3-8B-int4-ov',
    localSubdir: 'openvino/Qwen3-8B-int4-ov',
    requiredFiles: [
      'config.json',
      'generation_config.json',
      'openvino_model.xml',
      'openvino_model.bin',
      'openvino_tokenizer.xml',
      'openvino_tokenizer.bin',
      'openvino_detokenizer.xml',
      'openvino_detokenizer.bin',
      'tokenizer.json',
      'tokenizer_config.json',
      'special_tokens_map.json',
      'added_tokens.json',
      'vocab.json',
      'merges.txt',
      'chat_template.jinja',
      'openvino_config.json',
    ],
    runtime: 'openvino-llm-node',
    installMode: 'hf-direct',
    device: 'AUTO',
  }),
  createBuiltInModel({
    id: 'local_translate_openvino_qwen2_5_7b_instruct_int4_ov',
    type: 'translate',
    displayName: 'OpenVINO Qwen2.5 7B Instruct INT4',
    repoId: 'OpenVINO/Qwen2.5-7B-Instruct-int4-ov',
    localSubdir: 'openvino/Qwen2.5-7B-Instruct-int4-ov',
    requiredFiles: [
      'config.json',
      'generation_config.json',
      'openvino_model.xml',
      'openvino_model.bin',
      'openvino_tokenizer.xml',
      'openvino_tokenizer.bin',
      'openvino_detokenizer.xml',
      'openvino_detokenizer.bin',
      'tokenizer.json',
      'tokenizer_config.json',
      'special_tokens_map.json',
      'added_tokens.json',
      'vocab.json',
      'merges.txt',
    ],
    runtime: 'openvino-llm-node',
    installMode: 'hf-direct',
    device: 'AUTO',
  }),
  createBuiltInModel({
    id: 'local_translate_openvino_qwen2_5_1_5b_instruct_int4_ov',
    type: 'translate',
    displayName: 'OpenVINO Qwen2.5 1.5B Instruct INT4',
    repoId: 'OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov',
    localSubdir: 'openvino/Qwen2.5-1.5B-Instruct-int4-ov',
    requiredFiles: [
      'config.json',
      'generation_config.json',
      'openvino_model.xml',
      'openvino_model.bin',
      'openvino_tokenizer.xml',
      'openvino_tokenizer.bin',
      'openvino_detokenizer.xml',
      'openvino_detokenizer.bin',
      'tokenizer.json',
      'tokenizer_config.json',
      'special_tokens_map.json',
      'added_tokens.json',
      'vocab.json',
      'merges.txt',
    ],
    runtime: 'openvino-llm-node',
    installMode: 'hf-direct',
    device: 'AUTO',
  }),
  createBuiltInModel({
    id: 'local_translate_openvino_gemma_3_4b_it_int4_ov',
    type: 'translate',
    displayName: 'OpenVINO Gemma 3 4B IT INT4',
    repoId: 'OpenVINO/gemma-3-4b-it-int4-ov',
    localSubdir: 'openvino/gemma-3-4b-it-int4-ov',
    requiredFiles: [
      'config.json',
      'generation_config.json',
      'preprocessor_config.json',
      'openvino_language_model.xml',
      'openvino_language_model.bin',
      'openvino_text_embeddings_model.xml',
      'openvino_text_embeddings_model.bin',
      'openvino_vision_embeddings_model.xml',
      'openvino_vision_embeddings_model.bin',
      'openvino_tokenizer.xml',
      'openvino_tokenizer.bin',
      'openvino_detokenizer.xml',
      'openvino_detokenizer.bin',
      'tokenizer.json',
      'tokenizer_config.json',
      'special_tokens_map.json',
      'added_tokens.json',
      'chat_template.jinja',
      'openvino_config.json',
    ],
    runtime: 'openvino-llm-node',
    installMode: 'hf-direct',
    device: 'AUTO',
  }),
  createBuiltInModel({
    id: 'local_translate_openvino_qwen3_1_7b_int8_ov',
    type: 'translate',
    displayName: 'OpenVINO Qwen3 1.7B INT8',
    repoId: 'OpenVINO/Qwen3-1.7B-int8-ov',
    localSubdir: 'openvino/Qwen3-1.7B-int8-ov',
    requiredFiles: [
      'added_tokens.json',
      'config.json',
      'generation_config.json',
      'merges.txt',
      'openvino_detokenizer.bin',
      'openvino_detokenizer.xml',
      'openvino_model.bin',
      'openvino_model.xml',
      'openvino_tokenizer.bin',
      'openvino_tokenizer.xml',
      'special_tokens_map.json',
      'tokenizer.json',
      'tokenizer_config.json',
      'vocab.json',
    ],
    runtime: 'openvino-llm-node',
    installMode: 'hf-direct',
    device: 'AUTO',
  }),
];

function toFiniteNumber(input: unknown, min?: number, max?: number) {
  const value = Number(input);
  if (!Number.isFinite(value)) return undefined;
  const rounded = Number.isInteger(value) ? value : Number(value.toFixed(4));
  if (typeof min === 'number' && rounded < min) return undefined;
  if (typeof max === 'number' && rounded > max) return undefined;
  return rounded;
}

function sanitizeRuntimeHintEvidence(input: unknown): LocalModelRuntimeHintEvidence | null {
  if (!input || typeof input !== 'object') return null;
  const raw = input as Record<string, unknown>;
  const source = typeof raw.source === 'string' ? raw.source.trim() : '';
  if (!source) return null;
  const confidence =
    raw.confidence === 'high' || raw.confidence === 'medium' || raw.confidence === 'low'
      ? raw.confidence
      : undefined;
  const value: LocalModelRuntimeHintEvidence['value'] | undefined =
    typeof raw.value === 'string' ||
    typeof raw.value === 'number' ||
    typeof raw.value === 'boolean' ||
    raw.value === null
      ? (raw.value as LocalModelRuntimeHintEvidence['value'])
      : undefined;
  return {
    source,
    ...(typeof raw.key === 'string' && raw.key.trim() ? { key: raw.key.trim() } : {}),
    ...(value !== undefined ? { value } : {}),
    ...(confidence ? { confidence } : {}),
  };
}

function sanitizeRuntimeHints(input: unknown): LocalModelRuntimeHints | undefined {
  if (!input || typeof input !== 'object') return undefined;
  const raw = input as Record<string, any>;
  const generation = raw.generation && typeof raw.generation === 'object' ? raw.generation : null;
  const chatTemplate = raw.chatTemplate && typeof raw.chatTemplate === 'object' ? raw.chatTemplate : null;
  const batching = raw.batching && typeof raw.batching === 'object' ? raw.batching : null;
  const asr = raw.asr && typeof raw.asr === 'object' ? raw.asr : null;
  const modelCard = raw.modelCard && typeof raw.modelCard === 'object' ? raw.modelCard : null;
  const evidence = Array.isArray(raw.evidence)
    ? raw.evidence.map((item: unknown) => sanitizeRuntimeHintEvidence(item)).filter(Boolean) as LocalModelRuntimeHintEvidence[]
    : [];

  const sanitized: LocalModelRuntimeHints = {};
  if (typeof raw.inspectedAt === 'string' && raw.inspectedAt.trim()) sanitized.inspectedAt = raw.inspectedAt.trim();
  if (typeof raw.hfSha === 'string' && raw.hfSha.trim()) sanitized.hfSha = raw.hfSha.trim();
  const contextWindow = toFiniteNumber(raw.contextWindow, 1, 10_000_000);
  const maxInputTokens = toFiniteNumber(raw.maxInputTokens, 1, 10_000_000);
  const maxOutputTokens = toFiniteNumber(raw.maxOutputTokens, 1, 1_000_000);
  if (contextWindow != null) sanitized.contextWindow = Math.round(contextWindow);
  if (maxInputTokens != null) sanitized.maxInputTokens = Math.round(maxInputTokens);
  if (maxOutputTokens != null) sanitized.maxOutputTokens = Math.round(maxOutputTokens);

  if (modelCard) {
    const baseModel = Array.isArray(modelCard.baseModel)
      ? modelCard.baseModel.map((item: unknown) => String(item || '').trim()).filter(Boolean)
      : typeof modelCard.baseModel === 'string' && modelCard.baseModel.trim()
        ? modelCard.baseModel.trim()
        : null;
    sanitized.modelCard = {
      license: typeof modelCard.license === 'string' && modelCard.license.trim() ? modelCard.license.trim() : null,
      baseModel,
      pipelineTag: typeof modelCard.pipelineTag === 'string' && modelCard.pipelineTag.trim() ? modelCard.pipelineTag.trim() : null,
      libraryName: typeof modelCard.libraryName === 'string' && modelCard.libraryName.trim() ? modelCard.libraryName.trim() : null,
      summary: typeof modelCard.summary === 'string' && modelCard.summary.trim() ? modelCard.summary.trim().slice(0, 1200) : null,
    };
  }

  if (generation) {
    const nextGeneration: NonNullable<LocalModelRuntimeHints['generation']> = {};
    if (typeof generation.doSample === 'boolean') nextGeneration.doSample = generation.doSample;
    const temperature = toFiniteNumber(generation.temperature, 0, 2);
    const topP = toFiniteNumber(generation.topP, 0, 1);
    const topK = toFiniteNumber(generation.topK, 0, 1000);
    const minP = toFiniteNumber(generation.minP, 0, 1);
    const repetitionPenalty = toFiniteNumber(generation.repetitionPenalty, 0, 10);
    const presencePenalty = toFiniteNumber(generation.presencePenalty, -2, 2);
    const frequencyPenalty = toFiniteNumber(generation.frequencyPenalty, -2, 2);
    const noRepeatNgramSize = toFiniteNumber(generation.noRepeatNgramSize, 0, 64);
    if (temperature != null) nextGeneration.temperature = temperature;
    if (topP != null) nextGeneration.topP = topP;
    if (topK != null) nextGeneration.topK = Math.round(topK);
    if (minP != null) nextGeneration.minP = minP;
    if (repetitionPenalty != null) nextGeneration.repetitionPenalty = repetitionPenalty;
    if (presencePenalty != null) nextGeneration.presencePenalty = presencePenalty;
    if (frequencyPenalty != null) nextGeneration.frequencyPenalty = frequencyPenalty;
    if (noRepeatNgramSize != null) nextGeneration.noRepeatNgramSize = Math.round(noRepeatNgramSize);
    if (Object.keys(nextGeneration).length > 0) sanitized.generation = nextGeneration;
  }

  if (chatTemplate) {
    const kwargs =
      chatTemplate.kwargs && typeof chatTemplate.kwargs === 'object' && !Array.isArray(chatTemplate.kwargs)
        ? { ...chatTemplate.kwargs }
        : undefined;
    sanitized.chatTemplate = {
      ...(typeof chatTemplate.available === 'boolean' ? { available: chatTemplate.available } : {}),
      ...(typeof chatTemplate.supportsThinking === 'boolean' ? { supportsThinking: chatTemplate.supportsThinking } : {}),
      ...(typeof chatTemplate.defaultEnableThinking === 'boolean'
        ? { defaultEnableThinking: chatTemplate.defaultEnableThinking }
        : {}),
      templateSource:
        typeof chatTemplate.templateSource === 'string' && chatTemplate.templateSource.trim()
          ? chatTemplate.templateSource.trim()
          : null,
      templateHash:
        typeof chatTemplate.templateHash === 'string' && chatTemplate.templateHash.trim()
          ? chatTemplate.templateHash.trim()
          : null,
      ...(kwargs ? { kwargs } : {}),
    };
  }

  if (batching) {
    const inputTokenBudget = toFiniteNumber(batching.inputTokenBudget, 1, 10_000_000);
    const outputTokenBudget = toFiniteNumber(batching.outputTokenBudget, 1, 1_000_000);
    const safetyReserveTokens = toFiniteNumber(batching.safetyReserveTokens, 0, 1_000_000);
    const maxLines = toFiniteNumber(batching.maxLines, 1, 1000);
    const charBudget = toFiniteNumber(batching.charBudget, 1, 1_000_000);
    sanitized.batching = {
      mode: batching.mode === 'fixed_lines' ? 'fixed_lines' : batching.mode === 'token_aware' ? 'token_aware' : undefined,
      ...(inputTokenBudget != null ? { inputTokenBudget: Math.round(inputTokenBudget) } : {}),
      ...(outputTokenBudget != null ? { outputTokenBudget: Math.round(outputTokenBudget) } : {}),
      ...(safetyReserveTokens != null ? { safetyReserveTokens: Math.round(safetyReserveTokens) } : {}),
      ...(maxLines != null ? { maxLines: Math.round(maxLines) } : {}),
      ...(charBudget != null ? { charBudget: Math.round(charBudget) } : {}),
      ...(batching.confidence === 'high' || batching.confidence === 'medium' || batching.confidence === 'low'
        ? { confidence: batching.confidence }
        : {}),
    };
  }

  if (asr) {
    const chunkLengthSec = toFiniteNumber(asr.chunkLengthSec, 1, 3600);
    const samplingRate = toFiniteNumber(asr.samplingRate, 1, 384000);
    const maxTargetPositions = toFiniteNumber(asr.maxTargetPositions, 1, 1000000);
    sanitized.asr = {
      ...(asr.task === 'transcribe' || asr.task === 'translate' ? { task: asr.task } : {}),
      ...(typeof asr.returnTimestamps === 'boolean' ? { returnTimestamps: asr.returnTimestamps } : {}),
      ...(typeof asr.wordTimestamps === 'boolean' ? { wordTimestamps: asr.wordTimestamps } : {}),
      ...(chunkLengthSec != null ? { chunkLengthSec } : {}),
      ...(samplingRate != null ? { samplingRate: Math.round(samplingRate) } : {}),
      ...(maxTargetPositions != null ? { maxTargetPositions: Math.round(maxTargetPositions) } : {}),
      ...(asr.confidence === 'high' || asr.confidence === 'medium' || asr.confidence === 'low'
        ? { confidence: asr.confidence }
        : {}),
    };
  }

  if (evidence.length > 0) sanitized.evidence = evidence.slice(0, 30);
  return Object.keys(sanitized).length > 0 ? sanitized : undefined;
}

export function sanitizeLocalModelDefinition(input: any): LocalModelDefinition | null {
  if (!input || typeof input !== 'object') return null;

  const id = typeof input.id === 'string' ? input.id.trim() : '';
  const type = input.type === 'asr' || input.type === 'translate' ? input.type : null;
  const repoId = typeof input.repoId === 'string' ? input.repoId.trim() : '';
  const rawDisplayName = typeof input.displayName === 'string' ? input.displayName.trim() : '';
  const displayName =
    !rawDisplayName || rawDisplayName.toLowerCase() === repoId.toLowerCase()
      ? buildDefaultDisplayName(repoId)
      : rawDisplayName;
  const downloadRepoId = typeof input.downloadRepoId === 'string' ? input.downloadRepoId.trim() : '';
  const localSubdir = typeof input.localSubdir === 'string' ? input.localSubdir.trim() : '';
  const requiredFiles = Array.isArray(input.requiredFiles)
    ? normalizeFileList(input.requiredFiles.map((file: any) => String(file || '').trim()))
    : [];
  const runtimeLayout =
    input.runtimeLayout === 'asr-whisper' ||
    input.runtimeLayout === 'asr-ctc' ||
    input.runtimeLayout === 'asr-qwen3-official' ||
    input.runtimeLayout === 'translate-llm' ||
    input.runtimeLayout === 'translate-seq2seq' ||
    input.runtimeLayout === 'translate-vlm'
      ? input.runtimeLayout
      : inferRuntimeLayoutFromStoredDefinition({ ...input, type, requiredFiles });
  const runtime = normalizeRuntimeValue(input, runtimeLayout);
  const installMode =
    input.installMode === 'hf-qwen3-asr-convert' || input.installMode === 'hf-direct' || input.installMode === 'hf-auto-convert'
      ? input.installMode
      : 'hf-direct';
  const fileSet = new Set(requiredFiles);
  const sourceFormat =
    input.sourceFormat === 'openvino-ir' ||
    input.sourceFormat === 'onnx' ||
    input.sourceFormat === 'tensorflow' ||
    input.sourceFormat === 'tensorflow-lite' ||
    input.sourceFormat === 'paddle' ||
    input.sourceFormat === 'pytorch' ||
    input.sourceFormat === 'jax-flax' ||
    input.sourceFormat === 'keras' ||
    input.sourceFormat === 'gguf' ||
    input.sourceFormat === 'unknown'
      ? input.sourceFormat
      : inferSourceFormatFromStoredDefinition(input, fileSet);
  const conversionMethod = inferConversionMethodFromStoredDefinition(input, sourceFormat, runtimeLayout);
  const device = input.device === 'AUTO' ? 'AUTO' : null;
  const source = input.source === 'builtin' ? 'builtin' : input.source === 'custom' ? 'custom' : null;
  const runtimeHints = sanitizeRuntimeHints(input.runtimeHints);

  if (
    !id ||
    !type ||
    !displayName ||
    !repoId ||
    !localSubdir ||
    !runtime ||
    !runtimeLayout ||
    !device ||
    !source ||
    requiredFiles.length === 0
  ) {
    return null;
  }

  return {
    id,
    type,
    displayName,
    repoId,
    downloadRepoId: downloadRepoId || undefined,
    localSubdir,
    requiredFiles,
    runtime,
    runtimeLayout,
    installMode,
    sourceFormat,
    conversionMethod,
    device,
    source,
    ...(runtimeHints ? { runtimeHints } : {}),
  };
}

export function getBuiltinLocalModelByRepo(type: LocalModelType, repoId: string) {
  const normalizedRepoId = String(repoId || '').trim().toLowerCase();
  return BUILTIN_LOCAL_MODELS.find(
    (model) => model.type === type && model.repoId.toLowerCase() === normalizedRepoId
  ) || null;
}

export function getLocalModelInstallDir(model: LocalModelDefinition) {
  return path.join(PathManager.getModelsPath(), model.localSubdir);
}

export function buildLocalModelId(type: LocalModelType, repoId: string) {
  const normalized = String(repoId || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return `local_${type}_${normalized}`;
}

export function buildLocalModelSubdir(repoId: string) {
  const [ownerRaw, nameRaw] = String(repoId || '').trim().split('/');
  const owner = String(ownerRaw || '').trim();
  const name = String(nameRaw || '').trim();
  if (!owner || !name) {
    throw new Error('Invalid Hugging Face model id.');
  }
  if (owner.toLowerCase() === 'openvino') {
    return `openvino/${name}`;
  }
  return `openvino/${owner}--${name}`;
}

export function getDefaultLocalModelSelection(): LocalModelSelection {
  return {
    asrSelectedId: '',
    translateSelectedId: '',
    installed: [],
  };
}

export function inferLocalModelDefinitionFromInstalledDir(
  type: LocalModelType,
  repoId: string,
  modelDir: string,
  overrides?: Partial<
    Pick<
      LocalModelDefinition,
      | 'id'
      | 'displayName'
      | 'localSubdir'
      | 'downloadRepoId'
      | 'installMode'
      | 'sourceFormat'
      | 'conversionMethod'
      | 'source'
      | 'runtimeHints'
    >
  >
) {
  if (!fs.existsSync(modelDir) || !fs.statSync(modelDir).isDirectory()) {
    return null;
  }
  const fileSet = new Set(listRelativeFilesRecursively(modelDir));
  return buildDefinitionFromArtifactFiles(type, repoId, fileSet, overrides);
}

export function inferLocalModelDefinitionFromHf(
  type: LocalModelType,
  repoId: string,
  metadata: HuggingFaceModelMetadata
): LocalModelDefinition {
  const builtin = getBuiltinLocalModelByRepo(type, repoId);
  if (builtin) {
    return {
      ...builtin,
      requiredFiles: [...builtin.requiredFiles],
    };
  }

  const allFiles = collectSiblingFiles(metadata);
  const fileSet = new Set(allFiles);
  const pipelineTag = String(metadata?.pipeline_tag || '').trim().toLowerCase();
  const tags = normalizeTagSet(metadata);

  if (type === 'translate' && (pipelineTag === 'automatic-speech-recognition' || tags.has('automatic-speech-recognition'))) {
    throw new Error('This Hugging Face model is an ASR model, not a translation model.');
  }
  if (type === 'asr' && pipelineTag && pipelineTag !== 'automatic-speech-recognition' && !tags.has('automatic-speech-recognition')) {
    throw new Error('This Hugging Face model is not tagged as an ASR model.');
  }

  const directDefinition = buildDefinitionFromArtifactFiles(type, repoId, fileSet, {
    installMode: 'hf-direct',
    sourceFormat: 'openvino-ir',
    conversionMethod: 'direct-download',
    source: 'custom',
  });
  if (directDefinition) {
    return directDefinition;
  }

  if (type === 'translate' && !isSupportedTranslatePipelineTag(pipelineTag)) {
    throw new Error(
      `This Hugging Face model uses pipeline_tag="${pipelineTag}", which is not supported by ArcSub local translation runtime.`
    );
  }

  const sourceFormat = detectSourceFormatFromHfMetadata(metadata);
  const runtimeLayout = guessRuntimeLayoutFromMetadata(type, repoId, metadata);
  const runtime = inferRuntimeFromLayout(runtimeLayout);
  const conversionMethod = inferConversionMethodFromHfMetadata(type, repoId, metadata, sourceFormat);
  const baseModelRepoId =
    runtimeLayout === 'asr-whisper'
      ? inferWhisperBaseModelRepoId(repoId, metadata)
      : extractBaseModelRepoId(metadata);
  const downloadRepoId =
    type === 'asr' &&
    runtimeLayout === 'asr-whisper' &&
    (sourceFormat === 'onnx' || sourceFormat === 'tensorflow-lite') &&
    conversionMethod === 'optimum-export-openvino' &&
    /whisper/i.test(baseModelRepoId)
      ? baseModelRepoId
      : undefined;

  if (!runtime) {
    throw new Error('Unable to infer a supported OpenVINO runtime layout for this Hugging Face model.');
  }

  if (conversionMethod === 'unsupported') {
    if (sourceFormat === 'gguf') {
      throw new Error(
        'This Hugging Face model is packaged as GGUF. ArcSub installs only OpenVINO artifacts and requires non-OV sources to be converted into OpenVINO INT8 first.'
      );
    }
    throw new Error('This Hugging Face model does not expose a currently supported OpenVINO installation path.');
  }

  return {
    id: buildLocalModelId(type, repoId),
    type,
    displayName: buildDefaultDisplayName(repoId),
    repoId,
    downloadRepoId,
    localSubdir: buildLocalModelSubdir(repoId),
    requiredFiles: getRequiredFilesForRuntimeLayout(runtimeLayout),
    runtime,
    runtimeLayout,
    installMode: conversionMethod === 'openvino-qwen3-asr-export' ? 'hf-qwen3-asr-convert' : 'hf-auto-convert',
    sourceFormat,
    conversionMethod,
    device: 'AUTO',
    source: 'custom',
  };
}
