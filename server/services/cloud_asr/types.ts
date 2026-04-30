export type CloudAsrProvider =
  | 'openai-whisper'
  | 'whispercpp-inference'
  | 'elevenlabs'
  | 'deepgram'
  | 'gladia'
  | 'github-models'
  | 'google-cloud-speech'
  | 'google-gemini';

export interface CloudAsrProviderCapabilities {
  nativeWordTimestamps: boolean;
  nativeDiarization: boolean;
  supportsBatchAudio: boolean;
  requiresVadTimestamping: boolean;
  bypassesLocalAdvancedProcessing: boolean;
  nativeProviderLabel?: string | null;
  legacyNativeFeatureMetaKey?: string | null;
  vadTimestampSourceMetaKey?: string | null;
  vadTimestampingMetaKey?: string | null;
  maxUploadBytes?: number | null;
  maxDurationSec?: number | null;
  profileId?: string | null;
  profileFamily?: string | null;
}

export interface CloudAsrProviderMatchInput {
  rawUrl: string;
  parsedUrl: URL;
  hostname: string;
  pathname: string;
  modelName: string;
}

export interface CloudAsrConnectionProbe {
  method?: 'GET' | 'POST';
  url?: string;
  headers?: Record<string, string>;
  body?: BodyInit;
  expectedValidation?: 'generic' | 'google' | 'deepgram' | 'gladia' | null;
  expectedCatalogModelId?: string | null;
  timeoutMs?: number;
}

export interface CloudAsrProviderDefinition {
  provider: CloudAsrProvider;
  defaultModel: string;
  capabilities: CloudAsrProviderCapabilities;
  detect(input: CloudAsrProviderMatchInput): boolean;
  buildEndpointUrl(rawUrl: string, model?: string): string;
  buildHeaders(key?: string): Record<string, string>;
  buildConnectionProbe?(input: ResolvedCloudAsrProvider): CloudAsrConnectionProbe;
  preflight?(input: CloudAsrPreflightInput): Promise<CloudAsrPreflightResult> | CloudAsrPreflightResult;
  shouldChunkOnError?(input: CloudAsrChunkErrorInput): boolean;
  runtime?: CloudAsrAdapter;
  request?(input: CloudAsrProviderRequestInput): Promise<CloudAsrProviderResult>;
  batchRequest?(input: CloudAsrBatchRequestInput): Promise<CloudAsrBatchProviderResponse>;
}

export interface ResolveCloudAsrProviderInput {
  url: string;
  modelName?: string;
  model?: string;
}

export interface ResolvedCloudAsrProvider {
  provider: CloudAsrProvider;
  providerKey: CloudAsrProvider;
  endpointUrl: string;
  effectiveModel: string;
  defaultModel: string;
  capabilities: CloudAsrProviderCapabilities;
  profileId: string | null;
  profileFamily: string | null;
}

export interface CloudAsrAdapterRequestOptions {
  language?: string;
  prompt?: string;
  segmentation?: boolean;
  wordAlignment?: boolean;
  vad?: boolean;
  diarization?: boolean;
  diarizationOptions?: {
    mode?: 'auto' | 'fixed' | 'range' | 'many';
    exactSpeakerCount?: number;
    minSpeakers?: number;
    maxSpeakers?: number;
  } | null;
  audioDurationSec?: number | null;
  decodePolicy?: {
    pipelineMode?: 'stable' | 'throughput';
    alignmentStrategy?: 'provider-first' | 'alignment-first';
    temperature?: number | null;
    beamSize?: number | null;
    noSpeechThreshold?: number | null;
    conditionOnPreviousText?: boolean | null;
  };
}

export interface CloudAsrStructuredSegment {
  start_ts: number;
  end_ts?: number;
  text: string;
  speaker?: string;
  words?: CloudAsrWordSegment[];
  [key: string]: any;
}

export interface CloudAsrWordSegment {
  text: string;
  start_ts: number;
  end_ts?: number;
  speaker?: string;
  [key: string]: any;
}

export interface CloudAsrTranscriptChunk {
  start_ts: number;
  end_ts?: number;
  text: string;
  speaker?: string;
  word_start_index?: number;
  word_end_index?: number;
  [key: string]: any;
}

export interface CloudAsrStructuredTranscript {
  text?: string;
  chunks: CloudAsrTranscriptChunk[];
  segments: CloudAsrStructuredSegment[];
  word_segments: CloudAsrWordSegment[];
  debug?: Record<string, any>;
}

export interface CloudAsrProviderResult extends CloudAsrStructuredTranscript {
  meta: Record<string, any>;
}

export interface CloudAsrBatchAudioInput {
  filePath: string;
  index: number;
  audioDurationSec?: number | null;
}

export interface CloudAsrBatchProviderResult {
  input: CloudAsrBatchAudioInput;
  result: CloudAsrProviderResult;
}

export interface CloudAsrBatchProviderResponse {
  results: CloudAsrBatchProviderResult[];
  meta: Record<string, any>;
}

export interface CloudAsrAdapterDeps {
  createAbortSignalWithTimeout(timeoutMs: number, signal?: AbortSignal): { signal: AbortSignal; dispose: () => void };
  extractStructuredTranscript(data: any, rawLanguage?: string): CloudAsrStructuredTranscript;
  disableWordAlignment(
    transcript: CloudAsrStructuredTranscript,
    rawLanguage?: string
  ): CloudAsrStructuredTranscript;
}

export interface CloudAsrAdapterBuildInput {
  filePath: string;
  fileBuffer: Buffer;
  config: any;
  options: CloudAsrAdapterRequestOptions;
  includeLanguage: boolean;
  responseFormat: string;
}

export interface CloudAsrAdapterResultMetaInput {
  normalizedData: any;
  transcript: CloudAsrStructuredTranscript;
  options: CloudAsrAdapterRequestOptions;
  resolvedProvider: ResolvedCloudAsrProvider;
  requestTimeoutMs: number;
}

export interface CloudAsrAdapter {
  provider: CloudAsrProvider;
  getPreferredResponseFormats(options: CloudAsrAdapterRequestOptions): string[];
  buildRequestUrl?(endpointUrl: string, input: CloudAsrAdapterBuildInput): string;
  buildFormData?(input: CloudAsrAdapterBuildInput): FormData;
  buildRawBody?(input: CloudAsrAdapterBuildInput): BodyInit;
  buildJsonBody?(input: CloudAsrAdapterBuildInput): Record<string, unknown>;
  getRequestHeaders?(input: CloudAsrAdapterBuildInput): Record<string, string>;
  normalizeResponse?(data: any): any;
  buildResultMeta?(input: CloudAsrAdapterResultMetaInput): Record<string, unknown>;
}

export interface CloudAsrProviderRequestInput {
  filePath: string;
  resolvedProvider: ResolvedCloudAsrProvider;
  config: any;
  options: CloudAsrAdapterRequestOptions;
  deps: CloudAsrAdapterDeps;
  signal?: AbortSignal;
}

export interface CloudAsrBatchRequestInput {
  audioInputs: CloudAsrBatchAudioInput[];
  resolvedProvider: ResolvedCloudAsrProvider;
  config: any;
  options: CloudAsrAdapterRequestOptions;
  deps: Pick<CloudAsrAdapterDeps, 'createAbortSignalWithTimeout'>;
  signal?: AbortSignal;
}

export interface CloudAsrChunkingPolicy {
  initialChunkSec?: number;
  minChunkSec?: number;
  maxChunks?: number;
  audioFormat?: 'wav' | 'mp3';
  audioBitrate?: string;
  reason?: string;
  profileId?: string | null;
  modelId?: string | null;
}

export interface CloudAsrFileInfo {
  sizeBytes: number | null;
  durationSec: number | null;
}

export interface CloudAsrPreflightInput {
  filePath: string;
  resolvedProvider: ResolvedCloudAsrProvider;
  config: any;
  options: CloudAsrAdapterRequestOptions;
  chunkingPolicy: CloudAsrChunkingPolicy | null;
  getFileInfo(): Promise<CloudAsrFileInfo>;
}

export type CloudAsrPreflightResult =
  | {
      action: 'direct';
      skipGenericPreemptiveChunking?: boolean;
    }
  | {
      action: 'chunk';
      message: string;
      chunkingPolicy: CloudAsrChunkingPolicy;
    }
  | {
      action: 'reject';
      error: Error;
    };

export interface CloudAsrChunkErrorInput {
  error: unknown;
  resolvedProvider: ResolvedCloudAsrProvider;
  config: any;
  options: CloudAsrAdapterRequestOptions;
  chunkingPolicy: CloudAsrChunkingPolicy | null;
}
