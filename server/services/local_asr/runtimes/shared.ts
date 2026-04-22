import { OpenvinoRuntimeManager } from '../../../openvino_runtime_manager.js';

type AsrWordSegment = {
  text: string;
  start_ts: number;
  end_ts?: number;
  probability?: number;
  speaker?: string;
  source_segment_index?: number;
  token_index?: number;
};

type AsrStructuredSegment = {
  text: string;
  start_ts: number;
  end_ts?: number;
  speaker?: string;
  words?: AsrWordSegment[];
};

type AsrTranscriptChunk = {
  text: string;
  start_ts: number;
  end_ts?: number;
  speaker?: string;
  word_start_index?: number;
  word_end_index?: number;
  source_segment_indices?: number[];
};

type AsrStructuredTranscript = {
  text?: string;
  chunks: AsrTranscriptChunk[];
  segments: AsrStructuredSegment[];
  word_segments: AsrWordSegment[];
  debug?: Record<string, any>;
};

export type LocalAsrProviderResult = AsrStructuredTranscript & {
  meta: Record<string, any>;
};

export interface LocalAsrRuntimeInput {
  filePath: string;
  localModelId: string;
  localModelPath: string;
  localModelRuntime?: string;
  language?: string;
  prompt?: string;
  segmentation?: boolean;
  wordAlignment?: boolean;
  decodePolicy?: {
    pipelineMode?: 'stable' | 'throughput';
    alignmentStrategy?: 'provider-first' | 'alignment-first';
    temperature?: number | null;
    beamSize?: number | null;
    noSpeechThreshold?: number | null;
    conditionOnPreviousText?: boolean | null;
  };
  extractStructuredTranscript: (
    transcript: { text?: string; segments?: any[] },
    language?: string,
    options?: { enableSparseNoSpaceNativeRecovery?: boolean }
  ) => AsrStructuredTranscript;
  toFiniteNumber: (value: any, fallback?: number) => number;
}

export async function runOpenvinoLocalAsr(input: LocalAsrRuntimeInput) {
  return OpenvinoRuntimeManager.transcribeWithLocalAsr({
    modelId: input.localModelId,
    modelPath: input.localModelPath,
    runtime: input.localModelRuntime,
    audioPath: input.filePath,
    language: input.language,
    prompt: input.prompt,
    segmentation: input.segmentation,
    wordAlignment: input.wordAlignment,
    decodePolicy: input.decodePolicy,
  });
}

export function buildStructuredLocalAsrResult(input: {
  localResult: any;
  language?: string;
  extractStructuredTranscript: LocalAsrRuntimeInput['extractStructuredTranscript'];
  toFiniteNumber: LocalAsrRuntimeInput['toFiniteNumber'];
}) {
  const structuredStartedAt = Date.now();
  const structured = input.extractStructuredTranscript(
    {
      text: input.localResult?.text || '',
      segments:
        Array.isArray((input.localResult as any)?.segments) && (input.localResult as any).segments.length > 0
          ? (input.localResult as any).segments
          : Array.isArray(input.localResult?.chunks)
            ? input.localResult.chunks
            : [],
    },
    input.language,
    {
      enableSparseNoSpaceNativeRecovery: true,
    }
  );
  const structuredTranscriptMs = Math.max(0, Date.now() - structuredStartedAt);
  const baseProviderMs = input.toFiniteNumber((input.localResult as any)?.meta?.timing?.providerMs, 0);

  return {
    text: structured.text || String((input.localResult as any)?.text || '').trim(),
    chunks: structured.chunks,
    segments: structured.segments,
    word_segments: structured.word_segments,
    meta: {
      ...(input.localResult?.meta || {}),
      rawWordCount: input.toFiniteNumber((input.localResult as any)?.meta?.rawWordCount, 0),
      nativeWordTimestamps: Boolean((input.localResult as any)?.meta?.nativeWordTimestamps),
      syntheticWordFallbackUsed: Boolean(
        structured.debug?.cjkWordDiagnostics?.syntheticWordAppliedSegments ||
          (structured.word_segments.length > 0 && !Boolean((input.localResult as any)?.meta?.nativeWordTimestamps))
      ),
      cjkWordDiagnostics: structured.debug?.cjkWordDiagnostics || null,
      timing: {
        ...((input.localResult as any)?.meta?.timing || {}),
        structuredTranscriptMs,
        providerMs: baseProviderMs + structuredTranscriptMs,
      },
    },
  } satisfies LocalAsrProviderResult;
}
