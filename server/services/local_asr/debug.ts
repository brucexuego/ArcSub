import type { LocalResolvedAsrProfile } from './profile.js';
import type { RunIssue } from '../../../shared/run_monitor.js';

interface BuildAsrWarningsInput {
  segmentationForcedForDiarization: boolean;
  effectiveSegmentation: boolean;
  providerMeta: any;
  timestampsSynthesized: boolean;
  effectiveVad: boolean;
  vadWindows: Array<{ start: number; end: number }>;
  windowedTranscript: { chunks?: unknown[] } | null;
  allowVadWindowedTranscription: boolean;
  speechSegments: Array<{ start: number; end: number }>;
  alignmentDiagnostics: any;
}

interface BuildAsrDebugInfoInput {
  requestedFeatures: Record<string, any>;
  userPrompt: string | undefined;
  effectivePrompt: string;
  localAsrPromptApplied: boolean;
  providerMeta: any;
  localProfile: LocalResolvedAsrProfile | null;
  effectiveSegmentation: boolean;
  effectiveWordAlignment: boolean;
  effectiveVad: boolean;
  diarizationApplied: boolean;
  result: {
    chunks?: unknown[];
    segments?: unknown[];
    word_segments?: unknown[];
  };
  speechSegments: Array<{ start: number; end: number }>;
  vadWindows: Array<{ start: number; end: number }>;
  diarizationDiagnostics: any;
  alignmentDiagnostics: any;
  resolvedAlignmentLanguage: string | null;
  elapsedMs: number;
  vadMs: number;
  providerWallMs: number;
  alignmentWallMs: number;
  diarizationMs: number;
  alignmentError: string | null;
  vadError: string | null;
  diarizationError: string | null;
  warnings: string[];
  hasUsableChunkTimestamps: (chunks: unknown[]) => boolean;
  toFiniteNumber: (value: any, fallback?: number) => number;
}

export function buildAsrWarningCodes(input: BuildAsrWarningsInput) {
  const warnings: string[] = [];
  if (input.segmentationForcedForDiarization) warnings.push('segmentation_forced_for_diarization');
  if (input.effectiveSegmentation && input.providerMeta?.isWhisperCppInference && !input.timestampsSynthesized) {
    warnings.push('segmentation_ignored_by_provider');
  }
  if (input.timestampsSynthesized) warnings.push('segmentation_timestamp_synthesized');
  if (input.providerMeta?.autoLanguageFallbackUsed) warnings.push('auto_language_retried_without_language');
  if (input.providerMeta?.fileLimitFallbackUsed) warnings.push('provider_file_limit_chunked');
  if (input.effectiveVad && input.vadWindows.length > 0 && input.windowedTranscript?.chunks?.length) {
    warnings.push('vad_segmented_transcription');
  }
  if (input.effectiveVad && input.speechSegments.length > 0 && !input.allowVadWindowedTranscription) {
    warnings.push('local_vad_windowing_disabled');
  }
  if (input.alignmentDiagnostics?.applied) warnings.push('forced_alignment_applied');
  return warnings;
}

export function buildAsrDebugInfo(input: BuildAsrDebugInfoInput) {
  const warningIssues: RunIssue[] = input.warnings.map((code) => ({
    code,
    severity:
      code === 'forced_alignment_applied' || code === 'segmentation_timestamp_synthesized'
        ? 'info'
        : 'warning',
    area:
      code === 'forced_alignment_applied'
        ? 'alignment'
        : code.includes('diarization')
          ? 'diarization'
          : code.includes('vad')
            ? 'runtime'
            : 'provider',
  }));
  const errorIssues: RunIssue[] = [
    input.alignmentError
      ? {
          code: 'asr.alignment.failed',
          severity: 'error',
          area: 'alignment',
          technicalMessage: input.alignmentError,
        }
      : null,
    input.vadError
      ? {
          code: 'asr.vad.failed',
          severity: 'error',
          area: 'runtime',
          technicalMessage: input.vadError,
        }
      : null,
    input.diarizationError
      ? {
          code: 'asr.diarization.failed',
          severity: 'error',
          area: 'diarization',
          technicalMessage: input.diarizationError,
        }
      : null,
  ].filter(Boolean) as RunIssue[];
  const nativeWordCount = input.toFiniteNumber(
    input.providerMeta?.rawWordCount,
    input.toFiniteNumber(input.providerMeta?.cjkWordDiagnostics?.nativeWordCount, 0)
  );
  const syntheticWordCount = input.toFiniteNumber(input.providerMeta?.cjkWordDiagnostics?.syntheticWordCount, 0);
  return {
    requested: input.requestedFeatures,
    prompt: {
      userProvided: Boolean(String(input.userPrompt || '').trim()),
      effectiveProvided: Boolean(String(input.effectivePrompt || '').trim()),
      localAsrPromptApplied: input.localAsrPromptApplied,
    },
    provider: {
      ...(input.providerMeta || {}),
      profileId: input.localProfile?.baseProfile.profileId ?? null,
      profileFamily: input.localProfile?.baseProfile.profileFamily ?? null,
    },
    applied: {
      segmentation: input.effectiveSegmentation && input.hasUsableChunkTimestamps(input.result?.chunks || []),
      wordAlignment: input.effectiveWordAlignment && Array.isArray(input.result.word_segments) && input.result.word_segments.length > 0,
      forcedAlignment: Boolean(input.alignmentDiagnostics?.applied),
      vad: input.effectiveVad && input.speechSegments.length > 0,
      diarization: input.requestedFeatures.diarization && input.diarizationApplied,
      localModelProfileId: input.localProfile?.modelProfile?.id ?? null,
      localBaselineConfidence: input.localProfile?.baseProfile.baseline.baselineConfidence ?? null,
      localBaselineTaskFamily: input.localProfile?.baseProfile.baseline.taskFamily ?? null,
      localFallbackBaseline: input.localProfile?.baseProfile.usedFallbackBaseline ?? false,
    },
    quality: {
      timestampReliability: input.providerMeta?.rawHasTimestamps
        ? 'native'
        : input.alignmentDiagnostics?.applied
          ? 'forced'
          : input.providerMeta?.syntheticWordFallbackUsed
            ? 'synthetic'
            : 'none',
      alignmentConfidence:
        typeof input.alignmentDiagnostics?.avgConfidence === 'number' ? input.alignmentDiagnostics.avgConfidence : null,
      nativeWordCoverage:
        nativeWordCount + syntheticWordCount > 0
          ? Number((nativeWordCount / (nativeWordCount + syntheticWordCount)).toFixed(4))
          : null,
      diarizationConfidence:
        typeof input.diarizationDiagnostics?.selectedPass?.threshold === 'number'
          ? input.diarizationDiagnostics.selectedPass.threshold
          : null,
    },
    stats: {
      chunkCount: Array.isArray(input.result.chunks) ? input.result.chunks.length : 0,
      segmentCount: Array.isArray(input.result.segments) ? input.result.segments.length : 0,
      wordCount: Array.isArray(input.result.word_segments) ? input.result.word_segments.length : 0,
      vadDetectedSegments: input.speechSegments.length,
      vadWindowCount: input.vadWindows.length,
      vadTrimmed: false,
      fileLimitChunkCount: input.toFiniteNumber(input.providerMeta?.fileLimitChunkCount, 0),
      fileLimitFallbackUsed: Boolean(input.providerMeta?.fileLimitFallbackUsed),
      diarizationTaggedChunks: Array.isArray(input.result.chunks)
        ? input.result.chunks.filter((chunk: any) => typeof chunk?.speaker === 'string' && chunk.speaker.trim()).length
        : 0,
      diarizationProvider: input.diarizationDiagnostics?.provider ?? null,
      diarizationSource: input.diarizationDiagnostics?.selectedSource ?? null,
      alignmentAttemptedSegments: input.toFiniteNumber(input.alignmentDiagnostics?.attemptedSegmentCount, 0),
      alignmentAlignedSegments: input.toFiniteNumber(input.alignmentDiagnostics?.alignedSegmentCount, 0),
      alignmentAlignedWordCount: input.toFiniteNumber(input.alignmentDiagnostics?.alignedWordCount, 0),
      nativeWordTimestamps: Boolean(input.providerMeta?.nativeWordTimestamps),
      nativeWordCount,
      syntheticWordCount,
      syntheticWordAppliedSegments: input.toFiniteNumber(input.providerMeta?.cjkWordDiagnostics?.syntheticWordAppliedSegments, 0),
      syntheticWordFallbackUsed: Boolean(input.providerMeta?.syntheticWordFallbackUsed),
      cjkRawWordCount: input.toFiniteNumber(input.providerMeta?.cjkWordDiagnostics?.rawWordCount, 0),
      cjkMergedWordCount: input.toFiniteNumber(input.providerMeta?.cjkWordDiagnostics?.mergedWordCount, 0),
      cjkLexicalWordCount: input.toFiniteNumber(input.providerMeta?.cjkWordDiagnostics?.lexicalWordCount, 0),
      cjkReplacementCharCount: input.toFiniteNumber(input.providerMeta?.cjkWordDiagnostics?.replacementCharCount, 0),
      cjkSplitSegmentCount: input.toFiniteNumber(input.providerMeta?.cjkWordDiagnostics?.splitSegmentCount, 0),
    },
    cjkWordDiagnostics: input.providerMeta?.cjkWordDiagnostics || null,
    alignment: input.alignmentDiagnostics,
    alignmentResolvedLanguage: input.resolvedAlignmentLanguage,
    diarization: input.diarizationDiagnostics,
    diagnostics: {
      alignment: input.alignmentDiagnostics,
      alignmentResolvedLanguage: input.resolvedAlignmentLanguage,
      diarization: input.diarizationDiagnostics,
      cjk: input.providerMeta?.cjkWordDiagnostics || null,
      helperChunking: input.providerMeta?.helperChunking || null,
    },
    warnings: input.warnings,
    warningIssues,
    timing: {
      elapsedMs: input.elapsedMs,
      elapsedSec: Number((input.elapsedMs / 1000).toFixed(3)),
      vadMs: input.vadMs,
      vadSec: Number((input.vadMs / 1000).toFixed(3)),
      providerMs: input.toFiniteNumber(input.providerMeta?.timing?.providerMs, input.providerWallMs),
      providerSec: Number((input.toFiniteNumber(input.providerMeta?.timing?.providerMs, input.providerWallMs) / 1000).toFixed(3)),
      providerAudioDecodeMs: input.toFiniteNumber(input.providerMeta?.timing?.audioDecodeMs, 0),
      providerAudioDecodeSec: Number((input.toFiniteNumber(input.providerMeta?.timing?.audioDecodeMs, 0) / 1000).toFixed(3)),
      providerAudioDecodeCacheHit: Boolean(input.providerMeta?.timing?.audioDecodeCacheHit),
      providerGenerateMs: input.toFiniteNumber(input.providerMeta?.timing?.asrGenerateMs, 0),
      providerGenerateSec: Number((input.toFiniteNumber(input.providerMeta?.timing?.asrGenerateMs, 0) / 1000).toFixed(3)),
      providerLoadMs: input.toFiniteNumber(input.providerMeta?.timing?.pipelineLoadMs, 0),
      providerLoadSec: Number((input.toFiniteNumber(input.providerMeta?.timing?.pipelineLoadMs, 0) / 1000).toFixed(3)),
      structuredTranscriptMs: input.toFiniteNumber(input.providerMeta?.timing?.structuredTranscriptMs, 0),
      structuredTranscriptSec: Number((input.toFiniteNumber(input.providerMeta?.timing?.structuredTranscriptMs, 0) / 1000).toFixed(3)),
      diarizationMs: input.diarizationMs,
      diarizationSec: Number((input.diarizationMs / 1000).toFixed(3)),
      alignmentMs: input.alignmentWallMs,
      alignmentSec: Number((input.alignmentWallMs / 1000).toFixed(3)),
      alignmentElapsedMs: input.toFiniteNumber(input.alignmentDiagnostics?.elapsedMs, 0),
      alignmentElapsedSec: Number((input.toFiniteNumber(input.alignmentDiagnostics?.elapsedMs, 0) / 1000).toFixed(3)),
    },
    errors: {
      alignment: input.alignmentError,
      vad: input.vadError,
      diarization: input.diarizationError,
    },
    errorIssues,
    artifacts: {
      hasTimecodes: input.hasUsableChunkTimestamps(input.result?.chunks || []),
      hasWordSegments: Array.isArray(input.result.word_segments) && input.result.word_segments.length > 0,
      hasSpeakerTags: Array.isArray(input.result.chunks)
        ? input.result.chunks.some((chunk: any) => typeof chunk?.speaker === 'string' && chunk.speaker.trim())
        : false,
    },
  };
}
