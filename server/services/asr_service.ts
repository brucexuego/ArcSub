import fs from 'fs-extra';
import path from 'path';
import { spawn } from 'child_process';
import { PathManager } from '../path_manager.js';
import { resolveToolCommand } from '../runtime_tools.js';
import { SettingsManager } from './settings_manager.js';
import { VadService } from '../vad_service.js';
import { DiarizationOptions, DiarizationService } from '../diarization_service.js';
import { LocalModelService } from './local_model_service.js';
import { getLocalModelInstallDir } from '../local_model_catalog.js';
import { AlignmentService } from '../alignment_service.js';
import { getSegmenterLocale, isNoSpaceLanguage as isPolicyNoSpaceLanguage } from '../language/resolver.js';
import { LanguageAlignmentRegistry } from '../language_alignment/registry.js';
import { genericFallbackSegmentNoSpaceLexicalUnits, normalizeNoSpaceAlignmentText } from '../language_alignment/shared/no_space_utils.js';
import { buildCloudAsrRequestHeaders, requestCloudAsr, requestGeminiCloudAsrBatch } from './cloud_asr_adapter.js';
import { resolveCloudAsrProvider } from './cloud_asr_provider.js';
import { extractErrorMessage, hasAuthFailureHint, hasPayloadValidationHint } from '../http/text_utils.js';
import { buildAsrDebugInfo, buildAsrWarningCodes } from './local_asr/debug.js';
import { resolveLocalAsrProfile } from './local_asr/profile.js';
import { transcribeWithLocalAsrRuntime } from './local_asr/runtime.js';
import { ProjectManager } from './project_manager.js';
import { PROJECT_STATUS } from '../../src/project_status.js';

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

type AsrProviderResult = AsrStructuredTranscript & {
  meta: Record<string, any>;
};

type AsrPipelineMode = 'stable' | 'throughput';

type AsrAlignmentStrategy = 'provider-first' | 'alignment-first';

type AsrDecodePolicy = {
  pipelineMode: AsrPipelineMode;
  alignmentStrategy: AsrAlignmentStrategy;
  temperature: number | null;
  beamSize: number | null;
  noSpeechThreshold: number | null;
  conditionOnPreviousText: boolean | null;
};

type AsrVadWindowTask = {
  index: number;
  start: number;
  end: number;
  durationSec: number;
  bucket: 'short' | 'medium' | 'long';
};

type AsrWindowScheduleStats = {
  enabled: boolean;
  schedulerEnabled: boolean;
  mode: AsrPipelineMode;
  batchSize: number;
  concurrency: number;
  maxRetries: number;
  taskCount: number;
  providerRequestCount?: number;
  geminiMultiAudioBatchSize?: number;
  attemptedCount: number;
  succeededCount: number;
  skippedCount: number;
  retries: number;
};

type AsrCloudChunkingPolicy = {
  initialChunkSec?: number;
  minChunkSec?: number;
  maxChunks?: number;
  audioFormat?: 'wav' | 'mp3';
  audioBitrate?: string;
  reason?: string;
};

export class AsrService {
  private static pipeline: any = null;

  private static formatProjectOriginalSubtitles(result: AsrStructuredTranscript) {
    const rawChunks = Array.isArray(result?.chunks) ? result.chunks : [];
    if (rawChunks.length > 0) {
      const lines = rawChunks.flatMap((chunk: any) =>
        String(chunk?.text ?? chunk?.transcript ?? chunk?.content ?? '')
          .replace(/\r\n/g, '\n')
          .replace(/\r/g, '\n')
          .split('\n')
          .map((line) => line.trim())
          .filter(Boolean)
      );
      if (lines.length > 0) {
        return lines.join('\n');
      }
    }

    return String(result?.text ?? '')
      .replace(/\r\n/g, '\n')
      .replace(/\r/g, '\n')
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
      .join('\n');
  }

  private static throwIfAborted(signal?: AbortSignal) {
    if (!signal?.aborted) return;
    throw new Error('ASR request aborted.');
  }

  private static createAbortSignalWithTimeout(timeoutMs: number, signal?: AbortSignal) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    const combinedSignal =
      signal && typeof AbortSignal.any === 'function'
        ? AbortSignal.any([signal, controller.signal])
        : signal || controller.signal;
    return {
      signal: combinedSignal,
      dispose: () => clearTimeout(timeout),
    };
  }

  private static isAbortError(error: unknown) {
    const message = error instanceof Error ? error.message : String(error || '');
    return /aborted/i.test(message);
  }

  private static getFfmpegBinary() {
    return resolveToolCommand('ffmpeg');
  }

  private static isWhisperCppInferenceUrl(rawUrl: string) {
    try {
      const parsed = new URL(String(rawUrl || ''));
      return parsed.pathname.toLowerCase().includes('/inference');
    } catch {
      return false;
    }
  }

  private static isQwen3AsrLocalModel(input: {
    localModelId?: string | null;
    localModelRuntime?: string | null;
    localModelPath?: string | null;
  }) {
    const signature = [
      input.localModelRuntime,
      input.localModelId,
      input.localModelPath,
    ]
      .map((value) => String(value || '').toLowerCase())
      .join(' ');
    return signature.includes('openvino-qwen3-asr') || signature.includes('qwen3-asr');
  }

  private static getEnvNumber(name: string, fallback: number, min?: number, max?: number) {
    const raw = process.env[name];
    if (typeof raw !== 'string' || !raw.trim()) return fallback;
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) return fallback;
    if (typeof min === 'number' && parsed < min) return min;
    if (typeof max === 'number' && parsed > max) return max;
    return parsed;
  }

  private static getEnvBoolean(name: string, fallback: boolean) {
    const raw = process.env[name];
    if (typeof raw !== 'string') return fallback;
    const normalized = raw.trim().toLowerCase();
    if (!normalized) return fallback;
    if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
    if (['0', 'false', 'no', 'off'].includes(normalized)) return false;
    return fallback;
  }

  private static getEnvTrimmedString(name: string, fallback: string) {
    const raw = process.env[name];
    if (typeof raw !== 'string') return fallback;
    const trimmed = raw.trim();
    return trimmed || fallback;
  }

  private static resolvePipelineMode(rawMode?: string): AsrPipelineMode {
    const requested = String(rawMode || process.env.ASR_PIPELINE_MODE || '')
      .trim()
      .toLowerCase();
    if (requested === 'throughput') return 'throughput';
    return 'stable';
  }

  private static resolveAlignmentStrategy(mode: AsrPipelineMode): AsrAlignmentStrategy {
    const envValue = String(process.env.ASR_ALIGNMENT_STRATEGY || '')
      .trim()
      .toLowerCase();
    if (envValue === 'provider-first') return 'provider-first';
    if (envValue === 'alignment-first') return 'alignment-first';
    return mode === 'stable' ? 'alignment-first' : 'provider-first';
  }

  private static buildDecodePolicy(input: {
    mode: AsrPipelineMode;
    effectiveWordAlignment: boolean;
  }): AsrDecodePolicy {
    const mode = input.mode;
    const stableDefaults = {
      temperature: 0,
      beamSize: 1,
      noSpeechThreshold: 0.5,
      conditionOnPreviousText: true,
    };
    const throughputDefaults = {
      temperature: 0,
      beamSize: 1,
      noSpeechThreshold: 0.5,
      conditionOnPreviousText: false,
    };
    const defaults = mode === 'throughput' ? throughputDefaults : stableDefaults;
    const temperature = this.getEnvNumber(
      mode === 'throughput' ? 'ASR_DECODE_TEMPERATURE_THROUGHPUT' : 'ASR_DECODE_TEMPERATURE_STABLE',
      defaults.temperature,
      0,
      1
    );
    const beamSize = Math.round(
      this.getEnvNumber(
        mode === 'throughput' ? 'ASR_DECODE_BEAM_SIZE_THROUGHPUT' : 'ASR_DECODE_BEAM_SIZE_STABLE',
        defaults.beamSize,
        1,
        10
      )
    );
    const noSpeechThreshold = this.getEnvNumber(
      mode === 'throughput' ? 'ASR_DECODE_NO_SPEECH_THROUGHPUT' : 'ASR_DECODE_NO_SPEECH_STABLE',
      defaults.noSpeechThreshold,
      0,
      1
    );
    const conditionOnPreviousText = this.getEnvBoolean(
      mode === 'throughput'
        ? 'ASR_DECODE_CONDITION_ON_PREV_TEXT_THROUGHPUT'
        : 'ASR_DECODE_CONDITION_ON_PREV_TEXT_STABLE',
      defaults.conditionOnPreviousText
    );
    const alignmentStrategy = this.resolveAlignmentStrategy(mode);
    return {
      pipelineMode: mode,
      alignmentStrategy: input.effectiveWordAlignment ? alignmentStrategy : 'provider-first',
      temperature: Number.isFinite(temperature) ? temperature : null,
      beamSize: Number.isFinite(beamSize) ? beamSize : null,
      noSpeechThreshold: Number.isFinite(noSpeechThreshold) ? noSpeechThreshold : null,
      conditionOnPreviousText:
        typeof conditionOnPreviousText === 'boolean' ? conditionOnPreviousText : null,
    };
  }

  private static getVadWindowBatchSize(mode: AsrPipelineMode) {
    return Math.round(
      this.getEnvNumber(
        mode === 'throughput' ? 'ASR_VAD_WINDOW_BATCH_SIZE_THROUGHPUT' : 'ASR_VAD_WINDOW_BATCH_SIZE_STABLE',
        mode === 'throughput' ? 4 : 1,
        1,
        64
      )
    );
  }

  private static getGeminiMultiAudioBatchSize() {
    return 50;
  }

  private static getVadWindowConcurrency(mode: AsrPipelineMode, useLocalProvider: boolean) {
    const defaultConcurrency = useLocalProvider ? 1 : mode === 'throughput' ? 4 : 1;
    return Math.round(
      this.getEnvNumber(
        mode === 'throughput' ? 'ASR_VAD_WINDOW_CONCURRENCY_THROUGHPUT' : 'ASR_VAD_WINDOW_CONCURRENCY_STABLE',
        defaultConcurrency,
        1,
        16
      )
    );
  }

  private static buildVadWindowTasks(windows: Array<{ start: number; end: number }>) {
    return windows.map((window, index) => {
      const durationSec = Math.max(0, this.toFiniteNumber(window.end, 0) - this.toFiniteNumber(window.start, 0));
      const bucket: AsrVadWindowTask['bucket'] =
        durationSec <= 6 ? 'short' : durationSec <= 20 ? 'medium' : 'long';
      return {
        index,
        start: this.toFiniteNumber(window.start, 0),
        end: this.toFiniteNumber(window.end, 0),
        durationSec,
        bucket,
      } satisfies AsrVadWindowTask;
    });
  }

  private static collapseTranscriptToVadWindow(
    transcript: AsrStructuredTranscript,
    durationSec: number,
    language?: string
  ): AsrStructuredTranscript {
    const text = this.buildVadWindowTranscriptText(transcript, language);
    if (!text) {
      return {
        text: '',
        chunks: [],
        segments: [],
        word_segments: [],
      };
    }

    const endTs = Number(Math.max(0.1, durationSec).toFixed(3));
    const segment = {
      text,
      start_ts: 0,
      end_ts: endTs,
    } satisfies AsrStructuredSegment;
    return {
      text,
      chunks: [{
        text,
        start_ts: 0,
        end_ts: endTs,
        source_segment_indices: [0],
      }],
      segments: [segment],
      word_segments: [],
    };
  }

  private static buildVadWindowTranscriptText(transcript: AsrStructuredTranscript, language?: string) {
    const candidates = [
      this.normalizeVadWindowText((transcript.chunks || []).map((chunk) => chunk?.text || '').filter(Boolean).join(' ')),
      this.normalizeVadWindowText((transcript.segments || []).map((segment) => segment?.text || '').filter(Boolean).join(' ')),
      this.normalizeVadWindowText(transcript.text || ''),
      this.buildTranscriptTextFromChunks(transcript.chunks || [], language),
    ].filter(Boolean);

    if (candidates.length === 0) return '';
    return candidates
      .map((text, index) => ({ text, index, score: this.scoreVadWindowTextCandidate(text) }))
      .sort((a, b) => b.score - a.score || a.index - b.index)[0].text;
  }

  private static normalizeVadWindowText(value: string) {
    const cleaned = this.normalizeChunkText(value);
    if (!cleaned) return '';
    return cleaned
      .replace(/\s+/g, ' ')
      .replace(/([\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af])\s+([\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af])/gu, '$1$2')
      .replace(/\s+([,.;:!?%。、！？])/g, '$1')
      .replace(/([\(\[\{¿¡])\s+/g, '$1')
      .trim();
  }

  private static scoreVadWindowTextCandidate(text: string) {
    const latinWordSpaces = text.match(/[A-Za-z]{2,}\s+[A-Za-z]{2,}/g)?.length ?? 0;
    const latinCamelJoints = text.match(/[a-z][A-Z]/g)?.length ?? 0;
    const longLatinRuns = text.match(/[A-Za-z]{14,}/g)?.length ?? 0;
    return latinWordSpaces * 10 - latinCamelJoints * 3 - longLatinRuns * 2;
  }

  private static chunkTasksByBatchSize(tasks: AsrVadWindowTask[], batchSize: number) {
    if (batchSize <= 1 || tasks.length <= 1) return tasks.map((task) => [task]);
    const buckets = {
      short: tasks.filter((task) => task.bucket === 'short'),
      medium: tasks.filter((task) => task.bucket === 'medium'),
      long: tasks.filter((task) => task.bucket === 'long'),
    };
    const ordered = [...buckets.short, ...buckets.medium, ...buckets.long];
    const batches: AsrVadWindowTask[][] = [];
    for (let index = 0; index < ordered.length; index += batchSize) {
      batches.push(ordered.slice(index, index + batchSize));
    }
    return batches;
  }

  private static async runTaskPool<T>(
    tasks: AsrVadWindowTask[],
    concurrency: number,
    worker: (task: AsrVadWindowTask) => Promise<T>
  ) {
    if (tasks.length === 0) return [] as T[];
    const boundedConcurrency = Math.max(1, Math.min(concurrency, tasks.length));
    const results = new Array<T>(tasks.length);
    let cursor = 0;
    let firstError: unknown = null;
    const runner = async () => {
      while (cursor < tasks.length) {
        if (firstError) return;
        const current = cursor;
        cursor += 1;
        try {
          results[current] = await worker(tasks[current]);
        } catch (error) {
          if (!firstError) {
            firstError = error;
          }
          return;
        }
      }
    };
    await Promise.all(Array.from({ length: boundedConcurrency }, () => runner()));
    if (firstError) {
      throw firstError;
    }
    return results;
  }

  private static dedupeWordSegmentsByBoundary(words: AsrWordSegment[]) {
    const sorted = words
      .map((word) => ({
        ...word,
        start_ts: this.toFiniteNumber(word.start_ts, 0),
        end_ts: Number.isFinite(Number(word.end_ts)) ? this.toFiniteNumber(word.end_ts, 0) : undefined,
        text: this.normalizeWordText(String(word.text || '')),
      }))
      .filter((word) => word.text.length > 0)
      .sort((a, b) => a.start_ts - b.start_ts);
    const deduped: AsrWordSegment[] = [];
    for (const word of sorted) {
      const previous = deduped[deduped.length - 1];
      if (!previous) {
        deduped.push(word);
        continue;
      }
      const sameText = this.normalizeNoSpaceAlignmentText(previous.text) === this.normalizeNoSpaceAlignmentText(word.text);
      const nearBoundary = Math.abs(previous.start_ts - word.start_ts) <= 0.12;
      if (sameText && nearBoundary) {
        const prevEnd = this.toFiniteNumber(previous.end_ts, previous.start_ts);
        const nextEnd = this.toFiniteNumber(word.end_ts, word.start_ts);
        previous.end_ts = Math.max(prevEnd, nextEnd);
        continue;
      }
      deduped.push(word);
    }
    return deduped;
  }

  private static reconcileBoundaryChunks(
    chunks: AsrTranscriptChunk[],
    language?: string,
    overlapSec = 0.18
  ) {
    const sorted = [...chunks]
      .map((chunk) => ({
        ...chunk,
        text: this.normalizeChunkText(chunk.text || ''),
        start_ts: this.toFiniteNumber(chunk.start_ts, 0),
        end_ts: Number.isFinite(Number(chunk.end_ts)) ? this.toFiniteNumber(chunk.end_ts, 0) : undefined,
      }))
      .filter((chunk) => chunk.text.length > 0)
      .sort((a, b) => a.start_ts - b.start_ts);
    const output: AsrTranscriptChunk[] = [];
    for (const chunk of sorted) {
      const previous = output[output.length - 1];
      if (!previous) {
        output.push(chunk);
        continue;
      }
      const previousEnd = this.toFiniteNumber(previous.end_ts, previous.start_ts);
      const gap = chunk.start_ts - previousEnd;
      const normalizedPrevious = this.normalizeNoSpaceAlignmentText(previous.text);
      const normalizedCurrent = this.normalizeNoSpaceAlignmentText(chunk.text);
      const nearBoundary = gap <= overlapSec;
      const sameLine = normalizedPrevious.length > 0 && normalizedPrevious === normalizedCurrent;
      if (nearBoundary && sameLine) {
        const nextEnd = this.toFiniteNumber(chunk.end_ts, chunk.start_ts);
        previous.end_ts = Math.max(previousEnd, nextEnd);
        continue;
      }
      if (chunk.start_ts < previousEnd) {
        chunk.start_ts = Number((previousEnd + 0.02).toFixed(3));
        if (Number.isFinite(Number(chunk.end_ts)) && this.toFiniteNumber(chunk.end_ts, chunk.start_ts) <= chunk.start_ts) {
          chunk.end_ts = Number((chunk.start_ts + 0.1).toFixed(3));
        }
      }
      output.push(chunk);
    }
    return this.mergeDisplayChunks(output, language);
  }

  private static reconcileWindowedTranscript(
    transcript: AsrStructuredTranscript,
    language?: string
  ): AsrStructuredTranscript {
    const overlapSec = this.getEnvNumber('ASR_VAD_RECONCILE_OVERLAP_SEC', 0.18, 0.02, 1.5);
    const reconcileEnabled = this.getEnvBoolean('ASR_VAD_RECONCILE_ENABLED', true);
    if (!reconcileEnabled) {
      return transcript;
    }
    const reconciledChunks = this.reconcileBoundaryChunks(transcript.chunks || [], language, overlapSec);
    const reconciledWords = this.dedupeWordSegmentsByBoundary(transcript.word_segments || []);
    let tokenIndex = 0;
    const normalizedWords = reconciledWords.map((word) => ({
      ...word,
      source_segment_index: undefined,
      token_index: tokenIndex++,
    }));
    const resolvedSegments =
      Array.isArray(transcript.segments) && transcript.segments.length > 0
        ? transcript.segments
            .map((segment) => ({
              ...segment,
              text: this.normalizeChunkText(segment.text || ''),
              start_ts: this.toFiniteNumber(segment.start_ts, 0),
              end_ts: Number.isFinite(Number(segment.end_ts)) ? this.toFiniteNumber(segment.end_ts, 0) : undefined,
            }))
            .filter((segment) => segment.text.length > 0)
            .sort((a, b) => a.start_ts - b.start_ts)
        : reconciledChunks.map((chunk) => ({
            text: chunk.text,
            start_ts: chunk.start_ts,
            end_ts: chunk.end_ts,
          }));
    return {
      ...transcript,
      chunks: reconciledChunks,
      segments: resolvedSegments,
      word_segments: normalizedWords,
      text: this.buildTranscriptTextFromChunks(reconciledChunks, language),
    };
  }

  private static getCloudAsrPreemptiveChunkThresholdBytes() {
    const defaultBytes = 25 * 1024 * 1024;
    return Math.round(this.getEnvNumber('ASR_FILE_LIMIT_BYTES', defaultBytes, 1024 * 1024));
  }

  private static sanitizeAudioBitrate(value: string, fallback = '32k') {
    const normalized = String(value || '').trim().toLowerCase();
    return /^\d{2,3}k$/.test(normalized) ? normalized : fallback;
  }

  private static getPhi4CloudChunkingPolicy(): AsrCloudChunkingPolicy {
    const initialChunkSec = Math.round(this.getEnvNumber('ASR_PHI4_FILE_LIMIT_CHUNK_SEC', 5, 2, 60));
    const minChunkSec = Math.round(this.getEnvNumber('ASR_PHI4_FILE_LIMIT_MIN_CHUNK_SEC', 2, 1, 30));
    return {
      initialChunkSec,
      minChunkSec: Math.min(initialChunkSec, minChunkSec),
      maxChunks: Math.round(this.getEnvNumber('ASR_PHI4_FILE_LIMIT_MAX_CHUNKS', 2000, 1, 10000)),
      audioFormat: 'mp3',
      audioBitrate: this.sanitizeAudioBitrate(this.getEnvTrimmedString('ASR_PHI4_AUDIO_BITRATE', '16k'), '16k'),
      reason: 'github_phi4_8000_token_limit',
    };
  }

  private static getDeepgramCloudChunkingPolicy(): AsrCloudChunkingPolicy {
    const initialChunkSec = Math.round(this.getEnvNumber('ASR_DEEPGRAM_FILE_LIMIT_CHUNK_SEC', 480, 30, 1800));
    const minChunkSec = Math.round(this.getEnvNumber('ASR_DEEPGRAM_FILE_LIMIT_MIN_CHUNK_SEC', 60, 10, 600));
    return {
      initialChunkSec,
      minChunkSec: Math.min(initialChunkSec, minChunkSec),
      maxChunks: Math.round(this.getEnvNumber('ASR_DEEPGRAM_FILE_LIMIT_MAX_CHUNKS', 200, 1, 2000)),
      audioFormat: 'mp3',
      audioBitrate: this.sanitizeAudioBitrate(this.getEnvTrimmedString('ASR_DEEPGRAM_AUDIO_BITRATE', '32k'), '32k'),
      reason: 'deepgram_upload_or_processing_limit',
    };
  }

  private static getGladiaCloudChunkingPolicy(): AsrCloudChunkingPolicy {
    const initialChunkSec = Math.round(this.getEnvNumber('ASR_GLADIA_FILE_LIMIT_CHUNK_SEC', 3600, 600, 8100));
    const minChunkSec = Math.round(this.getEnvNumber('ASR_GLADIA_FILE_LIMIT_MIN_CHUNK_SEC', 600, 60, 1800));
    return {
      initialChunkSec,
      minChunkSec: Math.min(initialChunkSec, minChunkSec),
      maxChunks: Math.round(this.getEnvNumber('ASR_GLADIA_FILE_LIMIT_MAX_CHUNKS', 200, 1, 2000)),
      audioFormat: 'mp3',
      audioBitrate: this.sanitizeAudioBitrate(this.getEnvTrimmedString('ASR_GLADIA_AUDIO_BITRATE', '32k'), '32k'),
      reason: 'gladia_upload_or_duration_limit',
    };
  }

  private static buildVadWindows(
    segments: Array<{ start: number; end: number }>,
    options: { overflowMode?: 'collapse' | 'rebalance' } = {}
  ) {
    const sorted = segments
      .map((s) => ({
        start: this.toFiniteNumber(s?.start, 0),
        end: this.toFiniteNumber(s?.end, 0),
      }))
      .filter((s) => s.end > s.start)
      .sort((a, b) => a.start - b.start);

    if (sorted.length === 0) return [];

    const defaultMinSegmentSec = this.getEnvNumber('VAD_MIN_SPEECH_MS', 1000, 30, 30000) / 1000;
    const defaultMergeGapSec = this.getEnvNumber('VAD_MERGE_GAP_MS', 250, 0, 3000) / 1000;
    const minSegmentSec = this.getEnvNumber('VAD_TRANSCRIBE_MIN_SEGMENT_SEC', defaultMinSegmentSec, 0.05, 10);
    const mergeGapSec = this.getEnvNumber('VAD_TRANSCRIBE_MERGE_GAP_SEC', defaultMergeGapSec, 0, 10);
    const maxWindowSec = this.getEnvNumber('VAD_TRANSCRIBE_MAX_WINDOW_SEC', 120, 5, 1200);
    const maxWindows = Math.round(this.getEnvNumber('VAD_TRANSCRIBE_MAX_WINDOWS', 80, 1, 1000));

    const normalized: Array<{ start: number; end: number }> = [];
    for (const segment of sorted) {
      const duration = segment.end - segment.start;
      if (duration <= maxWindowSec) {
        normalized.push(segment);
        continue;
      }

      let cursor = segment.start;
      while (cursor < segment.end) {
        const nextEnd = Math.min(segment.end, cursor + maxWindowSec);
        if (nextEnd - cursor >= minSegmentSec) {
          normalized.push({ start: cursor, end: nextEnd });
        }
        cursor = nextEnd;
      }
    }

    if (normalized.length === 0) return [];

    const windows: Array<{ start: number; end: number }> = [];
    let current: { start: number; end: number } | null = null;

    for (const segment of normalized) {
      if (segment.end - segment.start < minSegmentSec) {
        continue;
      }

      if (!current) {
        current = { ...segment };
        continue;
      }

      const canMergeByGap = segment.start - current.end <= mergeGapSec;
      const mergedDuration = segment.end - current.start;
      if (canMergeByGap && mergedDuration <= maxWindowSec) {
        current.end = Math.max(current.end, segment.end);
      } else {
        windows.push(current);
        current = { ...segment };
      }
    }

    if (current) windows.push(current);

    if (windows.length > maxWindows && options.overflowMode === 'rebalance') {
      return this.rebalanceVadWindows(windows, maxWindows);
    }

    if (windows.length > maxWindows) {
      return [{ start: windows[0].start, end: windows[windows.length - 1].end }];
    }

    return windows;
  }

  private static rebalanceVadWindows(windows: Array<{ start: number; end: number }>, maxWindows: number) {
    if (windows.length <= maxWindows) return windows;
    const groupSize = Math.max(1, Math.ceil(windows.length / Math.max(1, maxWindows)));
    const rebalanced: Array<{ start: number; end: number }> = [];
    for (let index = 0; index < windows.length; index += groupSize) {
      const group = windows.slice(index, index + groupSize);
      const first = group[0];
      const last = group[group.length - 1];
      if (first && last && last.end > first.start) {
        rebalanced.push({ start: first.start, end: last.end });
      }
    }
    return rebalanced.length > 0 ? rebalanced : [{ start: windows[0].start, end: windows[windows.length - 1].end }];
  }

  private static async extractAudioWindow(
    sourcePath: string,
    startSec: number,
    endSec: number,
    suffix: string,
    options: { audioFormat?: 'wav' | 'mp3'; audioBitrate?: string } = {}
  ) {
    const audioFormat = options.audioFormat === 'mp3' ? 'mp3' : 'wav';
    const outputPath = path.join(path.dirname(sourcePath), `vad_window_${suffix}_${Date.now()}.${audioFormat}`);
    const audioArgs =
      audioFormat === 'mp3'
        ? [
            '-ac', '1',
            '-ar', '16000',
            '-c:a', 'libmp3lame',
            '-b:a', this.sanitizeAudioBitrate(options.audioBitrate || '32k'),
          ]
        : [
            '-ac', '1',
            '-ar', '16000',
            '-c:a', 'pcm_s16le',
          ];
    const args = [
      '-i', sourcePath,
      '-ss', startSec.toFixed(3),
      '-to', endSec.toFixed(3),
      '-vn',
      ...audioArgs,
      '-y',
      outputPath,
    ];

    await new Promise<void>((resolve, reject) => {
      const cp = spawn(this.getFfmpegBinary(), args);
      let stderr = '';
      cp.stderr.on('data', (chunk: Buffer) => {
        stderr += chunk.toString('utf8');
      });
      cp.on('close', (code) => {
        if (code === 0) return resolve();
        reject(new Error(`Audio window extraction failed (${code}): ${stderr.trim()}`));
      });
    });

    const stat = await fs.stat(outputPath);
    if (!stat.isFile() || stat.size < 256) {
      await fs.remove(outputPath).catch(() => {});
      throw new Error(`Audio window is empty (${path.basename(outputPath)}).`);
    }

    return outputPath;
  }

  private static mergeProviderMeta(metaList: any[]) {
    const first = metaList[0] || {};
    const callCount = metaList.reduce((sum: number, m: any) => {
      const nested = this.toFiniteNumber(m?.callCount, 1);
      return sum + Math.max(1, nested);
    }, 0);
    const timingKeys = [
      'providerMs',
      'audioDecodeMs',
      'asrGenerateMs',
      'pipelineLoadMs',
      'structuredTranscriptMs',
      'audioSampleCount',
    ] as const;
    const mergedTiming = timingKeys.reduce<Record<string, number>>((acc, key) => {
      acc[key] = metaList.reduce((sum: number, item: any) => sum + this.toFiniteNumber(item?.timing?.[key], 0), 0);
      return acc;
    }, {});
    const geminiBatchingItems = metaList
      .map((item: any) => item?.geminiMultiAudioBatching)
      .filter((item: any) => item && typeof item === 'object');
    return {
      ...first,
      callCount,
      rawSegmentCount: metaList.reduce((sum: number, m: any) => sum + this.toFiniteNumber(m?.rawSegmentCount, 0), 0),
      rawWordCount: metaList.reduce((sum: number, m: any) => sum + this.toFiniteNumber(m?.rawWordCount, 0), 0),
      rawHasTimestamps: metaList.some((m: any) => Boolean(m?.rawHasTimestamps)),
      autoLanguageFallbackUsed: metaList.some((m: any) => Boolean(m?.autoLanguageFallbackUsed)),
      fileLimitFallbackUsed: metaList.some((m: any) => Boolean(m?.fileLimitFallbackUsed)),
      fileLimitChunkCount: metaList.reduce((sum: number, m: any) => sum + this.toFiniteNumber(m?.fileLimitChunkCount, 0), 0),
      geminiMultiAudioBatching: geminiBatchingItems.length > 0
        ? {
            enabled: true,
            batchCount: geminiBatchingItems.length,
            requestCount: geminiBatchingItems.reduce((sum: number, item: any) => sum + this.toFiniteNumber(item?.requestCount, 0), 0),
            inputCount: geminiBatchingItems.reduce((sum: number, item: any) => sum + this.toFiniteNumber(item?.inputCount, 0), 0),
            totalDurationSec: Number(geminiBatchingItems.reduce((sum: number, item: any) => sum + this.toFiniteNumber(item?.totalDurationSec, 0), 0).toFixed(3)),
            rawItemCount: geminiBatchingItems.reduce((sum: number, item: any) => sum + this.toFiniteNumber(item?.rawItemCount, 0), 0),
            missingItemCount: geminiBatchingItems.reduce((sum: number, item: any) => sum + this.toFiniteNumber(item?.missingItemCount, 0), 0),
            recoveredFromXml: geminiBatchingItems.some((item: any) => Boolean(item?.recoveredFromXml)),
            maxBatchSize: Math.max(...geminiBatchingItems.map((item: any) => this.toFiniteNumber(item?.maxBatchSize, 0))),
          }
        : first?.geminiMultiAudioBatching,
      timing: {
        ...mergedTiming,
        audioDecodeCacheHit: metaList.some((item: any) => Boolean(item?.timing?.audioDecodeCacheHit)),
      },
    };
  }

  private static parseDurationSecondsFromFfmpegLog(stderr: string): number | null {
    const matched = String(stderr || '').match(/Duration:\s*(\d{2}):(\d{2}):(\d{2})(?:[.,](\d+))?/i);
    if (!matched) return null;
    const hh = Number(matched[1]);
    const mm = Number(matched[2]);
    const ss = Number(matched[3]);
    const fractionRaw = matched[4] ? Number(`0.${matched[4]}`) : 0;
    if (![hh, mm, ss, fractionRaw].every(Number.isFinite)) return null;
    const total = hh * 3600 + mm * 60 + ss + fractionRaw;
    return total > 0 ? total : null;
  }

  private static async getAudioDurationSeconds(filePath: string): Promise<number | null> {
    return new Promise((resolve) => {
      const args = ['-i', filePath, '-f', 'null', '-'];
      const cp = spawn(this.getFfmpegBinary(), args);
      let stderr = '';
      cp.stderr.on('data', (chunk: Buffer) => {
        stderr += chunk.toString('utf8');
      });
      cp.on('close', () => {
        resolve(this.parseDurationSecondsFromFfmpegLog(stderr));
      });
      cp.on('error', () => resolve(null));
    });
  }

  private static async assertElevenLabsSpeechToTextLimits(filePath: string) {
    const maxFileBytes = 3 * 1024 * 1024 * 1024;
    const maxDurationSec = 10 * 60 * 60;
    const stat = await fs.stat(filePath).catch(() => null);
    if (stat?.isFile() && stat.size >= maxFileBytes) {
      const sizeGb = stat.size / (1024 * 1024 * 1024);
      throw new Error(
        `ElevenLabs Speech to Text accepts files under 3GB. Current file is ${sizeGb.toFixed(2)}GB; please split or compress it before using ElevenLabs ASR.`
      );
    }

    const durationSec = await this.getAudioDurationSeconds(filePath).catch(() => null);
    if (Number.isFinite(durationSec || Number.NaN) && Number(durationSec) > maxDurationSec) {
      const durationHours = Number(durationSec) / 3600;
      throw new Error(
        `ElevenLabs Speech to Text standard mode accepts audio up to 10 hours. Current audio is ${durationHours.toFixed(2)} hours; please split it before using ElevenLabs ASR.`
      );
    }
  }

  private static async assertDeepgramSpeechToTextLimits(filePath: string) {
    const maxFileBytes = 2 * 1024 * 1024 * 1024;
    const stat = await fs.stat(filePath).catch(() => null);
    if (stat?.isFile() && stat.size >= maxFileBytes) {
      const sizeGb = stat.size / (1024 * 1024 * 1024);
      throw new Error(
        `Deepgram pre-recorded Speech to Text accepts files under 2GB. Current file is ${sizeGb.toFixed(2)}GB; please split or compress it before using Deepgram ASR.`
      );
    }
  }

  private static hasUsableChunkTimestamps(chunks: unknown[]): boolean {
    if (!Array.isArray(chunks) || chunks.length === 0) return false;

    let prevStart: number | null = null;
    let increasingCount = 0;

    for (const chunk of chunks) {
      const start = this.toFiniteNumber((chunk as any)?.start_ts, Number.NaN);
      const end = this.toFiniteNumber((chunk as any)?.end_ts, Number.NaN);

      if (Number.isFinite(start) && Number.isFinite(end) && end > start + 0.01) {
        return true;
      }
      if (Number.isFinite(start)) {
        if (prevStart == null || start > prevStart + 0.01) {
          increasingCount += 1;
          prevStart = start;
        }
      }
    }

    return increasingCount > 1;
  }

  private static async synthesizeChunkTimestamps(
    chunks: Array<{ start_ts: number; end_ts?: number; text: string }>,
    audioPath: string
  ) {
    if (!Array.isArray(chunks) || chunks.length === 0) {
      return { chunks, synthesized: false as const };
    }

    const durationFromAudio = await this.getAudioDurationSeconds(audioPath);
    const fallbackDuration = Math.max(2, chunks.length * 2);
    const totalDuration = Number.isFinite(durationFromAudio || Number.NaN)
      ? Math.max(1, Number(durationFromAudio))
      : fallbackDuration;

    const weights = chunks.map((chunk) => {
      const plain = String(chunk?.text || '').replace(/\s+/g, '');
      return Math.max(1, plain.length);
    });
    const totalWeight = Math.max(1, weights.reduce((sum, value) => sum + value, 0));

    let cumulative = 0;
    const mapped = chunks.map((chunk, index) => {
      const start = (cumulative / totalWeight) * totalDuration;
      cumulative += weights[index];
      let end = (cumulative / totalWeight) * totalDuration;
      if (!Number.isFinite(end) || end <= start) {
        end = start + 0.4;
      }
      return {
        ...chunk,
        start_ts: Number(start.toFixed(3)),
        end_ts: Number(end.toFixed(3)),
      };
    });

    return {
      chunks: mapped,
      synthesized: true as const,
      durationSeconds: totalDuration,
    };
  }

  private static parseCloudAsrStatus(error: unknown): number | null {
    const message = error instanceof Error ? error.message : String(error || '');
    const match = message.match(/Cloud ASR error \((\d{3})\):/i);
    if (!match) return null;
    const status = Number(match[1]);
    return Number.isFinite(status) ? status : null;
  }

  private static isCloudAsrFileLimitError(error: unknown): boolean {
    const message = (error instanceof Error ? error.message : String(error || '')).toLowerCase();
    const status = this.parseCloudAsrStatus(error);

    if (status === 413) return true;
    const fileLimitHint = /(too large|payload too large|entity too large|request body too large|max(?:imum)? (?:file|audio|payload|content)|file size|audio length|duration limit|too long)/i;
    if ((status === 400 || status === 422) && fileLimitHint.test(message)) return true;
    return fileLimitHint.test(message) && /cloud asr error/i.test(message);
  }

  private static isDeepgramChunkableError(error: unknown): boolean {
    const status = this.parseCloudAsrStatus(error);
    if (status === 413 || status === 504) return true;
    return this.isCloudAsrFileLimitError(error);
  }

  private static isGladiaChunkableError(error: unknown): boolean {
    const status = this.parseCloudAsrStatus(error);
    if (status === 408 || status === 413 || status === 504) return true;
    return this.isCloudAsrFileLimitError(error);
  }

  private static isCloudAsrChunkRetryError(error: unknown, chunkingPolicy: AsrCloudChunkingPolicy): boolean {
    if (chunkingPolicy.reason === 'deepgram_upload_or_processing_limit') {
      return this.isDeepgramChunkableError(error);
    }
    if (chunkingPolicy.reason === 'gladia_upload_or_duration_limit') {
      return this.isGladiaChunkableError(error);
    }
    return this.isCloudAsrFileLimitError(error);
  }

  private static async shouldChunkGladiaPreRecorded(filePath: string) {
    const maxSizeBytes = 1000 * 1024 * 1024;
    const proactiveSizeBytes = Math.floor(maxSizeBytes * 0.95);
    const maxDurationSec = 135 * 60;
    const proactiveDurationSec = 130 * 60;
    const stat = await fs.stat(filePath).catch(() => null);
    if (stat?.isFile() && stat.size >= proactiveSizeBytes) {
      return {
        shouldChunk: true,
        reason: `file_size_${(stat.size / (1024 * 1024)).toFixed(1)}mb`,
      };
    }
    const durationSec = await this.getAudioDurationSeconds(filePath).catch(() => null);
    if (Number.isFinite(durationSec || Number.NaN) && Number(durationSec) >= proactiveDurationSec) {
      return {
        shouldChunk: true,
        reason: `duration_${(Number(durationSec) / 60).toFixed(1)}min`,
      };
    }
    return {
      shouldChunk: false,
      reason: `limits_${Math.round(maxSizeBytes / (1024 * 1024))}mb_${Math.round(maxDurationSec / 60)}min`,
    };
  }

  private static isEmptyAudioWindowError(error: unknown) {
    const message = error instanceof Error ? error.message : String(error || '');
    return /Audio window is empty/i.test(message);
  }

  private static isCloudAsrEmptyTranscriptError(error: unknown) {
    const message = error instanceof Error ? error.message : String(error || '');
    return /Cloud ASR response does not contain transcript text/i.test(message);
  }

  private static applyChunkWindowFallbackBounds(transcript: AsrStructuredTranscript, fallbackEndSec: number) {
    const applyEnd = <T extends { start_ts: number; end_ts?: number }>(item: T): T => {
      const start = this.toFiniteNumber(item.start_ts, 0);
      const end = this.toFiniteNumber(item.end_ts, Number.NaN);
      if (Number.isFinite(end) && end > start) return item;
      return {
        ...item,
        end_ts: Number(Math.max(start + 0.1, fallbackEndSec).toFixed(3)),
      };
    };

    return {
      ...transcript,
      chunks: (transcript.chunks || []).map((chunk) => applyEnd(chunk)),
      segments: (transcript.segments || []).map((segment) => applyEnd(segment)),
    };
  }

  private static async transcribeCloudChunked(
    filePath: string,
    config: any,
    options: {
      language?: string;
      prompt?: string;
      segmentation?: boolean;
      wordAlignment?: boolean;
      vad?: boolean;
      diarization?: boolean;
      diarizationOptions?: DiarizationOptions | null;
      audioDurationSec?: number | null;
      decodePolicy?: AsrDecodePolicy;
    },
    onProgress: (msg: string) => void,
    signal?: AbortSignal,
    chunkingPolicy: AsrCloudChunkingPolicy = {}
  ) {
    const initialChunkSec = Math.round(
      chunkingPolicy.initialChunkSec ?? this.getEnvNumber('ASR_FILE_LIMIT_CHUNK_SEC', 480, 30, 3600)
    );
    const minChunkSec = Math.round(
      chunkingPolicy.minChunkSec ?? this.getEnvNumber('ASR_FILE_LIMIT_MIN_CHUNK_SEC', 60, 10, 1800)
    );
    const maxChunks = Math.round(
      chunkingPolicy.maxChunks ?? this.getEnvNumber('ASR_FILE_LIMIT_MAX_CHUNKS', 200, 1, 2000)
    );
    const audioFormat = chunkingPolicy.audioFormat === 'mp3' ? 'mp3' : 'wav';
    const audioBitrate = this.sanitizeAudioBitrate(chunkingPolicy.audioBitrate || '32k');
    const sourceDurationSec = await this.getAudioDurationSeconds(filePath).catch(() => null);
    const hasKnownDuration = Number.isFinite(sourceDurationSec || Number.NaN) && Number(sourceDurationSec) > 0;
    let chunkSec = Math.max(minChunkSec, initialChunkSec);

    while (true) {
      this.throwIfAborted(signal);
      const mergedTranscript: AsrStructuredTranscript = {
        text: '',
        chunks: [],
        segments: [],
        word_segments: [],
      };
      const providerMetaList: any[] = [];
      let producedChunkCount = 0;
      let shouldRetryWithSmallerChunks = false;

      for (let i = 0; i < maxChunks; i += 1) {
        this.throwIfAborted(signal);
        const startSec = i * chunkSec;
        if (hasKnownDuration && startSec >= Number(sourceDurationSec) - 0.05) {
          break;
        }
        const endSec = hasKnownDuration
          ? Math.min(startSec + chunkSec, Number(sourceDurationSec))
          : startSec + chunkSec;
        const label = `file_limit_${chunkSec}s_${i + 1}`;
        let chunkPath: string | null = null;

        try {
          chunkPath = await this.extractAudioWindow(filePath, startSec, endSec, label, {
            audioFormat,
            audioBitrate,
          });
        } catch (extractErr) {
          if (this.isEmptyAudioWindowError(extractErr)) {
            break;
          }
          throw extractErr;
        }

        try {
          onProgress(`Calling ASR provider (file-limit chunk ${i + 1})...`);
          const cloudResult = await this.transcribeCloud(
            chunkPath,
            config,
            { ...options, audioDurationSec: Math.max(0, endSec - startSec) },
            signal
          );
          providerMetaList.push(cloudResult.meta);
          producedChunkCount += 1;
          const shifted = this.offsetStructuredTranscript(cloudResult, startSec);
          const bounded = chunkingPolicy.reason === 'github_phi4_8000_token_limit'
            ? this.applyChunkWindowFallbackBounds(shifted, endSec)
            : shifted;
          mergedTranscript.chunks.push(...bounded.chunks);
          mergedTranscript.segments.push(...bounded.segments);
          mergedTranscript.word_segments.push(...bounded.word_segments);
        } catch (chunkErr) {
          if (this.isCloudAsrChunkRetryError(chunkErr, chunkingPolicy) && chunkSec > minChunkSec) {
            shouldRetryWithSmallerChunks = true;
            break;
          }
          if (
            chunkingPolicy.reason === 'github_phi4_8000_token_limit' &&
            this.isCloudAsrEmptyTranscriptError(chunkErr)
          ) {
            continue;
          }
          throw chunkErr;
        } finally {
          if (chunkPath) {
            await fs.remove(chunkPath).catch(() => {});
          }
        }
      }

      if (shouldRetryWithSmallerChunks) {
        const nextChunkSec = Math.max(minChunkSec, Math.floor(chunkSec / 2));
        if (nextChunkSec >= chunkSec) {
          throw new Error(`Cloud ASR still rejects chunked upload at minimum chunk size (${chunkSec}s).`);
        }
        chunkSec = nextChunkSec;
        onProgress(`Chunk upload still exceeded provider limits or timed out, reducing chunk size to ${chunkSec}s and retrying...`);
        continue;
      }

      if (producedChunkCount === 0 || mergedTranscript.chunks.length === 0) {
        throw new Error('Chunk fallback completed but no transcript text was returned.');
      }

      onProgress(`ASR provider completed (${producedChunkCount} file-limit chunks).`);
      return {
        ...mergedTranscript,
        text: mergedTranscript.chunks.map((chunk) => chunk.text).filter(Boolean).join(' ').trim(),
        chunks: mergedTranscript.chunks.sort((a, b) => this.toFiniteNumber(a.start_ts, 0) - this.toFiniteNumber(b.start_ts, 0)),
        segments: mergedTranscript.segments.sort((a, b) => this.toFiniteNumber(a.start_ts, 0) - this.toFiniteNumber(b.start_ts, 0)),
        word_segments: mergedTranscript.word_segments.sort((a, b) => this.toFiniteNumber(a.start_ts, 0) - this.toFiniteNumber(b.start_ts, 0)),
        meta: {
          ...this.mergeProviderMeta(providerMetaList),
          fileLimitFallbackUsed: true,
          fileLimitChunkCount: producedChunkCount,
          fileLimitChunkSec: chunkSec,
          fileLimitAudioFormat: audioFormat,
          fileLimitReason: chunkingPolicy.reason || null,
        },
      };
    }
  }

  private static async transcribeCloudWithFileLimitFallback(
    filePath: string,
    config: any,
    options: {
      language?: string;
      prompt?: string;
      segmentation?: boolean;
      wordAlignment?: boolean;
      vad?: boolean;
      diarization?: boolean;
      diarizationOptions?: DiarizationOptions | null;
      audioDurationSec?: number | null;
      decodePolicy?: AsrDecodePolicy;
    },
    onProgress: (msg: string) => void,
    signal?: AbortSignal
  ) {
    this.throwIfAborted(signal);
    const resolvedProvider = resolveCloudAsrProvider({
      url: String(config?.url || ''),
      modelName: String(config?.name || ''),
      model: String(config?.model || ''),
    });
    const phi4ChunkingPolicy =
      resolvedProvider.provider === 'github-models-phi4-multimodal'
        ? this.getPhi4CloudChunkingPolicy()
        : null;
    const deepgramChunkingPolicy =
      resolvedProvider.provider === 'deepgram-listen'
        ? this.getDeepgramCloudChunkingPolicy()
        : null;
    const gladiaChunkingPolicy =
      resolvedProvider.provider === 'gladia-pre-recorded'
        ? this.getGladiaCloudChunkingPolicy()
        : null;
    const isElevenLabsScribe = resolvedProvider.provider === 'elevenlabs-scribe';
    const isDeepgramListen = resolvedProvider.provider === 'deepgram-listen';
    const isGladiaPreRecorded = resolvedProvider.provider === 'gladia-pre-recorded';

    if (isElevenLabsScribe) {
      await this.assertElevenLabsSpeechToTextLimits(filePath);
    }
    if (isDeepgramListen) {
      await this.assertDeepgramSpeechToTextLimits(filePath);
    }

    if (phi4ChunkingPolicy) {
      const proactiveChunkSec = Math.max(
        phi4ChunkingPolicy.minChunkSec || 10,
        phi4ChunkingPolicy.initialChunkSec || 30
      );
      onProgress(
        `Phi-4 ASR uses ${proactiveChunkSec}s ${String(phi4ChunkingPolicy.audioFormat || 'mp3').toUpperCase()} chunks to stay under the provider token limit...`
      );
      return this.transcribeCloudChunked(filePath, config, options, onProgress, signal, phi4ChunkingPolicy);
    }

    if (isGladiaPreRecorded && gladiaChunkingPolicy) {
      const preflight = await this.shouldChunkGladiaPreRecorded(filePath);
      if (preflight.shouldChunk) {
        const proactiveChunkSec = Math.max(
          gladiaChunkingPolicy.minChunkSec || 600,
          gladiaChunkingPolicy.initialChunkSec || 3600
        );
        onProgress(
          `Gladia ASR input is near provider limits (${preflight.reason}); using ${Math.round(proactiveChunkSec / 60)} minute MP3 chunks...`
        );
        return this.transcribeCloudChunked(filePath, config, options, onProgress, signal, gladiaChunkingPolicy);
      }
    }

    if (!isElevenLabsScribe && !isDeepgramListen && !isGladiaPreRecorded) {
      try {
        const stat = await fs.stat(filePath);
        const preemptiveThresholdBytes = this.getCloudAsrPreemptiveChunkThresholdBytes();
        if (stat.isFile() && stat.size > preemptiveThresholdBytes) {
          const sizeMb = (stat.size / (1024 * 1024)).toFixed(1);
          const thresholdMb = (preemptiveThresholdBytes / (1024 * 1024)).toFixed(0);
          onProgress(`Audio file is ${sizeMb}MB (> ${thresholdMb}MB), using chunked upload proactively...`);
          return this.transcribeCloudChunked(filePath, config, options, onProgress, signal, phi4ChunkingPolicy || {});
        }
      } catch {
        // Non-fatal. If file stat fails, keep existing request-first fallback behavior.
      }
    }

    try {
      return await this.transcribeCloud(filePath, config, options, signal);
    } catch (err) {
      const shouldChunk =
        gladiaChunkingPolicy
          ? this.isGladiaChunkableError(err)
          : deepgramChunkingPolicy
          ? this.isDeepgramChunkableError(err)
          : this.isCloudAsrFileLimitError(err);
      if (!shouldChunk) {
        throw err;
      }
      onProgress('Provider rejected audio size/duration limit, retrying with chunked upload...');
      return this.transcribeCloudChunked(filePath, config, options, onProgress, signal, phi4ChunkingPolicy || deepgramChunkingPolicy || gladiaChunkingPolicy || {});
    }
  }

  private static async transcribeWithProvider(
    filePath: string,
    input: {
      useLocalProvider: boolean;
      localModelId?: string;
      localModelRuntime?: string;
      localModelPath?: string;
      cloudModelConfig?: any;
      options: {
        language?: string;
        prompt?: string;
        segmentation?: boolean;
        wordAlignment?: boolean;
        vad?: boolean;
        diarization?: boolean;
        diarizationOptions?: DiarizationOptions | null;
        audioDurationSec?: number | null;
        decodePolicy?: AsrDecodePolicy;
      };
      onProgress: (msg: string) => void;
      signal?: AbortSignal;
    }
  ) {
    this.throwIfAborted(input.signal);
    if (input.useLocalProvider) {
      if (!input.localModelId || !input.localModelPath) {
        throw new Error('Local ASR runtime is not configured.');
      }
      const preferAlignmentFirstNoSpace = Boolean(input.options.wordAlignment) && !this.isQwen3AsrLocalModel({
        localModelId: input.localModelId,
        localModelRuntime: input.localModelRuntime,
        localModelPath: input.localModelPath,
      });
      return transcribeWithLocalAsrRuntime({
        filePath,
        localModelId: input.localModelId,
        localModelPath: input.localModelPath,
        localModelRuntime: input.localModelRuntime,
        language: input.options.language,
        prompt: input.options.prompt,
        segmentation: input.options.segmentation,
        wordAlignment: input.options.wordAlignment,
        decodePolicy: input.options.decodePolicy,
        extractStructuredTranscript: (transcript, transcriptLanguage, extractOptions = {}) =>
          this.extractStructuredTranscript(transcript, transcriptLanguage, {
            ...extractOptions,
            preferAlignmentFirstNoSpace,
          }),
        toFiniteNumber: this.toFiniteNumber.bind(this),
      });
    }

    return this.transcribeCloudWithFileLimitFallback(
      filePath,
      input.cloudModelConfig,
      input.options,
      input.onProgress,
      input.signal
    );
  }

  static async transcribe(
    projectId: string,
    options: {
      modelId?: string,
      assetName?: string,
      language?: string,
      prompt?: string,
      segmentation?: boolean,
      wordAlignment?: boolean,
      vad?: boolean,
      diarization?: boolean,
      pipelineMode?: AsrPipelineMode,
      diarizationOptions?: DiarizationOptions
    },
    onProgress: (msg: string) => void,
    signal?: AbortSignal
  ) {
    this.throwIfAborted(signal);
    const {
      modelId,
      assetName,
      language,
      prompt,
      segmentation,
      wordAlignment,
      vad,
      diarization,
      pipelineMode: requestedPipelineMode,
      diarizationOptions,
    } = options;
    const startedAt = Date.now();
    const effectiveSegmentation = Boolean(segmentation) || Boolean(diarization);
    const requestedWordAlignment = wordAlignment !== false;
    const effectiveWordAlignment = effectiveSegmentation && requestedWordAlignment;
    const pipelineMode = this.resolvePipelineMode(requestedPipelineMode);
    const decodePolicy = this.buildDecodePolicy({
      mode: pipelineMode,
      effectiveWordAlignment,
    });
    const segmentationForcedForDiarization = Boolean(diarization) && !Boolean(segmentation);
    const requestedFeatures = {
      segmentation: Boolean(segmentation),
      wordAlignment: requestedWordAlignment,
      vad: Boolean(vad),
      diarization: Boolean(diarization),
      pipelineMode,
      diarizationOptions: diarizationOptions || null,
    };

    const audioPath = assetName
      ? PathManager.resolveProjectFile(projectId, 'assets', assetName, { createProject: false })
      : PathManager.resolveProjectFile(projectId, 'assets', 'audio.wav', { createProject: false });

    if (!(await fs.pathExists(audioPath))) {
      throw new Error(`Audio file not found: ${path.basename(audioPath)}`);
    }

    const settings = await SettingsManager.getSettings({ mask: false });
    const localModel = await LocalModelService.resolveLocalModelForRequest('asr', modelId, settings);
    const localProfile = localModel ? resolveLocalAsrProfile(localModel) : null;
    const useLocalProvider = Boolean(localModel);
    const localModelPath = localModel ? getLocalModelInstallDir(localModel) : undefined;
    const allowVadWindowedTranscription = this.getEnvBoolean('ASR_VAD_WINDOWING_ENABLED', true);
    const schedulerEnabled = this.getEnvBoolean('ASR_VAD_SCHEDULER_ENABLED', true);
    const vadWindowBatchSize = this.getVadWindowBatchSize(pipelineMode);
    const vadWindowConcurrency = this.getVadWindowConcurrency(pipelineMode, useLocalProvider);
    const vadWindowMaxRetries = Math.round(this.getEnvNumber('ASR_VAD_WINDOW_MAX_RETRIES', 1, 0, 5));
    const effectivePrompt = useLocalProvider
      ? this.buildEffectiveLocalAsrPrompt(language, prompt)
      : String(prompt || '').trim();
    const localAsrPromptApplied = useLocalProvider && Boolean(this.getBuiltInLocalAsrPrompt(language));

    const modelConfig = useLocalProvider
      ? null
      : settings.asrModels.find((m: any) => m.id === modelId) || settings.asrModels[0];
    if (!useLocalProvider) {
      if (!modelConfig) {
        throw new Error('No ASR model configured. Please configure one in Settings.');
      }
      if (!modelConfig.url) {
        throw new Error('ASR model URL is missing.');
      }
    }
    const resolvedCloudProvider = !useLocalProvider && modelConfig
      ? resolveCloudAsrProvider({
          url: String(modelConfig?.url || ''),
          modelName: String(modelConfig?.name || ''),
          model: String(modelConfig?.model || ''),
        })
      : null;
    const isGeminiCloudAsr = resolvedCloudProvider?.provider === 'google-gemini-audio';
    const isElevenLabsCloudAsr = resolvedCloudProvider?.provider === 'elevenlabs-scribe';
    const isDeepgramCloudAsr = resolvedCloudProvider?.provider === 'deepgram-listen';
    const isGladiaCloudAsr = resolvedCloudProvider?.provider === 'gladia-pre-recorded';
    const usesCloudNativeAdvancedAsr = isElevenLabsCloudAsr || isDeepgramCloudAsr || isGladiaCloudAsr;
    const effectiveVad = usesCloudNativeAdvancedAsr
      ? false
      : Boolean(vad) || Boolean(diarization);
    const geminiVadTimestampingRequested = Boolean(isGeminiCloudAsr && effectiveSegmentation);
    const shouldRunVadPipeline = effectiveVad || geminiVadTimestampingRequested;
    const shouldUseVadWindowedTranscription = allowVadWindowedTranscription || geminiVadTimestampingRequested;
    if (useLocalProvider) {
      onProgress(`Loading local ASR model (${localModel!.displayName})...`);
    }

    let speechSegments: Array<{ start: number; end: number }> = [];
    let vadWindows: Array<{ start: number; end: number }> = [];
    let windowedTranscript: AsrStructuredTranscript | null = null;
    let providerMetaFromWindows: any = null;
    let vadError: string | null = null;
    let diarizationError: string | null = null;
    let diarizationApplied = false;
    let alignmentError: string | null = null;
    let alignmentDiagnostics: any = null;
    let deferredSyntheticTimecodes = false;
    let vadMs = 0;
    let providerWallMs = 0;
    let alignmentWallMs = 0;
    let diarizationMs = 0;
    const windowScheduleStats: AsrWindowScheduleStats = {
      enabled: shouldRunVadPipeline,
      schedulerEnabled,
      mode: pipelineMode,
      batchSize: vadWindowBatchSize,
      concurrency: vadWindowConcurrency,
      maxRetries: vadWindowMaxRetries,
      taskCount: 0,
      attemptedCount: 0,
      succeededCount: 0,
      skippedCount: 0,
      retries: 0,
    };

    if (shouldRunVadPipeline) {
      const vadStartedAt = Date.now();
      this.throwIfAborted(signal);
      onProgress(
        geminiVadTimestampingRequested && !effectiveVad
          ? 'Running voice activity detection for Gemini timestamp windows...'
          : 'Running voice activity detection...'
      );
      try {
        speechSegments = await VadService.detectSpeech(audioPath);

        if (speechSegments.length > 0) {
          vadWindows = this.buildVadWindows(speechSegments, {
            overflowMode: geminiVadTimestampingRequested ? 'rebalance' : 'collapse',
          });
          const windowTasks = this.buildVadWindowTasks(vadWindows);
          windowScheduleStats.taskCount = windowTasks.length;
          onProgress(`VAD detected ${speechSegments.length} speech segments (${vadWindows.length} windows).`);

          if (!shouldUseVadWindowedTranscription) {
            onProgress('VAD windows are used only for diagnostics/diarization; ASR keeps full-audio transcription.');
          }

          const mergedWindowTranscript: AsrStructuredTranscript = {
            text: '',
            chunks: [],
            segments: [],
            word_segments: [],
          };
          const providerMetaList: any[] = [];
          const skippedWindowErrors: string[] = [];

          if (shouldUseVadWindowedTranscription) {
            if (geminiVadTimestampingRequested && !useLocalProvider && modelConfig) {
              const geminiBatchSize = this.getGeminiMultiAudioBatchSize();
              const batches: AsrVadWindowTask[][] = [];
              for (let index = 0; index < windowTasks.length; index += geminiBatchSize) {
                batches.push(windowTasks.slice(index, index + geminiBatchSize));
              }
              windowScheduleStats.batchSize = geminiBatchSize;
              windowScheduleStats.concurrency = 1;
              windowScheduleStats.geminiMultiAudioBatchSize = geminiBatchSize;
              windowScheduleStats.providerRequestCount = 0;

              for (let batchIndex = 0; batchIndex < batches.length; batchIndex += 1) {
                this.throwIfAborted(signal);
                const batch = batches[batchIndex];
                onProgress(`Calling Gemini ASR provider (batch ${batchIndex + 1}/${batches.length}, ${batch.length} audio windows)...`);

                let attempts = 0;
                while (true) {
                  attempts += 1;
                  windowScheduleStats.attemptedCount += batch.length;
                  const providerRequestCountBefore: number = windowScheduleStats.providerRequestCount || 0;
                  windowScheduleStats.providerRequestCount = providerRequestCountBefore + 1;
                  const extractedInputs: Array<{ task: AsrVadWindowTask; audioPath: string }> = [];

                  try {
                    for (const task of batch) {
                      const windowLabel = `${task.index + 1}_${Math.round(task.start * 1000)}_${Math.round(task.end * 1000)}`;
                      extractedInputs.push({
                        task,
                        audioPath: await this.extractAudioWindow(audioPath, task.start, task.end, windowLabel),
                      });
                    }

                    const providerStartedAt = Date.now();
                    const batchResult = await this.transcribeGeminiCloudWindowBatchAdaptive(
                      extractedInputs,
                      modelConfig,
                      {
                        language,
                        prompt: effectivePrompt,
                        segmentation: effectiveSegmentation,
                        wordAlignment: effectiveWordAlignment,
                        vad: effectiveVad,
                        diarization,
                        diarizationOptions,
                        decodePolicy,
                      },
                      onProgress,
                      signal
                    );
                    providerWallMs += Math.max(0, Date.now() - providerStartedAt);
                    providerMetaList.push(batchResult.meta);
                    const actualProviderRequests = this.toFiniteNumber(
                      batchResult.meta?.geminiMultiAudioBatching?.requestCount,
                      1
                    );
                    windowScheduleStats.providerRequestCount =
                      providerRequestCountBefore + Math.max(1, actualProviderRequests);

                    const resultByInputIndex = new Map(
                      batchResult.results.map((item) => [item.input.index, item.result] as const)
                    );
                    for (const task of batch) {
                      const providerResult = resultByInputIndex.get(task.index);
                      if (providerResult && providerResult.chunks.length > 0) {
                        const shifted = this.offsetStructuredTranscript(providerResult, task.start);
                        windowScheduleStats.succeededCount += 1;
                        mergedWindowTranscript.chunks.push(...shifted.chunks);
                        mergedWindowTranscript.segments.push(...shifted.segments);
                        mergedWindowTranscript.word_segments.push(...shifted.word_segments);
                      } else {
                        windowScheduleStats.skippedCount += 1;
                        skippedWindowErrors.push(`window_${task.index + 1}: Gemini batch returned empty transcript item`);
                      }
                    }
                    break;
                  } catch (windowErr) {
                    if (this.isAbortError(windowErr)) {
                      throw windowErr;
                    }
                    const message = windowErr instanceof Error ? windowErr.message : String(windowErr);
                    if (attempts <= vadWindowMaxRetries) {
                      windowScheduleStats.retries += 1;
                      onProgress(
                        `Gemini ASR batch ${batchIndex + 1}/${batches.length} failed (${message}), retrying (${attempts}/${vadWindowMaxRetries})...`
                      );
                      continue;
                    }
                    windowScheduleStats.skippedCount += batch.length;
                    for (const task of batch) {
                      skippedWindowErrors.push(`window_${task.index + 1}: ${message}`);
                    }
                    console.warn(`[ASR] Gemini ASR batch ${batchIndex + 1}/${batches.length} skipped: ${message}`);
                    break;
                  } finally {
                    await Promise.all(extractedInputs.map((input) => fs.remove(input.audioPath).catch(() => {})));
                  }
                }
              }
            } else {
              const batches = schedulerEnabled
                ? this.chunkTasksByBatchSize(windowTasks, vadWindowBatchSize)
                : windowTasks.map((task) => [task]);

              for (let batchIndex = 0; batchIndex < batches.length; batchIndex += 1) {
                this.throwIfAborted(signal);
                const batch = batches[batchIndex];
                const label = schedulerEnabled
                  ? `batch ${batchIndex + 1}/${batches.length}`
                  : `window group ${batchIndex + 1}/${batches.length}`;
                onProgress(`Calling ASR provider (${label}, ${batch.length} windows)...`);

                const results = await this.runTaskPool(
                  batch,
                  schedulerEnabled ? vadWindowConcurrency : 1,
                  async (task) => {
                    this.throwIfAborted(signal);
                    let attempts = 0;
                    while (true) {
                      attempts += 1;
                      windowScheduleStats.attemptedCount += 1;
                      windowScheduleStats.providerRequestCount = (windowScheduleStats.providerRequestCount || 0) + 1;
                      const windowLabel = `${task.index + 1}_${Math.round(task.start * 1000)}_${Math.round(task.end * 1000)}`;
                      const windowAudioPath = await this.extractAudioWindow(audioPath, task.start, task.end, windowLabel);
                      try {
                        const providerStartedAt = Date.now();
                        const providerResult = await this.transcribeWithProvider(windowAudioPath, {
                          useLocalProvider,
                          localModelId: localModel?.id,
                          localModelRuntime: localModel?.runtime,
                          localModelPath,
                          cloudModelConfig: modelConfig,
                          options: {
                            language,
                            prompt: effectivePrompt,
                            segmentation: effectiveSegmentation,
                            wordAlignment: effectiveWordAlignment,
                            vad: effectiveVad,
                            diarization,
                            diarizationOptions,
                            audioDurationSec: task.durationSec,
                            decodePolicy,
                          },
                          onProgress,
                          signal,
                        });
                        providerWallMs += Math.max(0, Date.now() - providerStartedAt);
                        const resolvedProviderResult =
                          geminiVadTimestampingRequested
                            ? {
                                ...this.collapseTranscriptToVadWindow(providerResult, task.durationSec, language),
                                meta: providerResult.meta,
                              }
                            : providerResult;
                        windowScheduleStats.succeededCount += 1;
                        return {
                          task,
                          ok: true as const,
                          providerResult: resolvedProviderResult,
                        };
                      } catch (windowErr) {
                        if (this.isAbortError(windowErr)) {
                          throw windowErr;
                        }
                        const message = windowErr instanceof Error ? windowErr.message : String(windowErr);
                        if (attempts <= vadWindowMaxRetries) {
                          windowScheduleStats.retries += 1;
                          onProgress(
                            `VAD window ${task.index + 1}/${vadWindows.length} failed (${message}), retrying (${attempts}/${vadWindowMaxRetries})...`
                          );
                          continue;
                        }
                        windowScheduleStats.skippedCount += 1;
                        return {
                          task,
                          ok: false as const,
                          message,
                        };
                      } finally {
                        await fs.remove(windowAudioPath).catch(() => {});
                      }
                    }
                  }
                );

                for (const item of results) {
                  if (item.ok) {
                    const shifted = this.offsetStructuredTranscript(item.providerResult, item.task.start);
                    providerMetaList.push(item.providerResult.meta);
                    mergedWindowTranscript.chunks.push(...shifted.chunks);
                    mergedWindowTranscript.segments.push(...shifted.segments);
                    mergedWindowTranscript.word_segments.push(...shifted.word_segments);
                  } else {
                    skippedWindowErrors.push(`window_${item.task.index + 1}: ${item.message}`);
                    console.warn(`[ASR] VAD window ${item.task.index + 1}/${vadWindows.length} skipped: ${item.message}`);
                  }
                }
              }
            }
          }

          if (skippedWindowErrors.length > 0) {
            vadError = skippedWindowErrors.join(' | ');
          }

          if (shouldUseVadWindowedTranscription && mergedWindowTranscript.chunks.length > 0) {
            windowedTranscript = this.reconcileWindowedTranscript({
              text: mergedWindowTranscript.chunks.map((chunk) => chunk.text).filter(Boolean).join(' ').trim(),
              chunks: mergedWindowTranscript.chunks.sort((a, b) => a.start_ts - b.start_ts),
              segments: mergedWindowTranscript.segments.sort((a, b) => a.start_ts - b.start_ts),
              word_segments: mergedWindowTranscript.word_segments.sort((a, b) => a.start_ts - b.start_ts),
            }, language);
            providerMetaFromWindows = this.mergeProviderMeta(providerMetaList);
            const providerRequestCount = windowScheduleStats.providerRequestCount || windowScheduleStats.succeededCount;
            onProgress(
              `ASR provider completed (${windowScheduleStats.succeededCount}/${vadWindows.length} VAD windows, ${providerRequestCount} provider requests, skipped ${windowScheduleStats.skippedCount}).`
            );
          } else {
            onProgress('VAD windows produced no transcript text, falling back to full-audio request.');
          }
        }
      } catch (vadErr) {
        if (this.isAbortError(vadErr)) {
          throw vadErr;
        }
        console.warn('[ASR] VAD pipeline failed, continuing without VAD windows:', vadErr);
        vadError = vadErr instanceof Error ? vadErr.message : String(vadErr);
      } finally {
        vadMs = Math.max(0, Date.now() - vadStartedAt);
      }
    }

    let result: AsrStructuredTranscript;
    let providerMeta: any = providerMetaFromWindows;
    if (windowedTranscript && windowedTranscript.chunks.length > 0) {
      result = windowedTranscript;
    } else {
      this.throwIfAborted(signal);
      onProgress(`Calling ASR provider (segmentation: ${effectiveSegmentation ? 'on' : 'off'})...`);
      const providerStartedAt = Date.now();
      const cloudResult = await this.transcribeWithProvider(audioPath, {
        useLocalProvider,
        localModelId: localModel?.id,
        localModelRuntime: localModel?.runtime,
        localModelPath,
        cloudModelConfig: modelConfig,
        options: {
          language,
          prompt: effectivePrompt,
          segmentation: effectiveSegmentation,
          wordAlignment: effectiveWordAlignment,
          vad: effectiveVad,
          diarization,
          diarizationOptions,
          decodePolicy,
        },
        onProgress,
        signal,
      });
      providerWallMs += Math.max(0, Date.now() - providerStartedAt);
      providerMeta = cloudResult.meta;
      result = {
        text: cloudResult.text,
        chunks: cloudResult.chunks,
        segments: cloudResult.segments,
        word_segments: cloudResult.word_segments,
      };
    }
    providerMeta = {
      ...(providerMeta || {}),
      pipelineMode,
      decodePolicy,
      windowScheduler: {
        ...windowScheduleStats,
      },
    };
    if (usesCloudNativeAdvancedAsr) {
      const nativeProviderLabel = isGladiaCloudAsr ? 'gladia' : isDeepgramCloudAsr ? 'deepgram' : 'elevenlabs';
      providerMeta = {
        ...providerMeta,
        cloudNativeAdvancedFeatures: {
          enabled: true,
          provider: nativeProviderLabel,
          vadRequested: Boolean(vad),
          localVadBypassed: Boolean(vad) || Boolean(diarization),
          wordAlignmentRequested: effectiveWordAlignment,
          localForcedAlignmentBypassed: effectiveWordAlignment,
          diarizationRequested: Boolean(diarization),
          localDiarizationBypassed: Boolean(diarization),
        },
        ...(isElevenLabsCloudAsr
          ? {
              elevenLabsNativeAdvancedFeatures: {
                enabled: true,
                vadRequested: Boolean(vad),
                localVadBypassed: Boolean(vad) || Boolean(diarization),
                wordAlignmentRequested: effectiveWordAlignment,
                localForcedAlignmentBypassed: effectiveWordAlignment,
                diarizationRequested: Boolean(diarization),
                localDiarizationBypassed: Boolean(diarization),
              },
            }
          : {}),
        ...(isDeepgramCloudAsr
          ? {
              deepgramNativeAdvancedFeatures: {
                enabled: true,
                vadRequested: Boolean(vad),
                localVadBypassed: Boolean(vad) || Boolean(diarization),
                wordAlignmentRequested: effectiveWordAlignment,
                localForcedAlignmentBypassed: effectiveWordAlignment,
                diarizationRequested: Boolean(diarization),
                localDiarizationBypassed: Boolean(diarization),
              },
            }
          : {}),
        ...(isGladiaCloudAsr
          ? {
              gladiaNativeAdvancedFeatures: {
                enabled: true,
                vadRequested: Boolean(vad),
                localVadBypassed: Boolean(vad) || Boolean(diarization),
                wordAlignmentRequested: effectiveWordAlignment,
                localForcedAlignmentBypassed: effectiveWordAlignment,
                diarizationRequested: Boolean(diarization),
                localDiarizationBypassed: Boolean(diarization),
              },
            }
          : {}),
      };
    }
    const geminiVadTimestampingApplied = Boolean(
      isGeminiCloudAsr &&
      geminiVadTimestampingRequested &&
      windowedTranscript &&
      windowedTranscript.chunks.length > 0
    );
    if (isGeminiCloudAsr) {
      providerMeta = {
        ...providerMeta,
        geminiTimestampSource: geminiVadTimestampingApplied ? 'vad_window' : 'provider',
        geminiVadTimestamping: {
          applied: geminiVadTimestampingApplied,
          requestedBecause: geminiVadTimestampingRequested ? 'segmentation' : null,
          forcedByProvider: geminiVadTimestampingRequested && !effectiveVad,
          requestedVad: Boolean(vad),
          detectedSpeechSegments: speechSegments.length,
          vadWindowCount: vadWindows.length,
          windowOverflowMode: geminiVadTimestampingRequested ? 'rebalance' : null,
          multiAudioBatching: providerMeta.geminiMultiAudioBatching || null,
          windowedTranscription: Boolean(geminiVadTimestampingApplied),
          fallbackReason:
            geminiVadTimestampingRequested && !geminiVadTimestampingApplied
              ? vadError
                ? 'vad_failed'
                : speechSegments.length === 0
                  ? 'no_speech_segments'
                  : 'windowed_transcript_unavailable'
              : null,
        },
      };
    }

    let timestampsSynthesized = false;
    const allowSyntheticSegmentationTimestamps =
      effectiveSegmentation &&
      (!effectiveWordAlignment || !this.isLanguageWithoutSpaces(language));
    const shouldPreferAlignmentFirst =
      decodePolicy.alignmentStrategy === 'alignment-first' && effectiveWordAlignment;
    if (
      allowSyntheticSegmentationTimestamps &&
      Array.isArray(result?.chunks) &&
      result.chunks.length > 0 &&
      !this.hasUsableChunkTimestamps(result.chunks)
    ) {
      if (shouldPreferAlignmentFirst) {
        deferredSyntheticTimecodes = true;
      } else {
        const synthesized = await this.synthesizeChunkTimestamps(result.chunks, audioPath);
        if (synthesized.synthesized) {
          result.chunks = synthesized.chunks;
          timestampsSynthesized = true;
          onProgress('Provider returned no timestamps, synthesized approximate timecodes for segmented transcript.');
        }
      }
    }

    const alignmentSampleText =
      Array.isArray(result.segments) && result.segments.length > 0
        ? result.segments.map((segment) => segment?.text || '').join(' ')
        : result.text || '';
    const resolvedAlignmentLanguage = effectiveWordAlignment
      ? AlignmentService.resolveForcedAlignmentLanguage(language, alignmentSampleText)
      : null;
    const hasProviderNativeAlignment = Boolean(providerMeta?.forcedAlignment);
    const nativeWordAlignmentProvider = providerMeta?.isElevenLabsScribe
      ? 'elevenlabs'
      : providerMeta?.isDeepgramListen
        ? 'deepgram'
        : providerMeta?.isGladiaPreRecorded
          ? 'gladia'
        : '';
    const nativeWordAlignmentDisplay =
      nativeWordAlignmentProvider === 'gladia'
        ? 'Gladia'
        : nativeWordAlignmentProvider === 'deepgram'
          ? 'Deepgram'
          : 'ElevenLabs';
    const nativeWordAlignmentProfile =
      nativeWordAlignmentProvider === 'gladia'
        ? 'gladia-pre-recorded-native'
        : nativeWordAlignmentProvider === 'deepgram'
          ? 'deepgram-listen-native'
          : 'elevenlabs-scribe-native';
    const useProviderNativeWordAlignment = Boolean(
      effectiveWordAlignment &&
      nativeWordAlignmentProvider &&
      providerMeta?.nativeWordTimestamps &&
      Array.isArray(result.word_segments) &&
      result.word_segments.length > 0
    );
    const shouldBypassLocalWordAlignmentForNativeProvider = Boolean(
      effectiveWordAlignment &&
      nativeWordAlignmentProvider
    );
    if (useProviderNativeWordAlignment) {
      alignmentDiagnostics = {
        applied: true,
        profileId: nativeWordAlignmentProfile,
        backend: 'native',
        modelId: String(providerMeta?.effectiveModel || nativeWordAlignmentProvider),
        language: String(providerMeta?.detectedLanguage || language || '').trim() || null,
        attemptedSegmentCount: Array.isArray(result.segments) ? result.segments.length : 0,
        alignedSegmentCount: Array.isArray(result.segments) ? result.segments.length : 0,
        skippedSegments: 0,
        failureCount: 0,
        alignedWordCount: result.word_segments.length,
        avgConfidence: null,
        elapsedMs: 0,
        providerNativeProvider: nativeWordAlignmentProvider,
      };
      onProgress(`Using ${nativeWordAlignmentDisplay} provider-native word timestamps.`);
    } else if (shouldBypassLocalWordAlignmentForNativeProvider) {
      alignmentDiagnostics = {
        applied: false,
        profileId: nativeWordAlignmentProfile,
        backend: 'native',
        modelId: String(providerMeta?.effectiveModel || nativeWordAlignmentProvider),
        language: String(providerMeta?.detectedLanguage || language || '').trim() || null,
        attemptedSegmentCount: Array.isArray(result.segments) ? result.segments.length : 0,
        alignedSegmentCount: 0,
        skippedSegments: Array.isArray(result.segments) ? result.segments.length : 0,
        failureCount: 0,
        alignedWordCount: 0,
        avgConfidence: null,
        elapsedMs: 0,
        providerNativeProvider: nativeWordAlignmentProvider,
        reason: providerMeta?.nativeWordTimestamps ? 'empty_provider_word_segments' : 'provider_word_timestamps_unavailable',
        localForcedAlignmentBypassed: true,
      };
      onProgress(`${nativeWordAlignmentDisplay} did not return usable provider-native word timestamps; local forced alignment is skipped for this provider.`);
    } else if (effectiveWordAlignment && !resolvedAlignmentLanguage && !hasProviderNativeAlignment) {
      alignmentDiagnostics = {
        applied: false,
        profileId: 'unresolved',
        backend: 'ctc',
        modelId: 'unresolved',
        language: String(language || '').trim() || null,
        attemptedSegmentCount: 0,
        alignedSegmentCount: 0,
        skippedSegments: 0,
        failureCount: 0,
        alignedWordCount: 0,
        avgConfidence: null,
        elapsedMs: 0,
        reason: 'unresolved_language_or_profile',
      };
      onProgress('Forced alignment unavailable for current language/profile, keeping provider timestamps.');
    }
    else if (
      effectiveWordAlignment &&
      (resolvedAlignmentLanguage || hasProviderNativeAlignment) &&
      Array.isArray(result.segments) &&
      result.segments.length > 0
    ) {
      this.throwIfAborted(signal);
      const alignmentStartedAt = Date.now();
      try {
        const aligned = await AlignmentService.alignTranscript(audioPath, result, {
          language: resolvedAlignmentLanguage || language,
          providerMeta,
          localModelId: localModel?.id || null,
          localModelRuntime: localModel?.runtime || null,
          onProgress,
          signal,
        });
        let alignmentAcceptDecision = { accepted: true, reason: null as string | null };
        if (aligned.applied && aligned.word_segments.length > 0) {
          alignmentAcceptDecision = this.evaluateNoSpaceAlignmentQualityGate(aligned.diagnostics, language);
        }
        alignmentDiagnostics =
          aligned.applied && !alignmentAcceptDecision.accepted
            ? {
                ...aligned.diagnostics,
                applied: false,
                downgradedByQualityGate: true,
                qualityGateReason: alignmentAcceptDecision.reason,
              }
            : aligned.diagnostics;
        if (aligned.applied && aligned.word_segments.length > 0 && alignmentAcceptDecision.accepted) {
          result.segments = aligned.segments;
          result.word_segments = aligned.word_segments;
          result.chunks = this.buildDisplayChunksFromSegments(result.segments, language);
          result.text = this.buildTranscriptTextFromChunks(result.chunks, language);
          if (aligned.diagnostics.backend === 'native') {
            onProgress('Provider-native forced alignment applied.');
          } else {
            onProgress(`Forced alignment applied (${aligned.diagnostics.alignedSegmentCount} spans).`);
          }
        } else if (aligned.applied && !alignmentAcceptDecision.accepted) {
          onProgress(
            `Forced alignment quality gate rejected CJK alignment (${alignmentAcceptDecision.reason || 'insufficient_quality'}), keeping provider timestamps.`
          );
        } else if (aligned.diagnostics.attemptedSegmentCount > 0) {
          onProgress('Forced alignment did not improve timing, keeping provider timestamps.');
        }
      } catch (alignErr) {
        if (this.isAbortError(alignErr)) {
          throw alignErr;
        }
        alignmentError = alignErr instanceof Error ? alignErr.message : String(alignErr);
        console.warn('[ASR] Forced alignment failed, continuing with provider timestamps:', alignErr);
      } finally {
        alignmentWallMs = Math.max(0, Date.now() - alignmentStartedAt);
      }
    }

    if (
      deferredSyntheticTimecodes &&
      allowSyntheticSegmentationTimestamps &&
      Array.isArray(result?.chunks) &&
      result.chunks.length > 0 &&
      !this.hasUsableChunkTimestamps(result.chunks)
    ) {
      const synthesized = await this.synthesizeChunkTimestamps(result.chunks, audioPath);
      if (synthesized.synthesized) {
        result.chunks = synthesized.chunks;
        timestampsSynthesized = true;
        onProgress('Forced alignment did not produce usable timestamps, synthesized approximate segmented timecodes.');
      }
    }

    let diarizationDiagnostics: any = null;
    const nativeDiarizationProvider = providerMeta?.isElevenLabsScribe
      ? 'elevenlabs'
      : providerMeta?.isDeepgramListen
        ? 'deepgram'
        : providerMeta?.isGladiaPreRecorded
          ? 'gladia'
        : '';
    const nativeDiarizationDisplay =
      nativeDiarizationProvider === 'gladia'
        ? 'Gladia'
        : nativeDiarizationProvider === 'deepgram'
          ? 'Deepgram'
          : 'ElevenLabs';
    const useProviderNativeDiarization = Boolean(
      diarization &&
      nativeDiarizationProvider
    );
    if (useProviderNativeDiarization) {
      if (providerMeta?.providerNativeDiarization && this.hasProviderSpeakerTags(result)) {
        this.applyProviderNativeSpeakerAssignments(result);
        diarizationApplied = true;
      }
      diarizationDiagnostics = this.buildProviderNativeDiarizationDiagnostics(
        result,
        speechSegments,
        vadWindows,
        nativeDiarizationProvider
      );
      onProgress(
        diarizationApplied
          ? `Using ${nativeDiarizationDisplay} provider-native speaker labels.`
          : `${nativeDiarizationDisplay} did not return speaker labels; local diarization fallback is skipped for this provider.`
      );
    } else if (diarization && Array.isArray(result.chunks) && result.chunks.length > 0) {
      this.throwIfAborted(signal);
      onProgress('Running speaker diarization...');
      const diarizationStartedAt = Date.now();
      try {
        const diarizationResult = await DiarizationService.performDiarization(result.chunks, audioPath, {
          speechSegments,
          vadWindows,
          options: diarizationOptions,
          onProgress,
          signal,
        });
        result.chunks = diarizationResult.chunks;
        diarizationDiagnostics = diarizationResult.diagnostics;
        diarizationApplied = true;
        this.applySpeakerAssignments(result);
      } catch (diaErr) {
        if (this.isAbortError(diaErr)) {
          throw diaErr;
        }
        console.warn('[ASR] Diarization failed, continuing without speaker tags:', diaErr);
        diarizationError = diaErr instanceof Error ? diaErr.message : String(diaErr);
      } finally {
        diarizationMs = Math.max(0, Date.now() - diarizationStartedAt);
      }
    }

    const warnings = buildAsrWarningCodes({
      segmentationForcedForDiarization,
      effectiveSegmentation,
      providerMeta,
      timestampsSynthesized,
      effectiveVad,
      vadWindows,
      windowedTranscript,
      allowVadWindowedTranscription,
      speechSegments,
      alignmentDiagnostics,
    });
    // Contract anchor: warnings.push('segmentation_forced_for_diarization');
    if (warnings.includes('segmentation_forced_for_diarization')) {
      onProgress('Segmentation was enabled automatically for diarization.');
    }
    if (warnings.includes('segmentation_ignored_by_provider')) {
      onProgress('Segmentation requested but provider endpoint may ignore it.');
    }
    if (warnings.includes('auto_language_retried_without_language')) {
      onProgress('Provider rejected language=auto, retried without language.');
    }
    if (warnings.includes('provider_file_limit_chunked')) {
      onProgress('Provider file size/duration fallback was applied (chunked upload).');
    }

    const elapsedMs = Math.max(0, Date.now() - startedAt);
    result.debug = buildAsrDebugInfo({
      requestedFeatures,
      userPrompt: prompt,
      effectivePrompt,
      localAsrPromptApplied,
      pipelineMode,
      decodePolicy,
      providerMeta,
      localProfile,
      effectiveSegmentation,
      effectiveWordAlignment,
      effectiveVad,
      diarizationApplied,
      result,
      speechSegments,
      vadWindows,
      windowScheduleStats,
      diarizationDiagnostics,
      alignmentDiagnostics,
      resolvedAlignmentLanguage,
      elapsedMs,
      vadMs,
      providerWallMs,
      alignmentWallMs,
      diarizationMs,
      alignmentError,
      vadError,
      diarizationError,
      warnings,
      hasUsableChunkTimestamps: this.hasUsableChunkTimestamps.bind(this),
      toFiniteNumber: this.toFiniteNumber.bind(this),
    });
    // Contract anchor: diarizationSource: diarizationDiagnostics?.selectedSource ?? null,

    const transcriptionPath = PathManager.resolveProjectFile(projectId, 'assets', 'transcription.json', { createProject: true });
    await fs.writeJson(transcriptionPath, result, { spaces: 2 });
    const resolvedSourceLanguage = String(language || '').trim();
    await ProjectManager.updateProject(projectId, {
      originalSubtitles: this.formatProjectOriginalSubtitles(result),
      transcriptionSourceLanguage:
        resolvedSourceLanguage && resolvedSourceLanguage.toLowerCase() !== 'auto'
          ? resolvedSourceLanguage
          : null,
      status: PROJECT_STATUS.TEXT_TRANSLATION,
    });
    onProgress('Transcription completed.');

    return result;
  }

  private static async transcribeCloud(
    filePath: string,
    config: any,
    options: {
      language?: string;
      prompt?: string;
      segmentation?: boolean;
      wordAlignment?: boolean;
      vad?: boolean;
      diarization?: boolean;
      diarizationOptions?: DiarizationOptions | null;
      audioDurationSec?: number | null;
      decodePolicy?: AsrDecodePolicy;
    },
    signal?: AbortSignal
  ) {
    const resolvedProvider = resolveCloudAsrProvider({
      url: String(config?.url || ''),
      modelName: String(config?.name || ''),
      model: String(config?.model || ''),
    });
    const cloudOptions = { ...options };
    if (
      (
        resolvedProvider.provider === 'google-gemini-audio' ||
        resolvedProvider.provider === 'elevenlabs-scribe' ||
        resolvedProvider.provider === 'deepgram-listen' ||
        resolvedProvider.provider === 'gladia-pre-recorded'
      ) &&
      !(Number.isFinite(Number(cloudOptions.audioDurationSec)) && Number(cloudOptions.audioDurationSec) > 0)
    ) {
      const audioDurationSec = await this.getAudioDurationSeconds(filePath).catch(() => null);
      if (Number.isFinite(audioDurationSec || Number.NaN) && Number(audioDurationSec) > 0) {
        cloudOptions.audioDurationSec = Number(audioDurationSec);
      }
    }

    const result = await requestCloudAsr(
      filePath,
      resolvedProvider,
      {
        ...config,
        url: resolvedProvider.endpointUrl,
        model: resolvedProvider.provider === 'openai-whisper' ? resolvedProvider.effectiveModel : config?.model,
      },
      cloudOptions,
      {
        createAbortSignalWithTimeout: this.createAbortSignalWithTimeout.bind(this),
        extractStructuredTranscript: (transcript, transcriptLanguage, extractOptions = {}) =>
          this.extractStructuredTranscript(transcript, transcriptLanguage, {
            ...extractOptions,
            preferAlignmentFirstNoSpace: Boolean(options.wordAlignment),
          }),
        disableWordAlignment: this.disableWordAlignment.bind(this),
      },
      signal
    );

    return {
      ...result,
      meta: {
        ...result.meta,
        isWhisperCppInference: resolvedProvider.provider === 'whispercpp-inference',
        isElevenLabsScribe: resolvedProvider.provider === 'elevenlabs-scribe',
        isDeepgramListen: resolvedProvider.provider === 'deepgram-listen',
        isGladiaPreRecorded: resolvedProvider.provider === 'gladia-pre-recorded',
        isGithubModelsPhi4Multimodal: resolvedProvider.provider === 'github-models-phi4-multimodal',
        isGoogleCloudChirp3: resolvedProvider.provider === 'google-cloud-chirp3',
        isGoogleGeminiAudio: resolvedProvider.provider === 'google-gemini-audio',
      },
    };
  }

  private static async transcribeGeminiCloudWindowBatch(
    inputs: Array<{ task: AsrVadWindowTask; audioPath: string }>,
    config: any,
    options: {
      language?: string;
      prompt?: string;
      segmentation?: boolean;
      wordAlignment?: boolean;
      vad?: boolean;
      diarization?: boolean;
      diarizationOptions?: DiarizationOptions | null;
      decodePolicy?: AsrDecodePolicy;
    },
    signal?: AbortSignal
  ) {
    const resolvedProvider = resolveCloudAsrProvider({
      url: String(config?.url || ''),
      modelName: String(config?.name || ''),
      model: String(config?.model || ''),
    });
    const totalDurationSec = inputs.reduce((sum, input) => sum + Math.max(0, input.task.durationSec), 0);
    return requestGeminiCloudAsrBatch(
      inputs.map((input) => ({
        filePath: input.audioPath,
        index: input.task.index,
        audioDurationSec: input.task.durationSec,
      })),
      resolvedProvider,
      {
        ...config,
        url: resolvedProvider.endpointUrl,
        model: config?.model,
      },
      {
        ...options,
        audioDurationSec: totalDurationSec,
      },
      {
        createAbortSignalWithTimeout: this.createAbortSignalWithTimeout.bind(this),
      },
      signal
    );
  }

  private static mergeGeminiCloudWindowBatchResponses(
    responses: Array<Awaited<ReturnType<typeof requestGeminiCloudAsrBatch>>>
  ) {
    const meta = this.mergeProviderMeta(responses.map((response) => response.meta));
    const batching = meta.geminiMultiAudioBatching || {};
    return {
      results: responses
        .flatMap((response) => response.results)
        .sort((a, b) => this.toFiniteNumber(a.input.index, 0) - this.toFiniteNumber(b.input.index, 0)),
      meta: {
        ...meta,
        geminiMultiAudioBatching: {
          ...batching,
          enabled: true,
          adaptiveSplit: true,
          leafBatchCount: responses.length,
        },
      },
    };
  }

  private static async transcribeGeminiCloudWindowBatchAdaptive(
    inputs: Array<{ task: AsrVadWindowTask; audioPath: string }>,
    config: any,
    options: {
      language?: string;
      prompt?: string;
      segmentation?: boolean;
      wordAlignment?: boolean;
      vad?: boolean;
      diarization?: boolean;
      diarizationOptions?: DiarizationOptions | null;
      decodePolicy?: AsrDecodePolicy;
    },
    onProgress: (msg: string) => void,
    signal?: AbortSignal,
    depth = 0
  ): Promise<Awaited<ReturnType<typeof requestGeminiCloudAsrBatch>>> {
    const result = await this.transcribeGeminiCloudWindowBatch(inputs, config, options, signal);
    const batching = result.meta?.geminiMultiAudioBatching || {};
    const missingItemCount = this.toFiniteNumber(batching.missingItemCount, 0);
    const rawItemCount = this.toFiniteNumber(batching.rawItemCount, result.results.length);
    const incomplete = missingItemCount > 0 || rawItemCount < inputs.length;

    if (!incomplete || inputs.length <= 1) {
      return result;
    }

    const midpoint = Math.ceil(inputs.length / 2);
    const left = inputs.slice(0, midpoint);
    const right = inputs.slice(midpoint);
    onProgress(
      `Gemini batch returned ${rawItemCount}/${inputs.length} mapped items; splitting into ${left.length}+${right.length} windows for accuracy...`
    );

    const responses = [];
    responses.push(await this.transcribeGeminiCloudWindowBatchAdaptive(left, config, options, onProgress, signal, depth + 1));
    if (right.length > 0) {
      responses.push(await this.transcribeGeminiCloudWindowBatchAdaptive(right, config, options, onProgress, signal, depth + 1));
    }
    const merged = this.mergeGeminiCloudWindowBatchResponses(responses);
    const discardedRequestCount = this.toFiniteNumber(result.meta?.geminiMultiAudioBatching?.requestCount, 1);
    const mergedRequestCount = this.toFiniteNumber(merged.meta.geminiMultiAudioBatching?.requestCount, 0);
    merged.meta.callCount = this.toFiniteNumber(merged.meta.callCount, 0) + discardedRequestCount;
    merged.meta.geminiMultiAudioBatching = {
      ...(merged.meta.geminiMultiAudioBatching || {}),
      adaptiveSplit: true,
      adaptiveSplitDepth: Math.max(
        depth + 1,
        this.toFiniteNumber(merged.meta.geminiMultiAudioBatching?.adaptiveSplitDepth, 0)
      ),
      requestCount: mergedRequestCount + discardedRequestCount,
      discardedRequestCount:
        this.toFiniteNumber(merged.meta.geminiMultiAudioBatching?.discardedRequestCount, 0) + discardedRequestCount,
      firstIncompleteRawItemCount: rawItemCount,
      firstIncompleteExpectedItemCount: inputs.length,
    };
    return merged;
  }

  static async testConnection(input: { url: string; key?: string; model?: string; name?: string }) {
    const resolvedProvider = resolveCloudAsrProvider({
      url: String(input.url || '').trim(),
      modelName: String(input.name || '').trim(),
      model: String(input.model || '').trim(),
    });

    const headers = buildCloudAsrRequestHeaders(resolvedProvider.provider, input.key);
    const isGithubModelsProvider = resolvedProvider.provider === 'github-models-phi4-multimodal';
    const isGoogleCloudChirp3Provider = resolvedProvider.provider === 'google-cloud-chirp3';
    const isGoogleGeminiAudioProvider = resolvedProvider.provider === 'google-gemini-audio';
    const isDeepgramProvider = resolvedProvider.provider === 'deepgram-listen';
    const isGladiaProvider = resolvedProvider.provider === 'gladia-pre-recorded';
    const testHeaders = (isGithubModelsProvider || isGoogleCloudChirp3Provider || isGoogleGeminiAudioProvider || isGladiaProvider)
      ? { ...headers, 'Content-Type': 'application/json' }
      : isDeepgramProvider
        ? { ...headers, 'Content-Type': 'audio/wav', Accept: 'application/json' }
      : headers;
    let testUrl = resolvedProvider.endpointUrl;
    let testBody: BodyInit | undefined;
    if (isGithubModelsProvider) {
      testBody = JSON.stringify({
          model: resolvedProvider.effectiveModel,
          messages: [{ role: 'user', content: 'Reply with OK.' }],
          temperature: 0,
          top_p: 1,
          max_tokens: 8,
          stream: false,
        });
    } else if (isGoogleCloudChirp3Provider) {
      testBody = JSON.stringify({
        config: {
          autoDecodingConfig: {},
          languageCodes: ['auto'],
          model: resolvedProvider.effectiveModel,
          features: {
            enableWordTimeOffsets: false,
            enableAutomaticPunctuation: true,
          },
        },
        content: '',
      });
    } else if (isGoogleGeminiAudioProvider) {
      testBody = JSON.stringify({
        contents: [
          {
            role: 'user',
            parts: [{ text: 'Reply with OK.' }],
          },
        ],
        generationConfig: {
          temperature: 0,
          maxOutputTokens: 8,
        },
      });
    } else if (isDeepgramProvider) {
      const nextUrl = new URL(resolvedProvider.endpointUrl);
      nextUrl.searchParams.set('model', resolvedProvider.effectiveModel);
      nextUrl.searchParams.set('smart_format', 'true');
      testUrl = nextUrl.toString();
      testBody = new Blob([new Uint8Array(0)], { type: 'audio/wav' });
    } else if (isGladiaProvider) {
      testBody = JSON.stringify({});
    }

    const request = this.createAbortSignalWithTimeout(8000);
    try {
      const response = await fetch(testUrl, {
        method: 'POST',
        headers: testHeaders,
        body: testBody,
        signal: request.signal,
        redirect: 'error',
      });

      const contentType = String(response.headers.get('content-type') || '');
      const responseText = await response.text();
      const detail = extractErrorMessage(responseText, contentType);
      const pathLower = new URL(resolvedProvider.endpointUrl).pathname.toLowerCase();
      const isOpenAiAsrEndpoint = pathLower.includes('/audio/transcriptions');
      const isWhisperCppInferenceEndpoint = pathLower.includes('/inference');
      const isElevenLabsScribeEndpoint = resolvedProvider.provider === 'elevenlabs-scribe' || pathLower.includes('/speech-to-text');
      const isGithubModelsEndpoint = resolvedProvider.provider === 'github-models-phi4-multimodal' || pathLower.includes('/inference/chat/completions');
      const isGoogleCloudChirp3Endpoint = resolvedProvider.provider === 'google-cloud-chirp3' || pathLower.includes('/recognizers/');
      const isGoogleGeminiAudioEndpoint = resolvedProvider.provider === 'google-gemini-audio' || pathLower.includes(':generatecontent');
      const isDeepgramEndpoint = resolvedProvider.provider === 'deepgram-listen' || pathLower.includes('/listen');
      const isGladiaEndpoint = resolvedProvider.provider === 'gladia-pre-recorded' || pathLower.includes('/v2/pre-recorded');
      const authFailed = hasAuthFailureHint(detail);
      const payloadValidationFailed = hasPayloadValidationHint(detail);
      const googlePayloadValidationFailed =
        payloadValidationFailed || /audio|content|recognize|recognition|config/i.test(detail || '');
      const deepgramPayloadValidationFailed =
        payloadValidationFailed || /audio|media|source|decode|encoding|listen/i.test(detail || '');
      const gladiaPayloadValidationFailed =
        payloadValidationFailed || /audio_url|audio|url|required|invalid|missing/i.test(detail || '');

      if (response.status >= 200 && response.status < 300) {
        return { success: true, message: 'Connection succeeded.' };
      }

      if (response.status === 401 || response.status === 403 || authFailed) {
        return {
          success: false,
          error: detail || `Authentication failed (${response.status}).`,
        };
      }

      if (response.status === 404) {
        return {
          success: false,
          error: `Endpoint not found (404). Please verify the API URL path.${detail ? ` ${detail}` : ''}`,
        };
      }

      if (response.status === 405) {
        return {
          success: false,
          error: `Method not allowed (405). Please verify this endpoint supports POST.${detail ? ` ${detail}` : ''}`,
        };
      }

      if (response.status === 429) {
        return {
          success: false,
          error: `Rate limited (429). Please retry later.${detail ? ` ${detail}` : ''}`,
        };
      }

      if (response.status >= 500 && !isWhisperCppInferenceEndpoint) {
        return {
          success: false,
          error: `Provider server error (${response.status}).${detail ? ` ${detail}` : ''}`,
        };
      }

      if (
        (isOpenAiAsrEndpoint || isWhisperCppInferenceEndpoint) &&
        payloadValidationFailed &&
        (
          response.status === 400 ||
          response.status === 415 ||
          response.status === 422 ||
          (isWhisperCppInferenceEndpoint && response.status === 500)
        )
      ) {
        return {
          success: true,
          message: 'Endpoint reachable. Request payload validation failed as expected.',
        };
      }

      if (
        isElevenLabsScribeEndpoint &&
        payloadValidationFailed &&
        (
          response.status === 400 ||
          response.status === 415 ||
          response.status === 422
        )
      ) {
        return {
          success: true,
          message: 'Endpoint reachable. Request payload validation failed as expected.',
        };
      }

      if (
        isGithubModelsEndpoint &&
        payloadValidationFailed &&
        (response.status === 400 || response.status === 415 || response.status === 422)
      ) {
        return {
          success: true,
          message: 'Endpoint reachable. Request payload validation failed as expected.',
        };
      }

      if (
        isGoogleCloudChirp3Endpoint &&
        googlePayloadValidationFailed &&
        (response.status === 400 || response.status === 415 || response.status === 422)
      ) {
        return {
          success: true,
          message: 'Endpoint reachable. Request payload validation failed as expected.',
        };
      }

      if (
        isGoogleGeminiAudioEndpoint &&
        payloadValidationFailed &&
        (response.status === 400 || response.status === 415 || response.status === 422)
      ) {
        return {
          success: true,
          message: 'Endpoint reachable. Request payload validation failed as expected.',
        };
      }

      if (
        isDeepgramEndpoint &&
        deepgramPayloadValidationFailed &&
        (response.status === 400 || response.status === 415 || response.status === 422)
      ) {
        return {
          success: true,
          message: 'Endpoint reachable. Request payload validation failed as expected.',
        };
      }

      if (
        isGladiaEndpoint &&
        gladiaPayloadValidationFailed &&
        (response.status === 400 || response.status === 415 || response.status === 422)
      ) {
        return {
          success: true,
          message: 'Endpoint reachable. Request payload validation failed as expected.',
        };
      }

      return {
        success: false,
        error: `Unexpected response (${response.status}).${detail ? ` ${detail}` : ''}`,
      };
    } catch (error: any) {
      if (error?.name === 'AbortError') {
        return { success: false, error: 'Connection timeout.' };
      }
      return { success: false, error: `Connection failed: ${error?.message || String(error)}` };
    } finally {
      request.dispose();
    }
  }

  private static toFiniteNumber(value: any, fallback = 0): number {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }

  private static firstString(candidates: any[]): string | null {
    for (const candidate of candidates) {
      if (typeof candidate === 'string' && candidate.trim()) {
        return candidate;
      }
    }
    return null;
  }

  private static normalizeChunkText(value: string): string {
    return value.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim();
  }

  private static normalizeDisplayText(value: string, language?: string): string {
    const cleaned = this.normalizeChunkText(value);
    if (!cleaned) return '';
    if (this.shouldTreatTextAsNoSpaceScript(cleaned, language)) {
      return cleaned
        .replace(/\s+/g, '')
        .replace(/([a-z])([A-Z])/g, '$1 $2')
        .replace(/([A-Za-z])([\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af])/gu, '$1 $2')
        .replace(/([\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af])([A-Za-z])/gu, '$1 $2')
        .trim();
    }

    return cleaned
      .replace(/([\p{L}\p{N}])\s*(['’])\s*(\p{L})/gu, '$1$2$3')
      .replace(/([\p{L}\p{N}])\s*-\s*(\p{L}|\d)/gu, '$1-$2')
      .replace(/(\d)\s*-\s*(\p{L}|\d)/gu, '$1-$2')
      .replace(/\s+([,.;:!?%])/g, '$1')
      .replace(/([\(\[\{¿¡])\s+/g, '$1')
      .replace(/\s+/g, ' ')
      .trim();
  }

  private static normalizeWordText(value: string): string {
    return String(value || '')
      .replace(/\r\n/g, '\n')
      .replace(/\r/g, '\n')
      .replace(/\u00a0/g, ' ')
      .trim();
  }

  private static getBuiltInLocalAsrPrompt(language?: string, sampleText?: string) {
    void language;
    void sampleText;
    return '';
  }

  private static buildEffectiveLocalAsrPrompt(language?: string, userPrompt?: string, sampleText?: string) {
    const builtInPrompt = this.getBuiltInLocalAsrPrompt(language, sampleText);
    const customPrompt = String(userPrompt || '').trim();
    if (builtInPrompt && customPrompt) {
      if (builtInPrompt === customPrompt) return builtInPrompt;
      return `${builtInPrompt}\n${customPrompt}`;
    }
    return builtInPrompt || customPrompt;
  }

  private static stripReplacementCharacters(value: string) {
    return String(value || '').replace(/\uFFFD+/g, '');
  }

  private static countReplacementCharacters(value: string) {
    const matched = String(value || '').match(/\uFFFD/g);
    return matched ? matched.length : 0;
  }

  private static isLanguageWithoutSpaces(language?: string): boolean {
    return isPolicyNoSpaceLanguage(language) || Boolean(LanguageAlignmentRegistry.getNoSpaceModule(language));
  }

  private static countNoSpaceScriptChars(value: string) {
    const matched = String(value || '').match(/[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]/g);
    return matched ? matched.length : 0;
  }

  private static countLatinScriptChars(value: string) {
    const matched = String(value || '').match(/[A-Za-z]/g);
    return matched ? matched.length : 0;
  }

  private static shouldTreatTextAsNoSpaceScript(value: string, language?: string) {
    if (this.isLanguageWithoutSpaces(language)) return true;
    const text = String(value || '');
    const noSpaceChars = this.countNoSpaceScriptChars(text);
    const latinChars = this.countLatinScriptChars(text);
    return noSpaceChars >= 2 && noSpaceChars >= latinChars;
  }

  private static shouldTreatWordListAsNoSpaceScript(words: Array<{ text?: string }>, language?: string) {
    if (this.isLanguageWithoutSpaces(language)) return true;
    const combined = words.map((word) => String(word?.text || '')).join('');
    return this.shouldTreatTextAsNoSpaceScript(combined, language);
  }

  private static isCjkTimingMergeLanguage(language?: string) {
    return Boolean(LanguageAlignmentRegistry.getNoSpaceModule(language)?.config.timingMerge);
  }

  private static isSentenceTerminalToken(token: string) {
    return /^[。！？!?]+$/.test(String(token || '').trim());
  }

  private static isBracketToken(token: string) {
    return /^[()\[\]{}「」『』（）【】〈〉《》]+$/.test(String(token || '').trim());
  }

  private static isPunctuationOnlyToken(token: string) {
    return /^[\p{P}\p{S}]+$/u.test(String(token || '').trim());
  }

  private static shouldPreferStandaloneCjkToken(token: string, language?: string) {
    return LanguageAlignmentRegistry.getNoSpaceModule(language, token)?.shouldPreferStandaloneToken(token, language) || false;
  }

  private static startsWithStandaloneCjkToken(token: string, language?: string) {
    return LanguageAlignmentRegistry.getNoSpaceModule(language, token)?.startsWithStandaloneToken(token, language) || false;
  }

  private static endsWithClosingBracketToken(token: string) {
    return /[\)）」』】〉》]$/.test(String(token || '').trim());
  }

  private static hasUnclosedBracketToken(token: string) {
    const text = String(token || '');
    const opens = (text.match(/[(「『（【〈《]/g) || []).length;
    const closes = (text.match(/[\)）」』】〉》]/g) || []).length;
    return opens > closes;
  }

  private static endsWithJapaneseLetter(token: string) {
    return /[\u3040-\u30ff\u3400-\u9fff]$/.test(String(token || '').trim());
  }

  private static isShortJapaneseContinuationToken(token: string, language?: string) {
    return LanguageAlignmentRegistry.getNoSpaceModule(language, token)?.isShortContinuationToken(token, language) || false;
  }

  private static shouldMergeJapaneseContinuation(
    bufferText: string,
    nextText: string,
    gap: number,
    combinedText: string,
    combinedDuration: number,
    language?: string
  ) {
    return (
      LanguageAlignmentRegistry.getNoSpaceModule(language, `${bufferText}${nextText}`)?.shouldMergeContinuation(
        bufferText,
        nextText,
        gap,
        combinedText,
        combinedDuration,
        language
      ) || false
    );
  }

  private static isShortJapaneseTailToken(token: string, language?: string) {
    return LanguageAlignmentRegistry.getNoSpaceModule(language, token)?.isShortTailToken(token, language) || false;
  }

  private static normalizeNoSpaceAlignmentText(value: string) {
    return normalizeNoSpaceAlignmentText(value);
  }

  private static getNoSpaceSegmenterLocale(language?: string, sampleText?: string) {
    return LanguageAlignmentRegistry.getNoSpaceModule(language, sampleText)?.config.segmenterLocale || getSegmenterLocale(language, 'ja');
  }

  private static getNoSpaceVariantDebug(language?: string, sampleText?: string) {
    return LanguageAlignmentRegistry.getNoSpaceModule(language, sampleText)?.getVariantDebug?.(language, sampleText) || null;
  }

  private static getNoSpaceVariantNumber(
    language: string | undefined,
    sampleText: string | undefined,
    key: string,
    fallback: number
  ) {
    const variantDebug = this.getNoSpaceVariantDebug(language, sampleText);
    const candidate = Number((variantDebug as Record<string, unknown> | null)?.[key]);
    return Number.isFinite(candidate) ? candidate : fallback;
  }

  private static evaluateNoSpaceAlignmentQualityGate(diagnostics: any, language?: string) {
    if (!this.isLanguageWithoutSpaces(language)) {
      return { accepted: true, reason: null as string | null };
    }

    const attempted = this.toFiniteNumber(diagnostics?.attemptedSegmentCount, 0);
    const aligned = this.toFiniteNumber(diagnostics?.alignedSegmentCount, 0);
    const rawAvgConfidence = diagnostics?.avgConfidence;
    const avgConfidence =
      rawAvgConfidence == null ? Number.NaN : this.toFiniteNumber(rawAvgConfidence, Number.NaN);
    const alignedRatio = attempted > 0 ? aligned / attempted : 0;

    const minAlignedRatio = this.getNoSpaceVariantNumber(language, undefined, 'alignmentMinAppliedRatio', 0.22);
    const minAvgConfidence = this.getNoSpaceVariantNumber(language, undefined, 'alignmentMinAvgConfidence', 0.5);

    if (attempted >= 3 && alignedRatio < minAlignedRatio) {
      return { accepted: false, reason: `aligned_ratio_below_${minAlignedRatio}` };
    }
    if (Number.isFinite(avgConfidence) && avgConfidence < minAvgConfidence) {
      return { accepted: false, reason: `avg_confidence_below_${minAvgConfidence}` };
    }

    return { accepted: true, reason: null as string | null };
  }

  private static fallbackSegmentNoSpaceLexicalUnits(text: string, language?: string) {
    return (
      LanguageAlignmentRegistry.getNoSpaceModule(language, text)?.fallbackSegmentLexicalUnits(text) ||
      genericFallbackSegmentNoSpaceLexicalUnits(text)
    );
  }

  private static segmentNoSpaceLexicalUnits(text: string, language?: string) {
    const normalized = this.normalizeNoSpaceAlignmentText(text);
    if (!normalized) {
      return {
        units: [] as string[],
        usedIntlSegmenter: false,
        locale: this.getNoSpaceSegmenterLocale(language, normalized),
      };
    }

    const locale = this.getNoSpaceSegmenterLocale(language, normalized);
    try {
      const segmenter = new Intl.Segmenter(locale, { granularity: 'word' });
      const units = Array.from(segmenter.segment(normalized))
        .map((part) => String(part.segment || '').trim())
        .filter(Boolean);
      if (units.length > 0) {
        return { units, usedIntlSegmenter: true, locale };
      }
    } catch {
      // Fall back to script-run segmentation below.
    }

    return {
      units: this.fallbackSegmentNoSpaceLexicalUnits(normalized, language),
      usedIntlSegmenter: false,
      locale,
    };
  }

  private static projectNoSpaceLexicalWordSegments(
    words: AsrWordSegment[],
    text: string,
    language?: string
  ): {
    words: AsrWordSegment[];
    appliedLexicalProjection: boolean;
    diagnostics: {
      lexicalWordCount: number;
      lexicalSingleCharCount: number;
      lexicalMismatchCount: number;
      lexicalProjectionAppliedSegments: number;
      usedIntlSegmenter: boolean;
      locale: string;
    };
  } {
    const diagnostics = {
      lexicalWordCount: words.length,
      lexicalSingleCharCount: words.filter((word) => this.normalizeNoSpaceAlignmentText(word?.text || '').length === 1).length,
      lexicalMismatchCount: 0,
      lexicalProjectionAppliedSegments: 0,
      usedIntlSegmenter: false,
      locale: this.getNoSpaceSegmenterLocale(language, text),
    };

    if (!Array.isArray(words) || words.length === 0 || !this.shouldTreatWordListAsNoSpaceScript(words, language)) {
      return { words, appliedLexicalProjection: false, diagnostics };
    }

    const { units, usedIntlSegmenter, locale } = this.segmentNoSpaceLexicalUnits(text, language);
    diagnostics.usedIntlSegmenter = usedIntlSegmenter;
    diagnostics.locale = locale;
    if (units.length === 0) {
      return { words, appliedLexicalProjection: false, diagnostics };
    }

    const baseWords = words
      .map((word) => ({
        ...word,
        normalizedText: this.normalizeNoSpaceAlignmentText(word.text || ''),
      }))
      .filter((word) => word.normalizedText.length > 0);
    const lexicalUnits = units
      .map((unit) => ({
        text: unit,
        normalizedText: this.normalizeNoSpaceAlignmentText(unit),
      }))
      .filter((unit) => unit.normalizedText.length > 0);

    const baseText = baseWords.map((word) => word.normalizedText).join('');
    const lexicalText = lexicalUnits.map((unit) => unit.normalizedText).join('');
    if (!baseText || !lexicalText || baseText !== lexicalText) {
      diagnostics.lexicalMismatchCount = Math.abs(baseText.length - lexicalText.length) || 1;
      return { words, appliedLexicalProjection: false, diagnostics };
    }

    const charSpans: Array<{ char: string; start_ts: number; end_ts: number; probability?: number }> = [];
    for (const word of baseWords) {
      const chars = Array.from(word.normalizedText);
      const start = this.toFiniteNumber(word.start_ts, 0);
      const end = this.getWordEnd(word);
      const duration = Math.max(0.04 * chars.length, end - start);
      for (let index = 0; index < chars.length; index += 1) {
        charSpans.push({
          char: chars[index],
          start_ts: Number((start + duration * (index / chars.length)).toFixed(3)),
          end_ts: Number((start + duration * ((index + 1) / chars.length)).toFixed(3)),
          probability: word.probability,
        });
      }
    }

    if (charSpans.length !== Array.from(lexicalText).length) {
      diagnostics.lexicalMismatchCount = Math.abs(charSpans.length - Array.from(lexicalText).length) || 1;
      return { words, appliedLexicalProjection: false, diagnostics };
    }

    const projected: AsrWordSegment[] = [];
    let charCursor = 0;
    for (const unit of lexicalUnits) {
      const unitChars = Array.from(unit.normalizedText);
      const slice = charSpans.slice(charCursor, charCursor + unitChars.length);
      if (slice.length !== unitChars.length || slice.map((item) => item.char).join('') !== unit.normalizedText) {
        diagnostics.lexicalMismatchCount += 1;
        return { words, appliedLexicalProjection: false, diagnostics };
      }
      const probabilityValues = slice
        .map((item) => this.toFiniteNumber(item.probability, Number.NaN))
        .filter((value) => Number.isFinite(value));
      projected.push({
        text: unit.text,
        start_ts: slice[0].start_ts,
        end_ts: slice[slice.length - 1].end_ts,
        probability:
          probabilityValues.length > 0
            ? Number((probabilityValues.reduce((sum, value) => sum + value, 0) / probabilityValues.length).toFixed(4))
            : undefined,
      });
      charCursor += unitChars.length;
    }

    diagnostics.lexicalWordCount = projected.length;
    diagnostics.lexicalSingleCharCount = projected.filter((word) => this.normalizeNoSpaceAlignmentText(word?.text || '').length === 1).length;
    diagnostics.lexicalProjectionAppliedSegments = projected.length > 0 ? 1 : 0;
    return {
      words: projected.length > 0 ? projected : words,
      appliedLexicalProjection: projected.length > 0,
      diagnostics,
    };
  }

  private static segmentWhitespaceLexicalUnits(text: string) {
    return Array.from(
      String(text || '')
        .normalize('NFKC')
        .matchAll(/[\p{L}\p{M}\p{N}]+(?:['’-][\p{L}\p{M}\p{N}]+)*|[^\s]/gu)
    )
      .map((match) => String(match[0] || '').trim())
      .filter(Boolean);
  }

  private static buildSyntheticWordSegmentsFromText(
    text: string,
    startTs: number,
    endTs: number | undefined,
    language?: string,
    sourceSegmentIndex?: number
  ): AsrWordSegment[] {
    const normalizedText = this.normalizeDisplayText(text, language);
    if (!normalizedText) return [];
    const start = this.toFiniteNumber(startTs, 0);
    const end = this.ensureValidRangeEnd(start, endTs);
    const duration = Math.max(0.04, end - start);

    const noSpaces = this.shouldTreatTextAsNoSpaceScript(normalizedText, language);
    const units = noSpaces
      ? this.segmentNoSpaceLexicalUnits(normalizedText, language).units
      : this.segmentWhitespaceLexicalUnits(normalizedText);
    const lexicalUnits = units.filter(Boolean);
    if (lexicalUnits.length === 0) {
      return [
        {
          text: normalizedText,
          start_ts: start,
          end_ts: end,
          source_segment_index: sourceSegmentIndex,
        },
      ];
    }

    const totalWeight = lexicalUnits.reduce((sum, unit) => sum + Math.max(1, Array.from(unit).length), 0);
    let cursor = 0;
    return lexicalUnits.map((unit) => {
      const weight = Math.max(1, Array.from(unit).length);
      const tokenStart = start + (duration * cursor) / totalWeight;
      cursor += weight;
      const tokenEnd = start + (duration * cursor) / totalWeight;
      return {
        text: unit,
        start_ts: Number(tokenStart.toFixed(3)),
        end_ts: Number(tokenEnd.toFixed(3)),
        source_segment_index: sourceSegmentIndex,
      } satisfies AsrWordSegment;
    });
  }

  private static shouldRecoverSparseNoSpaceNativeWords(
    words: AsrWordSegment[],
    text: string,
    language?: string
  ) {
    const normalizedText = normalizeNoSpaceAlignmentText(text);
    const minTextLength = this.getNoSpaceVariantNumber(language, normalizedText, 'cjkSparseMinTextLength', 4);
    const singleWordMinTextLength = this.getNoSpaceVariantNumber(language, normalizedText, 'cjkSparseSingleWordMinTextLength', 4);
    const lowCoverageRatio = this.getNoSpaceVariantNumber(language, normalizedText, 'cjkSparseLowCoverageRatio', 0.38);
    const mediumCoverageRatio = this.getNoSpaceVariantNumber(language, normalizedText, 'cjkSparseMediumCoverageRatio', 0.55);
    const mediumCoverageMaxWords = this.getNoSpaceVariantNumber(language, normalizedText, 'cjkSparseMediumCoverageMaxWords', 3);
    if (!normalizedText || normalizedText.length < minTextLength) return false;
    if (!this.shouldTreatTextAsNoSpaceScript(normalizedText, language)) return false;

    const nativeText = normalizeNoSpaceAlignmentText(this.joinWordSegments(words, language));
    if (!nativeText) return true;

    const coverageRatio = nativeText.length / Math.max(1, normalizedText.length);
    if (words.length <= 1 && normalizedText.length >= singleWordMinTextLength) return true;
    if (coverageRatio < lowCoverageRatio) return true;
    if (coverageRatio < mediumCoverageRatio && words.length <= mediumCoverageMaxWords) return true;
    return false;
  }

  private static splitNoSpaceSegmentByWords(segment: AsrStructuredSegment, language?: string): AsrStructuredSegment[] {
    const words = Array.isArray(segment?.words) ? segment.words : [];
    if (words.length <= 1 || !this.shouldTreatWordListAsNoSpaceScript(words, language)) {
      return [segment];
    }

    const sampleText = String(segment?.text || words.map((word) => word?.text || '').join(''));
    const maxDurationSec = this.getNoSpaceVariantNumber(language, sampleText, 'cjkSplitMaxDurationSec', 2.45);
    const strongPauseSec = this.getNoSpaceVariantNumber(language, sampleText, 'cjkSplitStrongPauseSec', 0.34);
    const clausePauseSec = this.getNoSpaceVariantNumber(language, sampleText, 'cjkSplitClausePauseSec', 0.18);
    const maxChars = this.getNoSpaceVariantNumber(language, sampleText, 'cjkSplitMaxChars', 14);
    const minCharsForBreak = this.getNoSpaceVariantNumber(language, sampleText, 'cjkSplitMinCharsForBreak', 6);
    const segments: AsrStructuredSegment[] = [];
    let bufferStart = 0;

    const finalize = (startIndex: number, endIndex: number) => {
      const slice = words.slice(startIndex, endIndex + 1);
      if (slice.length === 0) return;
      const text = this.joinWordSegments(slice, language);
      if (!text) return;
      segments.push({
        text,
        start_ts: this.toFiniteNumber(slice[0]?.start_ts, this.toFiniteNumber(segment?.start_ts, 0)),
        end_ts: this.ensureValidRangeEnd(
          this.toFiniteNumber(slice[0]?.start_ts, this.toFiniteNumber(segment?.start_ts, 0)),
          this.getWordEnd(slice[slice.length - 1])
        ),
        speaker: this.getDominantSpeaker(slice) || segment.speaker,
        words: slice.map((word) => ({ ...word })),
      });
    };

    for (let index = 1; index < words.length; index += 1) {
      const current = words[index];
      const previous = words[index - 1];
      const candidateWords = words.slice(bufferStart, index);
      const candidateText = this.joinWordSegments(candidateWords, language);
      const candidateChars = candidateText.length;
      const gap = this.toFiniteNumber(current?.start_ts, 0) - this.getWordEnd(previous);
      const nextDuration = this.getWordEnd(current) - this.toFiniteNumber(words[bufferStart]?.start_ts, 0);
      const endedSentence = /[。！？!?]$/.test(candidateText);
      const endedClause = /[、，；：]$/.test(candidateText);
      const speakerChanged = Boolean(previous?.speaker && current?.speaker && previous.speaker !== current.speaker);
      const shouldBreak =
        speakerChanged ||
        gap >= strongPauseSec ||
        (endedSentence && candidateChars >= minCharsForBreak) ||
        (endedClause && gap >= clausePauseSec && candidateChars >= minCharsForBreak) ||
        nextDuration > maxDurationSec ||
        candidateChars >= maxChars;

      if (shouldBreak) {
        finalize(bufferStart, index - 1);
        bufferStart = index;
      }
    }

    finalize(bufferStart, words.length - 1);
    return segments.length > 0 ? segments : [segment];
  }

  private static mergeCjkWordSegments(
    words: AsrWordSegment[],
    language?: string
  ): {
    words: AsrWordSegment[];
    appliedCjkMerging: boolean;
    diagnostics: {
      rawWordCount: number;
      mergedWordCount: number;
      droppedWordCount: number;
      replacementCharCount: number;
      rawSingleCharCount: number;
      mergedSingleCharCount: number;
      punctuationOnlyCount: number;
    };
  } {
    const prepared = words
      .map((word) => {
        const cleaned = this.stripReplacementCharacters(this.normalizeWordText(word?.text || ''));
        const compact = this.isLanguageWithoutSpaces(language) ? cleaned.replace(/\s+/g, '') : cleaned;
        return compact
          ? {
              ...word,
              text: compact,
            }
          : null;
      })
      .filter(Boolean) as AsrWordSegment[];

    const diagnostics = {
      rawWordCount: words.length,
      mergedWordCount: 0,
      droppedWordCount: Math.max(0, words.length - prepared.length),
      replacementCharCount: words.reduce((sum, word) => sum + this.countReplacementCharacters(word?.text || ''), 0),
      rawSingleCharCount: words.filter((word) => this.stripReplacementCharacters(this.normalizeWordText(word?.text || '')).length === 1).length,
      mergedSingleCharCount: 0,
      punctuationOnlyCount: prepared.filter((word) => this.isPunctuationOnlyToken(word.text)).length,
    };

    const shouldApplyCjkMerging = prepared.length > 0 && this.shouldTreatWordListAsNoSpaceScript(prepared, language);

    if (!shouldApplyCjkMerging) {
      diagnostics.mergedWordCount = prepared.length;
      diagnostics.mergedSingleCharCount = prepared.filter((word) => word.text.length === 1).length;
      return { words: prepared, appliedCjkMerging: false, diagnostics };
    }

    const sampleText = this.joinWordSegments(prepared, language);
    const softGapSec = this.getNoSpaceVariantNumber(language, sampleText, 'cjkMergeSoftGapSec', 0.11);
    const hardGapSec = this.getNoSpaceVariantNumber(language, sampleText, 'cjkMergeHardGapSec', 0.24);
    const maxPhraseChars = this.getNoSpaceVariantNumber(language, sampleText, 'cjkMergeMaxPhraseChars', 12);
    const maxPhraseDurationSec = this.getNoSpaceVariantNumber(language, sampleText, 'cjkMergeMaxPhraseDurationSec', 1.7);
    const merged: AsrWordSegment[] = [];
    let buffer: AsrWordSegment[] = [];

    const finalize = () => {
      if (buffer.length === 0) return;
      const first = buffer[0];
      const last = buffer[buffer.length - 1];
      const text = this.joinWordSegments(buffer, language).replace(/\uFFFD+/g, '').trim();
      if (text) {
        const probabilityValues = buffer
          .map((word) => this.toFiniteNumber(word?.probability, Number.NaN))
          .filter((value) => Number.isFinite(value));
        merged.push({
          ...first,
          text,
          start_ts: this.toFiniteNumber(first.start_ts, 0),
          end_ts: this.getWordEnd(last),
          probability:
            probabilityValues.length > 0
              ? Number((probabilityValues.reduce((sum, value) => sum + value, 0) / probabilityValues.length).toFixed(4))
              : undefined,
        });
      }
      buffer = [];
    };

    const shouldMergeIntoBuffer = (next: AsrWordSegment) => {
      if (buffer.length === 0) return true;
      const previous = buffer[buffer.length - 1];
      const gap = this.toFiniteNumber(next.start_ts, 0) - this.getWordEnd(previous);
      if (gap >= hardGapSec) return false;

      const bufferText = this.joinWordSegments(buffer, language);
      const combinedText = this.joinWordSegments([...buffer, next], language);
      const combinedDuration = this.getWordEnd(next) - this.toFiniteNumber(buffer[0]?.start_ts, 0);
      const nextText = this.normalizeWordText(next.text || '');

      if (this.isSentenceTerminalToken(previous.text) && gap > 0.04) {
        return false;
      }
      if (this.endsWithClosingBracketToken(bufferText) && gap > 0.04) {
        return false;
      }
      if (this.hasUnclosedBracketToken(bufferText) && combinedDuration <= 2.1 && combinedText.length <= maxPhraseChars + 4) {
        return true;
      }
      if (this.isBracketToken(previous.text) || this.isBracketToken(nextText)) {
        return combinedDuration <= maxPhraseDurationSec;
      }
      if (combinedText.length > maxPhraseChars || combinedDuration > maxPhraseDurationSec) {
        return false;
      }
      if (this.isPunctuationOnlyToken(nextText)) {
        return gap <= softGapSec;
      }
      if (gap <= softGapSec) {
        return true;
      }
      return combinedText.length <= 4 && gap <= 0.16;
    };

    for (const word of prepared) {
      if (shouldMergeIntoBuffer(word)) {
        buffer.push(word);
        if (this.isSentenceTerminalToken(word.text)) {
          finalize();
        }
        continue;
      }
      finalize();
      buffer.push(word);
      if (this.isSentenceTerminalToken(word.text)) {
        finalize();
      }
    }
    finalize();

    diagnostics.mergedWordCount = merged.length;
    diagnostics.mergedSingleCharCount = merged.filter((word) => word.text.length === 1).length;
    return { words: merged, appliedCjkMerging: true, diagnostics };
  }

  private static shouldAttachWithoutLeadingSpace(token: string): boolean {
    return /^[,.;:!?%)\]\}、。！？・，；：」』）】〕〉》〟’”]/.test(token);
  }

  private static shouldAttachWithoutTrailingSpace(text: string): boolean {
    return /[(\[\{「『（【〔〈《〝‘“'"-]$/.test(text);
  }

  private static joinWordSegments(words: AsrWordSegment[], language?: string): string {
    const noSpaces = this.shouldTreatWordListAsNoSpaceScript(words, language);
    let combined = '';

    for (const item of words) {
      const token = this.normalizeWordText(item?.text || '');
      if (!token) continue;

      if (!combined) {
        combined = noSpaces ? token.replace(/\s+/g, '') : token;
        continue;
      }

      if (noSpaces) {
        combined += token.replace(/\s+/g, '');
        continue;
      }

      if (this.shouldAttachWithoutLeadingSpace(token) || this.shouldAttachWithoutTrailingSpace(combined)) {
        combined += token;
      } else {
        combined += ` ${token}`;
      }
    }

    return this.normalizeDisplayText(combined.replace(/\s+/g, noSpaces ? '' : ' '), language);
  }

  private static getWordEnd(word: AsrWordSegment) {
    const end = this.toFiniteNumber(word?.end_ts, Number.NaN);
    if (Number.isFinite(end)) return end;
    return this.toFiniteNumber(word?.start_ts, 0);
  }

  private static ensureValidRangeEnd(startTs: number, endTs: number | undefined, minDurationSec = 0.04) {
    const start = this.toFiniteNumber(startTs, 0);
    const end = this.toFiniteNumber(endTs, start);
    return end > start ? end : Number((start + minDurationSec).toFixed(3));
  }

  private static makeChunkFromWords(words: AsrWordSegment[], language?: string): AsrTranscriptChunk | null {
    if (!Array.isArray(words) || words.length === 0) return null;
    const first = words[0];
    const last = words[words.length - 1];
    const text = this.joinWordSegments(words, language);
    if (!text) return null;
    const sourceSegmentIndices = Array.from(
      new Set(
        words
          .map((word) => this.toFiniteNumber(word?.source_segment_index, Number.NaN))
          .filter((value) => Number.isFinite(value))
      )
    );
    return {
      text,
      start_ts: this.toFiniteNumber(first.start_ts, 0),
      end_ts: this.ensureValidRangeEnd(this.toFiniteNumber(first.start_ts, 0), this.getWordEnd(last)),
      word_start_index: this.toFiniteNumber(first.token_index, 0),
      word_end_index: this.toFiniteNumber(last.token_index, 0),
      source_segment_indices: sourceSegmentIndices.length > 0 ? sourceSegmentIndices : undefined,
      speaker: this.getDominantSpeaker(words),
    };
  }

  private static buildChunksFromWordSegments(words: AsrWordSegment[], language?: string): AsrTranscriptChunk[] {
    if (!Array.isArray(words) || words.length === 0) return [];

    const noSpaces = this.isLanguageWithoutSpaces(language);
    const maxDurationSec = noSpaces ? 3.8 : 5.6;
    const strongPauseSec = noSpaces ? 0.55 : 0.45;
    const clausePauseSec = noSpaces ? 0.34 : 0.28;
    const maxChars = noSpaces ? 26 : 64;
    const minCharsForBreak = noSpaces ? 9 : 20;
    const maxTokens = noSpaces ? 32 : 14;
    const chunks: AsrTranscriptChunk[] = [];
    let bufferStart = 0;

    const finalize = (startIndex: number, endIndex: number) => {
      const chunk = this.makeChunkFromWords(words.slice(startIndex, endIndex + 1), language);
      if (chunk) chunks.push(chunk);
    };

    for (let index = 1; index < words.length; index += 1) {
      const current = words[index];
      const previous = words[index - 1];
      const candidateWords = words.slice(bufferStart, index);
      const candidateText = this.joinWordSegments(candidateWords, language);
      const candidateChars = candidateText.length;
      const candidateDuration = this.getWordEnd(previous) - this.toFiniteNumber(words[bufferStart]?.start_ts, 0);
      const nextDuration = this.getWordEnd(current) - this.toFiniteNumber(words[bufferStart]?.start_ts, 0);
      const gap = this.toFiniteNumber(current?.start_ts, 0) - this.getWordEnd(previous);
      const endedSentence = /[.!?…。！？]$/.test(candidateText);
      const endedClause = /[,;:、，；：]$/.test(candidateText);
      const endedBracketed = noSpaces && this.endsWithClosingBracketToken(candidateText);
      const crossedSegmentBoundary =
        this.toFiniteNumber(current?.source_segment_index, Number.NaN) !==
        this.toFiniteNumber(previous?.source_segment_index, Number.NaN);
      const speakerChanged = Boolean(previous?.speaker && current?.speaker && previous.speaker !== current.speaker);

      const shouldBreak =
        speakerChanged ||
        gap >= strongPauseSec ||
        (endedBracketed && gap >= 0.04) ||
        (endedSentence && candidateChars >= minCharsForBreak) ||
        (endedClause && gap >= clausePauseSec && candidateChars >= minCharsForBreak) ||
        (crossedSegmentBoundary && gap >= clausePauseSec && candidateChars >= minCharsForBreak) ||
        nextDuration > maxDurationSec ||
        candidateChars >= maxChars ||
        (!noSpaces && candidateWords.length >= maxTokens);

      if (shouldBreak) {
        finalize(bufferStart, index - 1);
        bufferStart = index;
      }
    }

    finalize(bufferStart, words.length - 1);
    return chunks;
  }

  private static buildChunksFromSegments(segments: AsrStructuredSegment[]): AsrTranscriptChunk[] {
    return segments
      .map((segment, index) => {
        const text = this.normalizeChunkText(segment?.text || '');
        if (!text) return null;
        return {
          text,
          start_ts: this.toFiniteNumber(segment?.start_ts, 0),
          end_ts: segment?.end_ts == null ? undefined : this.toFiniteNumber(segment.end_ts, 0),
          source_segment_indices: [index],
        } satisfies AsrTranscriptChunk;
      })
      .filter(Boolean) as AsrTranscriptChunk[];
  }

  private static buildTranscriptTextFromChunks(chunks: AsrTranscriptChunk[], language?: string) {
    if (!Array.isArray(chunks) || chunks.length === 0) return '';
    const noSpaces = this.shouldTreatTextAsNoSpaceScript(chunks.map((chunk) => chunk?.text || '').join(''), language);
    return this.normalizeDisplayText(
      chunks
        .map((chunk) => this.normalizeChunkText(chunk?.text || ''))
        .filter(Boolean)
        .join(noSpaces ? '' : ' '),
      language
    );
  }

  private static getNoSpaceChunkOverlapLength(previousText: string, currentText: string, language?: string) {
    const previous = Array.from(this.normalizeDisplayText(previousText || '', language).replace(/\s+/g, ''));
    const current = Array.from(this.normalizeDisplayText(currentText || '', language).replace(/\s+/g, ''));
    const maxLen = Math.min(previous.length, current.length);
    for (let size = maxLen; size >= 2; size -= 1) {
      const previousSuffix = previous.slice(previous.length - size).join('');
      const currentPrefix = current.slice(0, size).join('');
      if (previousSuffix === currentPrefix) {
        return size;
      }
    }
    return 0;
  }

  private static mergeDisplayChunks(chunks: AsrTranscriptChunk[], language?: string) {
    if (!Array.isArray(chunks) || chunks.length <= 1) return chunks;
    const noSpaces = this.shouldTreatTextAsNoSpaceScript(chunks.map((chunk) => chunk?.text || '').join(''), language);

    const merged: AsrTranscriptChunk[] = [];
    const getLastToken = (text: string) => {
      const parts = this.normalizeDisplayText(text, language).match(/[\p{L}\p{N}]+/gu);
      return parts && parts.length > 0 ? parts[parts.length - 1] : '';
    };
    const startsWithLowercase = (text: string) => /^[\s"'“‘(\[]*\p{Ll}/u.test(String(text || ''));
    const startsWithUppercaseWord = (text: string) => /^[\s"'“‘(\[]*\p{Lu}[\p{L}\p{M}]*/u.test(String(text || ''));
    const endsWithStrongPunctuation = (text: string) => /[.!?…。！？]$/.test(String(text || '').trim());
    const endsWithSoftPunctuation = (text: string) => /[,;:]$/.test(String(text || '').trim());

    for (const chunk of chunks) {
      const current = {
        ...chunk,
        text: this.normalizeDisplayText(chunk.text || '', language),
      };
      if (merged.length === 0) {
        merged.push(current);
        continue;
      }

      const previous = merged[merged.length - 1];
      const gap = this.toFiniteNumber(current.start_ts, 0) - this.toFiniteNumber(previous.end_ts, previous.start_ts);
      const speakerChanged = Boolean(previous?.speaker && current?.speaker && previous.speaker !== current.speaker);
      if (speakerChanged) {
        merged.push(current);
        continue;
      }
      if (noSpaces) {
        const previousText = this.normalizeDisplayText(previous.text || '', language).replace(/\s+/g, '');
        const currentText = this.normalizeDisplayText(current.text || '', language).replace(/\s+/g, '');
        const combinedDuration = this.toFiniteNumber(current.end_ts, current.start_ts) - this.toFiniteNumber(previous.start_ts, 0);
        const overlapLen = this.getNoSpaceChunkOverlapLength(previousText, currentText, language);
        const currentChars = Array.from(currentText);

        if (gap <= 0.5 && currentText && previousText.includes(currentText)) {
          merged[merged.length - 1] = {
            ...previous,
            end_ts: Math.max(
              this.toFiniteNumber(previous.end_ts, previous.start_ts),
              this.toFiniteNumber(current.end_ts, current.start_ts)
            ),
            word_end_index: current.word_end_index ?? previous.word_end_index,
            source_segment_indices: Array.from(
              new Set([...(previous.source_segment_indices || []), ...(current.source_segment_indices || [])])
            ),
          };
          continue;
        }

        const minRequiredOverlap = Math.max(2, Math.min(8, Math.floor(currentChars.length * 0.45)));
        const shouldMergeNoSpace =
          gap <= 0.5 &&
          combinedDuration <= 8.5 &&
          overlapLen >= minRequiredOverlap &&
          currentChars.length - overlapLen >= 1;

        if (shouldMergeNoSpace) {
          const tail = currentChars.slice(overlapLen).join('');
          merged[merged.length - 1] = {
            ...previous,
            text: this.normalizeDisplayText(`${previousText}${tail}`, language),
            end_ts: current.end_ts,
            word_end_index: current.word_end_index ?? previous.word_end_index,
            source_segment_indices: Array.from(
              new Set([...(previous.source_segment_indices || []), ...(current.source_segment_indices || [])])
            ),
          };
          continue;
        }

        merged.push(current);
        continue;
      }

      const combinedText = this.normalizeDisplayText(`${previous.text} ${current.text}`, language);
      const combinedChars = combinedText.length;
      const combinedDuration = this.toFiniteNumber(current.end_ts, current.start_ts) - this.toFiniteNumber(previous.start_ts, 0);
      const previousLastToken = getLastToken(previous.text);
      const previousEndsWithUpperToken = /^[\p{Lu}\p{Lt}][\p{L}\p{M}]{1,}$/u.test(previousLastToken);

      const shouldMerge =
        gap <= 0.24 &&
        combinedChars <= 96 &&
        combinedDuration <= 7.2 &&
        !endsWithStrongPunctuation(previous.text) &&
        (
          endsWithSoftPunctuation(previous.text) ||
          startsWithLowercase(current.text) ||
          (previousEndsWithUpperToken && startsWithUppercaseWord(current.text))
        );

      if (!shouldMerge) {
        merged.push(current);
        continue;
      }

      merged[merged.length - 1] = {
        ...previous,
        text: combinedText,
        end_ts: current.end_ts,
        word_end_index: current.word_end_index,
        source_segment_indices: Array.from(
          new Set([...(previous.source_segment_indices || []), ...(current.source_segment_indices || [])])
        ),
      };
    }

    return merged;
  }

  private static makeChunkFromSegments(segments: AsrStructuredSegment[], language?: string): AsrTranscriptChunk | null {
    if (!Array.isArray(segments) || segments.length === 0) return null;
    const noSpaces = this.shouldTreatTextAsNoSpaceScript(segments.map((segment) => this.normalizeChunkText(segment?.text || '')).join(''), language);
    const text = this.normalizeDisplayText(
      segments
        .map((segment) => this.normalizeChunkText(segment?.text || ''))
        .filter(Boolean)
        .join(noSpaces ? '' : ' '),
      language
    );
    if (!text) return null;

    const firstSegment = segments[0];
    const lastSegment = segments[segments.length - 1];
    const firstWord = this.findFirstTimedWord(segments);
    const lastWord = this.findLastTimedWord(segments);
    const segmentStart = this.getSegmentStartForDisplay(firstSegment);
    const segmentEnd = this.getSegmentEndForDisplay(lastSegment);
    const wordStart = this.toFiniteNumber(firstWord?.start_ts ?? segmentStart, segmentStart);
    const wordEnd = this.ensureValidRangeEnd(wordStart, lastWord != null ? this.getWordEnd(lastWord) : segmentEnd);
    const segmentDuration = Math.max(0, segmentEnd - segmentStart);
    const wordDuration = Math.max(0, wordEnd - wordStart);
    const collapsedWordTiming =
      text.length >= 12 &&
      segmentDuration >= 0.8 &&
      wordDuration < Math.max(0.12, segmentDuration * 0.22);
    const sourceSegmentIndices = Array.from(
      new Set(
        segments
          .flatMap((segment) =>
            (segment.words || [])
              .map((word) => this.toFiniteNumber(word?.source_segment_index, Number.NaN))
              .filter((value) => Number.isFinite(value))
          )
      )
    );

    return {
      text,
      start_ts: collapsedWordTiming ? segmentStart : wordStart,
      end_ts: collapsedWordTiming ? segmentEnd : wordEnd,
      word_start_index: firstWord?.token_index,
      word_end_index: lastWord?.token_index,
      source_segment_indices: sourceSegmentIndices.length > 0 ? sourceSegmentIndices : undefined,
      speaker: this.getDominantSpeaker(segments) || this.getDominantSpeaker(segments.flatMap((segment) => segment.words || [])),
    };
  }

  private static findFirstTimedWord(segments: AsrStructuredSegment[]) {
    for (const segment of segments) {
      for (const word of segment.words || []) {
        const start = this.toFiniteNumber(word?.start_ts, Number.NaN);
        if (Number.isFinite(start)) {
          return word;
        }
      }
    }
    return null;
  }

  private static findLastTimedWord(segments: AsrStructuredSegment[]) {
    for (let segmentIndex = segments.length - 1; segmentIndex >= 0; segmentIndex -= 1) {
      const words = segments[segmentIndex]?.words || [];
      for (let wordIndex = words.length - 1; wordIndex >= 0; wordIndex -= 1) {
        const word = words[wordIndex];
        const start = this.toFiniteNumber(word?.start_ts, Number.NaN);
        const end = this.toFiniteNumber(word?.end_ts, Number.NaN);
        if (Number.isFinite(end) || Number.isFinite(start)) {
          return word;
        }
      }
    }
    return null;
  }

  private static getSegmentStartForDisplay(segment: AsrStructuredSegment) {
    const firstWord = (segment.words || []).find((word) => Number.isFinite(this.toFiniteNumber(word?.start_ts, Number.NaN)));
    return this.toFiniteNumber(firstWord?.start_ts ?? segment.start_ts, 0);
  }

  private static getSegmentEndForDisplay(segment: AsrStructuredSegment) {
    const words = segment.words || [];
    for (let index = words.length - 1; index >= 0; index -= 1) {
      const candidate = this.getWordEnd(words[index]);
      if (Number.isFinite(candidate) && candidate > 0) {
        return candidate;
      }
    }
    return this.ensureValidRangeEnd(
      this.toFiniteNumber(segment.start_ts, 0),
      segment.end_ts == null ? undefined : this.toFiniteNumber(segment.end_ts, 0)
    );
  }

  private static buildDisplayChunksFromSegments(segments: AsrStructuredSegment[], language?: string): AsrTranscriptChunk[] {
    if (!Array.isArray(segments) || segments.length === 0) return [];

    const noSpaces = this.shouldTreatTextAsNoSpaceScript(segments.map((segment) => this.normalizeChunkText(segment?.text || '')).join(''), language);
    const maxDurationSec = noSpaces ? 3.4 : 5.6;
    const strongPauseSec = noSpaces ? 0.55 : 0.45;
    const clausePauseSec = noSpaces ? 0.34 : 0.28;
    const maxChars = noSpaces ? 22 : 68;
    const minCharsForBreak = noSpaces ? 8 : 22;
    const maxSegments = noSpaces ? 2 : 3;
    const chunks: AsrTranscriptChunk[] = [];
    let bufferStart = 0;

    const finalize = (startIndex: number, endIndex: number) => {
      const chunk = this.makeChunkFromSegments(segments.slice(startIndex, endIndex + 1), language);
      if (chunk) chunks.push(chunk);
    };

    for (let index = 1; index < segments.length; index += 1) {
      const current = segments[index];
      const previous = segments[index - 1];
      const candidateSegments = segments.slice(bufferStart, index);
      const candidateText = candidateSegments
        .map((segment) => this.normalizeChunkText(segment?.text || ''))
        .filter(Boolean)
        .join(noSpaces ? '' : ' ');
      const normalizedCandidateText = this.normalizeDisplayText(candidateText, language);
      const candidateChars = normalizedCandidateText.length;
      const previousText = this.normalizeChunkText(previous?.text || '');
      const currentText = this.normalizeChunkText(current?.text || '');
      const previousChars = previousText.length;
      const currentChars = currentText.length;
      const bufferStartTs = this.getSegmentStartForDisplay(segments[bufferStart]);
      const previousEnd = this.getSegmentEndForDisplay(previous);
      const currentStart = this.getSegmentStartForDisplay(current);
      const currentEnd = this.getSegmentEndForDisplay(current);
      const gap = currentStart - previousEnd;
      const nextDuration = currentEnd - bufferStartTs;
      const endedSentence = /[.!?…。！？]$/.test(normalizedCandidateText);
      const endedClause = /[,;:、，；：]$/.test(normalizedCandidateText);
      const endedBracketed = noSpaces && this.endsWithClosingBracketToken(normalizedCandidateText);
      const meaningfulCjkBoundary = noSpaces && previousChars >= 4 && currentChars >= 4;
      const speakerChanged = Boolean(previous?.speaker && current?.speaker && previous.speaker !== current.speaker);

      const shouldBreak =
        speakerChanged ||
        gap >= strongPauseSec ||
        endedBracketed ||
        (endedSentence && (candidateChars >= minCharsForBreak || noSpaces)) ||
        (endedClause && gap >= clausePauseSec && candidateChars >= minCharsForBreak) ||
        meaningfulCjkBoundary ||
        nextDuration > maxDurationSec ||
        candidateChars >= maxChars ||
        (candidateSegments.length >= maxSegments && candidateChars >= minCharsForBreak);

      if (shouldBreak) {
        finalize(bufferStart, index - 1);
        bufferStart = index;
      }
    }

    finalize(bufferStart, segments.length - 1);
    return this.mergeDisplayChunks(chunks, language);
  }

  private static buildFallbackSegmentsFromRootText(data: any): AsrStructuredTranscript {
    const rootText =
      this.firstString([
        data?.text,
        data?.transcript,
        data?.result?.text,
        data?.result?.transcript,
        data?.data?.text,
        data?.results?.channels?.[0]?.alternatives?.[0]?.transcript,
      ]) || '';

    const normalizedRoot = this.normalizeChunkText(rootText);
    if (!normalizedRoot) {
      return { text: '', chunks: [], segments: [], word_segments: [] };
    }

    const lines = normalizedRoot
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean);

    if (lines.length <= 1) {
      const segment = { start_ts: 0, text: normalizedRoot } satisfies AsrStructuredSegment;
      return {
        text: normalizedRoot,
        chunks: [{ start_ts: 0, text: normalizedRoot }],
        segments: [segment],
        word_segments: [],
      };
    }

    const segments = lines.map((line, index) => ({
      start_ts: index,
      end_ts: index + 1,
      text: line,
    })) satisfies AsrStructuredSegment[];

    return {
      text: normalizedRoot,
      chunks: this.buildChunksFromSegments(segments),
      segments,
      word_segments: [],
    };
  }

  private static extractStructuredTranscript(
    data: any,
    language?: string,
    options: {
      enableSparseNoSpaceNativeRecovery?: boolean;
      preferAlignmentFirstNoSpace?: boolean;
    } = {}
  ): AsrStructuredTranscript {
    const segmentLike = Array.isArray(data?.segments)
      ? data.segments
      : Array.isArray(data?.chunks)
        ? data.chunks
        : null;

    if (segmentLike && segmentLike.length > 0) {
      const cjkWordDiagnostics = {
        nativeWordCount: 0,
        syntheticWordCount: 0,
        syntheticWordAppliedSegments: 0,
        rawWordCount: 0,
        mergedWordCount: 0,
        droppedWordCount: 0,
        replacementCharCount: 0,
        rawSingleCharCount: 0,
        mergedSingleCharCount: 0,
        punctuationOnlyCount: 0,
        lexicalWordCount: 0,
        lexicalSingleCharCount: 0,
        lexicalMismatchCount: 0,
        lexicalProjectionAppliedSegments: 0,
        splitSegmentCount: 0,
        sparseNativeCoverageRecoveredSegments: 0,
        alignmentFirstMode: false,
        heuristicBypassedSegments: 0,
        usedIntlSegmenter: false,
        segmenterLocale: this.getNoSpaceSegmenterLocale(language),
        languagePackKey: null as string | null,
        languageVariant: null as string | null,
        variantConfig: null as Record<string, any> | null,
      };
      const noSpaceModule = LanguageAlignmentRegistry.getNoSpaceModule(language, segmentLike.map((s: any) => this.firstString([s?.text, s?.transcript, s?.content, s?.sentence]) || '').join(' '));
      if (noSpaceModule) {
        cjkWordDiagnostics.languagePackKey = String(noSpaceModule.config?.key || '').trim() || null;
        cjkWordDiagnostics.languageVariant = noSpaceModule.resolveVariant?.(
          language,
          segmentLike.map((s: any) => this.firstString([s?.text, s?.transcript, s?.content, s?.sentence]) || '').join(' ')
        ) || null;
        cjkWordDiagnostics.variantConfig = noSpaceModule.getVariantDebug?.(
          language,
          segmentLike.map((s: any) => this.firstString([s?.text, s?.transcript, s?.content, s?.sentence]) || '').join(' ')
        ) || null;
      }
      let cjkMergeApplied = false;
      const mappedSegments = segmentLike
        .flatMap((s: any, segmentIndex: number) => {
          const text = this.firstString([s?.text, s?.transcript, s?.content, s?.sentence]) || '';
          const normalizedText = this.normalizeChunkText(text);
          const start = this.toFiniteNumber(s?.start_ts ?? s?.start ?? s?.timestamp?.start, 0);
          const endRaw = s?.end_ts ?? s?.end ?? s?.timestamp?.end;
          const endNum = Number(endRaw);
          const end = Number.isFinite(endNum) ? endNum : undefined;
          const rawWords = Array.isArray(s?.words)
            ? s.words
                .map((word: any, wordIndex: number) => {
                  const rawText = this.firstString([word?.word, word?.text, word?.token, word?.content]) || '';
                  const normalizedWord = this.normalizeWordText(rawText);
                  if (!normalizedWord) return null;
                  const wordStart = this.toFiniteNumber(word?.start ?? word?.start_ts ?? word?.timestamp?.start, Number.NaN);
                  if (!Number.isFinite(wordStart)) return null;
                  const wordEndRaw = word?.end ?? word?.end_ts ?? word?.timestamp?.end;
                  const wordEndNum = Number(wordEndRaw);
                  return {
                    text: normalizedWord,
                    start_ts: wordStart,
                    end_ts: Number.isFinite(wordEndNum) ? wordEndNum : undefined,
                    speaker: this.firstString([word?.speaker, word?.speaker_id]) ?? undefined,
                    probability: Number.isFinite(Number(word?.probability)) ? Number(word?.probability) : undefined,
                    source_segment_index: segmentIndex,
                    token_index: 0,
                  } satisfies AsrWordSegment;
                })
                .filter(Boolean) as AsrWordSegment[]
            : [];
          cjkWordDiagnostics.nativeWordCount += rawWords.length;
          const alignmentFirstNoSpace =
            Boolean(options.preferAlignmentFirstNoSpace) &&
            this.shouldTreatTextAsNoSpaceScript(normalizedText || text, language);
          if (alignmentFirstNoSpace) {
            cjkWordDiagnostics.alignmentFirstMode = true;
          }
          const sparseNativeRecovered =
            !alignmentFirstNoSpace &&
            options.enableSparseNoSpaceNativeRecovery &&
            rawWords.length > 0 &&
            this.shouldRecoverSparseNoSpaceNativeWords(rawWords, normalizedText, language);
          const synthesizedWords =
            ((rawWords.length === 0) || sparseNativeRecovered) && normalizedText
              ? this.buildSyntheticWordSegmentsFromText(normalizedText, start, end, language, segmentIndex)
              : [];
          if (synthesizedWords.length > 0) {
            cjkWordDiagnostics.syntheticWordCount += synthesizedWords.length;
            cjkWordDiagnostics.syntheticWordAppliedSegments += 1;
            if (sparseNativeRecovered) {
              cjkWordDiagnostics.sparseNativeCoverageRecoveredSegments += 1;
            }
          }
          const baseWords = sparseNativeRecovered ? synthesizedWords : rawWords.length > 0 ? rawWords : synthesizedWords;
          let words = baseWords;
          let appliedCjkMerging = false;
          let lexicalProjectionApplied = false;
          if (!alignmentFirstNoSpace) {
            const { words: mergedWords, diagnostics, appliedCjkMerging: mergedApplied } = this.mergeCjkWordSegments(baseWords, language);
            appliedCjkMerging = mergedApplied;
            if (appliedCjkMerging) {
              cjkMergeApplied = true;
              cjkWordDiagnostics.rawWordCount += diagnostics.rawWordCount;
              cjkWordDiagnostics.mergedWordCount += diagnostics.mergedWordCount;
              cjkWordDiagnostics.droppedWordCount += diagnostics.droppedWordCount;
              cjkWordDiagnostics.replacementCharCount += diagnostics.replacementCharCount;
              cjkWordDiagnostics.rawSingleCharCount += diagnostics.rawSingleCharCount;
              cjkWordDiagnostics.mergedSingleCharCount += diagnostics.mergedSingleCharCount;
              cjkWordDiagnostics.punctuationOnlyCount += diagnostics.punctuationOnlyCount;
            }

            const lexicalProjection = this.projectNoSpaceLexicalWordSegments(
              mergedWords,
              normalizedText || this.joinWordSegments(mergedWords, language),
              language
            );
            lexicalProjectionApplied = lexicalProjection.appliedLexicalProjection;
            if (lexicalProjectionApplied) {
              cjkMergeApplied = true;
            }
            cjkWordDiagnostics.lexicalWordCount += lexicalProjection.diagnostics.lexicalWordCount;
            cjkWordDiagnostics.lexicalSingleCharCount += lexicalProjection.diagnostics.lexicalSingleCharCount;
            cjkWordDiagnostics.lexicalMismatchCount += lexicalProjection.diagnostics.lexicalMismatchCount;
            cjkWordDiagnostics.lexicalProjectionAppliedSegments += lexicalProjection.diagnostics.lexicalProjectionAppliedSegments;
            cjkWordDiagnostics.usedIntlSegmenter = cjkWordDiagnostics.usedIntlSegmenter || lexicalProjection.diagnostics.usedIntlSegmenter;
            cjkWordDiagnostics.segmenterLocale = lexicalProjection.diagnostics.locale || cjkWordDiagnostics.segmenterLocale;
            words = lexicalProjection.words;
          } else if (rawWords.length > 0) {
            cjkWordDiagnostics.heuristicBypassedSegments += 1;
          }
          const projectedText = words.length > 0 ? this.joinWordSegments(words, language) : '';
          const normalizedProjected = this.normalizeNoSpaceAlignmentText(projectedText);
          const normalizedSegment = this.normalizeNoSpaceAlignmentText(normalizedText);
          const preferProjectedText =
            words.length > 0 &&
            (
              sparseNativeRecovered ||
              lexicalProjectionApplied ||
              (appliedCjkMerging && normalizedProjected.length > 0 && normalizedProjected !== normalizedSegment)
            );
          const resolvedText = this.normalizeDisplayText(
            alignmentFirstNoSpace
              ? (normalizedText || projectedText)
              : preferProjectedText
              ? (projectedText || normalizedText)
              : (normalizedText || projectedText),
            language
          );

          const baseSegment = {
            start_ts: start,
            end_ts: end,
            text: resolvedText,
            speaker: this.firstString([s?.speaker, s?.speaker_id, words.find((word) => word.speaker)?.speaker]) ?? undefined,
            words: words.length > 0 ? words : undefined,
          } satisfies AsrStructuredSegment;
          const splitSegments = alignmentFirstNoSpace ? [baseSegment] : this.splitNoSpaceSegmentByWords(baseSegment, language);
          if (splitSegments.length > 1) {
            cjkMergeApplied = true;
            cjkWordDiagnostics.splitSegmentCount += splitSegments.length - 1;
          }
          return splitSegments;
        })
        .filter((segment: any) => segment.text.length > 0) as AsrStructuredSegment[];

      if (mappedSegments.length > 0) {
        let tokenIndex = 0;
        const indexedSegments = mappedSegments.map((segment, segmentIndex) => {
          const words = (segment.words || []).map((word) => ({
            ...word,
            source_segment_index: segmentIndex,
            token_index: tokenIndex++,
          }));
          return words.length > 0 ? { ...segment, words } : segment;
        });
        const wordSegments = indexedSegments.flatMap((segment) => segment.words || []);
        const chunks = this.buildDisplayChunksFromSegments(indexedSegments, language);
        const transcriptText = this.buildTranscriptTextFromChunks(chunks, language);
        return {
          text: transcriptText,
          chunks: chunks.length > 0 ? chunks : this.buildChunksFromSegments(indexedSegments),
          segments: indexedSegments,
          word_segments: wordSegments,
          debug: (cjkMergeApplied || cjkWordDiagnostics.alignmentFirstMode)
            ? {
                cjkWordDiagnostics: {
                  ...cjkWordDiagnostics,
                  mergeApplied: cjkMergeApplied,
                  chunkSource:
                    cjkWordDiagnostics.alignmentFirstMode
                      ? 'alignment_first_segments'
                      : wordSegments.length > 0
                      ? cjkWordDiagnostics.lexicalProjectionAppliedSegments > 0
                        ? 'lexical_cjk_segments'
                        : 'merged_cjk_segments'
                      : 'segment_only',
                },
              }
            : undefined,
        };
      }
    }

    return this.buildFallbackSegmentsFromRootText(data);
  }

  private static offsetStructuredTranscript(transcript: AsrStructuredTranscript, offsetSec: number): AsrStructuredTranscript {
    const shift = (value: number | undefined) => {
      if (!Number.isFinite(Number(value))) return undefined;
      return this.toFiniteNumber(value, 0) + offsetSec;
    };

    return {
      text: transcript.text,
      chunks: (transcript.chunks || []).map((chunk) => ({
        ...chunk,
        start_ts: shift(chunk.start_ts) ?? 0,
        end_ts: shift(chunk.end_ts),
      })),
      segments: (transcript.segments || []).map((segment) => ({
        ...segment,
        start_ts: shift(segment.start_ts) ?? 0,
        end_ts: shift(segment.end_ts),
        words: (segment.words || []).map((word) => ({
          ...word,
          start_ts: shift(word.start_ts) ?? 0,
          end_ts: shift(word.end_ts),
        })),
      })),
      word_segments: (transcript.word_segments || []).map((word) => ({
        ...word,
        start_ts: shift(word.start_ts) ?? 0,
        end_ts: shift(word.end_ts),
      })),
    };
  }

  private static applySpeakerAssignments(result: AsrStructuredTranscript) {
    if (!Array.isArray(result?.chunks) || result.chunks.length === 0) return;

    if (Array.isArray(result.word_segments) && result.word_segments.length > 0) {
      for (const chunk of result.chunks) {
        const speaker = typeof chunk?.speaker === 'string' && chunk.speaker.trim() ? chunk.speaker.trim() : '';
        if (!speaker) continue;
        const startIndex = this.toFiniteNumber(chunk.word_start_index, Number.NaN);
        const endIndex = this.toFiniteNumber(chunk.word_end_index, Number.NaN);
        if (Number.isFinite(startIndex) && Number.isFinite(endIndex) && endIndex >= startIndex) {
          for (let index = startIndex; index <= endIndex; index += 1) {
            if (result.word_segments[index]) {
              result.word_segments[index].speaker = speaker;
            }
          }
        }
      }

      for (const word of result.word_segments) {
        if (word.speaker) continue;
        const wordStart = this.toFiniteNumber(word.start_ts, 0);
        const wordEnd = this.getWordEnd(word);
        const ownerChunk = result.chunks.find((chunk) => {
          const chunkStart = this.toFiniteNumber(chunk.start_ts, 0);
          const chunkEnd = this.toFiniteNumber(chunk.end_ts, chunkStart);
          return chunkStart <= wordStart + 0.02 && chunkEnd + 0.02 >= wordEnd;
        });
        if (ownerChunk?.speaker) {
          word.speaker = ownerChunk.speaker;
        }
      }
    }

    if (Array.isArray(result.segments) && result.segments.length > 0) {
      result.segments = result.segments.map((segment, index) => {
        const segmentWords = (result.word_segments || []).filter(
          (word) => this.toFiniteNumber(word?.source_segment_index, Number.NaN) === index
        );
        const countedSpeakers = new Map<string, number>();
        for (const word of segmentWords) {
          const speaker = typeof word?.speaker === 'string' && word.speaker.trim() ? word.speaker.trim() : '';
          if (!speaker) continue;
          countedSpeakers.set(speaker, (countedSpeakers.get(speaker) || 0) + 1);
        }
        const dominantSpeaker = Array.from(countedSpeakers.entries()).sort((a, b) => b[1] - a[1])[0]?.[0];
        if (dominantSpeaker) {
          return { ...segment, speaker: dominantSpeaker };
        }

        const segmentStart = this.toFiniteNumber(segment.start_ts, 0);
        const segmentEnd = this.toFiniteNumber(segment.end_ts, segmentStart);
        const overlappingChunk = result.chunks.find((chunk) => {
          const chunkStart = this.toFiniteNumber(chunk.start_ts, 0);
          const chunkEnd = this.toFiniteNumber(chunk.end_ts, chunkStart);
          return chunkEnd >= segmentStart && chunkStart <= segmentEnd;
        });
        return overlappingChunk?.speaker ? { ...segment, speaker: overlappingChunk.speaker } : segment;
      });
    }
  }

  private static getDominantSpeaker(items: Array<{ speaker?: string }>) {
    const countedSpeakers = new Map<string, number>();
    for (const item of items) {
      const speaker = typeof item?.speaker === 'string' && item.speaker.trim() ? item.speaker.trim() : '';
      if (!speaker) continue;
      countedSpeakers.set(speaker, (countedSpeakers.get(speaker) || 0) + 1);
    }
    return Array.from(countedSpeakers.entries()).sort((a, b) => b[1] - a[1])[0]?.[0];
  }

  private static collectTranscriptSpeakers(result: AsrStructuredTranscript) {
    const speakers = new Set<string>();
    for (const item of [
      ...(result.chunks || []),
      ...(result.segments || []),
      ...(result.word_segments || []),
    ]) {
      const speaker = typeof item?.speaker === 'string' && item.speaker.trim() ? item.speaker.trim() : '';
      if (speaker) speakers.add(speaker);
    }
    return speakers;
  }

  private static hasProviderSpeakerTags(result: AsrStructuredTranscript) {
    return this.collectTranscriptSpeakers(result).size > 0;
  }

  private static applyProviderNativeSpeakerAssignments(result: AsrStructuredTranscript) {
    if (!Array.isArray(result?.chunks) || result.chunks.length === 0) return;
    const words = Array.isArray(result.word_segments) ? result.word_segments : [];
    const segments = Array.isArray(result.segments) ? result.segments : [];

    result.chunks = result.chunks.map((chunk) => {
      const existingSpeaker = typeof chunk?.speaker === 'string' && chunk.speaker.trim() ? chunk.speaker.trim() : '';
      if (existingSpeaker) return chunk;

      const startIndex = this.toFiniteNumber(chunk.word_start_index, Number.NaN);
      const endIndex = this.toFiniteNumber(chunk.word_end_index, Number.NaN);
      const wordsByIndex =
        Number.isFinite(startIndex) && Number.isFinite(endIndex) && endIndex >= startIndex
          ? words.slice(Math.max(0, Math.floor(startIndex)), Math.floor(endIndex) + 1)
          : [];
      const chunkStart = this.toFiniteNumber(chunk.start_ts, 0);
      const chunkEnd = this.toFiniteNumber(chunk.end_ts, chunkStart);
      const wordsByOverlap = wordsByIndex.length > 0
        ? wordsByIndex
        : words.filter((word) => {
            const wordStart = this.toFiniteNumber(word.start_ts, 0);
            const wordEnd = this.getWordEnd(word);
            return wordEnd >= chunkStart - 0.02 && wordStart <= chunkEnd + 0.02;
          });
      const segmentSpeakers = segments.filter((segment) => {
        const segmentStart = this.toFiniteNumber(segment.start_ts, 0);
        const segmentEnd = this.toFiniteNumber(segment.end_ts, segmentStart);
        return segmentEnd >= chunkStart - 0.02 && segmentStart <= chunkEnd + 0.02;
      });
      const speaker = this.getDominantSpeaker(wordsByOverlap) || this.getDominantSpeaker(segmentSpeakers);
      return speaker ? { ...chunk, speaker } : chunk;
    });

    this.applySpeakerAssignments(result);
  }

  private static buildProviderNativeDiarizationDiagnostics(
    result: AsrStructuredTranscript,
    speechSegments: Array<{ start: number; end: number }>,
    vadWindows: Array<{ start: number; end: number }>,
    providerName = 'provider'
  ) {
    const speakers = this.collectTranscriptSpeakers(result);
    return {
      provider: 'provider_native',
      selectedSource: 'provider_native',
      speechSegmentCount: speechSegments.length,
      vadWindowCount: vadWindows.length,
      providerNative: true,
      providerNativeProvider: providerName,
      selectedPass: {
        source: 'provider_native',
        regionCount: Array.isArray(result.chunks) ? result.chunks.length : 0,
        uniqueSpeakerCount: speakers.size,
        threshold: 1,
      },
    };
  }

  private static disableWordAlignment(transcript: AsrStructuredTranscript, language?: string): AsrStructuredTranscript {
    const coarseSegments = (transcript.segments || []).map((segment) => ({
      ...segment,
      words: undefined,
      speaker: segment.speaker,
    }));
    return {
      ...transcript,
      segments: coarseSegments,
      word_segments: [],
      chunks: this.buildDisplayChunksFromSegments(coarseSegments, language),
    };
  }

  static unload() {
    this.pipeline = null;
  }
}
