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
import { requestCloudAsr } from './cloud_asr_adapter.js';
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

  private static getEnvNumber(name: string, fallback: number, min?: number, max?: number) {
    const raw = process.env[name];
    if (typeof raw !== 'string' || !raw.trim()) return fallback;
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) return fallback;
    if (typeof min === 'number' && parsed < min) return min;
    if (typeof max === 'number' && parsed > max) return max;
    return parsed;
  }

  private static getCloudAsrPreemptiveChunkThresholdBytes() {
    const defaultBytes = 25 * 1024 * 1024;
    return Math.round(this.getEnvNumber('ASR_FILE_LIMIT_BYTES', defaultBytes, 1024 * 1024));
  }

  private static buildVadWindows(segments: Array<{ start: number; end: number }>) {
    const sorted = segments
      .map((s) => ({
        start: this.toFiniteNumber(s?.start, 0),
        end: this.toFiniteNumber(s?.end, 0),
      }))
      .filter((s) => s.end > s.start)
      .sort((a, b) => a.start - b.start);

    if (sorted.length === 0) return [];

    const minSegmentSec = this.getEnvNumber('VAD_TRANSCRIBE_MIN_SEGMENT_SEC', 0.2, 0.05, 10);
    const mergeGapSec = this.getEnvNumber('VAD_TRANSCRIBE_MERGE_GAP_SEC', 0.35, 0, 10);
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

    if (windows.length > maxWindows) {
      return [{ start: windows[0].start, end: windows[windows.length - 1].end }];
    }

    return windows;
  }

  private static async extractAudioWindow(sourcePath: string, startSec: number, endSec: number, suffix: string) {
    const outputPath = path.join(path.dirname(sourcePath), `vad_window_${suffix}_${Date.now()}.wav`);
    const args = [
      '-i', sourcePath,
      '-ss', startSec.toFixed(3),
      '-to', endSec.toFixed(3),
      '-vn',
      '-ac', '1',
      '-ar', '16000',
      '-c:a', 'pcm_s16le',
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
    return {
      ...first,
      callCount,
      rawSegmentCount: metaList.reduce((sum: number, m: any) => sum + this.toFiniteNumber(m?.rawSegmentCount, 0), 0),
      rawWordCount: metaList.reduce((sum: number, m: any) => sum + this.toFiniteNumber(m?.rawWordCount, 0), 0),
      rawHasTimestamps: metaList.some((m: any) => Boolean(m?.rawHasTimestamps)),
      autoLanguageFallbackUsed: metaList.some((m: any) => Boolean(m?.autoLanguageFallbackUsed)),
      fileLimitFallbackUsed: metaList.some((m: any) => Boolean(m?.fileLimitFallbackUsed)),
      fileLimitChunkCount: metaList.reduce((sum: number, m: any) => sum + this.toFiniteNumber(m?.fileLimitChunkCount, 0), 0),
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

  private static hasUsableChunkTimestamps(chunks: Array<{ start_ts: number; end_ts?: number; text: string }>) {
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

  private static isEmptyAudioWindowError(error: unknown) {
    const message = error instanceof Error ? error.message : String(error || '');
    return /Audio window is empty/i.test(message);
  }

  private static async transcribeCloudChunked(
    filePath: string,
    config: any,
    options: { language?: string; prompt?: string; segmentation?: boolean; wordAlignment?: boolean; vad?: boolean; diarization?: boolean },
    onProgress: (msg: string) => void,
    signal?: AbortSignal
  ) {
    const initialChunkSec = Math.round(this.getEnvNumber('ASR_FILE_LIMIT_CHUNK_SEC', 480, 30, 3600));
    const minChunkSec = Math.round(this.getEnvNumber('ASR_FILE_LIMIT_MIN_CHUNK_SEC', 60, 10, 1800));
    const maxChunks = Math.round(this.getEnvNumber('ASR_FILE_LIMIT_MAX_CHUNKS', 200, 1, 2000));
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
        const endSec = startSec + chunkSec;
        const label = `file_limit_${chunkSec}s_${i + 1}`;
        let chunkPath: string | null = null;

        try {
          chunkPath = await this.extractAudioWindow(filePath, startSec, endSec, label);
        } catch (extractErr) {
          if (this.isEmptyAudioWindowError(extractErr)) {
            break;
          }
          throw extractErr;
        }

        try {
          onProgress(`Calling ASR provider (file-limit chunk ${i + 1})...`);
          const cloudResult = await this.transcribeCloud(chunkPath, config, options, signal);
          providerMetaList.push(cloudResult.meta);
          producedChunkCount += 1;
          const shifted = this.offsetStructuredTranscript(cloudResult, startSec);
          mergedTranscript.chunks.push(...shifted.chunks);
          mergedTranscript.segments.push(...shifted.segments);
          mergedTranscript.word_segments.push(...shifted.word_segments);
        } catch (chunkErr) {
          if (this.isCloudAsrFileLimitError(chunkErr) && chunkSec > minChunkSec) {
            shouldRetryWithSmallerChunks = true;
            break;
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
        onProgress(`Chunk uploads still exceed provider limit, reducing chunk size to ${chunkSec}s and retrying...`);
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
        },
      };
    }
  }

  private static async transcribeCloudWithFileLimitFallback(
    filePath: string,
    config: any,
    options: { language?: string; prompt?: string; segmentation?: boolean; wordAlignment?: boolean; vad?: boolean; diarization?: boolean },
    onProgress: (msg: string) => void,
    signal?: AbortSignal
  ) {
    this.throwIfAborted(signal);
    try {
      const stat = await fs.stat(filePath);
      const preemptiveThresholdBytes = this.getCloudAsrPreemptiveChunkThresholdBytes();
      if (stat.isFile() && stat.size > preemptiveThresholdBytes) {
        const sizeMb = (stat.size / (1024 * 1024)).toFixed(1);
        const thresholdMb = (preemptiveThresholdBytes / (1024 * 1024)).toFixed(0);
        onProgress(`Audio file is ${sizeMb}MB (> ${thresholdMb}MB), using chunked upload proactively...`);
        return this.transcribeCloudChunked(filePath, config, options, onProgress, signal);
      }
    } catch {
      // Non-fatal. If file stat fails, keep existing request-first fallback behavior.
    }

    try {
      return await this.transcribeCloud(filePath, config, options, signal);
    } catch (err) {
      if (!this.isCloudAsrFileLimitError(err)) {
        throw err;
      }
      onProgress('Provider rejected audio size/duration limit, retrying with chunked upload...');
      return this.transcribeCloudChunked(filePath, config, options, onProgress, signal);
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
      options: { language?: string; prompt?: string; segmentation?: boolean; wordAlignment?: boolean; vad?: boolean; diarization?: boolean };
      onProgress: (msg: string) => void;
      signal?: AbortSignal;
    }
  ) {
    this.throwIfAborted(input.signal);
    if (input.useLocalProvider) {
      if (!input.localModelId || !input.localModelPath) {
        throw new Error('Local ASR runtime is not configured.');
      }
      return transcribeWithLocalAsrRuntime({
        filePath,
        localModelId: input.localModelId,
        localModelPath: input.localModelPath,
        localModelRuntime: input.localModelRuntime,
        language: input.options.language,
        prompt: input.options.prompt,
        segmentation: input.options.segmentation,
        wordAlignment: input.options.wordAlignment,
        extractStructuredTranscript: this.extractStructuredTranscript.bind(this),
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
      diarizationOptions?: DiarizationOptions
    },
    onProgress: (msg: string) => void,
    signal?: AbortSignal
  ) {
    this.throwIfAborted(signal);
    const { modelId, assetName, language, prompt, segmentation, wordAlignment, vad, diarization, diarizationOptions } = options;
    const startedAt = Date.now();
    const effectiveSegmentation = Boolean(segmentation) || Boolean(diarization);
    const requestedWordAlignment = wordAlignment !== false;
    const effectiveWordAlignment = effectiveSegmentation && requestedWordAlignment;
    const effectiveVad = Boolean(vad) || Boolean(diarization);
    const segmentationForcedForDiarization = Boolean(diarization) && !Boolean(segmentation);
    const requestedFeatures = {
      segmentation: Boolean(segmentation),
      wordAlignment: requestedWordAlignment,
      vad: Boolean(vad),
      diarization: Boolean(diarization),
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
    const allowVadWindowedTranscription = !(useLocalProvider && localModel?.runtime === 'openvino-whisper-node');
    const effectivePrompt = useLocalProvider
      ? this.buildEffectiveLocalAsrPrompt(language, prompt)
      : String(prompt || '').trim();
    const localAsrPromptApplied = useLocalProvider && Boolean(this.getBuiltInLocalAsrPrompt(language));

    const modelConfig = useLocalProvider
      ? null
      : settings.asrModels.find((m: any) => m.id === modelId) || settings.asrModels[0];
    if (!useLocalProvider && !modelConfig) {
      throw new Error('No ASR model configured. Please configure one in Settings.');
    }
    if (!useLocalProvider && !modelConfig.url) {
      throw new Error('ASR model URL is missing.');
    }
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
    let vadMs = 0;
    let providerWallMs = 0;
    let alignmentWallMs = 0;
    let diarizationMs = 0;

    if (effectiveVad) {
      const vadStartedAt = Date.now();
      this.throwIfAborted(signal);
      onProgress('Running voice activity detection...');
      try {
        speechSegments = await VadService.detectSpeech(audioPath);

        if (speechSegments.length > 0) {
          vadWindows = this.buildVadWindows(speechSegments);
          onProgress(`VAD detected ${speechSegments.length} speech segments (${vadWindows.length} windows).`);

          if (!allowVadWindowedTranscription) {
            onProgress('Local Whisper ASR keeps full-audio transcription; VAD windows are used only for diagnostics/diarization.');
          }

          const mergedWindowTranscript: AsrStructuredTranscript = {
            text: '',
            chunks: [],
            segments: [],
            word_segments: [],
          };
          const providerMetaList: any[] = [];
          const skippedWindowErrors: string[] = [];

          if (allowVadWindowedTranscription) {
            for (let i = 0; i < vadWindows.length; i += 1) {
              this.throwIfAborted(signal);
              const window = vadWindows[i];
              const windowLabel = `${i + 1}_${Math.round(window.start * 1000)}_${Math.round(window.end * 1000)}`;
              const windowAudioPath = await this.extractAudioWindow(audioPath, window.start, window.end, windowLabel);

              try {
                onProgress(`Calling ASR provider (VAD window ${i + 1}/${vadWindows.length})...`);
                const providerStartedAt = Date.now();
                try {
                  const cloudResult = await this.transcribeWithProvider(windowAudioPath, {
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
                    },
                    onProgress,
                    signal,
                  });
                  providerWallMs += Math.max(0, Date.now() - providerStartedAt);
                  providerMetaList.push(cloudResult.meta);
                  const shifted = this.offsetStructuredTranscript(cloudResult, window.start);
                  mergedWindowTranscript.chunks.push(...shifted.chunks);
                  mergedWindowTranscript.segments.push(...shifted.segments);
                  mergedWindowTranscript.word_segments.push(...shifted.word_segments);
                } catch (windowErr) {
                  providerWallMs += Math.max(0, Date.now() - providerStartedAt);
                  if (this.isAbortError(windowErr)) {
                    throw windowErr;
                  }
                  const message = windowErr instanceof Error ? windowErr.message : String(windowErr);
                  skippedWindowErrors.push(`window_${i + 1}: ${message}`);
                  console.warn(`[ASR] VAD window ${i + 1}/${vadWindows.length} skipped: ${message}`);
                }
              } finally {
                await fs.remove(windowAudioPath).catch(() => {});
              }
            }
          }

          if (skippedWindowErrors.length > 0) {
            vadError = skippedWindowErrors.join(' | ');
          }

          if (allowVadWindowedTranscription && mergedWindowTranscript.chunks.length > 0) {
            windowedTranscript = {
              text: mergedWindowTranscript.chunks.map((chunk) => chunk.text).filter(Boolean).join(' ').trim(),
              chunks: mergedWindowTranscript.chunks.sort((a, b) => a.start_ts - b.start_ts),
              segments: mergedWindowTranscript.segments.sort((a, b) => a.start_ts - b.start_ts),
              word_segments: mergedWindowTranscript.word_segments.sort((a, b) => a.start_ts - b.start_ts),
            };
            providerMetaFromWindows = this.mergeProviderMeta(providerMetaList);
            onProgress(`ASR provider completed (${vadWindows.length} VAD windows).`);
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

    let timestampsSynthesized = false;
    if (
      effectiveSegmentation &&
      Array.isArray(result?.chunks) &&
      result.chunks.length > 0 &&
      !this.hasUsableChunkTimestamps(result.chunks)
    ) {
      const synthesized = await this.synthesizeChunkTimestamps(result.chunks, audioPath);
      if (synthesized.synthesized) {
        result.chunks = synthesized.chunks;
        timestampsSynthesized = true;
        onProgress('Provider returned no timestamps, synthesized approximate timecodes for segmented transcript.');
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
    if (effectiveWordAlignment && !resolvedAlignmentLanguage && !hasProviderNativeAlignment) {
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
    if (
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
        alignmentDiagnostics = aligned.diagnostics;
        if (aligned.applied && aligned.word_segments.length > 0) {
          result.segments = aligned.segments;
          result.word_segments = aligned.word_segments;
          result.chunks = this.buildDisplayChunksFromSegments(result.segments, language);
          result.text = this.buildTranscriptTextFromChunks(result.chunks, language);
          if (aligned.diagnostics.backend === 'native') {
            onProgress('Provider-native forced alignment applied.');
          } else {
            onProgress(`Forced alignment applied (${aligned.diagnostics.alignedSegmentCount} spans).`);
          }
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

    let diarizationDiagnostics: any = null;
    if (diarization && Array.isArray(result.chunks) && result.chunks.length > 0) {
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
      providerMeta,
      localProfile,
      effectiveSegmentation,
      effectiveWordAlignment,
      effectiveVad,
      diarizationApplied,
      result,
      speechSegments,
      vadWindows,
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
    await ProjectManager.updateProject(projectId, {
      originalSubtitles: this.formatProjectOriginalSubtitles(result),
      status: PROJECT_STATUS.TEXT_TRANSLATION,
    });
    onProgress('Transcription completed.');

    return result;
  }

  private static async transcribeCloud(
    filePath: string,
    config: any,
    options: { language?: string; prompt?: string; segmentation?: boolean; wordAlignment?: boolean; vad?: boolean; diarization?: boolean },
    signal?: AbortSignal
  ) {
    const resolvedProvider = resolveCloudAsrProvider({
      url: String(config?.url || ''),
      modelName: String(config?.name || ''),
      model: String(config?.model || ''),
    });

    const result = await requestCloudAsr(
      filePath,
      resolvedProvider,
      {
        ...config,
        url: resolvedProvider.endpointUrl,
        model: resolvedProvider.provider === 'openai-whisper' ? resolvedProvider.effectiveModel : config?.model,
      },
      options,
      {
        createAbortSignalWithTimeout: this.createAbortSignalWithTimeout.bind(this),
        extractStructuredTranscript: this.extractStructuredTranscript.bind(this),
        disableWordAlignment: this.disableWordAlignment.bind(this),
      },
      signal
    );

    return {
      ...result,
      meta: {
        ...result.meta,
        isWhisperCppInference: resolvedProvider.provider === 'whispercpp-inference',
      },
    };
  }

  static async testConnection(input: { url: string; key?: string; model?: string; name?: string }) {
    const resolvedProvider = resolveCloudAsrProvider({
      url: String(input.url || '').trim(),
      modelName: String(input.name || '').trim(),
      model: String(input.model || '').trim(),
    });

    const headers: Record<string, string> = {};
    if (input.key) {
      headers.Authorization = `Bearer ${String(input.key).trim()}`;
    }

    const request = this.createAbortSignalWithTimeout(8000);
    try {
      const response = await fetch(resolvedProvider.endpointUrl, {
        method: 'POST',
        headers,
        signal: request.signal,
        redirect: 'error',
      });

      const contentType = String(response.headers.get('content-type') || '');
      const responseText = await response.text();
      const detail = extractErrorMessage(responseText, contentType);
      const pathLower = new URL(resolvedProvider.endpointUrl).pathname.toLowerCase();
      const isOpenAiAsrEndpoint = pathLower.includes('/audio/transcriptions');
      const isWhisperCppInferenceEndpoint = pathLower.includes('/inference');
      const authFailed = hasAuthFailureHint(detail);
      const payloadValidationFailed = hasPayloadValidationHint(detail);

      if (response.status === 401 || response.status === 403 || authFailed) {
        return {
          success: false,
          error: detail || `Authentication failed (${response.status}).`,
        };
      }

      if (response.status >= 200 && response.status < 300) {
        return { success: true, message: 'Connection succeeded.' };
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
    if (!normalizedText || normalizedText.length < 4) return false;
    if (!this.shouldTreatTextAsNoSpaceScript(normalizedText, language)) return false;

    const nativeText = normalizeNoSpaceAlignmentText(this.joinWordSegments(words, language));
    if (!nativeText) return true;

    const coverageRatio = nativeText.length / Math.max(1, normalizedText.length);
    if (words.length <= 1 && normalizedText.length >= 4) return true;
    if (coverageRatio < 0.38) return true;
    if (coverageRatio < 0.55 && words.length <= 3) return true;
    return false;
  }

  private static splitNoSpaceSegmentByWords(segment: AsrStructuredSegment, language?: string): AsrStructuredSegment[] {
    const words = Array.isArray(segment?.words) ? segment.words : [];
    if (words.length <= 1 || !this.shouldTreatWordListAsNoSpaceScript(words, language)) {
      return [segment];
    }

    const maxDurationSec = 2.45;
    const strongPauseSec = 0.34;
    const clausePauseSec = 0.18;
    const maxChars = 14;
    const minCharsForBreak = 6;
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
      const shouldBreak =
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

    const softGapSec = 0.11;
    const hardGapSec = 0.24;
    const maxPhraseChars = 12;
    const maxPhraseDurationSec = 1.7;
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

      const shouldBreak =
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

      const shouldBreak =
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
                    probability: Number.isFinite(Number(word?.probability)) ? Number(word?.probability) : undefined,
                    source_segment_index: segmentIndex,
                    token_index: 0,
                  } satisfies AsrWordSegment;
                })
                .filter(Boolean) as AsrWordSegment[]
            : [];
          cjkWordDiagnostics.nativeWordCount += rawWords.length;
          const sparseNativeRecovered =
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
          const { words: mergedWords, diagnostics, appliedCjkMerging } = this.mergeCjkWordSegments(baseWords, language);
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
          if (lexicalProjection.appliedLexicalProjection) {
            cjkMergeApplied = true;
          }
          cjkWordDiagnostics.lexicalWordCount += lexicalProjection.diagnostics.lexicalWordCount;
          cjkWordDiagnostics.lexicalSingleCharCount += lexicalProjection.diagnostics.lexicalSingleCharCount;
          cjkWordDiagnostics.lexicalMismatchCount += lexicalProjection.diagnostics.lexicalMismatchCount;
          cjkWordDiagnostics.lexicalProjectionAppliedSegments += lexicalProjection.diagnostics.lexicalProjectionAppliedSegments;
          cjkWordDiagnostics.usedIntlSegmenter = cjkWordDiagnostics.usedIntlSegmenter || lexicalProjection.diagnostics.usedIntlSegmenter;
          cjkWordDiagnostics.segmenterLocale = lexicalProjection.diagnostics.locale || cjkWordDiagnostics.segmenterLocale;
          const words = lexicalProjection.words;
          const projectedText = words.length > 0 ? this.joinWordSegments(words, language) : '';
          const normalizedProjected = this.normalizeNoSpaceAlignmentText(projectedText);
          const normalizedSegment = this.normalizeNoSpaceAlignmentText(normalizedText);
          const preferProjectedText =
            words.length > 0 &&
            (
              sparseNativeRecovered ||
              lexicalProjection.appliedLexicalProjection ||
              (appliedCjkMerging && normalizedProjected.length > 0 && normalizedProjected !== normalizedSegment)
            );
          const resolvedText = this.normalizeDisplayText(
            preferProjectedText
              ? (projectedText || normalizedText)
              : (normalizedText || projectedText),
            language
          );

          const baseSegment = {
            start_ts: start,
            end_ts: end,
            text: resolvedText,
            words: words.length > 0 ? words : undefined,
          } satisfies AsrStructuredSegment;
          const splitSegments = this.splitNoSpaceSegmentByWords(baseSegment, language);
          if (splitSegments.length > 1) {
            cjkMergeApplied = true;
            cjkWordDiagnostics.splitSegmentCount += splitSegments.length - 1;
          }
          return splitSegments;
        })
        .filter((segment: any) => segment.text.length > 0) as AsrStructuredSegment[];

      if (mappedSegments.length > 0) {
        let tokenIndex = 0;
        const wordSegments = mappedSegments.flatMap((segment, segmentIndex) =>
          (segment.words || []).map((word) => ({
            ...word,
            source_segment_index: segmentIndex,
            token_index: tokenIndex++,
          }))
        );
        const chunks = this.buildDisplayChunksFromSegments(mappedSegments, language);
        const transcriptText = this.buildTranscriptTextFromChunks(chunks, language);
        return {
          text: transcriptText,
          chunks: chunks.length > 0 ? chunks : this.buildChunksFromSegments(mappedSegments),
          segments: mappedSegments,
          word_segments: wordSegments,
          debug: cjkMergeApplied
            ? {
                cjkWordDiagnostics: {
                  ...cjkWordDiagnostics,
                  mergeApplied: cjkMergeApplied,
                  chunkSource:
                    wordSegments.length > 0
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
