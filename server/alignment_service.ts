import fs from 'fs-extra';
import path from 'path';
import { spawn } from 'child_process';
import { env, AutoModelForCTC, AutoProcessor, AutoTokenizer, Wav2Vec2FeatureExtractor } from '@huggingface/transformers';
import { OpenvinoBackend } from './openvino_backend.js';
import { OpenvinoRuntimeManager } from './openvino_runtime_manager.js';
import { PathManager } from './path_manager.js';
import { resolveToolCommand } from './runtime_tools.js';
import { LanguageAlignmentRegistry } from './language_alignment/registry.js';
import { AlignmentPlanRegistry } from './language_alignment/plan_registry.js';
import { AlignmentModelProfileRegistry } from './language_alignment/model_profiles.js';
import type { AlignmentCtcModelProfile } from './language_alignment/model_profiles.js';
import type { ResolvedAlignmentPlan } from './language_alignment/plan_registry.js';
import type {
  AlignmentRun as LanguageAlignmentRun,
  AlignmentSegmentLike,
  AlignmentWordLike,
  AlignmentWordToken as SharedAlignmentWordToken,
  ForcedAlignmentLanguageModule,
} from './language_alignment/shared/types.js';

type AlignmentWordSegment = AlignmentWordLike;

type AlignmentStructuredSegment = AlignmentSegmentLike<AlignmentWordSegment>;

type AlignmentStructuredTranscript = {
  text?: string;
  chunks: Array<{ text: string; start_ts: number; end_ts?: number; speaker?: string }>;
  segments: AlignmentStructuredSegment[];
  word_segments: AlignmentWordSegment[];
  debug?: Record<string, any>;
};

type AlignmentDiagnostics = {
  applied: boolean;
  profileId: string;
  backend: 'ctc' | 'native';
  modelId: string;
  language: string | null;
  attemptedSegmentCount: number;
  alignedSegmentCount: number;
  skippedSegments: number;
  failureCount: number;
  alignedWordCount: number;
  avgConfidence: number | null;
  elapsedMs: number;
};

type AlignmentResult = {
  applied: boolean;
  segments: AlignmentStructuredSegment[];
  word_segments: AlignmentWordSegment[];
  diagnostics: AlignmentDiagnostics;
};

type AlignmentResources = {
  backend: 'transformers' | 'openvino';
  processor: any;
  tokenizer: any;
  model: any;
  compiledModel?: any;
  inputPorts?: Array<{ name: string; elementType: string }>;
  outputName?: string | null;
  blankId: number;
  charToTokenId: Map<string, number>;
  wordDelimiterToken?: string | null;
};

type AlignmentWordToken = SharedAlignmentWordToken;

type SegmentAlignment = {
  words: AlignmentWordSegment[];
  start_ts: number;
  end_ts: number;
  avgConfidence: number | null;
};

export class AlignmentService {
  private static readonly sampleRate = 16000;
  private static readonly englishAutoStopwords = new Set([
    'the', 'and', 'you', 'that', 'this', 'with', 'have', 'are', 'not', 'but', 'for', 'your', 'what', 'just', 'they', 'from',
  ]);
  private static readonly englishAutoPriorityWords = new Set([
    'im', 'youre', 'were', 'theyre', 'its', 'dont', 'didnt', 'cant', 'really', 'people', 'think', 'going',
  ]);
  private static genericResourcesPromises = new Map<string, Promise<AlignmentResources>>();

  private static getAlignmentModelProgressLabel(profile: AlignmentCtcModelProfile) {
    return String(profile.modelId || profile.id || 'alignment model').trim();
  }

  static supportsLanguagePackForcedAlignment(language?: string, sampleText?: string) {
    return Boolean(LanguageAlignmentRegistry.getForcedAlignmentModule(language, sampleText));
  }

  static supportsJapaneseForcedAlignment(language?: string, sampleText?: string) {
    return this.supportsLanguagePackForcedAlignment(language, sampleText);
  }

  static supportsForcedAlignment(language?: string, sampleText?: string) {
    return Boolean(this.resolveForcedAlignmentLanguage(language, sampleText));
  }

  static resolveForcedAlignmentLanguage(language?: string, sampleText?: string) {
    const normalized = String(language || '').trim().toLowerCase();
    if (normalized && normalized !== 'auto' && this.supportsLanguagePackForcedAlignment(normalized, sampleText)) {
      return normalized;
    }
    if (normalized && normalized !== 'auto') {
      return null;
    }

    const bestPack = LanguageAlignmentRegistry.getBestForcedAlignmentTextMatch(sampleText);
    if (bestPack?.module) {
      return bestPack.module.aliases[0] || bestPack.module.key;
    }
    return null;
  }

  static async alignTranscript(
    audioPath: string,
    transcript: AlignmentStructuredTranscript,
    options: {
      language?: string;
      providerMeta?: Record<string, any> | null;
      localModelId?: string | null;
      localModelRuntime?: string | null;
      onProgress?: (msg: string) => void;
      signal?: AbortSignal;
    } = {}
  ): Promise<AlignmentResult> {
    const startedAt = Date.now();
    const language = String(options.language || '').trim().toLowerCase();
    const segments = Array.isArray(transcript?.segments) ? transcript.segments : [];
    const transcriptText =
      segments.length > 0
        ? segments.map((segment) => segment?.text || '').join(' ')
        : String(transcript?.text || '');
    const plan = AlignmentPlanRegistry.resolve({
      language,
      sampleText: transcriptText,
      providerMeta: options.providerMeta || null,
      localModelId: options.localModelId || null,
      localModelRuntime: options.localModelRuntime || null,
    });
    const emptyDiagnostics: AlignmentDiagnostics = {
      applied: false,
      profileId: plan?.profile.id || 'unresolved',
      backend: plan?.mode || 'ctc',
      modelId: plan?.profile.modelId || 'unknown',
      language: language || null,
      attemptedSegmentCount: 0,
      alignedSegmentCount: 0,
      skippedSegments: 0,
      failureCount: 0,
      alignedWordCount: 0,
      avgConfidence: null,
      elapsedMs: 0,
    };

    if (!plan) {
      return {
        applied: false,
        segments,
        word_segments: transcript.word_segments || [],
        diagnostics: emptyDiagnostics,
      };
    }

    if (plan.mode === 'native') {
      const nativeWords = Array.isArray(transcript.word_segments) ? transcript.word_segments : [];
      const applied = Boolean(options.providerMeta?.forcedAlignment?.applied) && nativeWords.length > 0;
      const elapsedMs = Math.max(0, Date.now() - startedAt);
      return {
        applied,
        segments,
        word_segments: nativeWords,
        diagnostics: {
          applied,
          profileId: plan.profile.id,
          backend: 'native',
          modelId: String(options.providerMeta?.forcedAlignment?.modelId || plan.profile.modelId),
          language: String(options.providerMeta?.forcedAlignment?.language || plan.language || '').trim() || null,
          attemptedSegmentCount: segments.length,
          alignedSegmentCount: applied ? segments.length : 0,
          skippedSegments: 0,
          failureCount: applied ? 0 : segments.length,
          alignedWordCount: nativeWords.length,
          avgConfidence: null,
          elapsedMs,
        },
      };
    }

    if (segments.length === 0) {
      return {
        applied: false,
        segments,
        word_segments: transcript.word_segments || [],
        diagnostics: {
          ...emptyDiagnostics,
          profileId: plan.profile.id,
          backend: 'ctc',
          modelId: plan.profile.modelId,
        },
      };
    }

    const primaryResult = await this.runCtcAlignmentPlan({
      audioPath,
      language,
      transcriptText,
      segments,
      plan,
      startedAt,
      onProgress: options.onProgress,
      signal: options.signal,
    });
    if (!this.shouldRetryZhCnWithMmsFallback(language, plan, primaryResult.diagnostics)) {
      return primaryResult;
    }

    const mmsProfile = AlignmentModelProfileRegistry.get('mms-300m-forced-aligner-v1');
    if (!mmsProfile || mmsProfile.kind !== 'ctc') {
      return primaryResult;
    }

    const fallbackPlan: ResolvedAlignmentPlan = {
      ...plan,
      profile: mmsProfile,
      mode: 'ctc',
      config: {
        ...plan.config,
        profileId: mmsProfile.id,
        progressLabel: `${plan.config.progressLabel} (MMS fallback)`,
      },
    };
    options.onProgress?.('zh-CN specialized aligner produced no aligned segments, retrying with MMS fallback.');
    const fallbackResult = await this.runCtcAlignmentPlan({
      audioPath,
      language,
      transcriptText,
      segments,
      plan: fallbackPlan,
      startedAt,
      onProgress: options.onProgress,
      signal: options.signal,
    });
    return fallbackResult.applied ? fallbackResult : primaryResult;
  }

  private static shouldRetryZhCnWithMmsFallback(
    language: string,
    plan: ResolvedAlignmentPlan,
    diagnostics: AlignmentDiagnostics
  ) {
    const normalizedLanguage = String(language || '').trim().toLowerCase();
    const isZhCnRequest = normalizedLanguage === 'zh-cn' || normalizedLanguage === 'zh-hans' || normalizedLanguage === 'zh-sg';
    const isZhCnProfile = plan.profile.id === 'zh-cn-jonatas-xlsr53-chinese-v1';
    return Boolean(
      isZhCnRequest &&
        isZhCnProfile &&
        diagnostics.attemptedSegmentCount > 0 &&
        diagnostics.alignedSegmentCount === 0 &&
        diagnostics.failureCount >= diagnostics.attemptedSegmentCount
    );
  }

  private static async runCtcAlignmentPlan(input: {
    audioPath: string;
    language: string;
    transcriptText: string;
    segments: AlignmentStructuredSegment[];
    plan: ResolvedAlignmentPlan;
    startedAt: number;
    onProgress?: (msg: string) => void;
    signal?: AbortSignal;
  }): Promise<AlignmentResult> {
    this.throwIfAborted(input.signal);
    input.onProgress?.(`Preparing ${input.plan.config.progressLabel}...`);

    const [resources, fullAudio] = await Promise.all([
      this.getGenericResources(input.plan.profile as AlignmentCtcModelProfile, input.onProgress),
      this.extractMonoAudio(input.audioPath),
    ]);
    input.onProgress?.(`Running ${input.plan.config.progressLabel}...`);
    const totalSec = fullAudio.length / this.sampleRate;

    let attemptedSegmentCount = 0;
    let alignedSegmentCount = 0;
    let failureCount = 0;
    let skippedSegments = 0;
    let alignedWordCount = 0;
    let confidenceSum = 0;
    let confidenceCount = 0;

    const languageModule = input.plan.module as ForcedAlignmentLanguageModule;
    const nextSegments: AlignmentStructuredSegment[] = [];
    for (let index = 0; index < input.segments.length; index += 1) {
      this.throwIfAborted(input.signal);
      const segment = this.cloneSegment(input.segments[index]);
      const projectedWords = await languageModule.projectWordReadings(segment, input.language, input.transcriptText);
      const words =
        projectedWords && projectedWords.length > 0
          ? projectedWords
          : Array.isArray(segment.words)
            ? segment.words.map((word) => ({ ...word }))
            : [];
      const hasTiming =
        Number.isFinite(Number(segment?.start_ts)) &&
        Number.isFinite(Number(segment?.end_ts)) &&
        Number(segment.end_ts) > Number(segment.start_ts);

      if (!hasTiming || words.length === 0) {
        skippedSegments += 1;
        nextSegments.push(segment);
        continue;
      }

      const runs = languageModule.extractAlignableRuns(words) as LanguageAlignmentRun[];
      if (runs.length === 0) {
        skippedSegments += 1;
        nextSegments.push(segment);
        continue;
      }

      let segmentAligned = false;
      let segmentFailure = false;
      const updatedWords = words.map((word) => ({ ...word }));
      for (const run of runs) {
        attemptedSegmentCount += 1;
        if (attemptedSegmentCount === 1 || attemptedSegmentCount % 25 === 0) {
          input.onProgress?.(`Running ${input.plan.config.progressLabel} (${attemptedSegmentCount})...`);
        }
        try {
          const aligned = await this.alignSegment(
            fullAudio,
            totalSec,
            this.toFiniteNumber(updatedWords[run.startIndex]?.start_ts, 0),
            this.getWordEnd(updatedWords[run.endIndex]),
            run.tokens,
            resources,
            input.plan.config.clipPaddingSec,
            input.plan.config.minRunConfidence ?? 0.5
          );
          if (!aligned || aligned.words.length !== run.tokens.length) {
            segmentFailure = true;
            failureCount += 1;
            continue;
          }

          for (let offset = 0; offset < aligned.words.length; offset += 1) {
            const targetIndex = run.startIndex + offset;
            updatedWords[targetIndex] = {
              ...updatedWords[targetIndex],
              start_ts: aligned.words[offset].start_ts,
              end_ts: aligned.words[offset].end_ts,
              probability: aligned.words[offset].probability,
            };
          }
          segmentAligned = true;
          alignedSegmentCount += 1;
          alignedWordCount += aligned.words.length;
          if (aligned.avgConfidence != null) {
            confidenceSum += aligned.avgConfidence;
            confidenceCount += 1;
          }
        } catch (error) {
          segmentFailure = true;
          failureCount += 1;
          console.warn(`[ALIGN] ${input.plan.config.progressLabel} skipped run in segment ${index + 1}:`, error);
        }
      }

      if (!segmentAligned && segmentFailure) {
        nextSegments.push(segment);
        continue;
      }

      if (segmentAligned) {
        segment.words = updatedWords;
        segment.start_ts = updatedWords[0]?.start_ts ?? segment.start_ts;
        segment.end_ts = updatedWords[updatedWords.length - 1]?.end_ts ?? segment.end_ts;
      }
      nextSegments.push(segment);
    }

    let tokenIndex = 0;
    const flattenedWords: AlignmentWordSegment[] = [];
    const normalizedSegments = nextSegments.map((segment, segmentIndex) => {
      const words = Array.isArray(segment.words)
        ? segment.words
            .map((word) => ({
              ...word,
              source_segment_index: segmentIndex,
              token_index: tokenIndex++,
            }))
        : undefined;

      if (Array.isArray(words) && words.length > 0) {
        flattenedWords.push(...words);
      }

      return {
        ...segment,
        words,
      };
    });

    const overallAvgConfidence = confidenceCount > 0 ? Number((confidenceSum / confidenceCount).toFixed(4)) : null;
    const requiredAlignedSegments = Math.max(1, Math.ceil(attemptedSegmentCount * input.plan.config.minOverallAlignedRatio));
    const applied =
      attemptedSegmentCount > 0 &&
      alignedSegmentCount >= requiredAlignedSegments &&
      overallAvgConfidence != null &&
      overallAvgConfidence >= input.plan.config.minOverallConfidence;
    const elapsedMs = Math.max(0, Date.now() - input.startedAt);
    return {
      applied,
      segments: normalizedSegments,
      word_segments: flattenedWords,
      diagnostics: {
        applied,
        profileId: input.plan.profile.id,
        backend: 'ctc',
        modelId: input.plan.profile.modelId,
        language: input.plan.variant || input.language || null,
        attemptedSegmentCount,
        alignedSegmentCount,
        skippedSegments,
        failureCount,
        alignedWordCount,
        avgConfidence: overallAvgConfidence,
        elapsedMs,
      },
    };
  }

  private static throwIfAborted(signal?: AbortSignal) {
    if (signal?.aborted) {
      throw new Error('Alignment request aborted.');
    }
  }

  private static getFfmpegBinary() {
    return resolveToolCommand('ffmpeg');
  }

  private static shouldUseOpenvinoAlignmentModel(profile: AlignmentCtcModelProfile) {
    return Boolean(profile.sourceFormat && profile.conversionMethod && profile.runtimeLayout);
  }

  private static getOpenvinoAlignmentModelDir(modelId: string) {
    const safeName = String(modelId || '')
      .trim()
      .replace(/[\\/]+/g, '--')
      .replace(/[^a-zA-Z0-9._-]/g, '-');
    return path.join(PathManager.getOpenvinoAlignmentModelsPath(), safeName);
  }

  private static async ensureOpenvinoAlignmentModel(
    profile: AlignmentCtcModelProfile,
    onProgress?: (msg: string) => void
  ) {
    if (!this.shouldUseOpenvinoAlignmentModel(profile)) {
      throw new Error(`Alignment profile does not define an OpenVINO conversion path: ${profile.id}`);
    }
    const modelLabel = this.getAlignmentModelProgressLabel(profile);
    const modelDir = this.getOpenvinoAlignmentModelDir(profile.modelId);
    onProgress?.(`Checking alignment model ${modelLabel}...`);
    const requiredPaths = [
      path.join(modelDir, 'openvino_model.xml'),
      path.join(modelDir, 'openvino_model.bin'),
      path.join(modelDir, 'config.json'),
      path.join(modelDir, 'preprocessor_config.json'),
      path.join(modelDir, 'tokenizer_config.json'),
    ];
    const hasTokenizer = await Promise.all([
      fs.pathExists(path.join(modelDir, 'tokenizer.json')),
      fs.pathExists(path.join(modelDir, 'vocab.json')),
      fs.pathExists(path.join(modelDir, 'vocab.txt')),
    ]).then((values) => values.some(Boolean));
    const hasBaseFiles = (await Promise.all(requiredPaths.map((candidate) => fs.pathExists(candidate)))).every(Boolean);
    if (hasBaseFiles && hasTokenizer) {
      return modelDir;
    }

    onProgress?.(`Downloading alignment model ${modelLabel}...`);
    onProgress?.(`Converting alignment model ${modelLabel} to OpenVINO...`);
    await OpenvinoRuntimeManager.convertHuggingFaceModel({
      repoId: profile.modelId,
      outputDir: modelDir,
      type: 'asr',
      sourceFormat: profile.sourceFormat,
      conversionMethod: profile.conversionMethod,
      runtimeLayout: profile.runtimeLayout,
      envOverrides: profile.envOverrides,
    });
    return modelDir;
  }

  private static parseOpenvinoAlignmentPorts(xmlPath: string) {
    const xml = fs.readFileSync(xmlPath, 'utf8');
    const inputPorts = Array.from(
      xml.matchAll(/<layer\b[^>]*name="([^"]+)"[^>]*type="Parameter"[^>]*>[\s\S]*?<data\b[^>]*element_type="([^"]+)"/g)
    ).map((match) => ({
      name: String(match[1] || '').trim(),
      elementType: String(match[2] || 'f32').trim().toLowerCase(),
    }));

    const outputName =
      xml.match(/<layer\b[^>]*type="Result"[^>]*output_names="([^"]+)"/)?.[1]?.trim() ||
      xml.match(/<layer\b[^>]*name="([^"]+)"[^>]*type="Result"/)?.[1]?.trim() ||
      null;

    return {
      inputPorts,
      outputName,
    };
  }

  private static async getOpenvinoGenericResources(
    profile: AlignmentCtcModelProfile,
    onProgress?: (msg: string) => void
  ): Promise<AlignmentResources> {
    const modelDir = await this.ensureOpenvinoAlignmentModel(profile, onProgress);
    const xmlPath = path.join(modelDir, 'openvino_model.xml');
    const { inputPorts, outputName } = this.parseOpenvinoAlignmentPorts(xmlPath);
    const device = OpenvinoBackend.getBaselineModelDevice();
    onProgress?.(`Loading alignment model ${this.getAlignmentModelProgressLabel(profile)}...`);
    const [processor, tokenizer, compiledModel, vocab] = await Promise.all([
      AutoProcessor.from_pretrained(modelDir, { local_files_only: true }).catch(() =>
        Wav2Vec2FeatureExtractor.from_pretrained(modelDir, { local_files_only: true })
      ),
      AutoTokenizer.from_pretrained(modelDir, { local_files_only: true }).catch(() => null),
      OpenvinoBackend.compileModel(xmlPath, device),
      this.loadAlignmentVocab(modelDir),
    ]);

    const charToTokenId = new Map<string, number>();
    const tokenizerVocab = tokenizer?._tokenizer?.model?.vocab || {};
    const mergedVocab = {
      ...(vocab || {}),
      ...tokenizerVocab,
    } as Record<string, number>;
    for (const [token, id] of Object.entries(mergedVocab)) {
      if (Number.isFinite(Number(id))) {
        const normalizedToken = String(token);
        charToTokenId.set(normalizedToken, Number(id));
        const lowered = normalizedToken.toLowerCase();
        if (!charToTokenId.has(lowered)) {
          charToTokenId.set(lowered, Number(id));
        }
      }
    }

    const config = await fs.readJson(path.join(modelDir, 'config.json')).catch(() => ({} as Record<string, any>));
    return {
      backend: 'openvino' as const,
      processor,
      tokenizer,
      model: null,
      compiledModel,
      inputPorts,
      outputName,
      blankId: Number.isFinite(Number(config?.pad_token_id)) ? Number(config?.pad_token_id) : 0,
      charToTokenId,
      wordDelimiterToken:
        tokenizer?.word_delimiter_token ||
        tokenizer?.wordDelimiterToken ||
        (charToTokenId.has('|') ? '|' : null),
    };
  }

  private static async getGenericResources(
    profile: AlignmentCtcModelProfile,
    onProgress?: (msg: string) => void
  ): Promise<AlignmentResources> {
    const cacheKey = profile.id;
    const cached = this.genericResourcesPromises.get(cacheKey);
    if (!cached) {
      const nextPromise = (async () => {
        if (this.shouldUseOpenvinoAlignmentModel(profile)) {
          return this.getOpenvinoGenericResources(profile, onProgress);
        }

        onProgress?.(`Checking alignment model ${this.getAlignmentModelProgressLabel(profile)}...`);
        onProgress?.(`Downloading alignment model ${this.getAlignmentModelProgressLabel(profile)}...`);
        env.cacheDir = PathManager.getTransformersCachePath();
        const [processor, tokenizer, model, vocab] = await Promise.all([
          AutoProcessor.from_pretrained(profile.modelId).catch(() => Wav2Vec2FeatureExtractor.from_pretrained(profile.modelId)),
          AutoTokenizer.from_pretrained(profile.modelId).catch(() => null),
          AutoModelForCTC.from_pretrained(profile.modelId, { dtype: 'fp32' }),
          this.fetchRawJson<Record<string, number>>(`https://huggingface.co/${profile.modelId}/raw/main/vocab.json`),
        ]);
        onProgress?.(`Loading alignment model ${this.getAlignmentModelProgressLabel(profile)}...`);

        const charToTokenId = new Map<string, number>();
        const tokenizerVocab = tokenizer?._tokenizer?.model?.vocab || {};
        const mergedVocab = {
          ...(vocab || {}),
          ...tokenizerVocab,
        } as Record<string, number>;
        for (const [token, id] of Object.entries(mergedVocab)) {
          if (Number.isFinite(Number(id))) {
            const normalizedToken = String(token);
            charToTokenId.set(normalizedToken, Number(id));
            const lowered = normalizedToken.toLowerCase();
            if (!charToTokenId.has(lowered)) {
              charToTokenId.set(lowered, Number(id));
            }
          }
        }

        const config = model?.config as Record<string, any> | undefined;
        return {
          backend: 'transformers' as const,
          processor,
          tokenizer,
          model,
          blankId: Number.isFinite(Number(config?.pad_token_id)) ? Number(config?.pad_token_id) : 0,
          charToTokenId,
          wordDelimiterToken:
            tokenizer?.word_delimiter_token ||
            tokenizer?.wordDelimiterToken ||
            (charToTokenId.has('|') ? '|' : null),
        };
      })();
      this.genericResourcesPromises.set(cacheKey, nextPromise);
    }

    return this.genericResourcesPromises.get(cacheKey)!;
  }

  private static async fetchRawJson<T>(url: string): Promise<T> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch alignment resource (${response.status}): ${url}`);
    }
    return response.json() as Promise<T>;
  }

  private static async loadAlignmentVocab(modelDir: string): Promise<Record<string, number>> {
    const vocabJsonPath = path.join(modelDir, 'vocab.json');
    if (await fs.pathExists(vocabJsonPath)) {
      return fs.readJson(vocabJsonPath);
    }

    const tokenizerJsonPath = path.join(modelDir, 'tokenizer.json');
    if (await fs.pathExists(tokenizerJsonPath)) {
      const tokenizerJson = await fs.readJson(tokenizerJsonPath).catch(() => null);
      const vocab = tokenizerJson?.model?.vocab;
      if (vocab && typeof vocab === 'object') {
        return vocab as Record<string, number>;
      }
    }

    const vocabTxtPath = path.join(modelDir, 'vocab.txt');
    if (await fs.pathExists(vocabTxtPath)) {
      const lines = (await fs.readFile(vocabTxtPath, 'utf8'))
        .split(/\r?\n/)
        .map((line) => String(line || '').trim())
        .filter(Boolean);
      return Object.fromEntries(lines.map((token, index) => [token, index]));
    }

    return {};
  }

  private static async getKuromojiTokenizer(): Promise<{ tokenize: (text: string) => Array<{ surface_form?: string; reading?: string }> }> {
    throw new Error('Kuromoji tokenizer access moved to the language alignment module.');
  }

  private static async extractMonoAudio(filePath: string): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
      const ffmpeg = spawn(this.getFfmpegBinary(), [
        '-v',
        'error',
        '-i',
        filePath,
        '-f',
        'f32le',
        '-ac',
        '1',
        '-ar',
        String(this.sampleRate),
        'pipe:1',
      ]);

      const stdoutChunks: Buffer[] = [];
      const stderrChunks: Buffer[] = [];
      ffmpeg.stdout.on('data', (chunk: Buffer) => stdoutChunks.push(chunk));
      ffmpeg.stderr.on('data', (chunk: Buffer) => stderrChunks.push(chunk));
      ffmpeg.on('close', (code) => {
        if (code === 0) {
          const buffer = Buffer.concat(stdoutChunks);
          resolve(new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4));
          return;
        }

        const stderr = Buffer.concat(stderrChunks).toString('utf8').trim();
        reject(new Error(stderr || 'Audio extraction failed.'));
      });
      ffmpeg.on('error', (error) => reject(error));
    });
  }

  private static toFiniteNumber(value: any, fallback = 0): number {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }

  private static getWordEnd(word: AlignmentWordSegment) {
    const end = this.toFiniteNumber(word?.end_ts, Number.NaN);
    if (Number.isFinite(end)) return end;
    return this.toFiniteNumber(word?.start_ts, 0);
  }

  private static cloneSegment(segment: AlignmentStructuredSegment): AlignmentStructuredSegment {
    return {
      ...segment,
      words: Array.isArray(segment?.words)
        ? segment.words.map((word) => ({ ...word }))
        : undefined,
    };
  }

  private static extractEnglishWordTokens(text: string): AlignmentWordToken[] {
    const source = String(text || '');
    const matches = source.match(/[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*/g) || [];
    return matches
      .map((match) => ({
        text: match,
        normalized: this.normalizeEnglishWord(match),
      }))
      .filter((item) => item.normalized.length > 0);
  }

  private static normalizeEnglishWord(value: string) {
    return String(value || '')
      .replace(/[’`]/g, '\'')
      .toUpperCase()
      .replace(/[^A-Z']/g, '');
  }

  private static scoreEnglishText(text?: string) {
    const tokens = (String(text || '').match(/[A-Za-z]+(?:'[A-Za-z]+)*/g) || [])
      .map((token) => token.toLowerCase().replace(/'/g, ''))
      .filter(Boolean);
    if (tokens.length === 0) return 0;

    let score = 0;
    for (const token of tokens) {
      if (this.englishAutoStopwords.has(token)) score += 1;
      if (this.englishAutoPriorityWords.has(token)) score += 2;
    }
    return score;
  }

  private static normalizeJapaneseKana(value: string) {
    return String(value || '')
      .normalize('NFKC')
      .replace(/\u00a0/g, ' ')
      .replace(/\s+/g, '')
      .replace(/[ァ-ヶ]/g, (char) => String.fromCharCode(char.charCodeAt(0) - 0x60))
      .replace(/[—―ｰ]/g, 'ー')
      .trim();
  }

  private static normalizeJapaneseSurface(value: string) {
    return String(value || '')
      .normalize('NFKC')
      .replace(/\u00a0/g, ' ')
      .replace(/\s+/g, '')
      .trim();
  }

  private static isJapaneseAlignmentTokenSupported(value: string) {
    return /^[ぁ-ゖー]+$/.test(value);
  }

  private static convertKatakanaToHiragana(value: string) {
    return String(value || '').replace(/[ァ-ヶ]/g, (char) => String.fromCharCode(char.charCodeAt(0) - 0x60));
  }

  private static async projectJapaneseWordReadings(segment: AlignmentStructuredSegment) {
    const words = Array.isArray(segment?.words) ? segment.words.map((word) => ({ ...word })) : [];
    const surfaceText = this.normalizeJapaneseSurface(
      words.length > 0 ? words.map((word) => word.text || '').join('') : segment?.text || ''
    );
    if (!surfaceText || words.length === 0) {
      return null;
    }

    const baseWords = words
      .map((word) => ({
        ...word,
        normalizedSurface: this.normalizeJapaneseSurface(word.text || ''),
      }))
      .filter((word) => word.normalizedSurface.length > 0);
    const baseSurface = baseWords.map((word) => word.normalizedSurface).join('');
    if (!baseSurface || baseSurface !== surfaceText) {
      return null;
    }

    const tokenizer = await this.getKuromojiTokenizer();
    const tokens = tokenizer.tokenize(surfaceText);
    const tokenSurface = tokens.map((token) => this.normalizeJapaneseSurface(token.surface_form || '')).join('');
    if (!tokenSurface || tokenSurface !== surfaceText) {
      return null;
    }

    const charSpans: Array<{ char: string; start_ts: number; end_ts: number; probability?: number }> = [];
    for (const word of baseWords) {
      const chars = Array.from(word.normalizedSurface);
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

    const projected: AlignmentWordSegment[] = [];
    let charCursor = 0;
    for (const token of tokens) {
      const surface = this.normalizeJapaneseSurface(token.surface_form || '');
      if (!surface) continue;
      const surfaceChars = Array.from(surface);
      const slice = charSpans.slice(charCursor, charCursor + surfaceChars.length);
      if (slice.length !== surfaceChars.length || slice.map((item) => item.char).join('') !== surface) {
        return null;
      }
      const rawReading = token.reading && token.reading !== '*' ? token.reading : surface;
      const normalizedReading = this.normalizeJapaneseKana(this.convertKatakanaToHiragana(rawReading));
      const usesReadingProjection =
        /[\u3400-\u9fff]/.test(surface) &&
        this.isJapaneseAlignmentTokenSupported(normalizedReading) &&
        normalizedReading.length > 0;
      const probabilityValues = slice
        .map((item) => this.toFiniteNumber(item.probability, Number.NaN))
        .filter((value) => Number.isFinite(value));
      projected.push({
        text: surface,
        normalizedReading: this.isJapaneseAlignmentTokenSupported(normalizedReading) ? normalizedReading : undefined,
        usesReadingProjection,
        start_ts: slice[0].start_ts,
        end_ts: slice[slice.length - 1].end_ts,
        probability:
          probabilityValues.length > 0
            ? Number((probabilityValues.reduce((sum, value) => sum + value, 0) / probabilityValues.length).toFixed(4))
            : undefined,
      });
      charCursor += surfaceChars.length;
    }

    return projected.length > 0 ? projected : null;
  }

  private static extractJapaneseAlignableRuns(words: AlignmentWordSegment[]) {
    const annotated = words.map((word, index) => {
      const normalized = this.normalizeJapaneseKana(word?.normalizedReading || word?.text || '');
      return {
        index,
        word,
        token: normalized,
        alignable: this.isJapaneseAlignmentTokenSupported(normalized),
        requiresProjection: Boolean(word?.usesReadingProjection),
      };
    });

    const runs: Array<{ startIndex: number; endIndex: number; tokens: AlignmentWordToken[] }> = [];
    let current: typeof annotated = [];
    const flush = () => {
      if (current.length > 0) {
        runs.push({
          startIndex: current[0].index,
          endIndex: current[current.length - 1].index,
          tokens: current.map((item) => ({
            text: item.word.text,
            normalized: item.token,
          })),
        });
      }
      current = [];
    };

    for (const item of annotated) {
      if (!item.alignable) {
        flush();
        continue;
      }
      if (current.length > 0) {
        const previous = current[current.length - 1];
        const gap = this.toFiniteNumber(item.word.start_ts, 0) - this.getWordEnd(previous.word);
        if (gap > 0.3) {
          flush();
        }
      }
      current.push(item);
    }
    flush();

    return runs.filter((run) => {
      if (!run.tokens.some((token) => token.normalized.length >= 2)) {
        return false;
      }
      const slice = annotated.slice(run.startIndex, run.endIndex + 1);
      return slice.some((item) => item.requiresProjection);
    });
  }

  private static sliceAudio(fullAudio: Float32Array, startSec: number, endSec: number) {
    const startSample = Math.max(0, Math.floor(startSec * this.sampleRate));
    const endSample = Math.min(fullAudio.length, Math.ceil(endSec * this.sampleRate));
    return fullAudio.subarray(startSample, Math.max(startSample + 1, endSample));
  }

  private static getLogProb(logProbs: Float32Array, frameIndex: number, vocabSize: number, tokenId: number) {
    return logProbs[frameIndex * vocabSize + tokenId] ?? Number.NEGATIVE_INFINITY;
  }

  private static computeLogSoftmax(logits: Float32Array, frames: number, vocabSize: number) {
    const logProbs = new Float32Array(logits.length);
    for (let frame = 0; frame < frames; frame += 1) {
      const rowOffset = frame * vocabSize;
      let maxLogit = Number.NEGATIVE_INFINITY;
      for (let vocabIndex = 0; vocabIndex < vocabSize; vocabIndex += 1) {
        const value = logits[rowOffset + vocabIndex] ?? Number.NEGATIVE_INFINITY;
        if (value > maxLogit) maxLogit = value;
      }

      let sumExp = 0;
      for (let vocabIndex = 0; vocabIndex < vocabSize; vocabIndex += 1) {
        sumExp += Math.exp((logits[rowOffset + vocabIndex] ?? Number.NEGATIVE_INFINITY) - maxLogit);
      }

      const logDenominator = maxLogit + Math.log(Math.max(sumExp, 1e-12));
      for (let vocabIndex = 0; vocabIndex < vocabSize; vocabIndex += 1) {
        logProbs[rowOffset + vocabIndex] = (logits[rowOffset + vocabIndex] ?? Number.NEGATIVE_INFINITY) - logDenominator;
      }
    }
    return logProbs;
  }

  private static alignTokenSequence(
    logProbs: Float32Array,
    frames: number,
    vocabSize: number,
    tokenIds: number[],
    blankId: number
  ) {
    if (frames <= 0 || tokenIds.length === 0) return null;

    const stateCount = tokenIds.length * 2 + 1;
    const backPointers = new Int32Array(frames * stateCount);
    backPointers.fill(-1);

    let previous = new Float64Array(stateCount);
    let current = new Float64Array(stateCount);
    previous.fill(Number.NEGATIVE_INFINITY);
    current.fill(Number.NEGATIVE_INFINITY);

    previous[0] = this.getLogProb(logProbs, 0, vocabSize, blankId);
    backPointers[0] = 0;
    if (stateCount > 1) {
      previous[1] = this.getLogProb(logProbs, 0, vocabSize, tokenIds[0]);
      backPointers[1] = 1;
    }

    for (let frame = 1; frame < frames; frame += 1) {
      current.fill(Number.NEGATIVE_INFINITY);
      for (let state = 0; state < stateCount; state += 1) {
        const emitTokenId = state % 2 === 0 ? blankId : tokenIds[(state - 1) >> 1];
        let bestState = state;
        let bestScore = previous[state];

        if (state > 0 && previous[state - 1] > bestScore) {
          bestScore = previous[state - 1];
          bestState = state - 1;
        }

        if (state % 2 === 1 && state > 1) {
          const tokenIndex = (state - 1) >> 1;
          const previousTokenIndex = (state - 3) >> 1;
          if (tokenIds[tokenIndex] !== tokenIds[previousTokenIndex] && previous[state - 2] > bestScore) {
            bestScore = previous[state - 2];
            bestState = state - 2;
          }
        }

        current[state] = bestScore + this.getLogProb(logProbs, frame, vocabSize, emitTokenId);
        backPointers[frame * stateCount + state] = bestState;
      }

      const temp = previous;
      previous = current;
      current = temp;
    }

    const terminalBlank = stateCount - 1;
    const terminalToken = Math.max(0, stateCount - 2);
    let bestTerminalState = terminalBlank;
    let bestTerminalScore = previous[terminalBlank];
    if (previous[terminalToken] > bestTerminalScore) {
      bestTerminalState = terminalToken;
      bestTerminalScore = previous[terminalToken];
    }
    if (!Number.isFinite(bestTerminalScore)) return null;

    const statePath = new Int32Array(frames);
    let currentState = bestTerminalState;
    for (let frame = frames - 1; frame >= 0; frame -= 1) {
      statePath[frame] = currentState;
      if (frame > 0) {
        currentState = backPointers[frame * stateCount + currentState];
        if (currentState < 0) return null;
      }
    }

    const charStarts = new Array<number>(tokenIds.length).fill(-1);
    const charEnds = new Array<number>(tokenIds.length).fill(-1);
    const charConfidenceSums = new Array<number>(tokenIds.length).fill(0);
    const charConfidenceCounts = new Array<number>(tokenIds.length).fill(0);
    for (let frame = 0; frame < frames; frame += 1) {
      const state = statePath[frame];
      if (state % 2 === 0) continue;
      const tokenIndex = (state - 1) >> 1;
      const tokenId = tokenIds[tokenIndex];
      if (charStarts[tokenIndex] < 0) charStarts[tokenIndex] = frame;
      charEnds[tokenIndex] = frame;
      charConfidenceSums[tokenIndex] += Math.exp(this.getLogProb(logProbs, frame, vocabSize, tokenId));
      charConfidenceCounts[tokenIndex] += 1;
    }

    if (charConfidenceCounts.some((count) => count === 0)) {
      return null;
    }

    const charConfidences = charConfidenceSums.map((sum, index) => sum / Math.max(1, charConfidenceCounts[index]));
    return {
      charStarts,
      charEnds,
      charConfidences,
    };
  }

  private static toOpenvinoTensorData(elementType: string, source: any) {
    const normalized = String(elementType || 'f32').trim().toLowerCase();
    const data = source?.data ?? source;
    if (!data) {
      throw new Error(`Alignment input tensor is missing data for element type ${normalized}.`);
    }

    if (normalized === 'i64') {
      return BigInt64Array.from(Array.from(data as ArrayLike<number>, (value) => BigInt(Math.round(Number(value) || 0))));
    }
    if (normalized === 'i32') {
      return Int32Array.from(Array.from(data as ArrayLike<number>, (value) => Math.round(Number(value) || 0)));
    }
    if (normalized === 'i8') {
      return Int8Array.from(Array.from(data as ArrayLike<number>, (value) => Math.round(Number(value) || 0)));
    }
    if (normalized === 'u8') {
      return Uint8Array.from(Array.from(data as ArrayLike<number>, (value) => Math.max(0, Math.round(Number(value) || 0))));
    }
    return data;
  }

  private static async runOpenvinoAlignmentModel(resources: AlignmentResources, audioSlice: Float32Array) {
    if (!resources.compiledModel || !Array.isArray(resources.inputPorts) || resources.inputPorts.length === 0) {
      throw new Error('OpenVINO alignment model is missing compiled resources.');
    }

    const processedInputs = await resources.processor(audioSlice);
    const feeds: Record<string, any> = {};
    for (const port of resources.inputPorts) {
      const sourceTensor = processedInputs?.[port.name];
      if (!sourceTensor) continue;
      const shape = Array.isArray(sourceTensor?.dims)
        ? Array.from(sourceTensor.dims, (value) => Number(value))
        : Array.isArray(sourceTensor?.shape)
          ? Array.from(sourceTensor.shape, (value) => Number(value))
          : [];
      const tensorData = this.toOpenvinoTensorData(port.elementType, sourceTensor);
      feeds[port.name] = await OpenvinoBackend.createTensor(port.elementType, shape, tensorData);
    }

    if (Object.keys(feeds).length === 0) {
      throw new Error('OpenVINO alignment processor did not produce any compatible inputs.');
    }

    const inferRequest = resources.compiledModel.createInferRequest();
    const outputs = inferRequest.infer(feeds);
    const outputTensor =
      (resources.outputName ? outputs?.[resources.outputName] : null) ||
      outputs?.logits ||
      Object.values(outputs || {})[0];
    const logits = OpenvinoBackend.getTensorData(outputTensor) as Float32Array | null;
    const shape = typeof outputTensor?.getShape === 'function' ? outputTensor.getShape() : (outputTensor?.shape || outputTensor?.dims || null);
    const dims = Array.isArray(shape) ? shape.map((value) => Number(value)) : [];
    const frames = Number(dims[1]);
    const vocabSize = Number(dims[2]);
    if (!(logits instanceof Float32Array) || !Number.isFinite(frames) || !Number.isFinite(vocabSize) || frames <= 0 || vocabSize <= 0) {
      throw new Error('OpenVINO alignment model returned invalid logits.');
    }

    return {
      logits,
      frames,
      vocabSize,
    };
  }

  private static async alignSegment(
    fullAudio: Float32Array,
    totalSec: number,
    baseStart: number,
    baseEnd: number,
    wordTokens: AlignmentWordToken[],
    resources: AlignmentResources,
    clipPaddingSec: number,
    minSegmentConfidence: number
  ): Promise<SegmentAlignment | null> {
    const transcript = wordTokens.map((word) => word.normalized).join(' ').trim();
    if (!transcript) return null;

    const tokenIds: number[] = [];
    for (const char of Array.from(transcript)) {
      if (char === ' ') {
        const delimiter = resources.wordDelimiterToken || (resources.charToTokenId.has('|') ? '|' : null);
        if (!delimiter) {
          continue;
        }
        const delimiterId = resources.charToTokenId.get(delimiter) ?? null;
        if (!Number.isFinite(delimiterId)) {
          continue;
        }
        tokenIds.push(Number(delimiterId));
        continue;
      }

      const tokenId =
        resources.charToTokenId.get(char) ??
        resources.charToTokenId.get(char.toLowerCase()) ??
        null;
      if (!Number.isFinite(tokenId)) {
        return null;
      }
      tokenIds.push(Number(tokenId));
    }
    if (tokenIds.length === 0) return null;

    const clipStart = Math.max(0, baseStart - clipPaddingSec);
    const clipEnd = Math.min(totalSec, baseEnd + clipPaddingSec);
    if (clipEnd <= clipStart) return null;

    const audioSlice = this.sliceAudio(fullAudio, clipStart, clipEnd);
    if (audioSlice.length < Math.round(this.sampleRate * 0.08)) return null;

    const openvinoLogits =
      resources.backend === 'openvino'
        ? await this.runOpenvinoAlignmentModel(resources, audioSlice)
        : null;
    const inputs = openvinoLogits ? null : await resources.processor(audioSlice);
    const outputs = openvinoLogits ? null : await resources.model(inputs);
    const frames = Number(openvinoLogits?.frames ?? outputs?.logits?.dims?.[1]);
    const vocabSize = Number(openvinoLogits?.vocabSize ?? outputs?.logits?.dims?.[2]);
    const logits = (openvinoLogits?.logits ?? outputs?.logits?.data) as Float32Array | undefined;
    if (!(logits instanceof Float32Array) || !Number.isFinite(frames) || !Number.isFinite(vocabSize) || frames <= 0 || vocabSize <= 0) {
      return null;
    }
    if (frames < tokenIds.length) return null;

    const logProbs = this.computeLogSoftmax(logits, frames, vocabSize);
    const aligned = this.alignTokenSequence(logProbs, frames, vocabSize, tokenIds, resources.blankId);
    if (!aligned) return null;

    const secondsPerFrame = (clipEnd - clipStart) / frames;
    if (!(secondsPerFrame > 0)) return null;

    let charCursor = 0;
    const words: AlignmentWordSegment[] = [];
    for (const word of wordTokens) {
      const startCharIndex = charCursor;
      const endCharIndex = charCursor + word.normalized.length - 1;
      const startFrame = aligned.charStarts[startCharIndex];
      const endFrame = aligned.charEnds[endCharIndex];
      if (startFrame < 0 || endFrame < startFrame) {
        return null;
      }

      let probabilitySum = 0;
      for (let charIndex = startCharIndex; charIndex <= endCharIndex; charIndex += 1) {
        probabilitySum += aligned.charConfidences[charIndex] ?? 0;
      }

      words.push({
        text: word.text,
        start_ts: Number((clipStart + startFrame * secondsPerFrame).toFixed(3)),
        end_ts: Number((clipStart + (endFrame + 1) * secondsPerFrame).toFixed(3)),
        probability: Number((probabilitySum / Math.max(1, word.normalized.length)).toFixed(4)),
      });

      charCursor = endCharIndex + 1;
      if (charCursor < transcript.length && transcript[charCursor] === ' ') {
        charCursor += 1;
      }
    }

    if (words.length === 0) return null;
    const avgConfidence = words.reduce((sum, word) => sum + (word.probability || 0), 0) / words.length;
    if (!(avgConfidence >= minSegmentConfidence)) {
      return null;
    }
    return {
      words,
      start_ts: words[0].start_ts,
      end_ts: Number(words[words.length - 1].end_ts ?? words[words.length - 1].start_ts),
      avgConfidence: Number(avgConfidence.toFixed(4)),
    };
  }
}
