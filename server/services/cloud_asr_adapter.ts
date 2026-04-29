import fs from 'fs-extra';
import crypto from 'node:crypto';
import path from 'node:path';
import type { CloudAsrProvider, ResolvedCloudAsrProvider } from './cloud_asr_provider.js';

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

interface CloudAsrAdapterBuildFormInput {
  filePath: string;
  fileBuffer: Buffer;
  config: any;
  options: CloudAsrAdapterRequestOptions;
  includeLanguage: boolean;
  responseFormat: string;
}

export interface CloudAsrAdapter {
  provider: CloudAsrProvider;
  getPreferredResponseFormats(options: CloudAsrAdapterRequestOptions): string[];
  buildRequestUrl?(endpointUrl: string, input: CloudAsrAdapterBuildFormInput): string;
  buildFormData?(input: CloudAsrAdapterBuildFormInput): FormData;
  buildRawBody?(input: CloudAsrAdapterBuildFormInput): BodyInit;
  buildJsonBody?(input: CloudAsrAdapterBuildFormInput): Record<string, unknown>;
  getRequestHeaders?(input: CloudAsrAdapterBuildFormInput): Record<string, string>;
  normalizeResponse?(data: any): any;
}

function bufferToBlobPart(buffer: Buffer): BlobPart {
  const copy = new ArrayBuffer(buffer.byteLength);
  new Uint8Array(copy).set(buffer);
  return copy;
}

function extractErrorMessage(errorText: string, fallback: string) {
  let errorMessage = fallback;
  try {
    const parsed = JSON.parse(errorText);
    const nested = parsed?.error;
    const nestedMessage = typeof nested === 'string' ? nested : nested?.message;
    errorMessage = nestedMessage || parsed?.message || parsed?.detail || errorMessage;
  } catch {
    errorMessage = errorText || errorMessage;
  }
  return String(errorMessage || '').trim() || fallback;
}

function sleepWithAbort(ms: number, signal?: AbortSignal) {
  const delayMs = Math.max(0, Math.round(ms));
  if (delayMs === 0) return Promise.resolve();
  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      const error = new Error('Aborted');
      error.name = 'AbortError';
      reject(error);
      return;
    }
    const timeout = setTimeout(() => {
      signal?.removeEventListener('abort', abort);
      resolve();
    }, delayMs);
    const abort = () => {
      clearTimeout(timeout);
      const error = new Error('Aborted');
      error.name = 'AbortError';
      reject(error);
    };
    signal?.addEventListener('abort', abort, { once: true });
  });
}

function parseRetryAfterMs(response: Response) {
  const raw = response.headers.get('retry-after');
  if (!raw) return null;
  const asSeconds = Number(raw);
  if (Number.isFinite(asSeconds) && asSeconds >= 0) {
    return Math.min(30_000, Math.round(asSeconds * 1000));
  }
  const asDate = Date.parse(raw);
  if (Number.isFinite(asDate)) {
    return Math.min(30_000, Math.max(0, asDate - Date.now()));
  }
  return null;
}

function shouldRetryTransientCloudAsrResponse(response: Response) {
  return response.status === 408 || response.status === 429 || response.status === 500 || response.status === 502 || response.status === 503 || response.status === 504;
}

function getTransientRetryDelayMs(response: Response, attemptIndex: number) {
  const retryAfterMs = parseRetryAfterMs(response);
  if (retryAfterMs != null) return retryAfterMs;
  return Math.min(8_000, 1_000 * 2 ** attemptIndex);
}

const GEMINI_FREE_TIER_LIMITS = {
  rpm: 5,
  tpm: 250_000,
  rpd: 20,
};
const GEMINI_AUDIO_TOKENS_PER_SECOND = 32;
const GEMINI_ASR_TEXT_TOKEN_BUFFER = 1024;

interface GeminiFreeTierLimiterState {
  queue: Promise<void>;
  requestTimestamps: number[];
  tokenUsages: Array<{ timestamp: number; tokens: number }>;
  dayKey: string;
  dailyCount: number;
}

const geminiFreeTierLimiterState = new Map<string, GeminiFreeTierLimiterState>();

function getPacificDayKey(now = new Date()) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'America/Los_Angeles',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(now);
  const byType = new Map(parts.map((part) => [part.type, part.value]));
  return `${byType.get('year') || '0000'}-${byType.get('month') || '00'}-${byType.get('day') || '00'}`;
}

function buildGeminiFreeTierLimiterKey(config: any, resolvedProvider: ResolvedCloudAsrProvider) {
  const keyHash = crypto
    .createHash('sha256')
    .update(String(config?.key || '').trim())
    .digest('hex')
    .slice(0, 16);
  return `${resolvedProvider.provider}:${resolvedProvider.effectiveModel}:${keyHash}`;
}

async function enforceGeminiFreeTierRateLimit(
  config: any,
  resolvedProvider: ResolvedCloudAsrProvider,
  estimatedInputTokens: number | null,
  signal?: AbortSignal
) {
  if (resolvedProvider.provider !== 'google-gemini-audio') return;

  if (estimatedInputTokens != null && estimatedInputTokens > GEMINI_FREE_TIER_LIMITS.tpm) {
    throw new Error(
      `Gemini free tier single request token estimate exceeds the local TPM limit (${estimatedInputTokens}/${GEMINI_FREE_TIER_LIMITS.tpm} input tokens for ${resolvedProvider.effectiveModel}). Use VAD/windowed transcription or shorter audio chunks.`
    );
  }

  const limiterKey = buildGeminiFreeTierLimiterKey(config, resolvedProvider);
  let state = geminiFreeTierLimiterState.get(limiterKey);
  if (!state) {
    state = {
      queue: Promise.resolve(),
      requestTimestamps: [],
      tokenUsages: [],
      dayKey: getPacificDayKey(),
      dailyCount: 0,
    };
    geminiFreeTierLimiterState.set(limiterKey, state);
  }

  const run = async () => {
    while (true) {
      if (signal?.aborted) {
        const error = new Error('Aborted');
        error.name = 'AbortError';
        throw error;
      }

      const now = Date.now();
      const currentDayKey = getPacificDayKey(new Date(now));
      if (state!.dayKey !== currentDayKey) {
        state!.dayKey = currentDayKey;
        state!.dailyCount = 0;
        state!.requestTimestamps = [];
        state!.tokenUsages = [];
      }

      state!.requestTimestamps = state!.requestTimestamps.filter((timestamp) => now - timestamp < 60_000);
      state!.tokenUsages = state!.tokenUsages.filter((usage) => now - usage.timestamp < 60_000);
      if (state!.dailyCount >= GEMINI_FREE_TIER_LIMITS.rpd) {
        throw new Error(
          `Gemini free tier daily request limit reached locally (${GEMINI_FREE_TIER_LIMITS.rpd} RPD for ${resolvedProvider.effectiveModel}). Daily quota resets at Pacific Time midnight.`
        );
      }
      const currentMinuteTokens = state!.tokenUsages.reduce((sum, usage) => sum + usage.tokens, 0);
      if (
        estimatedInputTokens != null &&
        currentMinuteTokens + estimatedInputTokens > GEMINI_FREE_TIER_LIMITS.tpm &&
        state!.tokenUsages.length > 0
      ) {
        const oldestTokenUsage = state!.tokenUsages[0];
        await sleepWithAbort(Math.max(0, oldestTokenUsage.timestamp + 60_000 - now), signal);
        continue;
      }

      const oldestRequest = state!.requestTimestamps[0];
      const nextBucketSlotMs =
        state!.requestTimestamps.length >= GEMINI_FREE_TIER_LIMITS.rpm && oldestRequest != null
          ? oldestRequest + 60_000
          : now;
      const minSpacingMs = Math.ceil(60_000 / GEMINI_FREE_TIER_LIMITS.rpm);
      const lastRequest = state!.requestTimestamps[state!.requestTimestamps.length - 1];
      const nextSmoothSlotMs = lastRequest != null ? lastRequest + minSpacingMs : now;
      const waitMs = Math.max(0, nextBucketSlotMs - now, nextSmoothSlotMs - now);

      if (waitMs <= 0) {
        state!.requestTimestamps.push(now);
        if (estimatedInputTokens != null) {
          state!.tokenUsages.push({ timestamp: now, tokens: estimatedInputTokens });
        }
        state!.dailyCount += 1;
        return;
      }
      await sleepWithAbort(waitMs, signal);
    }
  };

  const queued = state.queue.then(run, run);
  state.queue = queued.catch(() => {});
  await queued;
}

function estimateGeminiAudioAsrInputTokens(options: CloudAsrAdapterRequestOptions) {
  const durationSec = Number(options.audioDurationSec);
  if (!Number.isFinite(durationSec) || durationSec <= 0) return null;
  return Math.ceil(durationSec * GEMINI_AUDIO_TOKENS_PER_SECOND) + GEMINI_ASR_TEXT_TOKEN_BUFFER;
}

function estimateGeminiBatchAudioAsrInputTokens(inputs: CloudAsrBatchAudioInput[]) {
  const totalDurationSec = inputs.reduce((sum, input) => {
    const durationSec = Number(input.audioDurationSec);
    return sum + (Number.isFinite(durationSec) && durationSec > 0 ? durationSec : 0);
  }, 0);
  if (totalDurationSec <= 0) return null;
  return Math.ceil(totalDurationSec * GEMINI_AUDIO_TOKENS_PER_SECOND) + GEMINI_ASR_TEXT_TOKEN_BUFFER;
}

export function buildCloudAsrRequestHeaders(provider: CloudAsrProvider, key?: string): Record<string, string> {
  const trimmed = String(key || '').trim();
  if (!trimmed) return {};
  if (provider === 'elevenlabs-scribe') {
    return { 'xi-api-key': trimmed };
  }
  if (provider === 'deepgram-listen') {
    return { Authorization: `Token ${trimmed}` };
  }
  if (provider === 'gladia-pre-recorded') {
    return { 'x-gladia-key': trimmed };
  }
  if (provider === 'github-models-phi4-multimodal') {
    return {
      Accept: 'application/vnd.github+json',
      Authorization: `Bearer ${trimmed}`,
      'X-GitHub-Api-Version': '2026-03-10',
    };
  }
  if (provider === 'google-cloud-chirp3') {
    return { Authorization: `Bearer ${trimmed}` };
  }
  if (provider === 'google-gemini-audio') {
    return { 'x-goog-api-key': trimmed };
  }
  return { Authorization: `Bearer ${trimmed}` };
}

function toFiniteNumber(value: unknown, fallback = Number.NaN) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function getEnvNumber(name: string, fallback: number, min = Number.NEGATIVE_INFINITY, max = Number.POSITIVE_INFINITY) {
  const parsed = Number(process.env[name]);
  const value = Number.isFinite(parsed) ? parsed : fallback;
  return Math.max(min, Math.min(max, value));
}

function getCloudAsrRequestTimeoutMs(resolvedProvider: ResolvedCloudAsrProvider, options: CloudAsrAdapterRequestOptions) {
  const sharedTimeoutMs = Math.round(getEnvNumber('ASR_CLOUD_REQUEST_TIMEOUT_MS', 120000, 15000, 1800000));
  if (resolvedProvider.provider === 'deepgram-listen') {
    const deepgramFloorMs = Math.round(
      getEnvNumber('ASR_DEEPGRAM_REQUEST_TIMEOUT_MS', 600000, 60000, 1200000)
    );
    return Math.max(sharedTimeoutMs, deepgramFloorMs);
  }
  if (resolvedProvider.provider === 'gladia-pre-recorded') {
    const gladiaFloorMs = Math.round(
      getEnvNumber('ASR_GLADIA_REQUEST_TIMEOUT_MS', 900000, 60000, 7200000)
    );
    return Math.max(sharedTimeoutMs, gladiaFloorMs);
  }

  if (resolvedProvider.provider !== 'elevenlabs-scribe') {
    return sharedTimeoutMs;
  }

  const audioDurationSec = toFiniteNumber(options.audioDurationSec, Number.NaN);
  const estimatedTimeoutMs = Number.isFinite(audioDurationSec) && audioDurationSec > 0
    ? Math.ceil(audioDurationSec * 1000 * 2.2 + 120000)
    : 1800000;
  const elevenLabsFloorMs = Math.round(
    getEnvNumber('ASR_ELEVENLABS_REQUEST_TIMEOUT_MS', 1800000, 60000, 7200000)
  );
  return Math.round(Math.max(sharedTimeoutMs, elevenLabsFloorMs, Math.min(estimatedTimeoutMs, 7200000)));
}

function normalizeElevenLabsLanguageCode(language: string) {
  const normalized = String(language || '').trim().toLowerCase().replace(/_/g, '-');
  if (!normalized || normalized === 'auto' || normalized === 'detect') return '';
  if (normalized.startsWith('zh-')) return 'zh';
  if (normalized === 'jp') return 'ja';
  const primary = normalized.split('-')[0];
  return /^[a-z]{2,3}$/.test(primary) ? primary : '';
}

function getElevenLabsNumSpeakers(options: CloudAsrAdapterRequestOptions) {
  if (!options.diarization) return null;
  const diarizationOptions = options.diarizationOptions || {};
  const mode = String(diarizationOptions.mode || '').trim();
  const exactSpeakerCount = toFiniteNumber(diarizationOptions.exactSpeakerCount, Number.NaN);
  const maxSpeakers = toFiniteNumber(diarizationOptions.maxSpeakers, Number.NaN);
  const value =
    mode === 'fixed' && Number.isFinite(exactSpeakerCount) && exactSpeakerCount > 0
      ? exactSpeakerCount
      : (mode === 'range' || mode === 'many') && Number.isFinite(maxSpeakers) && maxSpeakers > 0
        ? maxSpeakers
        : Number.NaN;

  if (!Number.isFinite(value)) return null;
  return Math.max(1, Math.min(32, Math.round(value)));
}

function normalizeElevenLabsSpeaker(value: unknown) {
  const raw = String(value ?? '').trim();
  if (!raw) return undefined;
  const matched = raw.match(/^speaker[\s_-]*(\d+)$/i);
  return matched ? `spk${matched[1]}` : raw;
}

function isElevenLabsNonSpeechWord(raw: any) {
  const type = String(raw?.type || '').trim().toLowerCase();
  return type === 'spacing' || type === 'audio_event';
}

function isElevenLabsBoundaryText(text: string) {
  const value = String(text || '').trim();
  if (!value) return false;
  return /^[.?!,;:\u3001\u3002\uff0c\uff0e\uff01\uff1f\uff1a\uff1b]/u.test(value) ||
    /[.?!,;:\u3001\u3002\uff0c\uff0e\uff01\uff1f\uff1a\uff1b]$/u.test(value);
}

function normalizeElevenLabsWord(raw: any) {
  const text = String(raw?.text ?? raw?.word ?? '').trim();
  if (!text || isElevenLabsNonSpeechWord(raw)) return null;
  const start = toFiniteNumber(raw?.start);
  if (!Number.isFinite(start)) return null;
  const endRaw = toFiniteNumber(raw?.end);
  const end = Number.isFinite(endRaw) && endRaw > start ? endRaw : undefined;
  const speaker = normalizeElevenLabsSpeaker(raw?.speaker_id) || normalizeElevenLabsSpeaker(raw?.speaker);
  return {
    ...raw,
    text,
    start,
    end,
    speaker,
  };
}

function joinElevenLabsWords(words: Array<{ text: string }>) {
  let output = '';
  for (const word of words) {
    const text = String(word?.text || '').trim();
    if (!text) continue;
    if (!output) {
      output = text;
      continue;
    }
    if (/^[,.;:!?%)]$/.test(text) || /^[\u3001\u3002\uff0c\uff0e\uff01\uff1f\uff1a\uff1b\uff09\u3011\u300f\u300d]$/u.test(text)) {
      output += text;
      continue;
    }
    if (/[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]$/u.test(output) || /^[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]/u.test(text)) {
      output += text;
      continue;
    }
    output += ` ${text}`;
  }
  return output.trim();
}

function inferAudioFormat(filePath: string) {
  const ext = path.extname(filePath).replace(/^\./, '').toLowerCase();
  if (['mp3', 'wav', 'm4a', 'flac', 'ogg', 'webm'].includes(ext)) return ext;
  return 'mp3';
}

function inferAudioMimeType(filePath: string) {
  const ext = path.extname(filePath).replace(/^\./, '').toLowerCase();
  const mimeByExt: Record<string, string> = {
    aac: 'audio/aac',
    aiff: 'audio/aiff',
    flac: 'audio/flac',
    m4a: 'audio/m4a',
    mp3: 'audio/mp3',
    mp4: 'audio/mp4',
    mpeg: 'audio/mpeg',
    mpg: 'audio/mpeg',
    oga: 'audio/ogg',
    ogg: 'audio/ogg',
    opus: 'audio/ogg',
    pcm: 'audio/pcm',
    wav: 'audio/wav',
    webm: 'audio/webm',
  };
  return mimeByExt[ext] || 'audio/wav';
}

function normalizeDeepgramLanguageCode(language: string) {
  const normalized = String(language || '').trim().toLowerCase().replace(/_/g, '-');
  if (!normalized || normalized === 'auto' || normalized === 'detect') return '';
  if (normalized === 'jp') return 'ja';
  const primary = normalized.split('-')[0];
  return /^[a-z]{2,3}$/.test(primary) ? primary : normalized;
}

function getDeepgramUtteranceSplitSec() {
  const configured = Number(process.env.ASR_DEEPGRAM_UTT_SPLIT_SEC);
  if (Number.isFinite(configured) && configured > 0) {
    return Math.max(0.1, Math.min(5, configured));
  }
  const fallbackSilenceMs = getEnvNumber(
    'VAD_FIXED_SILENCE_MS',
    getEnvNumber('VAD_MIN_SILENCE_MS', 500, 50, 5000),
    50,
    5000
  );
  return Number(Math.max(0.1, Math.min(5, fallbackSilenceMs / 1000)).toFixed(3));
}

function buildDeepgramListenUrl(endpointUrl: string, input: CloudAsrAdapterBuildFormInput) {
  const { config, includeLanguage, options } = input;
  const next = new URL(endpointUrl);
  const setDefault = (key: string, value: string) => {
    if (!next.searchParams.has(key)) next.searchParams.set(key, value);
  };

  const configuredModel = String(config?.model || '').trim();
  if (configuredModel) {
    next.searchParams.set('model', configuredModel);
  } else {
    setDefault('model', 'nova-3');
  }

  setDefault('smart_format', 'true');
  setDefault('punctuate', 'true');
  setDefault('paragraphs', 'true');

  const wantsCloudUtterances = options.segmentation !== false || Boolean(options.vad) || Boolean(options.diarization);
  if (wantsCloudUtterances) {
    next.searchParams.set('utterances', 'true');
    if (!next.searchParams.has('utt_split')) {
      next.searchParams.set('utt_split', String(getDeepgramUtteranceSplitSec()));
    }
  }
  if (options.diarization) {
    next.searchParams.set('diarize', 'true');
  }

  const rawLanguage = typeof options.language === 'string' ? options.language.trim() : '';
  const normalizedLanguage = normalizeDeepgramLanguageCode(rawLanguage);
  if (includeLanguage && normalizedLanguage) {
    next.searchParams.set('language', normalizedLanguage);
  } else if (includeLanguage && rawLanguage.toLowerCase() === 'auto') {
    setDefault('detect_language', 'true');
  }

  return next.toString();
}

function buildGithubPhi4AsrPrompt(options: CloudAsrAdapterRequestOptions) {
  const parts = [
    'Transcribe the audio exactly as spoken.',
    'Return only the transcript text. Do not summarize, translate, explain, add timestamps, or add labels.',
  ];
  const rawLanguage = typeof options.language === 'string' ? options.language.trim() : '';
  if (rawLanguage && rawLanguage.toLowerCase() !== 'auto') {
    parts.push(`The expected spoken language is ${rawLanguage}.`);
  }
  if (options.prompt && options.prompt.trim()) {
    parts.push(`Vocabulary and context: ${options.prompt.trim()}`);
  }
  return parts.join(' ');
}

function getGithubPhi4MaxOutputTokens() {
  const parsed = Number(process.env.ASR_PHI4_MAX_OUTPUT_TOKENS || 512);
  if (!Number.isFinite(parsed)) return 512;
  return Math.max(128, Math.min(2048, Math.round(parsed)));
}

function buildGeminiAudioAsrPrompt(options: CloudAsrAdapterRequestOptions) {
  const parts = [
    'You are a specialized ASR data processor. Your only task is to generate a verbatim speech transcript for this audio.',
    'Do not summarize, translate, rewrite, explain, or add commentary.',
    'Preserve the original spoken language.',
    'Preserve normal word spacing for Latin-script text.',
    'Return JSON only. Segment by natural speech breaks or sentence boundaries.',
    'Use integer milliseconds from the beginning of the audio for start_ms and end_ms when possible.',
    'Do not use decimal fractions for timestamps.',
    'If timestamps are uncertain, still return the transcript text.',
  ];
  const rawLanguage = typeof options.language === 'string' ? options.language.trim() : '';
  if (rawLanguage && rawLanguage.toLowerCase() !== 'auto') {
    parts.push(`The expected spoken language is ${rawLanguage}.`);
  }
  if (options.prompt && options.prompt.trim()) {
    parts.push(`Vocabulary and context: ${options.prompt.trim()}`);
  }
  return parts.join(' ');
}

function buildGeminiAudioBatchAsrPrompt(options: CloudAsrAdapterRequestOptions, inputCount: number) {
  const parts = [
    'You are a specialized ASR data processor. Your only task is to process a batch of audio files.',
    `You are processing exactly ${inputCount} separate audio files in the exact physical order provided in this request.`,
    'Highest-priority rule: strict 1:1 atomic mapping.',
    'Return JSON only.',
    `Return exactly ${inputCount} items in the items array.`,
    'Every single audio file MUST correspond to exactly one item.',
    'Every item MUST include an index field.',
    'Use zero-based integer index values: the first audio file is index 0, the second is index 1, and so on.',
    `The items array MUST contain every index from 0 through ${Math.max(0, inputCount - 1)} exactly once, in ascending order.`,
    'Each item must contain the complete verbatim transcript for that one audio file in text.',
    'Do not split one audio file into multiple items.',
    'Do not merge multiple audio files into one item.',
    'Do not put speech from one audio file into the item for another audio file.',
    'Do not summarize, translate, rewrite, explain, or add commentary.',
    'Preserve the original spoken language for every audio file.',
    'Preserve normal word spacing for Latin-script text.',
    'If an audio file contains no speech, return an empty string for its text.',
    `Before outputting, count the input audio files and count the items. If the count is not exactly ${inputCount}, discard the draft and regenerate until the count matches.`,
  ];
  if (options.diarization) {
    parts.push('When a speaker can be identified, include a stable speaker label such as spk0, spk1, and reuse it across files for the same speaker.');
  }
  const rawLanguage = typeof options.language === 'string' ? options.language.trim() : '';
  if (rawLanguage && rawLanguage.toLowerCase() !== 'auto') {
    parts.push(`The expected spoken language is ${rawLanguage}.`);
  }
  if (options.prompt && options.prompt.trim()) {
    parts.push(`Vocabulary and context: ${options.prompt.trim()}`);
  }
  return parts.join(' ');
}

function firstString(candidates: unknown[]) {
  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.trim()) return candidate.trim();
  }
  return '';
}

function stripCodeFence(value: string) {
  const trimmed = String(value || '').trim();
  const matched = trimmed.match(/^```(?:json|text)?\s*([\s\S]*?)\s*```$/i);
  return matched ? matched[1].trim() : trimmed;
}

function normalizeGithubPhi4TranscriptText(raw: string) {
  let text = stripCodeFence(raw);
  try {
    const parsed = JSON.parse(text);
    text = firstString([
      parsed?.text,
      parsed?.transcript,
      parsed?.transcription,
      parsed?.result,
      Array.isArray(parsed?.segments)
        ? parsed.segments.map((segment: any) => firstString([segment?.text, segment?.transcript])).filter(Boolean).join('\n')
        : '',
    ]) || text;
  } catch {
    // Keep plain text responses.
  }
  return String(text || '')
    .replace(/^\s*(transcript|transcription|text|output)\s*[:：]\s*/i, '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .join('\n')
    .trim();
}

function normalizeGithubPhi4Transcript(data: any) {
  const rawContent = firstString([
    data?.choices?.[0]?.message?.content,
    data?.choices?.[0]?.delta?.content,
    data?.output_text,
    data?.text,
    data?.content,
  ]);
  return {
    ...data,
    text: normalizeGithubPhi4TranscriptText(rawContent),
  };
}

const GOOGLE_CHIRP_LANGUAGE_CODE_ALIASES: Record<string, string> = {
  auto: 'auto',
  en: 'en-US',
  ja: 'ja-JP',
  zh: 'cmn-Hans-CN',
  'zh-cn': 'cmn-Hans-CN',
  'zh-hans': 'cmn-Hans-CN',
  'zh-tw': 'cmn-Hant-TW',
  'zh-hant': 'cmn-Hant-TW',
  yue: 'yue-Hant-HK',
  ko: 'ko-KR',
  fr: 'fr-FR',
  de: 'de-DE',
  es: 'es-ES',
  it: 'it-IT',
  pt: 'pt-BR',
  'pt-br': 'pt-BR',
  fi: 'fi-FI',
};

function resolveGoogleCloudLanguageCodes(language?: string) {
  const raw = String(language || '').trim();
  if (!raw) return ['auto'];
  const parts = raw
    .split(/[,;]/)
    .map((part) => part.trim())
    .filter(Boolean);
  if (parts.length === 0) return ['auto'];
  return parts.map((part) => GOOGLE_CHIRP_LANGUAGE_CODE_ALIASES[part.toLowerCase()] || part);
}

function parseGoogleDurationSeconds(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const matched = value.trim().match(/^(-?\d+(?:\.\d+)?)s$/);
    if (!matched) return undefined;
    const parsed = Number(matched[1]);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  if (value && typeof value === 'object') {
    const raw = value as { seconds?: unknown; nanos?: unknown };
    const seconds = Number(raw.seconds || 0);
    const nanos = Number(raw.nanos || 0);
    if (Number.isFinite(seconds) && Number.isFinite(nanos)) {
      return seconds + nanos / 1_000_000_000;
    }
  }
  return undefined;
}

function normalizeGoogleCloudChirp3Word(raw: any) {
  const text = String(raw?.word ?? raw?.text ?? '').trim();
  if (!text) return null;
  const start = parseGoogleDurationSeconds(raw?.startOffset ?? raw?.start ?? raw?.start_ts);
  if (start == null || !Number.isFinite(start)) return null;
  const end = parseGoogleDurationSeconds(raw?.endOffset ?? raw?.end ?? raw?.end_ts);
  const probability = toFiniteNumber(raw?.confidence, Number.NaN);
  return {
    ...raw,
    word: text,
    text,
    start,
    start_ts: start,
    end: end != null && end > start ? end : undefined,
    end_ts: end != null && end > start ? end : undefined,
    probability: Number.isFinite(probability) ? probability : undefined,
  };
}

function normalizeGoogleCloudChirp3Transcript(data: any) {
  const results = Array.isArray(data?.results) ? data.results : [];
  const segments: any[] = [];
  const texts: string[] = [];
  let previousEnd = 0;
  let detectedLanguage = '';

  for (const result of results) {
    const alternative = Array.isArray(result?.alternatives) ? result.alternatives[0] : null;
    const text = String(alternative?.transcript || '').trim();
    if (!text) continue;
    const rawWords = Array.isArray(alternative?.words) ? alternative.words : [];
    const words = rawWords
      .map((word: any) => normalizeGoogleCloudChirp3Word(word))
      .filter(Boolean);
    const firstWord = words[0];
    const lastWord = words[words.length - 1];
    const start = Number.isFinite(Number(firstWord?.start)) ? Number(firstWord.start) : previousEnd;
    const resultEnd = parseGoogleDurationSeconds(result?.resultEndOffset);
    const wordEnd = Number.isFinite(Number(lastWord?.end)) ? Number(lastWord.end) : undefined;
    const end = wordEnd != null
      ? wordEnd
      : resultEnd != null && resultEnd > start
        ? resultEnd
        : undefined;
    const languageCode = typeof result?.languageCode === 'string' ? result.languageCode.trim() : '';
    if (!detectedLanguage && languageCode) detectedLanguage = languageCode;
    texts.push(text);
    segments.push({
      text,
      transcript: text,
      start,
      start_ts: start,
      end,
      end_ts: end,
      language_code: languageCode || undefined,
      confidence: Number.isFinite(Number(alternative?.confidence)) ? Number(alternative.confidence) : undefined,
      words: words.length > 0 ? words : undefined,
    });
    previousEnd = end != null ? Math.max(previousEnd, end) : previousEnd;
  }

  return {
    ...data,
    text: texts.join('\n').trim() || data?.text,
    language_code: detectedLanguage || data?.language_code,
    segments: segments.length > 0 ? segments : data?.segments,
  };
}

function parseFlexibleTimestampSeconds(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  if (!trimmed) return undefined;
  const secondsSuffix = trimmed.match(/^(\d+(?:\.\d+)?)\s*s$/i);
  if (secondsSuffix) {
    const parsed = Number(secondsSuffix[1]);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  if (/^\d+(?:\.\d+)?$/.test(trimmed)) {
    const parsed = Number(trimmed);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  const timeMatch = trimmed.match(/(\d{1,2})(?::(\d{2}))(?::(\d{2}(?:\.\d+)?))?/);
  if (!timeMatch) return undefined;
  const first = Number(timeMatch[1]);
  const second = Number(timeMatch[2]);
  const third = timeMatch[3] != null ? Number(timeMatch[3]) : undefined;
  if (!Number.isFinite(first) || !Number.isFinite(second)) return undefined;
  if (third == null) return first * 60 + second;
  if (!Number.isFinite(third)) return undefined;
  return first * 3600 + second * 60 + third;
}

function parseGeminiMillisecondsSeconds(value: unknown): number | undefined {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) return undefined;
  return parsed / 1000;
}

function parseGeminiAudioSegmentRange(segment: any) {
  const start =
    parseGeminiMillisecondsSeconds(segment?.start_ms ?? segment?.timestamp?.start_ms) ??
    parseFlexibleTimestampSeconds(segment?.start) ??
    parseFlexibleTimestampSeconds(segment?.start_ts) ??
    parseFlexibleTimestampSeconds(segment?.timestamp_start);
  const end =
    parseGeminiMillisecondsSeconds(segment?.end_ms ?? segment?.timestamp?.end_ms) ??
    parseFlexibleTimestampSeconds(segment?.end) ??
    parseFlexibleTimestampSeconds(segment?.end_ts) ??
    parseFlexibleTimestampSeconds(segment?.timestamp_end);

  if ((start != null || end != null) || typeof segment?.timestamp !== 'string') {
    return { start, end };
  }

  const timestamp = String(segment.timestamp || '');
  const rangeParts = timestamp.split(/\s*(?:-|to|-->|~|–|—)\s*/i).filter(Boolean);
  if (rangeParts.length >= 2) {
    return {
      start: parseFlexibleTimestampSeconds(rangeParts[0]),
      end: parseFlexibleTimestampSeconds(rangeParts[1]),
    };
  }

  return { start: parseFlexibleTimestampSeconds(timestamp), end: undefined };
}

function parseGeminiAudioJsonPayload(rawText: string) {
  const cleaned = stripCodeFence(rawText);
  try {
    return JSON.parse(cleaned);
  } catch {
    const objectMatch = cleaned.match(/\{[\s\S]*\}/);
    if (objectMatch) {
      try {
        return JSON.parse(objectMatch[0]);
      } catch {
        // Fall through to plain text.
      }
    }
  }
  return null;
}

function getGeminiAudioResponseText(data: any) {
  return firstString([
    data?.candidates?.[0]?.content?.parts?.map((part: any) => (typeof part?.text === 'string' ? part.text : '')).join(''),
    data?.text,
    data?.content,
  ]);
}

function isJsonLikeGeminiText(rawText: string) {
  return /^[\s`]*(?:json\s*)?[\[{]/i.test(stripCodeFence(rawText));
}

function decodeJsonStringLiteralBody(value: string) {
  try {
    return JSON.parse(`"${value}"`);
  } catch {
    return value
      .replace(/\\"/g, '"')
      .replace(/\\n/g, '\n')
      .replace(/\\r/g, '\r')
      .replace(/\\t/g, '\t');
  }
}

function salvageGeminiAudioSegmentsFromInvalidJson(rawText: string) {
  const cleaned = stripCodeFence(rawText);
  if (!isJsonLikeGeminiText(cleaned)) return [];

  const segments: any[] = [];
  const textFieldPattern = /"(?:text|transcript|sentence)"\s*:\s*"((?:\\.|[^"\\])*)"/g;
  let match: RegExpExecArray | null;
  while ((match = textFieldPattern.exec(cleaned)) !== null) {
    const text = String(decodeJsonStringLiteralBody(match[1]) || '').trim();
    if (!text || /^[\[{]\s*$/.test(text)) continue;
    segments.push({
      text,
      transcript: text,
    });
  }

  return segments;
}

function normalizeGeminiAudioWord(raw: any) {
  const text = String(raw?.text ?? raw?.word ?? raw?.token ?? '').trim();
  if (!text) return null;
  const start = parseFlexibleTimestampSeconds(raw?.start ?? raw?.start_ts ?? raw?.timestamp?.start);
  if (start == null || !Number.isFinite(start)) return null;
  const end = parseFlexibleTimestampSeconds(raw?.end ?? raw?.end_ts ?? raw?.timestamp?.end);
  return {
    ...raw,
    word: text,
    text,
    start,
    start_ts: start,
    end: end != null && end > start ? end : undefined,
    end_ts: end != null && end > start ? end : undefined,
  };
}

function normalizeGeminiAudioTranscript(data: any) {
  const rawContent = getGeminiAudioResponseText(data);
  const payload = parseGeminiAudioJsonPayload(rawContent);
  const source = payload && typeof payload === 'object' ? payload : {};
  const rawSegments = Array.isArray((source as any)?.segments)
    ? (source as any).segments
    : Array.isArray((source as any)?.chunks)
      ? (source as any).chunks
      : salvageGeminiAudioSegmentsFromInvalidJson(rawContent);
  const segments: any[] = [];
  const textLines: string[] = [];
  let previousEnd = 0;

  for (const [index, segment] of rawSegments.entries()) {
    const text = firstString([
      segment?.text,
      segment?.content,
      segment?.transcript,
      segment?.sentence,
    ]);
    if (!text) continue;
    const range = parseGeminiAudioSegmentRange(segment);
    const start = range.start != null && Number.isFinite(range.start) ? range.start : previousEnd;
    const end = range.end != null && Number.isFinite(range.end) && range.end > start ? range.end : undefined;
    const words = Array.isArray(segment?.words)
      ? segment.words.map((word: any) => normalizeGeminiAudioWord(word)).filter(Boolean)
      : [];
    textLines.push(text);
    segments.push({
      ...segment,
      text,
      transcript: text,
      start,
      start_ts: start,
      end,
      end_ts: end,
      speaker: typeof segment?.speaker === 'string' && segment.speaker.trim()
        ? segment.speaker.trim()
        : typeof segment?.speaker_id === 'string' && segment.speaker_id.trim()
          ? segment.speaker_id.trim()
          : undefined,
      words: words.length > 0 ? words : undefined,
      source_segment_index: index,
    });
    previousEnd = end != null ? Math.max(previousEnd, end) : previousEnd;
  }

  const text = firstString([
    (source as any)?.text,
    (source as any)?.transcript,
    (source as any)?.transcription,
    textLines.join('\n'),
    isJsonLikeGeminiText(rawContent) ? '' : rawContent,
  ]);

  return {
    ...data,
    text,
    language_code: firstString([(source as any)?.language_code, (source as any)?.languageCode, (source as any)?.language]) || data?.language_code,
    segments: segments.length > 0 ? segments : data?.segments,
    gemini_audio_payload: payload || undefined,
    gemini_audio_payload_recovered: !payload && segments.length > 0,
  };
}

function extractGeminiAudioBatchRawItems(data: any, expectedCount: number) {
  const rawContent = getGeminiAudioResponseText(data);
  const payload = parseGeminiAudioJsonPayload(rawContent);
  const source = payload && typeof payload === 'object' ? payload as any : {};
  const jsonItems = Array.isArray(source?.items)
    ? source.items
    : Array.isArray(source?.audio_texts)
      ? source.audio_texts
      : Array.isArray(source?.audioTexts)
        ? source.audioTexts
        : Array.isArray(source?.segments)
          ? source.segments
          : Array.isArray(source?.results)
            ? source.results
            : [];

  if (jsonItems.length > 0) {
    return {
      rawContent,
      payload,
      items: jsonItems,
      recoveredFromXml: false,
    };
  }

  const xmlItems = Array.from(rawContent.matchAll(/<audio_text\b[^>]*>([\s\S]*?)<\/audio_text>/gi)).map((match) => ({
    text: String(match[1] || '').trim(),
  }));
  if (xmlItems.length > 0) {
    return {
      rawContent,
      payload,
      items: xmlItems.slice(0, expectedCount),
      recoveredFromXml: true,
    };
  }

  return {
    rawContent,
    payload,
    items: [],
    recoveredFromXml: false,
  };
}

function getGeminiBatchItemIndex(raw: any) {
  const candidates = [raw?.index, raw?.audio_index, raw?.audioIndex, raw?.file_index, raw?.fileIndex, raw?.id];
  for (const candidate of candidates) {
    const parsed = Number(candidate);
    if (Number.isInteger(parsed)) return parsed;
  }
  return null;
}

function getGeminiBatchItemText(raw: any) {
  return firstString([
    raw?.text,
    raw?.transcript,
    raw?.transcription,
    raw?.content,
    raw?.sentence,
    typeof raw === 'string' ? raw : '',
  ]);
}

function normalizeGeminiAudioBatchItems(data: any, expectedCount: number) {
  const extracted = extractGeminiAudioBatchRawItems(data, expectedCount);
  const used = new Set<number>();

  const takeByPredicate = (predicate: (raw: any, position: number) => boolean) => {
    for (let position = 0; position < extracted.items.length; position += 1) {
      if (used.has(position)) continue;
      const raw = extracted.items[position];
      if (predicate(raw, position)) {
        used.add(position);
        return { raw, position };
      }
    }
    return null;
  };

  const normalized = Array.from({ length: expectedCount }, (_, index) => {
    const matched =
      takeByPredicate((raw) => getGeminiBatchItemIndex(raw) === index) ||
      takeByPredicate((raw) => getGeminiBatchItemIndex(raw) === index + 1) ||
      takeByPredicate((_raw, position) => position === index);
    const raw = matched?.raw || {};
    const text = getGeminiBatchItemText(raw);
    return {
      index,
      text,
      speaker: firstString([raw?.speaker, raw?.speaker_id, raw?.speakerId]),
      language_code: firstString([raw?.language_code, raw?.languageCode, raw?.language]),
      raw,
      missing: !matched,
    };
  });

  return {
    ...extracted,
    normalized,
    missingCount: normalized.filter((item) => item.missing).length,
  };
}

function buildGeminiBatchTranscriptFromItem(
  item: { text: string; speaker?: string },
  durationSec: number | null | undefined
): CloudAsrStructuredTranscript {
  const text = String(item.text || '').trim();
  const endTs = Math.max(0.1, Number.isFinite(Number(durationSec)) && Number(durationSec) > 0 ? Number(durationSec) : 0.1);
  if (!text) {
    return {
      text: '',
      chunks: [],
      segments: [],
      word_segments: [],
    };
  }
  const chunk = {
    text,
    start_ts: 0,
    end_ts: Number(endTs.toFixed(3)),
    speaker: item.speaker || undefined,
  };
  return {
    text,
    chunks: [chunk],
    segments: [chunk],
    word_segments: [],
  };
}

function normalizeDeepgramSpeaker(value: unknown) {
  if (typeof value === 'number' && Number.isFinite(value)) return `spk${Math.max(0, Math.round(value))}`;
  const raw = String(value ?? '').trim();
  if (!raw) return undefined;
  const numeric = raw.match(/^(?:speaker[\s_-]*)?(\d+)$/i);
  if (numeric) return `spk${Number(numeric[1])}`;
  return raw;
}

function normalizeDeepgramWord(raw: any) {
  const text = String(raw?.punctuated_word ?? raw?.word ?? raw?.text ?? '').trim();
  if (!text) return null;
  const start = toFiniteNumber(raw?.start ?? raw?.start_ts, Number.NaN);
  if (!Number.isFinite(start)) return null;
  const endRaw = toFiniteNumber(raw?.end ?? raw?.end_ts, Number.NaN);
  const end = Number.isFinite(endRaw) && endRaw > start ? endRaw : undefined;
  const probability = toFiniteNumber(raw?.confidence ?? raw?.probability, Number.NaN);
  const speaker = normalizeDeepgramSpeaker(raw?.speaker ?? raw?.speaker_id);
  return {
    ...raw,
    word: text,
    text,
    start,
    start_ts: start,
    end,
    end_ts: end,
    speaker,
    speaker_id: speaker,
    probability: Number.isFinite(probability) ? probability : undefined,
  };
}

function getDeepgramAlternativeItems(data: any) {
  const channels = Array.isArray(data?.results?.channels) ? data.results.channels : [];
  return channels.flatMap((channel: any) =>
    (Array.isArray(channel?.alternatives) ? channel.alternatives : []).map((alternative: any) => ({
      channel,
      alternative,
    }))
  );
}

function buildDeepgramSegmentsFromWords(words: ReturnType<typeof normalizeDeepgramWord>[]) {
  const normalizedWords = words.filter(Boolean) as Array<NonNullable<ReturnType<typeof normalizeDeepgramWord>>>;
  const segments: any[] = [];
  let buffer: typeof normalizedWords = [];
  const utteranceSplitSec = getDeepgramUtteranceSplitSec();

  const flush = () => {
    if (buffer.length === 0) return;
    const first = buffer[0];
    const last = buffer[buffer.length - 1];
    segments.push({
      text: joinElevenLabsWords(buffer),
      transcript: joinElevenLabsWords(buffer),
      start: first.start,
      start_ts: first.start,
      end: last.end,
      end_ts: last.end,
      speaker: first.speaker,
      words: buffer,
    });
    buffer = [];
  };

  for (const word of normalizedWords) {
    const previous = buffer[buffer.length - 1];
    const gap = previous?.end != null ? word.start - previous.end : 0;
    const speakerChanged = Boolean(previous?.speaker && word.speaker && previous.speaker !== word.speaker);
    if (buffer.length > 0 && (speakerChanged || gap >= utteranceSplitSec)) {
      flush();
    }
    buffer.push(word);

    const first = buffer[0];
    const end = word.end ?? word.start;
    const duration = Math.max(0, end - first.start);
    const currentText = joinElevenLabsWords(buffer);
    if (
      buffer.length > 0 &&
      (
        (duration >= 0.5 && isElevenLabsBoundaryText(word.text)) ||
        duration >= 8 ||
        currentText.length >= 90
      )
    ) {
      flush();
    }
  }
  flush();
  return segments;
}

function normalizeDeepgramTranscript(data: any) {
  const rawUtterances = Array.isArray(data?.results?.utterances) ? data.results.utterances : [];
  const alternativeItems = getDeepgramAlternativeItems(data);
  const transcriptTexts: string[] = [];
  const speakerIds = new Set<string>();
  let normalizedWordCount = 0;
  let detectedLanguage = firstString([
    data?.language_code,
    data?.metadata?.detected_language,
    alternativeItems.find((item: any) => typeof item?.channel?.detected_language === 'string')?.channel?.detected_language,
  ]);

  const segments = rawUtterances
    .map((utterance: any) => {
      const text = firstString([utterance?.transcript, utterance?.text]);
      if (!text) return null;
      const words = Array.isArray(utterance?.words)
        ? utterance.words.map((word: any) => normalizeDeepgramWord(word)).filter(Boolean)
        : [];
      normalizedWordCount += words.length;
      for (const word of words) {
        if (word?.speaker) speakerIds.add(word.speaker);
      }
      const speaker = normalizeDeepgramSpeaker(utterance?.speaker);
      if (speaker) speakerIds.add(speaker);
      transcriptTexts.push(text);
      return {
        ...utterance,
        text,
        transcript: text,
        start: toFiniteNumber(utterance?.start, 0),
        start_ts: toFiniteNumber(utterance?.start, 0),
        end: Number.isFinite(Number(utterance?.end)) ? toFiniteNumber(utterance?.end, 0) : undefined,
        end_ts: Number.isFinite(Number(utterance?.end)) ? toFiniteNumber(utterance?.end, 0) : undefined,
        speaker,
        words: words.length > 0 ? words : undefined,
      };
    })
    .filter(Boolean);

  if (segments.length === 0) {
    for (const { alternative, channel } of alternativeItems) {
      const text = firstString([alternative?.transcript, alternative?.text]);
      if (text) transcriptTexts.push(text);
      if (!detectedLanguage && typeof channel?.detected_language === 'string') {
        detectedLanguage = channel.detected_language.trim();
      }
      const words = Array.isArray(alternative?.words)
        ? alternative.words.map((word: any) => normalizeDeepgramWord(word)).filter(Boolean)
        : [];
      normalizedWordCount += words.length;
      for (const word of words) {
        if (word?.speaker) speakerIds.add(word.speaker);
      }
      segments.push(...buildDeepgramSegmentsFromWords(words));
    }
  }

  return {
    ...data,
    text: transcriptTexts.join('\n').trim() || data?.text,
    language_code: detectedLanguage || data?.language_code,
    segments: segments.length > 0 ? segments.sort((a: any, b: any) => toFiniteNumber(a.start, 0) - toFiniteNumber(b.start, 0)) : data?.segments,
    deepgram_diarization_applied: speakerIds.size > 0,
    deepgram_speaker_count: speakerIds.size,
    deepgram_word_count: normalizedWordCount,
    deepgram_utterance_count: rawUtterances.length,
  };
}

function normalizeGladiaLanguageCode(language: string) {
  const normalized = String(language || '').trim().toLowerCase().replace(/_/g, '-');
  if (!normalized || normalized === 'auto' || normalized === 'detect') return '';
  if (normalized === 'jp') return 'ja';
  const primary = normalized.split('-')[0];
  return /^[a-z]{2,3}$/.test(primary) ? primary : normalized;
}

function normalizeGladiaSpeaker(value: unknown) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return `spk${Math.max(0, Math.round(value))}`;
  }
  const raw = String(value ?? '').trim();
  if (!raw) return undefined;
  const numeric = raw.match(/^(?:speaker[\s_-]*)?(\d+)$/i);
  if (numeric) return `spk${Math.max(0, Number(numeric[1]))}`;
  return raw;
}

function normalizeGladiaWord(raw: any, fallbackSpeaker?: string) {
  const text = String(raw?.word ?? raw?.text ?? '').trim();
  if (!text) return null;
  const start = toFiniteNumber(raw?.start ?? raw?.start_ts, Number.NaN);
  if (!Number.isFinite(start)) return null;
  const endRaw = toFiniteNumber(raw?.end ?? raw?.end_ts, Number.NaN);
  const end = Number.isFinite(endRaw) && endRaw > start ? endRaw : undefined;
  const probability = toFiniteNumber(raw?.confidence ?? raw?.probability, Number.NaN);
  const speaker = normalizeGladiaSpeaker(raw?.speaker ?? raw?.speaker_id) || fallbackSpeaker;
  return {
    ...raw,
    word: text,
    text,
    start,
    start_ts: start,
    end,
    end_ts: end,
    speaker,
    speaker_id: speaker,
    probability: Number.isFinite(probability) ? probability : undefined,
  };
}

function parseSrtTimestampSeconds(raw: string) {
  const matched = String(raw || '').trim().match(/^(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})$/);
  if (!matched) return Number.NaN;
  const [, hours, minutes, seconds, millis] = matched;
  return (
    Number(hours) * 3600 +
    Number(minutes) * 60 +
    Number(seconds) +
    Number(millis.padEnd(3, '0').slice(0, 3)) / 1000
  );
}

function normalizeNoSpaceScriptDisplayText(text: string) {
  return String(text || '')
    .replace(/([\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af])\s+([\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af])/gu, '$1$2')
    .replace(/\s+([,.;:!?%\u3001\u3002\uff0c\uff0e\uff01\uff1f\uff1a\uff1b])/gu, '$1')
    .trim();
}

function parseGladiaSrtSegments(srtText: string, language?: string) {
  const blocks = String(srtText || '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean);
  const noSpaceLanguage = /^(zh|ja|jp|ko)(?:-|$)/i.test(String(language || ''));
  const segments: any[] = [];
  for (const block of blocks) {
    const lines = block.split('\n').map((line) => line.trim()).filter(Boolean);
    const timeIndex = lines.findIndex((line) => line.includes('-->'));
    if (timeIndex < 0) continue;
    const [rawStart, rawEnd] = lines[timeIndex].split('-->').map((part) => part.trim());
    const start = parseSrtTimestampSeconds(rawStart);
    const end = parseSrtTimestampSeconds(rawEnd);
    if (!Number.isFinite(start)) continue;
    const rawText = lines.slice(timeIndex + 1).join(' ').trim();
    const text = noSpaceLanguage ? normalizeNoSpaceScriptDisplayText(rawText) : rawText;
    if (!text) continue;
    segments.push({
      text,
      transcript: text,
      start,
      start_ts: start,
      end: Number.isFinite(end) ? end : undefined,
      end_ts: Number.isFinite(end) ? end : undefined,
    });
  }
  return segments;
}

function normalizeGladiaTranscript(data: any) {
  const transcription = data?.result?.transcription || data?.transcription || {};
  const transcriptionUtterances = Array.isArray(transcription?.utterances) ? transcription.utterances : [];
  const diarizationUtterances = Array.isArray(data?.result?.diarization?.results) ? data.result.diarization.results : [];
  const rawUtterances = transcriptionUtterances.length > 0 ? transcriptionUtterances : diarizationUtterances;
  const languages = Array.isArray(transcription?.languages) ? transcription.languages : [];
  const detectedLanguage = firstString([
    data?.language_code,
    transcription?.language,
    languages.find((language: unknown) => typeof language === 'string'),
  ]);
  const transcriptTexts: string[] = [];
  const speakerIds = new Set<string>();
  let normalizedWordCount = 0;

  const segments = rawUtterances
    .map((utterance: any, utteranceIndex: number) => {
      const text = firstString([utterance?.text, utterance?.transcript]);
      if (!text) return null;
      const pairedDiarizationUtterance = transcriptionUtterances.length > 0 ? diarizationUtterances[utteranceIndex] : null;
      const speaker = normalizeGladiaSpeaker(
        utterance?.speaker ??
        utterance?.speaker_id ??
        pairedDiarizationUtterance?.speaker ??
        pairedDiarizationUtterance?.speaker_id
      );
      if (speaker) speakerIds.add(speaker);
      const words = Array.isArray(utterance?.words)
        ? utterance.words.map((word: any) => normalizeGladiaWord(word, speaker)).filter(Boolean)
        : [];
      normalizedWordCount += words.length;
      for (const word of words) {
        if (word?.speaker) speakerIds.add(word.speaker);
      }
      transcriptTexts.push(text);
      return {
        ...utterance,
        text,
        transcript: text,
        start: toFiniteNumber(utterance?.start, 0),
        start_ts: toFiniteNumber(utterance?.start, 0),
        end: Number.isFinite(Number(utterance?.end)) ? toFiniteNumber(utterance?.end, 0) : undefined,
        end_ts: Number.isFinite(Number(utterance?.end)) ? toFiniteNumber(utterance?.end, 0) : undefined,
        speaker,
        words: words.length > 0 ? words : undefined,
      };
    })
    .filter(Boolean);

  if (segments.length === 0) {
    const subtitles = Array.isArray(transcription?.subtitles) ? transcription.subtitles : [];
    const srt = firstString(
      subtitles
        .filter((item: any) => !item?.format || String(item.format).toLowerCase() === 'srt')
        .map((item: any) => item?.subtitles)
    ) || firstString(subtitles.map((item: any) => item?.subtitles));
    segments.push(...parseGladiaSrtSegments(srt, detectedLanguage));
    transcriptTexts.push(...segments.map((segment) => segment.text).filter(Boolean));
  }

  return {
    ...data,
    text: transcriptTexts.join('\n').trim() || transcription?.full_transcript || data?.text,
    language_code: detectedLanguage || data?.language_code,
    segments: segments.length > 0 ? segments.sort((a: any, b: any) => toFiniteNumber(a.start, 0) - toFiniteNumber(b.start, 0)) : data?.segments,
    gladia_diarization_applied: speakerIds.size > 0,
    gladia_speaker_count: speakerIds.size,
    gladia_word_count: normalizedWordCount,
    gladia_utterance_count: rawUtterances.length,
  };
}

function normalizeElevenLabsTranscript(data: any) {
  const collectTranscripts = () => {
    if (Array.isArray(data?.transcripts)) return data.transcripts;
    if (data?.transcripts && typeof data.transcripts === 'object') return Object.values(data.transcripts);
    return [data];
  };

  const allSegments: any[] = [];
  const transcriptTexts: string[] = [];
  const speakerIds = new Set<string>();
  let normalizedWordCount = 0;

  for (const transcript of collectTranscripts()) {
    const rawWords = Array.isArray((transcript as any)?.words) ? (transcript as any).words : [];
    const words = rawWords.map((word: any) => normalizeElevenLabsWord(word)).filter(Boolean) as Array<{
      text: string;
      start: number;
      end?: number;
      speaker?: string;
      channel_index?: number;
    }>;
    normalizedWordCount += words.length;
    for (const word of words) {
      if (word.speaker) speakerIds.add(word.speaker);
    }
    const transcriptText = typeof (transcript as any)?.text === 'string'
      ? String((transcript as any).text).trim()
      : joinElevenLabsWords(words);
    if (transcriptText) transcriptTexts.push(transcriptText);
    if (words.length === 0) continue;

    let buffer: typeof words = [];
    const flush = () => {
      if (buffer.length === 0) return;
      const first = buffer[0];
      const last = buffer[buffer.length - 1];
      allSegments.push({
        text: joinElevenLabsWords(buffer),
        start: first.start,
        end: last.end,
        speaker: first.speaker,
        words: buffer.map((word) => ({
          ...word,
          speaker_id: word.speaker,
        })),
      });
      buffer = [];
    };

    for (const word of words) {
      const previous = buffer[buffer.length - 1];
      const gap = previous?.end != null ? word.start - previous.end : 0;
      const speakerChanged = Boolean(previous?.speaker && word.speaker && previous.speaker !== word.speaker);
      if (buffer.length > 0 && (speakerChanged || gap >= 0.2)) {
        flush();
      }
      buffer.push(word);

      const first = buffer[0];
      const end = word.end ?? word.start;
      const duration = Math.max(0, end - first.start);
      const currentText = joinElevenLabsWords(buffer);
      if (
        buffer.length > 0 &&
        (
          (duration >= 0.5 && isElevenLabsBoundaryText(word.text)) ||
          duration >= 8 ||
          currentText.length >= 90
        )
      ) {
        flush();
      }
    }
    flush();
  }

  return {
    ...data,
    text: transcriptTexts.join('\n').trim() || data?.text,
    language_code: data?.language_code ?? data?.language,
    segments: allSegments.length > 0 ? allSegments.sort((a, b) => toFiniteNumber(a.start, 0) - toFiniteNumber(b.start, 0)) : data?.segments,
    elevenlabs_diarization_applied: speakerIds.size > 0,
    elevenlabs_speaker_count: speakerIds.size,
    elevenlabs_word_count: normalizedWordCount,
  };
}

const cloudAsrAdapters: Record<CloudAsrProvider, CloudAsrAdapter> = {
  'openai-whisper': {
    provider: 'openai-whisper',
    getPreferredResponseFormats(options) {
      return [options.segmentation !== false ? 'verbose_json' : 'json'];
    },
    buildFormData(input) {
      const { filePath, fileBuffer, config, options, includeLanguage, responseFormat } = input;
      const blob = new Blob([bufferToBlobPart(fileBuffer)]);
      const formData = new FormData();
      formData.append('file', blob, path.basename(filePath));
      formData.append('model', config.model || 'whisper-1');
      formData.append('response_format', responseFormat);

      const rawLanguage = typeof options.language === 'string' ? options.language.trim() : '';
      if (includeLanguage && rawLanguage) {
        formData.append('language', rawLanguage);
      }
      if (options.prompt && options.prompt.trim()) {
        formData.append('prompt', options.prompt.trim());
      }
      if (Number.isFinite(Number(options.decodePolicy?.temperature))) {
        formData.append('temperature', String(Number(options.decodePolicy?.temperature)));
      }
      return formData;
    },
  },
  'whispercpp-inference': {
    provider: 'whispercpp-inference',
    getPreferredResponseFormats(options) {
      return options.segmentation !== false ? ['verbose_json', 'json'] : ['json'];
    },
    buildFormData(input) {
      const { filePath, fileBuffer, options, includeLanguage, responseFormat } = input;
      const blob = new Blob([bufferToBlobPart(fileBuffer)]);
      const formData = new FormData();
      formData.append('file', blob, path.basename(filePath));
      formData.append('response_format', responseFormat);
      formData.append('response-format', responseFormat);

      const rawLanguage = typeof options.language === 'string' ? options.language.trim() : '';
      if (includeLanguage && rawLanguage) {
        formData.append('language', rawLanguage);
      }
      if (options.prompt && options.prompt.trim()) {
        formData.append('prompt', options.prompt.trim());
      }
      if (options.segmentation !== false && options.wordAlignment !== false) {
        formData.append('word_timestamps', 'true');
        formData.append('timestamp_granularities[]', 'segment');
        formData.append('timestamp_granularities[]', 'word');
      }
      if (Number.isFinite(Number(options.decodePolicy?.temperature))) {
        formData.append('temperature', String(Number(options.decodePolicy?.temperature)));
      }
      if (Number.isFinite(Number(options.decodePolicy?.beamSize))) {
        formData.append('beam_size', String(Math.max(1, Math.round(Number(options.decodePolicy?.beamSize)))));
      }
      if (Number.isFinite(Number(options.decodePolicy?.noSpeechThreshold))) {
        formData.append('no_speech_thold', String(Number(options.decodePolicy?.noSpeechThreshold)));
      }
      return formData;
    },
  },
  'elevenlabs-scribe': {
    provider: 'elevenlabs-scribe',
    getPreferredResponseFormats() {
      return ['json'];
    },
    buildFormData(input) {
      const { filePath, fileBuffer, config, options, includeLanguage } = input;
      const blob = new Blob([bufferToBlobPart(fileBuffer)]);
      const formData = new FormData();
      formData.append('file', blob, path.basename(filePath));

      const rawLanguage = typeof options.language === 'string' ? options.language.trim() : '';
      const normalizedLanguage = normalizeElevenLabsLanguageCode(rawLanguage);
      const configuredModel = String(config.model || '').trim();
      const effectiveModel = configuredModel || (normalizedLanguage ? 'scribe_v2' : 'scribe_v1');
      formData.append('model_id', effectiveModel);

      if (includeLanguage && normalizedLanguage) {
        formData.append('language_code', normalizedLanguage);
      }
      formData.append('diarize', options.diarization ? 'true' : 'false');
      const numSpeakers = getElevenLabsNumSpeakers(options);
      if (numSpeakers != null) {
        formData.append('num_speakers', String(numSpeakers));
      }
      formData.append('tag_audio_events', 'false');
      if (options.segmentation === false && options.wordAlignment === false) {
        formData.append('timestamps_granularity', 'none');
      } else {
        formData.append('timestamps_granularity', 'word');
      }
      if (Number.isFinite(Number(options.decodePolicy?.temperature))) {
        formData.append('temperature', String(Number(options.decodePolicy?.temperature)));
      }
      return formData;
    },
    normalizeResponse(data) {
      return normalizeElevenLabsTranscript(data);
    },
  },
  'deepgram-listen': {
    provider: 'deepgram-listen',
    getPreferredResponseFormats() {
      return ['json'];
    },
    buildRequestUrl(endpointUrl, input) {
      return buildDeepgramListenUrl(endpointUrl, input);
    },
    getRequestHeaders(input) {
      return { Accept: 'application/json', 'Content-Type': inferAudioMimeType(input.filePath) };
    },
    buildRawBody(input) {
      return new Blob([bufferToBlobPart(input.fileBuffer)], { type: inferAudioMimeType(input.filePath) });
    },
    normalizeResponse(data) {
      return normalizeDeepgramTranscript(data);
    },
  },
  'gladia-pre-recorded': {
    provider: 'gladia-pre-recorded',
    getPreferredResponseFormats() {
      return ['json'];
    },
    getRequestHeaders() {
      return { Accept: 'application/json', 'Content-Type': 'application/json' };
    },
    buildJsonBody() {
      return {};
    },
    normalizeResponse(data) {
      return normalizeGladiaTranscript(data);
    },
  },
  'github-models-phi4-multimodal': {
    provider: 'github-models-phi4-multimodal',
    getPreferredResponseFormats() {
      return ['json'];
    },
    getRequestHeaders() {
      return { 'Content-Type': 'application/json' };
    },
    buildJsonBody(input) {
      const { fileBuffer, filePath, config, options } = input;
      const audioFormat = inferAudioFormat(filePath);
      return {
        model: config.model || 'microsoft/Phi-4-multimodal-instruct',
        messages: [
          {
            role: 'system',
            content: 'You are a precise ASR transcription engine. Output only the transcript text.',
          },
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: buildGithubPhi4AsrPrompt(options),
              },
              {
                type: 'input_audio',
                input_audio: {
                  data: fileBuffer.toString('base64'),
                  format: audioFormat,
                },
              },
            ],
          },
        ],
        temperature: Number.isFinite(Number(options.decodePolicy?.temperature))
          ? Number(options.decodePolicy?.temperature)
          : 0,
        top_p: 1,
        max_tokens: getGithubPhi4MaxOutputTokens(),
        stream: false,
      };
    },
    normalizeResponse(data) {
      return normalizeGithubPhi4Transcript(data);
    },
  },
  'google-cloud-chirp3': {
    provider: 'google-cloud-chirp3',
    getPreferredResponseFormats() {
      return ['json'];
    },
    getRequestHeaders() {
      return { Accept: 'application/json', 'Content-Type': 'application/json' };
    },
    buildJsonBody(input) {
      const { fileBuffer, config, options } = input;
      return {
        config: {
          autoDecodingConfig: {},
          languageCodes: resolveGoogleCloudLanguageCodes(options.language),
          model: config.model || 'chirp_3',
          features: {
            enableWordTimeOffsets: options.wordAlignment !== false,
            enableWordConfidence: options.wordAlignment !== false,
            enableAutomaticPunctuation: true,
          },
        },
        content: fileBuffer.toString('base64'),
      };
    },
    normalizeResponse(data) {
      return normalizeGoogleCloudChirp3Transcript(data);
    },
  },
  'google-gemini-audio': {
    provider: 'google-gemini-audio',
    getPreferredResponseFormats() {
      return ['json'];
    },
    getRequestHeaders() {
      return { Accept: 'application/json', 'Content-Type': 'application/json' };
    },
    buildJsonBody(input) {
      const { fileBuffer, filePath, options } = input;
      const temperature = Number.isFinite(Number(options.decodePolicy?.temperature))
        ? Number(options.decodePolicy?.temperature)
        : 0;
      return {
        contents: [
          {
            role: 'user',
            parts: [
              { text: buildGeminiAudioAsrPrompt(options) },
              {
                inlineData: {
                  mimeType: inferAudioMimeType(filePath),
                  data: fileBuffer.toString('base64'),
                },
              },
            ],
          },
        ],
        generationConfig: {
          temperature,
          topP: 1,
          maxOutputTokens: 8192,
          responseMimeType: 'application/json',
          responseSchema: {
            type: 'OBJECT',
            properties: {
              text: { type: 'STRING' },
              language_code: { type: 'STRING' },
              segments: {
                type: 'ARRAY',
                items: {
                  type: 'OBJECT',
                  properties: {
                    start_ms: { type: 'INTEGER' },
                    end_ms: { type: 'INTEGER' },
                    text: { type: 'STRING' },
                    speaker: { type: 'STRING' },
                    language_code: { type: 'STRING' },
                  },
                  required: ['text'],
                },
              },
            },
            required: ['segments'],
          },
        },
      };
    },
    normalizeResponse(data) {
      return normalizeGeminiAudioTranscript(data);
    },
  },
};

export function getCloudAsrAdapter(provider: CloudAsrProvider) {
  return cloudAsrAdapters[provider];
}

function buildGeminiMultiAudioJsonBody(
  inputs: Array<CloudAsrBatchAudioInput & { fileBuffer: Buffer }>,
  options: CloudAsrAdapterRequestOptions
) {
  const temperature = Number.isFinite(Number(options.decodePolicy?.temperature))
    ? Number(options.decodePolicy?.temperature)
    : 0;
  return {
    contents: [
      {
        role: 'user',
        parts: [
          ...inputs.flatMap((input, index) => [
            { text: `Audio file index ${index}:` },
            {
              inlineData: {
                mimeType: inferAudioMimeType(input.filePath),
                data: input.fileBuffer.toString('base64'),
              },
            },
          ]),
          { text: buildGeminiAudioBatchAsrPrompt(options, inputs.length) },
        ],
      },
    ],
    generationConfig: {
      temperature,
      topP: 1,
      maxOutputTokens: 65536,
      responseMimeType: 'application/json',
      responseSchema: {
        type: 'OBJECT',
        properties: {
          items: {
            type: 'ARRAY',
            items: {
              type: 'OBJECT',
              properties: {
                index: { type: 'INTEGER' },
                text: { type: 'STRING' },
                speaker: { type: 'STRING' },
                language_code: { type: 'STRING' },
              },
              required: ['index', 'text'],
            },
          },
        },
        required: ['items'],
      },
    },
  };
}

export async function requestGeminiCloudAsrBatch(
  audioInputs: CloudAsrBatchAudioInput[],
  resolvedProvider: ResolvedCloudAsrProvider,
  config: any,
  options: CloudAsrAdapterRequestOptions,
  deps: Pick<CloudAsrAdapterDeps, 'createAbortSignalWithTimeout'>,
  signal?: AbortSignal
): Promise<CloudAsrBatchProviderResponse> {
  if (resolvedProvider.provider !== 'google-gemini-audio') {
    throw new Error(`Gemini multi-audio ASR batch is not supported for provider "${resolvedProvider.provider}".`);
  }
  if (audioInputs.length === 0) {
    return {
      results: [],
      meta: {
        provider: resolvedProvider.provider,
        endpointUrl: resolvedProvider.endpointUrl,
        effectiveModel: resolvedProvider.effectiveModel,
        callCount: 0,
        rawSegmentCount: 0,
        rawWordCount: 0,
        rawHasTimestamps: false,
        geminiMultiAudioBatching: {
          enabled: true,
          inputCount: 0,
          requestCount: 0,
        },
      },
    };
  }

  const rawLanguage = typeof options.language === 'string' ? options.language.trim() : '';
  const inputs = await Promise.all(
    audioInputs.map(async (input) => ({
      ...input,
      fileBuffer: await fs.readFile(input.filePath),
    }))
  );
  const headers = {
    ...buildCloudAsrRequestHeaders(resolvedProvider.provider, config.key),
    Accept: 'application/json',
    'Content-Type': 'application/json',
  };
  const estimatedInputTokens = estimateGeminiBatchAudioAsrInputTokens(audioInputs);
  const cloudRequestTimeoutMs = getCloudAsrRequestTimeoutMs(resolvedProvider, options);
  const body = JSON.stringify(buildGeminiMultiAudioJsonBody(inputs, options));

  const sendOnce = async () => {
    await enforceGeminiFreeTierRateLimit(config, resolvedProvider, estimatedInputTokens, signal);
    const request = deps.createAbortSignalWithTimeout(cloudRequestTimeoutMs, signal);
    try {
      const response = await fetch(resolvedProvider.endpointUrl, {
        method: 'POST',
        headers,
        body,
        signal: request.signal,
        redirect: 'error',
      });
      const rawText = await response.text();
      return { response, rawText };
    } finally {
      request.dispose();
    }
  };

  let requestResult: Awaited<ReturnType<typeof sendOnce>> | null = null;
  let transientRetryCount = 0;
  for (let attemptIndex = 0; ; attemptIndex += 1) {
    requestResult = await sendOnce();
    if (!shouldRetryTransientCloudAsrResponse(requestResult.response) || attemptIndex >= 3) {
      break;
    }
    transientRetryCount += 1;
    await sleepWithAbort(getTransientRetryDelayMs(requestResult.response, attemptIndex), signal);
  }

  if (!requestResult) {
    throw new Error('Cloud ASR batch request failed before receiving response.');
  }
  if (!requestResult.response.ok) {
    const errorMessage = extractErrorMessage(requestResult.rawText, requestResult.response.statusText);
    const retryHint = transientRetryCount > 0 ? ` | transient_retries=${transientRetryCount}` : '';
    throw new Error(`Cloud ASR error (${requestResult.response.status}): ${errorMessage}${retryHint}`);
  }

  let data: any = null;
  try {
    data = JSON.parse(requestResult.rawText);
  } catch {
    data = { text: requestResult.rawText };
  }

  const parsed = normalizeGeminiAudioBatchItems(data, inputs.length);
  if (parsed.items.length === 0) {
    throw new Error('Cloud ASR response does not contain Gemini batch transcript items.');
  }

  const totalDurationSec = audioInputs.reduce((sum, input) => {
    const durationSec = Number(input.audioDurationSec);
    return sum + (Number.isFinite(durationSec) && durationSec > 0 ? durationSec : 0);
  }, 0);
  const meta = {
    provider: resolvedProvider.provider,
    endpointUrl: resolvedProvider.endpointUrl,
    effectiveModel: resolvedProvider.effectiveModel,
    responseFormat: 'json',
    rawHasTimestamps: false,
    rawSegmentCount: parsed.normalized.filter((item) => item.text.trim()).length,
    rawWordCount: 0,
    requestedLanguage: rawLanguage || null,
    sentLanguage: null,
    detectedLanguage: firstString(parsed.normalized.map((item) => item.language_code)) || null,
    autoLanguageFallbackUsed: false,
    responseFormatFallbackUsed: false,
    transientRetryCount,
    callCount: 1,
    geminiMultiAudioBatching: {
      enabled: true,
      inputCount: inputs.length,
      requestCount: 1,
      totalDurationSec: Number(totalDurationSec.toFixed(3)),
      rawItemCount: parsed.items.length,
      missingItemCount: parsed.missingCount,
      recoveredFromXml: parsed.recoveredFromXml,
      maxBatchSize: 50,
    },
    geminiFreeTierLimiter: {
      applied: true,
      ...GEMINI_FREE_TIER_LIMITS,
      audioTokensPerSecond: GEMINI_AUDIO_TOKENS_PER_SECOND,
      estimatedInputTokens,
    },
    cjkWordDiagnostics: null,
  };

  return {
    meta,
    results: inputs.map((input, position) => {
      const item = parsed.normalized[position] || { text: '', speaker: undefined };
      const transcript = buildGeminiBatchTranscriptFromItem(item, input.audioDurationSec);
      return {
        input,
        result: {
          ...transcript,
          meta: {
            ...meta,
            batchItemIndex: position,
            batchInputIndex: input.index,
            rawSegmentCount: transcript.chunks.length,
          },
        },
      };
    }),
  };
}

function getGladiaPollIntervalMs() {
  return Math.round(getEnvNumber('ASR_GLADIA_POLL_INTERVAL_MS', 2000, 500, 60000));
}

function buildGladiaEndpointPath(endpointUrl: string, targetPath: string) {
  const next = new URL(endpointUrl);
  const normalized = next.pathname.replace(/\/+$/, '');
  const suffixes = ['/v2/pre-recorded', '/v2/upload', '/v2'];
  const prefix = suffixes.reduce((current, suffix) => (
    current.endsWith(suffix) ? current.slice(0, -suffix.length) : current
  ), normalized);
  next.pathname = `${prefix}${targetPath}`.replace(/\/{2,}/g, '/');
  next.search = '';
  return next.toString();
}

function buildGladiaUploadUrl(endpointUrl: string) {
  return buildGladiaEndpointPath(endpointUrl, '/v2/upload');
}

function buildGladiaPollUrl(endpointUrl: string, job: any) {
  const resultUrl = firstString([job?.result_url, job?.url]);
  if (resultUrl) return resultUrl;
  const id = firstString([job?.id, job?.transcription_id]);
  if (!id) throw new Error('Gladia transcription job response did not include an id or result_url.');
  return buildGladiaEndpointPath(endpointUrl, `/v2/pre-recorded/${encodeURIComponent(id)}`);
}

function buildGladiaLanguageConfig(language: string) {
  const normalized = normalizeGladiaLanguageCode(language);
  return {
    languages: normalized ? [normalized] : [],
    code_switching: false,
  };
}

function buildGladiaDiarizationConfig(options: CloudAsrAdapterRequestOptions) {
  const diarizationOptions = options.diarizationOptions || {};
  const mode = String(diarizationOptions.mode || '').trim();
  const exactSpeakerCount = toFiniteNumber(diarizationOptions.exactSpeakerCount, Number.NaN);
  const minSpeakers = toFiniteNumber(diarizationOptions.minSpeakers, Number.NaN);
  const maxSpeakers = toFiniteNumber(diarizationOptions.maxSpeakers, Number.NaN);
  const config: Record<string, number> = {};

  if (mode === 'fixed' && Number.isFinite(exactSpeakerCount) && exactSpeakerCount > 0) {
    config.number_of_speakers = Math.round(exactSpeakerCount);
  } else {
    if ((mode === 'range' || mode === 'many') && Number.isFinite(minSpeakers) && minSpeakers > 0) {
      config.min_speakers = Math.round(minSpeakers);
    }
    if ((mode === 'range' || mode === 'many') && Number.isFinite(maxSpeakers) && maxSpeakers > 0) {
      config.max_speakers = Math.round(maxSpeakers);
    }
  }

  return Object.keys(config).length > 0 ? config : null;
}

function buildGladiaPreRecordedPayload(audioUrl: string, options: CloudAsrAdapterRequestOptions) {
  const payload: Record<string, unknown> = {
    audio_url: audioUrl,
    language_config: buildGladiaLanguageConfig(options.language || ''),
    subtitles: true,
    subtitles_config: {
      formats: ['srt'],
      minimum_duration: 1,
      maximum_duration: 15.5,
      maximum_characters_per_row: 80,
      maximum_rows_per_caption: 2,
      style: 'default',
    },
    sentences: true,
    punctuation_enhanced: true,
    diarization: Boolean(options.diarization),
  };
  const diarizationConfig = options.diarization ? buildGladiaDiarizationConfig(options) : null;
  if (diarizationConfig) {
    payload.diarization_config = diarizationConfig;
  }
  if (options.prompt && options.prompt.trim()) {
    payload.context_prompt = options.prompt.trim();
  }
  return payload;
}

async function readGladiaJsonResponse(response: Response, fallback: string) {
  const rawText = await response.text();
  if (!response.ok) {
    throw new Error(`Cloud ASR error (${response.status}): ${extractErrorMessage(rawText, fallback || response.statusText)}`);
  }
  try {
    return rawText ? JSON.parse(rawText) : {};
  } catch {
    throw new Error(`Cloud ASR error (${response.status}): Invalid JSON response from Gladia.`);
  }
}

async function fetchGladiaJson(
  url: string,
  init: RequestInit,
  timeoutMs: number,
  deps: Pick<CloudAsrAdapterDeps, 'createAbortSignalWithTimeout'>,
  signal?: AbortSignal
) {
  const request = deps.createAbortSignalWithTimeout(timeoutMs, signal);
  try {
    const response = await fetch(url, {
      ...init,
      signal: request.signal,
      redirect: 'error',
    });
    return response;
  } finally {
    request.dispose();
  }
}

async function requestGladiaPreRecordedAsr(
  filePath: string,
  resolvedProvider: ResolvedCloudAsrProvider,
  config: any,
  options: CloudAsrAdapterRequestOptions,
  deps: CloudAsrAdapterDeps,
  signal?: AbortSignal
): Promise<CloudAsrProviderResult> {
  const rawLanguage = typeof options.language === 'string' ? options.language.trim() : '';
  const fileBuffer = await fs.readFile(filePath);
  const cloudRequestTimeoutMs = getCloudAsrRequestTimeoutMs(resolvedProvider, options);
  const headers = buildCloudAsrRequestHeaders(resolvedProvider.provider, config.key);
  const uploadForm = new FormData();
  uploadForm.append(
    'audio',
    new Blob([bufferToBlobPart(fileBuffer)], { type: inferAudioMimeType(filePath) }),
    path.basename(filePath)
  );

  const uploadResponse = await fetchGladiaJson(
    buildGladiaUploadUrl(resolvedProvider.endpointUrl),
    {
      method: 'POST',
      headers,
      body: uploadForm,
    },
    cloudRequestTimeoutMs,
    deps,
    signal
  );
  const uploadData = await readGladiaJsonResponse(uploadResponse, 'Gladia upload failed.');
  const audioUrl = firstString([uploadData?.audio_url]);
  if (!audioUrl) {
    throw new Error('Cloud ASR error (502): Gladia upload response did not include audio_url.');
  }

  const initResponse = await fetchGladiaJson(
    resolvedProvider.endpointUrl,
    {
      method: 'POST',
      headers: {
        ...headers,
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(buildGladiaPreRecordedPayload(audioUrl, options)),
    },
    cloudRequestTimeoutMs,
    deps,
    signal
  );
  const jobData = await readGladiaJsonResponse(initResponse, 'Gladia transcription job creation failed.');
  const pollUrl = buildGladiaPollUrl(resolvedProvider.endpointUrl, jobData);
  const startedAt = Date.now();
  const pollIntervalMs = getGladiaPollIntervalMs();
  let pollCount = 0;
  let transientRetryCount = 0;
  let resultData: any = null;

  while (true) {
    if (signal?.aborted) {
      const error = new Error('Aborted');
      error.name = 'AbortError';
      throw error;
    }
    if (Date.now() - startedAt > cloudRequestTimeoutMs) {
      throw new Error(`Cloud ASR error (408): Gladia transcription timed out after ${cloudRequestTimeoutMs}ms.`);
    }

    if (pollCount > 0) {
      await sleepWithAbort(pollIntervalMs, signal);
    }
    pollCount += 1;
    const response = await fetchGladiaJson(
      pollUrl,
      {
        method: 'GET',
        headers: {
          ...headers,
          Accept: 'application/json',
        },
      },
      cloudRequestTimeoutMs,
      deps,
      signal
    );

    if (shouldRetryTransientCloudAsrResponse(response)) {
      transientRetryCount += 1;
      await sleepWithAbort(parseRetryAfterMs(response) ?? getTransientRetryDelayMs(response, transientRetryCount), signal);
      continue;
    }

    const data = await readGladiaJsonResponse(response, 'Gladia polling failed.');
    const status = String(data?.status || '').trim().toLowerCase();
    if (status === 'done' || status === 'completed' || status === 'success') {
      resultData = data;
      break;
    }
    if (status === 'error' || status === 'failed') {
      const errorMessage = firstString([data?.error, data?.message, data?.error_message]) || `Gladia job failed${data?.error_code ? ` (${data.error_code})` : ''}.`;
      throw new Error(`Cloud ASR error (${data?.error_code || 500}): ${errorMessage}`);
    }
  }

  const normalizedData = normalizeGladiaTranscript(resultData);
  const transcriptLanguage =
    typeof normalizedData?.language_code === 'string' && normalizedData.language_code.trim()
      ? normalizedData.language_code.trim()
      : rawLanguage;
  let transcript = deps.extractStructuredTranscript(normalizedData, transcriptLanguage);
  if (options.wordAlignment === false && transcript.word_segments.length > 0) {
    transcript = deps.disableWordAlignment(transcript, transcriptLanguage);
  }
  if (transcript.chunks.length === 0) {
    throw new Error('Cloud ASR response does not contain transcript text.');
  }
  const rawSegments = Array.isArray(normalizedData?.segments)
    ? normalizedData.segments
    : Array.isArray(normalizedData?.chunks)
      ? normalizedData.chunks
      : [];
  const rawHasTimestamps = rawSegments.some(
    (s: any) => s?.start_ts != null || s?.start != null || s?.timestamp?.start != null
  );

  return {
    ...transcript,
    meta: {
      provider: resolvedProvider.provider,
      endpointUrl: resolvedProvider.endpointUrl,
      effectiveModel: resolvedProvider.effectiveModel,
      responseFormat: 'json',
      rawHasTimestamps,
      rawSegmentCount: rawSegments.length,
      rawWordCount: transcript.word_segments.length,
      requestedLanguage: rawLanguage || null,
      sentLanguage: rawLanguage && rawLanguage.toLowerCase() !== 'auto' ? rawLanguage : null,
      detectedLanguage: normalizedData?.language_code || null,
      nativeWordTimestamps: transcript.word_segments.length > 0,
      providerNativeDiarization: Boolean(normalizedData?.gladia_diarization_applied),
      gladia: {
        jobId: firstString([jobData?.id, resultData?.id]) || null,
        pollUrl,
        pollCount,
        pollIntervalMs,
        requestTimeoutMs: cloudRequestTimeoutMs,
        uploadLimitMb: 1000,
        maxAudioLengthMinutes: 135,
        diarizationRequested: Boolean(options.diarization),
        requestParamsDiarization: typeof resultData?.request_params?.diarization === 'boolean'
          ? resultData.request_params.diarization
          : null,
        diarizationApplied: Boolean(normalizedData?.gladia_diarization_applied),
        speakerCount: toFiniteNumber(normalizedData?.gladia_speaker_count, 0),
        wordCount: toFiniteNumber(normalizedData?.gladia_word_count, transcript.word_segments.length),
        utteranceCount: toFiniteNumber(normalizedData?.gladia_utterance_count, 0),
        subtitlesRequested: true,
        sentencesRequested: true,
      },
      autoLanguageFallbackUsed: false,
      responseFormatFallbackUsed: false,
      transientRetryCount,
      cjkWordDiagnostics: transcript.debug?.cjkWordDiagnostics || null,
    },
  };
}

export async function requestCloudAsr(
  filePath: string,
  resolvedProvider: ResolvedCloudAsrProvider,
  config: any,
  options: CloudAsrAdapterRequestOptions,
  deps: CloudAsrAdapterDeps,
  signal?: AbortSignal
): Promise<CloudAsrProviderResult> {
  if (resolvedProvider.provider === 'gladia-pre-recorded') {
    return requestGladiaPreRecordedAsr(filePath, resolvedProvider, config, options, deps, signal);
  }

  const { language, wordAlignment } = options;
  const rawLanguage = typeof language === 'string' ? language.trim() : '';
  const normalizedLanguage = rawLanguage.toLowerCase();
  const adapter = getCloudAsrAdapter(resolvedProvider.provider);

  const fileBuffer = await fs.readFile(filePath);
  const preferredFormats = adapter.getPreferredResponseFormats(options);
  const headers = buildCloudAsrRequestHeaders(resolvedProvider.provider, config.key);
  const geminiEstimatedInputTokens = resolvedProvider.provider === 'google-gemini-audio'
    ? estimateGeminiAudioAsrInputTokens(options)
    : null;
  const cloudRequestTimeoutMs = getCloudAsrRequestTimeoutMs(resolvedProvider, options);

  const sendOnce = async (includeLanguage: boolean, responseFormat: string) => {
    const bodyInput = {
      filePath,
      fileBuffer,
      config,
      options,
      includeLanguage,
      responseFormat,
    };
    const requestHeaders = {
      ...headers,
      ...(adapter.getRequestHeaders ? adapter.getRequestHeaders(bodyInput) : {}),
    };
    const body = adapter.buildJsonBody
      ? JSON.stringify(adapter.buildJsonBody(bodyInput))
      : adapter.buildRawBody
        ? adapter.buildRawBody(bodyInput)
      : adapter.buildFormData?.(bodyInput);
    if (!body) {
      throw new Error(`Cloud ASR adapter "${resolvedProvider.provider}" did not provide a request body.`);
    }
    await enforceGeminiFreeTierRateLimit(config, resolvedProvider, geminiEstimatedInputTokens, signal);
    const requestUrl = adapter.buildRequestUrl
      ? adapter.buildRequestUrl(resolvedProvider.endpointUrl, bodyInput)
      : resolvedProvider.endpointUrl;

    const request = deps.createAbortSignalWithTimeout(cloudRequestTimeoutMs, signal);
    try {
      const response = await fetch(requestUrl, {
        method: 'POST',
        headers: requestHeaders,
        body,
        signal: request.signal,
        redirect: 'error',
      });
      const rawText = await response.text();
      return { response, rawText, includeLanguage, responseFormat };
    } finally {
      request.dispose();
    }
  };

  let attemptedAutoFallback = false;
  let attemptedFormatFallback = false;
  let firstErrorSummary = '';
  let requestResult: Awaited<ReturnType<typeof sendOnce>> | null = null;
  let lastErrorForFormat = '';
  let transientRetryCount = 0;
  const maxTransientRetries = resolvedProvider.provider === 'google-gemini-audio' ? 3 : 2;

  const sendWithTransientRetry = async (includeLanguage: boolean, responseFormat: string) => {
    let attemptIndex = 0;
    while (true) {
      const result = await sendOnce(includeLanguage, responseFormat);
      if (!shouldRetryTransientCloudAsrResponse(result.response) || attemptIndex >= maxTransientRetries) {
        return result;
      }
      transientRetryCount += 1;
      const delayMs = getTransientRetryDelayMs(result.response, attemptIndex);
      await sleepWithAbort(delayMs, signal);
      attemptIndex += 1;
    }
  };

  for (let i = 0; i < preferredFormats.length; i += 1) {
    const format = preferredFormats[i];
    requestResult = await sendWithTransientRetry(Boolean(rawLanguage), format);
    const canRetryAutoLanguage =
      normalizedLanguage === 'auto' &&
      requestResult.includeLanguage &&
      (requestResult.response.status === 400 || requestResult.response.status === 422);

    if (!requestResult.response.ok && canRetryAutoLanguage) {
      attemptedAutoFallback = true;
      firstErrorSummary = `first_attempt_${requestResult.response.status}: ${extractErrorMessage(
        requestResult.rawText,
        requestResult.response.statusText
      )}`;
      requestResult = await sendWithTransientRetry(false, format);
    }

    if (requestResult.response.ok) break;

    lastErrorForFormat = extractErrorMessage(requestResult.rawText, requestResult.response.statusText);
    const canFallbackToNextFormat =
      i + 1 < preferredFormats.length &&
      (requestResult.response.status === 400 ||
        requestResult.response.status === 415 ||
        requestResult.response.status === 422 ||
        requestResult.response.status === 500);
    if (canFallbackToNextFormat) {
      attemptedFormatFallback = true;
      continue;
    }
    break;
  }

  if (!requestResult) {
    throw new Error('Cloud ASR request failed before receiving response.');
  }

  if (!requestResult.response.ok) {
    const errorMessage = lastErrorForFormat || extractErrorMessage(requestResult.rawText, requestResult.response.statusText);
    const fallbackHint = attemptedAutoFallback ? ` | ${firstErrorSummary}` : '';
    const retryHint = transientRetryCount > 0 ? ` | transient_retries=${transientRetryCount}` : '';
    throw new Error(`Cloud ASR error (${requestResult.response.status}): ${errorMessage}${fallbackHint}${retryHint}`);
  }

  let data: any = null;
  try {
    data = JSON.parse(requestResult.rawText);
  } catch {
    data = { text: requestResult.rawText };
  }

  const normalizedData = adapter.normalizeResponse ? adapter.normalizeResponse(data) : data;
  const transcriptLanguage =
    typeof normalizedData?.language_code === 'string' && normalizedData.language_code.trim()
      ? normalizedData.language_code.trim()
      : rawLanguage;
  let transcript = deps.extractStructuredTranscript(normalizedData, transcriptLanguage);
  if (wordAlignment === false && transcript.word_segments.length > 0) {
    transcript = deps.disableWordAlignment(transcript, transcriptLanguage);
  }
  if (transcript.chunks.length === 0) {
    throw new Error('Cloud ASR response does not contain transcript text.');
  }

  const rawSegments = Array.isArray(normalizedData?.segments)
    ? normalizedData.segments
    : Array.isArray(normalizedData?.chunks)
      ? normalizedData.chunks
      : [];
  const rawHasTimestamps = rawSegments.some(
    (s: any) => s?.start_ts != null || s?.start != null || s?.timestamp?.start != null
  );
  const isElevenLabsScribe = resolvedProvider.provider === 'elevenlabs-scribe';
  const isDeepgramListen = resolvedProvider.provider === 'deepgram-listen';

  return {
    ...transcript,
    meta: {
      provider: resolvedProvider.provider,
      endpointUrl: resolvedProvider.endpointUrl,
      effectiveModel: resolvedProvider.effectiveModel,
      responseFormat: requestResult.responseFormat,
      rawHasTimestamps,
      rawSegmentCount: rawSegments.length,
      rawWordCount: transcript.word_segments.length,
      requestedLanguage: rawLanguage || null,
      sentLanguage: requestResult.includeLanguage ? rawLanguage : null,
      detectedLanguage: normalizedData?.language_code || null,
      ...(isElevenLabsScribe
        ? {
          nativeWordTimestamps: transcript.word_segments.length > 0,
          providerNativeDiarization: Boolean(normalizedData?.elevenlabs_diarization_applied),
          elevenLabs: {
            diarizationRequested: Boolean(options.diarization),
            numSpeakersHint: getElevenLabsNumSpeakers(options),
            diarizationApplied: Boolean(normalizedData?.elevenlabs_diarization_applied),
            speakerCount: toFiniteNumber(normalizedData?.elevenlabs_speaker_count, 0),
            wordCount: toFiniteNumber(normalizedData?.elevenlabs_word_count, transcript.word_segments.length),
            audioEventsTagged: false,
            requestTimeoutMs: cloudRequestTimeoutMs,
            standardModeLimitHours: 10,
            uploadLimitGb: 3,
          },
        }
        : {}),
      ...(isDeepgramListen
        ? {
          nativeWordTimestamps: transcript.word_segments.length > 0,
          providerNativeDiarization: Boolean(normalizedData?.deepgram_diarization_applied),
          deepgram: {
            diarizationRequested: Boolean(options.diarization),
            diarizationApplied: Boolean(normalizedData?.deepgram_diarization_applied),
            speakerCount: toFiniteNumber(normalizedData?.deepgram_speaker_count, 0),
            wordCount: toFiniteNumber(normalizedData?.deepgram_word_count, transcript.word_segments.length),
            utteranceCount: toFiniteNumber(normalizedData?.deepgram_utterance_count, 0),
            utterancesRequested: options.segmentation !== false || Boolean(options.vad) || Boolean(options.diarization),
            utteranceSplitSec: getDeepgramUtteranceSplitSec(),
            requestTimeoutMs: cloudRequestTimeoutMs,
            uploadLimitGb: 2,
            maxProcessingTimeSec: String(resolvedProvider.effectiveModel || '').toLowerCase().includes('whisper') ? 1200 : 600,
          },
        }
        : {}),
      autoLanguageFallbackUsed: attemptedAutoFallback,
      responseFormatFallbackUsed: attemptedFormatFallback,
      transientRetryCount,
      geminiFreeTierLimiter: resolvedProvider.provider === 'google-gemini-audio'
        ? {
            applied: true,
            ...GEMINI_FREE_TIER_LIMITS,
            audioTokensPerSecond: GEMINI_AUDIO_TOKENS_PER_SECOND,
            estimatedInputTokens: geminiEstimatedInputTokens,
          }
        : null,
      cjkWordDiagnostics: transcript.debug?.cjkWordDiagnostics || null,
    },
  };
}
