import crypto from 'node:crypto';
import type {
  CloudAsrAdapter,
  CloudAsrAdapterDeps,
  CloudAsrAdapterRequestOptions,
  CloudAsrBatchAudioInput,
  CloudAsrBatchProviderResponse,
  CloudAsrBatchRequestInput,
  CloudAsrStructuredTranscript,
  ResolvedCloudAsrProvider,
} from '../types.js';
import {
  extractCloudAsrErrorMessage,
  firstString,
  getCloudAsrRequestTimeoutMs,
  getTransientRetryDelayMs,
  inferAudioMimeType,
  parseFlexibleTimestampSeconds,
  readAudioFile,
  shouldRetryTransientCloudAsrResponse,
  sleepWithAbort,
  stripCodeFence,
} from '../runtime/shared.js';
import { createCloudAsrProviderDefinition, ensureEndpointPath, parseCloudAsrUrl } from './shared.js';

export const DEFAULT_GEMINI_FREE_TIER_LIMITS = {
  rpm: 5,
  tpm: 250_000,
  rpd: 20,
};
export const GEMINI_AUDIO_TOKENS_PER_SECOND = 32;
const GEMINI_ASR_TEXT_TOKEN_BUFFER = 1024;

export interface GeminiFreeTierLimiterLimits {
  enabled: boolean;
  rpm: number;
  tpm: number;
  rpd: number;
}

interface GeminiFreeTierLimiterState {
  queue: Promise<void>;
  requestTimestamps: number[];
  tokenUsages: Array<{ timestamp: number; tokens: number }>;
  dayKey: string;
  dailyCount: number;
}

const geminiFreeTierLimiterState = new Map<string, GeminiFreeTierLimiterState>();

function getEnvBoolean(name: string, fallback: boolean) {
  const raw = process.env[name];
  if (raw == null || raw.trim() === '') return fallback;
  const normalized = raw.trim().toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
  if (['0', 'false', 'no', 'off'].includes(normalized)) return false;
  return fallback;
}

function getEnvInteger(name: string, fallback: number, min: number, max: number) {
  const raw = process.env[name];
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, Math.floor(parsed)));
}

export function getGeminiFreeTierLimits(): GeminiFreeTierLimiterLimits {
  return {
    enabled: getEnvBoolean('ASR_GEMINI_FREE_TIER_LIMITER_ENABLED', true),
    rpm: getEnvInteger('ASR_GEMINI_FREE_TIER_RPM', DEFAULT_GEMINI_FREE_TIER_LIMITS.rpm, 1, 10_000),
    tpm: getEnvInteger('ASR_GEMINI_FREE_TIER_TPM', DEFAULT_GEMINI_FREE_TIER_LIMITS.tpm, 1, 100_000_000),
    rpd: getEnvInteger('ASR_GEMINI_FREE_TIER_RPD', DEFAULT_GEMINI_FREE_TIER_LIMITS.rpd, 1, 1_000_000),
  };
}

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

export async function enforceGeminiFreeTierRateLimit(
  config: any,
  resolvedProvider: ResolvedCloudAsrProvider,
  estimatedInputTokens: number | null,
  signal?: AbortSignal
) {
  const limits = getGeminiFreeTierLimits();
  if (!limits.enabled) return;

  if (estimatedInputTokens != null && estimatedInputTokens > limits.tpm) {
    throw new Error(
      `Gemini free tier single request token estimate exceeds the local TPM limit (${estimatedInputTokens}/${limits.tpm} input tokens for ${resolvedProvider.effectiveModel}). Use VAD/windowed transcription or shorter audio chunks.`
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
      if (state!.dailyCount >= limits.rpd) {
        throw new Error(
          `Gemini free tier daily request limit reached locally (${limits.rpd} RPD for ${resolvedProvider.effectiveModel}). Daily quota resets at Pacific Time midnight.`
        );
      }
      const currentMinuteTokens = state!.tokenUsages.reduce((sum, usage) => sum + usage.tokens, 0);
      if (
        estimatedInputTokens != null &&
        currentMinuteTokens + estimatedInputTokens > limits.tpm &&
        state!.tokenUsages.length > 0
      ) {
        const oldestTokenUsage = state!.tokenUsages[0];
        await sleepWithAbort(Math.max(0, oldestTokenUsage.timestamp + 60_000 - now), signal);
        continue;
      }

      const oldestRequest = state!.requestTimestamps[0];
      const nextBucketSlotMs =
        state!.requestTimestamps.length >= limits.rpm && oldestRequest != null
          ? oldestRequest + 60_000
          : now;
      const minSpacingMs = Math.ceil(60_000 / limits.rpm);
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

export function estimateGeminiAudioAsrInputTokens(options: CloudAsrAdapterRequestOptions) {
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
  const rangeParts = timestamp.split(/\s*(?:-->|-|to|~|–|—)\s*/i).filter(Boolean);
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

const googleGeminiAudioRuntime: CloudAsrAdapter = {
  provider: 'google-gemini',
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
};

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

async function requestGeminiCloudAsrBatch(input: CloudAsrBatchRequestInput): Promise<CloudAsrBatchProviderResponse> {
  const { audioInputs, resolvedProvider, config, options, deps, signal } = input;
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
    audioInputs.map(async (audioInput) => ({
      ...audioInput,
      fileBuffer: await readAudioFile(audioInput.filePath),
    }))
  );
  const headers = {
    'x-goog-api-key': String(config?.key || '').trim(),
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
    const errorMessage = extractCloudAsrErrorMessage(requestResult.rawText, requestResult.response.statusText);
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

  const totalDurationSec = audioInputs.reduce((sum, audioInput) => {
    const durationSec = Number(audioInput.audioDurationSec);
    return sum + (Number.isFinite(durationSec) && durationSec > 0 ? durationSec : 0);
  }, 0);
  const geminiFreeTierLimits = getGeminiFreeTierLimits();
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
      applied: geminiFreeTierLimits.enabled,
      ...geminiFreeTierLimits,
      audioTokensPerSecond: GEMINI_AUDIO_TOKENS_PER_SECOND,
      estimatedInputTokens,
    },
    cjkWordDiagnostics: null,
  };

  return {
    meta,
    results: inputs.map((audioInput, position) => {
      const item = parsed.normalized[position] || { text: '', speaker: undefined };
      const transcript = buildGeminiBatchTranscriptFromItem(item, audioInput.audioDurationSec);
      return {
        input: audioInput,
        result: {
          ...transcript,
          meta: {
            ...meta,
            batchItemIndex: position,
            batchInputIndex: audioInput.index,
            rawSegmentCount: transcript.chunks.length,
          },
        },
      };
    }),
  };
}

export const googleGeminiProvider = createCloudAsrProviderDefinition({
  provider: 'google-gemini',
  defaultModel: 'gemini-2.5-flash',
  runtime: googleGeminiAudioRuntime,
  batchRequest: requestGeminiCloudAsrBatch,
  detect(input) {
    return input.hostname === 'generativelanguage.googleapis.com' ||
      input.pathname.includes(':generatecontent') ||
      (input.modelName.includes('gemini') &&
        (input.modelName.includes('asr') || input.modelName.includes('audio') || input.modelName.includes('transcri')));
  },
  buildEndpointUrl(rawUrl, model) {
    const parsed = parseCloudAsrUrl(rawUrl);
    if (parsed.pathname.toLowerCase().includes(':generatecontent')) return parsed.toString();
    const targetModel = String(model || '').trim() || 'gemini-2.5-flash';
    return ensureEndpointPath(parsed, `/v1beta/models/${encodeURIComponent(targetModel)}:generateContent`).toString();
  },
  buildHeaders(key): Record<string, string> {
    const trimmed = String(key || '').trim();
    return trimmed ? { 'x-goog-api-key': trimmed } : {};
  },
  buildConnectionProbe() {
    return {
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
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
      }),
      expectedValidation: 'generic',
      timeoutMs: 18_000,
    };
  },
});
