import type { CloudAsrAdapter, CloudAsrAdapterBuildInput } from '../types.js';
import {
  bufferToBlobPart,
  firstString,
  getEnvNumber,
  inferAudioMimeType,
  toFiniteNumber,
} from '../runtime/shared.js';
import { isDeepgramChunkableError } from '../profiles/errors.js';
import { isElevenLabsBoundaryText, joinElevenLabsWords } from './elevenlabs.js';
import { createCloudAsrProviderDefinition, ensureEndpointPath, parseCloudAsrUrl } from './shared.js';

function normalizeDeepgramLanguageCode(language: string) {
  const normalized = String(language || '').trim().toLowerCase().replace(/_/g, '-');
  if (!normalized || normalized === 'auto' || normalized === 'detect') return '';
  if (normalized === 'jp') return 'ja';
  const primary = normalized.split('-')[0];
  return /^[a-z]{2,3}$/.test(primary) ? primary : normalized;
}

export function getDeepgramUtteranceSplitSec() {
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

function buildDeepgramListenUrl(endpointUrl: string, input: CloudAsrAdapterBuildInput) {
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

const deepgramRuntime: CloudAsrAdapter = {
  provider: 'deepgram',
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
  buildResultMeta({ normalizedData, options, transcript, resolvedProvider, requestTimeoutMs }) {
    return {
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
        requestTimeoutMs,
        uploadLimitGb: 2,
        maxProcessingTimeSec: String(resolvedProvider.effectiveModel || '').toLowerCase().includes('whisper') ? 1200 : 600,
      },
    };
  },
};

export const deepgramProvider = createCloudAsrProviderDefinition({
  provider: 'deepgram',
  defaultModel: 'nova-3',
  runtime: deepgramRuntime,
  async preflight(input) {
    const fileInfo = await input.getFileInfo();
    const maxSizeBytes = 2 * 1024 * 1024 * 1024;
    if (fileInfo.sizeBytes != null && fileInfo.sizeBytes > maxSizeBytes) {
      return { action: 'reject', error: new Error('Deepgram listen upload limit exceeded (2GB maximum).') };
    }
    return { action: 'direct', skipGenericPreemptiveChunking: true };
  },
  shouldChunkOnError(input) {
    return isDeepgramChunkableError(input.error);
  },
  detect(input) {
    return /(^|\.)deepgram\.com$/.test(input.hostname) ||
      input.pathname.includes('/listen') ||
      input.modelName.includes('deepgram') ||
      input.modelName.includes('nova-');
  },
  buildEndpointUrl(rawUrl) {
    const parsed = parseCloudAsrUrl(rawUrl);
    if (parsed.pathname.toLowerCase().includes('/listen')) return parsed.toString();
    return ensureEndpointPath(parsed, '/v1/listen').toString();
  },
  buildHeaders(key): Record<string, string> {
    const trimmed = String(key || '').trim();
    return trimmed ? { Authorization: `Token ${trimmed}` } : {};
  },
  buildConnectionProbe(input) {
    const nextUrl = new URL(input.endpointUrl);
    nextUrl.searchParams.set('model', input.effectiveModel);
    nextUrl.searchParams.set('smart_format', 'true');
    return {
      url: nextUrl.toString(),
      headers: { 'Content-Type': 'audio/wav', Accept: 'application/json' },
      body: new Blob([new Uint8Array(0)], { type: 'audio/wav' }),
      expectedValidation: 'deepgram',
    };
  },
});
