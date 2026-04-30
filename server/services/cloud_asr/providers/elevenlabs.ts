import path from 'node:path';
import type { CloudAsrAdapter, CloudAsrAdapterRequestOptions } from '../types.js';
import { bufferToBlobPart, toFiniteNumber } from '../runtime/shared.js';
import { createCloudAsrProviderDefinition, ensureEndpointPath, parseCloudAsrUrl } from './shared.js';

function hasScribeModelHint(value: string) {
  return /(^|[^a-z0-9])scribe(?:[\s._-]?v?\d+)?($|[^a-z0-9])/i.test(value);
}

function normalizeElevenLabsLanguageCode(language: string) {
  const normalized = String(language || '').trim().toLowerCase().replace(/_/g, '-');
  if (!normalized || normalized === 'auto' || normalized === 'detect') return '';
  if (normalized.startsWith('zh-')) return 'zh';
  if (normalized === 'jp') return 'ja';
  const primary = normalized.split('-')[0];
  return /^[a-z]{2,3}$/.test(primary) ? primary : '';
}

export function getElevenLabsNumSpeakers(options: CloudAsrAdapterRequestOptions) {
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

export function isElevenLabsBoundaryText(text: string) {
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

export function joinElevenLabsWords(words: Array<{ text: string }>) {
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

const elevenLabsRuntime: CloudAsrAdapter = {
  provider: 'elevenlabs',
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
  buildResultMeta({ normalizedData, options, transcript, requestTimeoutMs }) {
    return {
      nativeWordTimestamps: transcript.word_segments.length > 0,
      providerNativeDiarization: Boolean(normalizedData?.elevenlabs_diarization_applied),
      elevenLabs: {
        diarizationRequested: Boolean(options.diarization),
        numSpeakersHint: getElevenLabsNumSpeakers(options),
        diarizationApplied: Boolean(normalizedData?.elevenlabs_diarization_applied),
        speakerCount: toFiniteNumber(normalizedData?.elevenlabs_speaker_count, 0),
        wordCount: toFiniteNumber(normalizedData?.elevenlabs_word_count, transcript.word_segments.length),
        audioEventsTagged: false,
        requestTimeoutMs,
        standardModeLimitHours: 10,
        uploadLimitGb: 3,
      },
    };
  },
};

export const elevenLabsProvider = createCloudAsrProviderDefinition({
  provider: 'elevenlabs',
  defaultModel: 'scribe_v2',
  runtime: elevenLabsRuntime,
  async preflight(input) {
    const fileInfo = await input.getFileInfo();
    const maxSizeBytes = 3 * 1024 * 1024 * 1024;
    const maxDurationSec = 10 * 60 * 60;
    if (fileInfo.sizeBytes != null && fileInfo.sizeBytes > maxSizeBytes) {
      return { action: 'reject', error: new Error('ElevenLabs Scribe upload limit exceeded (3GB maximum).') };
    }
    if (fileInfo.durationSec != null && fileInfo.durationSec > maxDurationSec) {
      return { action: 'reject', error: new Error('ElevenLabs Scribe audio duration limit exceeded (10 hours maximum).') };
    }
    return { action: 'direct', skipGenericPreemptiveChunking: true };
  },
  detect(input) {
    return /(^|\.)elevenlabs\.io$/.test(input.hostname) ||
      input.pathname.includes('/speech-to-text') ||
      input.modelName.includes('elevenlabs') ||
      hasScribeModelHint(input.modelName);
  },
  buildEndpointUrl(rawUrl) {
    const parsed = parseCloudAsrUrl(rawUrl);
    if (parsed.pathname.toLowerCase().includes('/speech-to-text')) return parsed.toString();
    return ensureEndpointPath(parsed, '/v1/speech-to-text').toString();
  },
  buildHeaders(key): Record<string, string> {
    const trimmed = String(key || '').trim();
    return trimmed ? { 'xi-api-key': trimmed } : {};
  },
});
