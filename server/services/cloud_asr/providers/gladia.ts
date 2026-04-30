import path from 'node:path';
import type {
  CloudAsrAdapterDeps,
  CloudAsrAdapterRequestOptions,
  CloudAsrProviderRequestInput,
  CloudAsrProviderResult,
} from '../types.js';
import {
  bufferToBlobPart,
  extractCloudAsrErrorMessage,
  firstString,
  getCloudAsrRequestTimeoutMs,
  getEnvNumber,
  getTransientRetryDelayMs,
  inferAudioMimeType,
  parseRetryAfterMs,
  readAudioFile,
  shouldRetryTransientCloudAsrResponse,
  sleepWithAbort,
  stringifyErrorValue,
  toFiniteNumber,
} from '../runtime/shared.js';
import { isGladiaChunkableError } from '../profiles/errors.js';
import { createCloudAsrProviderDefinition, ensureEndpointPath, parseCloudAsrUrl } from './shared.js';

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
    transcriptTexts.push(...segments.map((segment: any) => segment.text).filter(Boolean));
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
    throw new Error(`Cloud ASR error (${response.status}): ${extractCloudAsrErrorMessage(rawText, fallback || response.statusText)}`);
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

async function requestGladiaPreRecordedAsr(input: CloudAsrProviderRequestInput): Promise<CloudAsrProviderResult> {
  const { filePath, resolvedProvider, config, options, deps, signal } = input;
  const rawLanguage = typeof options.language === 'string' ? options.language.trim() : '';
  const fileBuffer = await readAudioFile(filePath);
  const cloudRequestTimeoutMs = getCloudAsrRequestTimeoutMs(resolvedProvider, options);
  const headers = gladiaProvider.buildHeaders(config.key);
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
      const errorMessage = stringifyErrorValue([
        data?.error,
        data?.message,
        data?.error_message,
        data?.detail,
        data?.errors,
      ]) || `Gladia job failed${data?.error_code ? ` (${data.error_code})` : ''}.`;
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

export const gladiaProvider = createCloudAsrProviderDefinition({
  provider: 'gladia',
  defaultModel: 'gladia-v2',
  request: requestGladiaPreRecordedAsr,
  async preflight(input) {
    const policy = input.chunkingPolicy;
    const fileInfo = await input.getFileInfo();
    const maxSizeBytes = 1000 * 1024 * 1024;
    const proactiveSizeBytes = Math.floor(maxSizeBytes * 0.95);
    const proactiveDurationSec = 130 * 60;
    if (policy && fileInfo.sizeBytes != null && fileInfo.sizeBytes >= proactiveSizeBytes) {
      return {
        action: 'chunk',
        chunkingPolicy: policy,
        message: `Gladia ASR input is near provider limits (file_size_${(fileInfo.sizeBytes / (1024 * 1024)).toFixed(1)}mb); using ${Math.round(Math.max(policy.minChunkSec || 600, policy.initialChunkSec || 3600) / 60)} minute MP3 chunks...`,
      };
    }
    if (policy && fileInfo.durationSec != null && fileInfo.durationSec >= proactiveDurationSec) {
      return {
        action: 'chunk',
        chunkingPolicy: policy,
        message: `Gladia ASR input is near provider limits (duration_${(fileInfo.durationSec / 60).toFixed(1)}min); using ${Math.round(Math.max(policy.minChunkSec || 600, policy.initialChunkSec || 3600) / 60)} minute MP3 chunks...`,
      };
    }
    return { action: 'direct', skipGenericPreemptiveChunking: true };
  },
  shouldChunkOnError(input) {
    return isGladiaChunkableError(input.error);
  },
  detect(input) {
    return /(^|\.)gladia\.io$/.test(input.hostname) ||
      input.pathname.includes('/v2/pre-recorded') ||
      input.pathname.includes('/v2/upload') ||
      input.modelName.includes('gladia');
  },
  buildEndpointUrl(rawUrl) {
    const parsed = parseCloudAsrUrl(rawUrl);
    const path = parsed.pathname.toLowerCase();
    if (path.includes('/v2/pre-recorded')) return parsed.toString();
    if (path.replace(/\/+$/, '') === '/v2') {
      const next = new URL(parsed.toString());
      next.pathname = '/v2/pre-recorded';
      return next.toString();
    }
    if (path.includes('/v2/upload') || path.includes('/v2/transcription')) {
      const next = new URL(parsed.toString());
      next.pathname = '/v2/pre-recorded';
      return next.toString();
    }
    return ensureEndpointPath(parsed, '/v2/pre-recorded').toString();
  },
  buildHeaders(key): Record<string, string> {
    const trimmed = String(key || '').trim();
    return trimmed ? { 'x-gladia-key': trimmed } : {};
  },
  buildConnectionProbe() {
    return {
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
      expectedValidation: 'gladia',
    };
  },
});
