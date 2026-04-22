import fs from 'fs-extra';
import path from 'node:path';
import type { CloudAsrProvider, ResolvedCloudAsrProvider } from './cloud_asr_provider.js';

export interface CloudAsrAdapterRequestOptions {
  language?: string;
  prompt?: string;
  segmentation?: boolean;
  wordAlignment?: boolean;
  vad?: boolean;
  diarization?: boolean;
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
  buildFormData(input: CloudAsrAdapterBuildFormInput): FormData;
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

const cloudAsrAdapters: Record<CloudAsrProvider, CloudAsrAdapter> = {
  'openai-whisper': {
    provider: 'openai-whisper',
    getPreferredResponseFormats(options) {
      return [options.segmentation !== false ? 'verbose_json' : 'json'];
    },
    buildFormData(input) {
      const { filePath, fileBuffer, config, options, includeLanguage, responseFormat } = input;
      const blob = new Blob([fileBuffer]);
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
      const blob = new Blob([fileBuffer]);
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
};

export function getCloudAsrAdapter(provider: CloudAsrProvider) {
  return cloudAsrAdapters[provider];
}

export async function requestCloudAsr(
  filePath: string,
  resolvedProvider: ResolvedCloudAsrProvider,
  config: any,
  options: CloudAsrAdapterRequestOptions,
  deps: CloudAsrAdapterDeps,
  signal?: AbortSignal
): Promise<CloudAsrProviderResult> {
  const { language, wordAlignment } = options;
  const rawLanguage = typeof language === 'string' ? language.trim() : '';
  const normalizedLanguage = rawLanguage.toLowerCase();
  const adapter = getCloudAsrAdapter(resolvedProvider.provider);

  const fileBuffer = await fs.readFile(filePath);
  const preferredFormats = adapter.getPreferredResponseFormats(options);
  const headers: Record<string, string> = {};
  const cloudRequestTimeoutMs = (() => {
    const parsed = Number(process.env.ASR_CLOUD_REQUEST_TIMEOUT_MS || 120000);
    if (!Number.isFinite(parsed)) return 120000;
    return Math.max(15000, Math.min(1800000, Math.round(parsed)));
  })();
  if (config.key) {
    headers.Authorization = `Bearer ${config.key}`;
  }

  const sendOnce = async (includeLanguage: boolean, responseFormat: string) => {
    const request = deps.createAbortSignalWithTimeout(cloudRequestTimeoutMs, signal);
    try {
      const response = await fetch(resolvedProvider.endpointUrl, {
        method: 'POST',
        headers,
        body: adapter.buildFormData({
          filePath,
          fileBuffer,
          config,
          options,
          includeLanguage,
          responseFormat,
        }),
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

  for (let i = 0; i < preferredFormats.length; i += 1) {
    const format = preferredFormats[i];
    requestResult = await sendOnce(Boolean(rawLanguage), format);
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
      requestResult = await sendOnce(false, format);
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
    throw new Error(`Cloud ASR error (${requestResult.response.status}): ${errorMessage}${fallbackHint}`);
  }

  let data: any = null;
  try {
    data = JSON.parse(requestResult.rawText);
  } catch {
    data = { text: requestResult.rawText };
  }

  let transcript = deps.extractStructuredTranscript(data, rawLanguage);
  if (wordAlignment === false && transcript.word_segments.length > 0) {
    transcript = deps.disableWordAlignment(transcript, rawLanguage);
  }
  if (transcript.chunks.length === 0) {
    throw new Error('Cloud ASR response does not contain transcript text.');
  }

  const rawSegments = Array.isArray(data?.segments)
    ? data.segments
    : Array.isArray(data?.chunks)
      ? data.chunks
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
      responseFormat: requestResult.responseFormat,
      rawHasTimestamps,
      rawSegmentCount: rawSegments.length,
      rawWordCount: transcript.word_segments.length,
      requestedLanguage: rawLanguage || null,
      sentLanguage: requestResult.includeLanguage ? rawLanguage : null,
      autoLanguageFallbackUsed: attemptedAutoFallback,
      responseFormatFallbackUsed: attemptedFormatFallback,
      cjkWordDiagnostics: transcript.debug?.cjkWordDiagnostics || null,
    },
  };
}
