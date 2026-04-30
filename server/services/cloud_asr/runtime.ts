import { requireCloudAsrProviderDefinition } from './registry.js';
import { buildCloudAsrRequestHeaders } from './resolver.js';
import type {
  CloudAsrAdapter,
  CloudAsrAdapterDeps,
  CloudAsrAdapterRequestOptions,
  CloudAsrBatchAudioInput,
  CloudAsrBatchProviderResponse,
  CloudAsrProvider,
  CloudAsrProviderResult,
  ResolvedCloudAsrProvider,
} from './types.js';
import {
  enforceGeminiFreeTierRateLimit,
  estimateGeminiAudioAsrInputTokens,
  GEMINI_AUDIO_TOKENS_PER_SECOND,
  getGeminiFreeTierLimits,
} from './providers/google_gemini_audio.js';
import {
  extractCloudAsrErrorMessage,
  getCloudAsrRequestTimeoutMs,
  getTransientRetryDelayMs,
  readAudioFile,
  shouldRetryTransientCloudAsrResponse,
  sleepWithAbort,
} from './runtime/shared.js';

export { buildCloudAsrRequestHeaders } from './resolver.js';

export type {
  CloudAsrAdapter,
  CloudAsrAdapterDeps,
  CloudAsrAdapterRequestOptions,
  CloudAsrBatchAudioInput,
  CloudAsrBatchProviderResponse,
  CloudAsrProviderResult,
  CloudAsrStructuredTranscript,
} from './types.js';

export function getCloudAsrAdapter(provider: CloudAsrProvider): CloudAsrAdapter {
  const adapter = requireCloudAsrProviderDefinition(provider).runtime;
  if (!adapter) {
    throw new Error(`Cloud ASR provider "${provider}" does not expose a generic runtime adapter.`);
  }
  return adapter;
}

export async function requestGeminiCloudAsrBatch(
  audioInputs: CloudAsrBatchAudioInput[],
  resolvedProvider: ResolvedCloudAsrProvider,
  config: any,
  options: CloudAsrAdapterRequestOptions,
  deps: Pick<CloudAsrAdapterDeps, 'createAbortSignalWithTimeout'>,
  signal?: AbortSignal
): Promise<CloudAsrBatchProviderResponse> {
  const definition = requireCloudAsrProviderDefinition(resolvedProvider.provider);
  if (!definition.batchRequest) {
    throw new Error(`Cloud ASR provider "${resolvedProvider.provider}" does not support batch audio requests.`);
  }
  return definition.batchRequest({
    audioInputs,
    resolvedProvider,
    config,
    options,
    deps,
    signal,
  });
}

function hasProviderSpecificRequest(provider: CloudAsrProvider) {
  return Boolean(requireCloudAsrProviderDefinition(provider).request);
}

export async function requestCloudAsr(
  filePath: string,
  resolvedProvider: ResolvedCloudAsrProvider,
  config: any,
  options: CloudAsrAdapterRequestOptions,
  deps: CloudAsrAdapterDeps,
  signal?: AbortSignal
): Promise<CloudAsrProviderResult> {
  const definition = requireCloudAsrProviderDefinition(resolvedProvider.provider);
  if (definition.request) {
    return definition.request({
      filePath,
      resolvedProvider,
      config,
      options,
      deps,
      signal,
    });
  }

  const { language, wordAlignment } = options;
  const rawLanguage = typeof language === 'string' ? language.trim() : '';
  const normalizedLanguage = rawLanguage.toLowerCase();
  const adapter = definition.runtime;
  if (!adapter) {
    throw new Error(`Cloud ASR provider "${resolvedProvider.provider}" does not expose a request implementation.`);
  }

  const fileBuffer = await readAudioFile(filePath);
  const preferredFormats = adapter.getPreferredResponseFormats(options);
  const headers = buildCloudAsrRequestHeaders(resolvedProvider.provider, config.key);
  const geminiEstimatedInputTokens = resolvedProvider.provider === 'google-gemini'
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
    if (resolvedProvider.provider === 'google-gemini') {
      await enforceGeminiFreeTierRateLimit(config, resolvedProvider, geminiEstimatedInputTokens, signal);
    }
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
  const maxTransientRetries = resolvedProvider.provider === 'google-gemini' ? 3 : 2;

  const sendWithTransientRetry = async (includeLanguage: boolean, responseFormat: string) => {
    let attemptIndex = 0;
    while (true) {
      const result = await sendOnce(includeLanguage, responseFormat);
      if (!shouldRetryTransientCloudAsrResponse(result.response) || attemptIndex >= maxTransientRetries) {
        return result;
      }
      transientRetryCount += 1;
      await sleepWithAbort(getTransientRetryDelayMs(result.response, attemptIndex), signal);
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
      firstErrorSummary = `first_attempt_${requestResult.response.status}: ${extractCloudAsrErrorMessage(
        requestResult.rawText,
        requestResult.response.statusText
      )}`;
      requestResult = await sendWithTransientRetry(false, format);
    }

    if (requestResult.response.ok) break;

    lastErrorForFormat = extractCloudAsrErrorMessage(requestResult.rawText, requestResult.response.statusText);
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
    const errorMessage =
      lastErrorForFormat || extractCloudAsrErrorMessage(requestResult.rawText, requestResult.response.statusText);
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
  const providerSpecificMeta = adapter.buildResultMeta
    ? adapter.buildResultMeta({
        normalizedData,
        transcript,
        options,
        resolvedProvider,
        requestTimeoutMs: cloudRequestTimeoutMs,
      })
    : {};
  const geminiFreeTierLimits =
    resolvedProvider.provider === 'google-gemini' ? getGeminiFreeTierLimits() : null;

  return {
    ...transcript,
    meta: {
      provider: resolvedProvider.provider,
      providerKey: resolvedProvider.providerKey,
      endpointUrl: resolvedProvider.endpointUrl,
      effectiveModel: resolvedProvider.effectiveModel,
      profileId: resolvedProvider.profileId,
      capabilities: resolvedProvider.capabilities,
      responseFormat: requestResult.responseFormat,
      rawHasTimestamps,
      rawSegmentCount: rawSegments.length,
      rawWordCount: transcript.word_segments.length,
      requestedLanguage: rawLanguage || null,
      sentLanguage: requestResult.includeLanguage ? rawLanguage : null,
      detectedLanguage: normalizedData?.language_code || null,
      ...providerSpecificMeta,
      autoLanguageFallbackUsed: attemptedAutoFallback,
      responseFormatFallbackUsed: attemptedFormatFallback,
      transientRetryCount,
      geminiFreeTierLimiter: resolvedProvider.provider === 'google-gemini'
        ? {
            applied: Boolean(geminiFreeTierLimits?.enabled),
            ...geminiFreeTierLimits,
            audioTokensPerSecond: GEMINI_AUDIO_TOKENS_PER_SECOND,
            estimatedInputTokens: geminiEstimatedInputTokens,
          }
        : null,
      cjkWordDiagnostics: transcript.debug?.cjkWordDiagnostics || null,
      providerHasDedicatedRequest: hasProviderSpecificRequest(resolvedProvider.provider),
    },
  };
}
