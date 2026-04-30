import { extractErrorMessage, hasAuthFailureHint, hasPayloadValidationHint } from '../../http/text_utils.js';
import { requireCloudAsrProviderDefinition } from './registry.js';
import { resolveCloudAsrProvider } from './resolver.js';

export interface TestCloudAsrConnectionInput {
  url: string;
  key?: string;
  model?: string;
  name?: string;
}

function createAbortSignalWithTimeout(timeoutMs: number) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  return {
    signal: controller.signal,
    dispose: () => clearTimeout(timeout),
  };
}

function isValidationStatus(status: number, allowWhisperCppServerError = false) {
  return (
    status === 400 ||
    status === 415 ||
    status === 422 ||
    (allowWhisperCppServerError && status === 500)
  );
}

function matchesExpectedValidation(expectedValidation: string | null | undefined, detail: string) {
  const payloadValidationFailed = hasPayloadValidationHint(detail);
  if (expectedValidation === 'google') {
    return payloadValidationFailed || /audio|content|recognize|recognition|config/i.test(detail || '');
  }
  if (expectedValidation === 'deepgram') {
    return payloadValidationFailed || /audio|media|source|decode|encoding|listen/i.test(detail || '');
  }
  if (expectedValidation === 'gladia') {
    return payloadValidationFailed || /audio_url|audio|url|required|invalid|missing/i.test(detail || '');
  }
  return payloadValidationFailed;
}

function catalogContainsModel(responseText: string, modelId: string) {
  const expected = String(modelId || '').trim().toLowerCase();
  if (!expected) return true;
  try {
    const parsed = JSON.parse(responseText);
    return Array.isArray(parsed) && parsed.some((item: any) => String(item?.id || '').trim().toLowerCase() === expected);
  } catch {
    return false;
  }
}

export async function testCloudAsrConnection(input: TestCloudAsrConnectionInput) {
  const resolvedProvider = resolveCloudAsrProvider({
    url: String(input.url || '').trim(),
    modelName: String(input.name || '').trim(),
    model: String(input.model || '').trim(),
  });
  const definition = requireCloudAsrProviderDefinition(resolvedProvider.provider);
  const providerHeaders = definition.buildHeaders(input.key);
  const probe = definition.buildConnectionProbe?.(resolvedProvider) || {};
  const testUrl = probe.url || resolvedProvider.endpointUrl;
  const testMethod = probe.method || 'POST';
  const testHeaders = {
    ...providerHeaders,
    ...(probe.headers || {}),
  };
  const testBody = probe.body;
  const timeoutMs = Number.isFinite(Number(probe.timeoutMs)) && Number(probe.timeoutMs) > 0
    ? Math.round(Number(probe.timeoutMs))
    : 8000;

  const request = createAbortSignalWithTimeout(timeoutMs);
  try {
    const response = await fetch(testUrl, {
      method: testMethod,
      headers: testHeaders,
      body: testMethod === 'GET' ? undefined : testBody,
      signal: request.signal,
      redirect: 'error',
    });

    const contentType = String(response.headers.get('content-type') || '');
    const responseText = await response.text();
    const detail = extractErrorMessage(responseText, contentType);
    const authFailed = hasAuthFailureHint(detail);

    if (response.status >= 200 && response.status < 300) {
      if (
        probe.expectedCatalogModelId &&
        !catalogContainsModel(responseText, probe.expectedCatalogModelId)
      ) {
        return {
          success: false,
          error: `Model not found in provider catalog: ${probe.expectedCatalogModelId}. Please verify the model ID.`,
        };
      }
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
      const retryAfter = response.headers.get('retry-after');
      const retryHint = retryAfter ? ` Retry-After: ${retryAfter}.` : '';
      return {
        success: false,
        error: `Rate limited (429). Please retry later.${retryHint}${detail ? ` ${detail}` : ''}`,
      };
    }

    const allowServerValidation = resolvedProvider.provider === 'whispercpp-inference';
    if (
      isValidationStatus(response.status, allowServerValidation) &&
      matchesExpectedValidation(probe.expectedValidation || 'generic', detail)
    ) {
      return {
        success: true,
        message: 'Endpoint reachable. Request payload validation failed as expected.',
      };
    }

    if (response.status >= 500 && !allowServerValidation) {
      return {
        success: false,
        error: `Provider server error (${response.status}).${detail ? ` ${detail}` : ''}`,
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
