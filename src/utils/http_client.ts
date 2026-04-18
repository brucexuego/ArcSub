export class HttpRequestError extends Error {
  status: number;
  bodyText: string;

  constructor(message: string, status: number, bodyText: string) {
    super(message);
    this.status = status;
    this.bodyText = bodyText;
  }
}

type JsonLike = Record<string, unknown> | unknown[] | string | number | boolean | null;

interface JsonRequestOptions extends Omit<RequestInit, 'body' | 'method'> {
  body?: JsonLike;
  timeoutMs?: number;
  retries?: number;
  retryDelayMs?: number;
  dedupe?: boolean;
  dedupeKey?: string;
  cancelPreviousKey?: string;
}

const inFlightGetRequests = new Map<string, Promise<unknown>>();
const cancelableRequestControllers = new Map<string, AbortController>();
const cachedGetResponses = new Map<string, unknown>();
const cachedGetResponsesByUrl = new Map<string, unknown>();

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryableStatus(status: number) {
  return status === 408 || status === 429 || (status >= 500 && status < 600);
}

function mergeSignals(primary: AbortSignal | undefined, secondary: AbortSignal | undefined) {
  if (primary && secondary) {
    if (typeof AbortSignal.any === 'function') {
      return AbortSignal.any([primary, secondary]);
    }
    const fallback = new AbortController();
    const relay = () => fallback.abort();
    primary.addEventListener('abort', relay, { once: true });
    secondary.addEventListener('abort', relay, { once: true });
    return fallback.signal;
  }
  return primary || secondary;
}

async function requestJsonInternal<T>(
  method: 'GET' | 'POST' | 'PATCH' | 'DELETE',
  url: string,
  options: JsonRequestOptions = {}
): Promise<T> {
  const {
    body,
    timeoutMs = 10_000,
    retries = method === 'GET' ? 1 : 0,
    retryDelayMs = 300,
    dedupe = method === 'GET',
    dedupeKey = `${method}:${url}`,
    cancelPreviousKey,
    headers,
    signal,
    ...rest
  } = options;

  const existing = dedupe ? inFlightGetRequests.get(dedupeKey) : null;
  if (existing) {
    return existing as Promise<T>;
  }

  const requestTask = (async () => {
    let attempt = 0;
    let tried304Bypass = false;
    while (true) {
      const timeoutController = new AbortController();
      const timeoutId = setTimeout(() => timeoutController.abort(), timeoutMs);
      let requestController: AbortController | null = null;

      try {
        if (cancelPreviousKey) {
          cancelableRequestControllers.get(cancelPreviousKey)?.abort();
          requestController = new AbortController();
          cancelableRequestControllers.set(cancelPreviousKey, requestController);
        }

        const finalSignal = mergeSignals(signal, mergeSignals(timeoutController.signal, requestController?.signal));
        const finalHeaders = new Headers(headers || {});
        let serializedBody: string | undefined;
        if (body !== undefined) {
          serializedBody = JSON.stringify(body);
          if (!finalHeaders.has('Content-Type')) {
            finalHeaders.set('Content-Type', 'application/json');
          }
        }

        const requestInit: RequestInit = {
          ...rest,
          method,
          headers: finalHeaders,
          body: serializedBody,
          signal: finalSignal,
        };

        // When server replies 304 but this runtime has no cached body yet,
        // retry once with cache bypass to force a full payload.
        if (method === 'GET' && tried304Bypass) {
          requestInit.cache = 'no-store';
          finalHeaders.delete('If-None-Match');
          finalHeaders.delete('If-Modified-Since');
        }

        const response = await fetch(url, requestInit);

        const responseText = await response.text();
        const parseResponseJson = () => {
          if (!responseText) return null as T;
          try {
            return JSON.parse(responseText) as T;
          } catch {
            return responseText as T;
          }
        };

        if (response.ok) {
          const parsed = parseResponseJson();
          if (method === 'GET') {
            cachedGetResponses.set(dedupeKey, parsed);
            cachedGetResponsesByUrl.set(url, parsed);
          }
          return parsed;
        }

        if (method === 'GET' && response.status === 304) {
          if (cachedGetResponses.has(dedupeKey)) {
            return cachedGetResponses.get(dedupeKey) as T;
          }
          if (cachedGetResponsesByUrl.has(url)) {
            return cachedGetResponsesByUrl.get(url) as T;
          }
          if (!tried304Bypass) {
            tried304Bypass = true;
            continue;
          }
          return parseResponseJson();
        }

        if (attempt < retries && isRetryableStatus(response.status)) {
          await sleep(retryDelayMs * (attempt + 1));
          attempt += 1;
          continue;
        }

        throw new HttpRequestError(
          `Request failed (${response.status})`,
          response.status,
          responseText
        );
      } catch (error: any) {
        const isAbort = error?.name === 'AbortError';
        if (!isAbort && attempt < retries) {
          await sleep(retryDelayMs * (attempt + 1));
          attempt += 1;
          continue;
        }
        throw error;
      } finally {
        clearTimeout(timeoutId);
        if (cancelPreviousKey) {
          const current = cancelableRequestControllers.get(cancelPreviousKey);
          if (current === requestController) {
            cancelableRequestControllers.delete(cancelPreviousKey);
          }
        }
      }
    }
  })();

  if (dedupe) {
    inFlightGetRequests.set(dedupeKey, requestTask as Promise<unknown>);
  }

  try {
    return await requestTask;
  } finally {
    if (dedupe) {
      inFlightGetRequests.delete(dedupeKey);
    }
  }
}

export function getJson<T>(url: string, options: Omit<JsonRequestOptions, 'body'> = {}) {
  return requestJsonInternal<T>('GET', url, options);
}

export function postJson<T>(url: string, body: JsonLike, options: Omit<JsonRequestOptions, 'body'> = {}) {
  return requestJsonInternal<T>('POST', url, { ...options, body, dedupe: false });
}

export function patchJson<T>(url: string, body: JsonLike, options: Omit<JsonRequestOptions, 'body'> = {}) {
  return requestJsonInternal<T>('PATCH', url, { ...options, body, dedupe: false });
}

export function deleteJson<T>(url: string, options: Omit<JsonRequestOptions, 'body'> = {}) {
  return requestJsonInternal<T>('DELETE', url, { ...options, dedupe: false });
}

