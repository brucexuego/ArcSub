function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function firstString(values: unknown[]) {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) return value.trim();
  }
  return '';
}

export function extractCloudTranslateErrorMessage(raw: unknown, fallback = 'Provider request failed.') {
  if (typeof raw === 'string') {
    const text = raw.trim();
    if (!text) return fallback;
    try {
      return extractCloudTranslateErrorMessage(JSON.parse(text), text);
    } catch {
      return text;
    }
  }

  if (!isPlainObject(raw)) return fallback;
  const error = raw.error;
  if (typeof error === 'string' && error.trim()) return error.trim();
  if (isPlainObject(error)) {
    const message = firstString([
      error.message,
      error.detail,
      error.error_description,
      error.error,
      raw.message,
    ]);
    if (message) {
      const status = firstString([error.status, error.code]);
      return status && !message.includes(status) ? `${status}: ${message}` : message;
    }
  }

  const details = Array.isArray((raw as any).details) ? (raw as any).details : [];
  const detailMessage = details
    .map((detail) => isPlainObject(detail) ? firstString([detail.reason, detail.message, detail.type]) : '')
    .filter(Boolean)
    .join('; ');
  if (detailMessage) return detailMessage;

  return firstString([raw.message, raw.detail, raw.error_description]) || fallback;
}

export function extractErrorStatus(raw: unknown): number | undefined {
  const candidates: unknown[] = [];
  if (isPlainObject(raw)) {
    candidates.push(raw.status, raw.statusCode, raw.code);
    if (isPlainObject(raw.error)) {
      candidates.push(raw.error.status, raw.error.statusCode, raw.error.code);
    }
  }
  for (const candidate of candidates) {
    const parsed = Number(candidate);
    if (Number.isFinite(parsed) && parsed >= 100 && parsed <= 599) {
      return Math.round(parsed);
    }
  }
  return undefined;
}
