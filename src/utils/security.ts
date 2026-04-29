/**
 * Security Utility Functions
 * Provides input validation, sanitization, and secure data handling.
 */

export function sanitizeInput(input: string): string {
  if (!input) return '';
  return input.trim().replace(/[<>]/g, '').slice(0, 1000);
}

export function isValidUrl(url: string): boolean {
  if (!url) return false;
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
}

export function isValidProjectName(name: string): boolean {
  if (!name) return false;
  const sanitized = name.trim();
  return sanitized.length >= 1 && sanitized.length <= 100 && !/[\\/:*?"<>|]/.test(sanitized);
}

export function isValidApiKey(key: string): boolean {
  if (!key) return true;
  return key.length >= 8 && key.length <= 4096;
}

export function isMaskedApiKey(key: string): boolean {
  const v = String(key || '').trim();
  if (!v) return false;
  if (v.includes('****') || v.includes('•')) return true;
  if (v.includes(`?\uFF34`) || v.includes('\u0080')) return true; // legacy mojibake compatibility
  if (/^[A-Za-z0-9._-]{2,12}[*•?]{3,}[A-Za-z0-9._-]{0,12}$/.test(v)) return true;
  if (/[^\x20-\x7E]/.test(v) && v.length <= 64) return true;
  return false;
}

export function maskApiKey(key: string): string {
  if (!key) return '';
  if (isMaskedApiKey(key)) return key;
  if (key.length <= 8) return '****';
  return `${key.slice(0, 4)}****${key.slice(-4)}`;
}
