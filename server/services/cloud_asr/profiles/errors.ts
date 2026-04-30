export function parseCloudAsrStatus(error: unknown): number | null {
  const message = error instanceof Error ? error.message : String(error || '');
  const match = message.match(/Cloud ASR error \((\d{3})\):/i);
  if (!match) return null;
  const status = Number(match[1]);
  return Number.isFinite(status) ? status : null;
}

export function isCloudAsrFileLimitError(error: unknown): boolean {
  const message = (error instanceof Error ? error.message : String(error || '')).toLowerCase();
  const status = parseCloudAsrStatus(error);
  if (status === 413) return true;
  const fileLimitHint = /(too large|payload too large|entity too large|request body too large|max(?:imum)? (?:file|audio|payload|content)|file size|audio length|duration limit|too long)/i;
  if ((status === 400 || status === 422) && fileLimitHint.test(message)) return true;
  return fileLimitHint.test(message) && /cloud asr error/i.test(message);
}

export function isDeepgramChunkableError(error: unknown): boolean {
  const status = parseCloudAsrStatus(error);
  if (status === 413 || status === 504) return true;
  return isCloudAsrFileLimitError(error);
}

export function isGladiaChunkableError(error: unknown): boolean {
  const status = parseCloudAsrStatus(error);
  if (status === 408 || status === 413 || status === 504) return true;
  return isCloudAsrFileLimitError(error);
}
