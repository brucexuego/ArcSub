import fs from 'fs-extra';
import path from 'path';
import { PathManager } from '../path_manager.js';

function toSingleLineText(value: string) {
  return value.replace(/\s+/g, ' ').trim();
}

function limitText(value: string, max = 180) {
  if (value.length <= max) return value;
  return `${value.slice(0, max)}...`;
}

export function extractErrorMessage(bodyText: string, contentType: string) {
  const cleanBody = toSingleLineText(bodyText || '');
  if (!cleanBody) return '';

  if (contentType.includes('application/json')) {
    try {
      const parsed = JSON.parse(bodyText);
      const nestedError = parsed?.error;
      const nestedMessage =
        typeof nestedError === 'string'
          ? nestedError
          : nestedError?.message || parsed?.message || parsed?.detail;
      if (typeof nestedMessage === 'string' && nestedMessage.trim()) {
        return limitText(toSingleLineText(nestedMessage));
      }
    } catch {
      // Fall back to text body
    }
  }

  return limitText(cleanBody);
}

function normalizeTranslationSourceText(fileName: string, rawContent: string) {
  const clean = String(rawContent || '').replace(/^\uFEFF/, '').trim();
  if (!clean) return '';

  const ext = path.extname(fileName).toLowerCase();
  if (ext !== '.json') return clean;

  try {
    const parsed = JSON.parse(clean);
    if (typeof parsed === 'string') return parsed.trim();
    if (typeof parsed?.originalSubtitles === 'string') return parsed.originalSubtitles.trim();
    if (typeof parsed?.translatedSubtitles === 'string') return parsed.translatedSubtitles.trim();
    if (typeof parsed?.text === 'string') return parsed.text.trim();

    if (Array.isArray(parsed?.segments)) {
      const merged = parsed.segments
        .map((seg: any) => String(seg?.text || '').trim())
        .filter(Boolean)
        .join('\n');
      if (merged) return merged;
    }

    if (Array.isArray(parsed?.chunks)) {
      const merged = parsed.chunks
        .map((chunk: any) => String(chunk?.text || '').trim())
        .filter(Boolean)
        .join('\n');
      if (merged) return merged;
    }
  } catch {
    // Fallback to raw text.
  }

  return clean;
}

function parseBracketTimecodeSeconds(raw: string) {
  const matched = String(raw || '').match(/^(\d{2}):(\d{2}):(\d{2})$/);
  if (!matched) return null;
  const hh = Number(matched[1]);
  const mm = Number(matched[2]);
  const ss = Number(matched[3]);
  if (!Number.isFinite(hh) || !Number.isFinite(mm) || !Number.isFinite(ss)) return null;
  return hh * 3600 + mm * 60 + ss;
}

function formatSubtitleTime(seconds: number, type: 'srt' | 'vtt') {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const h = Math.floor(safe / 3600);
  const m = Math.floor((safe % 3600) / 60);
  const s = Math.floor(safe % 60);
  const ms = Math.floor((safe % 1) * 1000);
  const msSep = type === 'srt' ? ',' : '.';
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}${msSep}${ms.toString().padStart(3, '0')}`;
}

function buildTranslationArtifacts(sourceText: string) {
  const sourceLines = String(sourceText || '').split('\n');
  const timedRows: Array<{ start: number; text: string }> = [];
  const txtLines = sourceLines.map((line) => {
    const matched = String(line).match(/^\[(\d{2}:\d{2}:\d{2})\]\s*(.*)$/);
    if (!matched) return line;
    const seconds = parseBracketTimecodeSeconds(matched[1]);
    if (seconds === null) return line;
    timedRows.push({ start: seconds, text: String(matched[2] || '').trim() });
    return String(matched[2] || '').trim();
  });

  const txt = txtLines.join('\n').trim();
  const hasTimecodes = timedRows.length > 0 && timedRows.length === sourceLines.length;
  if (!hasTimecodes) {
    return { hasTimecodes: false as const, txt, srt: '', vtt: '' };
  }

  let srt = '';
  let vtt = 'WEBVTT\n\n';
  timedRows.forEach((row, idx) => {
    const nextStart = idx + 1 < timedRows.length ? timedRows[idx + 1].start : row.start + 2;
    const end = nextStart > row.start ? nextStart : row.start + 2;
    const payload = row.text || '';
    srt += `${idx + 1}\n${formatSubtitleTime(row.start, 'srt')} --> ${formatSubtitleTime(end, 'srt')}\n${payload}\n\n`;
    vtt += `${formatSubtitleTime(row.start, 'vtt')} --> ${formatSubtitleTime(end, 'vtt')}\n${payload}\n\n`;
  });

  return { hasTimecodes: true as const, txt, srt, vtt };
}

export async function persistTranslationArtifacts(projectId: string, translatedText: string) {
  const artifacts = buildTranslationArtifacts(translatedText);
  const txtPath = PathManager.resolveProjectFile(projectId, 'subtitles', 'translation.txt', { createProject: false });
  const srtPath = PathManager.resolveProjectFile(projectId, 'subtitles', 'translation.srt', { createProject: false });
  const vttPath = PathManager.resolveProjectFile(projectId, 'subtitles', 'translation.vtt', { createProject: false });

  await fs.outputFile(txtPath, artifacts.txt || '', 'utf8');
  if (artifacts.hasTimecodes) {
    await fs.outputFile(srtPath, artifacts.srt, 'utf8');
    await fs.outputFile(vttPath, artifacts.vtt, 'utf8');
  } else {
    if (await fs.pathExists(srtPath)) await fs.remove(srtPath);
    if (await fs.pathExists(vttPath)) await fs.remove(vttPath);
  }

  return artifacts;
}

export async function readProjectSourceAsset(projectId: string, subDir: 'assets' | 'subtitles', assetName: string) {
  const filePath = PathManager.resolveProjectFile(projectId, subDir, assetName, { createProject: false });
  if (!(await fs.pathExists(filePath))) return null;
  const content = await fs.readFile(filePath, 'utf8');
  return normalizeTranslationSourceText(assetName, content);
}

export function hasAuthFailureHint(message: string) {
  if (!message) return false;
  const lower = message.toLowerCase();
  const hints = [
    'invalid_api_key',
    'invalid api key',
    'incorrect api key',
    'authentication',
    'unauthorized',
    'forbidden',
    'bearer',
    'token',
    'permission denied',
    'access denied',
    'bad credentials',
  ];
  return hints.some((hint) => lower.includes(hint));
}

export function hasPayloadValidationHint(message: string) {
  if (!message) return false;
  const lower = message.toLowerCase();
  const hints = [
    'missing',
    'required',
    'validation',
    'invalid request',
    'request body',
    'content-type',
    'unsupported media type',
    'no file',
    'file is required',
    'model is required',
    'multipart',
    'form-data',
    'invalid multipart',
  ];
  return hints.some((hint) => lower.includes(hint));
}

export function toClientPath(absPath: string) {
  return PathManager.toClientPath(absPath);
}
