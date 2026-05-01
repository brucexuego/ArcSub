import fs from 'fs-extra';
import { PathManager } from '../path_manager.js';
import {
  normalizeSubtitleSourceForTranslation,
  rebuildTranslationWithSourceTimecodes,
} from '../../src/utils/subtitle_normalizer.js';

export { rebuildTranslationWithSourceTimecodes };

function toSingleLineText(value: string) {
  return value.replace(/\s+/g, ' ').trim();
}

function limitText(value: string, max = 800) {
  if (value.length <= max) return value;
  return `${value.slice(0, max)}...`;
}

function pushUniqueErrorPart(parts: string[], value: string) {
  const normalized = value.trim();
  if (normalized && !parts.includes(normalized)) {
    parts.push(normalized);
  }
}

function stringifyJsonErrorValue(value: unknown, seen = new WeakSet<object>()): string {
  if (value == null) return '';
  if (typeof value === 'string') return value.trim();
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (Array.isArray(value)) {
    const parts: string[] = [];
    value.forEach((item) => pushUniqueErrorPart(parts, stringifyJsonErrorValue(item, seen)));
    return parts.join(' | ');
  }
  if (typeof value !== 'object') return '';

  if (seen.has(value)) return '';
  seen.add(value);

  const record = value as Record<string, unknown>;
  const parts: string[] = [];
  [
    'message',
    'msg',
    'err_msg',
    'error_description',
    'description',
    'reason',
    'detail',
    'error',
  ].forEach((key) => pushUniqueErrorPart(parts, stringifyJsonErrorValue(record[key], seen)));
  ['status', 'code', 'type', 'error_code', 'err_code'].forEach((key) => {
    const text = stringifyJsonErrorValue(record[key], seen);
    if (text) pushUniqueErrorPart(parts, `${key}: ${text}`);
  });

  if (parts.length > 0) return parts.join(' | ');
  try {
    return JSON.stringify(value);
  } catch {
    return '';
  }
}

export function extractErrorMessage(bodyText: string, contentType: string) {
  const cleanBody = toSingleLineText(bodyText || '');
  if (!cleanBody) return '';

  if (contentType.includes('application/json')) {
    try {
      const parsed = JSON.parse(bodyText);
      const message = stringifyJsonErrorValue(parsed);
      if (message) {
        return limitText(toSingleLineText(message));
      }
    } catch {
      // Fall back to text body
    }
  }

  return limitText(cleanBody);
}

function normalizeTranslationSourceText(fileName: string, rawContent: string) {
  return normalizeSubtitleSourceForTranslation(fileName, rawContent);
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

function stripLineSafeMarkers(text: string) {
  return String(text || '').replace(/\[\[L\d{5}\]\]\s*/g, '').trim();
}

function buildTranslationArtifacts(sourceText: string) {
  const sourceLines = String(sourceText || '').split('\n').map((line) => stripLineSafeMarkers(line));
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
