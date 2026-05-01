export interface NormalizedSubtitleCue {
  startSeconds: number | null;
  text: string;
}

function normalizeNewlines(value: string) {
  return String(value || '')
    .replace(/^\uFEFF/, '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n');
}

function collapseCueText(lines: string[]) {
  return lines
    .map((line) => String(line || '').trim())
    .filter(Boolean)
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function stripHtmlTags(value: string) {
  return String(value || '').replace(/<[^>]+>/g, '').trim();
}

function stripAssText(value: string) {
  return String(value || '')
    .replace(/\{[^}]*\}/g, '')
    .replace(/\\[Nn]/g, ' ')
    .replace(/\\h/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function secondsFromParts(hours: number, minutes: number, seconds: number, fraction = 0) {
  if (![hours, minutes, seconds, fraction].every(Number.isFinite)) return null;
  return hours * 3600 + minutes * 60 + seconds + fraction;
}

export function parseSubtitleTimeToSeconds(raw: string) {
  const value = String(raw || '').trim();
  const standard = value.match(/^(\d{1,2}):(\d{2}):(\d{2})(?:[,.](\d{1,3}))?$/);
  if (standard) {
    const ms = standard[4] ? Number(standard[4].padEnd(3, '0')) / 1000 : 0;
    return secondsFromParts(Number(standard[1]), Number(standard[2]), Number(standard[3]), ms);
  }

  const ass = value.match(/^(\d+):(\d{2}):(\d{2})(?:\.(\d{1,3}))?$/);
  if (ass) {
    const cs = ass[4] ? Number(ass[4].padEnd(3, '0')) / 1000 : 0;
    return secondsFromParts(Number(ass[1]), Number(ass[2]), Number(ass[3]), cs);
  }

  return null;
}

export function formatStandardSubtitleTime(seconds: number | null) {
  const safe = Math.max(0, Number.isFinite(seconds || 0) ? Number(seconds || 0) : 0);
  const totalSeconds = Math.floor(safe);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const secs = totalSeconds % 60;
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function buildStandardSubtitleText(cues: NormalizedSubtitleCue[]) {
  return cues
    .map((cue) => {
      const text = collapseCueText([cue.text]);
      if (!text) return '';
      if (cue.startSeconds == null) return text;
      return `[${formatStandardSubtitleTime(cue.startSeconds)}] ${text}`;
    })
    .filter(Boolean)
    .join('\n')
    .trim();
}

function parseSrtCues(content: string): NormalizedSubtitleCue[] {
  const blocks = normalizeNewlines(content)
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean);
  const cues: NormalizedSubtitleCue[] = [];

  for (const block of blocks) {
    const lines = block.split('\n').map((line) => line.trim()).filter(Boolean);
    const timeIndex = lines.findIndex((line) => /-->/u.test(line));
    if (timeIndex < 0) continue;
    const startRaw = lines[timeIndex].split(/-->/u)[0]?.trim() || '';
    const startSeconds = parseSubtitleTimeToSeconds(startRaw);
    const text = collapseCueText(lines.slice(timeIndex + 1).map(stripHtmlTags));
    if (text) cues.push({ startSeconds, text });
  }

  return cues;
}

function parseVttCues(content: string): NormalizedSubtitleCue[] {
  const lines = normalizeNewlines(content).split('\n');
  const cues: NormalizedSubtitleCue[] = [];
  let index = 0;

  while (index < lines.length) {
    let line = lines[index].trim();
    if (!line || /^WEBVTT\b/i.test(line)) {
      index += 1;
      continue;
    }
    if (/^(NOTE|STYLE|REGION)\b/i.test(line)) {
      index += 1;
      while (index < lines.length && lines[index].trim()) index += 1;
      continue;
    }
    if (!/-->/u.test(line) && index + 1 < lines.length && /-->/u.test(lines[index + 1])) {
      index += 1;
      line = lines[index].trim();
    }
    if (!/-->/u.test(line)) {
      index += 1;
      continue;
    }

    const startRaw = line.split(/-->/u)[0]?.trim() || '';
    const startSeconds = parseSubtitleTimeToSeconds(startRaw);
    index += 1;
    const textLines: string[] = [];
    while (index < lines.length && lines[index].trim()) {
      textLines.push(stripHtmlTags(lines[index]));
      index += 1;
    }
    const text = collapseCueText(textLines);
    if (text) cues.push({ startSeconds, text });
  }

  return cues;
}

function splitAssDialogueValues(value: string, fieldCount: number) {
  const parts = String(value || '').split(',');
  if (fieldCount <= 1 || parts.length <= fieldCount) return parts;
  return [...parts.slice(0, fieldCount - 1), parts.slice(fieldCount - 1).join(',')];
}

function parseAssCues(content: string): NormalizedSubtitleCue[] {
  const lines = normalizeNewlines(content).split('\n');
  const cues: NormalizedSubtitleCue[] = [];
  let inEvents = false;
  let fields: string[] = [];

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;
    if (/^\[Events\]/i.test(line)) {
      inEvents = true;
      continue;
    }
    if (/^\[.+\]/.test(line)) {
      inEvents = false;
      continue;
    }
    if (!inEvents) continue;

    const formatMatch = line.match(/^Format:\s*(.+)$/i);
    if (formatMatch) {
      fields = formatMatch[1].split(',').map((field) => field.trim().toLowerCase());
      continue;
    }

    const dialogueMatch = line.match(/^Dialogue:\s*(.+)$/i);
    if (!dialogueMatch) continue;

    const effectiveFields = fields.length > 0
      ? fields
      : ['layer', 'start', 'end', 'style', 'name', 'marginl', 'marginr', 'marginv', 'effect', 'text'];
    const values = splitAssDialogueValues(dialogueMatch[1], effectiveFields.length);
    const startIndex = effectiveFields.indexOf('start');
    const textIndex = effectiveFields.indexOf('text');
    const startSeconds = startIndex >= 0 ? parseSubtitleTimeToSeconds(values[startIndex] || '') : null;
    const text = textIndex >= 0 ? stripAssText(values.slice(textIndex).join(',')) : '';
    if (text) cues.push({ startSeconds, text });
  }

  return cues;
}

function normalizeJsonSubtitleSource(content: string): string | null {
  try {
    const parsed = JSON.parse(content);
    if (typeof parsed === 'string') return parsed.trim();
    if (typeof parsed?.originalSubtitles === 'string') return parsed.originalSubtitles.trim();
    if (typeof parsed?.translatedSubtitles === 'string') return parsed.translatedSubtitles.trim();
    if (typeof parsed?.text === 'string') return parsed.text.trim();

    const segments = Array.isArray(parsed?.segments) ? parsed.segments : Array.isArray(parsed?.chunks) ? parsed.chunks : null;
    if (segments) {
      const cues = segments
        .map((segment: any) => {
          const text = String(segment?.text || '').trim();
          if (!text) return null;
          const start = Number(segment?.start ?? segment?.start_ts ?? segment?.timestamp?.[0]);
          return {
            startSeconds: Number.isFinite(start) ? start : null,
            text,
          } as NormalizedSubtitleCue;
        })
        .filter((cue: NormalizedSubtitleCue | null): cue is NormalizedSubtitleCue => Boolean(cue));
      if (cues.length > 0) return buildStandardSubtitleText(cues);
    }
  } catch {
    return null;
  }

  return null;
}

export function normalizeSubtitleSourceForTranslation(fileName: string, rawContent: string) {
  const clean = normalizeNewlines(rawContent).trim();
  if (!clean) return '';

  const lowerName = String(fileName || '').toLowerCase();
  const ext = lowerName.includes('.') ? lowerName.slice(lowerName.lastIndexOf('.')) : '';
  if (ext === '.json') {
    const normalized = normalizeJsonSubtitleSource(clean);
    if (normalized) return normalizeSubtitleSourceForTranslation('source.txt', normalized);
    return clean;
  }

  let cues: NormalizedSubtitleCue[] = [];
  if (ext === '.srt') cues = parseSrtCues(clean);
  if (ext === '.vtt') cues = parseVttCues(clean);
  if (ext === '.ass' || ext === '.ssa') cues = parseAssCues(clean);
  if (cues.length === 0 && /^\s*\[Events\]/im.test(clean) && /^Dialogue:/im.test(clean)) {
    cues = parseAssCues(clean);
  }
  if (cues.length === 0 && /^WEBVTT\b/im.test(clean)) {
    cues = parseVttCues(clean);
  }
  if (cues.length === 0 && /-->/u.test(clean)) {
    cues = parseSrtCues(clean);
  }

  const normalized = cues.length > 0 ? buildStandardSubtitleText(cues) : '';
  return normalized || clean;
}

export function stripLineSafeMarkers(text: string) {
  return String(text || '').replace(/\[\[L\d{5}\]\]\s*/g, '').trim();
}

export function extractLeadingStandardTimeTag(line: string) {
  const matched = String(line || '').match(/^(\[\d{2}:\d{2}:\d{2}(?:[.,]\d{1,3})?\])\s*(.*)$/);
  if (!matched) return null;
  return {
    tag: matched[1],
    text: String(matched[2] || '').trim(),
  };
}

export function stripLeadingStandardTimeTag(line: string) {
  const matched = extractLeadingStandardTimeTag(line);
  if (!matched) return String(line || '').trim();
  return matched.text;
}

export function cleanTranslatedSubtitleLinePayload(line: string) {
  const withoutMarker = stripLineSafeMarkers(line);
  return stripLineSafeMarkers(stripLeadingStandardTimeTag(withoutMarker));
}

export function rebuildTranslationWithSourceTimecodes(sourceText: string, translatedText: string) {
  const sourceLines = normalizeNewlines(sourceText).trim().split('\n').filter(Boolean);
  const translatedLines = normalizeNewlines(translatedText).trim().split('\n');
  if (sourceLines.length === 0) return String(translatedText || '').trim();

  const allSourceTimed = sourceLines.every((line) => Boolean(extractLeadingStandardTimeTag(line.trim())));
  if (!allSourceTimed) return String(translatedText || '').trim();

  const rebuilt = sourceLines.map((sourceLine, index) => {
    const parsed = extractLeadingStandardTimeTag(sourceLine.trim());
    if (!parsed) return cleanTranslatedSubtitleLinePayload(translatedLines[index] || '');
    const translatedPayload = cleanTranslatedSubtitleLinePayload(translatedLines[index] || '');
    if (!translatedPayload) return `${parsed.tag} ${parsed.text}`.trim();
    return `${parsed.tag} ${translatedPayload}`.trim();
  });

  return rebuilt.join('\n').trim();
}
