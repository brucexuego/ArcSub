export interface EditableSubtitleRow {
  id: string;
  timecode: string;
  text: string;
}

export type SubtitleEditorIssueCode =
  | 'timecode_format'
  | 'timecode_before_previous'
  | 'timecode_after_next'
  | 'text_required';

export interface SubtitleEditorIssue {
  index: number;
  field: 'timecode' | 'text';
  code: SubtitleEditorIssueCode;
}

const TIME_TAG_PATTERN = /^\[(\d{2}:\d{2}:\d{2}(?:[.,]\d{1,3})?)\]\s*([\s\S]*)$/;
const TIMECODE_PATTERN = /^(\d{2}):(\d{2}):(\d{2})(?:[.,](\d{1,3}))?$/;

let subtitleRowSeed = 0;

function normalizeText(input: string) {
  return String(input || '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n');
}

export function makeSubtitleRow(timecode = '', text = ''): EditableSubtitleRow {
  subtitleRowSeed += 1;
  return {
    id: `subtitle-row-${Date.now()}-${subtitleRowSeed}`,
    timecode: String(timecode || '').trim(),
    text: String(text || ''),
  };
}

export function parseSubtitleLine(line: string) {
  const source = String(line || '');
  const matched = source.match(TIME_TAG_PATTERN);
  if (!matched) {
    return {
      timecode: '',
      text: source.trim(),
    };
  }
  return {
    timecode: matched[1],
    text: String(matched[2] || '').trim(),
  };
}

export function subtitleRowsFromText(text: string) {
  return normalizeText(text)
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const parsed = parseSubtitleLine(line);
      return makeSubtitleRow(parsed.timecode, parsed.text);
    });
}

export function buildSubtitleLine(row: Pick<EditableSubtitleRow, 'timecode' | 'text'>) {
  const text = String(row.text || '').trim();
  const timecode = String(row.timecode || '').trim();
  if (!text) return '';
  if (!timecode) return text;
  return `[${timecode}] ${text}`;
}

export function subtitleRowsToLines(rows: Array<Pick<EditableSubtitleRow, 'timecode' | 'text'>>) {
  return rows
    .map((row) => buildSubtitleLine(row))
    .map((line) => line.trim())
    .filter(Boolean);
}

export function parseTimecodeToSeconds(timecode: string): number | null {
  const matched = String(timecode || '').trim().match(TIMECODE_PATTERN);
  if (!matched) return null;
  const hh = Number(matched[1]);
  const mm = Number(matched[2]);
  const ss = Number(matched[3]);
  const ms = matched[4] ? Number(matched[4].padEnd(3, '0')) : 0;
  if (![hh, mm, ss, ms].every(Number.isFinite)) return null;
  return hh * 3600 + mm * 60 + ss + ms / 1000;
}

export function hasStrictBracketTimecodes(text: string) {
  const lines = normalizeText(text).split('\n').map((line) => line.trim()).filter(Boolean);
  if (lines.length === 0) return false;
  return lines.every((line) => Boolean(TIME_TAG_PATTERN.test(line)));
}

export function extractLeadingTimeTag(line: string) {
  const matched = String(line || '').match(/^(\[\d{2}:\d{2}:\d{2}(?:[.,]\d{1,3})?\])\s*(.*)$/);
  if (!matched) return null;
  return {
    tag: matched[1],
    text: String(matched[2] || '').trim(),
  };
}

export function stripLeadingTimeTag(line: string) {
  const matched = extractLeadingTimeTag(line);
  if (!matched) return String(line || '').trim();
  return matched.text;
}

export function parseTimedSubtitleLine(line: string) {
  const matched = extractLeadingTimeTag(line);
  if (!matched) {
    return {
      timecode: null as string | null,
      startSeconds: null as number | null,
      text: String(line || '').trim(),
    };
  }

  const rawTimecode = matched.tag.replace(/^\[/, '').replace(/\]$/, '');
  return {
    timecode: rawTimecode,
    startSeconds: parseTimecodeToSeconds(rawTimecode),
    text: matched.text,
  };
}

export function validateSubtitleRows(rows: EditableSubtitleRow[]): SubtitleEditorIssue[] {
  const issues: SubtitleEditorIssue[] = [];
  if (!Array.isArray(rows) || rows.length === 0) return issues;

  const secondsByIndex: Array<number | null> = rows.map((row, index) => {
    const text = String(row.text || '').trim();
    const timecode = String(row.timecode || '').trim();

    if (!text) {
      issues.push({
        index,
        field: 'text',
        code: 'text_required',
      });
    }

    if (!timecode) return null;

    const seconds = parseTimecodeToSeconds(timecode);
    if (seconds == null) {
      issues.push({
        index,
        field: 'timecode',
        code: 'timecode_format',
      });
      return null;
    }
    return seconds;
  });

  for (let index = 0; index < rows.length; index += 1) {
    const current = secondsByIndex[index];
    if (current == null) continue;

    let prev: number | null = null;
    for (let i = index - 1; i >= 0; i -= 1) {
      if (secondsByIndex[i] != null) {
        prev = secondsByIndex[i];
        break;
      }
    }

    let next: number | null = null;
    for (let i = index + 1; i < rows.length; i += 1) {
      if (secondsByIndex[i] != null) {
        next = secondsByIndex[i];
        break;
      }
    }

    if (prev != null && current < prev) {
      issues.push({
        index,
        field: 'timecode',
        code: 'timecode_before_previous',
      });
    }
    if (next != null && current > next) {
      issues.push({
        index,
        field: 'timecode',
        code: 'timecode_after_next',
      });
    }
  }

  return issues;
}
