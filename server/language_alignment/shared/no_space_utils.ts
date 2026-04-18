export function normalizeNoSpaceAlignmentText(value: string) {
  return String(value || '')
    .normalize('NFKC')
    .replace(/\u00a0/g, ' ')
    .replace(/\s+/g, '')
    .replace(/[，､]/g, '、')
    .replace(/[．｡]/g, '。')
    .replace(/[！]/g, '!')
    .replace(/[？]/g, '?')
    .replace(/[〜～]/g, 'ー')
    .trim();
}

type ScriptMode = 'hiragana' | 'katakana' | 'han' | 'hangul' | 'latin' | 'digit' | 'punct' | 'other';

function getScriptMode(char: string): ScriptMode {
  if (/[\u3040-\u309f\u30fc]/u.test(char)) return 'hiragana';
  if (/[\u30a0-\u30ff]/u.test(char)) return 'katakana';
  if (/[\u3400-\u9fff]/u.test(char)) return 'han';
  if (/[\uac00-\ud7af]/u.test(char)) return 'hangul';
  if (/[A-Za-z]/.test(char)) return 'latin';
  if (/[0-9]/.test(char)) return 'digit';
  if (/[\p{P}\p{S}]/u.test(char)) return 'punct';
  return 'other';
}

export function genericFallbackSegmentNoSpaceLexicalUnits(text: string) {
  const normalized = normalizeNoSpaceAlignmentText(text);
  if (!normalized) return [];

  const units: string[] = [];
  let buffer = '';
  let bufferMode: ScriptMode | '' = '';
  const flush = () => {
    if (buffer) units.push(buffer);
    buffer = '';
    bufferMode = '';
  };

  for (const char of Array.from(normalized)) {
    const mode = getScriptMode(char);
    if (mode === 'punct') {
      flush();
      units.push(char);
      continue;
    }
    if (!buffer) {
      buffer = char;
      bufferMode = mode;
      continue;
    }
    if (
      mode === bufferMode ||
      (bufferMode === 'hiragana' && mode === 'katakana') ||
      (bufferMode === 'katakana' && mode === 'hiragana')
    ) {
      buffer += char;
      continue;
    }
    flush();
    buffer = char;
    bufferMode = mode;
  }
  flush();
  return units;
}
