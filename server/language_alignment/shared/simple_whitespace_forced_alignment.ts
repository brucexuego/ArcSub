import { loadJsonFromModuleDir, normalizeLanguageCode, tryLoadJsonFromModuleDir } from './json_loader.js';
import type {
  AlignmentRun,
  AlignmentSegmentLike,
  AlignmentWordLike,
  AlignmentWordToken,
  ForcedAlignmentLanguageModule,
  ForcedAlignmentPlanConfig,
  ForcedAlignmentSelectionContext,
} from './types.js';

export type SimpleWhitespaceForcedAlignmentConfig = {
  key: string;
  aliases: string[];
  alignmentProfileId: string;
  clipPaddingSec: number;
  minRunConfidence: number;
  minOverallConfidence: number;
  minOverallAlignedRatio: number;
  progressLabel: string;
  maxRunGapSec?: number;
  minRunTokenLength?: number;
  scriptPattern?: string;
};

type AutoTextClues = {
  stopwords?: string[];
  priorityWords?: string[];
  rawMarkers?: string[];
  minScore?: number;
};

type SimpleWhitespaceForcedAlignmentOptions<
  TWord extends AlignmentWordLike = AlignmentWordLike,
  TSegment extends AlignmentSegmentLike<TWord> = AlignmentSegmentLike<TWord>,
> = {
  normalizeToken?: (value: string) => string;
  supportsToken?: (value: string) => boolean;
  matchesText?: (text?: string) => boolean;
  projectWords?: (segment: TSegment) => Promise<TWord[] | null> | TWord[] | null;
};

const apostropheVariantsRegex = /[’‘`´]/gu;
const defaultWordRegex = /[\p{L}\p{M}\p{N}]+(?:['’][\p{L}\p{M}\p{N}]+)*/gu;
const latinSubstitutions: Record<string, string> = {
  '\u00df': 'ss',
  '\u00e6': 'ae',
  '\u0153': 'oe',
  '\u00f8': 'o',
  '\u0111': 'd',
  '\u0142': 'l',
  '\u00fe': 'th',
};

function defaultNormalizeToken(value: string) {
  return String(value || '')
    .normalize('NFKC')
    .replace(/\u00a0/g, ' ')
    .replace(apostropheVariantsRegex, "'")
    .replace(/[\u00df\u00e6\u0153\u00f8\u0111\u0142\u00fe]/gu, (char) => latinSubstitutions[char] || char)
    .normalize('NFKD')
    .replace(/\p{M}+/gu, '')
    .toLowerCase()
    .replace(/[^'\p{L}\p{M}\p{N}]+/gu, '')
    .replace(/^'+|'+$/g, '')
    .trim();
}

function defaultSupportsToken(value: string) {
  return /[\p{L}\p{N}]/u.test(String(value || ''));
}

function matchesAlias(normalizedLanguage: string, aliasSet: Set<string>) {
  if (!normalizedLanguage) return false;
  if (aliasSet.has(normalizedLanguage)) return true;
  return [...aliasSet].some((alias) => normalizedLanguage.startsWith(`${alias}-`));
}

function getWordEnd(word: AlignmentWordLike) {
  const end = Number(word?.end_ts);
  if (Number.isFinite(end)) return end;
  return Number(word?.start_ts) || 0;
}

function cloneWords<TWord extends AlignmentWordLike>(words: TWord[]) {
  return words.map((word) => ({ ...word }));
}

function projectWordsFromSegment<TWord extends AlignmentWordLike, TSegment extends AlignmentSegmentLike<TWord>>(segment: TSegment) {
  if (Array.isArray(segment?.words) && segment.words.length > 0) {
    const cloned = cloneWords(segment.words).filter((word) => String(word?.text || '').trim());
    return cloned.length > 0 ? cloned : null;
  }

  const text = String(segment?.text || '');
  const tokens = Array.from(text.matchAll(defaultWordRegex)).map((match) => match[0]).filter(Boolean);
  if (tokens.length === 0) {
    return null;
  }

  const start = Number(segment?.start_ts);
  const end = Number(segment?.end_ts);
  if (!(Number.isFinite(start) && Number.isFinite(end) && end > start)) {
    return tokens.map((token) => ({ text: token, start_ts: 0, end_ts: 0 })) as TWord[];
  }

  const totalChars = tokens.reduce((sum, token) => sum + token.length, 0) || tokens.length;
  let cursor = 0;
  return tokens.map((token) => {
    const tokenWeight = token.length || 1;
    const tokenStart = start + ((end - start) * cursor) / totalChars;
    cursor += tokenWeight;
    const tokenEnd = start + ((end - start) * cursor) / totalChars;
    return {
      text: token,
      start_ts: Number(tokenStart.toFixed(3)),
      end_ts: Number(tokenEnd.toFixed(3)),
    } as TWord;
  });
}

function tokenizeNormalizedWords(text: string, normalizeToken: (value: string) => string) {
  return Array.from(String(text || '').matchAll(defaultWordRegex))
    .map((match) => normalizeToken(match[0]))
    .filter(Boolean);
}

function scoreAutoText(
  text: string,
  normalizeToken: (value: string) => string,
  clues: AutoTextClues | null
) {
  if (!clues) return 0;

  const tokens = tokenizeNormalizedWords(text, normalizeToken);
  if (tokens.length === 0) return 0;

  const stopwordSet = new Set((clues.stopwords || []).map((token) => normalizeToken(token)).filter(Boolean));
  const prioritySet = new Set((clues.priorityWords || []).map((token) => normalizeToken(token)).filter(Boolean));
  const rawMarkers = (clues.rawMarkers || []).map((pattern) => {
    try {
      return new RegExp(pattern, 'iu');
    } catch {
      return null;
    }
  }).filter(Boolean) as RegExp[];

  let score = 0;
  const uniqueTokens = new Set(tokens);

  for (const token of tokens) {
    if (stopwordSet.has(token)) score += 1;
    if (prioritySet.has(token)) score += 2;
  }

  for (const marker of rawMarkers) {
    if (marker.test(text)) {
      score += 2;
    }
  }

  if (uniqueTokens.size <= 2 && score < (clues.minScore || 3)) {
    return 0;
  }

  return Number(score.toFixed(3));
}

export function createSimpleWhitespaceForcedAlignmentModule<
  TWord extends AlignmentWordLike = AlignmentWordLike,
  TSegment extends AlignmentSegmentLike<TWord> = AlignmentSegmentLike<TWord>,
>(
  moduleUrl: string,
  options: SimpleWhitespaceForcedAlignmentOptions<TWord, TSegment> = {}
): ForcedAlignmentLanguageModule<TWord, TSegment> {
  const config = loadJsonFromModuleDir<SimpleWhitespaceForcedAlignmentConfig>(moduleUrl, 'config.json');
  const aliasSet = new Set(config.aliases.map((alias) => normalizeLanguageCode(alias)));
  const autoClues = tryLoadJsonFromModuleDir<AutoTextClues>(moduleUrl, 'auto_clues.json');
  const scriptRegex = config.scriptPattern ? new RegExp(config.scriptPattern, 'u') : null;
  const normalizeToken = options.normalizeToken || defaultNormalizeToken;
  const supportsToken = options.supportsToken || defaultSupportsToken;
  const projectWords = options.projectWords || projectWordsFromSegment;
  const maxRunGapSec = Number.isFinite(Number(config.maxRunGapSec)) ? Number(config.maxRunGapSec) : 0.42;
  const minRunTokenLength = Math.max(1, Number(config.minRunTokenLength) || 2);

  return {
    key: config.key,
    aliases: config.aliases,
    matchesLanguage(language?: string) {
      return matchesAlias(normalizeLanguageCode(language), aliasSet);
    },
    matchesText(text?: string) {
      if (options.matchesText) {
        return options.matchesText(text);
      }
      return this.scoreText(text) > 0 || (scriptRegex ? scriptRegex.test(String(text || '')) : false);
    },
    scoreText(text?: string) {
      const source = String(text || '');
      const clueScore = scoreAutoText(source, normalizeToken, autoClues);
      if (clueScore > 0) return clueScore;
      return scriptRegex && scriptRegex.test(source) ? 1 : 0;
    },
    resolveVariant() {
      return null;
    },
    getPlanConfig(
      _language?: string,
      _sampleText?: string,
      _context?: ForcedAlignmentSelectionContext
    ): ForcedAlignmentPlanConfig {
      return {
        profileId: config.alignmentProfileId,
        clipPaddingSec: config.clipPaddingSec,
        minRunConfidence: config.minRunConfidence,
        minOverallConfidence: config.minOverallConfidence,
        minOverallAlignedRatio: config.minOverallAlignedRatio,
        progressLabel: config.progressLabel,
      };
    },
    async projectWordReadings(segment: TSegment) {
      const words = await projectWords(segment);
      return Array.isArray(words) && words.length > 0 ? words : null;
    },
    extractAlignableRuns(words: TWord[]) {
      const annotated = words
        .map((word, index) => ({
          index,
          word,
          normalized: normalizeToken(word?.text || ''),
        }))
        .filter((item) => item.normalized.length > 0 && supportsToken(item.normalized));

      const runs: AlignmentRun[] = [];
      let current: typeof annotated = [];

      const flush = () => {
        if (current.length > 0) {
          runs.push({
            startIndex: current[0].index,
            endIndex: current[current.length - 1].index,
            tokens: current.map(
              (item): AlignmentWordToken => ({
                text: item.word.text,
                normalized: item.normalized,
              })
            ),
          });
        }
        current = [];
      };

      for (const item of annotated) {
        if (current.length > 0) {
          const previous = current[current.length - 1];
          const gap = Number(item.word?.start_ts || 0) - getWordEnd(previous.word);
          if (gap > maxRunGapSec) {
            flush();
          }
        }
        current.push(item);
      }
      flush();

      return runs.filter((run) => run.tokens.some((token) => token.normalized.length >= minRunTokenLength));
    },
    normalizeAlignmentToken: normalizeToken,
    supportsAlignmentToken: supportsToken,
  };
}
