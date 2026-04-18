import { convert as romanizeHangul } from 'hangul-romanization';
import { createSimpleNoSpaceLanguagePack } from '../shared/simple_no_space_language.js';
import { loadJsonFromModuleDir, normalizeLanguageCode } from '../shared/json_loader.js';
import type {
  AlignmentRun,
  AlignmentSegmentLike,
  AlignmentWordLike,
  AlignmentWordToken,
  ForcedAlignmentLanguageModule,
  ForcedAlignmentPlanConfig,
  ForcedAlignmentSelectionContext,
  LanguageAlignmentPack,
  LocalAsrPromptModule,
  NoSpaceLanguageConfig,
} from '../shared/types.js';

type KoreanConfig = NoSpaceLanguageConfig & {
  alignmentProfileId: string;
  clipPaddingSec: number;
  minRunConfidence: number;
  minOverallConfidence: number;
  minOverallAlignedRatio: number;
  progressLabel: string;
  maxRunGapSec: number;
  minRunTokenLength: number;
  localAsrPrompt?: string;
};

const config = loadJsonFromModuleDir<KoreanConfig>(import.meta.url, 'config.json');
const aliasSet = new Set(config.aliases.map((alias) => normalizeLanguageCode(alias)));

function matchesLanguage(language?: string) {
  const normalized = normalizeLanguageCode(language);
  if (!normalized) return false;
  if (aliasSet.has(normalized)) return true;
  return [...aliasSet].some((alias) => normalized.startsWith(`${alias}-`));
}

function matchesKoreanText(text?: string) {
  return /[\uac00-\ud7af]/u.test(String(text || ''));
}

function scoreKoreanText(text?: string) {
  const hangulCount = String(text || '').match(/[\uac00-\ud7af]/gu)?.length || 0;
  return hangulCount * 3;
}

function normalizeKoreanAlignmentToken(value: string) {
  return String(value || '')
    .normalize('NFKC')
    .replace(/\u00a0/g, ' ')
    .replace(/\s+/g, '')
    .replace(/[’‘`´]/gu, "'")
    .replace(/[^A-Za-z0-9']+/gu, '')
    .toLowerCase()
    .replace(/^'+|'+$/g, '')
    .trim();
}

function romanizeKoreanToken(value: string) {
  const source = String(value || '').normalize('NFKC').trim();
  if (!source) return '';
  const romanized = romanizeHangul(source);
  return normalizeKoreanAlignmentToken(romanized);
}

function getWordEnd(word: AlignmentWordLike) {
  const end = Number(word?.end_ts);
  if (Number.isFinite(end)) return end;
  return Number(word?.start_ts) || 0;
}

function projectKoreanWords(segment: AlignmentSegmentLike<AlignmentWordLike>) {
  if (Array.isArray(segment?.words) && segment.words.length > 0) {
    const cloned = segment.words
      .map((word) => ({
        ...word,
        normalizedReading: romanizeKoreanToken(word.text || ''),
      }))
      .filter((word) => String(word.normalizedReading || '').length > 0);
    return cloned.length > 0 ? cloned : null;
  }

  const tokens = String(segment?.text || '')
    .split(/\s+/)
    .map((token) => token.trim())
    .filter(Boolean);
  if (tokens.length === 0) {
    return null;
  }

  const start = Number(segment?.start_ts);
  const end = Number(segment?.end_ts);
  if (!(Number.isFinite(start) && Number.isFinite(end) && end > start)) {
    return tokens.map((token) => ({
      text: token,
      normalizedReading: romanizeKoreanToken(token),
      start_ts: 0,
      end_ts: 0,
    }));
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
      normalizedReading: romanizeKoreanToken(token),
      start_ts: Number(tokenStart.toFixed(3)),
      end_ts: Number(tokenEnd.toFixed(3)),
    };
  });
}

const basePack = createSimpleNoSpaceLanguagePack(import.meta.url);

const forcedAlignment: ForcedAlignmentLanguageModule = {
  key: config.key,
  aliases: config.aliases,
  matchesLanguage,
  matchesText(text?: string) {
    return matchesKoreanText(text);
  },
  scoreText(text?: string) {
    return scoreKoreanText(text);
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
  async projectWordReadings(segment: AlignmentSegmentLike<AlignmentWordLike>) {
    const words = projectKoreanWords(segment);
    return Array.isArray(words) && words.length > 0 ? words : null;
  },
  extractAlignableRuns(words: AlignmentWordLike[]) {
    const annotated = words
      .map((word, index) => ({
        index,
        word,
        normalized: normalizeKoreanAlignmentToken(word?.normalizedReading || word?.text || ''),
      }))
      .filter((item) => item.normalized.length > 0);

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
        if (gap > config.maxRunGapSec) {
          flush();
        }
      }
      current.push(item);
    }
    flush();

    return runs.filter((run) => run.tokens.some((token) => token.normalized.length >= config.minRunTokenLength));
  },
  normalizeAlignmentToken: normalizeKoreanAlignmentToken,
  supportsAlignmentToken(value: string) {
    return /[A-Za-z0-9]/.test(String(value || ''));
  },
};

const localAsrPrompt: LocalAsrPromptModule = {
  key: config.key,
  aliases: config.aliases,
  matchesLanguage,
  matchesText(text?: string) {
    return matchesKoreanText(text);
  },
  scoreText(text?: string) {
    return scoreKoreanText(text);
  },
  resolveVariant() {
    return null;
  },
  getPrompt() {
    return config.localAsrPrompt || null;
  },
  getDebug() {
    return {
      hasPrompt: Boolean(config.localAsrPrompt),
    };
  },
};

export const languagePack: LanguageAlignmentPack = {
  key: basePack.key,
  aliases: basePack.aliases,
  noSpace: basePack.noSpace,
  forcedAlignment,
  localAsrPrompt,
};

export const koreanLanguagePack = languagePack;
