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
  ForcedAlignmentRunContext,
} from '../shared/types.js';

type KoreanConfig = NoSpaceLanguageConfig & {
  alignmentProfileId: string;
  clipPaddingSec: number;
  minRunConfidence: number;
  minOverallConfidence: number;
  minOverallAlignedRatio: number;
  alignmentMinAppliedRatio?: number;
  alignmentMinAvgConfidence?: number;
  progressLabel: string;
  maxRunGapSec: number;
  minRunTokenLength: number;
  cjkMergeSoftGapSec?: number;
  cjkMergeHardGapSec?: number;
  cjkMergeMaxPhraseChars?: number;
  cjkMergeMaxPhraseDurationSec?: number;
  cjkSplitMaxDurationSec?: number;
  cjkSplitStrongPauseSec?: number;
  cjkSplitClausePauseSec?: number;
  cjkSplitMaxChars?: number;
  cjkSplitMinCharsForBreak?: number;
  cjkSparseMinTextLength?: number;
  cjkSparseSingleWordMinTextLength?: number;
  cjkSparseLowCoverageRatio?: number;
  cjkSparseMediumCoverageRatio?: number;
  cjkSparseMediumCoverageMaxWords?: number;
  localAsrPrompt?: string;
  fallbackAlignmentProfileId?: string;
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

function normalizeKoreanHangulAlignmentToken(value: string) {
  return String(value || '')
    .normalize('NFKC')
    .replace(/\u00a0/g, ' ')
    .replace(/\s+/g, '')
    .replace(/[^\uac00-\ud7af]+/gu, '')
    .trim();
}

function shouldUseHangulAlignment(context?: ForcedAlignmentRunContext) {
  return String(context?.profileId || config.alignmentProfileId || '').trim() === 'ko-kresnik-xlsr-korean-v1';
}

function normalizeKoreanProfileToken(value: string, context?: ForcedAlignmentRunContext) {
  return shouldUseHangulAlignment(context)
    ? normalizeKoreanHangulAlignmentToken(value)
    : normalizeKoreanAlignmentToken(value);
}

function romanizeKoreanToken(value: string) {
  const source = String(value || '').normalize('NFKC').trim();
  if (!source) return '';
  const romanized = romanizeHangul(source);
  return normalizeKoreanAlignmentToken(romanized);
}

function projectKoreanToken(value: string, context?: ForcedAlignmentRunContext) {
  return shouldUseHangulAlignment(context)
    ? normalizeKoreanHangulAlignmentToken(value)
    : romanizeKoreanToken(value);
}

function getWordEnd(word: AlignmentWordLike) {
  const end = Number(word?.end_ts);
  if (Number.isFinite(end)) return end;
  return Number(word?.start_ts) || 0;
}

function projectKoreanWords(segment: AlignmentSegmentLike<AlignmentWordLike>, context?: ForcedAlignmentRunContext) {
  if (Array.isArray(segment?.words) && segment.words.length > 0) {
    const cloned = segment.words
      .map((word) => ({
        ...word,
        normalizedReading: projectKoreanToken(word.text || '', context),
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
      normalizedReading: projectKoreanToken(token, context),
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
      normalizedReading: projectKoreanToken(token, context),
      start_ts: Number(tokenStart.toFixed(3)),
      end_ts: Number(tokenEnd.toFixed(3)),
    };
  });
}

const basePack = createSimpleNoSpaceLanguagePack(import.meta.url);
if (!basePack.noSpace) {
  throw new Error('Korean no-space module is missing.');
}

const noSpace = {
  ...basePack.noSpace,
  getVariantDebug() {
    return {
      alignmentMinAppliedRatio: config.alignmentMinAppliedRatio ?? 0.22,
      alignmentMinAvgConfidence: config.alignmentMinAvgConfidence ?? 0.5,
      cjkMergeSoftGapSec: config.cjkMergeSoftGapSec ?? 0.11,
      cjkMergeHardGapSec: config.cjkMergeHardGapSec ?? 0.24,
      cjkMergeMaxPhraseChars: config.cjkMergeMaxPhraseChars ?? 12,
      cjkMergeMaxPhraseDurationSec: config.cjkMergeMaxPhraseDurationSec ?? 1.7,
      cjkSplitMaxDurationSec: config.cjkSplitMaxDurationSec ?? 2.45,
      cjkSplitStrongPauseSec: config.cjkSplitStrongPauseSec ?? 0.34,
      cjkSplitClausePauseSec: config.cjkSplitClausePauseSec ?? 0.18,
      cjkSplitMaxChars: config.cjkSplitMaxChars ?? 14,
      cjkSplitMinCharsForBreak: config.cjkSplitMinCharsForBreak ?? 6,
      cjkSparseMinTextLength: config.cjkSparseMinTextLength ?? 4,
      cjkSparseSingleWordMinTextLength: config.cjkSparseSingleWordMinTextLength ?? 4,
      cjkSparseLowCoverageRatio: config.cjkSparseLowCoverageRatio ?? 0.38,
      cjkSparseMediumCoverageRatio: config.cjkSparseMediumCoverageRatio ?? 0.55,
      cjkSparseMediumCoverageMaxWords: config.cjkSparseMediumCoverageMaxWords ?? 3,
    };
  },
};

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
      fallbackProfileId: config.fallbackAlignmentProfileId,
      fallbackProgressLabel: 'Korean MMS forced alignment fallback',
      clipPaddingSec: config.clipPaddingSec,
      minRunConfidence: config.minRunConfidence,
      minOverallConfidence: config.minOverallConfidence,
      minOverallAlignedRatio: config.minOverallAlignedRatio,
      progressLabel: config.progressLabel,
    };
  },
  async projectWordReadings(
    segment: AlignmentSegmentLike<AlignmentWordLike>,
    _language?: string,
    _sampleText?: string,
    context?: ForcedAlignmentRunContext
  ) {
    const words = projectKoreanWords(segment, context);
    return Array.isArray(words) && words.length > 0 ? words : null;
  },
  extractAlignableRuns(words: AlignmentWordLike[], context?: ForcedAlignmentRunContext) {
    const annotated = words
      .map((word, index) => ({
        index,
        word,
        normalized: normalizeKoreanProfileToken(word?.normalizedReading || word?.text || '', context),
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
  normalizeAlignmentToken: normalizeKoreanProfileToken,
  supportsAlignmentToken(value: string, context?: ForcedAlignmentRunContext) {
    return shouldUseHangulAlignment(context)
      ? /[\uac00-\ud7af]/u.test(String(value || ''))
      : /[A-Za-z0-9]/.test(String(value || ''));
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
  noSpace,
  forcedAlignment,
  localAsrPrompt,
};

export const koreanLanguagePack = languagePack;
