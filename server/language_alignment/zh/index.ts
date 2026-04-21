import { loadJsonFromModuleDir, normalizeLanguageCode } from '../shared/json_loader.js';
import { genericFallbackSegmentNoSpaceLexicalUnits, normalizeNoSpaceAlignmentText } from '../shared/no_space_utils.js';
import { addDict, pinyin } from 'pinyin-pro';
import traditionalDict from '@pinyin-pro/data/traditional';
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
  NoSpaceLanguageModule,
} from '../shared/types.js';

type ChineseConfig = NoSpaceLanguageConfig & {
  scriptPattern: string;
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
  maxContinuationGapSec: number;
  maxContinuationDurationSec: number;
  maxContinuationChars: number;
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
};

type ChineseConfigOverride = Partial<
  Pick<
    ChineseConfig,
    | 'alignmentProfileId'
    | 'clipPaddingSec'
    | 'minRunConfidence'
    | 'minOverallConfidence'
    | 'minOverallAlignedRatio'
    | 'alignmentMinAppliedRatio'
    | 'alignmentMinAvgConfidence'
    | 'progressLabel'
    | 'maxRunGapSec'
    | 'minRunTokenLength'
    | 'maxContinuationGapSec'
    | 'maxContinuationDurationSec'
    | 'maxContinuationChars'
    | 'cjkMergeSoftGapSec'
    | 'cjkMergeHardGapSec'
    | 'cjkMergeMaxPhraseChars'
    | 'cjkMergeMaxPhraseDurationSec'
    | 'cjkSplitMaxDurationSec'
    | 'cjkSplitStrongPauseSec'
    | 'cjkSplitClausePauseSec'
    | 'cjkSplitMaxChars'
    | 'cjkSplitMinCharsForBreak'
    | 'cjkSparseMinTextLength'
    | 'cjkSparseSingleWordMinTextLength'
    | 'cjkSparseLowCoverageRatio'
    | 'cjkSparseMediumCoverageRatio'
    | 'cjkSparseMediumCoverageMaxWords'
    | 'localAsrPrompt'
  >
>;

type ChineseTokenConfig = {
  standaloneTokens: string[];
  shortTailTokens: string[];
  continuationTokens: string[];
};

const config = loadJsonFromModuleDir<NoSpaceLanguageConfig>(import.meta.url, 'config.json');
const commonConfig = loadJsonFromModuleDir<ChineseConfig>(import.meta.url, 'config.common.json');
const zhCnConfig = loadJsonFromModuleDir<ChineseConfigOverride>(import.meta.url, 'config.zh-cn.json');
const zhTwConfig = loadJsonFromModuleDir<ChineseConfigOverride>(import.meta.url, 'config.zh-tw.json');
const commonTokenConfig = loadJsonFromModuleDir<ChineseTokenConfig>(import.meta.url, 'tokens.common.json');
const zhCnTokenConfig = loadJsonFromModuleDir<ChineseTokenConfig>(import.meta.url, 'tokens.zh-cn.json');
const zhTwTokenConfig = loadJsonFromModuleDir<ChineseTokenConfig>(import.meta.url, 'tokens.zh-tw.json');

const aliasSet = new Set(config.aliases.map((alias) => normalizeLanguageCode(alias)));
const scriptRegex = new RegExp(commonConfig.scriptPattern, 'u');
let pinyinDictInitialized = false;

type ChineseVariantKey = 'common' | 'zh-cn' | 'zh-tw';

type ChineseTokenSets = {
  standalone: Set<string>;
  shortTail: Set<string>;
  continuation: Set<string>;
};

const traditionalVariantRegex = /[\u5c0d\u9ede\u570b\u55ce\u55ce\u904e\u9084\u6c92\u500b\u500b\u6703\u61c9\u96fb\u8b93\u908a\u982d\u9ad4\u9ede]/u;
const simplifiedVariantRegex = /[\u5bf9\u70b9\u56fd\u5417\u8fc7\u8fd8\u6ca1\u4e2a\u4f1a\u5e94\u7535\u8ba9\u8fb9\u5934\u4f53]/u;

function scoreChineseText(text?: string) {
  const hanCount = String(text || '').match(/[\u3400-\u9fff]/gu)?.length || 0;
  return hanCount * 3;
}

function buildTokenSet(values: string[]) {
  return new Set(values.map((token) => token.trim()).filter(Boolean));
}

function mergeTokenSets(base: ChineseTokenConfig, override?: ChineseTokenConfig): ChineseTokenSets {
  return {
    standalone: buildTokenSet([...(base.standaloneTokens || []), ...(override?.standaloneTokens || [])]),
    shortTail: buildTokenSet([...(base.shortTailTokens || []), ...(override?.shortTailTokens || [])]),
    continuation: buildTokenSet([...(base.continuationTokens || []), ...(override?.continuationTokens || [])]),
  };
}

const tokenSetsByVariant: Record<ChineseVariantKey, ChineseTokenSets> = {
  common: mergeTokenSets(commonTokenConfig),
  'zh-cn': mergeTokenSets(commonTokenConfig, zhCnTokenConfig),
  'zh-tw': mergeTokenSets(commonTokenConfig, zhTwTokenConfig),
};

const configByVariant: Record<ChineseVariantKey, ChineseConfig> = {
  common: commonConfig,
  'zh-cn': {
    ...commonConfig,
    ...zhCnConfig,
  },
  'zh-tw': {
    ...commonConfig,
    ...zhTwConfig,
  },
};

function matchesLanguage(language?: string) {
  const normalized = normalizeLanguageCode(language);
  if (!normalized) return false;
  if (aliasSet.has(normalized)) return true;
  return [...aliasSet].some((alias) => normalized.startsWith(`${alias}-`));
}

function resolveVariant(language?: string): ChineseVariantKey {
  const normalized = normalizeLanguageCode(language);
  if (normalized === 'zh-cn' || normalized.startsWith('zh-cn-')) return 'zh-cn';
  if (normalized === 'zh-tw' || normalized.startsWith('zh-tw-')) return 'zh-tw';
  return 'common';
}

function detectVariantFromText(sampleText?: string): ChineseVariantKey {
  const text = String(sampleText || '');
  if (!text) return 'common';
  const traditionalMatches = text.match(traditionalVariantRegex)?.length || 0;
  const simplifiedMatches = text.match(simplifiedVariantRegex)?.length || 0;
  if (traditionalMatches > simplifiedMatches && traditionalMatches > 0) return 'zh-tw';
  if (simplifiedMatches > traditionalMatches && simplifiedMatches > 0) return 'zh-cn';
  return 'common';
}

function resolveVariantWithText(language?: string, sampleText?: string): ChineseVariantKey {
  const explicit = resolveVariant(language);
  if (explicit !== 'common') return explicit;
  return detectVariantFromText(sampleText);
}

function getTokenSets(language?: string, sampleText?: string) {
  return tokenSetsByVariant[resolveVariantWithText(language, sampleText)];
}

function getVariantConfig(language?: string, sampleText?: string) {
  return configByVariant[resolveVariantWithText(language, sampleText)];
}

function getWordEnd(word: AlignmentWordLike) {
  const end = Number(word?.end_ts);
  if (Number.isFinite(end)) return end;
  return Number(word?.start_ts) || 0;
}

function normalizeChineseAlignmentToken(value: string) {
  return String(value || '')
    .normalize('NFKC')
    .replace(/\u00a0/g, ' ')
    .replace(/\s+/g, '')
    .replace(/[’‘`´]/gu, "'")
    .replace(/[^A-Za-z0-9]+/gu, '')
    .toLowerCase()
    .trim();
}

function supportsChineseAlignmentToken(value: string) {
  return /[A-Za-z0-9]/u.test(String(value || ''));
}

function ensurePinyinTraditionalDict() {
  if (!pinyinDictInitialized) {
    addDict(traditionalDict as Record<string, string>);
    pinyinDictInitialized = true;
  }
}

function toChineseRomanizedToken(value: string, variant: ChineseVariantKey) {
  const source = String(value || '').normalize('NFKC').trim();
  if (!source) return '';
  ensurePinyinTraditionalDict();
  const romanized = pinyin(source, {
    toneType: 'none',
    type: 'array',
    nonZh: 'consecutive',
    v: false,
    traditional: variant === 'zh-tw',
  });
  const joined = Array.isArray(romanized) ? romanized.join('') : String(romanized || '');
  return normalizeChineseAlignmentToken(joined);
}

function projectChineseWords(segment: AlignmentSegmentLike<AlignmentWordLike>, language?: string, sampleText?: string) {
  const variant = resolveVariantWithText(language, sampleText || segment?.text || '');
  if (Array.isArray(segment?.words) && segment.words.length > 0) {
    const cloned = segment.words
      .map((word) => ({
        ...word,
        normalizedReading: toChineseRomanizedToken(word.text || '', variant),
      }))
      .filter((word) => String(word.normalizedReading || '').length > 0);
    return cloned.length > 0 ? cloned : null;
  }

  const lexicalUnits = genericFallbackSegmentNoSpaceLexicalUnits(segment?.text || '').filter((unit) => toChineseRomanizedToken(unit, variant).length > 0);
  if (lexicalUnits.length === 0) {
    return null;
  }

  const start = Number(segment?.start_ts);
  const end = Number(segment?.end_ts);
  if (!(Number.isFinite(start) && Number.isFinite(end) && end > start)) {
    return lexicalUnits.map((unit) => ({ text: unit, start_ts: 0, end_ts: 0 }));
  }

  const totalChars = lexicalUnits.reduce((sum, unit) => sum + unit.length, 0) || lexicalUnits.length;
  let cursor = 0;
  return lexicalUnits.map((unit) => {
    const tokenWeight = unit.length || 1;
    const unitStart = start + ((end - start) * cursor) / totalChars;
    cursor += tokenWeight;
    const unitEnd = start + ((end - start) * cursor) / totalChars;
    return {
      text: unit,
      normalizedReading: toChineseRomanizedToken(unit, variant),
      start_ts: Number(unitStart.toFixed(3)),
      end_ts: Number(unitEnd.toFixed(3)),
    };
  });
}

const noSpace: NoSpaceLanguageModule = {
  config,
  matchesLanguage,
  matchesText(text?: string) {
    return scriptRegex.test(String(text || ''));
  },
  resolveVariant(language?: string, sampleText?: string) {
    return resolveVariantWithText(language, sampleText);
  },
  getVariantDebug(language?: string, sampleText?: string) {
    const variant = resolveVariantWithText(language, sampleText);
    const variantConfig = getVariantConfig(language, sampleText);
    return {
      variant,
      alignmentMinAppliedRatio: variantConfig.alignmentMinAppliedRatio ?? 0.22,
      alignmentMinAvgConfidence: variantConfig.alignmentMinAvgConfidence ?? 0.5,
      maxContinuationGapSec: variantConfig.maxContinuationGapSec,
      maxContinuationDurationSec: variantConfig.maxContinuationDurationSec,
      maxContinuationChars: variantConfig.maxContinuationChars,
      cjkMergeSoftGapSec: variantConfig.cjkMergeSoftGapSec ?? 0.11,
      cjkMergeHardGapSec: variantConfig.cjkMergeHardGapSec ?? 0.24,
      cjkMergeMaxPhraseChars: variantConfig.cjkMergeMaxPhraseChars ?? 12,
      cjkMergeMaxPhraseDurationSec: variantConfig.cjkMergeMaxPhraseDurationSec ?? 1.7,
      cjkSplitMaxDurationSec: variantConfig.cjkSplitMaxDurationSec ?? 2.45,
      cjkSplitStrongPauseSec: variantConfig.cjkSplitStrongPauseSec ?? 0.34,
      cjkSplitClausePauseSec: variantConfig.cjkSplitClausePauseSec ?? 0.18,
      cjkSplitMaxChars: variantConfig.cjkSplitMaxChars ?? 14,
      cjkSplitMinCharsForBreak: variantConfig.cjkSplitMinCharsForBreak ?? 6,
      cjkSparseMinTextLength: variantConfig.cjkSparseMinTextLength ?? 4,
      cjkSparseSingleWordMinTextLength: variantConfig.cjkSparseSingleWordMinTextLength ?? 4,
      cjkSparseLowCoverageRatio: variantConfig.cjkSparseLowCoverageRatio ?? 0.38,
      cjkSparseMediumCoverageRatio: variantConfig.cjkSparseMediumCoverageRatio ?? 0.55,
      cjkSparseMediumCoverageMaxWords: variantConfig.cjkSparseMediumCoverageMaxWords ?? 3,
    };
  },
  normalizeAlignmentText: normalizeNoSpaceAlignmentText,
  fallbackSegmentLexicalUnits: genericFallbackSegmentNoSpaceLexicalUnits,
  shouldPreferStandaloneToken(token: string, language?: string) {
    return getTokenSets(language, token).standalone.has(String(token || '').trim());
  },
  startsWithStandaloneToken(token: string, language?: string) {
    const normalized = String(token || '').trim();
    return [...getTokenSets(language, token).standalone].some((entry) =>
      new RegExp(`^${entry}(?:[\\uFF0C\\u3002,.!?\\uFF1F\\uFF01\\s]|$)`, 'u').test(normalized)
    );
  },
  isShortContinuationToken(token: string, language?: string) {
    const normalized = String(token || '').trim();
    if (!normalized) return false;
    const tokenSets = getTokenSets(language, token);
    if (tokenSets.continuation.has(normalized)) return true;
    return /^[\u3400-\u9fff]{1,2}$/u.test(normalized) && !tokenSets.standalone.has(normalized);
  },
  shouldMergeContinuation(bufferText: string, nextText: string, gap: number, combinedText: string, combinedDuration: number, language?: string) {
    const variantConfig = getVariantConfig(language, combinedText);
    if (
      !(
        gap <= variantConfig.maxContinuationGapSec &&
        combinedDuration <= variantConfig.maxContinuationDurationSec &&
        combinedText.length <= variantConfig.maxContinuationChars
      )
    ) {
      return false;
    }
    if (!/[\u3400-\u9fff]$/u.test(String(bufferText || '').trim())) {
      return false;
    }
    const normalizedNext = String(nextText || '').trim();
    if (!this.isShortContinuationToken(normalizedNext, language)) {
      return false;
    }
    if (this.shouldPreferStandaloneToken(normalizedNext, language)) {
      return false;
    }
    return true;
  },
  isShortTailToken(token: string, language?: string) {
    return getTokenSets(language, token).shortTail.has(String(token || '').trim());
  },
};

const forcedAlignment: ForcedAlignmentLanguageModule = {
  key: config.key,
  aliases: config.aliases,
  matchesLanguage,
  matchesText(text?: string) {
    return scriptRegex.test(String(text || ''));
  },
  scoreText(text?: string) {
    return scoreChineseText(text);
  },
  resolveVariant(language?: string, sampleText?: string) {
    return resolveVariantWithText(language, sampleText);
  },
  getPlanConfig(
    language?: string,
    sampleText?: string,
    _context?: ForcedAlignmentSelectionContext
  ): ForcedAlignmentPlanConfig {
    const variantConfig = getVariantConfig(language, sampleText);
    return {
      profileId: variantConfig.alignmentProfileId,
      clipPaddingSec: variantConfig.clipPaddingSec,
      minRunConfidence: variantConfig.minRunConfidence,
      minOverallConfidence: variantConfig.minOverallConfidence,
      minOverallAlignedRatio: variantConfig.minOverallAlignedRatio,
      progressLabel: variantConfig.progressLabel,
    };
  },
  async projectWordReadings(segment: AlignmentSegmentLike<AlignmentWordLike>, language?: string, sampleText?: string) {
    const words = projectChineseWords(segment, language, sampleText);
    return Array.isArray(words) && words.length > 0 ? words : null;
  },
  extractAlignableRuns(words: AlignmentWordLike[]) {
    const annotated = words
      .map((word, index) => ({
        index,
        word,
        normalized: normalizeChineseAlignmentToken(word?.normalizedReading || word?.text || ''),
      }))
      .filter((item) => item.normalized.length > 0 && supportsChineseAlignmentToken(item.normalized));

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
        const currentText = `${current.map((entry) => entry.word.text || '').join('')}${item.word.text || ''}`;
        if (gap > getVariantConfig(undefined, currentText).maxRunGapSec) {
          flush();
        }
      }
      current.push(item);
    }
    flush();

    return runs.filter((run) => {
      const sampleText = run.tokens.map((token) => token.text || '').join('');
      const variantConfig = getVariantConfig(undefined, sampleText);
      return run.tokens.some((token) => token.normalized.length >= variantConfig.minRunTokenLength);
    });
  },
  normalizeAlignmentToken: normalizeChineseAlignmentToken,
  supportsAlignmentToken: supportsChineseAlignmentToken,
};

const localAsrPrompt: LocalAsrPromptModule = {
  key: config.key,
  aliases: config.aliases,
  matchesLanguage,
  matchesText(text?: string) {
    return scriptRegex.test(String(text || ''));
  },
  scoreText(text?: string) {
    return scoreChineseText(text);
  },
  resolveVariant(language?: string, sampleText?: string) {
    return resolveVariantWithText(language, sampleText);
  },
  getPrompt(language?: string, sampleText?: string) {
    return getVariantConfig(language, sampleText).localAsrPrompt || commonConfig.localAsrPrompt || null;
  },
  getDebug(language?: string, sampleText?: string) {
    const variant = resolveVariantWithText(language, sampleText);
    return {
      variant,
      hasPrompt: Boolean(getVariantConfig(language, sampleText).localAsrPrompt || commonConfig.localAsrPrompt),
    };
  },
};

export const languagePack: LanguageAlignmentPack = {
  key: config.key,
  aliases: config.aliases,
  noSpace,
  forcedAlignment,
  localAsrPrompt,
};

export const chineseLanguagePack = languagePack;
