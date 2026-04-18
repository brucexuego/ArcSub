import path from 'path';
import { createRequire } from 'module';
import type { IpadicFeatures, Tokenizer as KuromojiTokenizer } from 'kuromoji';
import { loadJsonFromModuleDir, normalizeLanguageCode } from '../shared/json_loader.js';
import { genericFallbackSegmentNoSpaceLexicalUnits, normalizeNoSpaceAlignmentText } from '../shared/no_space_utils.js';
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

const require = createRequire(import.meta.url);
const kuromoji = require('kuromoji') as typeof import('kuromoji');

type JapaneseConfig = NoSpaceLanguageConfig & {
  alignmentProfileId: string;
  clipPaddingSec: number;
  minRunConfidence: number;
  minOverallConfidence: number;
  minOverallAlignedRatio: number;
  localAsrPrompt?: string;
};

type JapaneseTokenConfig = {
  standaloneTokens: string[];
  shortTailTokens: string[];
};

type JapaneseAlignmentWord = AlignmentWordLike;

const config = loadJsonFromModuleDir<JapaneseConfig>(import.meta.url, 'config.json');
const tokenConfig = loadJsonFromModuleDir<JapaneseTokenConfig>(import.meta.url, 'tokens.json');
const aliasSet = new Set(config.aliases.map((alias) => normalizeLanguageCode(alias)));
const standaloneTokenSet = new Set(tokenConfig.standaloneTokens);
const shortTailTokenSet = new Set(tokenConfig.shortTailTokens);
let kuromojiTokenizerPromise: Promise<KuromojiTokenizer<IpadicFeatures>> | null = null;

function matchesLanguage(language?: string) {
  const normalized = normalizeLanguageCode(language);
  if (!normalized) return false;
  if (aliasSet.has(normalized)) return true;
  return [...aliasSet].some((alias) => normalized.startsWith(`${alias}-`));
}

function matchesJapaneseText(text?: string) {
  return /[\u3040-\u30ff]/u.test(String(text || ''));
}

function scoreJapaneseText(text?: string) {
  const source = String(text || '');
  const kanaCount = source.match(/[\u3040-\u30ff]/gu)?.length || 0;
  const kanjiCount = source.match(/[\u3400-\u9fff]/gu)?.length || 0;
  return kanaCount * 3 + kanjiCount;
}

function normalizeJapaneseSurface(value: string) {
  return String(value || '')
    .normalize('NFKC')
    .replace(/\u00a0/g, ' ')
    .replace(/\s+/g, '')
    .trim();
}

function normalizeJapaneseAlignmentToken(value: string) {
  return normalizeJapaneseSurface(value).replace(
    /[\u3001\u3002\uFF01\uFF1F!?.,\uFF0C\uFF0E\u300C\u300D\u300E\u300F\uFF08\uFF09()\uFF3B\uFF3D\u3010\u3011\u3008\u3009\u300A\u300B\u3014\u3015\u2026\u2025\u30FB:\uFF1A;\uFF1B"'`\u201C\u201D\u2018\u2019]/gu,
    ''
  );
}

function supportsSurfaceAlignmentToken(value: string) {
  const normalized = normalizeJapaneseAlignmentToken(value);
  if (!normalized) return false;
  return /^[A-Za-z0-9\u3040-\u30ff\u3400-\u9fff\u3005\u30F6\u30FC%\uFF05\uFFE5\u00A5\u5186]+$/u.test(normalized);
}

function getWordEnd(word: AlignmentWordLike) {
  const end = Number(word?.end_ts);
  if (Number.isFinite(end)) return end;
  return Number(word?.start_ts) || 0;
}

function toFiniteNumber(value: unknown, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function getKuromojiDictPath() {
  const packageDir = path.dirname(require.resolve('kuromoji/package.json'));
  return path.join(packageDir, 'dict');
}

async function getKuromojiTokenizer(): Promise<KuromojiTokenizer<IpadicFeatures>> {
  if (!kuromojiTokenizerPromise) {
    kuromojiTokenizerPromise = new Promise((resolve, reject) => {
      kuromoji.builder({ dicPath: getKuromojiDictPath() }).build((err, tokenizer) => {
        if (err || !tokenizer) {
          reject(err || new Error('Failed to initialize kuromoji tokenizer.'));
          return;
        }
        resolve(tokenizer);
      });
    });
  }
  return kuromojiTokenizerPromise;
}

const noSpace: NoSpaceLanguageModule = {
  config,
  matchesLanguage,
  matchesText: matchesJapaneseText,
  normalizeAlignmentText: normalizeNoSpaceAlignmentText,
  fallbackSegmentLexicalUnits: genericFallbackSegmentNoSpaceLexicalUnits,
  shouldPreferStandaloneToken(token: string) {
    return standaloneTokenSet.has(String(token || '').trim());
  },
  startsWithStandaloneToken(token: string) {
    const normalized = String(token || '').trim();
    return [...standaloneTokenSet].some((entry) =>
      new RegExp(`^${entry}(?:[\\u3001\\u3002,\\s!?]|$)`, 'u').test(normalized)
    );
  },
  isShortContinuationToken(token: string) {
    return /^[\u3040-\u309f\u30fc]{1,3}$/u.test(String(token || '').trim());
  },
  shouldMergeContinuation(bufferText: string, nextText: string, gap: number, combinedText: string, combinedDuration: number) {
    if (!(gap <= 0.14 && combinedDuration <= 1.9 && combinedText.length <= 14)) {
      return false;
    }
    if (!/[\u3040-\u30ff\u3400-\u9fff]$/u.test(String(bufferText || '').trim())) {
      return false;
    }
    if (!this.isShortContinuationToken(nextText)) {
      return false;
    }
    if (this.shouldPreferStandaloneToken(nextText)) {
      return false;
    }
    return true;
  },
  isShortTailToken(token: string) {
    return shortTailTokenSet.has(String(token || '').trim());
  },
};

const forcedAlignment: ForcedAlignmentLanguageModule<JapaneseAlignmentWord, AlignmentSegmentLike<JapaneseAlignmentWord>> = {
  key: config.key,
  aliases: config.aliases,
  matchesLanguage,
  matchesText(text?: string) {
    return matchesJapaneseText(text);
  },
  scoreText(text?: string) {
    return scoreJapaneseText(text);
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
      progressLabel: 'Japanese surface-form forced alignment',
    };
  },
  normalizeAlignmentToken: normalizeJapaneseAlignmentToken,
  supportsAlignmentToken: supportsSurfaceAlignmentToken,
  async projectWordReadings(segment: AlignmentSegmentLike<JapaneseAlignmentWord>) {
    const words = Array.isArray(segment?.words) ? segment.words.map((word) => ({ ...word })) : [];
    const surfaceText = normalizeJapaneseSurface(words.length > 0 ? words.map((word) => word.text || '').join('') : segment?.text || '');
    if (!surfaceText || words.length === 0) {
      return null;
    }

    const baseWords = words
      .map((word) => ({
        ...word,
        normalizedSurface: normalizeJapaneseSurface(word.text || ''),
      }))
      .filter((word) => word.normalizedSurface.length > 0);
    const baseSurface = baseWords.map((word) => word.normalizedSurface).join('');
    if (!baseSurface || baseSurface !== surfaceText) {
      return null;
    }

    const tokenizer = await getKuromojiTokenizer();
    const tokens = tokenizer.tokenize(surfaceText);
    const tokenSurface = tokens.map((token) => normalizeJapaneseSurface(token.surface_form || '')).join('');
    if (!tokenSurface || tokenSurface !== surfaceText) {
      return null;
    }

    const charSpans: Array<{ char: string; start_ts: number; end_ts: number; probability?: number }> = [];
    for (const word of baseWords) {
      const chars = Array.from(word.normalizedSurface);
      const start = toFiniteNumber(word.start_ts, 0);
      const end = getWordEnd(word);
      const duration = Math.max(0.04 * chars.length, end - start);
      for (let index = 0; index < chars.length; index += 1) {
        charSpans.push({
          char: chars[index],
          start_ts: Number((start + duration * (index / chars.length)).toFixed(3)),
          end_ts: Number((start + duration * ((index + 1) / chars.length)).toFixed(3)),
          probability: word.probability,
        });
      }
    }

    const projected: JapaneseAlignmentWord[] = [];
    let charCursor = 0;
    for (const token of tokens) {
      const surface = normalizeJapaneseSurface(token.surface_form || '');
      if (!surface) continue;
      const surfaceChars = Array.from(surface);
      const slice = charSpans.slice(charCursor, charCursor + surfaceChars.length);
      if (slice.length !== surfaceChars.length || slice.map((item) => item.char).join('') !== surface) {
        return null;
      }
      const probabilityValues = slice
        .map((item) => toFiniteNumber(item.probability, Number.NaN))
        .filter((value) => Number.isFinite(value));
      projected.push({
        text: surface,
        start_ts: slice[0].start_ts,
        end_ts: slice[slice.length - 1].end_ts,
        probability:
          probabilityValues.length > 0
            ? Number((probabilityValues.reduce((sum, value) => sum + value, 0) / probabilityValues.length).toFixed(4))
            : undefined,
      });
      charCursor += surfaceChars.length;
    }

    return projected.length > 0 ? projected : null;
  },
  extractAlignableRuns(words: JapaneseAlignmentWord[]) {
    const annotated = words.map((word, index) => {
      const normalized = normalizeJapaneseAlignmentToken(word?.text || '');
      return {
        index,
        word,
        token: normalized,
        alignable: supportsSurfaceAlignmentToken(normalized),
      };
    });

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
              normalized: item.token,
            })
          ),
        });
      }
      current = [];
    };

    for (const item of annotated) {
      if (!item.alignable) {
        flush();
        continue;
      }
      if (current.length > 0) {
        const previous = current[current.length - 1];
        const gap = toFiniteNumber(item.word.start_ts, 0) - getWordEnd(previous.word);
        if (gap > 0.3) {
          flush();
        }
      }
      current.push(item);
    }
    flush();

    return runs.filter((run) => run.tokens.reduce((sum, token) => sum + Array.from(token.normalized).length, 0) >= 2);
  },
};

const localAsrPrompt: LocalAsrPromptModule = {
  key: config.key,
  aliases: config.aliases,
  matchesLanguage,
  matchesText(text?: string) {
    return matchesJapaneseText(text);
  },
  scoreText(text?: string) {
    return scoreJapaneseText(text);
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
  key: config.key,
  aliases: config.aliases,
  noSpace,
  forcedAlignment,
  localAsrPrompt,
};
export const japaneseLanguagePack = languagePack;
