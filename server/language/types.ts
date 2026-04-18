export type TranslationMismatchHeuristic =
  | 'generic'
  | 'english'
  | 'latin-stopwords'
  | 'japanese'
  | 'korean'
  | 'chinese';

export interface AsrLanguagePolicy {
  noSpaceScript: boolean;
  segmenterLocale: string;
  whisperCode?: string;
  qwenName?: string;
  useBuiltInLocalPrompt: boolean;
}

export interface TranslationLanguagePolicy {
  stopwords: string[];
  instructionLines: string[];
  mismatchHeuristic: TranslationMismatchHeuristic;
  autoNaturalization: boolean;
  traditionalChineseTarget: boolean;
  simplifiedChineseTarget: boolean;
}

export interface LanguagePolicy {
  key: string;
  aliases: string[];
  englishName: string;
  nativeName: string;
  asr: AsrLanguagePolicy;
  translation: TranslationLanguagePolicy;
}
