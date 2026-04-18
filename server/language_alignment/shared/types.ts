export type AlignmentWordLike = {
  text: string;
  start_ts: number;
  end_ts?: number;
  probability?: number;
  speaker?: string;
  source_segment_index?: number;
  token_index?: number;
  normalizedReading?: string;
  usesReadingProjection?: boolean;
};

export type AlignmentSegmentLike<TWord extends AlignmentWordLike = AlignmentWordLike> = {
  text: string;
  start_ts: number;
  end_ts?: number;
  speaker?: string;
  words?: TWord[];
};

export type AlignmentWordToken = {
  text: string;
  normalized: string;
};

export type AlignmentRun = {
  startIndex: number;
  endIndex: number;
  tokens: AlignmentWordToken[];
};

export type ForcedAlignmentSelectionContext = {
  language?: string;
  sampleText?: string;
  providerMeta?: Record<string, any> | null;
  localModelId?: string | null;
  localModelRuntime?: string | null;
};

export type ForcedAlignmentPlanConfig = {
  profileId: string;
  clipPaddingSec: number;
  minSegmentConfidence?: number;
  minRunConfidence?: number;
  minOverallConfidence: number;
  minOverallAlignedRatio: number;
  progressLabel: string;
};

export type NoSpaceLanguageConfig = {
  key: string;
  aliases: string[];
  segmenterLocale: string;
  noSpaceScript: boolean;
  timingMerge: boolean;
};

export interface NoSpaceLanguageModule {
  readonly config: NoSpaceLanguageConfig;
  matchesLanguage(language?: string): boolean;
  matchesText(text?: string): boolean;
  resolveVariant?(language?: string, sampleText?: string): string | null;
  getVariantDebug?(language?: string, sampleText?: string): Record<string, unknown> | null;
  normalizeAlignmentText(value: string): string;
  fallbackSegmentLexicalUnits(text: string): string[];
  shouldPreferStandaloneToken(token: string, language?: string): boolean;
  startsWithStandaloneToken(token: string, language?: string): boolean;
  isShortContinuationToken(token: string, language?: string): boolean;
  shouldMergeContinuation(
    bufferText: string,
    nextText: string,
    gap: number,
    combinedText: string,
    combinedDuration: number,
    language?: string
  ): boolean;
  isShortTailToken(token: string, language?: string): boolean;
}

export interface ForcedAlignmentLanguageModule<
  TWord extends AlignmentWordLike = AlignmentWordLike,
  TSegment extends AlignmentSegmentLike<TWord> = AlignmentSegmentLike<TWord>,
> {
  readonly key: string;
  readonly aliases: string[];
  matchesLanguage(language?: string): boolean;
  matchesText?(text?: string): boolean;
  scoreText?(text?: string): number;
  resolveVariant?(language?: string, sampleText?: string): string | null;
  getPlanConfig(
    language?: string,
    sampleText?: string,
    context?: ForcedAlignmentSelectionContext
  ): ForcedAlignmentPlanConfig;
  projectWordReadings(segment: TSegment, language?: string, sampleText?: string): Promise<TWord[] | null>;
  extractAlignableRuns(words: TWord[]): AlignmentRun[];
  normalizeAlignmentToken(value: string): string;
  supportsAlignmentToken(value: string): boolean;
}

export interface LocalAsrPromptModule {
  readonly key: string;
  readonly aliases: string[];
  matchesLanguage(language?: string): boolean;
  matchesText?(text?: string): boolean;
  scoreText?(text?: string): number;
  resolveVariant?(language?: string, sampleText?: string): string | null;
  getPrompt(language?: string, sampleText?: string): string | null;
  getDebug?(language?: string, sampleText?: string): Record<string, unknown> | null;
}

export type LanguageAlignmentPack = {
  key: string;
  aliases: string[];
  noSpace?: NoSpaceLanguageModule;
  forcedAlignment?: ForcedAlignmentLanguageModule;
  localAsrPrompt?: LocalAsrPromptModule;
};
