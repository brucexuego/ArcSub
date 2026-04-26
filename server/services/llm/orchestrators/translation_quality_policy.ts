export type TranslationQualityMode = 'plain_probe' | 'template_validated' | 'json_strict';
export type TranslationPromptContract = 'none' | 'template' | 'custom' | 'template_custom';
export type TranslationAlignmentContract = 'line_safe' | 'line_safe_json_repair';

export interface TranslationQualityIntent {
  qualityMode: TranslationQualityMode;
  promptContract: TranslationPromptContract;
  alignmentContract: TranslationAlignmentContract;
  hasPromptTemplate: boolean;
  hasCustomPrompt: boolean;
  usesTemplatePrompt: boolean;
  usesJsonRepair: boolean;
  minimalQualityChecks: boolean;
}

export function normalizeTranslationPromptTemplateId(promptTemplateId?: string | null) {
  return String(promptTemplateId || '').trim().toLowerCase();
}

export function resolveTranslationQualityIntent(input: {
  promptTemplateId?: string | null;
  prompt?: string | null;
  enableJsonLineRepair?: boolean;
}): TranslationQualityIntent {
  const hasPromptTemplate = Boolean(normalizeTranslationPromptTemplateId(input.promptTemplateId));
  const hasCustomPrompt = Boolean(String(input.prompt || '').trim());
  const usesJsonRepair = input.enableJsonLineRepair !== false;
  const promptContract: TranslationPromptContract =
    hasPromptTemplate && hasCustomPrompt
      ? 'template_custom'
      : hasPromptTemplate
        ? 'template'
        : hasCustomPrompt
          ? 'custom'
          : 'none';
  const qualityMode: TranslationQualityMode = usesJsonRepair
    ? 'json_strict'
    : hasPromptTemplate || hasCustomPrompt
      ? 'template_validated'
      : 'plain_probe';

  return {
    qualityMode,
    promptContract,
    alignmentContract: usesJsonRepair ? 'line_safe_json_repair' : 'line_safe',
    hasPromptTemplate,
    hasCustomPrompt,
    usesTemplatePrompt: hasPromptTemplate || hasCustomPrompt,
    usesJsonRepair,
    minimalQualityChecks: qualityMode === 'plain_probe',
  };
}

export function resolveTranslationQualityMode(input: {
  promptTemplateId?: string | null;
  prompt?: string | null;
  enableJsonLineRepair?: boolean;
}): TranslationQualityMode {
  return resolveTranslationQualityIntent(input).qualityMode;
}

export function resolveLegacyTranslationQualityMode(input: {
  promptTemplateId?: string | null;
  prompt?: string | null;
  enableJsonLineRepair?: boolean;
}): TranslationQualityMode {
  if (input.enableJsonLineRepair !== false) {
    return 'json_strict';
  }
  if (normalizeTranslationPromptTemplateId(input.promptTemplateId) || String(input.prompt || '').trim()) {
    return 'template_validated';
  }
  return 'plain_probe';
}

export function isPlainTranslationProbeMode(mode: TranslationQualityMode) {
  return mode === 'plain_probe';
}

export function usesTemplateValidatedQualityChecks(mode: TranslationQualityMode) {
  return mode === 'template_validated';
}

export function usesJsonStrictAlignment(mode: TranslationQualityMode) {
  return mode === 'json_strict';
}
