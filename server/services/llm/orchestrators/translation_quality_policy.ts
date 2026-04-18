export type TranslationQualityMode = 'plain_probe' | 'template_validated' | 'json_strict';

export function normalizeTranslationPromptTemplateId(promptTemplateId?: string | null) {
  return String(promptTemplateId || '').trim().toLowerCase();
}

export function resolveTranslationQualityMode(input: {
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
