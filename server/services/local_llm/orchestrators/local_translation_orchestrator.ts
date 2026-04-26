import { getLocalModelInstallDir, type LocalModelDefinition } from '../../../local_model_catalog.js';
import { OpenvinoRuntimeManager } from '../../../openvino_runtime_manager.js';
import type { LocalTranslationBatchDebugInfo, TranslationDebugInfo, TranslationResult } from '../../translation_service.js';
import type { LocalOpenvinoResolvedProfile, LocalTranslateModelStrategy } from '../types.js';
import type { LocalTranslationResolvedProfile } from '../model_profiles.js';
import type { EffectiveLocalTranslationProfile } from '../effective_profile.js';
import type { TranslationQualityMode } from '../../llm/orchestrators/translation_quality_policy.js';
import type { RunIssue } from '../../../../shared/run_monitor.js';

type TranslateProgressFn = (message: string) => void;

interface LineSafeUnit {
  index: number;
  marker: string;
  prefix: string;
  content: string;
  speakerTag: string | null;
}

type TranslationQualityIssueCode =
  | 'empty_output'
  | 'repetition_loop'
  | 'adjacent_duplicate'
  | 'pass_through'
  | 'zh_tw_naturalization_needed'
  | 'target_lang_mismatch'
  | 'line_count_loss'
  | 'marker_loss';

interface TranslationQualityAssessmentContext {
  expectedLineCount?: number;
  requireMarkers?: boolean;
  sourceProfile?: unknown;
}

function getPassThroughRisk(warnings: string[]) {
  if (warnings.includes('quality_issue_pass_through')) return 'high' as const;
  if (warnings.includes('quality_retry_triggered')) return 'medium' as const;
  return 'low' as const;
}

function getRepetitionRisk(warnings: string[]) {
  if (warnings.includes('quality_issue_repetition_loop') || warnings.includes('quality_issue_adjacent_duplicate')) {
    return 'high' as const;
  }
  return 'low' as const;
}

function getMarkerPreservation(sourceHasStructuredPrefixes: boolean, warnings: string[]) {
  if (!sourceHasStructuredPrefixes) return undefined;
  return warnings.includes('quality_issue_marker_loss') ? ('lost' as const) : ('ok' as const);
}

function qualityIssueToWarningCode(issue: TranslationQualityIssueCode) {
  switch (issue) {
    case 'empty_output':
      return 'quality_issue_empty_output';
    case 'repetition_loop':
      return 'quality_issue_repetition_loop';
    case 'adjacent_duplicate':
      return 'quality_issue_adjacent_duplicate';
    case 'pass_through':
      return 'quality_issue_pass_through';
    case 'zh_tw_naturalization_needed':
      return 'quality_issue_zh_tw_naturalization_needed';
    case 'target_lang_mismatch':
      return 'quality_issue_target_lang_mismatch';
    case 'line_count_loss':
      return 'quality_issue_line_count_loss';
    case 'marker_loss':
      return 'quality_issue_marker_loss';
    default:
      return null;
  }
}

function buildTranslationWarningIssues(warnings: string[]): RunIssue[] {
  const qualityInfoCodes = new Set([
    'strict_retry_applied',
    'local_strict_retry_applied',
    'local_batch_translation_applied',
    'local_recursive_chunk_split_applied',
    'local_single_line_retry_applied',
    'translategemma_batch_translation_applied',
    'translategemma_recursive_chunk_split_applied',
    'translategemma_single_line_retry_applied',
    'residual_line_retry_applied',
    'line_safe_alignment_applied',
    'line_index_rebind_applied',
    'line_json_map_repair_applied',
    'line_json_map_pre_split_applied',
    'line_json_map_policy_split',
  ]);
  const qualityWarningCodes = new Set([
    'quality_retry_triggered',
    'post_repair_quality_retry_triggered',
    'residual_line_retry_triggered',
    'line_json_map_partial_fallback',
    'line_json_map_policy_single_line_fallback',
    'line_json_map_policy_source_fallback',
    'line_json_map_repair_disabled',
    'line_alignment_repair_failed',
    'local_repetition_loop_detected',
    'cloud_context_parse_failed',
    'cloud_context_chunk_split',
    'cloud_context_split_depth_exhausted',
    'cloud_context_single_line_fallback',
    'quality_issue_target_lang_mismatch',
    'quality_issue_pass_through',
    'quality_issue_empty_output',
    'quality_issue_line_count_loss',
    'quality_issue_marker_loss',
    'quality_issue_repetition_loop',
    'quality_issue_adjacent_duplicate',
    'quality_issue_zh_tw_naturalization_needed',
  ]);
  const providerInfoCodes = new Set(['transient_retry_applied']);
  const providerWarningCodes = new Set(['provider_fallback_applied']);

  return warnings.map((code) => {
    if (qualityInfoCodes.has(code)) {
      return { code, severity: 'info', area: 'quality' } satisfies RunIssue;
    }
    if (qualityWarningCodes.has(code)) {
      return { code, severity: 'warning', area: 'quality' } satisfies RunIssue;
    }
    if (providerInfoCodes.has(code)) {
      return { code, severity: 'info', area: 'provider' } satisfies RunIssue;
    }
    if (providerWarningCodes.has(code)) {
      return { code, severity: 'warning', area: 'provider' } satisfies RunIssue;
    }
    switch (code) {
      case 'quality_issue_target_lang_mismatch':
      case 'quality_issue_pass_through':
      case 'quality_issue_empty_output':
      case 'quality_issue_line_count_loss':
      case 'quality_issue_marker_loss':
      case 'quality_issue_repetition_loop':
      case 'quality_issue_adjacent_duplicate':
      case 'quality_issue_zh_tw_naturalization_needed':
        return { code, severity: 'warning', area: 'quality' } satisfies RunIssue;
      default:
        return { code, severity: 'warning', area: 'provider' } satisfies RunIssue;
    }
  });
}

export interface RunLocalTranslationOrchestratorInput {
  input: {
    text: string;
    targetLang: string;
    sourceLang?: string;
    glossary?: string;
    effectiveGlossary?: string | null;
    prompt?: string;
    promptTemplateId?: string;
    enableJsonLineRepair?: boolean;
    signal?: AbortSignal;
  };
  localModel: LocalModelDefinition;
  modelStrategy: LocalTranslateModelStrategy;
  localProfile: LocalOpenvinoResolvedProfile;
  localTranslationProfile: LocalTranslationResolvedProfile;
  effectiveLocalProfile?: EffectiveLocalTranslationProfile;
  translationQualityMode: TranslationQualityMode;
  residualRetryLimit: number;
  jsonRepairMaxLinesBeforeSplit: number;
}

export interface RunLocalTranslationOrchestratorDeps {
  throwIfAborted(signal?: AbortSignal): void;
  buildLineSafeUnits(sourceText: string): LineSafeUnit[];
  buildSourceChunkText(units: LineSafeUnit[]): string;
  buildLineSafeInput(units: LineSafeUnit[]): string;
  splitLineSafeUnits(units: LineSafeUnit[]): LineSafeUnit[][];
  splitLineSafeUnitsForLocalTranslation(
    units: LineSafeUnit[],
    input?: {
      localModel?: LocalModelDefinition;
      modelStrategy?: LocalTranslateModelStrategy;
      targetLang?: string;
      sourceLang?: string;
      glossary?: string;
      prompt?: string;
      promptTemplateId?: string;
      translationQualityMode?: TranslationQualityMode;
      onBatchPlan?: (plan: LocalTranslationBatchDebugInfo) => void;
    }
  ): Promise<LineSafeUnit[][]>;
  splitLineSafeUnitsForTranslateGemma(
    units: LineSafeUnit[],
    input: {
      localModel: LocalModelDefinition;
      modelStrategy: LocalTranslateModelStrategy;
      targetLang: string;
      sourceLang?: string;
      glossary?: string;
      prompt?: string;
      promptTemplateId?: string;
      translationQualityMode?: TranslationQualityMode;
      onBatchPlan?: (plan: LocalTranslationBatchDebugInfo) => void;
    }
  ): Promise<LineSafeUnit[][]>;
  stripStructuredPrefix(line: string): string;
  stripInjectedSpeakerContext(line: string): string;
  parseLocalTranslatedText(raw: string, lineSafeMode: boolean): string;
  parseLineSafeOutput(output: string, units: LineSafeUnit[]): string | null;
  rebindByLineIndex(units: LineSafeUnit[], translatedText: string): string | null;
  normalizeTargetLanguageOutput(text: string, targetLang: string): string;
  getTranslationQualityIssues(
    sourceText: string,
    translatedText: string,
    targetLang: string,
    context?: TranslationQualityAssessmentContext
  ): TranslationQualityIssueCode[];
  addQualityIssueWarnings(issues: TranslationQualityIssueCode[], addWarning: (code: string) => void): void;
  buildStrictRetryInstruction(
    targetLang: string,
    issues: TranslationQualityIssueCode[],
    context?: TranslationQualityAssessmentContext
  ): string;
  buildTargetLanguageDescriptor(targetLang: string): string;
  estimateLocalMaxNewTokens(input: {
    text: string;
    lineCount: number;
    lineSafeMode: boolean;
    localModel?: LocalModelDefinition;
    jsonMode?: boolean;
    strictMode?: boolean;
  }): number;
  buildLocalTranslationPrompt(input: {
    text: string;
    targetLang: string;
    sourceLang?: string;
    glossary?: string;
    lineSafeMode: boolean;
    modelStrategy: LocalTranslateModelStrategy;
    promptStyle?: LocalTranslateModelStrategy['promptStyle'];
    strictMode?: boolean;
    translationQualityMode?: TranslationQualityMode;
    promptOverride?: string;
    disableSystemPrompt?: boolean;
    promptTemplateId?: string;
    sourceText?: string;
  }): string;
  buildLocalTranslationMessages(input: {
    text: string;
    targetLang: string;
    sourceLang?: string;
    modelStrategy: LocalTranslateModelStrategy;
    promptStyle?: LocalTranslateModelStrategy['promptStyle'];
  }): Array<Record<string, unknown>> | null;
  buildLocalGenerationOptions(input: {
    localModel?: LocalModelDefinition;
    modelStrategy: LocalTranslateModelStrategy;
    jsonMode?: boolean;
    targetLang?: string;
    strictMode?: boolean;
    translationQualityMode?: TranslationQualityMode;
  }): any;
  repairLineAlignmentWithLocalJsonMap(
    units: LineSafeUnit[],
    localModel: LocalModelDefinition,
    options: {
      targetLang: string;
      glossary?: string;
      promptTemplateId?: string;
      modelStrategy: LocalTranslateModelStrategy;
    },
    onProgress?: TranslateProgressFn,
    signal?: AbortSignal
  ): Promise<{ text: string; missingCount: number; warnings?: string[] } | null>;
  isLikelyPassThroughTranslation(sourceText: string, translatedText: string, targetLang: string): boolean;
  isPlainTranslationProbeMode(mode: TranslationQualityMode): boolean;
  getTranslateRuntimeDebug(modelId: string): TranslationDebugInfo['runtime'];
}

export async function runLocalTranslationOrchestrator(
  params: RunLocalTranslationOrchestratorInput,
  deps: RunLocalTranslationOrchestratorDeps,
  onProgress?: TranslateProgressFn
): Promise<TranslationResult> {
  const startedAt = Date.now();
  const {
    input,
    localModel,
    modelStrategy,
    localProfile,
    localTranslationProfile,
    effectiveLocalProfile,
    translationQualityMode,
    residualRetryLimit,
    jsonRepairMaxLinesBeforeSplit,
  } = params;
  deps.throwIfAborted(input.signal);

  const sourceText = String(input.text || '');
  const sourceLineCount = sourceText.split('\n').length;
  const lineSafeUnits = deps.buildLineSafeUnits(sourceText);
  const useLineSafeMode = sourceLineCount > 1;
  const normalizedTargetLang = String(input.targetLang || '').trim().toLowerCase();
  const forceZhTwNormalization =
    normalizedTargetLang === 'zh-tw' ||
    normalizedTargetLang === 'zh-hant' ||
    normalizedTargetLang === 'zh-hk' ||
    normalizedTargetLang === 'zh-mo';
  const sourceHasStructuredPrefixes = lineSafeUnits.some((unit) => Boolean(unit.prefix));
  const sourceSpeakerTaggedLineCount = lineSafeUnits.filter((unit) => Boolean(unit.speakerTag)).length;
  const customPrompt = String(input.prompt || '').trim();
  const enableJsonLineRepair = input.enableJsonLineRepair !== false;
  const localPlainProbeMode = deps.isPlainTranslationProbeMode(translationQualityMode);
  let localBatchingDebug: LocalTranslationBatchDebugInfo | null = null;
  const captureLocalBatchPlan = (plan: LocalTranslationBatchDebugInfo) => {
    localBatchingDebug = {
      ...plan,
      durationsMs: [],
      totalDurationMs: null,
      maxDurationMs: null,
    };
  };
  const recordLocalBatchDuration = (batchIndex: number, durationMs: number) => {
    if (!localBatchingDebug || batchIndex < 0) return;
    const durations = [...(localBatchingDebug.durationsMs || [])];
    durations[batchIndex] = Math.max(0, Math.round(durationMs));
    const finiteDurations = durations.filter((value) => typeof value === 'number' && Number.isFinite(value));
    localBatchingDebug = {
      ...localBatchingDebug,
      durationsMs: durations,
      totalDurationMs: finiteDurations.length > 0 ? finiteDurations.reduce((sum, value) => sum + value, 0) : null,
      maxDurationMs: finiteDurations.length > 0 ? Math.max(...finiteDurations) : null,
    };
  };

  const runLocalTranslation = async (
    providerInputText: string,
    lineSafeMode: boolean,
    strictMode = false,
    promptOverride = customPrompt
  ) => {
    deps.throwIfAborted(input.signal);
    const lineCount = lineSafeMode ? providerInputText.split('\n').length : 1;
    const maxNewTokens = deps.estimateLocalMaxNewTokens({
      text: providerInputText,
      lineCount,
      lineSafeMode,
      localModel,
      strictMode,
    });
    const prompt = deps.buildLocalTranslationPrompt({
      text: providerInputText,
      sourceText: providerInputText,
      targetLang: input.targetLang,
      sourceLang: input.sourceLang,
      glossary: input.glossary,
      promptTemplateId: input.promptTemplateId,
      lineSafeMode,
      modelStrategy,
      promptStyle: localTranslationProfile.effectivePromptStyle,
      strictMode,
      translationQualityMode,
      promptOverride,
      disableSystemPrompt: false,
    });
    const raw = await OpenvinoRuntimeManager.translateWithLocalModel({
      modelId: localModel.id,
      modelPath: getLocalModelInstallDir(localModel),
      prompt,
      messages: deps.buildLocalTranslationMessages({
        text: providerInputText,
        targetLang: input.targetLang,
        sourceLang: input.sourceLang,
        modelStrategy,
        promptStyle: localTranslationProfile.effectivePromptStyle,
      }),
      maxNewTokens,
      generation: deps.buildLocalGenerationOptions({
        localModel,
        modelStrategy,
        jsonMode: false,
        targetLang: input.targetLang,
        strictMode,
        translationQualityMode,
      }),
    });
    return deps.parseLocalTranslatedText(raw, lineSafeMode);
  };

  const warnings: string[] = [];
  const addWarning = (code: string) => {
    if (!warnings.includes(code)) warnings.push(code);
  };
  let qualityRetryCount = 0;
  let strictRetrySucceeded = false;

  const translateSingleLocalUnit = async (unit: LineSafeUnit[], issues: TranslationQualityIssueCode[] = []) => {
    const [singleUnit] = unit;
    if (!singleUnit) return null;

    const sourceLine = deps.stripStructuredPrefix(singleUnit.content);
    if (!sourceLine) {
      return singleUnit.prefix ? singleUnit.prefix.trimEnd() : '';
    }

    try {
      const singleLinePrompt = [
        customPrompt,
        deps.buildStrictRetryInstruction(input.targetLang, issues),
        `Translate this single subtitle line into ${deps.buildTargetLanguageDescriptor(input.targetLang)}.`,
        'Do not leave this line in the source language unless the entire line is only a proper noun, brand, or code identifier.',
        'Return exactly one translated line.',
      ]
        .filter(Boolean)
        .join('\n\n');
      const prompt = deps.buildLocalTranslationPrompt({
        text: sourceLine,
        sourceText: sourceLine,
        targetLang: input.targetLang,
        sourceLang: input.sourceLang,
        glossary: input.glossary,
        promptTemplateId: input.promptTemplateId,
        lineSafeMode: false,
        modelStrategy,
        promptStyle: localTranslationProfile.effectivePromptStyle,
        strictMode: true,
        translationQualityMode,
        promptOverride: singleLinePrompt,
        disableSystemPrompt: false,
      });
      const raw = await OpenvinoRuntimeManager.translateWithLocalModel({
        modelId: localModel.id,
        modelPath: getLocalModelInstallDir(localModel),
        prompt,
        messages: deps.buildLocalTranslationMessages({
          text: sourceLine,
          targetLang: input.targetLang,
          sourceLang: input.sourceLang,
          modelStrategy,
          promptStyle: localTranslationProfile.effectivePromptStyle,
        }),
        maxNewTokens: deps.estimateLocalMaxNewTokens({
          text: sourceLine,
          lineCount: 1,
          lineSafeMode: false,
          localModel,
          strictMode: true,
        }),
        generation: deps.buildLocalGenerationOptions({
          localModel,
          modelStrategy,
          jsonMode: false,
          targetLang: input.targetLang,
          strictMode: true,
          translationQualityMode,
        }),
      });
      const strictOutput = deps.parseLocalTranslatedText(raw, false);
      const cleaned = deps.normalizeTargetLanguageOutput(
        deps.stripStructuredPrefix(deps.stripInjectedSpeakerContext(strictOutput)),
        input.targetLang
      );
      const singleIssues = deps.getTranslationQualityIssues(sourceLine, cleaned, input.targetLang);
      deps.addQualityIssueWarnings(singleIssues, addWarning);
      if (singleIssues.length === 0) {
        return singleUnit.prefix ? `${singleUnit.prefix}${cleaned}` : cleaned;
      }
    } catch {
      // caller fallback
    }

    return null;
  };

  const translateTranslateGemmaUnitsIndividually = async (units: LineSafeUnit[]) => {
    const translatedRows: string[] = [];

    for (const unit of units) {
      deps.throwIfAborted(input.signal);
      const sourceLine = deps.stripStructuredPrefix(unit.content);
      if (!sourceLine) {
        translatedRows.push(unit.prefix ? unit.prefix.trimEnd() : '');
        continue;
      }

      if (localPlainProbeMode) {
        const raw = await runLocalTranslation(sourceLine, false, false, '');
        const cleaned = deps.normalizeTargetLanguageOutput(
          deps.stripStructuredPrefix(deps.stripInjectedSpeakerContext(raw)),
          input.targetLang
        );
        translatedRows.push(unit.prefix ? `${unit.prefix}${cleaned}` : cleaned);
        continue;
      }

      const strictCandidate = await translateSingleLocalUnit([unit]);
      if (strictCandidate !== null) {
        addWarning('local_single_line_retry_applied');
        translatedRows.push(strictCandidate);
        continue;
      }

      const raw = await runLocalTranslation(sourceLine, false, false);
      const cleaned = deps.normalizeTargetLanguageOutput(
        deps.stripStructuredPrefix(deps.stripInjectedSpeakerContext(raw)),
        input.targetLang
      );
      const fallbackIssues = deps.getTranslationQualityIssues(sourceLine, cleaned, input.targetLang);
      deps.addQualityIssueWarnings(fallbackIssues, addWarning);
      translatedRows.push(unit.prefix ? `${unit.prefix}${cleaned}` : cleaned);
    }

    return translatedRows.join('\n');
  };

  const parseTranslateGemmaSubtitleLines = (rawText: string) => {
    const rows = String(rawText || '')
      .replace(/\r/g, '')
      .split('\n')
      .map((line) => deps.stripInjectedSpeakerContext(String(line || '').trim()));

    while (rows.length > 0 && !rows[0]) rows.shift();
    while (rows.length > 0 && !rows[rows.length - 1]) rows.pop();

    return rows;
  };

  const reattachTranslateGemmaSubtitleLines = (units: LineSafeUnit[], translatedLines: string[]) =>
    units
      .map((unit, index) => {
        const translated = String(translatedLines[index] || '').trim();
        return unit.prefix ? `${unit.prefix}${translated}` : translated;
      })
      .join('\n');

  const hasTranslateGemmaCoverageLoss = (units: LineSafeUnit[], translatedLines: string[]) => {
    if (translatedLines.length !== units.length) return true;

    return units.some((unit, index) => {
      const sourceLine = deps.stripStructuredPrefix(unit.content).trim();
      const translatedLine = String(translatedLines[index] || '').trim();
      if (!sourceLine) return translatedLine.length > 0;
      return !translatedLine;
    });
  };

  const translateTranslateGemmaSubtitleBatch = async (units: LineSafeUnit[]): Promise<string> => {
    if (units.length <= 0) return '';
    if (units.length === 1) {
      return await translateTranslateGemmaUnitsIndividually(units);
    }

    const sourceLines = units.map((unit) => deps.stripStructuredPrefix(unit.content).trim());
    const sourceChunkText = sourceLines.join('\n');
    const normalized = deps.normalizeTargetLanguageOutput(
      await runLocalTranslation(sourceChunkText, false, false, ''),
      input.targetLang
    );
    const translatedLines = parseTranslateGemmaSubtitleLines(normalized);

    if (!hasTranslateGemmaCoverageLoss(units, translatedLines)) {
      return reattachTranslateGemmaSubtitleLines(units, translatedLines);
    }

    addWarning('translategemma_recursive_chunk_split_applied');
    const [leftUnits, rightUnits] = deps.splitLineSafeUnits(units);
    const leftText = leftUnits.length > 0 ? await translateTranslateGemmaSubtitleBatch(leftUnits) : '';
    const rightText = rightUnits.length > 0 ? await translateTranslateGemmaSubtitleBatch(rightUnits) : '';
    return [leftText, rightText].filter(Boolean).join('\n');
  };

  const stripLineSafeMarker = (text: string) =>
    String(text || '')
      .replace(/^\s*\[\[L\d{5}\]\]\s*/, '')
      .replace(/^\s*L\d{5}\s*[:：-]\s*/, '')
      .trim();

  const processPlainProbeLineSafeChunk = async (units: LineSafeUnit[]): Promise<string> => {
    const providerInputText = deps.buildLineSafeInput(units);
    const chunkOutput = await runLocalTranslation(providerInputText, true, false, customPrompt);

    const restored = deps.parseLineSafeOutput(chunkOutput, units);
    if (restored !== null) {
      addWarning('line_safe_alignment_applied');
      return restored;
    }

    const rebound = deps.rebindByLineIndex(units, chunkOutput);
    if (rebound !== null) {
      addWarning('line_index_rebind_applied');
      return rebound;
    }

    if (units.length > 1) {
      addWarning('local_recursive_chunk_split_applied');
      const [leftUnits, rightUnits] = deps.splitLineSafeUnits(units);
      const leftText = leftUnits.length > 0 ? await processPlainProbeLineSafeChunk(leftUnits) : '';
      const rightText = rightUnits.length > 0 ? await processPlainProbeLineSafeChunk(rightUnits) : '';
      return [leftText, rightText].filter(Boolean).join('\n');
    }

    const [unit] = units;
    const firstLine =
      String(chunkOutput || '')
        .split('\n')
        .map((line) => stripLineSafeMarker(deps.stripInjectedSpeakerContext(line)))
        .filter(Boolean)[0] || deps.stripStructuredPrefix(unit?.content || '');
    const cleaned = forceZhTwNormalization ? deps.normalizeTargetLanguageOutput(firstLine, input.targetLang) : firstLine;
    return unit?.prefix ? `${unit.prefix}${cleaned}` : cleaned;
  };

  const processLineSafeChunk = async (units: LineSafeUnit[]): Promise<string> => {
    const sourceChunkText = deps.buildSourceChunkText(units);
    const candidateQualityContext: TranslationQualityAssessmentContext = {
      expectedLineCount: units.length,
    };
    const rawGenerationQualityContext: TranslationQualityAssessmentContext = {
      expectedLineCount: units.length,
      requireMarkers: true,
    };
    const finalizeChunkCandidate = async (candidate: string) => {
      const normalizedCandidate = deps.normalizeTargetLanguageOutput(candidate, input.targetLang);
      const candidateIssues = deps.getTranslationQualityIssues(
        sourceChunkText,
        normalizedCandidate,
        input.targetLang,
        candidateQualityContext
      );
      deps.addQualityIssueWarnings(candidateIssues, addWarning);
      if (candidateIssues.length === 0) {
        return normalizedCandidate;
      }

      addWarning('post_repair_quality_retry_triggered');

      if (units.length > 1) {
        addWarning('local_recursive_chunk_split_applied');
        const [leftUnits, rightUnits] = deps.splitLineSafeUnits(units);
        const leftText = leftUnits.length > 0 ? await processLineSafeChunk(leftUnits) : '';
        const rightText = rightUnits.length > 0 ? await processLineSafeChunk(rightUnits) : '';
        return [leftText, rightText].filter(Boolean).join('\n');
      }

      const singleLine = await translateSingleLocalUnit(units, candidateIssues);
      if (singleLine !== null) {
        addWarning('local_single_line_retry_applied');
        return singleLine;
      }

      return normalizedCandidate;
    };

    const providerInputText = deps.buildLineSafeInput(units);
    let chunkOutput = await runLocalTranslation(providerInputText, true, false);

    const initialChunkIssues = deps.getTranslationQualityIssues(
      sourceChunkText,
      chunkOutput,
      input.targetLang,
      rawGenerationQualityContext
    );
    deps.addQualityIssueWarnings(initialChunkIssues, addWarning);
    if (initialChunkIssues.length > 0) {
      deps.throwIfAborted(input.signal);
      addWarning('quality_retry_triggered');
      qualityRetryCount += 1;
      onProgress?.('Detected untranslated, repetitive, or target-language-mismatched output, retrying with stricter prompt...');
      try {
        const strictPrompt = [
          customPrompt,
          deps.buildStrictRetryInstruction(input.targetLang, initialChunkIssues, rawGenerationQualityContext),
        ]
          .filter(Boolean)
          .join('\n\n');
        const strictOutput = await runLocalTranslation(providerInputText, true, true, strictPrompt);
        const strictIssues = strictOutput
          ? deps.getTranslationQualityIssues(sourceChunkText, strictOutput, input.targetLang, rawGenerationQualityContext)
          : initialChunkIssues;
        deps.addQualityIssueWarnings(strictIssues, addWarning);
        if (strictOutput && strictIssues.length === 0) {
          chunkOutput = strictOutput;
          addWarning('strict_retry_applied');
          strictRetrySucceeded = true;
        }
      } catch {
        // keep original
      }
    }

    const restored = deps.parseLineSafeOutput(chunkOutput, units);
    if (restored !== null) {
      addWarning('line_safe_alignment_applied');
      return finalizeChunkCandidate(restored);
    }

    const rebound = deps.rebindByLineIndex(units, chunkOutput);
    if (rebound !== null) {
      addWarning('line_index_rebind_applied');
      return finalizeChunkCandidate(rebound);
    }

    if (enableJsonLineRepair) {
      if (jsonRepairMaxLinesBeforeSplit > 0 && units.length > jsonRepairMaxLinesBeforeSplit) {
        addWarning('line_json_map_pre_split_applied');
        const [leftUnits, rightUnits] = deps.splitLineSafeUnits(units);
        const leftText = leftUnits.length > 0 ? await processLineSafeChunk(leftUnits) : '';
        const rightText = rightUnits.length > 0 ? await processLineSafeChunk(rightUnits) : '';
        return [leftText, rightText].filter(Boolean).join('\n');
      }
      const repaired = await deps.repairLineAlignmentWithLocalJsonMap(
        units,
        localModel,
        {
          targetLang: input.targetLang,
          glossary: input.glossary,
          promptTemplateId: input.promptTemplateId,
          modelStrategy,
        },
        onProgress,
        input.signal
      );
      if (repaired) {
        for (const warning of repaired.warnings || []) addWarning(warning);
        addWarning('line_json_map_repair_applied');
        if (repaired.missingCount > 0) addWarning('line_json_map_partial_fallback');
        return finalizeChunkCandidate(repaired.text);
      }
    } else {
      addWarning('line_json_map_repair_disabled');
    }

    addWarning('line_alignment_repair_failed');
    return finalizeChunkCandidate(
      units
        .map((unit) => {
          const fallback = deps.stripStructuredPrefix(unit.content);
          return unit.prefix ? `${unit.prefix}${fallback}` : fallback;
        })
        .join('\n')
    );
  };

  const repairResidualLocalLines = async (translatedText: string) => {
    const translatedLines = String(translatedText || '').split('\n');
    const suspiciousIndexes = lineSafeUnits
      .map((unit, index) => {
        const sourceLine = deps.stripStructuredPrefix(unit.content);
        const translatedLine = String(translatedLines[index] || '');
        const translatedBody = deps.stripStructuredPrefix(translatedLine);
        if (!sourceLine.trim()) return -1;
        if (!translatedBody.trim()) return index;
        return deps.isLikelyPassThroughTranslation(sourceLine, translatedBody, input.targetLang) ? index : -1;
      })
      .filter((index) => index >= 0);

    if (suspiciousIndexes.length === 0) return translatedText;

    if (residualRetryLimit <= 0) {
      return translatedText;
    }

    addWarning('residual_line_retry_triggered');
    let repairedAny = false;

    for (const index of suspiciousIndexes.slice(0, residualRetryLimit)) {
      const unit = lineSafeUnits[index];
      const current = deps.stripStructuredPrefix(unit.content);
      if (!current.trim()) continue;
      const prev = index > 0 ? deps.stripStructuredPrefix(lineSafeUnits[index - 1].content) : '';
      const next = index + 1 < lineSafeUnits.length ? deps.stripStructuredPrefix(lineSafeUnits[index + 1].content) : '';
      const prompt = deps.buildLocalTranslationPrompt({
        text: [prev ? `PREV: ${prev}` : '', `CURRENT: ${current}`, next ? `NEXT: ${next}` : ''].filter(Boolean).join('\n'),
        sourceText: current,
        targetLang: input.targetLang,
        sourceLang: input.sourceLang,
        glossary: input.glossary,
        promptTemplateId: input.promptTemplateId,
        lineSafeMode: false,
        modelStrategy,
        promptStyle: localTranslationProfile.effectivePromptStyle,
        strictMode: true,
        translationQualityMode,
        promptOverride: [
          customPrompt,
          deps.buildStrictRetryInstruction(input.targetLang, ['pass_through']),
          `Translate only the CURRENT subtitle line into ${deps.buildTargetLanguageDescriptor(input.targetLang)}.`,
          'PREV and NEXT lines are context only. Do not repeat or translate them.',
          'Return only the translated CURRENT line.',
        ]
          .filter(Boolean)
          .join('\n\n'),
        disableSystemPrompt: false,
      });

      try {
        const retryContext = [prev ? `PREV: ${prev}` : '', `CURRENT: ${current}`, next ? `NEXT: ${next}` : '']
          .filter(Boolean)
          .join('\n');
        const raw = await OpenvinoRuntimeManager.translateWithLocalModel({
          modelId: localModel.id,
          modelPath: getLocalModelInstallDir(localModel),
          prompt,
          messages: deps.buildLocalTranslationMessages({
            text: current,
            targetLang: input.targetLang,
            sourceLang: input.sourceLang,
            modelStrategy,
            promptStyle: localTranslationProfile.effectivePromptStyle,
          }),
          maxNewTokens: deps.estimateLocalMaxNewTokens({
            text: retryContext,
            lineCount: 3,
            lineSafeMode: false,
            localModel,
            strictMode: true,
          }),
          generation: deps.buildLocalGenerationOptions({
            localModel,
            modelStrategy,
            jsonMode: false,
            targetLang: input.targetLang,
            strictMode: true,
            translationQualityMode,
          }),
        });
        const parsedLines = deps
          .parseLocalTranslatedText(raw, false)
          .split('\n')
          .map((line) => line.trim())
          .filter(Boolean);
        const labeledCurrent = parsedLines.find((line) => /^current\s*:/i.test(line));
        const candidate = labeledCurrent
          ? labeledCurrent.replace(/^current\s*:\s*/i, '').trim()
          : parsedLines.find((line) => !/^(prev|next)\s*:/i.test(line)) || '';
        const repaired = deps.normalizeTargetLanguageOutput(
          candidate.replace(/^(translation|output)\s*:\s*/i, '').trim(),
          input.targetLang
        );
        const repairedIssues = deps.getTranslationQualityIssues(current, repaired, input.targetLang);
        deps.addQualityIssueWarnings(repairedIssues, addWarning);
        if (!repaired || repairedIssues.length > 0) continue;
        translatedLines[index] = unit.prefix ? `${unit.prefix}${repaired}` : repaired;
        repairedAny = true;
      } catch {
        // best effort
      }
    }

    if (repairedAny) addWarning('residual_line_retry_applied');
    return translatedLines.join('\n');
  };

  onProgress?.('Calling translation provider (openvino-local)...');
  let output = '';

  if (modelStrategy.family === 'translategemma') {
    if (useLineSafeMode && localPlainProbeMode) {
      const localBatches = await deps.splitLineSafeUnitsForTranslateGemma(lineSafeUnits, {
        localModel,
        modelStrategy,
        targetLang: input.targetLang,
        sourceLang: input.sourceLang,
        glossary: input.glossary,
        prompt: customPrompt,
        promptTemplateId: input.promptTemplateId,
        translationQualityMode,
        onBatchPlan: captureLocalBatchPlan,
      });
      if (localBatches.length > 1) {
        addWarning('translategemma_batch_translation_applied');
      }
      const merged: string[] = [];
      for (let batchIndex = 0; batchIndex < localBatches.length; batchIndex += 1) {
        onProgress?.(`Translating TranslateGemma subtitle batch (${batchIndex + 1}/${localBatches.length})...`);
        const batchStartedAt = Date.now();
        const chunk = await processPlainProbeLineSafeChunk(localBatches[batchIndex]);
        recordLocalBatchDuration(batchIndex, Date.now() - batchStartedAt);
        merged.push(...chunk.split('\n'));
      }
      output = merged.join('\n');
    } else if (useLineSafeMode) {
      const localBatches = await deps.splitLineSafeUnitsForTranslateGemma(lineSafeUnits, {
        localModel,
        modelStrategy,
        targetLang: input.targetLang,
        sourceLang: input.sourceLang,
        glossary: input.glossary,
        prompt: customPrompt,
        promptTemplateId: input.promptTemplateId,
        translationQualityMode,
        onBatchPlan: captureLocalBatchPlan,
      });
      if (localBatches.length > 1) {
        addWarning('translategemma_batch_translation_applied');
      }
      const merged: string[] = [];
      for (let batchIndex = 0; batchIndex < localBatches.length; batchIndex += 1) {
        onProgress?.(`Translating TranslateGemma subtitle batch (${batchIndex + 1}/${localBatches.length})...`);
        const batchStartedAt = Date.now();
        const chunk = await translateTranslateGemmaSubtitleBatch(localBatches[batchIndex]);
        recordLocalBatchDuration(batchIndex, Date.now() - batchStartedAt);
        merged.push(...chunk.split('\n'));
      }
      output = merged.join('\n');
    } else {
      output = deps.normalizeTargetLanguageOutput(await runLocalTranslation(sourceText, false, false, ''), input.targetLang);
      if (!localPlainProbeMode) {
        const wholeRequestIssues = deps
          .getTranslationQualityIssues(sourceText, output, input.targetLang)
          .filter((code) => code === 'empty_output' || code === 'target_lang_mismatch' || code === 'pass_through');
        deps.addQualityIssueWarnings(wholeRequestIssues, addWarning);
        if (wholeRequestIssues.length > 0) {
          addWarning('translategemma_single_line_retry_applied');
          output = await translateTranslateGemmaUnitsIndividually(lineSafeUnits);
        }
      }
    }
  } else if (localPlainProbeMode && useLineSafeMode) {
    const localBatches = await deps.splitLineSafeUnitsForLocalTranslation(lineSafeUnits, {
      localModel,
      modelStrategy,
      targetLang: input.targetLang,
      sourceLang: input.sourceLang,
      glossary: input.glossary,
      prompt: customPrompt,
      promptTemplateId: input.promptTemplateId,
      translationQualityMode,
      onBatchPlan: captureLocalBatchPlan,
    });
    if (localBatches.length > 1) {
      addWarning('local_batch_translation_applied');
    }
    const merged: string[] = [];
    for (let batchIndex = 0; batchIndex < localBatches.length; batchIndex += 1) {
      if (localBatches.length > 1) {
        onProgress?.(`Translating local subtitle batch (${batchIndex + 1}/${localBatches.length})...`);
      }
      const batchStartedAt = Date.now();
      const chunk = await processPlainProbeLineSafeChunk(localBatches[batchIndex]);
      recordLocalBatchDuration(batchIndex, Date.now() - batchStartedAt);
      merged.push(...chunk.split('\n'));
    }
    output = merged.join('\n');
  } else if (localPlainProbeMode) {
    output = await runLocalTranslation(sourceText, false, false);
  } else if (useLineSafeMode) {
    const localBatches = await deps.splitLineSafeUnitsForLocalTranslation(lineSafeUnits, {
      localModel,
      modelStrategy,
      targetLang: input.targetLang,
      sourceLang: input.sourceLang,
      glossary: input.glossary,
      prompt: customPrompt,
      promptTemplateId: input.promptTemplateId,
      translationQualityMode,
      onBatchPlan: captureLocalBatchPlan,
    });
    if (localBatches.length > 1) {
      addWarning('local_batch_translation_applied');
      const merged: string[] = [];
      for (let batchIndex = 0; batchIndex < localBatches.length; batchIndex += 1) {
        onProgress?.(`Translating local subtitle batch (${batchIndex + 1}/${localBatches.length})...`);
        const batchStartedAt = Date.now();
        const chunk = await processLineSafeChunk(localBatches[batchIndex]);
        recordLocalBatchDuration(batchIndex, Date.now() - batchStartedAt);
        merged.push(...chunk.split('\n'));
      }
      output = merged.join('\n');
    } else {
      const batchStartedAt = Date.now();
      output = await processLineSafeChunk(lineSafeUnits);
      recordLocalBatchDuration(0, Date.now() - batchStartedAt);
    }
  } else {
    output = await runLocalTranslation(sourceText, false, false);
    const wholeRequestIssues = deps.getTranslationQualityIssues(sourceText, output, input.targetLang);
    deps.addQualityIssueWarnings(wholeRequestIssues, addWarning);
    if (wholeRequestIssues.length > 0) {
      addWarning('quality_retry_triggered');
      qualityRetryCount += 1;
      onProgress?.('Detected untranslated, repetitive, or target-language-mismatched output, retrying with stricter prompt...');
      try {
        const strictPrompt = [customPrompt, deps.buildStrictRetryInstruction(input.targetLang, wholeRequestIssues)]
          .filter(Boolean)
          .join('\n\n');
        const strictOutput = await runLocalTranslation(sourceText, false, true, strictPrompt);
        const strictIssues = strictOutput
          ? deps.getTranslationQualityIssues(sourceText, strictOutput, input.targetLang)
          : wholeRequestIssues;
        deps.addQualityIssueWarnings(strictIssues, addWarning);
        if (strictOutput && strictIssues.length === 0) {
          output = strictOutput;
          addWarning('strict_retry_applied');
          strictRetrySucceeded = true;
        }
      } catch {
        // keep first output
      }
    }
  }

  if (!localPlainProbeMode || forceZhTwNormalization) {
    output = deps.normalizeTargetLanguageOutput(output, input.targetLang);
  }
  if (useLineSafeMode && !localPlainProbeMode && modelStrategy.family !== 'translategemma') {
    output = await repairResidualLocalLines(output);
  }

  onProgress?.('Translation completed.');

  const runtimeDebug = deps.getTranslateRuntimeDebug(localModel.id);
  const elapsedMs = Date.now() - startedAt;
  const errors = {
    request: null,
  };
  const finalQualityIssues = localPlainProbeMode
    ? []
    : deps.getTranslationQualityIssues(sourceText, output, input.targetLang, {
        expectedLineCount: sourceLineCount,
      });
  const finalQualityWarnings = finalQualityIssues
    .map((issue) => qualityIssueToWarningCode(issue))
    .filter(Boolean) as string[];

  return {
    translatedText: output,
    debug: {
      requested: {
        sourceLang: input.sourceLang ? String(input.sourceLang) : undefined,
        sourceLanguageDescriptor: input.sourceLang ? deps.buildTargetLanguageDescriptor(input.sourceLang) : null,
        targetLang: input.targetLang,
        targetLanguageDescriptor: deps.buildTargetLanguageDescriptor(input.targetLang),
        lineCount: sourceLineCount,
        charCount: sourceText.length,
        hasGlossary: Boolean(input.glossary && input.glossary.trim()),
        effectiveGlossary: input.effectiveGlossary ?? null,
        hasPrompt: Boolean(customPrompt),
        promptTemplateId: input.promptTemplateId ? String(input.promptTemplateId) : null,
        jsonLineRepairEnabled: enableJsonLineRepair,
        sourceHasStructuredPrefixes,
        sourceHasSpeakerTags: sourceSpeakerTaggedLineCount > 0,
        sourceSpeakerTaggedLineCount,
      },
      provider: {
        name: 'openvino-local',
        modelId: localModel.id,
        model: localModel.repoId,
        endpoint: 'local://openvino/translate',
        adapterKey: null,
        profileId: localProfile.profileId,
        profileFamily: localProfile.profileFamily,
      },
      runtime: runtimeDebug,
      applied: {
        retryCount: 0,
        fallback: false,
        fallbackType: null,
        translationQualityMode,
        qualityRetryCount,
        strictRetrySucceeded,
        localModelFamily: modelStrategy.family,
        localModelProfileId: localTranslationProfile.modelProfile?.id || null,
        localPromptStyle: localTranslationProfile.effectivePromptStyle,
        localGenerationStyle: modelStrategy.generationStyle,
        localBypassChecks: localPlainProbeMode,
        localPromptContract: effectiveLocalProfile?.intent.promptContract,
        localAlignmentContract: effectiveLocalProfile?.intent.alignmentContract,
        localRuntimeHintsApplied: effectiveLocalProfile?.runtimeHintsApplied,
        localBaselineConfidence: localProfile.baseline.baselineConfidence,
        localBaselineTaskFamily: localProfile.baseline.taskFamily,
        localFallbackBaseline: localProfile.usedFallbackBaseline,
        localBatching: localBatchingDebug,
        localBatchingMode: localBatchingDebug?.mode ?? null,
        localBatchCount: localBatchingDebug?.batchCount ?? null,
        localBatchLineCounts: localBatchingDebug?.lineCounts ?? [],
        localBatchPromptTokens: localBatchingDebug?.promptTokens ?? [],
      },
      quality: {
        lineCountMatch: sourceLineCount <= 1 ? true : output.split('\n').length >= sourceLineCount,
        targetLanguageMatch: !finalQualityWarnings.includes('quality_issue_target_lang_mismatch'),
        passThroughRisk: getPassThroughRisk(finalQualityWarnings),
        repetitionRisk: getRepetitionRisk(finalQualityWarnings),
        markerPreservation: getMarkerPreservation(sourceHasStructuredPrefixes, finalQualityWarnings),
        strictRetryTriggered: qualityRetryCount > 0,
      },
      stats: {
        outputLineCount: output.split('\n').length,
        outputCharCount: output.length,
      },
      timing: {
        elapsedMs,
        elapsedSec: Number((elapsedMs / 1000).toFixed(3)),
        providerMs:
          typeof runtimeDebug?.lastPerfMetrics?.generateDurationMs === 'number'
            ? runtimeDebug.lastPerfMetrics.generateDurationMs
            : null,
        providerSec:
          typeof runtimeDebug?.lastPerfMetrics?.generateDurationMs === 'number'
            ? Number((runtimeDebug.lastPerfMetrics.generateDurationMs / 1000).toFixed(3))
            : null,
        repairMs: null,
        repairSec: null,
        qualityRetryMs: null,
        qualityRetrySec: null,
      },
      diagnostics: {
        qualityIssueCodes: warnings.filter((warning) => warning.startsWith('quality_issue_')),
        runtimeSource: 'local',
      },
      warnings,
      warningIssues: buildTranslationWarningIssues(warnings),
      errors,
      errorIssues: [],
      artifacts: {
        hasTimecodes: false,
      },
    },
  };
}
