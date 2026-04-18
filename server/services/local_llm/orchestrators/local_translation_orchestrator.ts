import { getLocalModelInstallDir, type LocalModelDefinition } from '../../../local_model_catalog.js';
import { OpenvinoRuntimeManager } from '../../../openvino_runtime_manager.js';
import type { TranslationDebugInfo, TranslationResult } from '../../translation_service.js';
import type { LocalOpenvinoResolvedProfile, LocalTranslateModelStrategy } from '../types.js';
import type { LocalTranslationResolvedProfile } from '../model_profiles.js';
import type { TranslationQualityMode } from '../../llm/orchestrators/translation_quality_policy.js';

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

export interface RunLocalTranslationOrchestratorInput {
  input: {
    text: string;
    targetLang: string;
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
      translationQualityMode?: TranslationQualityMode;
    }
  ): LineSafeUnit[][];
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
    jsonMode?: boolean;
    strictMode?: boolean;
  }): number;
  buildLocalTranslationPrompt(input: {
    text: string;
    targetLang: string;
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
  const {
    input,
    localModel,
    modelStrategy,
    localProfile,
    localTranslationProfile,
    translationQualityMode,
    residualRetryLimit,
    jsonRepairMaxLinesBeforeSplit,
  } = params;
  deps.throwIfAborted(input.signal);

  const sourceText = String(input.text || '');
  const sourceLineCount = sourceText.split('\n').length;
  const lineSafeUnits = deps.buildLineSafeUnits(sourceText);
  const useLineSafeMode = sourceLineCount > 1;
  const sourceHasStructuredPrefixes = lineSafeUnits.some((unit) => Boolean(unit.prefix));
  const sourceSpeakerTaggedLineCount = lineSafeUnits.filter((unit) => Boolean(unit.speakerTag)).length;
  const customPrompt = String(input.prompt || '').trim();
  const enableJsonLineRepair = input.enableJsonLineRepair !== false;
  const localPlainProbeMode = deps.isPlainTranslationProbeMode(translationQualityMode);

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
      strictMode,
    });
    const prompt = deps.buildLocalTranslationPrompt({
      text: providerInputText,
      sourceText: providerInputText,
      targetLang: input.targetLang,
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
        maxNewTokens: deps.estimateLocalMaxNewTokens({
          text: sourceLine,
          lineCount: 1,
          lineSafeMode: false,
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
          maxNewTokens: deps.estimateLocalMaxNewTokens({
            text: retryContext,
            lineCount: 3,
            lineSafeMode: false,
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

  if (localPlainProbeMode) {
    output = await runLocalTranslation(sourceText, false, false);
  } else if (useLineSafeMode) {
    const localBatches = deps.splitLineSafeUnitsForLocalTranslation(lineSafeUnits, {
      localModel,
      modelStrategy,
      translationQualityMode,
    });
    if (localBatches.length > 1) {
      addWarning('local_batch_translation_applied');
      const merged: string[] = [];
      for (let batchIndex = 0; batchIndex < localBatches.length; batchIndex += 1) {
        onProgress?.(`Translating local subtitle batch (${batchIndex + 1}/${localBatches.length})...`);
        const chunk = await processLineSafeChunk(localBatches[batchIndex]);
        merged.push(...chunk.split('\n'));
      }
      output = merged.join('\n');
    } else {
      output = await processLineSafeChunk(lineSafeUnits);
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

  if (!localPlainProbeMode) {
    output = deps.normalizeTargetLanguageOutput(output, input.targetLang);
  }
  if (useLineSafeMode && !localPlainProbeMode) {
    output = await repairResidualLocalLines(output);
  }

  onProgress?.('Translation completed.');

  return {
    translatedText: output,
    debug: {
      requested: {
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
        model: localModel.repoId,
        endpoint: 'local://openvino/translate',
        adapterKey: null,
        profileId: localProfile.profileId,
        profileFamily: localProfile.profileFamily,
      },
      runtime: deps.getTranslateRuntimeDebug(localModel.id),
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
        localBaselineConfidence: localProfile.baseline.baselineConfidence,
        localBaselineTaskFamily: localProfile.baseline.taskFamily,
        localFallbackBaseline: localProfile.usedFallbackBaseline,
      },
      stats: {
        outputLineCount: output.split('\n').length,
        outputCharCount: output.length,
      },
      warnings,
      errors: {
        request: null,
      },
    },
  };
}
