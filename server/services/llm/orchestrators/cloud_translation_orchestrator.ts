import type { CloudTranslateProvider } from '../../cloud_translate_provider.js';
import type { CloudTranslateBatchingSource } from '../../cloud_translate/types.js';
import type { ApiModelRequestOptions } from '../../../../src/types.js';
import {
  usesJsonStrictAlignment,
  usesTemplateValidatedQualityChecks,
  type TranslationQualityMode,
} from './translation_quality_policy.js';

export type CloudTranslationStrategy = 'plain' | 'line_locked' | 'cloud_strict';

export type CloudTranslationBatchSource = CloudTranslateBatchingSource;
export type CloudTranslationProviderBatchMode = 'line_safe' | 'plain_ordered';

export interface CloudTranslationLineSafeUnit {
  index: number;
  marker: string;
  prefix: string;
  content: string;
  speakerTag: string | null;
}

export interface CloudTranslationJsonRepairResult {
  text: string;
  missingCount: number;
  warnings?: string[];
}

export interface CloudTranslationContextChunk {
  start: number;
  length: number;
}

export interface CloudTranslationContextConfig {
  enabled: boolean;
  targetLines: number;
  minTargetLines: number;
  contextWindow: number;
  charBudget: number;
  maxSplitDepth: number;
}

export interface CloudTranslationProviderBatchConfig {
  enabled: boolean;
  source: CloudTranslationBatchSource;
  targetLines: number;
  minTargetLines: number;
  charBudget: number;
  maxSplitDepth: number;
  maxOutputTokens: number;
  timeoutMs: number;
  stream: boolean;
}

export interface CloudTranslationBatchDebugInfo {
  source: CloudTranslationBatchSource;
  mode: CloudTranslationProviderBatchMode;
  batchCount: number;
  lineCounts: number[];
  charCounts: number[];
  estimatedOutputTokens: number[];
  durationsMs: number[];
  maxLines: number;
  minLines: number;
  charBudget: number;
  maxOutputTokens: number;
  timeoutMs: number;
  stream: boolean;
  splitCount: number;
  totalDurationMs: number;
  maxDurationMs: number;
}

export interface CloudTranslationQuotaDebugInfo {
  applied: boolean;
  profileId: string | null;
  tokenEstimator: string | null;
  estimatedInputTokens: number | null;
  estimatedTotalTokens: number | null;
  waitedMs: number;
  waitReason: string | null;
  waitEvents: Array<{
    reason: string;
    waitedMs: number;
  }>;
}

export interface CloudTranslationProviderResult {
  text: string;
  meta: {
    endpointUrl: string;
    fallbackUsed: boolean;
    fallbackType: string | null;
    requestWarnings?: string[];
    quota?: CloudTranslationQuotaDebugInfo | null;
    responseHeaders?: Record<string, string>;
  };
}

export interface CloudTranslationProviderRequest {
  provider: CloudTranslateProvider;
  endpointUrl: string;
  key?: string;
  model?: string;
  modelOptions?: ApiModelRequestOptions;
}

export interface CloudTranslationOrchestratorInput {
  text: string;
  targetLang: string;
  sourceLang?: string;
  glossary?: string;
  prompt?: string;
  promptTemplateId?: string;
  enableJsonLineRepair: boolean;
  qualityMode: TranslationQualityMode;
  supportsContextMode: boolean;
  isConnectionTest?: boolean;
  providerRequest: CloudTranslationProviderRequest;
  providerBatching?: CloudTranslationProviderBatchConfig | null;
  initialWarnings?: string[];
  signal?: AbortSignal;
}

export interface CloudTranslationOrchestratorDeps {
  getCloudContextConfig(): CloudTranslationContextConfig;
  getRemoteBatchSize(): number;
  getRetryConfig(): {
    maxRetries: number;
    baseRetryMs: number;
    rateLimitRetryMs: number;
  };
  throwIfAborted(signal?: AbortSignal): void;
  sleep(ms: number, signal?: AbortSignal): Promise<void>;
  mapConnectionError(error: unknown): string;
  isRetryableError(error: unknown): boolean;
  isProviderHttpError(error: unknown): error is { status: number; retryAfterMs: number | null };
  isProviderContentFilterError(error: unknown): boolean;
  requestTranslationByProvider(
    provider: CloudTranslateProvider,
    endpointUrl: string,
    options: {
      text: string;
      targetLang: string;
      sourceLang?: string;
      glossary?: string;
      prompt?: string;
      promptTemplateId?: string;
      key?: string;
      model?: string;
      modelOptions?: ApiModelRequestOptions;
      isConnectionTest?: boolean;
      lineSafeMode?: boolean;
      systemPromptOverride?: string;
      disableSystemPrompt?: boolean;
      signal?: AbortSignal;
    },
    onProgress?: (message: string) => void
  ): Promise<CloudTranslationProviderResult>;
  buildLineSafeUnits(text: string): CloudTranslationLineSafeUnit[];
  stripStructuredPrefix(value: string): string;
  buildLineSafeInput(units: CloudTranslationLineSafeUnit[]): string;
  parseLineSafeOutput(output: string, units: CloudTranslationLineSafeUnit[]): string | null;
  normalizeTargetLanguageOutput(output: string, targetLang: string): string;
  repairLineAlignmentWithJsonMap(
    provider: CloudTranslateProvider,
    endpointUrl: string,
    units: CloudTranslationLineSafeUnit[],
    options: {
      targetLang: string;
      sourceLang?: string;
      glossary?: string;
      promptTemplateId?: string;
      key?: string;
      model?: string;
      modelOptions?: ApiModelRequestOptions;
    },
    onProgress?: (message: string) => void,
    signal?: AbortSignal
  ): Promise<CloudTranslationJsonRepairResult | null>;
  rebindByLineIndex(units: CloudTranslationLineSafeUnit[], translatedText: string): string | null;
  buildCloudContextInput(
    units: CloudTranslationLineSafeUnit[],
    start: number,
    length: number,
    contextWindow: number
  ): {
    text: string;
    targetUnits: CloudTranslationLineSafeUnit[];
  };
  buildCloudContextSystemPrompt(input: {
    targetLang: string;
    sourceLang?: string;
    glossary?: string;
    promptTemplateId?: string;
    prompt?: string;
  }): string;
  parseCloudContextOutput(raw: string, targetUnits: CloudTranslationLineSafeUnit[]): string | null;
  buildCloudContextChunks(
    units: CloudTranslationLineSafeUnit[],
    targetLines: number,
    charBudget: number
  ): CloudTranslationContextChunk[];
}

export interface CloudTranslationOrchestratorResult {
  output: string;
  warnings: string[];
  retryCount: number;
  lastError: string | null;
  fallbackUsed: boolean;
  fallbackType: string | null;
  endpointUsed: string;
  cloudStrategy: CloudTranslationStrategy;
  cloudContextChunkCount: number;
  cloudContextFallbackCount: number;
  cloudBatching: CloudTranslationBatchDebugInfo | null;
  cloudQuota: CloudTranslationQuotaDebugInfo | null;
  cloudStrategyReason: string;
  sourceLineCount: number;
  sourceHasStructuredPrefixes: boolean;
  sourceSpeakerTaggedLineCount: number;
  relaxedWholeRequestApplied: boolean;
  relaxedWholeRequestFallback: boolean;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function stripHeader(headers: Record<string, string>, target: string) {
  const targetLower = target.toLowerCase();
  for (const key of Object.keys(headers)) {
    if (key.toLowerCase() === targetLower) {
      delete headers[key];
    }
  }
}

function buildProviderBatchModelOptions(
  base: ApiModelRequestOptions | undefined,
  config: CloudTranslationProviderBatchConfig,
  estimatedOutputTokens: number
): ApiModelRequestOptions {
  const sampling = {
    ...(isPlainObject(base?.sampling) ? base?.sampling : {}),
    maxOutputTokens: Math.max(1, Math.min(config.maxOutputTokens, estimatedOutputTokens)),
  };
  const body = {
    ...(isPlainObject(base?.body) ? base?.body : {}),
    stream: config.stream,
  };
  const headers: Record<string, string> = {
    ...(isPlainObject(base?.headers) ? (base?.headers as Record<string, string>) : {}),
  };

  if (config.stream) {
    headers.Accept = 'text/event-stream';
  } else {
    stripHeader(headers, 'accept');
  }

  return {
    ...(base || {}),
    sampling,
    body,
    headers,
    timeoutMs: Math.max(Number(base?.timeoutMs) || 0, config.timeoutMs),
  };
}

function estimateCloudOutputTokens(text: string, lineCount: number, maxOutputTokens: number) {
  const value = String(text || '');
  const cjkChars = (value.match(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu) || [])
    .length;
  const latinLikeChars = Math.max(0, value.replace(/\s+/g, ' ').length - cjkChars);
  const sourceUnits = Math.ceil(cjkChars + latinLikeChars / 4);
  const estimated = Math.ceil(sourceUnits * 1.35) + lineCount * 10;
  return Math.max(256, Math.min(maxOutputTokens, estimated));
}

function getUnitSourceChars(unit: CloudTranslationLineSafeUnit) {
  return String(unit.content || '').trim().length;
}

function buildProviderBatchChunks(
  units: CloudTranslationLineSafeUnit[],
  targetLines: number,
  charBudget: number
): CloudTranslationLineSafeUnit[][] {
  const chunks: CloudTranslationLineSafeUnit[][] = [];
  let offset = 0;

  while (offset < units.length) {
    const chunk: CloudTranslationLineSafeUnit[] = [];
    let usedChars = 0;

    while (offset + chunk.length < units.length && chunk.length < targetLines) {
      const nextUnit = units[offset + chunk.length];
      const nextChars = getUnitSourceChars(nextUnit);
      if (chunk.length > 0 && usedChars + nextChars > charBudget) break;
      chunk.push(nextUnit);
      usedChars += nextChars;
    }

    if (chunk.length === 0) {
      chunk.push(units[offset]);
    }
    chunks.push(chunk);
    offset += chunk.length;
  }

  return chunks;
}

function stripLineSafeMarkers(text: string) {
  return String(text || '').replace(/\[\[L\d{5}\]\]\s*/g, '').trim();
}

function mergeQuotaDebug(
  current: CloudTranslationQuotaDebugInfo | null,
  next: CloudTranslationQuotaDebugInfo | null | undefined
) {
  if (!next) return current;
  if (!current) {
    return {
      ...next,
      waitEvents: [...(next.waitEvents || [])],
    } satisfies CloudTranslationQuotaDebugInfo;
  }
  return {
    applied: current.applied || next.applied,
    profileId: current.profileId || next.profileId,
    tokenEstimator: current.tokenEstimator || next.tokenEstimator,
    estimatedInputTokens:
      (current.estimatedInputTokens ?? 0) + (next.estimatedInputTokens ?? 0),
    estimatedTotalTokens:
      (current.estimatedTotalTokens ?? 0) + (next.estimatedTotalTokens ?? 0),
    waitedMs: current.waitedMs + next.waitedMs,
    waitReason: next.waitReason || current.waitReason,
    waitEvents: [...(current.waitEvents || []), ...(next.waitEvents || [])],
  } satisfies CloudTranslationQuotaDebugInfo;
}

function startProviderProgressPulse(input: {
  provider: CloudTranslateProvider;
  signal?: AbortSignal;
  onProgress?: (message: string) => void;
}) {
  if (!input.onProgress) return null;
  const startedAt = Date.now();
  return setInterval(() => {
    if (input.signal?.aborted) return;
    const elapsedSec = Math.max(1, Math.round((Date.now() - startedAt) / 1000));
    input.onProgress?.(`Waiting for translation provider (${input.provider}) response (${elapsedSec}s)...`);
  }, 15_000);
}

export async function runCloudTranslationOrchestrator(
  input: CloudTranslationOrchestratorInput,
  deps: CloudTranslationOrchestratorDeps,
  onProgress?: (message: string) => void
): Promise<CloudTranslationOrchestratorResult> {
  deps.throwIfAborted(input.signal);

  const lineSafeUnits = deps.buildLineSafeUnits(input.text);
  const sourceLineCount = String(input.text).split('\n').length;
  const sourceHasStructuredPrefixes = lineSafeUnits.some((unit) => Boolean(unit.prefix));
  const sourceSpeakerTaggedLineCount = lineSafeUnits.filter((unit) => Boolean(unit.speakerTag)).length;
  const hasMultipleSourceLines = sourceLineCount > 1;
  const useCloudStrictAlignment = input.enableJsonLineRepair && usesJsonStrictAlignment(input.qualityMode);
  const useCloudLineLocked = !useCloudStrictAlignment && usesTemplateValidatedQualityChecks(input.qualityMode);
  const cloudContextConfig = deps.getCloudContextConfig();
  const useCloudContextMode =
    hasMultipleSourceLines &&
    useCloudLineLocked &&
    cloudContextConfig.enabled &&
    input.supportsContextMode;
  const providerBatching =
    input.providerBatching?.enabled && !input.isConnectionTest ? input.providerBatching : null;
  const providerBatchTotalChars = lineSafeUnits.reduce((sum, unit) => sum + getUnitSourceChars(unit), 0);
  const useProviderBatchMode = Boolean(
    providerBatching &&
      !useCloudStrictAlignment &&
      !useCloudContextMode &&
      hasMultipleSourceLines &&
      (lineSafeUnits.length > providerBatching.targetLines || providerBatchTotalChars > providerBatching.charBudget)
  );
  const cloudStrategyReason = useCloudStrictAlignment
    ? 'json_line_repair_enabled'
    : useCloudContextMode
      ? 'template_or_custom_prompt_with_context_profile'
      : useProviderBatchMode
        ? 'provider_batch_profile_threshold'
        : useCloudLineLocked
          ? 'template_or_custom_prompt'
          : 'plain_cloud_request';

  const { maxRetries, baseRetryMs, rateLimitRetryMs } = deps.getRetryConfig();

  const warnings: string[] = [...(input.initialWarnings || [])];
  const addWarning = (code: string) => {
    if (!warnings.includes(code)) warnings.push(code);
  };

  let retryCount = 0;
  let lastError: string | null = null;
  let output = '';
  let fallbackUsed = false;
  let fallbackType: string | null = null;
  let endpointUsed = input.providerRequest.endpointUrl;
  let cloudStrategy: CloudTranslationStrategy = useCloudStrictAlignment
    ? 'cloud_strict'
    : useCloudLineLocked
      ? 'line_locked'
      : 'plain';
  let cloudContextChunkCount = 0;
  let cloudContextFallbackCount = 0;
  let cloudBatching: CloudTranslationBatchDebugInfo | null = null;
  let cloudQuota: CloudTranslationQuotaDebugInfo | null = null;

  const requestCloudTranslation = async (requestOptions: {
    text: string;
    lineSafeMode: boolean;
    promptOverride?: string;
    progressMessage?: string;
    disableSystemPrompt?: boolean;
    modelOptionsOverride?: ApiModelRequestOptions;
    retrySameRequest?: boolean;
  }) => {
    let attempt = 0;
    while (true) {
      deps.throwIfAborted(input.signal);
      try {
        onProgress?.(requestOptions.progressMessage || `Calling translation provider (${input.providerRequest.provider})...`);
        const progressPulse = startProviderProgressPulse({
          provider: input.providerRequest.provider,
          signal: input.signal,
          onProgress,
        });
        const result = await deps.requestTranslationByProvider(
          input.providerRequest.provider,
          endpointUsed,
          {
            text: requestOptions.text,
            targetLang: input.targetLang,
            sourceLang: input.sourceLang,
            glossary: input.glossary,
            prompt: requestOptions.promptOverride ?? input.prompt,
            promptTemplateId: input.promptTemplateId,
            key: input.providerRequest.key,
            model: input.providerRequest.model,
            modelOptions: requestOptions.modelOptionsOverride ?? input.providerRequest.modelOptions,
            lineSafeMode: requestOptions.lineSafeMode,
            isConnectionTest: input.isConnectionTest,
            systemPromptOverride: String(requestOptions.promptOverride ?? input.prompt ?? '').trim(),
            disableSystemPrompt: requestOptions.disableSystemPrompt === true,
            signal: input.signal,
          },
          onProgress
        ).finally(() => {
          if (progressPulse) clearInterval(progressPulse);
        });
        for (const warning of result.meta.requestWarnings || []) {
          addWarning(warning);
        }
        cloudQuota = mergeQuotaDebug(cloudQuota, result.meta.quota);
        if (result.meta.fallbackUsed) {
          fallbackUsed = true;
          fallbackType = result.meta.fallbackType;
          addWarning('provider_fallback_applied');
        }
        endpointUsed = result.meta.endpointUrl;
        lastError = null;
        return result.text;
      } catch (error) {
        lastError = deps.mapConnectionError(error);
        const canRetrySameRequest =
          requestOptions.retrySameRequest !== false ||
          (deps.isProviderHttpError(error) && error.status === 429);
        if (!canRetrySameRequest || !deps.isRetryableError(error) || attempt >= maxRetries) {
          throw error;
        }
        attempt += 1;
        retryCount += 1;
        addWarning('transient_retry_applied');
        const retryDelay =
          deps.isProviderHttpError(error) && error.status === 429
            ? Math.max(error.retryAfterMs || 0, rateLimitRetryMs * attempt)
            : baseRetryMs * attempt;
        const status = deps.isProviderHttpError(error) ? error.status : 'network';
        onProgress?.(`Retrying translation request (${attempt}/${maxRetries}) after transient error (${status})...`);
        await deps.sleep(retryDelay, input.signal);
      }
    }
  };

  const translateSingleCloudUnit = async (unit: CloudTranslationLineSafeUnit) => {
    cloudContextFallbackCount += 1;
    addWarning('cloud_context_single_line_fallback');
    let raw = '';
    try {
      raw = await requestCloudTranslation({
        text: deps.stripStructuredPrefix(unit.content),
        lineSafeMode: false,
        progressMessage: `Retrying cloud translation as single line (${unit.index})...`,
      });
    } catch (error) {
      if (!deps.isProviderContentFilterError(error)) {
        throw error;
      }
      addWarning('line_json_map_policy_source_fallback');
      const fallback = deps.stripStructuredPrefix(unit.content);
      return unit.prefix ? `${unit.prefix}${fallback}` : fallback;
    }
    const normalized = deps.normalizeTargetLanguageOutput(raw, input.targetLang).replace(/\s*\n+\s*/g, ' ').trim();
    return unit.prefix ? `${unit.prefix}${normalized}` : normalized;
  };

  const processCloudLineSafeChunk = async (units: CloudTranslationLineSafeUnit[]): Promise<string> => {
    const providerInputText = deps.buildLineSafeInput(units);
    let chunkOutput = '';
    try {
      chunkOutput = await requestCloudTranslation({
        text: providerInputText,
        lineSafeMode: true,
        progressMessage: `Calling translation provider (${input.providerRequest.provider})...`,
      });
    } catch (error) {
      if (!deps.isProviderContentFilterError(error)) {
        throw error;
      }

      if (input.enableJsonLineRepair) {
        const repaired = await deps.repairLineAlignmentWithJsonMap(
          input.providerRequest.provider,
          endpointUsed,
          units,
          {
            targetLang: input.targetLang,
            sourceLang: input.sourceLang,
            glossary: input.glossary,
            promptTemplateId: input.promptTemplateId,
            key: input.providerRequest.key,
            model: input.providerRequest.model,
            modelOptions: input.providerRequest.modelOptions,
          },
          onProgress,
          input.signal
        );

        if (repaired) {
          for (const warning of repaired.warnings || []) {
            addWarning(warning);
          }
          addWarning('line_json_map_repair_applied');
          if (repaired.missingCount > 0) {
            addWarning('line_json_map_partial_fallback');
          }
          return deps.normalizeTargetLanguageOutput(repaired.text, input.targetLang);
        }
      }

      if (units.length > 1) {
        const midpoint = Math.ceil(units.length / 2);
        const left = await processCloudLineSafeChunk(units.slice(0, midpoint));
        const right = await processCloudLineSafeChunk(units.slice(midpoint));
        return [left, right].filter(Boolean).join('\n');
      }

      return translateSingleCloudUnit(units[0]);
    }
    chunkOutput = deps.normalizeTargetLanguageOutput(chunkOutput, input.targetLang);

    const restored = deps.parseLineSafeOutput(chunkOutput, units);
    if (restored !== null) {
      addWarning('line_safe_alignment_applied');
      return deps.normalizeTargetLanguageOutput(restored, input.targetLang);
    }

    if (input.enableJsonLineRepair) {
      const repaired = await deps.repairLineAlignmentWithJsonMap(
        input.providerRequest.provider,
        endpointUsed,
        units,
        {
          targetLang: input.targetLang,
          sourceLang: input.sourceLang,
          glossary: input.glossary,
            promptTemplateId: input.promptTemplateId,
            key: input.providerRequest.key,
            model: input.providerRequest.model,
            modelOptions: input.providerRequest.modelOptions,
          },
          onProgress,
          input.signal
      );

      if (repaired) {
        for (const warning of repaired.warnings || []) {
          addWarning(warning);
        }
        addWarning('line_json_map_repair_applied');
        if (repaired.missingCount > 0) {
          addWarning('line_json_map_partial_fallback');
        }
        return deps.normalizeTargetLanguageOutput(repaired.text, input.targetLang);
      }
    } else {
      addWarning('line_json_map_repair_disabled');
    }

    const rebound = deps.rebindByLineIndex(units, chunkOutput);
    if (rebound !== null) {
      addWarning('line_index_rebind_applied');
      return deps.normalizeTargetLanguageOutput(rebound, input.targetLang);
    }

    addWarning('line_alignment_repair_failed');
    return stripLineSafeMarkers(chunkOutput);
  };

  const runCloudLineSafeFlow = async () => {
    const remoteBatchSize = deps.getRemoteBatchSize();
    if (lineSafeUnits.length > remoteBatchSize) {
      const totalBatches = Math.ceil(lineSafeUnits.length / remoteBatchSize);
      const merged: string[] = [];
      for (let offset = 0; offset < lineSafeUnits.length; offset += remoteBatchSize) {
        const batchUnits = lineSafeUnits.slice(offset, offset + remoteBatchSize);
        const batchIndex = Math.floor(offset / remoteBatchSize) + 1;
        onProgress?.(`Translating remote subtitle batch (${batchIndex}/${totalBatches})...`);
        const chunk = await processCloudLineSafeChunk(batchUnits);
        merged.push(...chunk.split('\n'));
      }
      output = merged.join('\n');
    } else {
      output = await processCloudLineSafeChunk(lineSafeUnits);
    }
  };

  const processCloudContextChunk = async (
    start: number,
    length: number,
    contextWindow: number,
    splitDepth: number
  ): Promise<string> => {
    const { text: chunkInput, targetUnits } = deps.buildCloudContextInput(lineSafeUnits, start, length, contextWindow);
    if (targetUnits.length === 0) return '';

    cloudContextChunkCount += 1;
    const raw = await requestCloudTranslation({
      text: chunkInput,
      lineSafeMode: false,
      promptOverride: deps.buildCloudContextSystemPrompt({
        targetLang: input.targetLang,
        sourceLang: input.sourceLang,
        glossary: input.glossary,
        promptTemplateId: input.promptTemplateId,
        prompt: input.prompt,
      }),
      progressMessage: `Calling translation provider (${input.providerRequest.provider}) with cloud context window...`,
      disableSystemPrompt: false,
    });

    const restored = deps.parseCloudContextOutput(raw, targetUnits);
    if (restored !== null) {
      return deps.normalizeTargetLanguageOutput(restored, input.targetLang);
    }

    addWarning('cloud_context_parse_failed');

    if (targetUnits.length === 1) {
      return translateSingleCloudUnit(targetUnits[0]);
    }

    const nextContextWindow = Math.max(0, contextWindow - 1);
    if (splitDepth < cloudContextConfig.maxSplitDepth) {
      addWarning('cloud_context_chunk_split');
      const leftLength = Math.max(1, Math.ceil(length / 2));
      const rightLength = Math.max(0, length - leftLength);
      const merged = [await processCloudContextChunk(start, leftLength, nextContextWindow, splitDepth + 1)];
      if (rightLength > 0) {
        merged.push(await processCloudContextChunk(start + leftLength, rightLength, nextContextWindow, splitDepth + 1));
      }
      return merged.join('\n');
    }

    addWarning('cloud_context_split_depth_exhausted');
    const fallbackLines: string[] = [];
    for (const unit of targetUnits) {
      fallbackLines.push(await translateSingleCloudUnit(unit));
    }
    return fallbackLines.join('\n');
  };

  const runCloudContextFlow = async () => {
    const chunks = deps.buildCloudContextChunks(
      lineSafeUnits,
      Math.max(cloudContextConfig.targetLines, cloudContextConfig.minTargetLines),
      cloudContextConfig.charBudget
    );
    const totalChunks = chunks.length;
    const merged: string[] = [];
    for (let index = 0; index < chunks.length; index += 1) {
      const chunk = chunks[index];
      onProgress?.(`Translating remote subtitle batch with cloud context (${index + 1}/${totalChunks})...`);
      merged.push(await processCloudContextChunk(chunk.start, chunk.length, cloudContextConfig.contextWindow, 0));
    }
    output = merged.join('\n');
  };

  const runProviderBatchFlow = async () => {
    if (!providerBatching) {
      output = await requestCloudTranslation({
        text: input.text,
        lineSafeMode: false,
        progressMessage: `Calling translation provider (${input.providerRequest.provider})...`,
      });
      return;
    }

    addWarning('cloud_provider_batch_translation_applied');
    const providerBatchMode: CloudTranslationProviderBatchMode =
      useCloudLineLocked ? 'line_safe' : 'plain_ordered';
    const durationsMs: number[] = [];
    const lineCounts: number[] = [];
    const charCounts: number[] = [];
    const estimatedOutputTokens: number[] = [];
    let splitCount = 0;

    const processProviderBatchChunk = async (
      units: CloudTranslationLineSafeUnit[],
      splitDepth: number
    ): Promise<string> => {
      const providerInputText =
        providerBatchMode === 'plain_ordered'
          ? units.map((unit) => deps.stripStructuredPrefix(unit.content).trim()).join('\n')
          : deps.buildLineSafeInput(units);
      const charCount = units.reduce((sum, unit) => sum + getUnitSourceChars(unit), 0);
      const outputTokenLimit = estimateCloudOutputTokens(
        providerInputText,
        units.length,
        providerBatching.maxOutputTokens
      );
      const modelOptionsOverride = buildProviderBatchModelOptions(
        input.providerRequest.modelOptions,
        providerBatching,
        outputTokenLimit
      );

      const splitAndRetry = async () => {
        if (units.length <= Math.max(1, providerBatching.minTargetLines) || splitDepth >= providerBatching.maxSplitDepth) {
          return null;
        }
        splitCount += 1;
        addWarning('cloud_provider_batch_split_applied');
        const midpoint = Math.ceil(units.length / 2);
        const left = await processProviderBatchChunk(units.slice(0, midpoint), splitDepth + 1);
        const right = await processProviderBatchChunk(units.slice(midpoint), splitDepth + 1);
        return [left, right].filter(Boolean).join('\n');
      };

      const startedAt = Date.now();
      let raw = '';
      try {
        raw = await requestCloudTranslation({
          text: providerInputText,
          lineSafeMode: providerBatchMode === 'line_safe',
          progressMessage: `Calling translation provider (${input.providerRequest.provider}) with hosted batch...`,
          modelOptionsOverride,
          retrySameRequest: false,
        });
      } catch (error) {
        if (deps.isRetryableError(error)) {
          const splitOutput = await splitAndRetry();
          if (splitOutput !== null) return splitOutput;
        }
        throw error;
      }

      const normalized = deps.normalizeTargetLanguageOutput(raw, input.targetLang);
      const restored = providerBatchMode === 'line_safe' ? deps.parseLineSafeOutput(normalized, units) : null;
      if (restored !== null) {
        addWarning('line_safe_alignment_applied');
        durationsMs.push(Date.now() - startedAt);
        lineCounts.push(units.length);
        charCounts.push(charCount);
        estimatedOutputTokens.push(outputTokenLimit);
        return deps.normalizeTargetLanguageOutput(restored, input.targetLang);
      }

      const rebound = deps.rebindByLineIndex(units, normalized);
      if (rebound !== null) {
        addWarning('line_index_rebind_applied');
        durationsMs.push(Date.now() - startedAt);
        lineCounts.push(units.length);
        charCounts.push(charCount);
        estimatedOutputTokens.push(outputTokenLimit);
        return deps.normalizeTargetLanguageOutput(rebound, input.targetLang);
      }

      const splitOutput = await splitAndRetry();
      if (splitOutput !== null) return splitOutput;

      addWarning('line_alignment_repair_failed');
      durationsMs.push(Date.now() - startedAt);
      lineCounts.push(units.length);
      charCounts.push(charCount);
      estimatedOutputTokens.push(outputTokenLimit);
      return providerBatchMode === 'line_safe' ? stripLineSafeMarkers(normalized) : normalized;
    };

    const chunks = buildProviderBatchChunks(
      lineSafeUnits,
      providerBatching.targetLines,
      providerBatching.charBudget
    );
    const totalBatches = chunks.length;
    const merged: string[] = [];
    for (let index = 0; index < chunks.length; index += 1) {
      onProgress?.(`Translating hosted cloud subtitle batch (${index + 1}/${totalBatches})...`);
      const chunk = await processProviderBatchChunk(chunks[index], 0);
      merged.push(...chunk.split('\n'));
    }

    output = merged.join('\n');
    const totalDurationMs = durationsMs.reduce((sum, value) => sum + value, 0);
    cloudBatching = {
      source: providerBatching.source,
      mode: providerBatchMode,
      batchCount: lineCounts.length,
      lineCounts,
      charCounts,
      estimatedOutputTokens,
      durationsMs,
      maxLines: providerBatching.targetLines,
      minLines: providerBatching.minTargetLines,
      charBudget: providerBatching.charBudget,
      maxOutputTokens: providerBatching.maxOutputTokens,
      timeoutMs: providerBatching.timeoutMs,
      stream: providerBatching.stream,
      splitCount,
      totalDurationMs,
      maxDurationMs: durationsMs.length > 0 ? Math.max(...durationsMs) : 0,
    };
  };

  if (useCloudStrictAlignment) {
    await runCloudLineSafeFlow();
  } else if (useCloudContextMode) {
    await runCloudContextFlow();
  } else if (useProviderBatchMode) {
    await runProviderBatchFlow();
  } else if (useCloudLineLocked) {
    await runCloudLineSafeFlow();
  } else {
    output = await requestCloudTranslation({
      text: input.text,
      lineSafeMode: false,
      progressMessage: `Calling translation provider (${input.providerRequest.provider})...`,
    });
  }

  output = deps.normalizeTargetLanguageOutput(output, input.targetLang);

  return {
    output,
    warnings,
    retryCount,
    lastError,
    fallbackUsed,
    fallbackType,
    endpointUsed,
    cloudStrategy,
    cloudContextChunkCount,
    cloudContextFallbackCount,
    cloudBatching,
    cloudQuota,
    cloudStrategyReason,
    sourceLineCount,
    sourceHasStructuredPrefixes,
    sourceSpeakerTaggedLineCount,
    relaxedWholeRequestApplied: false,
    relaxedWholeRequestFallback: false,
  };
}
