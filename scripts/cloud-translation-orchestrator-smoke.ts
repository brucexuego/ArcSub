import {
  runCloudTranslationOrchestrator,
  type CloudTranslationLineSafeUnit,
  type CloudTranslationOrchestratorDeps,
  type CloudTranslationProviderResult,
} from '../server/services/llm/orchestrators/cloud_translation_orchestrator.js';
import { getCloudTranslateAdapter } from '../server/services/cloud_translate_adapter.js';

class SmokeProviderHttpError extends Error {
  status: number;
  detail: string;
  retryAfterMs: number | null;

  constructor(prefix: string, status: number, detail: string, retryAfterMs: number | null = null) {
    super(`${prefix} (${status}): ${detail}`);
    this.status = status;
    this.detail = detail;
    this.retryAfterMs = retryAfterMs;
  }
}

function assert(condition: unknown, message: string) {
  if (!condition) throw new Error(message);
}

function buildLineSafeUnits(text: string): CloudTranslationLineSafeUnit[] {
  return String(text || '')
    .split(/\r?\n/)
    .map((line, index) => {
      const prefixMatch = line.match(/^(\[[^\]]+\]\s*)?(.*)$/);
      const prefix = prefixMatch?.[1] || '';
      const content = prefixMatch?.[2] || line;
      return {
        index: index + 1,
        marker: `[[L${String(index + 1).padStart(5, '0')}]]`,
        prefix,
        content,
        speakerTag: null,
      };
    });
}

function buildLineSafeInput(units: CloudTranslationLineSafeUnit[]) {
  return units.map((unit) => `${unit.marker} ${unit.prefix}${unit.content}`.trim()).join('\n');
}

function parseLineSafeOutput(output: string, units: CloudTranslationLineSafeUnit[]) {
  const map = new Map<number, string>();
  for (const line of String(output || '').split(/\r?\n/)) {
    const match = line.match(/\[\[L(\d{5})\]\]\s*(.*)$/);
    if (!match) continue;
    map.set(Number(match[1]), match[2].trim());
  }
  if (units.some((unit) => !map.has(unit.index))) return null;
  return units.map((unit) => `${unit.prefix}${map.get(unit.index) || ''}`.trim()).join('\n');
}

function buildDeps(mode: 'normal' | 'strict-repair'): CloudTranslationOrchestratorDeps {
  return {
    getCloudContextConfig: () => ({
      enabled: false,
      targetLines: 24,
      minTargetLines: 6,
      contextWindow: 2,
      charBudget: 2400,
      maxSplitDepth: 2,
    }),
    getRemoteBatchSize: () => 100,
    getRetryConfig: () => ({ maxRetries: 0, baseRetryMs: 1, rateLimitRetryMs: 1 }),
    throwIfAborted: () => {},
    sleep: async () => {},
    mapConnectionError: (error) => (error instanceof Error ? error.message : String(error || '')),
    isRetryableError: () => false,
    isProviderHttpError: (error): error is { status: number; retryAfterMs: number | null } =>
      error instanceof SmokeProviderHttpError,
    isProviderContentFilterError: () => false,
    requestTranslationByProvider: async (_provider, endpointUrl, options): Promise<CloudTranslationProviderResult> => {
      const text = String(options.text || '');
      if (mode === 'strict-repair' && options.lineSafeMode) {
        return {
          text: 'markerless strict output',
          meta: {
            endpointUrl,
            fallbackUsed: false,
            fallbackType: null,
            quota: {
              applied: true,
              profileId: 'smoke-quota',
              tokenEstimator: 'chars_heuristic',
              estimatedInputTokens: 10,
              estimatedTotalTokens: 20,
              waitedMs: 0,
              waitReason: null,
              waitEvents: [],
            },
          },
        };
      }
      if (options.lineSafeMode) {
        const translated = text
          .split(/\r?\n/)
          .map((line) => {
            const match = line.match(/^(\[\[L\d{5}\]\])\s*(.*)$/);
            return match ? `${match[1]} translated ${match[2]}` : `translated ${line}`;
          })
          .join('\n');
        return {
          text: translated,
          meta: { endpointUrl, fallbackUsed: false, fallbackType: null },
        };
      }
      return {
        text: `plain translated ${text}`,
        meta: { endpointUrl, fallbackUsed: false, fallbackType: null },
      };
    },
    buildLineSafeUnits,
    stripStructuredPrefix: (value) => String(value || '').replace(/^\[[^\]]+\]\s*/, ''),
    buildLineSafeInput,
    parseLineSafeOutput,
    normalizeTargetLanguageOutput: (output) => String(output || '').trim(),
    repairLineAlignmentWithJsonMap: async (_provider, _endpointUrl, units) => ({
      text: units.map((unit) => `${unit.prefix}strict repaired ${unit.content}`.trim()).join('\n'),
      missingCount: 0,
      warnings: ['line_json_map_repair_applied'],
    }),
    rebindByLineIndex: (units, translatedText) => {
      const lines = String(translatedText || '').split(/\r?\n/);
      if (lines.length < units.length) return null;
      return units.map((unit, index) => `${unit.prefix}${lines[index] || ''}`.trim()).join('\n');
    },
    buildCloudContextInput: (units) => ({ text: buildLineSafeInput(units), targetUnits: units }),
    buildCloudContextSystemPrompt: () => 'Translate target subtitle lines.',
    parseCloudContextOutput: parseLineSafeOutput,
    buildCloudContextChunks: (units) => [{ start: 0, length: units.length }],
  };
}

async function assertOpenAiSseErrorFrameFails() {
  const adapter = getCloudTranslateAdapter('openai-compatible');
  let failed = false;
  try {
    await adapter.request(
      'https://example.test/v1/chat/completions',
      {
        text: 'ping',
        targetLang: 'English',
        key: 'test-key',
        model: 'test-model',
        modelOptions: { body: { stream: true } },
      },
      {
        throwIfAborted: () => {},
        resolveRequestTimeoutMs: () => 1000,
        fetchWithTimeout: async () =>
          new Response('data: {"error":{"message":"stream exploded","code":429}}\n\n', {
            status: 200,
            headers: { 'content-type': 'text/event-stream' },
          }),
        parseRetryAfterMs: () => null,
        extractErrorMessage: (rawText, fallback) => rawText || fallback,
        resolveSystemPrompt: () => 'Translate.',
        normalizeDeepLTargetLanguage: (lang) => lang,
        parseOpenAiLikeContent: (content) => {
          if (typeof content === 'string') return content;
          return '';
        },
        parseGeminiContent: () => '',
        parseAnthropicContent: () => '',
        parseOllamaContent: () => '',
        parseResponsesContent: () => '',
        hasAnthropicEnvelope: () => false,
        hasGeminiEnvelope: () => false,
        hasOllamaEnvelope: () => false,
        hasOpenAiChatEnvelope: () => false,
        hasResponsesEnvelope: () => false,
        shouldFallbackToResponses: () => false,
        getOpenAiResponsesEndpoint: (url) => url,
        getOllamaFallbackEndpoint: (url) => url,
        makeProviderHttpError: (prefix, status, detail, retryAfterMs) =>
          new SmokeProviderHttpError(prefix, status, detail, retryAfterMs),
        isProviderHttpError: (error): error is SmokeProviderHttpError => error instanceof SmokeProviderHttpError,
      }
    );
  } catch (error) {
    failed = error instanceof SmokeProviderHttpError && /stream exploded/.test(error.detail);
  }
  assert(failed, 'OpenAI-compatible SSE error frame was not treated as a provider error.');
}

async function main() {
  const sample = ['[00:00:00] alpha', '[00:00:01] beta', '[00:00:02] gamma'].join('\n');

  const plain = await runCloudTranslationOrchestrator(
    {
      text: sample,
      targetLang: 'English',
      enableJsonLineRepair: false,
      qualityMode: 'plain_probe',
      supportsContextMode: false,
      providerRequest: {
        provider: 'openai-compatible',
        endpointUrl: 'https://example.test/v1/chat/completions',
      },
    },
    buildDeps('normal')
  );
  assert(plain.cloudStrategy === 'plain', 'plain request did not use plain strategy.');
  assert(!/\[\[L\d{5}\]\]/.test(plain.output), 'plain request leaked line-safe markers.');

  const lineLocked = await runCloudTranslationOrchestrator(
    {
      text: sample,
      targetLang: 'English',
      promptTemplateId: 'subtitle_general',
      enableJsonLineRepair: false,
      qualityMode: 'template_validated',
      supportsContextMode: false,
      providerRequest: {
        provider: 'openai-compatible',
        endpointUrl: 'https://example.test/v1/chat/completions',
      },
    },
    buildDeps('normal')
  );
  assert(lineLocked.cloudStrategy === 'line_locked', 'template request did not use line_locked strategy.');
  assert(lineLocked.warnings.includes('line_safe_alignment_applied'), 'line_locked did not apply line-safe parsing.');
  assert(!/\[\[L\d{5}\]\]/.test(lineLocked.output), 'line_locked output leaked internal markers.');

  const strict = await runCloudTranslationOrchestrator(
    {
      text: sample,
      targetLang: 'English',
      promptTemplateId: 'subtitle_general',
      enableJsonLineRepair: true,
      qualityMode: 'json_strict',
      supportsContextMode: false,
      providerRequest: {
        provider: 'openai-compatible',
        endpointUrl: 'https://example.test/v1/chat/completions',
      },
    },
    buildDeps('strict-repair')
  );
  assert(strict.cloudStrategy === 'cloud_strict', 'strict request did not use cloud_strict strategy.');
  assert(strict.warnings.includes('line_json_map_repair_applied'), 'cloud_strict did not use JSON repair fallback.');
  assert(strict.cloudQuota?.estimatedTotalTokens === 20, 'quota estimate was not preserved in debug output.');

  const batched = await runCloudTranslationOrchestrator(
    {
      text: sample,
      targetLang: 'English',
      promptTemplateId: 'subtitle_general',
      enableJsonLineRepair: false,
      qualityMode: 'template_validated',
      supportsContextMode: false,
      providerRequest: {
        provider: 'openai-compatible',
        endpointUrl: 'https://example.test/v1/chat/completions',
      },
      providerBatching: {
        enabled: true,
        source: 'github-models',
        targetLines: 2,
        minTargetLines: 1,
        charBudget: 20,
        maxSplitDepth: 2,
        maxOutputTokens: 512,
        timeoutMs: 30000,
        stream: false,
      },
    },
    buildDeps('normal')
  );
  assert(batched.cloudStrategy === 'line_locked', 'batched template request should still report line_locked.');
  assert(batched.cloudBatching?.mode === 'line_safe', 'provider batch did not keep line_safe mode in line_locked.');
  assert(batched.cloudBatching?.source === 'github-models', 'provider batch source was not preserved.');

  await assertOpenAiSseErrorFrameFails();

  console.log('cloud translation orchestrator smoke passed');
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
