import {
  runCloudTranslationOrchestrator,
  type CloudTranslationOrchestratorDeps,
  type CloudTranslationOrchestratorInput,
} from './cloud_translation_orchestrator.js';
import {
  redactUrlSecrets,
  resolveCloudTranslateProvider,
  type ResolvedCloudTranslateProvider,
} from '../../cloud_translate_provider.js';
import type { ApiModelRequestOptions } from '../../../../src/types.js';

export interface CloudTranslationConnectionProbeInput {
  url: string;
  key?: string;
  model?: string;
  name?: string;
  options?: ApiModelRequestOptions;
}

export interface CloudTranslationConnectionProbeResult {
  success: boolean;
  message?: string;
  error?: string;
  resolvedProvider?: ResolvedCloudTranslateProvider;
}

export interface CloudTranslationConnectionProbeDeps {
  buildOrchestratorInput(
    input: {
      text: string;
      targetLang: string;
      enableJsonLineRepair: boolean;
      isConnectionTest: boolean;
    },
    resolvedProvider: ResolvedCloudTranslateProvider,
    model: { key?: string; model?: string; options?: ApiModelRequestOptions }
  ): CloudTranslationOrchestratorInput;
  buildOrchestratorDeps(): CloudTranslationOrchestratorDeps;
  mapConnectionError(error: unknown): string;
}

function sanitizeResolvedProviderForResponse(
  resolvedProvider: ResolvedCloudTranslateProvider
): ResolvedCloudTranslateProvider {
  return {
    ...resolvedProvider,
    endpointUrl: redactUrlSecrets(resolvedProvider.endpointUrl),
  };
}

export async function runCloudTranslationConnectionProbe(
  input: CloudTranslationConnectionProbeInput,
  deps: CloudTranslationConnectionProbeDeps
): Promise<CloudTranslationConnectionProbeResult> {
  const url = String(input.url || '').trim();
  const key = String(input.key || '').trim();
  const model = String(input.model || '').trim();
  const name = String(input.name || '').trim();
  const options = input.options;

  try {
    const resolvedProvider = resolveCloudTranslateProvider({
      url,
      modelName: name,
      model,
      apiKey: key,
      options,
    });
    await runCloudTranslationOrchestrator(
      deps.buildOrchestratorInput(
        {
          text: 'ping',
          targetLang: 'en',
          enableJsonLineRepair: false,
          isConnectionTest: true,
        },
        resolvedProvider,
        {
          key,
          model: resolvedProvider.effectiveModel || model,
          options,
        }
      ),
      deps.buildOrchestratorDeps()
    );
    return {
      success: true,
      message: 'Connection succeeded.',
      resolvedProvider: sanitizeResolvedProviderForResponse(resolvedProvider),
    };
  } catch (error) {
    return {
      success: false,
      error: deps.mapConnectionError(error),
    };
  }
}
