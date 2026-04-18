import {
  runCloudTranslationOrchestrator,
  type CloudTranslationOrchestratorDeps,
  type CloudTranslationOrchestratorInput,
} from './cloud_translation_orchestrator.js';
import { resolveCloudTranslateProvider, type ResolvedCloudTranslateProvider } from '../../cloud_translate_provider.js';

export interface CloudTranslationConnectionProbeInput {
  url: string;
  key?: string;
  model?: string;
  name?: string;
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
    model: { key?: string; model?: string }
  ): CloudTranslationOrchestratorInput;
  buildOrchestratorDeps(): CloudTranslationOrchestratorDeps;
  mapConnectionError(error: unknown): string;
}

export async function runCloudTranslationConnectionProbe(
  input: CloudTranslationConnectionProbeInput,
  deps: CloudTranslationConnectionProbeDeps
): Promise<CloudTranslationConnectionProbeResult> {
  const url = String(input.url || '').trim();
  const key = String(input.key || '').trim();
  const model = String(input.model || '').trim();
  const name = String(input.name || '').trim();

  try {
    const resolvedProvider = resolveCloudTranslateProvider({
      url,
      modelName: name,
      model,
      apiKey: key,
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
        }
      ),
      deps.buildOrchestratorDeps()
    );
    return {
      success: true,
      message: 'Connection succeeded.',
      resolvedProvider,
    };
  } catch (error) {
    return {
      success: false,
      error: deps.mapConnectionError(error),
    };
  }
}
