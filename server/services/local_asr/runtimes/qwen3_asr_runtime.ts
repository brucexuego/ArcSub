import {
  buildStructuredLocalAsrResult,
  type LocalAsrProviderResult,
  type LocalAsrRuntimeInput,
  runOpenvinoLocalAsr,
} from './shared.js';

export async function transcribeWithLocalQwen3AsrRuntime(input: LocalAsrRuntimeInput): Promise<LocalAsrProviderResult> {
  const localResult = await runOpenvinoLocalAsr(input);
  return buildStructuredLocalAsrResult({
    localResult,
    language: input.language,
    extractStructuredTranscript: input.extractStructuredTranscript,
    toFiniteNumber: input.toFiniteNumber,
  });
}
