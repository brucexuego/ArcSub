import { transcribeWithLocalCtcAsrRuntime } from './runtimes/ctc_asr_runtime.js';
import { transcribeWithLocalQwen3AsrRuntime } from './runtimes/qwen3_asr_runtime.js';
import { transcribeWithLocalWhisperRuntime } from './runtimes/whisper_runtime.js';
import type { LocalAsrRuntimeInput } from './runtimes/shared.js';

export async function transcribeWithLocalAsrRuntime(input: LocalAsrRuntimeInput) {
  const isQwen3AsrRuntime = input.localModelRuntime === 'openvino-qwen3-asr';
  const isCtcAsrRuntime = input.localModelRuntime === 'openvino-ctc-asr';
  const isCohereAsrRuntime = input.localModelRuntime === 'openvino-cohere-asr';
  const isHfTransformersAsrRuntime = input.localModelRuntime === 'hf-transformers-asr';

  if (isQwen3AsrRuntime) {
    return transcribeWithLocalQwen3AsrRuntime(input);
  }
  if (isCtcAsrRuntime || isCohereAsrRuntime || isHfTransformersAsrRuntime) {
    return transcribeWithLocalCtcAsrRuntime(input);
  }

  return transcribeWithLocalWhisperRuntime(input);
}
