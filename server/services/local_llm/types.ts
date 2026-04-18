export type LocalTranslatePromptStyle =
  | 'generic'
  | 'qwen_chatml'
  | 'qwen3_non_thinking'
  | 'deepseek_r1_distill_qwen'
  | 'deepseek_r1_plain'
  | 'phi4_chat'
  | 'gemma_plain';

export type LocalTranslateGenerationStyle = 'generic' | 'qwen' | 'qwen3' | 'deepseek_r1';

export interface LocalTranslateModelStrategy {
  family: 'generic' | 'seq2seq' | 'qwen2_5' | 'qwen3' | 'deepseek_r1_distill_qwen' | 'phi4' | 'gemma3';
  promptStyle: LocalTranslatePromptStyle;
  generationStyle: LocalTranslateGenerationStyle;
  qwenOptimized: boolean;
  qwen3Optimized: boolean;
}

export interface LocalTranslatePromptSignature {
  fingerprint: string;
  tokenizerConfigPath: string | null;
  chatTemplatePath: string | null;
}

export interface LocalTranslateGenerationOptions {
  doSample?: boolean;
  temperature?: number;
  topP?: number;
  topK?: number;
  minP?: number;
  maxNgramSize?: number;
  numAssistantTokens?: number;
  assistantConfidenceThreshold?: number;
  repetitionPenalty?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  noRepeatNgramSize?: number;
  applyChatTemplate?: boolean;
}

export type LocalOpenvinoBaselineConfidence = 'full_public' | 'partial_public' | 'gated_partial';

export interface LocalOpenvinoOfficialBaseline {
  id: string;
  repoId: string;
  runtimeFamily:
    | 'openvino-llm-node'
    | 'openvino-seq2seq-translate'
    | 'openvino-whisper-node'
    | 'openvino-ctc-asr'
    | 'openvino-qwen3-asr';
  taskFamily: 'translate' | 'asr' | 'vlm';
  baselineConfidence: LocalOpenvinoBaselineConfidence;
  sourceLinks: string[];
  officialGeneration?: {
    doSample?: boolean;
    temperature?: number;
    topP?: number;
    topK?: number;
    repetitionPenalty?: number;
    maxNewTokensExample?: number;
  };
  officialAsr?: {
    task?: 'transcribe' | 'translate';
    returnTimestamps?: boolean;
    chunkLengthSec?: number;
    samplingRate?: number;
    maxTargetPositions?: number;
  };
  notes?: string[];
}

export interface LocalOpenvinoResolvedProfile {
  baseline: LocalOpenvinoOfficialBaseline;
  profileId: string;
  profileFamily: string;
  usedFallbackBaseline: boolean;
}
