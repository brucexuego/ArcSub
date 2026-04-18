export type CanonicalMessageRole = 'system' | 'developer' | 'user' | 'assistant' | 'tool';

export type CanonicalTextPart = {
  type: 'text';
  text: string;
};

export type CanonicalImageUrlPart = {
  type: 'image_url';
  url: string;
  detail?: 'low' | 'high' | 'auto';
};

export type CanonicalFileUrlPart = {
  type: 'file_url';
  url: string;
  mimeType?: string;
  filename?: string;
};

export type CanonicalToolCallPart = {
  type: 'tool_call';
  name: string;
  arguments: unknown;
  callId?: string;
};

export type CanonicalToolResultPart = {
  type: 'tool_result';
  output: unknown;
  callId?: string;
};

export type CanonicalMessagePart =
  | CanonicalTextPart
  | CanonicalImageUrlPart
  | CanonicalFileUrlPart
  | CanonicalToolCallPart
  | CanonicalToolResultPart;

export interface CanonicalMessage {
  role: CanonicalMessageRole;
  parts: CanonicalMessagePart[];
}

export type CanonicalStructuredOutput =
  | { mode: 'text' }
  | { mode: 'json_object' }
  | {
      mode: 'json_schema';
      schema: unknown;
      name?: string;
      strict?: boolean;
    };

export interface CanonicalSamplingConfig {
  temperature?: number;
  topP?: number;
  topK?: number;
  seed?: number;
  maxOutputTokens?: number;
  stop?: string[];
  frequencyPenalty?: number;
  presencePenalty?: number;
}

export interface CanonicalReasoningConfig {
  mode?: 'off' | 'on' | 'adaptive';
  effort?: 'low' | 'medium' | 'high' | 'max';
  budgetTokens?: number;
  exposeSummary?: boolean;
}

export interface CanonicalToolDefinition {
  name: string;
  description?: string;
  inputSchema?: unknown;
}

export type CanonicalToolChoice = 'auto' | 'none' | 'required' | { name: string };

export interface CanonicalToolConfig {
  tools?: CanonicalToolDefinition[];
  choice?: CanonicalToolChoice;
  strict?: boolean;
  allowParallel?: boolean;
}

export interface CanonicalLlmRequest {
  model: string;
  instructions?: string;
  messages: CanonicalMessage[];
  structuredOutput?: CanonicalStructuredOutput;
  sampling?: CanonicalSamplingConfig;
  reasoning?: CanonicalReasoningConfig;
  tooling?: CanonicalToolConfig;
  metadata?: Record<string, string>;
  conversationRef?: string;
  providerHints?: Record<string, unknown>;
}

export interface CanonicalToolCall {
  id?: string;
  name: string;
  arguments: unknown;
}

export interface CanonicalTokenUsage {
  inputTokens?: number;
  outputTokens?: number;
  reasoningTokens?: number;
  cachedInputTokens?: number;
}

export interface CanonicalLlmResponse {
  providerFamily: string;
  model: string;
  outputText?: string;
  outputJson?: unknown;
  toolCalls?: CanonicalToolCall[];
  finishReason?: string;
  usage?: CanonicalTokenUsage;
  reasoningSummary?: string;
  responseRef?: string;
  warnings?: string[];
  rawProviderMeta?: Record<string, unknown>;
}
