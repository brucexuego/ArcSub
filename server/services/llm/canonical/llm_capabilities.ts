export type LlmAdapterKey =
  | 'openai-responses'
  | 'openai-chat'
  | 'anthropic-messages'
  | 'gemini-native'
  | 'mistral-chat'
  | 'cohere-chat'
  | 'xai-chat'
  | 'openai-compatible-chat';

export type LlmRequestFamily = 'responses-items' | 'chat-messages' | 'contents-parts';

export type SystemPromptLocation =
  | 'instructions'
  | 'message-system-role'
  | 'message-developer-role'
  | 'top-level-system'
  | 'system-instruction';

export interface CanonicalProviderCapabilities {
  adapterKey: LlmAdapterKey;
  requestFamily: LlmRequestFamily;
  systemPromptLocation: SystemPromptLocation;
  supportsJsonObject: boolean;
  supportsJsonSchema: boolean;
  supportsTools: boolean;
  supportsStrictToolSchema: boolean;
  supportsReasoningConfig: boolean;
  supportsTopK: boolean;
  supportsTopP: boolean;
  supportsSeed: boolean;
  supportsConversationReference: boolean;
  supportsParallelToolCalls: boolean;
  supportsDeveloperRole: boolean;
}

export const CANONICAL_PROVIDER_CAPABILITIES: Record<LlmAdapterKey, CanonicalProviderCapabilities> = {
  'openai-responses': {
    adapterKey: 'openai-responses',
    requestFamily: 'responses-items',
    systemPromptLocation: 'instructions',
    supportsJsonObject: true,
    supportsJsonSchema: true,
    supportsTools: true,
    supportsStrictToolSchema: true,
    supportsReasoningConfig: true,
    supportsTopK: false,
    supportsTopP: true,
    supportsSeed: true,
    supportsConversationReference: true,
    supportsParallelToolCalls: true,
    supportsDeveloperRole: true,
  },
  'openai-chat': {
    adapterKey: 'openai-chat',
    requestFamily: 'chat-messages',
    systemPromptLocation: 'message-system-role',
    supportsJsonObject: true,
    supportsJsonSchema: true,
    supportsTools: true,
    supportsStrictToolSchema: true,
    supportsReasoningConfig: true,
    supportsTopK: false,
    supportsTopP: true,
    supportsSeed: true,
    supportsConversationReference: false,
    supportsParallelToolCalls: true,
    supportsDeveloperRole: true,
  },
  'anthropic-messages': {
    adapterKey: 'anthropic-messages',
    requestFamily: 'chat-messages',
    systemPromptLocation: 'top-level-system',
    supportsJsonObject: true,
    supportsJsonSchema: true,
    supportsTools: true,
    supportsStrictToolSchema: true,
    supportsReasoningConfig: true,
    supportsTopK: true,
    supportsTopP: true,
    supportsSeed: false,
    supportsConversationReference: false,
    supportsParallelToolCalls: false,
    supportsDeveloperRole: false,
  },
  'gemini-native': {
    adapterKey: 'gemini-native',
    requestFamily: 'contents-parts',
    systemPromptLocation: 'system-instruction',
    supportsJsonObject: true,
    supportsJsonSchema: true,
    supportsTools: true,
    supportsStrictToolSchema: false,
    supportsReasoningConfig: false,
    supportsTopK: true,
    supportsTopP: true,
    supportsSeed: true,
    supportsConversationReference: false,
    supportsParallelToolCalls: false,
    supportsDeveloperRole: false,
  },
  'mistral-chat': {
    adapterKey: 'mistral-chat',
    requestFamily: 'chat-messages',
    systemPromptLocation: 'message-system-role',
    supportsJsonObject: true,
    supportsJsonSchema: true,
    supportsTools: true,
    supportsStrictToolSchema: false,
    supportsReasoningConfig: true,
    supportsTopK: false,
    supportsTopP: true,
    supportsSeed: true,
    supportsConversationReference: false,
    supportsParallelToolCalls: true,
    supportsDeveloperRole: false,
  },
  'cohere-chat': {
    adapterKey: 'cohere-chat',
    requestFamily: 'chat-messages',
    systemPromptLocation: 'message-system-role',
    supportsJsonObject: true,
    supportsJsonSchema: true,
    supportsTools: true,
    supportsStrictToolSchema: true,
    supportsReasoningConfig: true,
    supportsTopK: true,
    supportsTopP: true,
    supportsSeed: true,
    supportsConversationReference: false,
    supportsParallelToolCalls: false,
    supportsDeveloperRole: false,
  },
  'xai-chat': {
    adapterKey: 'xai-chat',
    requestFamily: 'chat-messages',
    systemPromptLocation: 'message-system-role',
    supportsJsonObject: true,
    supportsJsonSchema: true,
    supportsTools: true,
    supportsStrictToolSchema: false,
    supportsReasoningConfig: true,
    supportsTopK: false,
    supportsTopP: true,
    supportsSeed: true,
    supportsConversationReference: false,
    supportsParallelToolCalls: true,
    supportsDeveloperRole: false,
  },
  'openai-compatible-chat': {
    adapterKey: 'openai-compatible-chat',
    requestFamily: 'chat-messages',
    systemPromptLocation: 'message-system-role',
    supportsJsonObject: true,
    supportsJsonSchema: false,
    supportsTools: true,
    supportsStrictToolSchema: false,
    supportsReasoningConfig: false,
    supportsTopK: false,
    supportsTopP: true,
    supportsSeed: true,
    supportsConversationReference: false,
    supportsParallelToolCalls: false,
    supportsDeveloperRole: false,
  },
};

export function getCanonicalProviderCapabilities(adapterKey: LlmAdapterKey) {
  return CANONICAL_PROVIDER_CAPABILITIES[adapterKey];
}

export function listCanonicalProviderCapabilities() {
  return Object.values(CANONICAL_PROVIDER_CAPABILITIES);
}
