import { llmAdapterRegistry } from './registry.js';
import { openAiChatAdapter, openAiCompatibleChatAdapter } from './openai_chat_adapter.js';
import { openAiResponsesAdapter } from './openai_responses_adapter.js';
import { anthropicMessagesAdapter } from './anthropic_messages_adapter.js';
import { geminiNativeAdapter } from './gemini_native_adapter.js';
import { mistralChatAdapter } from './mistral_chat_adapter.js';
import { cohereChatAdapter } from './cohere_chat_adapter.js';
import { xaiChatAdapter } from './xai_chat_adapter.js';

const BUILTIN_LLM_ADAPTERS = [
  openAiChatAdapter,
  openAiCompatibleChatAdapter,
  openAiResponsesAdapter,
  anthropicMessagesAdapter,
  geminiNativeAdapter,
  mistralChatAdapter,
  cohereChatAdapter,
  xaiChatAdapter,
];

for (const adapter of BUILTIN_LLM_ADAPTERS) {
  if (!llmAdapterRegistry.has(adapter.key)) {
    llmAdapterRegistry.register(adapter);
  }
}

export function ensureBuiltinLlmAdaptersRegistered() {
  return llmAdapterRegistry;
}
