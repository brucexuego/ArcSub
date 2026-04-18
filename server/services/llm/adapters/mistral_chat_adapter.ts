import { getCanonicalProviderCapabilities } from '../canonical/llm_capabilities.js';
import type { LlmAdapter } from './base.js';
import { openAiCompatibleChatAdapter } from './openai_chat_adapter.js';

export const mistralChatAdapter: LlmAdapter = {
  key: 'mistral-chat',
  capabilities: getCanonicalProviderCapabilities('mistral-chat'),
  buildRequest(input, context) {
    return openAiCompatibleChatAdapter.buildRequest(input, context);
  },
  parseResponse(response, context) {
    const parsed = openAiCompatibleChatAdapter.parseResponse(response, context);
    return {
      ...parsed,
      providerFamily: 'mistral-chat',
    };
  },
};
