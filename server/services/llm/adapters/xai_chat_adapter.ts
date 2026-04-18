import { getCanonicalProviderCapabilities } from '../canonical/llm_capabilities.js';
import type { LlmAdapter } from './base.js';
import { openAiCompatibleChatAdapter } from './openai_chat_adapter.js';

export const xaiChatAdapter: LlmAdapter = {
  key: 'xai-chat',
  capabilities: getCanonicalProviderCapabilities('xai-chat'),
  buildRequest(input, context) {
    return openAiCompatibleChatAdapter.buildRequest(input, context);
  },
  parseResponse(response, context) {
    const parsed = openAiCompatibleChatAdapter.parseResponse(response, context);
    return {
      ...parsed,
      providerFamily: 'xai-chat',
    };
  },
};
