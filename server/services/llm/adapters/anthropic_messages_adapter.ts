import { getCanonicalProviderCapabilities } from '../canonical/llm_capabilities.js';
import type { CanonicalLlmResponse } from '../canonical/llm_types.js';
import { buildAnthropicMessages, getCanonicalInstructions } from '../mapping/provider_payloads.js';
import type { LlmAdapter, ProviderHttpResponse } from './base.js';

function parseAnthropicContent(data: any) {
  const blocks = data?.content;
  if (!Array.isArray(blocks)) return '';
  return blocks
    .map((block: any) => (block?.type === 'text' && typeof block?.text === 'string' ? block.text : ''))
    .join('');
}

export const anthropicMessagesAdapter: LlmAdapter = {
  key: 'anthropic-messages',
  capabilities: getCanonicalProviderCapabilities('anthropic-messages'),
  buildRequest(input, context) {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'anthropic-version': '2023-06-01',
    };
    if (context.apiKey) headers['x-api-key'] = context.apiKey;

    const body: Record<string, any> = {
      model: context.modelOverride || input.model,
      max_tokens: input.sampling?.maxOutputTokens ?? 4096,
      temperature: input.sampling?.temperature ?? 0.2,
      messages: buildAnthropicMessages(input),
    };
    if (input.sampling?.topP != null) {
      body.top_p = input.sampling.topP;
    }
    if (input.sampling?.topK != null) {
      body.top_k = input.sampling.topK;
    }
    if (Array.isArray(input.sampling?.stop) && input.sampling?.stop.length > 0) {
      body.stop_sequences = input.sampling.stop;
    }
    const instructions = getCanonicalInstructions(input);
    if (instructions) {
      body.system = instructions;
    }
    if (input.reasoning && input.reasoning.mode !== 'off') {
      body.thinking = {
        type: 'enabled',
        budget_tokens: input.reasoning.budgetTokens ?? 1024,
      };
    }

    return {
      method: 'POST' as const,
      url: context.endpointUrl,
      headers,
      body: JSON.stringify(body),
      timeoutMs: 120000,
    };
  },
  parseResponse(response: ProviderHttpResponse): CanonicalLlmResponse {
    const data: any = response.body;
    return {
      providerFamily: 'anthropic-messages',
      model: String(data?.model || ''),
      outputText: parseAnthropicContent(data).trim() || undefined,
      finishReason: typeof data?.stop_reason === 'string' ? data.stop_reason : undefined,
      usage: data?.usage
        ? {
            inputTokens: Number.isFinite(Number(data.usage?.input_tokens)) ? Number(data.usage.input_tokens) : undefined,
            outputTokens: Number.isFinite(Number(data.usage?.output_tokens)) ? Number(data.usage.output_tokens) : undefined,
          }
        : undefined,
      responseRef: typeof data?.id === 'string' ? data.id : undefined,
      rawProviderMeta: {
        type: data?.type,
      },
    };
  },
};
