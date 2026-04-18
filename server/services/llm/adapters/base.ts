import type { CanonicalProviderCapabilities, LlmAdapterKey } from '../canonical/llm_capabilities.js';
import type { CanonicalLlmRequest, CanonicalLlmResponse } from '../canonical/llm_types.js';

export interface ProviderHttpRequest {
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  url: string;
  headers?: Record<string, string>;
  body?: unknown;
  timeoutMs?: number;
}

export interface ProviderHttpResponse {
  status: number;
  headers?: Record<string, string>;
  body: unknown;
}

export interface LlmAdapterContext {
  endpointUrl: string;
  apiKey?: string;
  providerName?: string;
  modelOverride?: string;
}

export interface LlmAdapter {
  key: LlmAdapterKey;
  capabilities: CanonicalProviderCapabilities;
  buildRequest(input: CanonicalLlmRequest, context: LlmAdapterContext): ProviderHttpRequest;
  parseResponse(response: ProviderHttpResponse, context: LlmAdapterContext): CanonicalLlmResponse;
}
