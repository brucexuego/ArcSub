import type { CanonicalProviderCapabilities, LlmAdapterKey } from '../canonical/llm_capabilities.js';
import { getCanonicalProviderCapabilities, listCanonicalProviderCapabilities } from '../canonical/llm_capabilities.js';
import type { LlmAdapter } from './base.js';

export class LlmAdapterRegistry {
  private readonly adapters = new Map<LlmAdapterKey, LlmAdapter>();

  register(adapter: LlmAdapter) {
    this.adapters.set(adapter.key, adapter);
    return adapter;
  }

  has(adapterKey: LlmAdapterKey) {
    return this.adapters.has(adapterKey);
  }

  get(adapterKey: LlmAdapterKey) {
    return this.adapters.get(adapterKey) || null;
  }

  require(adapterKey: LlmAdapterKey) {
    const adapter = this.get(adapterKey);
    if (!adapter) {
      throw new Error(`LLM adapter not registered: ${adapterKey}`);
    }
    return adapter;
  }

  list() {
    return Array.from(this.adapters.values());
  }

  listKeys() {
    return Array.from(this.adapters.keys());
  }
}

export const llmAdapterRegistry = new LlmAdapterRegistry();

export function getLlmAdapterCapabilities(adapterKey: LlmAdapterKey): CanonicalProviderCapabilities {
  return getCanonicalProviderCapabilities(adapterKey);
}

export function listLlmAdapterCapabilities() {
  return listCanonicalProviderCapabilities();
}
