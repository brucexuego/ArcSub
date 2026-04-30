import { requireCloudAsrProviderDefinition } from './registry.js';
import type {
  CloudAsrAdapterRequestOptions,
  CloudAsrChunkingPolicy,
  CloudAsrFileInfo,
  CloudAsrPreflightResult,
  ResolvedCloudAsrProvider,
} from './types.js';
import { isCloudAsrFileLimitError } from './profiles/errors.js';

export interface RunCloudAsrPreflightInput {
  filePath: string;
  resolvedProvider: ResolvedCloudAsrProvider;
  config: any;
  options: CloudAsrAdapterRequestOptions;
  chunkingPolicy: CloudAsrChunkingPolicy | null;
  getFileInfo(): Promise<CloudAsrFileInfo>;
}

export async function runCloudAsrPreflight(input: RunCloudAsrPreflightInput): Promise<CloudAsrPreflightResult> {
  const definition = requireCloudAsrProviderDefinition(input.resolvedProvider.provider);
  if (!definition.preflight) {
    return { action: 'direct' };
  }
  return definition.preflight(input);
}

export function shouldChunkCloudAsrError(
  error: unknown,
  input: Omit<RunCloudAsrPreflightInput, 'filePath' | 'getFileInfo'>
) {
  const definition = requireCloudAsrProviderDefinition(input.resolvedProvider.provider);
  if (definition.shouldChunkOnError) {
    return definition.shouldChunkOnError({
      error,
      resolvedProvider: input.resolvedProvider,
      config: input.config,
      options: input.options,
      chunkingPolicy: input.chunkingPolicy,
    });
  }
  return isCloudAsrFileLimitError(error);
}
