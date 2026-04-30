import type { CloudAsrProvider, CloudAsrProviderDefinition, CloudAsrProviderMatchInput } from './types.js';
import { deepgramProvider } from './providers/deepgram.js';
import { elevenLabsProvider } from './providers/elevenlabs.js';
import { githubModelsProvider } from './providers/github_models.js';
import { gladiaProvider } from './providers/gladia.js';
import { googleCloudSpeechProvider } from './providers/google_cloud_speech.js';
import { googleGeminiProvider } from './providers/google_gemini_audio.js';
import { openAiWhisperProvider } from './providers/openai_whisper.js';
import { whisperCppProvider } from './providers/whispercpp.js';

const PROVIDERS: CloudAsrProviderDefinition[] = [
  elevenLabsProvider,
  deepgramProvider,
  gladiaProvider,
  githubModelsProvider,
  googleCloudSpeechProvider,
  googleGeminiProvider,
  whisperCppProvider,
  openAiWhisperProvider,
];

const PROVIDER_BY_KEY = new Map<CloudAsrProvider, CloudAsrProviderDefinition>(
  PROVIDERS.map((provider) => [provider.provider, provider])
);

export function listCloudAsrProviders() {
  return [...PROVIDERS];
}

export function getCloudAsrProviderDefinition(provider: CloudAsrProvider) {
  return PROVIDER_BY_KEY.get(provider) || null;
}

export function requireCloudAsrProviderDefinition(provider: CloudAsrProvider) {
  const definition = getCloudAsrProviderDefinition(provider);
  if (!definition) {
    throw new Error(`Cloud ASR provider is not registered: ${provider}`);
  }
  return definition;
}

export function matchCloudAsrProvider(input: CloudAsrProviderMatchInput): CloudAsrProviderDefinition {
  return PROVIDERS.find((provider) => provider.detect(input)) || openAiWhisperProvider;
}
