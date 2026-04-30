import type { CloudAsrProvider, CloudAsrProviderCapabilities } from '../types.js';

const BASE_CAPABILITIES: CloudAsrProviderCapabilities = {
  nativeWordTimestamps: false,
  nativeDiarization: false,
  supportsBatchAudio: false,
  requiresVadTimestamping: false,
  bypassesLocalAdvancedProcessing: false,
  nativeProviderLabel: null,
  legacyNativeFeatureMetaKey: null,
  vadTimestampSourceMetaKey: null,
  vadTimestampingMetaKey: null,
  maxUploadBytes: null,
  maxDurationSec: null,
  profileId: null,
  profileFamily: null,
};

const CAPABILITIES: Record<CloudAsrProvider, CloudAsrProviderCapabilities> = {
  'openai-whisper': {
    ...BASE_CAPABILITIES,
    profileId: 'openai-whisper',
    profileFamily: 'openai-compatible-audio-transcriptions',
  },
  'whispercpp-inference': {
    ...BASE_CAPABILITIES,
    profileId: 'whispercpp-inference',
    profileFamily: 'whispercpp-inference',
  },
  'elevenlabs': {
    ...BASE_CAPABILITIES,
    nativeWordTimestamps: true,
    nativeDiarization: true,
    bypassesLocalAdvancedProcessing: true,
    nativeProviderLabel: 'elevenlabs',
    legacyNativeFeatureMetaKey: 'elevenLabsNativeAdvancedFeatures',
    maxUploadBytes: 3 * 1024 * 1024 * 1024,
    maxDurationSec: 10 * 60 * 60,
    profileId: 'elevenlabs-scribe',
    profileFamily: 'provider-native-speech-to-text',
  },
  'deepgram': {
    ...BASE_CAPABILITIES,
    nativeWordTimestamps: true,
    nativeDiarization: true,
    bypassesLocalAdvancedProcessing: true,
    nativeProviderLabel: 'deepgram',
    legacyNativeFeatureMetaKey: 'deepgramNativeAdvancedFeatures',
    maxUploadBytes: 2 * 1024 * 1024 * 1024,
    profileId: 'deepgram-listen',
    profileFamily: 'provider-native-speech-to-text',
  },
  'gladia': {
    ...BASE_CAPABILITIES,
    nativeWordTimestamps: true,
    nativeDiarization: true,
    bypassesLocalAdvancedProcessing: true,
    nativeProviderLabel: 'gladia',
    legacyNativeFeatureMetaKey: 'gladiaNativeAdvancedFeatures',
    maxUploadBytes: 1000 * 1024 * 1024,
    maxDurationSec: 135 * 60,
    profileId: 'gladia-pre-recorded',
    profileFamily: 'provider-native-speech-to-text',
  },
  'github-models': {
    ...BASE_CAPABILITIES,
    profileId: 'github-phi4-multimodal',
    profileFamily: 'multimodal-chat-asr',
  },
  'google-cloud-speech': {
    ...BASE_CAPABILITIES,
    nativeWordTimestamps: true,
    nativeProviderLabel: 'google-cloud',
    profileId: 'google-chirp3',
    profileFamily: 'google-cloud-speech',
  },
  'google-gemini': {
    ...BASE_CAPABILITIES,
    supportsBatchAudio: true,
    requiresVadTimestamping: true,
    nativeProviderLabel: 'gemini',
    vadTimestampSourceMetaKey: 'geminiTimestampSource',
    vadTimestampingMetaKey: 'geminiVadTimestamping',
    profileId: 'google-gemini-audio',
    profileFamily: 'multimodal-generative-asr',
  },
};

export function getCloudAsrProviderCapabilities(provider: CloudAsrProvider): CloudAsrProviderCapabilities {
  return CAPABILITIES[provider] || BASE_CAPABILITIES;
}

export function getCloudAsrLegacyMetaFlags(provider: CloudAsrProvider) {
  return {
    isWhisperCppInference: provider === 'whispercpp-inference',
    isElevenLabsScribe: provider === 'elevenlabs',
    isDeepgramListen: provider === 'deepgram',
    isGladiaPreRecorded: provider === 'gladia',
    isGithubModelsPhi4Multimodal: provider === 'github-models',
    isGoogleCloudChirp3: provider === 'google-cloud-speech',
    isGoogleGeminiAudio: provider === 'google-gemini',
  };
}
