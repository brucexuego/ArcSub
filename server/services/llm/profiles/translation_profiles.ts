import { llmModelProfileRegistry } from './registry.js';

export type ProviderTranslationProfileConfidence =
  | 'official_translation_specific'
  | 'official_general_only'
  | 'arcsub_empirical';

export interface ProviderTranslationRecommendationBundle {
  verifiedOn: string;
  sourceLinks: string[];
  promptNotes?: string[];
  parameterNotes?: string[];
  constraints?: string[];
}

export interface ProviderTranslationArcSubDefaults {
  temperature?: number;
  topP?: number;
  topK?: number;
  preferContextMode?: boolean;
  preferSchemaMode?: boolean;
  preferJsonStrictForSubtitles?: boolean;
  notes?: string[];
}

export interface ProviderTranslationProfile {
  id: string;
  providerFamily: string;
  modelFamily?: string;
  runtimeFamily?: string;
  confidence: ProviderTranslationProfileConfidence;
  officialRecommendations: ProviderTranslationRecommendationBundle;
  arcsubDefaults: ProviderTranslationArcSubDefaults;
}

export interface ResolveProviderTranslationProfileInput {
  providerFamily?: string;
  runtimeFamily?: string;
  modelFamily?: string;
  modelName?: string;
}

export const PROVIDER_TRANSLATION_PROFILES: ProviderTranslationProfile[] = [
  {
    id: 'deepl',
    providerFamily: 'deepl',
    confidence: 'official_translation_specific',
    officialRecommendations: {
      verifiedOn: '2026-04-15',
      sourceLinks: [
        'https://developers.deepl.com/api-reference',
        'https://developers.deepl.com/api-reference/multilingual-glossaries',
        'https://developers.deepl.com/docs/xml-and-html-handling/tag-handling-v2',
      ],
      promptNotes: [
        'Prefer dedicated translation controls over chat-style prompt tuning.',
        'Use provider glossary and context features when available.',
      ],
      parameterNotes: [
        'Use context for disambiguation.',
        'Use glossary_id for terminology control.',
        'Use model_type where higher-quality translation is required.',
      ],
    },
    arcsubDefaults: {
      preferContextMode: true,
      preferJsonStrictForSubtitles: false,
      notes: [
        'DeepL is already a translation-native provider, so ArcSub should avoid forcing LLM-style schema behavior onto it.',
      ],
    },
  },
  {
    id: 'claude',
    providerFamily: 'anthropic',
    modelFamily: 'claude',
    runtimeFamily: 'anthropic-messages',
    confidence: 'official_translation_specific',
    officialRecommendations: {
      verifiedOn: '2026-04-15',
      sourceLinks: [
        'https://docs.anthropic.com/en/resources/prompt-library/polyglot-superpowers',
        'https://docs.anthropic.com/en/api/messages',
      ],
      promptNotes: [
        'Use a translator role/system prompt.',
        'Keep task instructions explicit and output-only.',
      ],
      parameterNotes: ['Anthropic translation prompt example uses temperature=0.2.'],
    },
    arcsubDefaults: {
      temperature: 0.2,
      topP: 0.95,
      topK: 40,
      preferSchemaMode: true,
      preferJsonStrictForSubtitles: true,
    },
  },
  {
    id: 'cohere-command-a-translate',
    providerFamily: 'cohere',
    modelFamily: 'command-a-translate',
    runtimeFamily: 'cohere-chat',
    confidence: 'official_translation_specific',
    officialRecommendations: {
      verifiedOn: '2026-04-15',
      sourceLinks: [
        'https://docs.cohere.com/page/command-a-translate',
        'https://docs.cohere.com/docs/model-vault',
      ],
      promptNotes: ['Prefer the dedicated translation model over generic chat models when translation quality is the primary goal.'],
      parameterNotes: ['Favor translation-native model selection before additional prompt complexity.'],
    },
    arcsubDefaults: {
      temperature: 0.2,
      topP: 1,
      topK: 0,
      preferJsonStrictForSubtitles: true,
    },
  },
  {
    id: 'openai-gpt',
    providerFamily: 'openai',
    modelFamily: 'gpt',
    runtimeFamily: 'openai-chat',
    confidence: 'official_general_only',
    officialRecommendations: {
      verifiedOn: '2026-04-15',
      sourceLinks: [
        'https://platform.openai.com/docs/guides/prompt-engineering/strategies-for-better-results',
        'https://platform.openai.com/docs/guides/fine-tuning',
      ],
      promptNotes: [
        'Make output requirements explicit.',
        'Use evaluation and model pinning for production stability.',
      ],
      parameterNotes: ['No official generic GPT translation-specific parameter recipe was identified in the reviewed public docs.'],
      constraints: ['Treat ArcSub subtitle defaults as empirical product defaults, not official OpenAI translation presets.'],
    },
    arcsubDefaults: {
      temperature: 0.2,
      topP: 1,
      preferSchemaMode: true,
      preferJsonStrictForSubtitles: true,
    },
  },
  {
    id: 'gemini',
    providerFamily: 'google',
    modelFamily: 'gemini',
    runtimeFamily: 'gemini-native',
    confidence: 'official_general_only',
    officialRecommendations: {
      verifiedOn: '2026-04-15',
      sourceLinks: [
        'https://ai.google.dev/gemini-api/docs/system-instructions',
        'https://ai.google.dev/guide/prompt_best_practices',
      ],
      promptNotes: [
        'Use system instructions and explicit output formatting requirements.',
        'Use clear task framing instead of relying on vague prompt bias.',
      ],
      parameterNotes: ['No official Gemini translation-specific subtitle parameter recipe was identified in the reviewed public docs.'],
    },
    arcsubDefaults: {
      temperature: 0.2,
      topP: 0.95,
      topK: 40,
      preferSchemaMode: true,
      preferJsonStrictForSubtitles: true,
    },
  },
  {
    id: 'gemma-google-native',
    providerFamily: 'google',
    modelFamily: 'gemma',
    runtimeFamily: 'gemini-native',
    confidence: 'arcsub_empirical',
    officialRecommendations: {
      verifiedOn: '2026-04-30',
      sourceLinks: ['https://ai.google.dev/gemini-api/docs/models/gemma'],
      promptNotes: [
        'Use direct translation wording and collect only non-thought output parts.',
        'Keep subtitle marker requirements short and explicit.',
      ],
      parameterNotes: [
        'Streaming is useful for long requests, but thought parts must not be treated as final text.',
      ],
    },
    arcsubDefaults: {
      temperature: 0.2,
      topP: 0.95,
      topK: 40,
      preferSchemaMode: true,
      preferJsonStrictForSubtitles: true,
      notes: [
        'Gemma hosted through Gemini native APIs is more sensitive to long rule lists, so ArcSub keeps concise provider-native prompting available.',
      ],
    },
  },
  {
    id: 'mistral',
    providerFamily: 'mistral',
    modelFamily: 'mistral',
    runtimeFamily: 'mistral-chat',
    confidence: 'official_general_only',
    officialRecommendations: {
      verifiedOn: '2026-04-15',
      sourceLinks: ['https://docs.mistral.ai/capabilities/completion/prompting_capabilities'],
      promptNotes: ['Use system prompts, explicit formatting requirements, and few-shot prompting when needed.'],
      parameterNotes: ['No official Mistral translation-specific parameter recipe was identified in the reviewed public docs.'],
    },
    arcsubDefaults: {
      temperature: 0.2,
      topP: 1,
      preferSchemaMode: true,
      preferJsonStrictForSubtitles: true,
    },
  },
  {
    id: 'xai-grok',
    providerFamily: 'xai',
    modelFamily: 'grok',
    runtimeFamily: 'xai-chat',
    confidence: 'official_general_only',
    officialRecommendations: {
      verifiedOn: '2026-04-15',
      sourceLinks: [
        'https://docs.x.ai/developers/api-reference',
        'https://docs.x.ai/developers/advanced-api-usage/prompt-caching/best-practices',
      ],
      promptNotes: ['Use explicit task framing and stable prompt structure.'],
      parameterNotes: ['No official xAI translation-specific parameter recipe was identified in the reviewed public docs.'],
    },
    arcsubDefaults: {
      temperature: 0.2,
      topP: 1,
      preferSchemaMode: true,
      preferJsonStrictForSubtitles: true,
    },
  },
  {
    id: 'openai-compatible-generic',
    providerFamily: 'openai-compatible',
    runtimeFamily: 'openai-compatible-chat',
    confidence: 'arcsub_empirical',
    officialRecommendations: {
      verifiedOn: '2026-04-15',
      sourceLinks: [],
      parameterNotes: ['No stable provider-level translation preset is assumed for generic OpenAI-compatible runtimes.'],
      constraints: ['Model behavior must be treated as runtime-specific and empirically validated.'],
    },
    arcsubDefaults: {
      temperature: 0.2,
      topP: 1,
      preferSchemaMode: true,
      preferJsonStrictForSubtitles: true,
      notes: [
        'Use family defaults plus model-family overrides for DeepSeek, Qwen, Llama, GitHub Models, and similar runtimes.',
      ],
    },
  },
];

export function listProviderTranslationProfiles() {
  return [...PROVIDER_TRANSLATION_PROFILES];
}

export function getProviderTranslationProfileById(id: string) {
  return PROVIDER_TRANSLATION_PROFILES.find((profile) => profile.id === id) || null;
}

export function getProviderTranslationProfilesByFamily(providerFamily: string) {
  const normalized = String(providerFamily || '').trim().toLowerCase();
  return PROVIDER_TRANSLATION_PROFILES.filter((profile) => profile.providerFamily.toLowerCase() === normalized);
}

function normalizeProviderFamily(providerFamily: string | null | undefined) {
  const normalized = String(providerFamily || '').trim().toLowerCase();
  if (normalized === 'gemini-native') return 'google';
  if (normalized === 'anthropic') return 'anthropic';
  if (normalized === 'mistral-chat') return 'mistral';
  if (normalized === 'cohere-chat') return 'cohere';
  if (normalized === 'xai-chat') return 'xai';
  if (normalized === 'openai-compatible') return 'openai-compatible';
  if (normalized === 'deepl') return 'deepl';
  if (normalized === 'openai') return 'openai';
  return normalized;
}

function scoreProviderTranslationProfile(
  profile: ProviderTranslationProfile,
  input: {
    providerFamily: string;
    runtimeFamily: string;
    modelFamily: string;
  }
) {
  if (profile.providerFamily.toLowerCase() !== input.providerFamily) return -1;
  let score = 1;
  if (profile.runtimeFamily && profile.runtimeFamily.toLowerCase() === input.runtimeFamily) score += 4;
  if (profile.modelFamily && profile.modelFamily.toLowerCase() === input.modelFamily) score += 4;
  if (!profile.runtimeFamily) score += 1;
  if (!profile.modelFamily) score += 1;
  return score;
}

export function resolveProviderTranslationProfile(input: ResolveProviderTranslationProfileInput) {
  const providerFamily = normalizeProviderFamily(input.providerFamily);
  const runtimeFamily = String(input.runtimeFamily || '').trim().toLowerCase();
  const inferredModelFamily =
    String(input.modelFamily || '').trim().toLowerCase()
    || String(llmModelProfileRegistry.match(String(input.modelName || '').trim())?.family || '').trim().toLowerCase();

  const candidates = PROVIDER_TRANSLATION_PROFILES
    .map((profile) => ({
      profile,
      score: scoreProviderTranslationProfile(profile, {
        providerFamily,
        runtimeFamily,
        modelFamily: inferredModelFamily,
      }),
    }))
    .filter((entry) => entry.score >= 0)
    .sort((left, right) => right.score - left.score);

  return candidates[0]?.profile || null;
}
