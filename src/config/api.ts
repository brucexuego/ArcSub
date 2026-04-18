/**
 * API Configuration Defaults
 * These are used as initial values for the application's API settings.
 * In a production environment, sensitive keys should be provided via environment variables.
 */

export const DEFAULT_ASR_MODELS = [
  { 
    id: '1', 
    name: 'OpenAI Whisper', 
    url: 'https://api.openai.com/v1/audio/transcriptions', 
    key: '',
    model: 'whisper-1' 
  }
];

export const DEFAULT_TRANSLATE_MODELS = [
  { 
    id: '1', 
    name: 'DeepL API', 
    url: 'https://api-free.deepl.com/v2/translate', 
    key: '' 
  },
  { 
    id: '2', 
    name: 'OpenAI GPT-4', 
    url: 'https://api.openai.com/v1/chat/completions', 
    key: '' 
  }
];
