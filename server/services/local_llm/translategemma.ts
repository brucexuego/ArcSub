import { normalizeLanguageKey } from '../../language/resolver.js';
import type { LocalTranslatePromptStyle, LocalTranslateStructuredMessage } from './types.js';

function toTranslateGemmaLanguageCode(language?: string): string | null {
  const normalized = normalizeLanguageKey(language);
  if (!normalized) return null;

  const exactMap: Record<string, string> = {
    ar: 'ar',
    de: 'de',
    el: 'el',
    en: 'en',
    es: 'es',
    fa: 'fa',
    fi: 'fi',
    fr: 'fr',
    he: 'he',
    hi: 'hi',
    id: 'id',
    it: 'it',
    ja: 'ja',
    jp: 'ja',
    ko: 'ko',
    nl: 'nl',
    pl: 'pl',
    pt: 'pt',
    ptbr: 'pt-BR',
    'pt-br': 'pt-BR',
    ru: 'ru',
    sv: 'sv',
    th: 'th',
    tr: 'tr',
    uk: 'uk',
    vi: 'vi',
    zh: 'zh',
    'zh-cn': 'zh-CN',
    'zh-hans': 'zh-CN',
    'zh-sg': 'zh-CN',
    'zh-tw': 'zh-TW',
    'zh-hant': 'zh-TW',
    'zh-hk': 'zh-TW',
    'zh-mo': 'zh-TW',
  };

  if (exactMap[normalized]) return exactMap[normalized];

  const primary = normalized.split('-')[0];
  return exactMap[primary] || primary || null;
}

export function inferTranslateGemmaSourceLanguageCode(text: string): string {
  const sample = String(text || '');
  if (/[\u3040-\u30ff]/u.test(sample)) return 'ja';
  if (/[\uac00-\ud7af]/u.test(sample)) return 'ko';
  if (/[\u0e00-\u0e7f]/u.test(sample)) return 'th';
  if (/[\u0590-\u05ff]/u.test(sample)) return 'he';
  if (/[\u0600-\u06ff]/u.test(sample)) return 'ar';
  if (/[\u0900-\u097f]/u.test(sample)) return 'hi';
  if (/[\u0370-\u03ff]/u.test(sample)) return 'el';
  if (/[\u0400-\u04ff]/u.test(sample)) return 'ru';
  if (/[\u4e00-\u9fff]/u.test(sample)) return 'zh';
  return 'en';
}

export function buildTranslateGemmaMessages(input: {
  text: string;
  sourceLang?: string;
  targetLang: string;
  promptStyle?: LocalTranslatePromptStyle;
}): LocalTranslateStructuredMessage[] | null {
  const text = String(input.text || '').trim();
  if (!text) return null;

  const targetLangCode = toTranslateGemmaLanguageCode(input.targetLang);
  const sourceLangCode =
    toTranslateGemmaLanguageCode(input.sourceLang) || inferTranslateGemmaSourceLanguageCode(text);

  if (!sourceLangCode || !targetLangCode) return null;

  if (input.promptStyle === 'translategemma_vllm') {
    return [
      {
        role: 'user',
        content: `<<<source>>>${sourceLangCode}<<<target>>>${targetLangCode}<<<text>>>${text}`,
      },
    ];
  }

  return [
    {
      role: 'user',
      content: [
        {
          type: 'text',
          text,
          source_lang_code: sourceLangCode,
          target_lang_code: targetLangCode,
        },
      ],
    },
  ];
}
