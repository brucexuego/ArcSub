import { LANGUAGE_POLICIES, QWEN_ASR_LANGUAGE_ALIASES, UNKNOWN_LANGUAGE_POLICY } from './policies.js';
import type { LanguagePolicy } from './types.js';

const LANGUAGE_POLICY_BY_KEY = new Map<string, LanguagePolicy>();
const LANGUAGE_POLICY_BY_ALIAS = new Map<string, LanguagePolicy>();

for (const policy of LANGUAGE_POLICIES) {
  LANGUAGE_POLICY_BY_KEY.set(policy.key, policy);
  LANGUAGE_POLICY_BY_ALIAS.set(policy.key, policy);
  for (const alias of policy.aliases) {
    LANGUAGE_POLICY_BY_ALIAS.set(alias, policy);
  }
}

export function normalizeLanguageKey(language?: string) {
  return String(language || '')
    .trim()
    .toLowerCase()
    .replace(/_/g, '-');
}

function isStructuredLocaleKey(value: string) {
  const normalized = normalizeLanguageKey(value);
  if (!normalized) return false;
  const parts = normalized.split('-').filter(Boolean);
  if (parts.length === 0) return false;
  if (!/^[a-z]{2,3}$/.test(parts[0])) return false;
  return parts.every((part, index) =>
    index === 0 ? /^[a-z]{2,3}$/.test(part) : /^[a-z0-9]{2,8}$/.test(part)
  );
}

function compareLocaleSpecificity(left: string, right: string) {
  const leftParts = left.split('-');
  const rightParts = right.split('-');
  if (leftParts.length !== rightParts.length) {
    return leftParts.length - rightParts.length;
  }

  const leftHasScript = leftParts.some((part, index) => index > 0 && /^[a-z]{4}$/.test(part));
  const rightHasScript = rightParts.some((part, index) => index > 0 && /^[a-z]{4}$/.test(part));
  if (leftHasScript !== rightHasScript) {
    return leftHasScript ? -1 : 1;
  }

  return left.localeCompare(right);
}

function inferRelatedGlossaryLocaleKeys(value: string) {
  const normalized = normalizeLanguageKey(value);
  if (!isStructuredLocaleKey(normalized)) return [];

  const parts = normalized.split('-');
  if (parts[0] !== 'zh') return [];

  const related = new Set<string>();
  if (parts.includes('hant') || parts.includes('tw') || parts.includes('hk') || parts.includes('mo')) {
    related.add('zh-hant');
  }
  if (parts.includes('hans') || parts.includes('cn') || parts.includes('sg')) {
    related.add('zh-hans');
  }

  return Array.from(related);
}

export function buildTranslationGlossaryLocaleChain(language?: string) {
  const normalized = normalizeLanguageKey(language);
  const policy = resolveLanguagePolicy(normalized);
  const localeKeys = new Set<string>();
  const preferredPrimary =
    policy.key !== 'unknown' && isStructuredLocaleKey(policy.key)
      ? policy.key.split('-')[0]
      : isStructuredLocaleKey(normalized)
        ? normalized.split('-')[0]
        : '';

  const addLocaleKey = (value?: string) => {
    const candidate = normalizeLanguageKey(value);
    if (!isStructuredLocaleKey(candidate)) return;
    localeKeys.add(candidate);
    for (const related of inferRelatedGlossaryLocaleKeys(candidate)) {
      localeKeys.add(related);
    }
  };

  addLocaleKey(normalized);
  if (policy.key !== 'unknown') {
    addLocaleKey(policy.key);
    for (const alias of policy.aliases) {
      addLocaleKey(alias);
    }
  }

  const chain: string[] = ['_shared'];
  const seen = new Set<string>(chain);
  const orderedLocaleKeys = Array.from(localeKeys).sort((left, right) => {
    const leftPrimaryMatches = preferredPrimary ? left.split('-')[0] === preferredPrimary : false;
    const rightPrimaryMatches = preferredPrimary ? right.split('-')[0] === preferredPrimary : false;
    if (leftPrimaryMatches !== rightPrimaryMatches) {
      return leftPrimaryMatches ? -1 : 1;
    }
    return compareLocaleSpecificity(left, right);
  });

  for (const localeKey of orderedLocaleKeys) {
    const parts = localeKey.split('-');
    for (let index = 1; index <= parts.length; index += 1) {
      const partial = parts.slice(0, index).join('-');
      if (!partial || seen.has(partial)) continue;
      seen.add(partial);
      chain.push(partial);
    }
  }

  return chain;
}

export function resolveLanguagePolicy(language?: string): LanguagePolicy {
  const normalized = normalizeLanguageKey(language);
  if (!normalized || normalized === 'none' || normalized === 'null') {
    return UNKNOWN_LANGUAGE_POLICY;
  }
  return LANGUAGE_POLICY_BY_ALIAS.get(normalized) || UNKNOWN_LANGUAGE_POLICY;
}

export function matchesLanguageAlias(language: string | undefined, alias: string) {
  const normalizedLanguage = normalizeLanguageKey(language);
  const normalizedAlias = normalizeLanguageKey(alias);
  if (!normalizedLanguage || !normalizedAlias) return false;

  const languagePolicy = resolveLanguagePolicy(normalizedLanguage);
  const aliasPolicy = resolveLanguagePolicy(normalizedAlias);
  if (languagePolicy.key !== 'unknown' && aliasPolicy.key !== 'unknown') {
    return languagePolicy.key === aliasPolicy.key;
  }

  return normalizedLanguage === normalizedAlias || normalizedLanguage.startsWith(`${normalizedAlias}-`);
}

export function isNoSpaceLanguage(language?: string) {
  return resolveLanguagePolicy(language).asr.noSpaceScript;
}

export function getSegmenterLocale(language?: string, fallback = 'en') {
  const locale = resolveLanguagePolicy(language).asr.segmenterLocale;
  return locale || fallback;
}

export function buildWhisperLanguageToken(language?: string) {
  const raw = String(language || '').trim();
  if (!raw) return undefined;
  if (raw.startsWith('<|') && raw.endsWith('|>')) {
    return raw;
  }

  const normalized = normalizeLanguageKey(raw);
  if (!normalized || normalized === 'auto' || normalized === 'none' || normalized === 'null') {
    return undefined;
  }

  const policy = resolveLanguagePolicy(normalized);
  const whisperCode = policy.asr.whisperCode || normalized.split('-')[0] || normalized;
  return whisperCode ? `<|${whisperCode}|>` : undefined;
}

export function resolveQwenAsrLanguageName(language?: string) {
  const normalized = normalizeLanguageKey(language);
  if (!normalized || normalized === 'auto' || normalized === 'none' || normalized === 'null') {
    return undefined;
  }

  const policy = resolveLanguagePolicy(normalized);
  return policy.asr.qwenName || QWEN_ASR_LANGUAGE_ALIASES[normalized];
}

export function normalizeTargetLanguageEnglishName(targetLang: string) {
  const policy = resolveLanguagePolicy(targetLang);
  if (policy.key !== 'unknown') return policy.englishName;
  return targetLang || UNKNOWN_LANGUAGE_POLICY.englishName;
}

export function normalizeTargetLanguageNativeName(targetLang: string) {
  const policy = resolveLanguagePolicy(targetLang);
  if (policy.key !== 'unknown') return policy.nativeName;
  return normalizeTargetLanguageEnglishName(targetLang);
}

export function buildTargetLanguageDescriptor(targetLang: string) {
  const english = normalizeTargetLanguageEnglishName(targetLang);
  const native = normalizeTargetLanguageNativeName(targetLang);
  if (!native || native === english) return english;
  return `${english} (${native})`;
}

export function getTargetLanguageInstructionLines(targetLang: string) {
  return [...resolveLanguagePolicy(targetLang).translation.instructionLines];
}

export function getTranslationStopwords(targetLang: string) {
  return new Set(resolveLanguagePolicy(targetLang).translation.stopwords);
}

export function isTraditionalChineseTarget(targetLang: string) {
  return resolveLanguagePolicy(targetLang).translation.traditionalChineseTarget;
}

export function isSimplifiedChineseTarget(targetLang: string) {
  return resolveLanguagePolicy(targetLang).translation.simplifiedChineseTarget;
}
