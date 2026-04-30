import type { CloudAsrAdapter } from '../types.js';
import { toFiniteNumber } from '../runtime/shared.js';
import { createCloudAsrProviderDefinition, parseCloudAsrUrl } from './shared.js';

const GOOGLE_CHIRP_LANGUAGE_CODE_ALIASES: Record<string, string> = {
  auto: 'auto',
  en: 'en-US',
  ja: 'ja-JP',
  zh: 'cmn-Hans-CN',
  'zh-cn': 'cmn-Hans-CN',
  'zh-hans': 'cmn-Hans-CN',
  'zh-tw': 'cmn-Hant-TW',
  'zh-hant': 'cmn-Hant-TW',
  yue: 'yue-Hant-HK',
  ko: 'ko-KR',
  fr: 'fr-FR',
  de: 'de-DE',
  es: 'es-ES',
  it: 'it-IT',
  pt: 'pt-BR',
  'pt-br': 'pt-BR',
  fi: 'fi-FI',
};

function resolveGoogleCloudLanguageCodes(language?: string) {
  const raw = String(language || '').trim();
  if (!raw) return ['auto'];
  const parts = raw
    .split(/[,;]/)
    .map((part) => part.trim())
    .filter(Boolean);
  if (parts.length === 0) return ['auto'];
  return parts.map((part) => GOOGLE_CHIRP_LANGUAGE_CODE_ALIASES[part.toLowerCase()] || part);
}

function parseGoogleDurationSeconds(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const matched = value.trim().match(/^(-?\d+(?:\.\d+)?)s$/);
    if (!matched) return undefined;
    const parsed = Number(matched[1]);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  if (value && typeof value === 'object') {
    const raw = value as { seconds?: unknown; nanos?: unknown };
    const seconds = Number(raw.seconds || 0);
    const nanos = Number(raw.nanos || 0);
    if (Number.isFinite(seconds) && Number.isFinite(nanos)) {
      return seconds + nanos / 1_000_000_000;
    }
  }
  return undefined;
}

function normalizeGoogleCloudChirp3Word(raw: any) {
  const text = String(raw?.word ?? raw?.text ?? '').trim();
  if (!text) return null;
  const start = parseGoogleDurationSeconds(raw?.startOffset ?? raw?.start ?? raw?.start_ts);
  if (start == null || !Number.isFinite(start)) return null;
  const end = parseGoogleDurationSeconds(raw?.endOffset ?? raw?.end ?? raw?.end_ts);
  const probability = toFiniteNumber(raw?.confidence, Number.NaN);
  return {
    ...raw,
    word: text,
    text,
    start,
    start_ts: start,
    end: end != null && end > start ? end : undefined,
    end_ts: end != null && end > start ? end : undefined,
    probability: Number.isFinite(probability) ? probability : undefined,
  };
}

function normalizeGoogleCloudChirp3Transcript(data: any) {
  const results = Array.isArray(data?.results) ? data.results : [];
  const segments: any[] = [];
  const texts: string[] = [];
  let previousEnd = 0;
  let detectedLanguage = '';

  for (const result of results) {
    const alternative = Array.isArray(result?.alternatives) ? result.alternatives[0] : null;
    const text = String(alternative?.transcript || '').trim();
    if (!text) continue;
    const rawWords = Array.isArray(alternative?.words) ? alternative.words : [];
    const words = rawWords
      .map((word: any) => normalizeGoogleCloudChirp3Word(word))
      .filter(Boolean);
    const firstWord = words[0];
    const lastWord = words[words.length - 1];
    const start = Number.isFinite(Number(firstWord?.start)) ? Number(firstWord.start) : previousEnd;
    const resultEnd = parseGoogleDurationSeconds(result?.resultEndOffset);
    const wordEnd = Number.isFinite(Number(lastWord?.end)) ? Number(lastWord.end) : undefined;
    const end = wordEnd != null
      ? wordEnd
      : resultEnd != null && resultEnd > start
        ? resultEnd
        : undefined;
    const languageCode = typeof result?.languageCode === 'string' ? result.languageCode.trim() : '';
    if (!detectedLanguage && languageCode) detectedLanguage = languageCode;
    texts.push(text);
    segments.push({
      text,
      transcript: text,
      start,
      start_ts: start,
      end,
      end_ts: end,
      language_code: languageCode || undefined,
      confidence: Number.isFinite(Number(alternative?.confidence)) ? Number(alternative.confidence) : undefined,
      words: words.length > 0 ? words : undefined,
    });
    previousEnd = end != null ? Math.max(previousEnd, end) : previousEnd;
  }

  return {
    ...data,
    text: texts.join('\n').trim() || data?.text,
    language_code: detectedLanguage || data?.language_code,
    segments: segments.length > 0 ? segments : data?.segments,
  };
}

const googleChirp3Runtime: CloudAsrAdapter = {
  provider: 'google-cloud-speech',
  getPreferredResponseFormats() {
    return ['json'];
  },
  getRequestHeaders() {
    return { Accept: 'application/json', 'Content-Type': 'application/json' };
  },
  buildJsonBody(input) {
    const { fileBuffer, config, options } = input;
    return {
      config: {
        autoDecodingConfig: {},
        languageCodes: resolveGoogleCloudLanguageCodes(options.language),
        model: config.model || 'chirp_3',
        features: {
          enableWordTimeOffsets: options.wordAlignment !== false,
          enableWordConfidence: options.wordAlignment !== false,
          enableAutomaticPunctuation: true,
        },
      },
      content: fileBuffer.toString('base64'),
    };
  },
  normalizeResponse(data) {
    return normalizeGoogleCloudChirp3Transcript(data);
  },
};

export const googleCloudSpeechProvider = createCloudAsrProviderDefinition({
  provider: 'google-cloud-speech',
  defaultModel: 'chirp_3',
  runtime: googleChirp3Runtime,
  detect(input) {
    return input.hostname.endsWith('speech.googleapis.com') ||
      input.pathname.includes('/recognizers/') ||
      input.modelName.includes('chirp');
  },
  buildEndpointUrl(rawUrl) {
    const parsed = parseCloudAsrUrl(rawUrl);
    if (parsed.pathname.toLowerCase().endsWith(':recognize')) return parsed.toString();
    const trimmed = parsed.pathname.replace(/\/+$/, '');
    parsed.pathname = `${trimmed}:recognize`;
    return parsed.toString();
  },
  buildHeaders(key): Record<string, string> {
    const trimmed = String(key || '').trim();
    return trimmed ? { 'x-goog-api-key': trimmed } : {};
  },
  buildConnectionProbe(input) {
    return {
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config: {
          autoDecodingConfig: {},
          languageCodes: ['en-US'],
          model: input.effectiveModel,
        },
        content: '',
      }),
      expectedValidation: 'google',
    };
  },
});
