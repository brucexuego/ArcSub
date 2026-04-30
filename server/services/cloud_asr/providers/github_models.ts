import type { CloudAsrAdapter, CloudAsrAdapterRequestOptions } from '../types.js';
import { firstString, inferAudioFormat, stripCodeFence } from '../runtime/shared.js';
import { createCloudAsrProviderDefinition, parseCloudAsrUrl } from './shared.js';

function buildGitHubModelsChatEndpoint(parsed: URL) {
  const next = new URL(parsed.toString());
  const parts = next.pathname
    .split('/')
    .map((part) => part.trim())
    .filter(Boolean);
  const orgIndex = parts.findIndex((part) => part.toLowerCase() === 'orgs');
  const org = orgIndex >= 0 ? parts[orgIndex + 1] : '';
  next.pathname = org
    ? `/orgs/${encodeURIComponent(org)}/inference/chat/completions`
    : '/inference/chat/completions';
  return next.toString();
}

function buildGitHubModelsCatalogEndpoint(parsed: URL) {
  const next = new URL(parsed.toString());
  next.pathname = '/catalog/models';
  next.search = '';
  return next.toString();
}

function buildGithubPhi4AsrPrompt(options: CloudAsrAdapterRequestOptions) {
  const parts = [
    'Transcribe the audio exactly as spoken.',
    'Return only the transcript text. Do not summarize, translate, explain, add timestamps, or add labels.',
  ];
  const rawLanguage = typeof options.language === 'string' ? options.language.trim() : '';
  if (rawLanguage && rawLanguage.toLowerCase() !== 'auto') {
    parts.push(`The expected spoken language is ${rawLanguage}.`);
  }
  if (options.prompt && options.prompt.trim()) {
    parts.push(`Vocabulary and context: ${options.prompt.trim()}`);
  }
  return parts.join(' ');
}

function getGithubPhi4MaxOutputTokens() {
  const parsed = Number(process.env.ASR_PHI4_MAX_OUTPUT_TOKENS || 512);
  if (!Number.isFinite(parsed)) return 512;
  return Math.max(128, Math.min(2048, Math.round(parsed)));
}

function normalizeGithubPhi4TranscriptText(raw: string) {
  let text = stripCodeFence(raw);
  try {
    const parsed = JSON.parse(text);
    text = firstString([
      parsed?.text,
      parsed?.transcript,
      parsed?.transcription,
      parsed?.result,
      Array.isArray(parsed?.segments)
        ? parsed.segments.map((segment: any) => firstString([segment?.text, segment?.transcript])).filter(Boolean).join('\n')
        : '',
    ]) || text;
  } catch {
    // Keep plain text responses.
  }
  return String(text || '')
    .replace(/^\s*(transcript|transcription|text|output)\s*[:：]\s*/i, '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .join('\n')
    .trim();
}

function normalizeGithubPhi4Transcript(data: any) {
  const rawContent = firstString([
    data?.choices?.[0]?.message?.content,
    data?.choices?.[0]?.delta?.content,
    data?.output_text,
    data?.text,
    data?.content,
  ]);
  return {
    ...data,
    text: normalizeGithubPhi4TranscriptText(rawContent),
  };
}

const githubPhi4Runtime: CloudAsrAdapter = {
  provider: 'github-models',
  getPreferredResponseFormats() {
    return ['json'];
  },
  getRequestHeaders() {
    return { 'Content-Type': 'application/json' };
  },
  buildJsonBody(input) {
    const { fileBuffer, filePath, config, options } = input;
    const audioFormat = inferAudioFormat(filePath);
    return {
      model: String(config.model || 'microsoft/phi-4-multimodal-instruct').trim().toLowerCase(),
      messages: [
        {
          role: 'system',
          content: 'You are a precise ASR transcription engine. Output only the transcript text.',
        },
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: buildGithubPhi4AsrPrompt(options),
            },
            {
              type: 'input_audio',
              input_audio: {
                data: fileBuffer.toString('base64'),
                format: audioFormat,
              },
            },
          ],
        },
      ],
      temperature: Number.isFinite(Number(options.decodePolicy?.temperature))
        ? Number(options.decodePolicy?.temperature)
        : 0,
      top_p: 1,
      max_tokens: getGithubPhi4MaxOutputTokens(),
      stream: false,
    };
  },
  normalizeResponse(data) {
    return normalizeGithubPhi4Transcript(data);
  },
};

export const githubModelsProvider = createCloudAsrProviderDefinition({
  provider: 'github-models',
  defaultModel: 'microsoft/phi-4-multimodal-instruct',
  runtime: githubPhi4Runtime,
  preflight(input) {
    if (input.chunkingPolicy?.profileId === 'github-phi4-multimodal') {
      const proactiveChunkSec = Math.max(
        input.chunkingPolicy.minChunkSec || 10,
        input.chunkingPolicy.initialChunkSec || 30
      );
      return {
        action: 'chunk',
        chunkingPolicy: input.chunkingPolicy,
        message: `Phi-4 ASR uses ${proactiveChunkSec}s ${String(input.chunkingPolicy.audioFormat || 'mp3').toUpperCase()} chunks to stay under the provider token limit...`,
      };
    }
    return { action: 'direct' };
  },
  detect(input) {
    return input.hostname === 'models.github.ai' ||
      input.pathname.includes('/inference/chat/completions') ||
      (input.modelName.includes('github') && (input.modelName.includes('phi-4') || input.modelName.includes('phi4')));
  },
  buildEndpointUrl(rawUrl) {
    const parsed = parseCloudAsrUrl(rawUrl);
    if (parsed.pathname.toLowerCase().includes('/inference/chat/completions')) return parsed.toString();
    return buildGitHubModelsChatEndpoint(parsed);
  },
  buildHeaders(key): Record<string, string> {
    const trimmed = String(key || '').trim();
    if (!trimmed) return {};
    return {
      Accept: 'application/vnd.github+json',
      Authorization: `Bearer ${trimmed}`,
      'X-GitHub-Api-Version': '2026-03-10',
    };
  },
  buildConnectionProbe(input) {
    const parsed = parseCloudAsrUrl(input.endpointUrl);
    return {
      method: 'GET',
      url: buildGitHubModelsCatalogEndpoint(parsed),
      expectedValidation: 'generic',
      expectedCatalogModelId: input.effectiveModel,
    };
  },
});
