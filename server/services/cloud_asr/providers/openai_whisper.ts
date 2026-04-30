import path from 'node:path';
import type { CloudAsrAdapter } from '../types.js';
import { bufferToBlobPart } from '../runtime/shared.js';
import { buildBearerHeaders, createCloudAsrProviderDefinition, ensureEndpointPath, parseCloudAsrUrl } from './shared.js';

const openAiWhisperRuntime: CloudAsrAdapter = {
  provider: 'openai-whisper',
  getPreferredResponseFormats(options) {
    return [options.segmentation !== false ? 'verbose_json' : 'json'];
  },
  buildFormData(input) {
    const { filePath, fileBuffer, config, options, includeLanguage, responseFormat } = input;
    const blob = new Blob([bufferToBlobPart(fileBuffer)]);
    const formData = new FormData();
    formData.append('file', blob, path.basename(filePath));
    formData.append('model', config.model || 'whisper-1');
    formData.append('response_format', responseFormat);

    const rawLanguage = typeof options.language === 'string' ? options.language.trim() : '';
    if (includeLanguage && rawLanguage) {
      formData.append('language', rawLanguage);
    }
    if (options.prompt && options.prompt.trim()) {
      formData.append('prompt', options.prompt.trim());
    }
    if (Number.isFinite(Number(options.decodePolicy?.temperature))) {
      formData.append('temperature', String(Number(options.decodePolicy?.temperature)));
    }
    return formData;
  },
};

export const openAiWhisperProvider = createCloudAsrProviderDefinition({
  provider: 'openai-whisper',
  defaultModel: 'whisper-1',
  runtime: openAiWhisperRuntime,
  detect() {
    return false;
  },
  buildEndpointUrl(rawUrl) {
    const parsed = parseCloudAsrUrl(rawUrl);
    if (parsed.pathname.toLowerCase().includes('/audio/transcriptions')) return parsed.toString();
    return ensureEndpointPath(parsed, '/v1/audio/transcriptions').toString();
  },
  buildHeaders: buildBearerHeaders,
});
