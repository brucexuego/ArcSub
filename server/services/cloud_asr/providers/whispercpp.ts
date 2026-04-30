import path from 'node:path';
import type { CloudAsrAdapter } from '../types.js';
import { bufferToBlobPart } from '../runtime/shared.js';
import { buildBearerHeaders, createCloudAsrProviderDefinition, ensureEndpointPath, parseCloudAsrUrl } from './shared.js';

const whisperCppRuntime: CloudAsrAdapter = {
  provider: 'whispercpp-inference',
  getPreferredResponseFormats(options) {
    return options.segmentation !== false ? ['verbose_json', 'json'] : ['json'];
  },
  buildFormData(input) {
    const { filePath, fileBuffer, options, includeLanguage, responseFormat } = input;
    const blob = new Blob([bufferToBlobPart(fileBuffer)]);
    const formData = new FormData();
    formData.append('file', blob, path.basename(filePath));
    formData.append('response_format', responseFormat);
    formData.append('response-format', responseFormat);

    const rawLanguage = typeof options.language === 'string' ? options.language.trim() : '';
    if (includeLanguage && rawLanguage) {
      formData.append('language', rawLanguage);
    }
    if (options.prompt && options.prompt.trim()) {
      formData.append('prompt', options.prompt.trim());
    }
    if (options.segmentation !== false && options.wordAlignment !== false) {
      formData.append('word_timestamps', 'true');
      formData.append('timestamp_granularities[]', 'segment');
      formData.append('timestamp_granularities[]', 'word');
    }
    if (Number.isFinite(Number(options.decodePolicy?.temperature))) {
      formData.append('temperature', String(Number(options.decodePolicy?.temperature)));
    }
    if (Number.isFinite(Number(options.decodePolicy?.beamSize))) {
      formData.append('beam_size', String(Math.max(1, Math.round(Number(options.decodePolicy?.beamSize)))));
    }
    if (Number.isFinite(Number(options.decodePolicy?.noSpeechThreshold))) {
      formData.append('no_speech_thold', String(Number(options.decodePolicy?.noSpeechThreshold)));
    }
    return formData;
  },
};

export const whisperCppProvider = createCloudAsrProviderDefinition({
  provider: 'whispercpp-inference',
  defaultModel: 'whispercpp',
  runtime: whisperCppRuntime,
  detect(input) {
    return input.pathname.includes('/inference') ||
      input.modelName.includes('whisper.cpp') ||
      input.modelName.includes('whispercpp');
  },
  buildEndpointUrl(rawUrl) {
    const parsed = parseCloudAsrUrl(rawUrl);
    if (parsed.pathname.toLowerCase().endsWith('/inference')) return parsed.toString();
    return ensureEndpointPath(parsed, '/inference').toString();
  },
  buildHeaders: buildBearerHeaders,
});
