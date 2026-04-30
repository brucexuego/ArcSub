import fs from 'fs-extra';
import { getLocalModelInstallDir } from '../server/local_model_catalog.js';
import { AsrService } from '../server/services/asr_service.js';
import { LocalModelService } from '../server/services/local_model_service.js';
import { SettingsManager } from '../server/services/settings_manager.js';

const DEFAULT_AUDIO_CANDIDATES = [
  'runtime/projects/cohere_ov_asr_smoke_1777386189847/assets/audio.wav',
  'runtime/projects/z56rpa3gv/assets/audio.wav',
];

async function findAudioPath() {
  const configured = String(process.env.ASR_LOCAL_SMOKE_AUDIO || '').trim();
  const candidates = configured ? [configured] : DEFAULT_AUDIO_CANDIDATES;
  for (const candidate of candidates) {
    if (await fs.pathExists(candidate)) return candidate;
  }
  return null;
}

async function main() {
  const audioPath = await findAudioPath();
  if (!audioPath) {
    console.warn('ASR local smoke skipped: no local smoke audio found.');
    return;
  }

  const settings = await SettingsManager.getSettings({ mask: false });
  const modelId = String(process.env.ASR_LOCAL_SMOKE_MODEL_ID || settings.localModels?.asrSelectedId || '').trim();
  const localModel = await LocalModelService.resolveLocalModelForRequest('asr', modelId, settings);
  if (!localModel) {
    console.warn('ASR local smoke skipped: no installed local ASR model resolved.');
    return;
  }

  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => {
    throw new Error('Unexpected cloud fetch from local ASR smoke.');
  }) as typeof fetch;

  try {
    const result = await (AsrService as any).transcribeWithProvider(audioPath, {
      useLocalProvider: true,
      localModelId: localModel.id,
      localModelRuntime: localModel.runtime,
      localModelPath: getLocalModelInstallDir(localModel),
      options: {
        language: 'auto',
        prompt: '',
        segmentation: true,
        wordAlignment: false,
        vad: false,
        diarization: false,
        decodePolicy: {
          pipelineMode: 'stable',
          alignmentStrategy: 'provider-first',
          temperature: null,
          beamSize: null,
          noSpeechThreshold: null,
          conditionOnPreviousText: null,
        },
      },
      onProgress: () => {},
    });

    const chunkCount = Array.isArray(result?.chunks) ? result.chunks.length : 0;
    const text = String(result?.text || '').trim();
    if (chunkCount === 0 || !text) {
      throw new Error('Local ASR smoke returned no transcript text.');
    }

    console.log(JSON.stringify({
      ok: true,
      audioPath,
      modelId: localModel.id,
      runtime: localModel.runtime,
      chunkCount,
      segmentCount: Array.isArray(result?.segments) ? result.segments.length : 0,
      wordCount: Array.isArray(result?.word_segments) ? result.word_segments.length : 0,
      textPreview: text.slice(0, 120),
    }));
  } finally {
    globalThis.fetch = originalFetch;
  }
}

void main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
