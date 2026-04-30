import assert from 'node:assert/strict';
import fs from 'fs-extra';
import os from 'node:os';
import path from 'node:path';
import { listCloudAsrProviders, requireCloudAsrProviderDefinition } from '../server/services/cloud_asr/registry.js';
import {
  buildCloudAsrRequestHeaders,
  resolveCloudAsrProvider,
} from '../server/services/cloud_asr/resolver.js';
import { resolveCloudAsrChunkingPolicy } from '../server/services/cloud_asr/profiles/chunking.js';
import { testCloudAsrConnection } from '../server/services/cloud_asr/connection_probe.js';
import { requestCloudAsr, type CloudAsrAdapterDeps } from '../server/services/cloud_asr/runtime.js';
import { getGeminiFreeTierLimits } from '../server/services/cloud_asr/providers/google_gemini_audio.js';

function resolve(input: { url: string; modelName?: string; model?: string }) {
  return resolveCloudAsrProvider(input);
}

function makeChunkingEnv(overrides: Record<string, string | number> = {}) {
  return {
    getNumber(name: string, fallback: number, min?: number, max?: number) {
      const raw = overrides[name];
      const parsed = Number(raw);
      const value = Number.isFinite(parsed) ? parsed : fallback;
      return Math.max(min ?? Number.NEGATIVE_INFINITY, Math.min(max ?? Number.POSITIVE_INFINITY, value));
    },
    getString(name: string, fallback: string) {
      const raw = overrides[name];
      return typeof raw === 'string' && raw.trim() ? raw.trim() : fallback;
    },
    sanitizeAudioBitrate(value: string, fallback = '32k') {
      const normalized = String(value || '').trim().toLowerCase();
      return /^\d{2,3}k$/.test(normalized) ? normalized : fallback;
    },
  };
}

async function withMockFetch(response: Response, run: () => Promise<void>) {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => response.clone()) as typeof fetch;
  try {
    await run();
  } finally {
    globalThis.fetch = originalFetch;
  }
}

async function withMockFetchSequence(responses: Response[], run: () => Promise<void>) {
  const originalFetch = globalThis.fetch;
  let index = 0;
  globalThis.fetch = (async () => {
    const response = responses[index];
    index += 1;
    if (!response) {
      throw new Error(`Unexpected fetch call ${index}.`);
    }
    return response.clone();
  }) as typeof fetch;
  try {
    await run();
    assert.equal(index, responses.length);
  } finally {
    globalThis.fetch = originalFetch;
  }
}

function makeCloudAsrDeps(): CloudAsrAdapterDeps {
  return {
    createAbortSignalWithTimeout(timeoutMs: number, signal?: AbortSignal) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), Math.max(0, timeoutMs));
      const combinedSignal = signal && typeof AbortSignal.any === 'function'
        ? AbortSignal.any([signal, controller.signal])
        : signal || controller.signal;
      return {
        signal: combinedSignal,
        dispose() {
          clearTimeout(timeout);
        },
      };
    },
    extractStructuredTranscript() {
      return { chunks: [], segments: [], word_segments: [] };
    },
    disableWordAlignment(transcript) {
      return transcript;
    },
  };
}

function makeSuccessfulCloudAsrDeps(): CloudAsrAdapterDeps {
  return {
    ...makeCloudAsrDeps(),
    extractStructuredTranscript(data) {
      const text = String((data as any)?.text || 'ok').trim() || 'ok';
      return {
        text,
        chunks: [{ start_ts: 0, end_ts: 1, text }],
        segments: [{ start_ts: 0, end_ts: 1, text }],
        word_segments: [],
      };
    },
  };
}

async function withTempAudioFile(run: (audioPath: string) => Promise<void>) {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'arcsub-asr-provider-'));
  const audioPath = path.join(tempDir, 'tiny.wav');
  try {
    await fs.writeFile(audioPath, Buffer.from('not-a-real-wav'));
    await run(audioPath);
  } finally {
    await fs.remove(tempDir);
  }
}

async function withEnv(overrides: Record<string, string | undefined>, run: () => Promise<void>) {
  const originals = new Map<string, string | undefined>();
  for (const key of Object.keys(overrides)) {
    originals.set(key, process.env[key]);
  }
  try {
    for (const [key, value] of Object.entries(overrides)) {
      if (value == null) {
        delete process.env[key];
      } else {
        process.env[key] = value;
      }
    }
    await run();
  } finally {
    for (const [key, value] of originals.entries()) {
      if (value == null) {
        delete process.env[key];
      } else {
        process.env[key] = value;
      }
    }
  }
}

async function assertCommonRuntimeObjectErrorIsReadable(input: {
  url: string;
  model?: string;
  name?: string;
  status: number;
  body: unknown;
  expected: RegExp[];
}) {
  await withTempAudioFile(async (audioPath) => {
    const resolvedProvider = resolve({ url: input.url, modelName: input.name, model: input.model });
    await withMockFetch(
      new Response(JSON.stringify(input.body), {
        status: input.status,
        headers: { 'Content-Type': 'application/json' },
      }),
      async () => {
        await assert.rejects(
          () => requestCloudAsr(
            audioPath,
            resolvedProvider,
            {
              key: 'bad',
              model: input.model,
              url: resolvedProvider.endpointUrl,
            },
            {
              language: 'auto',
              segmentation: true,
              wordAlignment: false,
            },
            makeCloudAsrDeps()
          ),
          (error) => {
            const message = error instanceof Error ? error.message : String(error);
            assert.match(message, new RegExp(`Cloud ASR error \\(${input.status}\\)`));
            for (const pattern of input.expected) {
              assert.match(message, pattern);
            }
            assert.doesNotMatch(message, /\[object Object\]/);
            return true;
          }
        );
      }
    );
  });
}

async function assertElevenLabsObjectErrorIsReadable() {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'arcsub-asr-provider-'));
  const audioPath = path.join(tempDir, 'tiny.wav');
  try {
    await fs.writeFile(audioPath, Buffer.from('not-a-real-wav'));
    const resolvedProvider = resolve({ url: 'https://api.elevenlabs.io/v1/speech-to-text', model: 'scribe_v2' });
    await withMockFetch(
      new Response(JSON.stringify({
        detail: {
          status: 'quota_exceeded',
          message: 'Not enough credits',
        },
      }), { status: 401 }),
      async () => {
        await assert.rejects(
          () => requestCloudAsr(
            audioPath,
            resolvedProvider,
            {
              key: 'bad',
              model: 'scribe_v2',
              url: resolvedProvider.endpointUrl,
            },
            {
              language: 'auto',
              segmentation: true,
              wordAlignment: false,
            },
            makeCloudAsrDeps()
          ),
          (error) => {
            const message = error instanceof Error ? error.message : String(error);
            assert.match(message, /Cloud ASR error \(401\)/);
            assert.match(message, /Not enough credits/);
            assert.match(message, /quota_exceeded/);
            assert.doesNotMatch(message, /\[object Object\]/);
            return true;
          }
        );
      }
    );
  } finally {
    await fs.remove(tempDir);
  }
}

async function assertDeepgramObjectErrorIsReadable() {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'arcsub-asr-provider-'));
  const audioPath = path.join(tempDir, 'tiny.wav');
  try {
    await fs.writeFile(audioPath, Buffer.from('not-a-real-wav'));
    const resolvedProvider = resolve({ url: 'https://api.deepgram.com/v1/listen', model: 'nova-3' });
    await withMockFetch(
      new Response(JSON.stringify({
        err_code: 'INVALID_AUTH',
        err_msg: 'Invalid API key',
      }), { status: 401 }),
      async () => {
        await assert.rejects(
          () => requestCloudAsr(
            audioPath,
            resolvedProvider,
            {
              key: 'bad',
              model: 'nova-3',
              url: resolvedProvider.endpointUrl,
            },
            {
              language: 'auto',
              segmentation: true,
              wordAlignment: false,
            },
            makeCloudAsrDeps()
          ),
          (error) => {
            const message = error instanceof Error ? error.message : String(error);
            assert.match(message, /Cloud ASR error \(401\)/);
            assert.match(message, /Invalid API key/);
            assert.match(message, /INVALID_AUTH/);
            assert.doesNotMatch(message, /\[object Object\]/);
            return true;
          }
        );
      }
    );
  } finally {
    await fs.remove(tempDir);
  }
}

async function assertGladiaJobObjectErrorIsReadable() {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'arcsub-asr-provider-'));
  const audioPath = path.join(tempDir, 'tiny.wav');
  try {
    await fs.writeFile(audioPath, Buffer.from('not-a-real-wav'));
    const resolvedProvider = resolve({ url: 'https://api.gladia.io/v2/pre-recorded', model: 'gladia-v2' });
    await withMockFetchSequence(
      [
        new Response(JSON.stringify({ audio_url: 'https://uploads.example/audio.wav' }), { status: 200 }),
        new Response(JSON.stringify({ id: 'job-1' }), { status: 200 }),
        new Response(JSON.stringify({
          status: 'failed',
          error_code: 422,
          error: {
            message: 'Unsupported audio codec',
            code: 'invalid_audio',
          },
        }), { status: 200 }),
      ],
      async () => {
        await assert.rejects(
          () => requestCloudAsr(
            audioPath,
            resolvedProvider,
            {
              key: 'bad',
              model: 'gladia-v2',
              url: resolvedProvider.endpointUrl,
            },
            {
              language: 'auto',
              segmentation: true,
              wordAlignment: false,
            },
            makeCloudAsrDeps()
          ),
          (error) => {
            const message = error instanceof Error ? error.message : String(error);
            assert.match(message, /Cloud ASR error \(422\)/);
            assert.match(message, /Unsupported audio codec/);
            assert.match(message, /invalid_audio/);
            assert.doesNotMatch(message, /\[object Object\]/);
            return true;
          }
        );
      }
    );
  } finally {
    await fs.remove(tempDir);
  }
}

async function assertGeminiLimiterDoesNotApplyToWhisperCpp() {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'arcsub-asr-provider-'));
  const audioPath = path.join(tempDir, 'tiny.wav');
  try {
    await fs.writeFile(audioPath, Buffer.from('not-a-real-wav'));
    const resolvedProvider = resolve({ url: 'https://host.local/inference', model: 'whispercpp' });
    const responses = Array.from({ length: 21 }, () =>
      new Response(JSON.stringify({ text: 'ok' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    );
    await withMockFetchSequence(responses, async () => {
      for (let i = 0; i < responses.length; i += 1) {
        const result = await requestCloudAsr(
          audioPath,
          resolvedProvider,
          {
            key: '',
            model: 'whispercpp',
            url: resolvedProvider.endpointUrl,
          },
          {
            language: 'auto',
            segmentation: true,
            wordAlignment: false,
          },
          makeSuccessfulCloudAsrDeps()
        );
        assert.equal(result.meta.provider, 'whispercpp-inference');
        assert.equal(result.meta.geminiFreeTierLimiter, null);
      }
    });
  } finally {
    await fs.remove(tempDir);
  }
}

async function assertGeminiLimiterEnvOverrides() {
  await withEnv(
    {
      ASR_GEMINI_FREE_TIER_LIMITER_ENABLED: '0',
      ASR_GEMINI_FREE_TIER_RPM: '15',
      ASR_GEMINI_FREE_TIER_TPM: '250000',
      ASR_GEMINI_FREE_TIER_RPD: '500',
    },
    async () => {
      assert.deepEqual(getGeminiFreeTierLimits(), {
        enabled: false,
        rpm: 15,
        tpm: 250_000,
        rpd: 500,
      });
    }
  );

  await withEnv(
    {
      ASR_GEMINI_FREE_TIER_LIMITER_ENABLED: '1',
      ASR_GEMINI_FREE_TIER_RPM: '0',
      ASR_GEMINI_FREE_TIER_TPM: '-1',
      ASR_GEMINI_FREE_TIER_RPD: 'not-a-number',
    },
    async () => {
      assert.deepEqual(getGeminiFreeTierLimits(), {
        enabled: true,
        rpm: 1,
        tpm: 1,
        rpd: 20,
      });
    }
  );
}

async function assertGeminiLimiterDisabledSkipsLocalQuotaBlock() {
  await withEnv(
    {
      ASR_GEMINI_FREE_TIER_LIMITER_ENABLED: '0',
      ASR_GEMINI_FREE_TIER_RPM: '1',
      ASR_GEMINI_FREE_TIER_TPM: '1',
      ASR_GEMINI_FREE_TIER_RPD: '1',
    },
    async () => {
      await withTempAudioFile(async (audioPath) => {
        const resolvedProvider = resolve({ url: 'https://generativelanguage.googleapis.com', model: 'gemini-3.1-flash-lite-preview' });
        await withMockFetch(
          new Response(JSON.stringify({
            candidates: [
              {
                content: {
                  parts: [
                    {
                      text: JSON.stringify({
                        segments: [{ start: 0, end: 1, text: 'ok' }],
                      }),
                    },
                  ],
                },
              },
            ],
          }), { status: 200, headers: { 'Content-Type': 'application/json' } }),
          async () => {
            const result = await requestCloudAsr(
              audioPath,
              resolvedProvider,
              {
                key: 'test',
                model: 'gemini-3.1-flash-lite-preview',
                url: resolvedProvider.endpointUrl,
              },
              {
                language: 'auto',
                segmentation: true,
                wordAlignment: false,
                audioDurationSec: 120,
              },
              makeSuccessfulCloudAsrDeps()
            );
            assert.equal(result.meta.provider, 'google-gemini');
            assert.equal(result.meta.geminiFreeTierLimiter?.applied, false);
            assert.equal(result.meta.geminiFreeTierLimiter?.enabled, false);
            assert.equal(result.meta.geminiFreeTierLimiter?.tpm, 1);
          }
        );
      });
    }
  );
}

async function main() {
  for (const provider of listCloudAsrProviders()) {
    assert.equal(typeof provider.provider, 'string', 'provider key is required');
    assert.equal(typeof provider.defaultModel, 'string', `${provider.provider} defaultModel is required`);
    assert.equal(typeof provider.detect, 'function', `${provider.provider} detect hook is required`);
    assert.equal(typeof provider.buildEndpointUrl, 'function', `${provider.provider} endpoint hook is required`);
    assert.equal(typeof provider.buildHeaders, 'function', `${provider.provider} header hook is required`);
    assert.ok(provider.capabilities, `${provider.provider} capabilities are required`);
    assert.ok(provider.runtime || provider.request, `${provider.provider} must expose runtime or request`);
  }

  assert.equal(resolve({ url: 'https://api.openai.com' }).provider, 'openai-whisper');
  assert.equal(resolve({ url: 'https://host.local/inference' }).provider, 'whispercpp-inference');
  assert.equal(resolve({ url: 'https://api.elevenlabs.io', modelName: 'Scribe v2' }).provider, 'elevenlabs');
  assert.equal(resolve({ url: 'https://api.deepgram.com' }).provider, 'deepgram');
  assert.equal(resolve({ url: 'https://api.gladia.io/v2/upload' }).provider, 'gladia');
  assert.equal(resolve({ url: 'https://models.github.ai' }).provider, 'github-models');
  assert.equal(
    resolve({ url: 'https://models.github.ai/inference', model: 'microsoft/Phi-4-multimodal-instruct' }).effectiveModel,
    'microsoft/phi-4-multimodal-instruct'
  );
  assert.equal(
    resolve({ url: 'https://us-speech.googleapis.com/v2/projects/p/locations/us/recognizers/_' }).endpointUrl,
    'https://us-speech.googleapis.com/v2/projects/p/locations/us/recognizers/_:recognize'
  );
  assert.equal(resolve({ url: 'https://generativelanguage.googleapis.com', model: 'gemini-2.5-flash' }).provider, 'google-gemini');

  assert.equal(
    resolve({ url: 'https://api.openai.com' }).endpointUrl,
    'https://api.openai.com/v1/audio/transcriptions'
  );
  assert.equal(
    resolve({ url: 'https://api.deepgram.com' }).endpointUrl,
    'https://api.deepgram.com/v1/listen'
  );
  assert.equal(
    resolve({ url: 'https://api.gladia.io/v2' }).endpointUrl,
    'https://api.gladia.io/v2/pre-recorded'
  );
  assert.equal(
    resolve({ url: 'https://generativelanguage.googleapis.com', model: 'gemini-2.5-flash' }).endpointUrl,
    'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent'
  );

  assert.deepEqual(buildCloudAsrRequestHeaders('elevenlabs', 'abc'), { 'xi-api-key': 'abc' });
  assert.deepEqual(buildCloudAsrRequestHeaders('deepgram', 'abc'), { Authorization: 'Token abc' });
  assert.equal(buildCloudAsrRequestHeaders('google-gemini', 'abc')['x-goog-api-key'], 'abc');

  const eleven = resolve({ url: 'https://api.elevenlabs.io' });
  assert.equal(eleven.capabilities.nativeWordTimestamps, true);
  assert.equal(eleven.capabilities.nativeDiarization, true);
  assert.equal(eleven.capabilities.bypassesLocalAdvancedProcessing, true);
  const gemini = resolve({ url: 'https://generativelanguage.googleapis.com', model: 'gemini-2.5-flash' });
  assert.equal(gemini.capabilities.supportsBatchAudio, true);
  assert.equal(gemini.capabilities.requiresVadTimestamping, true);

  const phi4Policy = resolveCloudAsrChunkingPolicy(resolve({
    url: 'https://models.github.ai/inference',
    model: 'microsoft/phi-4-multimodal-instruct',
  }), makeChunkingEnv({
    ASR_PHI4_FILE_LIMIT_CHUNK_SEC: 6,
    ASR_PHI4_FILE_LIMIT_MIN_CHUNK_SEC: 3,
    ASR_PHI4_AUDIO_BITRATE: '24k',
  }));
  assert.equal(phi4Policy?.reason, 'github_phi4_8000_token_limit');
  assert.equal(phi4Policy?.profileId, 'github-phi4-multimodal');
  assert.equal(phi4Policy?.initialChunkSec, 6);
  assert.equal(phi4Policy?.audioFormat, 'mp3');
  assert.equal(phi4Policy?.audioBitrate, '24k');
  assert.equal(resolveCloudAsrChunkingPolicy(resolve({
    url: 'https://models.github.ai/inference',
    model: 'openai/gpt-4.1',
  }), makeChunkingEnv()), null);

  const githubDef = requireCloudAsrProviderDefinition('github-models');
  const githubProbe = githubDef.buildConnectionProbe?.(resolve({ url: 'https://models.github.ai' }));
  assert.equal(githubProbe?.method, 'GET');
  assert.equal(githubProbe?.url, 'https://models.github.ai/catalog/models');
  assert.equal(githubProbe?.expectedCatalogModelId, 'microsoft/phi-4-multimodal-instruct');
  const githubPreflight = await githubDef.preflight?.({
    filePath: 'unused.wav',
    resolvedProvider: resolve({ url: 'https://models.github.ai/inference', model: 'microsoft/phi-4-multimodal-instruct' }),
    config: {},
    options: {},
    chunkingPolicy: phi4Policy,
    getFileInfo: async () => ({ sizeBytes: 1, durationSec: 1 }),
  });
  assert.equal(githubPreflight?.action, 'chunk');

  const deepgramDef = requireCloudAsrProviderDefinition('deepgram');
  const deepgramProbe = deepgramDef.buildConnectionProbe?.(resolve({ url: 'https://api.deepgram.com', model: 'nova-3' }));
  assert.ok(String(deepgramProbe?.url || '').includes('model=nova-3'));
  assert.ok(deepgramProbe?.body instanceof Blob);
  const deepgramPreflight = await deepgramDef.preflight?.({
    filePath: 'unused.wav',
    resolvedProvider: resolve({ url: 'https://api.deepgram.com', model: 'nova-3' }),
    config: {},
    options: {},
    chunkingPolicy: resolveCloudAsrChunkingPolicy('deepgram', makeChunkingEnv()),
    getFileInfo: async () => ({ sizeBytes: 2 * 1024 * 1024 * 1024 + 1, durationSec: 1 }),
  });
  assert.equal(deepgramPreflight?.action, 'reject');

  const geminiDef = requireCloudAsrProviderDefinition('google-gemini');
  const geminiProbe = geminiDef.buildConnectionProbe?.(resolve({ url: 'https://generativelanguage.googleapis.com', model: 'gemini-3.1-flash-lite-preview' }));
  assert.equal(geminiProbe?.timeoutMs, 18_000);

  const gladiaDef = requireCloudAsrProviderDefinition('gladia');
  const gladiaPolicy = resolveCloudAsrChunkingPolicy('gladia', makeChunkingEnv());
  const gladiaPreflight = await gladiaDef.preflight?.({
    filePath: 'unused.wav',
    resolvedProvider: resolve({ url: 'https://api.gladia.io/v2/pre-recorded', model: 'gladia-v2' }),
    config: {},
    options: {},
    chunkingPolicy: gladiaPolicy,
    getFileInfo: async () => ({ sizeBytes: 960 * 1024 * 1024, durationSec: 1 }),
  });
  assert.equal(gladiaPreflight?.action, 'chunk');

  await withMockFetch(
    new Response(JSON.stringify({ error: { message: 'missing required audio_url' } }), { status: 400 }),
    async () => {
      const result = await testCloudAsrConnection({
        url: 'https://api.gladia.io/v2/pre-recorded',
        key: 'test',
        model: 'gladia-v2',
        name: 'Gladia',
      });
      assert.equal(result.success, true);
    }
  );

  await withMockFetch(
    new Response(JSON.stringify({ error: { message: 'invalid api key' } }), { status: 401 }),
    async () => {
      const result = await testCloudAsrConnection({
        url: 'https://api.openai.com/v1/audio/transcriptions',
        key: 'bad',
        model: 'whisper-1',
        name: 'OpenAI',
      });
      assert.equal(result.success, false);
      assert.match(result.error || '', /invalid api key|Authentication failed/i);
    }
  );

  await withMockFetch(
    new Response(JSON.stringify([
      { id: 'openai/gpt-4.1' },
      { id: 'microsoft/phi-4-multimodal-instruct' },
    ]), { status: 200, headers: { 'Content-Type': 'application/json' } }),
    async () => {
      const result = await testCloudAsrConnection({
        url: 'https://models.github.ai/inference',
        key: 'ok',
        model: 'microsoft/Phi-4-multimodal-instruct',
        name: 'Phi 4 multimodal instruct',
      });
      assert.equal(result.success, true);
    }
  );

  await withMockFetch(
    new Response(JSON.stringify([{ id: 'openai/gpt-4.1' }]), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }),
    async () => {
      const result = await testCloudAsrConnection({
        url: 'https://models.github.ai/inference',
        key: 'ok',
        model: 'microsoft/Phi-4-multimodal-instruct',
        name: 'Phi 4 multimodal instruct',
      });
      assert.equal(result.success, false);
      assert.match(result.error || '', /Model not found/);
      assert.match(result.error || '', /microsoft\/phi-4-multimodal-instruct/);
    }
  );

  await withMockFetch(
    new Response(JSON.stringify({
      detail: {
        status: 'quota_exceeded',
        message: 'Not enough credits for this request',
      },
    }), {
      status: 401,
      headers: { 'Content-Type': 'application/json' },
    }),
    async () => {
      const result = await testCloudAsrConnection({
        url: 'https://api.elevenlabs.io/v1/speech-to-text',
        key: 'bad',
        model: 'scribe_v2',
        name: 'ElevenLabs',
      });
      assert.equal(result.success, false);
      assert.match(result.error || '', /Not enough credits/);
      assert.match(result.error || '', /quota_exceeded/);
      assert.doesNotMatch(result.error || '', /\[object Object\]/);
    }
  );

  await withMockFetch(
    new Response(JSON.stringify({ ok: true }), { status: 200 }),
    async () => {
      const result = await testCloudAsrConnection({
        url: 'https://generativelanguage.googleapis.com',
        key: 'ok',
        model: 'gemini-2.5-flash',
        name: 'Gemini audio',
      });
      assert.equal(result.success, true);
    }
  );

  await assertCommonRuntimeObjectErrorIsReadable({
    url: 'https://api.openai.com/v1/audio/transcriptions',
    model: 'whisper-1',
    status: 401,
    body: { error: { message: 'Invalid API key', code: 'invalid_api_key' } },
    expected: [/Invalid API key/, /invalid_api_key/],
  });
  await assertCommonRuntimeObjectErrorIsReadable({
    url: 'http://localhost:8080/inference',
    status: 400,
    body: { detail: { message: 'Model is not loaded', code: 'model_unavailable' } },
    expected: [/Model is not loaded/, /model_unavailable/],
  });
  await assertCommonRuntimeObjectErrorIsReadable({
    url: 'https://models.github.ai/inference',
    model: 'microsoft/Phi-4-multimodal-instruct',
    status: 403,
    body: { error: { message: 'Model access denied', code: 'forbidden' } },
    expected: [/Model access denied/, /forbidden/],
  });
  await assertCommonRuntimeObjectErrorIsReadable({
    url: 'https://us-speech.googleapis.com/v2/projects/p/locations/us/recognizers/_:recognize',
    model: 'chirp_3',
    status: 403,
    body: { error: { message: 'Permission denied', status: 'PERMISSION_DENIED' } },
    expected: [/Permission denied/, /PERMISSION_DENIED/],
  });
  await assertCommonRuntimeObjectErrorIsReadable({
    url: 'https://generativelanguage.googleapis.com',
    model: 'gemini-2.5-flash',
    status: 400,
    body: { error: { message: 'Quota exceeded', status: 'RESOURCE_EXHAUSTED' } },
    expected: [/Quota exceeded/, /RESOURCE_EXHAUSTED/],
  });

  await assertElevenLabsObjectErrorIsReadable();
  await assertDeepgramObjectErrorIsReadable();
  await assertGladiaJobObjectErrorIsReadable();
  await assertGeminiLimiterDoesNotApplyToWhisperCpp();
  await assertGeminiLimiterEnvOverrides();
  await assertGeminiLimiterDisabledSkipsLocalQuotaBlock();

  console.log('ASR provider modularization checks passed.');
}

void main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
