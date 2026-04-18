#!/usr/bin/env node

import process from 'node:process';

let pipeline = null;
let runtimeKind = 'llm';
let modelPath = '';
let device = 'AUTO';

function writeResponse(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

function summarizePerfMetrics(perfMetrics) {
  if (!perfMetrics || typeof perfMetrics !== 'object') return null;

  const readScalar = (getter) => {
    if (typeof getter !== 'function') return null;
    try {
      const value = getter.call(perfMetrics);
      return typeof value === 'number' && Number.isFinite(value) ? value : null;
    } catch {
      return null;
    }
  };

  const readMean = (getter) => {
    if (typeof getter !== 'function') return null;
    try {
      const value = getter.call(perfMetrics);
      return typeof value?.mean === 'number' && Number.isFinite(value.mean) ? value.mean : null;
    } catch {
      return null;
    }
  };

  return {
    loadTimeMs: readScalar(perfMetrics.getLoadTime),
    generatedTokens: readScalar(perfMetrics.getNumGeneratedTokens),
    inputTokens: readScalar(perfMetrics.getNumInputTokens),
    ttftMs: readMean(perfMetrics.getTTFT),
    tpotMs: readMean(perfMetrics.getTPOT),
    ipotMs: readMean(perfMetrics.getIPOT),
    throughputTokensPerSec: readMean(perfMetrics.getThroughput),
    generateDurationMs: readMean(perfMetrics.getGenerateDuration),
    inferenceDurationMs: readMean(perfMetrics.getInferenceDuration),
    tokenizationDurationMs: readMean(perfMetrics.getTokenizationDuration),
    detokenizationDurationMs: readMean(perfMetrics.getDetokenizationDuration),
  };
}

async function disposePipeline() {
  if (!pipeline) return;
  const current = pipeline;
  const disposer = current.dispose || current.release || current.close || current.delete;
  pipeline = null;
  if (typeof disposer === 'function') {
    await Promise.resolve(disposer.call(current));
  }
}

async function loadModel(params) {
  const nextModelPath = String(params.modelPath || '').trim();
  const nextDevice = String(params.device || 'AUTO').trim() || 'AUTO';
  const nextRuntimeKind = String(params.runtimeKind || 'llm').trim().toLowerCase();
  const properties =
    params.properties && typeof params.properties === 'object' ? { ...params.properties } : {};

  if (!nextModelPath) {
    throw new Error('modelPath is required.');
  }
  if (nextRuntimeKind !== 'llm' && nextRuntimeKind !== 'vlm') {
    throw new Error(`Unsupported runtimeKind: ${nextRuntimeKind}`);
  }

  await disposePipeline();

  const openvinoGenai = await import('openvino-genai-node');
  const Factory =
    nextRuntimeKind === 'vlm'
      ? openvinoGenai?.VLMPipeline || openvinoGenai?.default?.VLMPipeline
      : openvinoGenai?.LLMPipeline || openvinoGenai?.default?.LLMPipeline || openvinoGenai?.default;
  if (typeof Factory !== 'function') {
    throw new Error(`openvino-genai-node ${nextRuntimeKind.toUpperCase()} pipeline is unavailable.`);
  }

  pipeline = await Promise.resolve(Factory(nextModelPath, nextDevice, properties));
  modelPath = nextModelPath;
  device = typeof pipeline?.device === 'string' && pipeline.device.trim() ? pipeline.device : nextDevice;
  runtimeKind = nextRuntimeKind;
  return {
    loaded: true,
    modelPath,
    device,
    runtimeKind,
  };
}

async function generate(params) {
  if (!pipeline) {
    throw new Error('Translation pipeline is not loaded.');
  }
  const prompt = String(params.prompt || '');
  if (!prompt.trim()) {
    throw new Error('prompt is required.');
  }
  const generationConfig =
    params.generationConfig && typeof params.generationConfig === 'object'
      ? { ...params.generationConfig }
      : {};

  const result =
    runtimeKind === 'vlm'
      ? await pipeline.generate(prompt, { generationConfig })
      : await pipeline.generate(prompt, generationConfig);

  const asTexts = Array.isArray(result?.texts) ? result.texts : [];
  const text =
    typeof result === 'string'
      ? result
      : typeof result?.text === 'string'
        ? result.text
        : typeof result?.translatedText === 'string'
          ? result.translatedText
          : typeof result?.toString === 'function'
            ? result.toString()
            : '';

  return {
    texts: asTexts.length > 0 ? asTexts : text ? [text] : [],
    text: String(text || '').trim(),
    device,
    runtimeKind,
    perfMetrics: summarizePerfMetrics(result?.perfMetrics),
  };
}

function health() {
  return {
    ready: true,
    modelLoaded: Boolean(pipeline),
    modelPath,
    device,
    runtimeKind,
    nodeVersion: process.version,
  };
}

async function handleRequest(request) {
  const requestId = String(request?.requestId || '');
  const method = String(request?.method || '').trim();
  const params = request?.params && typeof request.params === 'object' ? request.params : {};
  if (!requestId) throw new Error('requestId is required.');
  if (!method) throw new Error('method is required.');

  if (method === 'health') {
    return { requestId, ok: true, result: health() };
  }
  if (method === 'load') {
    return { requestId, ok: true, result: await loadModel(params) };
  }
  if (method === 'generate') {
    return { requestId, ok: true, result: await generate(params) };
  }
  if (method === 'unload') {
    await disposePipeline();
    modelPath = '';
    device = 'AUTO';
    runtimeKind = 'llm';
    return { requestId, ok: true, result: { unloaded: true } };
  }
  if (method === 'shutdown') {
    await disposePipeline();
    writeResponse({ requestId, ok: true, result: { shuttingDown: true } });
    process.exit(0);
  }
  throw new Error(`Unknown method: ${method}`);
}

if (typeof process.stdin.setEncoding === 'function') {
  process.stdin.setEncoding('utf8');
}

let buffer = '';
process.stdin.on('data', async (chunk) => {
  buffer += String(chunk || '');
  while (buffer.includes('\n')) {
    const lineEnd = buffer.indexOf('\n');
    const rawLine = buffer.slice(0, lineEnd).trim();
    buffer = buffer.slice(lineEnd + 1);
    if (!rawLine) continue;

    let requestId = '';
    try {
      const request = JSON.parse(rawLine);
      requestId = String(request?.requestId || '');
      const response = await handleRequest(request);
      writeResponse(response);
    } catch (error) {
      writeResponse({
        requestId,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }
});

process.stdin.on('end', async () => {
  await disposePipeline().catch(() => {});
  process.exit(0);
});
