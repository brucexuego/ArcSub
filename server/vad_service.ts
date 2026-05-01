import * as ort from "onnxruntime-node";
import { spawn } from "child_process";
import { createRequire } from "module";
import path from "path";
import fs from "fs-extra";
import { PathManager } from "./path_manager.js";
import { OpenvinoBackend } from "./openvino_backend.js";
import { resolveToolCommand } from "./runtime_tools.js";
import { ResourceManager } from "./services/resource_manager.js";

export interface SpeechSegment {
  start: number;
  end: number;
}

type VadEngine = "silero" | "ten" | "auto" | "compare";

const require = createRequire(import.meta.url);

export class VadService {
  private static compiledModel: any | null = null;
  private static session: ort.InferenceSession | null = null;
  private static tenVadModule: any | null = null;
  private static backend: "openvino" | "onnxruntime" | "ten-onnx" | null = null;
  private static modelPath = path.join(PathManager.getModelsPath(), "silero_vad.onnx");
  private static tenModelPath = path.join(PathManager.getModelsPath(), "ten-vad.onnx");
  private static readonly sampleRate = 16000;
  private static readonly adaptivePauseHistoryMax = 50;

  private static getEnvNumber(name: string, fallback: number, min?: number, max?: number) {
    const raw = process.env[name];
    if (typeof raw !== "string" || !raw.trim()) return fallback;

    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) return fallback;
    if (typeof min === "number" && parsed < min) return min;
    if (typeof max === "number" && parsed > max) return max;
    return parsed;
  }

  private static getConfig() {
    const frameSizeRaw = Math.round(this.getEnvNumber("VAD_FRAME_SIZE", 512, 64, 2048));
    const frameSize = frameSizeRaw % 2 === 0 ? frameSizeRaw : frameSizeRaw + 1;
    const startThreshold = this.getEnvNumber("VAD_START_THRESHOLD", 0.5, 0.05, 0.95);
    const endThresholdRaw = this.getEnvNumber("VAD_END_THRESHOLD", 0.4, 0.01, 0.9);
    const silenceModeRaw = String(process.env.VAD_SILENCE_MODE ?? "auto").trim().toLowerCase();
    const silenceMode: "auto" | "fixed" = silenceModeRaw === "fixed" ? "fixed" : "auto";
    const minSilenceMs = this.getEnvNumber("VAD_MIN_SILENCE_MS", 800, 30, 5000);
    const fixedSilenceMs = this.getEnvNumber("VAD_FIXED_SILENCE_MS", 800, 30, 5000);
    const initialSilenceMs = silenceMode === "fixed" ? fixedSilenceMs : minSilenceMs;

    return {
      frameSize,
      startThreshold,
      endThreshold: Math.min(endThresholdRaw, startThreshold - 0.01),
      minSpeechMs: this.getEnvNumber("VAD_MIN_SPEECH_MS", 1000, 30, 30000),
      minSilenceMs,
      fixedSilenceMs,
      initialSilenceMs,
      silenceMode,
      preSpeechMs: this.getEnvNumber("VAD_PRE_SPEECH_MS", 96, 0, 2000),
      adaptiveMinSilenceMs: this.getEnvNumber("VAD_ADAPTIVE_MIN_SILENCE_MS", 300, 50, 5000),
      adaptiveMaxSilenceMs: this.getEnvNumber("VAD_ADAPTIVE_MAX_SILENCE_MS", 2000, 100, 10000),
      progressiveTier1Sec: this.getEnvNumber("VAD_PROGRESSIVE_TIER1_SEC", 3, 0.1, 120),
      progressiveTier2Sec: this.getEnvNumber("VAD_PROGRESSIVE_TIER2_SEC", 6, 0.1, 300),
      progressiveTier3Sec: this.getEnvNumber("VAD_PROGRESSIVE_TIER3_SEC", 10, 0.1, 600),
      padMs: this.getEnvNumber("VAD_PAD_MS", 120, 0, 1000),
      mergeGapMs: this.getEnvNumber("VAD_MERGE_GAP_MS", 250, 0, 3000),
      logStats: String(process.env.VAD_LOG_STATS ?? "1").toLowerCase() !== "0",
    };
  }

  private static getEngine(): VadEngine {
    const raw = String(process.env.VAD_ENGINE ?? "silero").trim().toLowerCase();
    if (raw === "ten" || raw === "auto" || raw === "compare") return raw;
    return "silero";
  }

  private static getFfmpegBinary() {
    return resolveToolCommand("ffmpeg");
  }

  private static normalizeSamplesForVad(samples: Float32Array) {
    let peak = 0;
    for (let i = 0; i < samples.length; i += 1) {
      const abs = Math.abs(samples[i]);
      if (abs > peak) peak = abs;
    }
    if (!Number.isFinite(peak) || peak <= 0) {
      return { samples, gain: 1, peak: 0 };
    }
    const gain = this.clamp(0.9 / peak, 1, 8);
    if (gain <= 1.01) {
      return { samples, gain: 1, peak };
    }

    const normalized = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i += 1) {
      normalized[i] = this.clamp(samples[i] * gain, -1, 1);
    }
    return { samples: normalized, gain, peak };
  }

  private static async getOpenvinoModelPath() {
    const envPath = process.env.VAD_OPENVINO_MODEL_PATH?.trim();
    if (envPath) {
      const resolved = path.isAbsolute(envPath) ? envPath : path.resolve(PathManager.getRoot(), envPath);
      if (await fs.pathExists(resolved)) {
        return resolved;
      }
      console.warn(`[VAD] VAD_OPENVINO_MODEL_PATH not found: ${resolved}`);
    }

    const xmlCandidate = this.modelPath.replace(/\.onnx$/i, ".xml");
    if (xmlCandidate !== this.modelPath && (await fs.pathExists(xmlCandidate))) {
      return xmlCandidate;
    }

    return null;
  }

  private static async getTenModelPath(downloadIfMissing: boolean) {
    const envPath = process.env.TEN_VAD_MODEL_PATH?.trim();
    const resolved = envPath
      ? (path.isAbsolute(envPath) ? envPath : path.resolve(PathManager.getRoot(), envPath))
      : this.tenModelPath;

    if (await fs.pathExists(resolved)) {
      return resolved;
    }

    if (!downloadIfMissing) {
      return null;
    }

    try {
      await ResourceManager.ensureBaselineAsset("ten-vad");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.warn(`[VAD] Could not download Ten VAD model: ${message}`);
    }

    if (await fs.pathExists(resolved)) {
      return resolved;
    }

    if (resolved !== this.tenModelPath && (await fs.pathExists(this.tenModelPath))) {
      return this.tenModelPath;
    }

    return null;
  }

  private static clamp(value: number, min: number, max: number) {
    return Math.min(Math.max(value, min), max);
  }

  private static toProbability(value: unknown) {
    const num = Number(value);
    if (!Number.isFinite(num)) return 0;
    return this.clamp(num, 0, 1);
  }

  private static pushSegment(
    segments: SpeechSegment[],
    startSec: number,
    endSec: number,
    totalSec: number,
    minSpeechSec: number,
    padSec: number
  ) {
    const paddedStart = this.clamp(startSec - padSec, 0, totalSec);
    const paddedEnd = this.clamp(endSec + padSec, 0, totalSec);
    if (paddedEnd - paddedStart < minSpeechSec) return;
    segments.push({ start: paddedStart, end: paddedEnd });
  }

  private static mergeAdjacentSegments(segments: SpeechSegment[], mergeGapSec: number) {
    if (segments.length < 2) return segments;

    const merged: SpeechSegment[] = [segments[0]];
    for (let i = 1; i < segments.length; i += 1) {
      const prev = merged[merged.length - 1];
      const current = segments[i];

      if (current.start - prev.end <= mergeGapSec) {
        prev.end = Math.max(prev.end, current.end);
      } else {
        merged.push(current);
      }
    }

    return merged;
  }

  private static getProgressiveSilenceMultiplier(
    speechDurationSec: number,
    config: {
      progressiveTier1Sec: number;
      progressiveTier2Sec: number;
      progressiveTier3Sec: number;
    }
  ) {
    let multiplier = 1;
    if (speechDurationSec >= config.progressiveTier1Sec) multiplier = 1;
    if (speechDurationSec >= config.progressiveTier2Sec) multiplier = 0.5;
    if (speechDurationSec >= config.progressiveTier3Sec) multiplier = 0.25;
    return multiplier;
  }

  private static clampAdaptiveSilenceFrames(
    frames: number,
    frameDurationSec: number,
    config: {
      adaptiveMinSilenceMs: number;
      adaptiveMaxSilenceMs: number;
    }
  ) {
    const minFrames = Math.max(1, Math.round((config.adaptiveMinSilenceMs / 1000) / frameDurationSec));
    const maxFrames = Math.max(minFrames, Math.round((config.adaptiveMaxSilenceMs / 1000) / frameDurationSec));
    return this.clamp(frames, minFrames, maxFrames);
  }

  private static updateAdaptiveSilenceFrames(
    pauseHistory: number[],
    currentFrames: number,
    frameDurationSec: number,
    config: {
      silenceMode: "auto" | "fixed";
      adaptiveMinSilenceMs: number;
      adaptiveMaxSilenceMs: number;
      logStats: boolean;
    }
  ) {
    if (config.silenceMode !== "auto" || pauseHistory.length < 3) {
      return currentFrames;
    }
    const sorted = [...pauseHistory].sort((a, b) => a - b);
    const index = Math.min(sorted.length - 1, Math.floor(sorted.length * 0.75));
    const p75 = sorted[index];
    const targetSec = this.clamp(p75 * 1.2, config.adaptiveMinSilenceMs / 1000, config.adaptiveMaxSilenceMs / 1000);
    const nextFrames = this.clampAdaptiveSilenceFrames(
      Math.max(1, Math.round(targetSec / frameDurationSec)),
      frameDurationSec,
      config
    );

    if (config.logStats && nextFrames !== currentFrames) {
      console.log(`[VAD] adaptive silence updated: ${targetSec.toFixed(2)}s (${nextFrames} frames), p75=${p75.toFixed(2)}s`);
    }
    return nextFrames;
  }

  private static finalizeSpeechSegments(
    probabilities: number[],
    totalSec: number,
    config: ReturnType<typeof VadService.getConfig>
  ) {
    const frameSize = config.frameSize;
    const frameDurationSec = frameSize / this.sampleRate;
    const minSpeechSec = config.minSpeechMs / 1000;
    const preSpeechFrames = Math.max(0, Math.round((config.preSpeechMs / 1000) / frameDurationSec));
    let silenceLimitFrames = Math.max(1, Math.round((config.initialSilenceMs / 1000) / frameDurationSec));
    silenceLimitFrames = this.clampAdaptiveSilenceFrames(silenceLimitFrames, frameDurationSec, config);

    const segmentsRaw: SpeechSegment[] = [];
    const preSpeechStartQueue: number[] = [];
    const pauseHistory: number[] = [];

    let isSpeech = false;
    let speechStart = 0;
    let speechFrameCount = 0;
    let silenceFrameCount = 0;
    let lastSpeechEnd = 0;

    for (let frameIndex = 0; frameIndex < probabilities.length; frameIndex += 1) {
      const probability = probabilities[frameIndex];
      const currentStart = (frameIndex * frameSize) / this.sampleRate;
      const currentEnd = ((frameIndex + 1) * frameSize) / this.sampleRate;

      if (!isSpeech && probability >= config.startThreshold) {
        isSpeech = true;
        speechStart = preSpeechStartQueue.length > 0 ? preSpeechStartQueue[0] : currentStart;
        speechFrameCount = 1 + preSpeechStartQueue.length;
        silenceFrameCount = 0;
        lastSpeechEnd = currentEnd;
        preSpeechStartQueue.length = 0;
        continue;
      }

      if (!isSpeech) {
        if (preSpeechFrames > 0) {
          preSpeechStartQueue.push(currentStart);
          while (preSpeechStartQueue.length > preSpeechFrames) {
            preSpeechStartQueue.shift();
          }
        }
        continue;
      }

      if (probability > config.endThreshold) {
        if (silenceFrameCount > 0) {
          const pauseDurationSec = silenceFrameCount * frameDurationSec;
          if (pauseDurationSec >= 0.1) {
            pauseHistory.push(pauseDurationSec);
            if (pauseHistory.length > this.adaptivePauseHistoryMax) {
              pauseHistory.shift();
            }
            silenceLimitFrames = this.updateAdaptiveSilenceFrames(
              pauseHistory,
              silenceLimitFrames,
              frameDurationSec,
              config
            );
          }
        }
        speechFrameCount += 1;
        silenceFrameCount = 0;
        lastSpeechEnd = currentEnd;
        continue;
      }

      silenceFrameCount += 1;
      const speechDurationSec = Math.max(frameDurationSec, currentEnd - speechStart);
      const multiplier = this.getProgressiveSilenceMultiplier(speechDurationSec, config);
      const effectiveSilenceFrames = Math.max(1, Math.round(silenceLimitFrames * multiplier));
      if (silenceFrameCount >= effectiveSilenceFrames) {
        if (speechFrameCount * frameDurationSec >= minSpeechSec) {
          this.pushSegment(
            segmentsRaw,
            speechStart,
            lastSpeechEnd,
            totalSec,
            minSpeechSec,
            config.padMs / 1000
          );
        }

        isSpeech = false;
        speechFrameCount = 0;
        silenceFrameCount = 0;
      }
    }

    if (isSpeech && speechFrameCount * frameDurationSec >= minSpeechSec) {
      this.pushSegment(
        segmentsRaw,
        speechStart,
        Math.max(lastSpeechEnd, totalSec),
        totalSec,
        minSpeechSec,
        config.padMs / 1000
      );
    }

    return {
      segments: this.mergeAdjacentSegments(segmentsRaw, config.mergeGapMs / 1000),
      preSpeechFrames,
      silenceLimitFrames,
    };
  }

  private static buildRescueConfig(
    baseConfig: ReturnType<typeof VadService.getConfig>,
    maxProb: number
  ) {
    const rescueStart = this.clamp(Math.max(0.05, maxProb * 0.5), 0.05, Math.min(0.25, baseConfig.startThreshold));
    const rescueEnd = this.clamp(rescueStart * 0.5, 0.02, rescueStart - 0.01);
    return {
      ...baseConfig,
      startThreshold: rescueStart,
      endThreshold: rescueEnd,
      minSpeechMs: Math.min(baseConfig.minSpeechMs, 180),
      initialSilenceMs: Math.min(baseConfig.initialSilenceMs, 280),
      minSilenceMs: Math.min(baseConfig.minSilenceMs, 280),
      adaptiveMinSilenceMs: Math.min(baseConfig.adaptiveMinSilenceMs, 120),
      adaptiveMaxSilenceMs: Math.min(baseConfig.adaptiveMaxSilenceMs, 600),
    };
  }

  private static percentile(sortedValues: number[], ratio: number) {
    if (sortedValues.length === 0) return 0;
    const clampedRatio = this.clamp(ratio, 0, 1);
    const index = Math.min(
      sortedValues.length - 1,
      Math.max(0, Math.round((sortedValues.length - 1) * clampedRatio))
    );
    return sortedValues[index];
  }

  private static smoothSignal(values: number[], radius: number) {
    if (values.length === 0 || radius <= 0) return values;
    const smoothed = new Array<number>(values.length).fill(0);
    for (let i = 0; i < values.length; i += 1) {
      const start = Math.max(0, i - radius);
      const end = Math.min(values.length - 1, i + radius);
      let sum = 0;
      let count = 0;
      for (let j = start; j <= end; j += 1) {
        sum += values[j];
        count += 1;
      }
      smoothed[i] = count > 0 ? sum / count : values[i];
    }
    return smoothed;
  }

  private static buildEnergyFallbackSegments(
    samples: Float32Array,
    totalSec: number,
    config: ReturnType<typeof VadService.getConfig>
  ) {
    const frameSize = config.frameSize;
    const energies: number[] = [];

    for (let i = 0; i < samples.length; i += frameSize) {
      const frameData = samples.subarray(i, i + frameSize);
      if (frameData.length < frameSize) break;

      let energy = 0;
      for (let j = 0; j < frameData.length; j += 1) {
        const sample = frameData[j];
        energy += sample * sample;
      }
      energies.push(Math.sqrt(energy / frameData.length));
    }

    if (energies.length === 0) {
      return {
        segments: [] as SpeechSegment[],
        stats: { p35: 0, p80: 0, p95: 0, startThreshold: 0, endThreshold: 0 },
      };
    }

    const smoothed = this.smoothSignal(energies, 2);
    const sorted = [...smoothed].sort((a, b) => a - b);
    const p35 = this.percentile(sorted, 0.35);
    const p80 = this.percentile(sorted, 0.8);
    const p95 = this.percentile(sorted, 0.95);
    const dynamicRange = p95 - p35;
    if (p95 < 0.005 || dynamicRange < 0.0015) {
      return {
        segments: [] as SpeechSegment[],
        stats: { p35, p80, p95, startThreshold: 0, endThreshold: 0 },
      };
    }

    const normalized = smoothed.map((value) => this.clamp(value / Math.max(p95, 1e-6), 0, 1));
    const normalizedP35 = p35 / Math.max(p95, 1e-6);
    const normalizedP80 = p80 / Math.max(p95, 1e-6);
    const startThreshold = this.clamp(
      Math.max(0.08, normalizedP35 * 2.2, normalizedP80 * 0.55),
      0.08,
      0.72
    );
    const endThreshold = this.clamp(startThreshold * 0.6, 0.04, startThreshold - 0.01);
    const energyConfig = {
      ...config,
      startThreshold,
      endThreshold,
      minSpeechMs: Math.min(config.minSpeechMs, 420),
      initialSilenceMs: Math.min(config.initialSilenceMs, 420),
      minSilenceMs: Math.min(config.minSilenceMs, 420),
      adaptiveMinSilenceMs: Math.min(config.adaptiveMinSilenceMs, 180),
      adaptiveMaxSilenceMs: Math.min(config.adaptiveMaxSilenceMs, 800),
      mergeGapMs: Math.max(config.mergeGapMs, 320),
      padMs: Math.max(config.padMs, 160),
    };
    const pass = this.finalizeSpeechSegments(normalized, totalSec, energyConfig);
    return {
      segments: pass.segments,
      stats: { p35, p80, p95, startThreshold, endThreshold },
    };
  }

  private static async initSilero() {
    if (this.compiledModel || this.session) return;
    let openvinoModelPath = await this.getOpenvinoModelPath();
    if (!openvinoModelPath && !(await fs.pathExists(this.modelPath))) {
      try {
        await ResourceManager.ensureBaselineAsset("vad");
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.warn(`[VAD] Could not download baseline VAD model, continuing with existing fallback behavior: ${message}`);
      }
      openvinoModelPath = await this.getOpenvinoModelPath();
    }

    if (!openvinoModelPath && !(await fs.pathExists(this.modelPath))) {
      throw new Error(`[VAD] model not found: ${this.modelPath}`);
    }

    let openvinoError: unknown = null;
    if (openvinoModelPath) {
      try {
        const device = OpenvinoBackend.getBaselineModelDevice();
        console.log(`[VAD] Trying OpenVINO device: ${device} (${path.basename(openvinoModelPath)})`);
        this.compiledModel = await OpenvinoBackend.compileModel(openvinoModelPath, device);
        this.backend = "openvino";
        console.log(`[VAD] Using inference backend: OpenVINO ${device}`);
        return;
      } catch (error) {
        openvinoError = error;
        const message = error instanceof Error ? error.message : String(error);
        console.warn(`[VAD] OpenVINO VAD model failed, falling back to ONNX Runtime. ${message}`);
      }
    } else {
      console.log("[VAD] No OpenVINO-compatible VAD IR model found, using ONNX Runtime fallback.");
    }

    const providers = process.platform === "win32" ? ["dml", "webgpu", "cpu"] : ["webgpu", "cpu"];
    for (const provider of providers) {
      try {
        const label = provider === "dml" ? "directml" : provider;
        console.log(`[VAD] Trying ONNX Runtime execution provider: ${label}`);
        this.session = await ort.InferenceSession.create(this.modelPath, {
          executionProviders: [provider],
        });
        this.backend = "onnxruntime";
        console.log(`[VAD] Using inference backend: ONNX Runtime (${label})`);
        return;
      } catch {
        const label = provider === "dml" ? "directml" : provider;
        console.warn(`[VAD] Provider ${label} unavailable, trying next...`);
      }
    }

    const detail = openvinoError instanceof Error ? openvinoError.message : String(openvinoError ?? "not attempted");
    throw new Error(`[VAD] Failed to initialize OpenVINO and ONNX Runtime backends. OpenVINO error: ${detail}`);
  }

  private static async detectSpeechWithSilero(filePath: string): Promise<SpeechSegment[]> {
    await this.initSilero();
    if (!this.compiledModel && !this.session) throw new Error("VAD runtime missing");

    const config = this.getConfig();
    const rawSamples = await this.extractAudioForVad(filePath);
    const normalizedAudio = this.normalizeSamplesForVad(rawSamples);
    const samples = normalizedAudio.samples;
    const totalSec = samples.length / this.sampleRate;
    const frameSize = config.frameSize;
    const probabilities: number[] = [];

    const inferRequest = this.compiledModel ? this.compiledModel.createInferRequest() : null;
    const inputPorts = this.compiledModel && Array.isArray(this.compiledModel.inputs) ? this.compiledModel.inputs : [];
    const outputPorts = this.compiledModel && Array.isArray(this.compiledModel.outputs) ? this.compiledModel.outputs : [];
    const inputName = this.compiledModel
      ? OpenvinoBackend.getPortName(inputPorts[0], "input")
      : (this.session?.inputNames[0] ?? "input");
    const srName = this.compiledModel
      ? OpenvinoBackend.getPortName(
          inputPorts.find((port: any) => OpenvinoBackend.getPortName(port, "").toLowerCase().includes("sr")),
          "sr"
        )
      : (this.session?.inputNames.find((name) => name.toLowerCase().includes("sr")) ?? "sr");
    const stateName = this.compiledModel
      ? OpenvinoBackend.getPortName(
          inputPorts.find((port: any) => OpenvinoBackend.getPortName(port, "").toLowerCase().includes("state")),
          "state"
        )
      : (this.session?.inputNames.find((name) => name.toLowerCase().includes("state")) ?? "state");
    const stateInputIndex = !this.compiledModel && this.session ? this.session.inputNames.indexOf(stateName) : -1;
    const stateMetadata = stateInputIndex >= 0 ? this.session?.inputMetadata[stateInputIndex] : undefined;
    const stateDimFromMetadata = stateMetadata?.isTensor ? Number(stateMetadata.shape?.[2]) : 128;
    const stateDim = this.compiledModel
      ? Number(
          (
            inputPorts.find((port: any) => OpenvinoBackend.getPortName(port, "").toLowerCase().includes("state"))?.getShape?.() ??
            [2, 1, 128]
          ).at(-1)
        ) || 128
      : stateDimFromMetadata || 128;
    const probName = this.compiledModel
      ? OpenvinoBackend.getPortName(
          outputPorts.find((port: any) => {
            const name = OpenvinoBackend.getPortName(port, "").toLowerCase();
            return name === "output" || name.includes("prob");
          }) || outputPorts[0],
          "output"
        )
      : (this.session?.outputNames.find((name) => name === "output" || name.toLowerCase().includes("prob")) ??
        this.session?.outputNames[0] ??
        "output");
    const outStateName = this.compiledModel
      ? OpenvinoBackend.getPortName(
          outputPorts.find((port: any) => {
            const name = OpenvinoBackend.getPortName(port, "").toLowerCase();
            return name.includes("state") || name.includes("hn");
          }) || outputPorts[1],
          "state_out"
        )
      : (this.session?.outputNames.find((name) => name.toLowerCase().includes("state") || name.toLowerCase().includes("hn")) ??
        this.session?.outputNames[1] ??
        "state_out");

    let state: any = this.compiledModel
      ? await OpenvinoBackend.createTensor("f32", [2, 1, stateDim], new Float32Array(2 * stateDim).fill(0))
      : new ort.Tensor("float32", new Float32Array(2 * stateDim).fill(0), [2, 1, stateDim]);
    const sr: any = this.compiledModel
      ? await OpenvinoBackend.createTensor("i64", [1], BigInt64Array.from([BigInt(this.sampleRate)]))
      : new ort.Tensor("int64", BigInt64Array.from([BigInt(this.sampleRate)]), [1]);

    let isSpeech = false;
    let speechStart = 0;
    let speechFrameCount = 0;
    let silenceFrameCount = 0;
    let lastSpeechEnd = 0;

    let maxProb = 0;
    let sumProb = 0;
    let probCount = 0;

    try {
      for (let i = 0; i < samples.length; i += frameSize) {
        const frameData = samples.slice(i, i + frameSize);
        if (frameData.length < frameSize) break;

        let results: Record<string, any>;
        if (this.compiledModel && inferRequest) {
          const input = await OpenvinoBackend.createTensor("f32", [1, frameSize], frameData);
          results = inferRequest.infer({
            [inputName]: input,
            [srName]: sr,
            [stateName]: state,
          });
        } else if (this.session) {
          const input = new ort.Tensor("float32", frameData, [1, frameSize]);
          results = await this.session.run({
            [inputName]: input,
            [srName]: sr,
            [stateName]: state,
          });
        } else {
          throw new Error("VAD runtime missing during inference.");
        }

        const resultValues = Object.values(results);
        state = (results[outStateName] ?? resultValues[1]) as any;
        const probabilityTensorData = this.compiledModel
          ? (OpenvinoBackend.getTensorData(results[probName] ?? resultValues[0]) as any)
          : ((results[probName] ?? resultValues[0])?.data as any);
        const probability = this.toProbability(probabilityTensorData?.[0]);
        probabilities.push(probability);

        maxProb = Math.max(maxProb, probability);
        sumProb += probability;
        probCount += 1;
      }
      let segmentPass = this.finalizeSpeechSegments(probabilities, totalSec, config);
      let merged = segmentPass.segments;
      let usedRescueConfig = false;
      let usedEnergyFallback = false;
      let energyStats: {
        p35: number;
        p80: number;
        p95: number;
        startThreshold: number;
        endThreshold: number;
      } | null = null;
      if (merged.length === 0 && maxProb >= 0.05 && maxProb < config.startThreshold) {
        const rescueConfig = this.buildRescueConfig(config, maxProb);
        const rescuePass = this.finalizeSpeechSegments(probabilities, totalSec, rescueConfig);
        if (rescuePass.segments.length > 0) {
          merged = rescuePass.segments;
          segmentPass = rescuePass;
          usedRescueConfig = true;
          if (config.logStats) {
            console.log(
              `[VAD] rescue thresholds applied: startTh=${rescueConfig.startThreshold.toFixed(4)}, endTh=${rescueConfig.endThreshold.toFixed(4)}, minSpeechMs=${rescueConfig.minSpeechMs}`
            );
          }
        }
      }

      if (merged.length === 0) {
        const energyFallback = this.buildEnergyFallbackSegments(samples, totalSec, config);
        if (energyFallback.segments.length > 0) {
          merged = energyFallback.segments;
          usedEnergyFallback = true;
          energyStats = energyFallback.stats;
          if (config.logStats) {
            console.log(
              `[VAD] energy fallback applied: segments=${merged.length}, p35=${energyStats.p35.toFixed(5)}, p80=${energyStats.p80.toFixed(5)}, p95=${energyStats.p95.toFixed(5)}, startTh=${energyStats.startThreshold.toFixed(4)}, endTh=${energyStats.endThreshold.toFixed(4)}`
            );
          }
        }
      }

      if (config.logStats) {
        const avgProb = probCount > 0 ? sumProb / probCount : 0;
        console.log(
          `[VAD] backend=${this.backend ?? "unknown"}, segments=${merged.length}, frameSize=${frameSize}, maxProb=${maxProb.toFixed(4)}, avgProb=${avgProb.toFixed(4)}, gain=${normalizedAudio.gain.toFixed(2)}, peak=${normalizedAudio.peak.toFixed(4)}, startTh=${config.startThreshold}, endTh=${config.endThreshold}, silenceMode=${config.silenceMode}, silenceFrames=${segmentPass.silenceLimitFrames}, preSpeechFrames=${segmentPass.preSpeechFrames}, rescue=${usedRescueConfig ? "on" : "off"}, energyFallback=${usedEnergyFallback ? "on" : "off"}`
        );
      }

      return merged;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error("[VAD] detection failed:", message);
      return [];
    }
  }

  private static getTenConfig(baseConfig: ReturnType<typeof VadService.getConfig>) {
    const windowSizeRaw = Math.round(this.getEnvNumber("TEN_VAD_WINDOW_SIZE", 256, 160, 512));
    // TEN_VAD_* values are optional overrides; otherwise Ten follows shared VAD_* semantics.
    return {
      provider: String(process.env.TEN_VAD_PROVIDER || "cpu").trim() || "cpu",
      windowSize: windowSizeRaw,
      threshold: this.getEnvNumber("TEN_VAD_THRESHOLD", baseConfig.startThreshold, 0.05, 0.95),
      minSilenceDuration: this.getEnvNumber(
        "TEN_VAD_MIN_SILENCE_MS",
        baseConfig.initialSilenceMs,
        30,
        5000
      ) / 1000,
      minSpeechDuration: this.getEnvNumber("TEN_VAD_MIN_SPEECH_MS", baseConfig.minSpeechMs, 30, 30000) / 1000,
      maxSpeechDuration: this.getEnvNumber("TEN_VAD_MAX_SPEECH_SEC", 20, 1, 1200),
    };
  }

  private static loadSherpaOnnx() {
    if (!this.tenVadModule) {
      this.tenVadModule = require("sherpa-onnx");
    }
    return this.tenVadModule;
  }

  private static postProcessExternalSegments(
    segments: SpeechSegment[],
    totalSec: number,
    config: ReturnType<typeof VadService.getConfig>
  ) {
    const normalized: SpeechSegment[] = [];
    const minSpeechSec = config.minSpeechMs / 1000;
    const padSec = config.padMs / 1000;
    for (const segment of segments) {
      this.pushSegment(
        normalized,
        this.clamp(segment.start, 0, totalSec),
        this.clamp(segment.end, 0, totalSec),
        totalSec,
        minSpeechSec,
        padSec
      );
    }
    return this.mergeAdjacentSegments(normalized, config.mergeGapMs / 1000);
  }

  private static async detectSpeechWithTen(filePath: string, downloadIfMissing: boolean): Promise<SpeechSegment[]> {
    const config = this.getConfig();
    const modelPath = await this.getTenModelPath(downloadIfMissing);
    if (!modelPath) {
      throw new Error(`[VAD] Ten VAD model not found: ${this.tenModelPath}`);
    }

    const rawSamples = await this.extractAudioForVad(filePath);
    const normalizedAudio = this.normalizeSamplesForVad(rawSamples);
    const samples = normalizedAudio.samples;
    const totalSec = samples.length / this.sampleRate;
    const tenConfig = this.getTenConfig(config);
    const sherpa = this.loadSherpaOnnx();
    const vad = sherpa.createVad({
      tenVad: {
        model: modelPath,
        threshold: tenConfig.threshold,
        minSilenceDuration: tenConfig.minSilenceDuration,
        minSpeechDuration: tenConfig.minSpeechDuration,
        maxSpeechDuration: tenConfig.maxSpeechDuration,
        windowSize: tenConfig.windowSize,
      },
      sileroVad: {
        model: "",
        threshold: config.startThreshold,
        minSilenceDuration: config.initialSilenceMs / 1000,
        minSpeechDuration: config.minSpeechMs / 1000,
        maxSpeechDuration: tenConfig.maxSpeechDuration,
        windowSize: config.frameSize,
      },
      sampleRate: this.sampleRate,
      numThreads: 1,
      provider: tenConfig.provider,
      debug: 0,
      bufferSizeInSeconds: Math.max(30, Math.ceil(totalSec) + 1),
    });

    try {
      const chunkSize = tenConfig.windowSize;
      for (let offset = 0; offset < samples.length; offset += chunkSize) {
        vad.acceptWaveform(samples.subarray(offset, Math.min(samples.length, offset + chunkSize)));
      }
      vad.flush();

      const rawSegments: SpeechSegment[] = [];
      while (!vad.isEmpty()) {
        const segment = vad.front();
        const start = Number(segment?.start || 0) / this.sampleRate;
        const end = (Number(segment?.start || 0) + Number(segment?.samples?.length || 0)) / this.sampleRate;
        if (Number.isFinite(start) && Number.isFinite(end) && end > start) {
          rawSegments.push({ start, end });
        }
        vad.pop();
      }

      const merged = this.postProcessExternalSegments(rawSegments, totalSec, config);
      if (config.logStats) {
        console.log(
          `[VAD] backend=ten-onnx, segments=${merged.length}, rawSegments=${rawSegments.length}, windowSize=${tenConfig.windowSize}, threshold=${tenConfig.threshold}, silence=${tenConfig.minSilenceDuration.toFixed(2)}s, gain=${normalizedAudio.gain.toFixed(2)}, peak=${normalizedAudio.peak.toFixed(4)}`
        );
      }
      this.backend = "ten-onnx";
      return merged;
    } finally {
      vad.free();
    }
  }

  private static summarizeSegments(segments: SpeechSegment[]) {
    const totalSec = segments.reduce((sum, segment) => sum + Math.max(0, segment.end - segment.start), 0);
    return {
      count: segments.length,
      totalSec: Number(totalSec.toFixed(3)),
      avgSec: segments.length > 0 ? Number((totalSec / segments.length).toFixed(3)) : 0,
    };
  }

  static async detectSpeech(filePath: string): Promise<SpeechSegment[]> {
    const engine = this.getEngine();
    if (engine === "ten") {
      return this.detectSpeechWithTen(filePath, true);
    }

    if (engine === "auto") {
      try {
        const tenSegments = await this.detectSpeechWithTen(filePath, true);
        if (tenSegments.length > 0) {
          return tenSegments;
        }
        console.warn("[VAD] Ten VAD returned no speech segments, falling back to Silero.");
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.warn(`[VAD] Ten VAD unavailable, falling back to Silero: ${message}`);
      }
      return this.detectSpeechWithSilero(filePath);
    }

    if (engine === "compare") {
      const [silero, ten] = await Promise.all([
        this.detectSpeechWithSilero(filePath),
        this.detectSpeechWithTen(filePath, true).catch((error) => {
          const message = error instanceof Error ? error.message : String(error);
          console.warn(`[VAD] Ten VAD compare failed: ${message}`);
          return [] as SpeechSegment[];
        }),
      ]);
      console.log(
        `[VAD] compare silero=${JSON.stringify(this.summarizeSegments(silero))} ten=${JSON.stringify(this.summarizeSegments(ten))}`
      );
      return ten.length > 0 ? ten : silero;
    }

    return this.detectSpeechWithSilero(filePath);
  }

  private static async extractAudioForVad(filePath: string): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
      const ffmpeg = spawn(this.getFfmpegBinary(), [
        "-i",
        filePath,
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        String(this.sampleRate),
        "pipe:1",
      ]);

      const chunks: Buffer[] = [];
      const stderrChunks: Buffer[] = [];

      ffmpeg.stdout.on("data", (chunk: Buffer) => chunks.push(chunk));
      ffmpeg.stderr.on("data", (chunk: Buffer) => stderrChunks.push(chunk));

      ffmpeg.on("close", (code) => {
        if (code === 0) {
          const buffer = Buffer.concat(chunks);
          resolve(new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4));
          return;
        }

        const stderrText = Buffer.concat(stderrChunks).toString("utf8").trim();
        reject(new Error(`FFmpeg failed (${code}). ${stderrText}`.trim()));
      });
    });
  }
}
