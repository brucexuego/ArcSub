import fs from 'fs-extra';
import path from 'path';
import { spawn } from 'child_process';
import { AutoProcessor, Tensor } from '@huggingface/transformers';
import { AudioProcessor } from './audio_processor.js';
import { ClusteringService } from './clustering_service.js';
import { OpenvinoBackend } from './openvino_backend.js';
import { PathManager } from './path_manager.js';
import { VadService } from './vad_service.js';
import { resolveToolCommand } from './runtime_tools.js';

export interface PyannoteSpeakerTurn {
  start: number;
  end: number;
  speaker: string;
  confidence: number;
}

export interface PyannoteDiarizationDiagnostics {
  modelSource: string;
  clusteringStrategy: string;
  clusteringBackend: 'vbx_plda' | 'cosine_fallback';
  audioDurationSec: number;
  windowDurationSec: number;
  windowHopSec: number;
  windowCount: number;
  turnCount: number;
  clusteredSpeakerCount: number;
  autoSpeakerCount?: number;
  requestedSpeakerCount: {
    exactSpeakerCount: number | null;
    minSpeakers: number;
    maxSpeakers: number;
  };
  clusterThreshold: number;
  similarity: {
    pairCount: number;
    min: number;
    max: number;
    avg: number;
    p90: number;
  };
}

export interface PyannoteDiarizationResult {
  turns: PyannoteSpeakerTurn[];
  diagnostics: PyannoteDiarizationDiagnostics;
}

interface RawWindowTurn {
  start: number;
  end: number;
  confidence: number;
}

interface PyannoteVbxConfig {
  mean1: number[];
  mean2: number[];
  lda: number[][];
  plda_mu: number[];
  plda_tr: number[][];
  phi: number[];
  fa: number;
  fb: number;
  threshold: number;
  max_iters: number;
  init_smoothing: number;
  lda_dimension: number;
}

export interface PyannoteSpeakerCountOptions {
  exactSpeakerCount?: number | null;
  minSpeakers?: number;
  maxSpeakers?: number;
}

export class PyannoteDiarizationService {
  private static readonly sampleRate = 16000;
  private static readonly segmentationWindowSec = 10;
  private static readonly segmentationHopSec = 5;
  private static readonly embeddingWindowSec = 3;
  private static readonly minTurnSec = 0.35;
  private static readonly mergeGapSec = 0.0;
  private static readonly speechTimelineStepSec = 0.02;
  private static readonly speechProbabilityThreshold = 0.35;
  private static readonly minSpeechIntervalSec = 0.08;
  private static readonly maxSpeechGapBridgeSec = 0.08;
  private static readonly vadRefineMinTurnDurationSec = 8;
  private static readonly vadRefineSplitGapSec = 0.35;

  private static segmentationProcessorPromise: Promise<any> | null = null;
  private static segmentationModelPromise: Promise<any> | null = null;
  private static embeddingModelPromise: Promise<any> | null = null;
  private static vbxConfigPromise: Promise<PyannoteVbxConfig> | null = null;
  private static vadSegmentsPromiseByAudioPath = new Map<string, Promise<Array<{ start: number; end: number }>>>();
  private static pythonDepsPromise: Promise<void> | null = null;

  private static throwIfAborted(signal?: AbortSignal) {
    if (!signal?.aborted) return;
    throw new Error('Pyannote diarization aborted.');
  }

  private static getFfmpegBinary() {
    return resolveToolCommand('ffmpeg');
  }

  private static getPythonCommand() {
    const configured = String(process.env.OPENVINO_HELPER_PYTHON || '').trim();
    if (configured) return configured;
    return process.platform === 'win32' ? 'python' : 'python3';
  }

  private static getRootDir() {
    return path.join(PathManager.getModelsPath(), 'pyannote');
  }

  private static getLegacyRootDir() {
    return path.join(PathManager.getModelsPath(), 'pyannote-community-1');
  }

  private static getSegmentationDir() {
    return path.join(this.getRootDir(), 'segmentation');
  }

  private static getEmbeddingDir() {
    return path.join(this.getRootDir(), 'embedding');
  }

  private static getSegmentationIrPath() {
    return path.join(this.getSegmentationDir(), 'model.xml');
  }

  private static getEmbeddingIrPath() {
    return path.join(this.getEmbeddingDir(), 'model.xml');
  }

  private static getVbxConfigPath() {
    return path.join(this.getRootDir(), 'plda', 'vbx.json');
  }

  private static getHfToken() {
    const direct = String(process.env.HF_TOKEN || '').trim();
    if (direct) return direct;

    const envPath = PathManager.resolveDotEnvPath();
    if (fs.existsSync(envPath)) {
      const content = fs.readFileSync(envPath, 'utf8');
      const matched = content.match(/^HF_TOKEN=(.+)$/m);
      const fileToken = String(matched?.[1] || '').trim();
      if (fileToken) return fileToken;
    }

    throw new Error('HF_TOKEN is required to prepare pyannote assets.');
  }

  private static async downloadFile(url: string, outputPath: string, token?: string) {
    await fs.ensureDir(path.dirname(outputPath));
    const response = await fetch(url, {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    });
    if (!response.ok) {
      throw new Error(`Failed to download ${url} (${response.status})`);
    }
    const buffer = Buffer.from(await response.arrayBuffer());
    await fs.writeFile(outputPath, buffer);
  }

  private static async runPythonChecked(args: string[], fallbackErrorMessage: string) {
    const workspaceRoot = PathManager.getRoot();

    return new Promise<{ stdout: string; stderr: string }>((resolve, reject) => {
      const child = spawn(this.getPythonCommand(), args, {
        cwd: workspaceRoot,
        env: {
          ...process.env,
          ...PathManager.getRuntimeTempEnv(),
          PYTHONUTF8: '1',
          PYTHONIOENCODING: 'utf-8',
        },
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let stdout = '';
      let stderr = '';
      child.stdout.on('data', (chunk: Buffer) => {
        stdout += chunk.toString('utf8');
      });
      child.stderr.on('data', (chunk: Buffer) => {
        stderr += chunk.toString('utf8');
      });
      child.on('close', (code) => {
        if (code === 0) {
          resolve({ stdout, stderr });
          return;
        }
        reject(new Error(stderr.trim() || stdout.trim() || fallbackErrorMessage));
      });
      child.on('error', (error) => {
        reject(new Error(String(error?.message || error || fallbackErrorMessage)));
      });
    });
  }

  private static async ensurePythonDependencies() {
    if (this.pythonDepsPromise) {
      return this.pythonDepsPromise;
    }

    this.pythonDepsPromise = (async () => {
      const tempScriptPath = path.join(PathManager.getTmpPath(), `pyannote-deps-probe-${Date.now()}.py`);
      const probeScript = [
        'import importlib.util',
        '',
        'def _missing(name):',
        '    try:',
        '        return importlib.util.find_spec(name) is None',
        '    except ModuleNotFoundError:',
        '        return True',
        '',
        'modules = ["torch", "torchaudio", "numpy", "scipy", "openvino", "onnx", "pyannote.audio"]',
        'missing = [name for name in modules if _missing(name)]',
        'print("\\n".join(missing))',
        '',
      ].join('\n');

      await fs.ensureDir(path.dirname(tempScriptPath));
      await fs.writeFile(tempScriptPath, probeScript, 'utf8');

      let probeResult: { stdout: string; stderr: string };
      try {
        probeResult = await this.runPythonChecked(
          [tempScriptPath],
          'Failed to inspect pyannote Python dependencies.'
        );
      } finally {
        await fs.remove(tempScriptPath).catch(() => {});
      }
      const missing = String(probeResult.stdout || '')
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean);

      if (missing.length === 0) {
        return;
      }

      const packageMap: Record<string, string[]> = {
        torch: ['torch'],
        torchaudio: ['torchaudio'],
        numpy: ['numpy'],
        scipy: ['scipy'],
        openvino: ['openvino'],
        onnx: ['onnx'],
        'pyannote.audio': ['pyannote.audio'],
      };
      const packages = Array.from(new Set(missing.flatMap((name) => packageMap[name] || [])));
      if (packages.length === 0) {
        return;
      }

      await this.runPythonChecked(
        ['-m', 'pip', 'install', '--upgrade', ...packages],
        'Failed to install pyannote Python dependencies.'
      );
    })();

    try {
      await this.pythonDepsPromise;
    } catch (error) {
      this.pythonDepsPromise = null;
      throw error;
    }
  }

  private static async runPythonOpenvinoConversion(
    sourcePath: string,
    outputXmlPath: string,
    shape: number[]
  ) {
    const workspaceRoot = PathManager.getRoot();
    const toolPath = PathManager.resolveToolsSourcePath();
    const script = [
      'import sys',
      `sys.path.insert(0, ${JSON.stringify(toolPath)})`,
      'from openvino_asr_env import prepare_openvino_env',
      'prepare_openvino_env()',
      'from openvino import Core, save_model',
      'core = Core()',
      `model = core.read_model(${JSON.stringify(sourcePath)})`,
      'input_port = model.inputs[0]',
      `model.reshape({input_port: ${JSON.stringify(shape)}})`,
      `save_model(model, ${JSON.stringify(outputXmlPath)}, compress_to_fp16=False)`,
    ].join('\n');

    await new Promise<void>((resolve, reject) => {
      const child = spawn(this.getPythonCommand(), ['-c', script], {
        cwd: workspaceRoot,
        env: {
          ...process.env,
          ...PathManager.getRuntimeTempEnv(),
          PYTHONUTF8: '1',
          PYTHONIOENCODING: 'utf-8',
        },
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let stderr = '';
      child.stderr.on('data', (chunk: Buffer) => {
        stderr += chunk.toString('utf8');
      });
      child.on('close', (code) => {
        if (code === 0) return resolve();
        reject(new Error(stderr.trim() || `OpenVINO conversion failed (${code}).`));
      });
      child.on('error', reject);
    });
  }

  private static async runPythonVbxPreparation(transformPath: string, pldaPath: string, outputJsonPath: string) {
    const workspaceRoot = PathManager.getRoot();
    const scriptPath = PathManager.resolveToolsSourcePath('prepare_pyannote_vbx.py');

    await new Promise<void>((resolve, reject) => {
      const child = spawn(
        this.getPythonCommand(),
        [scriptPath, '--transform-npz', transformPath, '--plda-npz', pldaPath, '--output-json', outputJsonPath],
        {
          cwd: workspaceRoot,
          env: {
            ...process.env,
            ...PathManager.getRuntimeTempEnv(),
            PYTHONUTF8: '1',
            PYTHONIOENCODING: 'utf-8',
          },
          stdio: ['ignore', 'pipe', 'pipe'],
        }
      );
      let stderr = '';
      child.stderr.on('data', (chunk: Buffer) => {
        stderr += chunk.toString('utf8');
      });
      child.on('close', (code) => {
        if (code === 0) return resolve();
        reject(new Error(stderr.trim() || `Pyannote VBx prep failed (${code}).`));
      });
      child.on('error', reject);
    });
  }

  private static async runPythonExport(subfolder: 'segmentation' | 'embedding', outputDir: string) {
    const workspaceRoot = PathManager.getRoot();
    const scriptPath = PathManager.resolveToolsSourcePath('export_pyannote.py');
    const token = this.getHfToken();

    await new Promise<void>((resolve, reject) => {
      const child = spawn(
        this.getPythonCommand(),
        [scriptPath, '--output-dir', outputDir, '--subfolder', subfolder, '--token', token],
        {
          cwd: workspaceRoot,
          env: {
            ...process.env,
            ...PathManager.getRuntimeTempEnv(),
            PYTHONUTF8: '1',
            PYTHONIOENCODING: 'utf-8',
          },
          stdio: ['ignore', 'pipe', 'pipe'],
        }
      );
      let stderr = '';
      child.stderr.on('data', (chunk: Buffer) => {
        stderr += chunk.toString('utf8');
      });
      child.on('close', (code) => {
        if (code === 0) return resolve();
        reject(new Error(stderr.trim() || `Pyannote export failed (${code}).`));
      });
      child.on('error', reject);
    });
  }

  private static async ensureModelsPrepared() {
    const legacyRootDir = this.getLegacyRootDir();
    const rootDir = this.getRootDir();
    if (!(await fs.pathExists(rootDir)) && (await fs.pathExists(legacyRootDir))) {
      await fs.move(legacyRootDir, rootDir);
    }

    const segmentationDir = this.getSegmentationDir();
    const embeddingDir = this.getEmbeddingDir();
    const pldaDir = path.join(this.getRootDir(), 'plda');
    const hfToken = this.getHfToken();

    await fs.ensureDir(segmentationDir);
    await fs.ensureDir(embeddingDir);
    await fs.ensureDir(pldaDir);

    const segmentationConfigPath = path.join(segmentationDir, 'config.json');
    const segmentationPreprocessorPath = path.join(segmentationDir, 'preprocessor_config.json');
    const segmentationOnnxPath = path.join(segmentationDir, 'model.onnx');
    const embeddingOnnxPath = path.join(embeddingDir, 'model.onnx');
    const pldaPath = path.join(pldaDir, 'plda.npz');
    const xvecTransformPath = path.join(pldaDir, 'xvec_transform.npz');
    const vbxConfigPath = this.getVbxConfigPath();

    await this.ensurePythonDependencies();

    if (!(await fs.pathExists(segmentationConfigPath))) {
      await this.downloadFile(
        'https://huggingface.co/onnx-community/pyannote-segmentation-3.0/raw/main/config.json',
        segmentationConfigPath
      );
    }
    if (!(await fs.pathExists(segmentationPreprocessorPath))) {
      await this.downloadFile(
        'https://huggingface.co/onnx-community/pyannote-segmentation-3.0/raw/main/preprocessor_config.json',
        segmentationPreprocessorPath
      );
    }
    if (!(await fs.pathExists(segmentationOnnxPath))) {
      await this.runPythonExport('segmentation', segmentationDir);
    }
    if (!(await fs.pathExists(embeddingOnnxPath))) {
      await this.runPythonExport('embedding', embeddingDir);
    }
    if (!(await fs.pathExists(pldaPath))) {
      await this.downloadFile(
        'https://huggingface.co/pyannote/speaker-diarization-community-1/resolve/main/plda/plda.npz?download=true',
        pldaPath,
        hfToken
      );
    }
    if (!(await fs.pathExists(xvecTransformPath))) {
      await this.downloadFile(
        'https://huggingface.co/pyannote/speaker-diarization-community-1/resolve/main/plda/xvec_transform.npz?download=true',
        xvecTransformPath,
        hfToken
      );
    }
    if (!(await fs.pathExists(vbxConfigPath))) {
      await this.runPythonVbxPreparation(xvecTransformPath, pldaPath, vbxConfigPath);
    }

    if (!(await fs.pathExists(this.getSegmentationIrPath()))) {
      await this.runPythonOpenvinoConversion(segmentationOnnxPath, this.getSegmentationIrPath(), [1, 1, 160000]);
    }
    if (!(await fs.pathExists(this.getEmbeddingIrPath()))) {
      await this.runPythonOpenvinoConversion(embeddingOnnxPath, this.getEmbeddingIrPath(), [1, 298, 80]);
    }
  }

  static async ensureDeploymentAssets() {
    await this.ensureModelsPrepared();
  }

  private static async getSegmentationProcessor() {
    if (!this.segmentationProcessorPromise) {
      this.segmentationProcessorPromise = this.ensureModelsPrepared().then(() =>
        AutoProcessor.from_pretrained(this.getSegmentationDir(), { local_files_only: true })
      );
    }
    return this.segmentationProcessorPromise;
  }

  private static async getSegmentationModel() {
    if (!this.segmentationModelPromise) {
      this.segmentationModelPromise = this.ensureModelsPrepared().then(() =>
        OpenvinoBackend.compileModel(this.getSegmentationIrPath(), 'CPU')
      );
    }
    return this.segmentationModelPromise;
  }

  private static async getEmbeddingModel() {
    if (!this.embeddingModelPromise) {
      this.embeddingModelPromise = this.ensureModelsPrepared().then(() =>
        OpenvinoBackend.compileModel(this.getEmbeddingIrPath(), 'CPU')
      );
    }
    return this.embeddingModelPromise;
  }

  private static async getVbxConfig() {
    if (!this.vbxConfigPromise) {
      this.vbxConfigPromise = this.ensureModelsPrepared().then(() => fs.readJson(this.getVbxConfigPath()));
    }
    return this.vbxConfigPromise;
  }

  private static async getVadSpeechSegments(audioPath: string) {
    const cached = this.vadSegmentsPromiseByAudioPath.get(audioPath);
    if (cached) return cached;

    const pending = VadService.detectSpeech(audioPath)
      .catch((error) => {
        this.vadSegmentsPromiseByAudioPath.delete(audioPath);
        throw error;
      });
    this.vadSegmentsPromiseByAudioPath.set(audioPath, pending);
    return pending;
  }

  private static async extractMonoAudio(filePath: string): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
      const ffmpeg = spawn(this.getFfmpegBinary(), [
        '-v',
        'error',
        '-i',
        filePath,
        '-f',
        'f32le',
        '-ac',
        '1',
        '-ar',
        String(this.sampleRate),
        'pipe:1',
      ]);

      const stdoutChunks: Buffer[] = [];
      const stderrChunks: Buffer[] = [];
      ffmpeg.stdout.on('data', (chunk: Buffer) => stdoutChunks.push(chunk));
      ffmpeg.stderr.on('data', (chunk: Buffer) => stderrChunks.push(chunk));
      ffmpeg.on('close', (code) => {
        if (code === 0) {
          const buffer = Buffer.concat(stdoutChunks);
          resolve(new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4));
          return;
        }

        const stderr = Buffer.concat(stderrChunks).toString('utf8').trim();
        reject(new Error(stderr || 'Audio extraction failed.'));
      });
    });
  }

  private static padOrTrimWindow(samples: Float32Array, targetSamples: number) {
    if (samples.length === targetSamples) return samples;
    if (samples.length > targetSamples) return samples.slice(0, targetSamples);
    const padded = new Float32Array(targetSamples);
    padded.set(samples);
    return padded;
  }

  private static extractSegmentClip(
    fullAudio: Float32Array,
    startSec: number,
    endSec: number,
    targetSamples: number
  ) {
    const midpoint = (startSec + endSec) / 2;
    const halfDurationSec = targetSamples / this.sampleRate / 2;
    let clipStart = Math.max(0, midpoint - halfDurationSec);
    let clipEnd = clipStart + targetSamples / this.sampleRate;
    const totalSec = fullAudio.length / this.sampleRate;
    if (clipEnd > totalSec) {
      clipEnd = totalSec;
      clipStart = Math.max(0, clipEnd - targetSamples / this.sampleRate);
    }

    const startSample = Math.max(0, Math.floor(clipStart * this.sampleRate));
    const endSample = Math.min(fullAudio.length, startSample + targetSamples);
    const slice = fullAudio.slice(startSample, endSample);
    return this.padOrTrimWindow(slice, targetSamples);
  }

  private static async inferSegmentationLogits(windowSamples: Float32Array) {
    const model = await this.getSegmentationModel();
    const tensor = await OpenvinoBackend.createTensor('f32', [1, 1, 160000], windowSamples);
    const inferRequest = model.createInferRequest();
    const result = inferRequest.infer({ input_values: tensor });
    return OpenvinoBackend.getTensorData(result.logits ?? Object.values(result)[0]) as Float32Array;
  }

  private static async inferEmbedding(windowSamples: Float32Array) {
    const model = await this.getEmbeddingModel();
    const rawFeatures = await AudioProcessor.extractFeatures(windowSamples);
    const targetFrames = 298;
    const targetBins = 80;
    const expectedFeatureLength = targetFrames * targetBins;
    let finalFeatures: Float32Array;
    if (rawFeatures.length === expectedFeatureLength) {
      finalFeatures = rawFeatures;
    } else if (rawFeatures.length > expectedFeatureLength) {
      finalFeatures = rawFeatures.slice(0, expectedFeatureLength);
    } else {
      finalFeatures = new Float32Array(expectedFeatureLength);
      finalFeatures.set(rawFeatures);
    }

    const tensor = await OpenvinoBackend.createTensor('f32', [1, targetFrames, targetBins], finalFeatures);
    const inferRequest = model.createInferRequest();
    const result = inferRequest.infer({ fbank: tensor });
    const data = OpenvinoBackend.getTensorData(result.embeddings ?? Object.values(result)[0]) as Float32Array;
    return Array.from(data);
  }

  private static computeSimilarityStats(embeddings: number[][]) {
    const values: number[] = [];
    for (let i = 0; i < embeddings.length; i += 1) {
      for (let j = i + 1; j < embeddings.length; j += 1) {
        values.push(ClusteringService.cosineSimilarity(embeddings[i], embeddings[j]));
      }
    }
    if (values.length === 0) {
      return { pairCount: 0, min: 0, max: 0, avg: 0, p90: 0 };
    }
    const sorted = [...values].sort((a, b) => a - b);
    const p90Index = Math.min(sorted.length - 1, Math.max(0, Math.floor(sorted.length * 0.9)));
    const sum = values.reduce((acc, value) => acc + value, 0);
    return {
      pairCount: values.length,
      min: Math.min(...values),
      max: Math.max(...values),
      avg: sum / values.length,
      p90: sorted[p90Index],
    };
  }

  private static resolveClusterThreshold(stats: { avg: number }) {
    return Math.min(0.72, Math.max(0.28, stats.avg + 0.04));
  }

  private static l2Normalize(vector: number[]) {
    let norm = 0;
    for (const value of vector) norm += value * value;
    const scale = Math.sqrt(norm);
    if (!Number.isFinite(scale) || scale <= 0) return [...vector];
    return vector.map((value) => value / scale);
  }

  private static subtractVectors(left: number[], right: number[]) {
    return left.map((value, index) => value - (right[index] ?? 0));
  }

  private static multiplyMatrixTranspose(vector: number[], matrix: number[][]) {
    if (matrix.length === 0) return [];
    const output = new Array<number>(matrix[0].length).fill(0);
    for (let row = 0; row < matrix.length; row += 1) {
      const value = vector[row] ?? 0;
      const matrixRow = matrix[row];
      for (let column = 0; column < output.length; column += 1) {
        output[column] += value * (matrixRow[column] ?? 0);
      }
    }
    return output;
  }

  private static multiplyVectorByTransposedMatrix(vector: number[], matrix: number[][], limit = matrix.length) {
    const output = new Array<number>(Math.min(limit, matrix.length)).fill(0);
    for (let row = 0; row < output.length; row += 1) {
      let sum = 0;
      const matrixRow = matrix[row];
      for (let column = 0; column < vector.length; column += 1) {
        sum += (vector[column] ?? 0) * (matrixRow[column] ?? 0);
      }
      output[row] = sum;
    }
    return output;
  }

  private static transformEmbeddingForPlda(embedding: number[], config: PyannoteVbxConfig) {
    const firstPass = this.l2Normalize(this.subtractVectors(embedding, config.mean1));
    const scaledFirstPass = firstPass.map((value) => value * Math.sqrt(config.lda.length));
    const ldaProjected = this.multiplyMatrixTranspose(scaledFirstPass, config.lda);
    const secondPass = this.l2Normalize(this.subtractVectors(ldaProjected, config.mean2)).map(
      (value) => value * Math.sqrt(config.lda[0]?.length || config.lda_dimension)
    );
    const pldaInput = this.subtractVectors(secondPass, config.plda_mu);
    return this.multiplyVectorByTransposedMatrix(pldaInput, config.plda_tr, config.lda_dimension);
  }

  private static clusterAhcByCentroidDistance(embeddings: number[][], threshold: number) {
    if (embeddings.length === 0) return [];
    const normalizedEmbeddings = embeddings.map((embedding) => this.l2Normalize(embedding));
    const clusters = normalizedEmbeddings.map((embedding, index) => ({
      indices: [index],
      centroid: [...embedding],
    }));

    const euclideanDistance = (left: number[], right: number[]) => {
      let sum = 0;
      for (let i = 0; i < left.length; i += 1) {
        const diff = (left[i] ?? 0) - (right[i] ?? 0);
        sum += diff * diff;
      }
      return Math.sqrt(sum);
    };

    while (clusters.length > 1) {
      let bestI = -1;
      let bestJ = -1;
      let bestDistance = Number.POSITIVE_INFINITY;
      for (let i = 0; i < clusters.length; i += 1) {
        for (let j = i + 1; j < clusters.length; j += 1) {
          const distance = euclideanDistance(clusters[i].centroid, clusters[j].centroid);
          if (distance < bestDistance) {
            bestDistance = distance;
            bestI = i;
            bestJ = j;
          }
        }
      }
      if (bestI < 0 || bestJ < 0 || bestDistance > threshold) break;

      const left = clusters[bestI];
      const right = clusters[bestJ];
      const mergedIndices = [...left.indices, ...right.indices];
      const mergedCentroid = new Array<number>(left.centroid.length).fill(0);
      for (const index of mergedIndices) {
        const embedding = normalizedEmbeddings[index];
        for (let k = 0; k < mergedCentroid.length; k += 1) {
          mergedCentroid[k] += embedding[k] ?? 0;
        }
      }
      for (let k = 0; k < mergedCentroid.length; k += 1) {
        mergedCentroid[k] /= mergedIndices.length;
      }
      clusters[bestI] = { indices: mergedIndices, centroid: mergedCentroid };
      clusters.splice(bestJ, 1);
    }

    const labels = new Array<number>(embeddings.length).fill(0);
    clusters.forEach((cluster, clusterIndex) => {
      for (const index of cluster.indices) labels[index] = clusterIndex;
    });
    return labels;
  }

  private static softmax(values: number[]) {
    const max = Math.max(...values);
    const exps = values.map((value) => Math.exp(value - max));
    const sum = exps.reduce((acc, value) => acc + value, 0) || 1;
    return exps.map((value) => value / sum);
  }

  private static logSumExp(values: number[]) {
    const max = Math.max(...values);
    const sum = values.reduce((acc, value) => acc + Math.exp(value - max), 0);
    return max + Math.log(sum || 1);
  }

  private static accumulateSpeechProbabilities(
    logits: Float32Array,
    windowStartSec: number,
    actualWindowSec: number,
    timelineSums: number[],
    timelineCounts: number[]
  ) {
    const labelCount = 7;
    const frameCount = Math.floor(logits.length / labelCount);
    if (frameCount <= 0 || actualWindowSec <= 0) return;
    const frameDurationSec = actualWindowSec / frameCount;

    for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
      const offset = frameIndex * labelCount;
      const scores = Array.from(logits.slice(offset, offset + labelCount));
      const probabilities = this.softmax(scores);
      const speechProbability = 1 - (probabilities[0] ?? 0);
      const frameCenterSec = windowStartSec + (frameIndex + 0.5) * frameDurationSec;
      const timelineIndex = Math.max(
        0,
        Math.min(timelineSums.length - 1, Math.round(frameCenterSec / this.speechTimelineStepSec))
      );
      timelineSums[timelineIndex] += speechProbability;
      timelineCounts[timelineIndex] += 1;
    }
  }

  private static buildSpeechIntervals(timelineSums: number[], timelineCounts: number[], totalDurationSec: number) {
    const scores = timelineSums.map((sum, index) => (timelineCounts[index] > 0 ? sum / timelineCounts[index] : 0));
    const smoothedScores = scores.map((_, index) => {
      let weightedSum = 0;
      let weightTotal = 0;
      for (let neighbor = Math.max(0, index - 1); neighbor <= Math.min(scores.length - 1, index + 1); neighbor += 1) {
        const weight = neighbor === index ? 2 : 1;
        weightedSum += scores[neighbor] * weight;
        weightTotal += weight;
      }
      return weightTotal > 0 ? weightedSum / weightTotal : 0;
    });

    const binary = smoothedScores.map((score) => score >= this.speechProbabilityThreshold);
    const maxGapBins = Math.max(1, Math.round(this.maxSpeechGapBridgeSec / this.speechTimelineStepSec));
    const minSpeechBins = Math.max(1, Math.round(this.minSpeechIntervalSec / this.speechTimelineStepSec));

    for (let start = 0; start < binary.length; ) {
      if (binary[start]) {
        start += 1;
        continue;
      }
      let end = start;
      while (end < binary.length && !binary[end]) end += 1;
      const leftActive = start > 0 && binary[start - 1];
      const rightActive = end < binary.length && binary[end];
      if (leftActive && rightActive && end - start <= maxGapBins) {
        for (let index = start; index < end; index += 1) binary[index] = true;
      }
      start = end;
    }

    for (let start = 0; start < binary.length; ) {
      if (!binary[start]) {
        start += 1;
        continue;
      }
      let end = start;
      while (end < binary.length && binary[end]) end += 1;
      if (end - start < minSpeechBins) {
        for (let index = start; index < end; index += 1) binary[index] = false;
      }
      start = end;
    }

    const intervals: Array<{ start: number; end: number }> = [];
    for (let start = 0; start < binary.length; ) {
      if (!binary[start]) {
        start += 1;
        continue;
      }
      let end = start;
      while (end < binary.length && binary[end]) end += 1;
      intervals.push({
        start: start * this.speechTimelineStepSec,
        end: Math.min(totalDurationSec, end * this.speechTimelineStepSec),
      });
      start = end;
    }
    return intervals;
  }

  private static splitTurnsBySpeechIntervals(
    turns: RawWindowTurn[],
    intervals: Array<{ start: number; end: number }>
  ): RawWindowTurn[] {
    if (turns.length === 0 || intervals.length === 0) return turns;
    const splitTurns: RawWindowTurn[] = [];
    let intervalIndex = 0;

    for (const turn of turns) {
      while (intervalIndex < intervals.length && intervals[intervalIndex].end <= turn.start) {
        intervalIndex += 1;
      }

      let matched = false;
      let scanIndex = intervalIndex;
      while (scanIndex < intervals.length && intervals[scanIndex].start < turn.end) {
        const interval = intervals[scanIndex];
        const start = Math.max(turn.start, interval.start);
        const end = Math.min(turn.end, interval.end);
        if (end - start >= this.minTurnSec) {
          splitTurns.push({
            start,
            end,
            confidence: turn.confidence,
          });
          matched = true;
        }
        scanIndex += 1;
      }

      if (!matched) {
        splitTurns.push(turn);
      }
    }

    return splitTurns;
  }

  private static splitLongTurnsByVad(
    turns: RawWindowTurn[],
    vadSegments: Array<{ start: number; end: number }>
  ): RawWindowTurn[] {
    if (turns.length === 0 || vadSegments.length === 0) return turns;

    const refinedTurns: RawWindowTurn[] = [];
    let vadIndex = 0;

    for (const turn of turns) {
      const turnDuration = turn.end - turn.start;
      if (turnDuration < this.vadRefineMinTurnDurationSec) {
        refinedTurns.push(turn);
        continue;
      }

      while (vadIndex < vadSegments.length && vadSegments[vadIndex].end <= turn.start) {
        vadIndex += 1;
      }

      const overlaps: Array<{ start: number; end: number }> = [];
      let scanIndex = vadIndex;
      while (scanIndex < vadSegments.length && vadSegments[scanIndex].start < turn.end) {
        const overlapStart = Math.max(turn.start, vadSegments[scanIndex].start);
        const overlapEnd = Math.min(turn.end, vadSegments[scanIndex].end);
        if (overlapEnd - overlapStart >= this.minTurnSec) {
          const previous = overlaps[overlaps.length - 1];
          if (previous && overlapStart - previous.end <= this.vadRefineSplitGapSec) {
            previous.end = Math.max(previous.end, overlapEnd);
          } else {
            overlaps.push({ start: overlapStart, end: overlapEnd });
          }
        }
        scanIndex += 1;
      }

      if (overlaps.length <= 1) {
        refinedTurns.push(turn);
        continue;
      }

      for (const overlap of overlaps) {
        refinedTurns.push({
          start: overlap.start,
          end: overlap.end,
          confidence: turn.confidence,
        });
      }
    }

    return refinedTurns;
  }

  private static runVbxResponsibilities(
    features: number[][],
    phi: number[],
    initialClusters: number[],
    options: { fa: number; fb: number; maxIters: number; initSmoothing: number }
  ) {
    const speakerCount = Math.max(1, Math.max(...initialClusters) + 1);
    const frameCount = features.length;
    const dimension = features[0]?.length || 0;
    const qInit = Array.from({ length: frameCount }, () => new Array<number>(speakerCount).fill(0));
    for (let i = 0; i < frameCount; i += 1) {
      qInit[i][initialClusters[i]] = 1;
    }
    let gamma = qInit.map((row) => this.softmax(row.map((value) => value * options.initSmoothing)));
    let priors = new Array<number>(speakerCount).fill(1 / speakerCount);

    const g = features.map((feature) => {
      let squareSum = 0;
      for (const value of feature) squareSum += value * value;
      return -0.5 * (squareSum + dimension * Math.log(2 * Math.PI));
    });
    const sqrtPhi = phi.map((value) => Math.sqrt(Math.max(0, value)));
    const rho = features.map((feature) => feature.map((value, index) => value * (sqrtPhi[index] ?? 0)));

    for (let iter = 0; iter < options.maxIters; iter += 1) {
      const gammaSums = new Array<number>(speakerCount).fill(0);
      for (const row of gamma) {
        for (let speaker = 0; speaker < speakerCount; speaker += 1) gammaSums[speaker] += row[speaker] ?? 0;
      }

      const invL = Array.from({ length: speakerCount }, () => new Array<number>(dimension).fill(0));
      const alpha = Array.from({ length: speakerCount }, () => new Array<number>(dimension).fill(0));
      for (let speaker = 0; speaker < speakerCount; speaker += 1) {
        for (let d = 0; d < dimension; d += 1) {
          const phiValue = phi[d] ?? 0;
          const inv = 1 / (1 + (options.fa / options.fb) * gammaSums[speaker] * phiValue);
          invL[speaker][d] = inv;
        }
      }
      for (let speaker = 0; speaker < speakerCount; speaker += 1) {
        for (let frame = 0; frame < frameCount; frame += 1) {
          const responsibility = gamma[frame][speaker] ?? 0;
          for (let d = 0; d < dimension; d += 1) {
            alpha[speaker][d] += responsibility * (rho[frame][d] ?? 0);
          }
        }
        for (let d = 0; d < dimension; d += 1) {
          alpha[speaker][d] = (options.fa / options.fb) * invL[speaker][d] * alpha[speaker][d];
        }
      }

      const nextGamma = Array.from({ length: frameCount }, () => new Array<number>(speakerCount).fill(0));
      const nextPriors = new Array<number>(speakerCount).fill(0);
      for (let frame = 0; frame < frameCount; frame += 1) {
        const logSpeakerProbabilities = new Array<number>(speakerCount).fill(0);
        for (let speaker = 0; speaker < speakerCount; speaker += 1) {
          let alphaTerm = 0;
          let invLTerm = 0;
          for (let d = 0; d < dimension; d += 1) {
            alphaTerm += (rho[frame][d] ?? 0) * (alpha[speaker][d] ?? 0);
            invLTerm += ((invL[speaker][d] ?? 0) + (alpha[speaker][d] ?? 0) ** 2) * (phi[d] ?? 0);
          }
          logSpeakerProbabilities[speaker] =
            options.fa * (alphaTerm - 0.5 * invLTerm + g[frame]) +
            Math.log((priors[speaker] ?? 0) + 1e-8);
        }
        const rowLogNorm = this.logSumExp(logSpeakerProbabilities);
        for (let speaker = 0; speaker < speakerCount; speaker += 1) {
          const value = Math.exp(logSpeakerProbabilities[speaker] - rowLogNorm);
          nextGamma[frame][speaker] = value;
          nextPriors[speaker] += value;
        }
      }

      const priorSum = nextPriors.reduce((acc, value) => acc + value, 0) || 1;
      gamma = nextGamma;
      priors = nextPriors.map((value) => value / priorSum);
    }

    return { responsibilities: gamma, priors };
  }

  private static computeWeightedCentroids(embeddings: number[][], responsibilities: number[][], priors: number[]) {
    const keptSpeakers = priors
      .map((value, index) => ({ value, index }))
      .filter((entry) => entry.value > 1e-7);
    if (keptSpeakers.length === 0) {
      return [this.averageEmbeddings(embeddings)];
    }

    return keptSpeakers.map(({ index }) => {
      const centroid = new Array<number>(embeddings[0]?.length || 0).fill(0);
      let weightSum = 0;
      for (let row = 0; row < embeddings.length; row += 1) {
        const weight = responsibilities[row][index] ?? 0;
        weightSum += weight;
        for (let d = 0; d < centroid.length; d += 1) {
          centroid[d] += weight * (embeddings[row][d] ?? 0);
        }
      }
      if (weightSum <= 0) return this.averageEmbeddings(embeddings);
      for (let d = 0; d < centroid.length; d += 1) centroid[d] /= weightSum;
      return centroid;
    });
  }

  private static averageEmbeddings(embeddings: number[][]) {
    if (embeddings.length === 0) return [];
    const average = new Array<number>(embeddings[0].length).fill(0);
    for (const embedding of embeddings) {
      for (let i = 0; i < average.length; i += 1) average[i] += embedding[i] ?? 0;
    }
    for (let i = 0; i < average.length; i += 1) average[i] /= embeddings.length;
    return average;
  }

  private static assignEmbeddingsToCentroids(embeddings: number[][], centroids: number[][]) {
    if (centroids.length === 0) return embeddings.map(() => 0);
    return embeddings.map((embedding) => {
      let bestIndex = 0;
      let bestSimilarity = Number.NEGATIVE_INFINITY;
      for (let i = 0; i < centroids.length; i += 1) {
        const similarity = ClusteringService.cosineSimilarity(embedding, centroids[i]);
        if (similarity > bestSimilarity) {
          bestSimilarity = similarity;
          bestIndex = i;
        }
      }
      return bestIndex;
    });
  }

  private static runKMeans(embeddings: number[][], targetClusters: number, maxIterations = 20) {
    if (embeddings.length === 0) return [];
    const normalizedEmbeddings = embeddings.map((embedding) => this.l2Normalize(embedding));
    let centroids = Array.from({ length: Math.min(targetClusters, normalizedEmbeddings.length) }, (_, index) => [
      ...normalizedEmbeddings[Math.floor((index * normalizedEmbeddings.length) / Math.max(1, targetClusters))],
    ]);

    let labels = this.assignEmbeddingsToCentroids(normalizedEmbeddings, centroids);
    for (let iter = 0; iter < maxIterations; iter += 1) {
      const nextCentroids = Array.from({ length: centroids.length }, () => new Array<number>(centroids[0].length).fill(0));
      const counts = new Array<number>(centroids.length).fill(0);
      for (let i = 0; i < normalizedEmbeddings.length; i += 1) {
        const label = labels[i] ?? 0;
        counts[label] += 1;
        for (let d = 0; d < nextCentroids[label].length; d += 1) {
          nextCentroids[label][d] += normalizedEmbeddings[i][d] ?? 0;
        }
      }
      for (let cluster = 0; cluster < nextCentroids.length; cluster += 1) {
        if (counts[cluster] <= 0) {
          nextCentroids[cluster] = [...centroids[cluster]];
          continue;
        }
        for (let d = 0; d < nextCentroids[cluster].length; d += 1) {
          nextCentroids[cluster][d] /= counts[cluster];
        }
      }
      centroids = nextCentroids;
      const nextLabels = this.assignEmbeddingsToCentroids(normalizedEmbeddings, centroids);
      if (nextLabels.every((value, index) => value === labels[index])) break;
      labels = nextLabels;
    }
    return labels;
  }

  private static resolveTargetClusterCount(
    autoClusterCount: number,
    options: Required<PyannoteSpeakerCountOptions>
  ) {
    if (options.exactSpeakerCount) return options.exactSpeakerCount;
    if (autoClusterCount < options.minSpeakers) return options.minSpeakers;
    if (autoClusterCount > options.maxSpeakers) return options.maxSpeakers;
    return null;
  }

  private static normalizeSpeakerCountOptions(options?: PyannoteSpeakerCountOptions): Required<PyannoteSpeakerCountOptions> {
    const exactSpeakerCount = Number.isFinite(options?.exactSpeakerCount as number)
      ? Math.max(1, Math.round(options!.exactSpeakerCount as number))
      : null;
    const minSpeakers = Math.max(
      1,
      Number.isFinite(options?.minSpeakers as number) ? Math.round(options!.minSpeakers as number) : 1
    );
    const maxSpeakers = Math.max(
      minSpeakers,
      Number.isFinite(options?.maxSpeakers as number) ? Math.round(options!.maxSpeakers as number) : 8
    );

    if (exactSpeakerCount) {
      return {
        exactSpeakerCount,
        minSpeakers: exactSpeakerCount,
        maxSpeakers: exactSpeakerCount,
      };
    }

    return {
      exactSpeakerCount: null,
      minSpeakers,
      maxSpeakers,
    };
  }

  private static async clusterWithVbx(embeddings: number[][], options: Required<PyannoteSpeakerCountOptions>) {
    const config = await this.getVbxConfig();
    if (embeddings.length <= 1) {
      return {
        labels: embeddings.map(() => 0),
        threshold: config.threshold,
        vbxSpeakerCount: embeddings.length,
        appliedSpeakerCount: Math.min(Math.max(embeddings.length, 1), options.maxSpeakers),
      };
    }

    const ahcInit = this.clusterAhcByCentroidDistance(embeddings, config.threshold);
    const transformed = embeddings.map((embedding) => this.transformEmbeddingForPlda(embedding, config));
    const { responsibilities, priors } = this.runVbxResponsibilities(transformed, config.phi, ahcInit, {
      fa: config.fa,
      fb: config.fb,
      maxIters: config.max_iters,
      initSmoothing: config.init_smoothing,
    });

    let centroids = this.computeWeightedCentroids(embeddings, responsibilities, priors);
    const autoClusterCount = centroids.length;
    const forcedClusterCount = this.resolveTargetClusterCount(autoClusterCount, options);
    let labels = this.assignEmbeddingsToCentroids(embeddings, centroids);

    if (forcedClusterCount && forcedClusterCount !== autoClusterCount) {
      labels = this.runKMeans(embeddings, forcedClusterCount);
      centroids = this.computeWeightedCentroids(
        embeddings,
        embeddings.map((_, row) => centroids.map((__, cluster) => (labels[row] === cluster ? 1 : 0))),
        new Array(Math.max(...labels) + 1).fill(1)
      );
      labels = this.assignEmbeddingsToCentroids(embeddings, centroids);
    }

    return {
      labels,
      threshold: config.threshold,
      vbxSpeakerCount: autoClusterCount,
      appliedSpeakerCount: new Set(labels).size,
    };
  }

  private static mergeAdjacentTurns(turns: PyannoteSpeakerTurn[]) {
    if (turns.length < 2) return turns;
    const merged: PyannoteSpeakerTurn[] = [turns[0]];
    for (let i = 1; i < turns.length; i += 1) {
      const previous = merged[merged.length - 1];
      const current = turns[i];
      if (
        previous.speaker === current.speaker &&
        current.start - previous.end <= this.mergeGapSec
      ) {
        previous.end = Math.max(previous.end, current.end);
        previous.confidence = Math.max(previous.confidence, current.confidence);
      } else {
        merged.push({ ...current });
      }
    }
    return merged;
  }

  static async diarizeAudio(
    audioPath: string,
    options?: PyannoteSpeakerCountOptions,
    onProgress?: (message: string) => void,
    signal?: AbortSignal
  ): Promise<PyannoteDiarizationResult> {
    this.throwIfAborted(signal);
    await this.ensureModelsPrepared();

    const normalizedSpeakerOptions = this.normalizeSpeakerCountOptions(options);
    const processor = await this.getSegmentationProcessor();
    const fullAudio = await this.extractMonoAudio(audioPath);
    const totalSamples = fullAudio.length;
    const totalDurationSec = totalSamples / this.sampleRate;
    const windowSamples = this.segmentationWindowSec * this.sampleRate;
    const hopSamples = this.segmentationHopSec * this.sampleRate;

    const rawTurns: RawWindowTurn[] = [];
    let windowCount = 0;
    const timelineBinCount = Math.max(1, Math.ceil(totalDurationSec / this.speechTimelineStepSec) + 2);
    const speechTimelineSums = new Array<number>(timelineBinCount).fill(0);
    const speechTimelineCounts = new Array<number>(timelineBinCount).fill(0);

    for (let startSample = 0; startSample < totalSamples; startSample += hopSamples) {
      this.throwIfAborted(signal);
      const endSample = Math.min(totalSamples, startSample + windowSamples);
      const actualWindow = fullAudio.slice(startSample, endSample);
      const paddedWindow = this.padOrTrimWindow(actualWindow, windowSamples);
      const logits = await this.inferSegmentationLogits(paddedWindow);
      this.accumulateSpeechProbabilities(
        logits,
        startSample / this.sampleRate,
        actualWindow.length / this.sampleRate,
        speechTimelineSums,
        speechTimelineCounts
      );
      const tensor = new Tensor('float32', logits, [1, logits.length / 7, 7]);
      const turns = processor.post_process_speaker_diarization(tensor, actualWindow.length)?.[0] || [];

      for (const turn of turns) {
        const start = startSample / this.sampleRate + Number(turn.start || 0);
        const end = startSample / this.sampleRate + Number(turn.end || 0);
        if (end - start < this.minTurnSec) continue;
        rawTurns.push({
          start,
          end,
          confidence: Number(turn.confidence || 0),
        });
      }

      windowCount += 1;
      onProgress?.(`Pyannote window ${windowCount} processed.`);
      if (endSample >= totalSamples) break;
    }

    rawTurns.sort((a, b) => a.start - b.start);
    const uniqueTurns: RawWindowTurn[] = [];
    for (const turn of rawTurns) {
      const previous = uniqueTurns[uniqueTurns.length - 1];
      if (
        previous &&
        Math.abs(previous.start - turn.start) <= 0.15 &&
        Math.abs(previous.end - turn.end) <= 0.15
      ) {
        previous.confidence = Math.max(previous.confidence, turn.confidence);
        continue;
      }
      uniqueTurns.push({ ...turn });
    }

    const speechIntervals = this.buildSpeechIntervals(speechTimelineSums, speechTimelineCounts, totalDurationSec);
    let speechBoundedTurns = this.splitTurnsBySpeechIntervals(uniqueTurns, speechIntervals);
    try {
      this.throwIfAborted(signal);
      const vadSpeechSegments = await this.getVadSpeechSegments(audioPath);
      speechBoundedTurns = this.splitLongTurnsByVad(speechBoundedTurns, vadSpeechSegments);
    } catch (error) {
      if (String(error instanceof Error ? error.message : error || '').toLowerCase().includes('aborted')) {
        throw error;
      }
      console.warn('[Pyannote] VAD refinement unavailable, continuing with segmentation-only turn boundaries:', error);
    }

    const embeddingTargetSamples = this.embeddingWindowSec * this.sampleRate;
    const embeddings: number[][] = [];
    for (let i = 0; i < speechBoundedTurns.length; i += 1) {
      this.throwIfAborted(signal);
      const turn = speechBoundedTurns[i];
      const clip = this.extractSegmentClip(fullAudio, turn.start, turn.end, embeddingTargetSamples);
      embeddings.push(await this.inferEmbedding(clip));
      if (i % 10 === 0 || i === speechBoundedTurns.length - 1) {
        onProgress?.(`Pyannote embeddings ${i + 1}/${speechBoundedTurns.length}.`);
      }
    }

    const similarity = this.computeSimilarityStats(embeddings);
    const clusterThresholdFallback = this.resolveClusterThreshold(similarity);
    let labels: number[] = [];
    let clusterThreshold = clusterThresholdFallback;
    let autoSpeakerCount: number | undefined;
    let clusteringBackend: PyannoteDiarizationDiagnostics['clusteringBackend'] = 'vbx_plda';
    try {
      const clustered = await this.clusterWithVbx(embeddings, normalizedSpeakerOptions);
      labels = clustered.labels;
      clusterThreshold = clustered.threshold;
      autoSpeakerCount = clustered.vbxSpeakerCount;
    } catch (error) {
      console.warn('[Pyannote] VBx/PLDA clustering failed, falling back to cosine clustering:', error);
      clusteringBackend = 'cosine_fallback';
      labels = embeddings.length > 0 ? ClusteringService.cluster(embeddings, clusterThresholdFallback) : [];
      if (new Set(labels).size > normalizedSpeakerOptions.maxSpeakers) {
        labels = ClusteringService.mergeTinyClusters(embeddings, labels, {
          maxClusters: normalizedSpeakerOptions.maxSpeakers,
          minClusterSize: 2,
          minCoverageRatio: 0.04,
        });
      }
      const fallbackSoftCap = normalizedSpeakerOptions.exactSpeakerCount
        ? normalizedSpeakerOptions.exactSpeakerCount
        : Math.min(normalizedSpeakerOptions.maxSpeakers, Math.max(2, normalizedSpeakerOptions.minSpeakers + 2));
      if (new Set(labels).size > fallbackSoftCap) {
        labels = ClusteringService.mergeTinyClusters(embeddings, labels, {
          maxClusters: fallbackSoftCap,
          minClusterSize: 3,
          minCoverageRatio: 0.08,
        });
      }
    }

    const turns = speechBoundedTurns.map((turn, index) => ({
      start: turn.start,
      end: turn.end,
      confidence: turn.confidence,
      speaker: `Speaker ${(labels[index] ?? 0) + 1}`,
    }));
    const mergedTurns = this.mergeAdjacentTurns(turns);

    return {
      turns: mergedTurns,
      diagnostics: {
        modelSource: 'pyannote/speaker-diarization-community-1',
        clusteringStrategy: 'pyannote segmentation + official WeSpeaker resnet weights + local log-mel frontend + VBx/PLDA clustering',
        clusteringBackend,
        audioDurationSec: totalDurationSec,
        windowDurationSec: this.segmentationWindowSec,
        windowHopSec: this.segmentationHopSec,
        windowCount,
        turnCount: mergedTurns.length,
        clusteredSpeakerCount: new Set(labels).size,
        autoSpeakerCount,
        requestedSpeakerCount: {
          exactSpeakerCount: normalizedSpeakerOptions.exactSpeakerCount,
          minSpeakers: normalizedSpeakerOptions.minSpeakers,
          maxSpeakers: normalizedSpeakerOptions.maxSpeakers,
        },
        clusterThreshold,
        similarity,
      },
    };
  }
}
