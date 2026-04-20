import fs from 'fs-extra';
import path from 'path';
import { spawn } from 'node:child_process';
import { WhisperFeatureExtractor } from '@huggingface/transformers';
import { buildWhisperLanguageToken, resolveQwenAsrLanguageName } from './language/resolver.js';
import { OpenvinoBackend } from './openvino_backend.js';
import { PathManager } from './path_manager.js';
import { getBundledToolPath, resolveToolCommand } from './runtime_tools.js';
import { SystemMonitor, type AcceleratorSnapshot, type SystemResourceSnapshot } from './services/system_monitor.js';
import type { LocalTranslateGenerationOptions } from './services/local_llm/types.js';

interface HelperRequestPayload {
  requestId: string;
  method: string;
  params?: Record<string, unknown>;
}

interface HelperResponsePayload {
  requestId?: string;
  ok?: boolean;
  result?: any;
  error?: string;
}

interface LocalAsrChunk {
  start_ts: number;
  end_ts?: number;
  text: string;
}

interface Qwen3AsrPromptTemplate {
  prefix_ids: number[];
  suffix_ids: number[];
  n_audio_tokens: number;
  n_samples: number;
  nb_frames: number;
  audio_pad_id: number;
  eos_id?: number;
  eot_id?: number;
  asr_text_id?: number;
  special_ids?: number[];
  language_suffix_ids?: Record<string, number[]>;
  supported_languages?: string[];
}

interface Qwen3AsrTokenizerConfig {
  added_tokens_decoder?: Record<string, { content?: string; special?: boolean }>;
}

interface Qwen3AsrGenerationConfig {
  eos_token_id?: number | number[];
  pad_token_id?: number;
}

interface Qwen3AsrPreprocessorConfig {
  feature_extractor_type?: string;
  feature_size?: number;
  hop_length?: number;
  n_fft?: number;
  n_samples?: number;
  nb_max_frames?: number;
  sampling_rate?: number;
  chunk_length?: number;
  dither?: number;
  padding_side?: string;
  padding_value?: number;
  return_attention_mask?: boolean;
}

interface Qwen3AsrCompiledModel {
  compiledModel: any;
  inputNames: string[];
  outputNames: string[];
}

interface Qwen3AsrRuntime {
  modelId: string;
  modelPath: string;
  featureExtractor: any;
  audioEncoder: Qwen3AsrCompiledModel;
  thinkerEmbeddings: Qwen3AsrCompiledModel;
  decoderPrefill: Qwen3AsrCompiledModel;
  decoderKv: Qwen3AsrCompiledModel;
  promptTemplate: Qwen3AsrPromptTemplate;
  preprocessorConfig: Qwen3AsrPreprocessorConfig;
  tokenizerConfig: Qwen3AsrTokenizerConfig;
  generationConfig: Qwen3AsrGenerationConfig | null;
  tokenIdToContent: Map<number, string>;
  vocabSize: number;
  specialTokenIds: Set<number>;
  eosTokenIds: Set<number>;
  supportedLanguages: Map<string, string>;
}

class OpenVinoAsrHelperClient {
  private static child: ReturnType<typeof spawn> | null = null;
  private static requestQueue: Promise<void> = Promise.resolve();
  private static pending = new Map<
    string,
    {
      resolve: (value: any) => void;
      reject: (error: Error) => void;
      timeout: NodeJS.Timeout;
    }
  >();
  private static readBuffer = '';
  private static stderrTail: string[] = [];
  private static requestSeq = 0;
  private static expectedExitPids = new Set<number>();

  private static getTimeoutMs(method: string, fallbackMs: number) {
    const keyMap: Record<string, string> = {
      health: 'OPENVINO_HELPER_HEALTH_TIMEOUT_MS',
      load: 'OPENVINO_HELPER_LOAD_TIMEOUT_MS',
      transcribe: 'OPENVINO_HELPER_TRANSCRIBE_TIMEOUT_MS',
      convertOfficialQwen3Asr: 'OPENVINO_HELPER_CONVERT_TIMEOUT_MS',
      unload: 'OPENVINO_HELPER_UNLOAD_TIMEOUT_MS',
      shutdown: 'OPENVINO_HELPER_SHUTDOWN_TIMEOUT_MS',
    };
    const envKey = keyMap[method];
    if (!envKey) return fallbackMs;
    const raw = process.env[envKey];
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed < 1000) return fallbackMs;
    return Math.floor(parsed);
  }

  private static terminateHelper(reason: string) {
    const target = this.child;
    if (target && !target.killed) {
      if (typeof target.pid === 'number' && target.pid > 0) {
        this.expectedExitPids.add(target.pid);
      }
      try {
        target.kill();
      } catch {
        // Fall through to hard kill path.
      }

      if (process.platform === 'win32' && target.pid) {
        // Ensure GPU context is released even if child ignores normal termination.
        const killer = spawn('taskkill', ['/PID', String(target.pid), '/T', '/F'], {
          windowsHide: true,
          stdio: 'ignore',
          detached: true,
        });
        killer.unref();
      }
    }
    this.child = null;
    this.readBuffer = '';
    this.stderrTail = [];
    this.rejectAllPending(reason);
  }

  private static getBundledHelperPath() {
    return getBundledToolPath('openvino_whisper_helper');
  }

  private static getSourceHelperPath() {
    return PathManager.resolveToolsSourcePath('openvino_whisper_helper.py');
  }

  private static getPythonCommand() {
    const configured = String(process.env.OPENVINO_HELPER_PYTHON || '').trim();
    return configured || 'python';
  }

  private static getHelperLaunchSpec() {
    const bundledPath = this.getBundledHelperPath();
    const sourcePath = this.getSourceHelperPath();
    const preferBundled = String(process.env.OPENVINO_HELPER_USE_EXE || '').trim() === '1';

    if (!preferBundled && fs.existsSync(sourcePath)) {
      return {
        command: this.getPythonCommand(),
        args: [sourcePath],
        displayPath: sourcePath,
        requiredPath: sourcePath,
      };
    }

    if (fs.existsSync(bundledPath)) {
      return {
        command: bundledPath,
        args: [] as string[],
        displayPath: bundledPath,
        requiredPath: bundledPath,
      };
    }

    return {
      command: this.getPythonCommand(),
      args: [sourcePath],
      displayPath: sourcePath,
      requiredPath: sourcePath,
    };
  }

  static getHelperPath() {
    return this.getHelperLaunchSpec().displayPath;
  }

  private static rejectAllPending(message: string) {
    for (const [, item] of this.pending) {
      clearTimeout(item.timeout);
      item.reject(new Error(message));
    }
    this.pending.clear();
  }

  private static handleStdoutChunk(chunk: Buffer | string) {
    this.readBuffer += chunk.toString('utf8');
    let lineEnd = this.readBuffer.indexOf('\n');

    while (lineEnd >= 0) {
      const rawLine = this.readBuffer.slice(0, lineEnd).trim();
      this.readBuffer = this.readBuffer.slice(lineEnd + 1);
      lineEnd = this.readBuffer.indexOf('\n');
      if (!rawLine) continue;

      let payload: HelperResponsePayload | null = null;
      try {
        payload = JSON.parse(rawLine) as HelperResponsePayload;
      } catch {
        payload = null;
      }
      if (!payload?.requestId) continue;

      const pendingItem = this.pending.get(payload.requestId);
      if (!pendingItem) continue;

      clearTimeout(pendingItem.timeout);
      this.pending.delete(payload.requestId);

      if (payload.ok) {
        pendingItem.resolve(payload.result ?? {});
      } else {
        pendingItem.reject(new Error(String(payload.error || 'ASR helper returned an unknown error.')));
      }
    }
  }

  private static async ensureStarted() {
    if (this.child && !this.child.killed) return this.child;

    const launchSpec = this.getHelperLaunchSpec();
    if (!(await fs.pathExists(launchSpec.requiredPath))) {
      throw new Error(`ASR helper not found: ${launchSpec.displayPath}`);
    }

    this.readBuffer = '';
    this.stderrTail = [];
    const child = spawn(launchSpec.command, launchSpec.args, {
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
      env: {
        ...process.env,
        PYTHONIOENCODING: 'utf-8',
      },
    });
    this.child = child;

    child.stdout?.on('data', (chunk) => this.handleStdoutChunk(chunk));
    child.stderr?.on('data', (chunk) => {
      const text = chunk.toString('utf8').trim();
      if (!text) return;
      console.warn('[openvino-asr-helper]', text);
      for (const line of text.split(/\r?\n/).map((item) => item.trim()).filter(Boolean)) {
        this.stderrTail.push(line);
      }
      if (this.stderrTail.length > 40) {
        this.stderrTail = this.stderrTail.slice(-40);
      }
    });
    child.on('exit', (code, signal) => {
      const childPid = typeof child.pid === 'number' ? child.pid : 0;
      const expectedExit = childPid > 0 ? this.expectedExitPids.delete(childPid) : false;
      const isCurrentChild = this.child === child;
      if (isCurrentChild) {
        this.child = null;
      }
      if (expectedExit || !isCurrentChild) {
        return;
      }
      const stderrDetail = this.stderrTail.length > 0 ? `\n${this.stderrTail.join('\n')}` : '';
      this.rejectAllPending(
        `ASR helper exited unexpectedly (code=${code ?? 'null'}, signal=${signal ?? 'null'}).${stderrDetail}`
      );
    });
    child.on('error', (error) => {
      const childPid = typeof child.pid === 'number' ? child.pid : 0;
      const expectedExit = childPid > 0 ? this.expectedExitPids.delete(childPid) : false;
      const isCurrentChild = this.child === child;
      if (isCurrentChild) {
        this.child = null;
      }
      if (expectedExit || !isCurrentChild) {
        return;
      }
      this.rejectAllPending(`ASR helper error: ${error.message}`);
    });

    return child;
  }

  private static async requestNow(method: string, params: Record<string, unknown> = {}, timeoutMs = 180000) {
    const child = await this.ensureStarted();
    if (!child.stdin) {
      throw new Error('ASR helper stdin is not available.');
    }
    const effectiveTimeoutMs = this.getTimeoutMs(method, timeoutMs);

    const requestId = `req_${Date.now()}_${++this.requestSeq}`;
    const payload: HelperRequestPayload = { requestId, method, params };
    const body = `${JSON.stringify(payload)}\n`;

    return new Promise<any>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(requestId);
        this.terminateHelper(`ASR helper request timed out (${method}).`);
        reject(new Error(`ASR helper request timed out (${method}).`));
      }, effectiveTimeoutMs);

      this.pending.set(requestId, { resolve, reject, timeout });
      child.stdin!.write(body, 'utf8', (error) => {
        if (!error) return;
        clearTimeout(timeout);
        this.pending.delete(requestId);
        reject(new Error(`Failed to send request to ASR helper: ${error.message}`));
      });
    });
  }

  private static enqueueRequest<T>(task: () => Promise<T>) {
    const run = this.requestQueue.then(task, task);
    this.requestQueue = run.then(
      () => undefined,
      () => undefined
    );
    return run;
  }

  private static async request(method: string, params: Record<string, unknown> = {}, timeoutMs = 180000) {
    return this.enqueueRequest(() => this.requestNow(method, params, timeoutMs));
  }

  static async healthCheck(timeoutMs = 30000) {
    return this.request('health', {}, timeoutMs);
  }

  static async loadModel(modelPath: string, device = 'AUTO', cacheDir?: string, timeoutMs = 60000) {
    return this.request('load', { modelPath, device, cacheDir }, timeoutMs);
  }

  static async transcribe(input: {
    audioPath: string;
    language?: string;
    prompt?: string;
    returnTimestamps?: boolean;
    wordTimestamps?: boolean;
    task?: string;
  }, timeoutMs?: number) {
    // Large local files may need long processing; default 30 minutes.
    return this.request('transcribe', input, timeoutMs ?? 1800000);
  }

  static async convertOfficialQwen3Asr(input: {
    repoId: string;
    outputDir: string;
    useLocalDir?: boolean;
  }) {
    return this.request('convertOfficialQwen3Asr', input, 7200000);
  }

  static async unloadModel() {
    return this.request('unload', {}, 15000);
  }

  static async shutdown() {
    await this.enqueueRequest(async () => {
      try {
        await this.requestNow('shutdown', {}, 4000);
      } catch {
        // If helper does not support shutdown command, fallback to process kill.
      } finally {
        this.terminateHelper('ASR helper was shut down.');
      }
    });
  }

  static forceShutdownNow(reason = 'ASR helper was forcefully shut down.') {
    this.terminateHelper(reason);
  }
}

type OpenvinoPipelineFactory = (
  modelPath: string,
  device?: string,
  properties?: Record<string, unknown>
) => Promise<any> | any;
type TranslatePipelineKind = 'llm' | 'vlm' | 'seq2seq';

class OpenVinoTranslateHelperClient {
  private static child: ReturnType<typeof spawn> | null = null;
  private static requestQueue: Promise<void> = Promise.resolve();
  private static pending = new Map<
    string,
    {
      resolve: (value: any) => void;
      reject: (error: Error) => void;
      timeout: NodeJS.Timeout;
    }
  >();
  private static readBuffer = '';
  private static requestSeq = 0;
  private static expectedExitPids = new Set<number>();

  private static getTimeoutMs(method: string, fallbackMs: number) {
    const keyMap: Record<string, string> = {
      health: 'OPENVINO_TRANSLATE_HELPER_HEALTH_TIMEOUT_MS',
      load: 'OPENVINO_TRANSLATE_HELPER_LOAD_TIMEOUT_MS',
      generate: 'OPENVINO_TRANSLATE_HELPER_GENERATE_TIMEOUT_MS',
      unload: 'OPENVINO_TRANSLATE_HELPER_UNLOAD_TIMEOUT_MS',
      shutdown: 'OPENVINO_TRANSLATE_HELPER_SHUTDOWN_TIMEOUT_MS',
    };
    const envKey = keyMap[method];
    if (!envKey) return fallbackMs;
    const raw = process.env[envKey];
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed < 1000) return fallbackMs;
    return Math.floor(parsed);
  }

  private static getPythonCommand() {
    const configured = String(process.env.OPENVINO_HELPER_PYTHON || '').trim();
    return configured || 'python';
  }

  private static getHelperPath() {
    return PathManager.resolveToolsSourcePath('openvino_translate_helper.py');
  }

  private static rejectAllPending(message: string) {
    for (const [, item] of this.pending) {
      clearTimeout(item.timeout);
      item.reject(new Error(message));
    }
    this.pending.clear();
  }

  private static terminateHelper(reason: string) {
    const target = this.child;
    if (target && !target.killed) {
      if (typeof target.pid === 'number' && target.pid > 0) {
        this.expectedExitPids.add(target.pid);
      }
      try {
        target.kill();
      } catch {
        // Ignore and continue with cleanup.
      }

      if (process.platform === 'win32' && target.pid) {
        const killer = spawn('taskkill', ['/PID', String(target.pid), '/T', '/F'], {
          windowsHide: true,
          stdio: 'ignore',
          detached: true,
        });
        killer.unref();
      }
    }
    this.child = null;
    this.readBuffer = '';
    this.rejectAllPending(reason);
  }

  private static handleStdoutChunk(chunk: Buffer | string) {
    this.readBuffer += chunk.toString('utf8');
    let lineEnd = this.readBuffer.indexOf('\n');

    while (lineEnd >= 0) {
      const rawLine = this.readBuffer.slice(0, lineEnd).trim();
      this.readBuffer = this.readBuffer.slice(lineEnd + 1);
      lineEnd = this.readBuffer.indexOf('\n');
      if (!rawLine) continue;

      let payload: HelperResponsePayload | null = null;
      try {
        payload = JSON.parse(rawLine) as HelperResponsePayload;
      } catch {
        continue;
      }
      if (!payload?.requestId) continue;
      const pending = this.pending.get(payload.requestId);
      if (!pending) continue;

      clearTimeout(pending.timeout);
      this.pending.delete(payload.requestId);
      if (payload.ok) {
        pending.resolve(payload.result);
      } else {
        pending.reject(new Error(payload.error || 'OpenVINO translate helper request failed.'));
      }
    }
  }

  private static ensureHelper() {
    if (this.child && !this.child.killed) {
      return this.child;
    }

    const helperPath = this.getHelperPath();
    if (!fs.existsSync(helperPath)) {
      throw new Error(`OpenVINO translate helper not found: ${helperPath}`);
    }

    const child = spawn(this.getPythonCommand(), [helperPath], {
      cwd: PathManager.getRoot(),
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: {
        ...process.env,
        PYTHONIOENCODING: 'utf-8',
        PYTHONUTF8: '1',
      },
    });

    child.stdout.on('data', (chunk) => this.handleStdoutChunk(chunk));
    child.stderr.on('data', (chunk) => {
      const text = chunk.toString('utf8').trim();
      if (text) {
        console.warn('[openvino-translate-helper]', text);
      }
    });
    child.on('error', (error) => {
      this.terminateHelper(`Translate helper failed: ${error.message}`);
    });
    child.on('exit', (code, signal) => {
      const pid = typeof child.pid === 'number' ? child.pid : 0;
      const expected = pid > 0 && this.expectedExitPids.delete(pid);
      this.child = null;
      this.readBuffer = '';
      if (expected) {
        this.rejectAllPending('Translate helper was restarted.');
        return;
      }
      this.rejectAllPending(
        `Translate helper exited unexpectedly (code=${String(code)}, signal=${String(signal)}).`
      );
    });

    this.child = child;
    return child;
  }

  private static async requestNow(method: string, params: Record<string, unknown>, timeoutMs: number) {
    const child = this.ensureHelper();
    if (!child.stdin || child.stdin.destroyed) {
      this.terminateHelper('Translate helper stdin is unavailable.');
      throw new Error('Translate helper stdin is unavailable.');
    }

    const requestId = `translate-${Date.now()}-${++this.requestSeq}`;
    return await new Promise<any>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(requestId);
        const error = new Error(`Translate helper request timed out after ${timeoutMs} ms (${method}).`);
        this.terminateHelper(error.message);
        reject(error);
      }, timeoutMs);

      this.pending.set(requestId, {
        resolve,
        reject,
        timeout,
      });

      try {
        child.stdin.write(`${JSON.stringify({ requestId, method, params })}\n`, 'utf8');
      } catch (error: any) {
        clearTimeout(timeout);
        this.pending.delete(requestId);
        reject(new Error(`Failed to write to translate helper stdin: ${String(error?.message || error)}`));
      }
    });
  }

  private static async enqueueRequest<T>(fn: () => Promise<T>) {
    const previous = this.requestQueue;
    let release: (() => void) | null = null;
    this.requestQueue = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous.catch(() => {});
    try {
      return await fn();
    } finally {
      release?.();
    }
  }

  private static async request(method: string, params: Record<string, unknown>, fallbackMs: number) {
    const timeoutMs = this.getTimeoutMs(method, fallbackMs);
    return this.enqueueRequest(() => this.requestNow(method, params, timeoutMs));
  }

  static async healthCheck(timeoutMs = 5000) {
    return this.request('health', {}, timeoutMs);
  }

  static async loadModel(modelPath: string, device: string, timeoutMs = 180000) {
    return this.request('load', { modelPath, device }, timeoutMs);
  }

  static async generate(prompt: string, generationConfig: Record<string, unknown>) {
    return this.request('generate', { prompt, generationConfig }, 300000);
  }

  static async unload() {
    return this.request('unload', {}, 15000);
  }

  static async shutdown() {
    await this.enqueueRequest(async () => {
      try {
        await this.requestNow('shutdown', {}, 4000);
      } catch {
        // Fallback to kill below.
      } finally {
        this.terminateHelper('Translate helper was shut down.');
      }
    });
  }

  static forceShutdownNow(reason = 'Translate helper was forcefully shut down.') {
    this.terminateHelper(reason);
  }
}

class OpenVinoGenaiTranslateHelperClient {
  private static child: ReturnType<typeof spawn> | null = null;
  private static requestQueue: Promise<void> = Promise.resolve();
  private static pending = new Map<
    string,
    {
      resolve: (value: any) => void;
      reject: (error: Error) => void;
      timeout: NodeJS.Timeout;
    }
  >();
  private static readBuffer = '';
  private static requestSeq = 0;
  private static expectedExitPids = new Set<number>();

  private static getTimeoutMs(method: string, fallbackMs: number) {
    const keyMap: Record<string, string> = {
      health: 'OPENVINO_TRANSLATE_HELPER_HEALTH_TIMEOUT_MS',
      load: 'OPENVINO_TRANSLATE_HELPER_LOAD_TIMEOUT_MS',
      generate: 'OPENVINO_TRANSLATE_HELPER_GENERATE_TIMEOUT_MS',
      unload: 'OPENVINO_TRANSLATE_HELPER_UNLOAD_TIMEOUT_MS',
      shutdown: 'OPENVINO_TRANSLATE_HELPER_SHUTDOWN_TIMEOUT_MS',
    };
    const envKey = keyMap[method];
    if (!envKey) return fallbackMs;
    const raw = process.env[envKey];
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed < 1000) return fallbackMs;
    return Math.floor(parsed);
  }

  private static getHelperPath() {
    return PathManager.resolveToolsSourcePath('openvino_genai_translate_helper.mjs');
  }

  private static rejectAllPending(message: string) {
    for (const [, item] of this.pending) {
      clearTimeout(item.timeout);
      item.reject(new Error(message));
    }
    this.pending.clear();
  }

  private static terminateHelper(reason: string) {
    const target = this.child;
    if (target && !target.killed) {
      if (typeof target.pid === 'number' && target.pid > 0) {
        this.expectedExitPids.add(target.pid);
      }
      try {
        target.kill();
      } catch {
        // Ignore and continue cleanup.
      }

      if (process.platform === 'win32' && target.pid) {
        const killer = spawn('taskkill', ['/PID', String(target.pid), '/T', '/F'], {
          windowsHide: true,
          stdio: 'ignore',
          detached: true,
        });
        killer.unref();
      }
    }
    this.child = null;
    this.readBuffer = '';
    this.rejectAllPending(reason);
  }

  private static handleStdoutChunk(chunk: Buffer | string) {
    this.readBuffer += chunk.toString('utf8');
    let lineEnd = this.readBuffer.indexOf('\n');

    while (lineEnd >= 0) {
      const rawLine = this.readBuffer.slice(0, lineEnd).trim();
      this.readBuffer = this.readBuffer.slice(lineEnd + 1);
      lineEnd = this.readBuffer.indexOf('\n');
      if (!rawLine) continue;

      let payload: HelperResponsePayload | null = null;
      try {
        payload = JSON.parse(rawLine) as HelperResponsePayload;
      } catch {
        continue;
      }
      if (!payload?.requestId) continue;
      const pending = this.pending.get(payload.requestId);
      if (!pending) continue;

      clearTimeout(pending.timeout);
      this.pending.delete(payload.requestId);
      if (payload.ok) {
        pending.resolve(payload.result);
      } else {
        pending.reject(new Error(payload.error || 'OpenVINO GenAI translate helper request failed.'));
      }
    }
  }

  private static ensureHelper() {
    if (this.child && !this.child.killed) {
      return this.child;
    }

    const helperPath = this.getHelperPath();
    if (!fs.existsSync(helperPath)) {
      throw new Error(`OpenVINO GenAI translate helper not found: ${helperPath}`);
    }

    const child = spawn(process.execPath, [helperPath], {
      cwd: PathManager.getRoot(),
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: {
        ...process.env,
        NODE_NO_WARNINGS: process.env.NODE_NO_WARNINGS || '1',
      },
    });

    child.stdout.on('data', (chunk) => this.handleStdoutChunk(chunk));
    child.stderr.on('data', (chunk) => {
      const text = chunk.toString('utf8').trim();
      if (text) {
        console.warn('[openvino-genai-translate-helper]', text);
      }
    });
    child.on('error', (error) => {
      this.terminateHelper(`OpenVINO GenAI translate helper failed: ${error.message}`);
    });
    child.on('exit', (code, signal) => {
      const pid = typeof child.pid === 'number' ? child.pid : 0;
      const expected = pid > 0 && this.expectedExitPids.delete(pid);
      this.child = null;
      this.readBuffer = '';
      if (expected) {
        this.rejectAllPending('OpenVINO GenAI translate helper was restarted.');
        return;
      }
      this.rejectAllPending(
        `OpenVINO GenAI translate helper exited unexpectedly (code=${String(code)}, signal=${String(signal)}).`
      );
    });

    this.child = child;
    return child;
  }

  private static async requestNow(method: string, params: Record<string, unknown>, timeoutMs: number) {
    const child = this.ensureHelper();
    if (!child.stdin || child.stdin.destroyed) {
      this.terminateHelper('OpenVINO GenAI translate helper stdin is unavailable.');
      throw new Error('OpenVINO GenAI translate helper stdin is unavailable.');
    }

    const requestId = `genai-translate-${Date.now()}-${++this.requestSeq}`;
    return await new Promise<any>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(requestId);
        const error = new Error(`OpenVINO GenAI translate helper request timed out after ${timeoutMs} ms (${method}).`);
        this.terminateHelper(error.message);
        reject(error);
      }, timeoutMs);

      this.pending.set(requestId, {
        resolve,
        reject,
        timeout,
      });

      try {
        child.stdin.write(`${JSON.stringify({ requestId, method, params })}\n`, 'utf8');
      } catch (error: any) {
        clearTimeout(timeout);
        this.pending.delete(requestId);
        reject(new Error(`Failed to write to OpenVINO GenAI translate helper stdin: ${String(error?.message || error)}`));
      }
    });
  }

  private static async enqueueRequest<T>(fn: () => Promise<T>) {
    const previous = this.requestQueue;
    let release: (() => void) | null = null;
    this.requestQueue = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous.catch(() => {});
    try {
      return await fn();
    } finally {
      release?.();
    }
  }

  private static async request(method: string, params: Record<string, unknown>, fallbackMs: number) {
    const timeoutMs = this.getTimeoutMs(method, fallbackMs);
    return this.enqueueRequest(() => this.requestNow(method, params, timeoutMs));
  }

  static async healthCheck(timeoutMs = 5000) {
    return this.request('health', {}, timeoutMs);
  }

  static async loadModel(
    modelPath: string,
    device: string,
    runtimeKind: 'llm' | 'vlm',
    properties: Record<string, unknown>,
    timeoutMs = 180000
  ) {
    return this.request('load', { modelPath, device, runtimeKind, properties }, timeoutMs);
  }

  static async generate(input: { prompt?: string; messages?: Array<Record<string, unknown>> }, generationConfig: Record<string, unknown>) {
    return this.request(
      'generate',
      {
        prompt: input.prompt || '',
        messages: Array.isArray(input.messages) ? input.messages : undefined,
        generationConfig,
      },
      300000
    );
  }

  static async unload() {
    return this.request('unload', {}, 15000);
  }

  static async shutdown() {
    await this.enqueueRequest(async () => {
      try {
        await this.requestNow('shutdown', {}, 4000);
      } catch {
        // Fallback to kill below.
      } finally {
        this.terminateHelper('OpenVINO GenAI translate helper was shut down.');
      }
    });
  }

  static forceShutdownNow(reason = 'OpenVINO GenAI translate helper was forcefully shut down.') {
    this.terminateHelper(reason);
  }
}

export interface OpenvinoAcceleratorInference {
  observedAt: string;
  source: 'model-load-delta' | 'model-load-snapshot' | 'post-generate-snapshot';
  acceleratorModel: string;
  luid?: string;
  memorySource?: 'dedicated' | 'shared';
  vramUsedGB?: number;
  vramTotalGB?: number;
  utilization?: number;
  physIndex?: number;
}

export interface OpenvinoTranslateRuntimeDebug {
  modelId: string;
  modelPath: string;
  pipelineKind: TranslatePipelineKind;
  requestedDevice: string;
  pipelineDevice: string | null;
  cacheDir: string | null;
  promptLookupEnabled: boolean;
  schedulerConfig: Record<string, unknown> | null;
  loadInference: OpenvinoAcceleratorInference | null;
  lastInference: OpenvinoAcceleratorInference | null;
  lastPerfMetrics: {
    loadTimeMs: number | null;
    generatedTokens: number | null;
    inputTokens: number | null;
    ttftMs: number | null;
    tpotMs: number | null;
    ipotMs: number | null;
    throughputTokensPerSec: number | null;
    generateDurationMs: number | null;
    inferenceDurationMs: number | null;
    tokenizationDurationMs: number | null;
    detokenizationDurationMs: number | null;
  } | null;
}

interface CachedTranslatePipeline {
  kind: TranslatePipelineKind;
  pipeline: any;
  runtimeDebug: OpenvinoTranslateRuntimeDebug;
}

interface CachedAsrPipeline {
  modelId: string;
  modelPath: string;
  pipeline: any;
  requestedDevice: string;
  cacheDir: string | null;
  loadMs: number;
  wordTimestampsEnabled: boolean;
}

interface LocalAsrWordTiming {
  text: string;
  start_ts: number;
  end_ts?: number;
}

interface LocalAsrSegment extends LocalAsrChunk {
  words?: LocalAsrWordTiming[];
}

interface LocalAudioDecodeCacheEntry {
  cacheKey: string;
  audio: Float32Array;
  sampleCount: number;
  cachedAt: number;
}

interface LocalAudioDecodeResult {
  audio: Float32Array;
  decodeMs: number;
  cacheHit: boolean;
  sampleCount: number;
  cacheKey: string;
}

export class OpenvinoRuntimeManager {
  private static readonly utf8Decoder = new TextDecoder('utf-8');
  private static translatePipelineCache = new Map<string, CachedTranslatePipeline>();
  private static activeTranslateModelId: string | null = null;
  private static llmTranslatePipelineFactory: OpenvinoPipelineFactory | null = null;
  private static vlmTranslatePipelineFactory: OpenvinoPipelineFactory | null = null;
  private static whisperMultilingualCache = new Map<string, boolean>();
  private static whisperAsrPipelineFactory: OpenvinoPipelineFactory | null = null;
  private static asrPipelineCache: CachedAsrPipeline | null = null;
  private static qwenAsrRuntimeCache: Qwen3AsrRuntime | null = null;
  private static qwenByteDecoderMap: Map<string, number> | null = null;
  private static openvinoStatusCache: { at: number; value: any } | null = null;
  private static audioDecodeCache = new Map<string, LocalAudioDecodeCacheEntry>();

  private static async getInstalledNodePackageVersion(packageName: string) {
    try {
      const packageJsonPath = PathManager.resolveNodeModulesPath(packageName, 'package.json');
      const packageJson = await fs.readJson(packageJsonPath);
      const version = String(packageJson?.version || '').trim();
      return version || null;
    } catch {
      return null;
    }
  }

  private static log(message: string, extra?: Record<string, unknown>) {
    const payload = extra ? ` ${JSON.stringify(extra)}` : '';
    console.log(`[${new Date().toISOString()}] [LocalRuntime] ${message}${payload}`);
  }

  private static shouldTraceLocalTranslate() {
    return this.getEnvBoolean('OPENVINO_LOCAL_TRANSLATE_TRACE', false);
  }

  private static traceLocalTranslate(message: string, extra?: Record<string, unknown>) {
    if (!this.shouldTraceLocalTranslate()) return;
    this.log(`translate-trace ${message}`, extra);
  }

  private static getEnvNumber(name: string, fallback: number, min?: number, max?: number) {
    const raw = process.env[name];
    if (typeof raw !== 'string' || !raw.trim()) return fallback;
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) return fallback;
    if (typeof min === 'number' && parsed < min) return min;
    if (typeof max === 'number' && parsed > max) return max;
    return parsed;
  }

  private static getEnvBoolean(name: string, fallback = false) {
    const raw = String(process.env[name] || '').trim();
    if (!raw) return fallback;
    if (/^(1|true|yes|on)$/i.test(raw)) return true;
    if (/^(0|false|no|off)$/i.test(raw)) return false;
    return fallback;
  }

  private static getAsrTimeoutMs(kind: 'load' | 'generate') {
    if (kind === 'load') {
      const direct = this.getEnvNumber('OPENVINO_ASR_LOAD_TIMEOUT_MS', 0, 0, 300000);
      if (direct > 0) return direct;
      return this.getEnvNumber('OPENVINO_HELPER_LOAD_TIMEOUT_MS', 60000, 1000, 300000);
    }

    const direct = this.getEnvNumber('OPENVINO_ASR_TIMEOUT_MS', 0, 0, 7200000);
    if (direct > 0) return direct;
    return this.getEnvNumber('OPENVINO_HELPER_TRANSCRIBE_TIMEOUT_MS', 1800000, 1000, 7200000);
  }

  private static getTranslateLoadTimeoutMs(kind: TranslatePipelineKind) {
    if (kind === 'llm' || kind === 'vlm') {
      return this.getEnvNumber('OPENVINO_TRANSLATE_HELPER_HEAVY_LOAD_TIMEOUT_MS', 600000, 30000, 1800000);
    }
    return this.getEnvNumber('OPENVINO_TRANSLATE_HELPER_LOAD_TIMEOUT_MS', 180000, 1000, 1800000);
  }

  private static normalizeRequestedDevice(raw: string | null | undefined, fallback = 'AUTO') {
    const normalized = String(raw || '')
      .trim()
      .toUpperCase();
    return normalized || fallback;
  }

  private static getConfiguredLocalAsrDevice() {
    return this.normalizeRequestedDevice(
      process.env.OPENVINO_LOCAL_ASR_DEVICE,
      ''
    );
  }

  private static isOpenvinoGpuAllocationLimitError(error: unknown) {
    const message = String((error as any)?.message || error || '');
    return (
      /Exceeded max size of memory object allocation/i.test(message) ||
      (/intel_gpu/i.test(message) && /max alloc size/i.test(message)) ||
      (/requested .* bytes/i.test(message) && /max alloc size supported by device/i.test(message))
    );
  }

  private static shouldFallbackQwenAsrToCpu(requestedDevice: string, error: unknown) {
    const normalized = this.normalizeRequestedDevice(requestedDevice, 'AUTO');
    if (normalized === 'CPU') return false;
    return this.isOpenvinoGpuAllocationLimitError(error);
  }

  private static isOpenvinoRemoteTensorNotImplementedError(error: unknown) {
    const message = String((error as any)?.message || error || '');
    return (
      /iremote_tensor\.hpp/i.test(message) ||
      (/remote[_ ]tensor/i.test(message) && /not implemented/i.test(message)) ||
      (/whisperPerformInferenceThread/i.test(message) && /Not Implemented/i.test(message))
    );
  }

  private static shouldFallbackWhisperAsrToCpu(requestedDevice: string, error: unknown) {
    const normalized = this.normalizeRequestedDevice(requestedDevice, 'AUTO');
    if (normalized === 'CPU') return false;
    return this.isOpenvinoGpuAllocationLimitError(error) || this.isOpenvinoRemoteTensorNotImplementedError(error);
  }

  private static shouldFallbackTranslateToCpu(requestedDevice: string, error: unknown) {
    const normalized = this.normalizeRequestedDevice(requestedDevice, 'AUTO');
    if (normalized === 'CPU') return false;
    return this.isOpenvinoGpuAllocationLimitError(error);
  }

  private static async loadAsrHelperModelWithFallback(input: {
    modelPath: string;
    requestedDevice: string;
    cacheDir: string;
    helperRuntimeKind: 'qwen3_asr_official' | 'ctc_asr' | null;
  }) {
    const requestedDevice = this.normalizeRequestedDevice(input.requestedDevice, 'AUTO');
    const loadTimeoutMs =
      input.helperRuntimeKind === 'qwen3_asr_official' || input.helperRuntimeKind === 'ctc_asr'
        ? this.getEnvNumber('OPENVINO_HELPER_HEAVY_LOAD_TIMEOUT_MS', 600000, 30000, 1800000)
        : this.getEnvNumber('OPENVINO_HELPER_LOAD_TIMEOUT_MS', 180000, 1000, 1800000);
    try {
      await OpenVinoAsrHelperClient.loadModel(input.modelPath, requestedDevice, input.cacheDir, loadTimeoutMs);
      return {
        effectiveDevice: requestedDevice,
        fallbackToCpu: false,
      };
    } catch (error) {
      const serializationCacheError =
        input.cacheDir && this.isOpenvinoSerializationCacheError(error);
      if (serializationCacheError) {
        this.log('asr helper load hit serialization cache error, retrying without cache', {
          requestedDevice,
          modelPath: input.modelPath,
          cacheDir: input.cacheDir,
          message: String((error as any)?.message || error || 'OpenVINO serialization cache failure'),
        });
        try {
          await fs.remove(input.cacheDir);
        } catch {
          // Ignore cache cleanup failures and still retry without CACHE_DIR.
        }
        OpenVinoAsrHelperClient.forceShutdownNow('Retrying ASR helper without CACHE_DIR after serialization failure.');
        await OpenVinoAsrHelperClient.loadModel(input.modelPath, requestedDevice, undefined, loadTimeoutMs);
        return {
          effectiveDevice: requestedDevice,
          fallbackToCpu: false,
        };
      }

      if (input.helperRuntimeKind !== 'qwen3_asr_official' || !this.shouldFallbackQwenAsrToCpu(requestedDevice, error)) {
        throw error;
      }

      this.log('qwen asr helper gpu allocation failed, retrying on cpu', {
        requestedDevice,
        modelPath: input.modelPath,
        message: String((error as any)?.message || error || 'OpenVINO GPU allocation failure'),
      });
      OpenVinoAsrHelperClient.forceShutdownNow('Retrying Qwen3-ASR helper on CPU after GPU allocation failure.');
      await OpenVinoAsrHelperClient.loadModel(input.modelPath, 'CPU', input.cacheDir, loadTimeoutMs);
      return {
        effectiveDevice: 'CPU',
        fallbackToCpu: true,
      };
    }
  }


  private static async getLocalAsrDevice() {
    const configured = this.getConfiguredLocalAsrDevice();
    if (configured) return configured;
    return 'AUTO';
  }

  private static getConfiguredLocalTranslateDevice() {
    return this.normalizeRequestedDevice(
      process.env.OPENVINO_LOCAL_TRANSLATE_DEVICE,
      ''
    );
  }

  private static async getLocalTranslateDevice() {
    const configured = this.getConfiguredLocalTranslateDevice();
    if (configured) return configured;
    return 'AUTO';
  }

  private static async loadGenaiTranslateModelWithFallback(input: {
    modelPath: string;
    requestedDevice: string;
    runtimeKind: 'llm' | 'vlm';
    properties: Record<string, unknown>;
    timeoutMs: number;
  }) {
    const requestedDevice = this.normalizeRequestedDevice(input.requestedDevice, 'AUTO');
    try {
      const result = await OpenVinoGenaiTranslateHelperClient.loadModel(
        input.modelPath,
        requestedDevice,
        input.runtimeKind,
        input.properties,
        input.timeoutMs
      );
      return {
        result,
        effectiveDevice: requestedDevice,
      };
    } catch (error) {
      if (!this.shouldFallbackTranslateToCpu(requestedDevice, error)) {
        throw error;
      }
      this.log('translate helper gpu allocation failed, retrying on cpu', {
        requestedDevice,
        runtimeKind: input.runtimeKind,
        modelPath: input.modelPath,
        message: String((error as any)?.message || error || 'OpenVINO GPU allocation failure'),
      });
      OpenVinoGenaiTranslateHelperClient.forceShutdownNow('Retrying translate helper on CPU after GPU allocation failure.');
      const result = await OpenVinoGenaiTranslateHelperClient.loadModel(
        input.modelPath,
        'CPU',
        input.runtimeKind,
        input.properties,
        input.timeoutMs
      );
      return {
        result,
        effectiveDevice: 'CPU',
      };
    }
  }

  private static async loadSeq2SeqTranslateModelWithFallback(input: {
    modelPath: string;
    requestedDevice: string;
    timeoutMs: number;
  }) {
    const requestedDevice = this.normalizeRequestedDevice(input.requestedDevice, 'AUTO');
    try {
      const result = await OpenVinoTranslateHelperClient.loadModel(
        input.modelPath,
        requestedDevice,
        input.timeoutMs
      );
      return {
        result,
        effectiveDevice: requestedDevice,
      };
    } catch (error) {
      if (!this.shouldFallbackTranslateToCpu(requestedDevice, error)) {
        throw error;
      }
      this.log('seq2seq translate helper gpu allocation failed, retrying on cpu', {
        requestedDevice,
        modelPath: input.modelPath,
        message: String((error as any)?.message || error || 'OpenVINO GPU allocation failure'),
      });
      OpenVinoTranslateHelperClient.forceShutdownNow('Retrying seq2seq translate helper on CPU after GPU allocation failure.');
      const result = await OpenVinoTranslateHelperClient.loadModel(input.modelPath, 'CPU', input.timeoutMs);
      return {
        result,
        effectiveDevice: 'CPU',
      };
    }
  }

  private static async runGenaiTranslateHelperOnce(input: {
    modelPath: string;
    prompt: string;
    generationConfig: Record<string, unknown>;
    runtimeKind: 'llm' | 'vlm';
    modelId?: string;
  }) {
    const helperPath = PathManager.resolveToolsSourcePath('openvino_genai_translate_helper.mjs');
    if (!(await fs.pathExists(helperPath))) {
      throw new Error(`OpenVINO GenAI translate helper not found: ${helperPath}`);
    }

    const requestedDevice = await this.getLocalTranslateDevice();
    const isTranslateGemma =
      /translategemma/i.test(String(input.modelId || '')) || /translategemma/i.test(String(input.modelPath || ''));
    // Dedicated translation models are run as isolated helper jobs without the
    // shared CACHE_DIR / scheduler hints used by generic local LLM routes.
    const properties = isTranslateGemma ? {} : this.getLocalTranslatePipelineProperties(input.modelId);
    const child = spawn(process.execPath, [helperPath], {
      cwd: PathManager.getRoot(),
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: {
        ...process.env,
        NODE_NO_WARNINGS: process.env.NODE_NO_WARNINGS || '1',
      },
    });

    let readBuffer = '';
    const pending = new Map<
      string,
      {
        resolve: (value: any) => void;
        reject: (error: Error) => void;
        timeout: NodeJS.Timeout;
      }
    >();
    let stderr = '';
    let requestSeq = 0;

    const rejectAll = (message: string) => {
      for (const [, item] of pending) {
        clearTimeout(item.timeout);
        item.reject(new Error(message));
      }
      pending.clear();
    };

    child.stdout.on('data', (chunk) => {
      readBuffer += chunk.toString('utf8');
      let lineEnd = readBuffer.indexOf('\n');
      while (lineEnd >= 0) {
        const rawLine = readBuffer.slice(0, lineEnd).trim();
        readBuffer = readBuffer.slice(lineEnd + 1);
        lineEnd = readBuffer.indexOf('\n');
        if (!rawLine) continue;
        let payload: HelperResponsePayload | null = null;
        try {
          payload = JSON.parse(rawLine) as HelperResponsePayload;
        } catch {
          payload = null;
        }
        if (!payload?.requestId) continue;
        const item = pending.get(payload.requestId);
        if (!item) continue;
        clearTimeout(item.timeout);
        pending.delete(payload.requestId);
        if (payload.ok) {
          item.resolve(payload.result);
        } else {
          item.reject(new Error(String(payload.error || 'OpenVINO GenAI translate helper request failed.')));
        }
      }
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString('utf8');
    });
    child.on('error', (error) => {
      rejectAll(`OpenVINO GenAI translate helper failed: ${error.message}`);
    });
    child.on('exit', (code, signal) => {
      rejectAll(`OpenVINO GenAI translate helper exited unexpectedly (code=${String(code)}, signal=${String(signal)}).`);
    });

    const request = (method: string, params: Record<string, unknown>, timeoutMs: number) =>
      new Promise<any>((resolve, reject) => {
        if (!child.stdin || child.stdin.destroyed) {
          reject(new Error('OpenVINO GenAI translate helper stdin is unavailable.'));
          return;
        }
        const requestId = `fresh-genai-${Date.now()}-${++requestSeq}`;
        const timeout = setTimeout(() => {
          pending.delete(requestId);
          reject(new Error(`OpenVINO GenAI translate helper request timed out after ${timeoutMs} ms (${method}).`));
        }, timeoutMs);
        pending.set(requestId, { resolve, reject, timeout });
        child.stdin.write(`${JSON.stringify({ requestId, method, params })}\n`, 'utf8', (error) => {
          if (!error) return;
          clearTimeout(timeout);
          pending.delete(requestId);
          reject(new Error(`Failed to write to OpenVINO GenAI translate helper stdin: ${String(error.message || error)}`));
        });
      });

    try {
      const loadResult = await request(
        'load',
        {
          modelPath: input.modelPath,
          device: requestedDevice,
          runtimeKind: input.runtimeKind,
          properties,
        },
        this.getTranslateLoadTimeoutMs(input.runtimeKind)
      );
      const generateResult = await request(
        'generate',
        {
          prompt: input.prompt,
          generationConfig: input.generationConfig,
        },
        this.getEnvNumber('OPENVINO_TRANSLATE_HELPER_GENERATE_TIMEOUT_MS', 300000, 1000, 1800000)
      );
      await request('shutdown', {}, 5000).catch(() => {});
      return {
        loadResult,
        generateResult,
        stderr,
      };
    } finally {
      if (!child.killed) {
        try {
          child.kill();
        } catch {
          // ignore cleanup failures
        }
      }
    }
  }

  private static getLocalTranslateCacheDir(modelId?: string) {
    const configured = String(
      process.env.OPENVINO_LOCAL_TRANSLATE_CACHE_DIR || ''
    ).trim();
    const cacheRoot = configured || path.join(PathManager.getModelsPath(), 'openvino-cache', 'translate');
    const safeModelId = String(modelId || 'default')
      .trim()
      .replace(/[^a-z0-9._-]+/gi, '_');
    const resolved = path.join(cacheRoot, safeModelId);
    fs.ensureDirSync(resolved);
    return resolved;
  }

  private static getLocalTranslatePipelineProperties(modelId?: string) {
    const properties: Record<string, unknown> = {
      CACHE_DIR: this.getLocalTranslateCacheDir(modelId),
    };

    const performanceHint = String(
      process.env.OPENVINO_LOCAL_TRANSLATE_PERFORMANCE_HINT || ''
    )
      .trim()
      .toUpperCase();
    if (performanceHint) {
      properties.PERFORMANCE_HINT = performanceHint;
    }

    const numStreams = Math.round(
      this.getEnvNumber('OPENVINO_LOCAL_TRANSLATE_NUM_STREAMS', 0, 0, 64)
    );
    if (numStreams > 0) {
      properties.NUM_STREAMS = String(numStreams);
    }

    if (this.getEnvBoolean('OPENVINO_LOCAL_TRANSLATE_PROMPT_LOOKUP', false)) {
      properties.prompt_lookup = true;
    }

    const schedulerConfig: Record<string, unknown> = {};
    const maxNumBatchedTokens = Math.round(
      this.getEnvNumber(
        'OPENVINO_LOCAL_TRANSLATE_MAX_BATCHED_TOKENS',
        0,
        0,
        65536
      )
    );
    if (maxNumBatchedTokens > 0) {
      schedulerConfig.max_num_batched_tokens = maxNumBatchedTokens;
    }

    const cacheSize = this.getEnvNumber(
      'OPENVINO_LOCAL_TRANSLATE_SCHEDULER_CACHE_SIZE_GB',
      0,
      0,
      256
    );
    if (cacheSize > 0) {
      schedulerConfig.cache_size = Number(cacheSize.toFixed(3));
    }

    const dynamicSplitFuseRaw = process.env.OPENVINO_LOCAL_TRANSLATE_DYNAMIC_SPLIT_FUSE;
    if (typeof dynamicSplitFuseRaw === 'string' && dynamicSplitFuseRaw.trim()) {
      schedulerConfig.dynamic_split_fuse = this.getEnvBoolean('OPENVINO_LOCAL_TRANSLATE_DYNAMIC_SPLIT_FUSE', true);
    }

    if (Object.keys(schedulerConfig).length > 0) {
      properties.schedulerConfig = schedulerConfig;
    }

    return properties;
  }

  private static isTranslateGemmaModelRef(modelId?: string, modelPath?: string) {
    return (
      /translategemma/i.test(String(modelId || '')) ||
      /translategemma/i.test(String(modelPath || ''))
    );
  }

  private static getLocalAsrCacheDir(modelId?: string) {
    const configured = String(
      process.env.OPENVINO_LOCAL_ASR_CACHE_DIR || ''
    ).trim();
    const cacheRoot = configured || path.join(PathManager.getModelsPath(), 'openvino-cache', 'asr');
    const safeModelId = String(modelId || 'default')
      .trim()
      .replace(/[^a-z0-9._-]+/gi, '_');
    const resolved = path.join(cacheRoot, safeModelId);
    fs.ensureDirSync(resolved);
    return resolved;
  }

  private static getAsrPipelineProperties(modelId?: string, wordTimestampsEnabled = true, useCache = true) {
    const properties: Record<string, unknown> = {
      word_timestamps: wordTimestampsEnabled,
    };
    if (useCache) {
      properties.CACHE_DIR = this.getLocalAsrCacheDir(modelId);
    }
    return properties;
  }

  private static isOpenvinoSerializationCacheError(error: unknown) {
    const message = String((error as any)?.message || error || '');
    return (
      /Unsupported attribute type for serialization/i.test(message) ||
      /xml_serialize_util/i.test(message) ||
      (/serialization/i.test(message) && /inputs/i.test(message))
    );
  }

  private static getAudioDecodeCacheMaxItems() {
    return Math.max(0, Math.round(this.getEnvNumber('OPENVINO_ASR_AUDIO_CACHE_MAX_ITEMS', 12, 0, 128)));
  }

  private static buildAudioDecodeCacheKey(filePath: string, stats: { size: number; mtimeMs: number }) {
    return `${path.resolve(filePath).toLowerCase()}|${Math.round(stats.size)}|${Math.round(stats.mtimeMs)}`;
  }

  private static trimAudioDecodeCache() {
    const maxItems = this.getAudioDecodeCacheMaxItems();
    if (maxItems <= 0) {
      this.audioDecodeCache.clear();
      return;
    }
    while (this.audioDecodeCache.size > maxItems) {
      const firstKey = this.audioDecodeCache.keys().next().value;
      if (!firstKey) break;
      this.audioDecodeCache.delete(firstKey);
    }
  }

  private static getPythonCommand() {
    const configured = String(process.env.OPENVINO_HELPER_PYTHON || '').trim();
    return configured || 'python';
  }

  private static getOfficialQwenConvertScriptPath() {
    return PathManager.resolveToolsSourcePath('convert_official_qwen3_asr.py');
  }

  private static getGenericHfConvertScriptPath() {
    return PathManager.resolveToolsSourcePath('convert_hf_model_to_openvino.py');
  }

  private static getChatTemplateRenderScriptPath() {
    return PathManager.resolveToolsSourcePath('render_hf_chat_template.py');
  }

  private static getInputTokenCountScriptPath() {
    return PathManager.resolveToolsSourcePath('count_hf_input_tokens.py');
  }

  private static async withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(message));
      }, timeoutMs);

      promise.then(
        (value) => {
          clearTimeout(timer);
          resolve(value);
        },
        (error) => {
          clearTimeout(timer);
          reject(error);
        }
      );
    });
  }

  private static normalizeTranslatedText(raw: unknown) {
    const text = String(raw ?? '').trim();
    if (!text) return '';

    const fenced = text.match(/^```(?:text|markdown)?\s*([\s\S]*?)\s*```$/i);
    if (fenced) return String(fenced[1] || '').trim();
    return text;
  }

  private static normalizeLocalAsrText(raw: unknown) {
    if (Array.isArray(raw)) {
      return raw
        .map((item) => this.normalizeLocalAsrText(item))
        .filter(Boolean)
        .join(' ')
        .trim();
    }

    if (raw && typeof raw === 'object') {
      const candidateObject = raw as Record<string, unknown>;
      const directCandidates = [
        candidateObject.text,
        candidateObject.transcript,
        candidateObject.transcription,
        candidateObject.output_text,
        candidateObject.outputText,
        candidateObject.translatedText,
        candidateObject.texts,
        candidateObject.chunks,
        candidateObject.segments,
        candidateObject.results,
      ];
      for (const candidate of directCandidates) {
        const normalized = this.normalizeLocalAsrText(candidate);
        if (normalized) return normalized;
      }
    }

    const text = String(raw ?? '').trim();
    return text;
  }

  private static isAutoLanguageRequest(language?: string) {
    const normalized = String(language || '').trim().toLowerCase();
    return !normalized || normalized === 'auto' || normalized === 'none' || normalized === 'null';
  }

  private static getWhisperAutoFallbackLanguages() {
    const raw = String(
      process.env.OPENVINO_WHISPER_AUTO_FALLBACK_LANGUAGES || 'ja,zh,en,ko,es,fr,de,it,pt,fi'
    ).trim();
    const seen = new Set<string>();
    const ordered: string[] = [];
    for (const item of raw.split(/[,\s]+/)) {
      const normalized = String(item || '')
        .trim()
        .toLowerCase()
        .replace(/_/g, '-');
      if (!normalized || seen.has(normalized) || this.isAutoLanguageRequest(normalized)) continue;
      seen.add(normalized);
      ordered.push(normalized);
    }
    return ordered;
  }

  private static async isVlmModelPath(modelPath: string) {
    const hasLanguageModel = await fs.pathExists(path.join(modelPath, 'openvino_language_model.xml'));
    const hasTextEmbeddingsModel = await fs.pathExists(path.join(modelPath, 'openvino_text_embeddings_model.xml'));
    return hasLanguageModel && hasTextEmbeddingsModel;
  }

  private static async isSeq2SeqTranslateModelPath(modelPath: string) {
    const checks = await Promise.all([
      fs.pathExists(path.join(modelPath, 'openvino_encoder_model.xml')),
      fs.pathExists(path.join(modelPath, 'openvino_encoder_model.bin')),
      fs.pathExists(path.join(modelPath, 'openvino_decoder_model.xml')),
      fs.pathExists(path.join(modelPath, 'openvino_decoder_model.bin')),
      fs.pathExists(path.join(modelPath, 'tokenizer_config.json')),
    ]);
    return checks.every(Boolean);
  }

  private static async safeSystemSnapshot(forceFresh = false): Promise<SystemResourceSnapshot | null> {
    try {
      return await this.withTimeout(
        SystemMonitor.getSnapshot(forceFresh),
        2000,
        'System resource snapshot timed out.'
      );
    } catch {
      return null;
    }
  }

  private static toGpuAccelerators(snapshot: SystemResourceSnapshot | null) {
    return Array.isArray(snapshot?.accelerators)
      ? snapshot!.accelerators.filter((item): item is AcceleratorSnapshot => item.kind === 'gpu')
      : [];
  }

  private static acceleratorKey(accelerator: AcceleratorSnapshot) {
    return `${String(accelerator.model || '').trim().toLowerCase()}|${String(accelerator.luid || accelerator.id || '').trim().toLowerCase()}`;
  }

  private static toAcceleratorInference(
    accelerator: AcceleratorSnapshot,
    source: OpenvinoAcceleratorInference['source']
  ): OpenvinoAcceleratorInference {
    return {
      observedAt: new Date().toISOString(),
      source,
      acceleratorModel: accelerator.model,
      luid: accelerator.luid,
      memorySource: accelerator.memorySource,
      vramUsedGB: accelerator.vramUsedGB,
      vramTotalGB: accelerator.vramTotalGB,
      utilization: accelerator.utilization,
      physIndex: accelerator.physIndex,
    };
  }

  private static inferAcceleratorByDelta(
    before: SystemResourceSnapshot | null,
    after: SystemResourceSnapshot | null
  ): OpenvinoAcceleratorInference | null {
    const afterGpus = this.toGpuAccelerators(after);
    if (afterGpus.length === 0) return null;
    const beforeMap = new Map(this.toGpuAccelerators(before).map((item) => [this.acceleratorKey(item), item]));

    let best: { accelerator: AcceleratorSnapshot; score: number } | null = null;
    for (const accelerator of afterGpus) {
      const prev = beforeMap.get(this.acceleratorKey(accelerator));
      const deltaVram = Math.max(0, (accelerator.vramUsedGB || 0) - (prev?.vramUsedGB || 0));
      const deltaUtil = Math.max(0, accelerator.utilization - (prev?.utilization || 0));
      const score =
        deltaVram * 20 +
        deltaUtil * 0.08 +
        ((accelerator.memorySource === 'dedicated' ? 1 : 0) * 2) +
        ((accelerator.engineTypes || []).includes('compute') ? 2 : 0);

      if (!best || score > best.score) {
        best = { accelerator, score };
      }
    }

    if (!best || best.score < 0.35) return null;
    return this.toAcceleratorInference(best.accelerator, 'model-load-delta');
  }

  private static pickLikelyActiveAccelerator(
    snapshot: SystemResourceSnapshot | null,
    source: OpenvinoAcceleratorInference['source']
  ): OpenvinoAcceleratorInference | null {
    const gpus = this.toGpuAccelerators(snapshot);
    if (gpus.length === 0) return null;

    let best: { accelerator: AcceleratorSnapshot; score: number } | null = null;
    for (const accelerator of gpus) {
      const score =
        (accelerator.utilization || 0) * 0.12 +
        (accelerator.vramUsedGB || 0) * 3 +
        ((accelerator.memorySource === 'dedicated' ? 1 : 0) * 1.5) +
        ((accelerator.engineTypes || []).includes('compute') ? 2 : 0);
      if (!best || score > best.score) {
        best = { accelerator, score };
      }
    }

    if (!best || best.score <= 0.2) return null;
    return this.toAcceleratorInference(best.accelerator, source);
  }

  static getTranslateRuntimeDebug(modelId: string): OpenvinoTranslateRuntimeDebug | null {
    const cached = this.translatePipelineCache.get(modelId);
    return cached ? { ...cached.runtimeDebug } : null;
  }

  private static summarizePerfMetrics(perfMetrics: any) {
    if (!perfMetrics || typeof perfMetrics !== 'object') return null;

    const readScalar = (getter: (() => unknown) | undefined) => {
      if (typeof getter !== 'function') return null;
      try {
        const value = getter.call(perfMetrics);
        return typeof value === 'number' && Number.isFinite(value) ? value : null;
      } catch {
        return null;
      }
    };

    const readMean = (getter: (() => { mean?: unknown }) | undefined) => {
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

  private static async getOrCreateTranslatePipeline(modelId: string, modelPath: string): Promise<CachedTranslatePipeline> {
    this.traceLocalTranslate('getOrCreateTranslatePipeline:start', { modelId, modelPath });
    const cached = this.translatePipelineCache.get(modelId);
    if (cached) {
      this.traceLocalTranslate('getOrCreateTranslatePipeline:cache-hit', { modelId });
      this.activeTranslateModelId = modelId;
      return cached;
    }

    // Keep translation runtime behavior consistent with ASR:
    // when switching model ids, unload previous local translation pipelines first.
    if (this.translatePipelineCache.size > 0 && this.activeTranslateModelId !== modelId) {
      this.log('switching translate model, releasing previous pipeline cache', {
        fromModelId: this.activeTranslateModelId,
        toModelId: modelId,
        pipelineCount: this.translatePipelineCache.size,
      });
      const previousPipelines = Array.from(this.translatePipelineCache.values()).map((item) => item.pipeline);
      this.translatePipelineCache.clear();
      const settled = await Promise.allSettled(
        previousPipelines.map((pipeline) => this.disposeTranslatePipeline(pipeline))
      );
      const failed = settled.filter((item) => item.status === 'rejected').length;
      if (failed > 0) {
        throw new Error(`Failed to release ${failed} previous local translation pipeline(s) before model switch.`);
      }
      this.activeTranslateModelId = null;
    }

    const isVlm = await this.isVlmModelPath(modelPath);
    const isSeq2Seq = !isVlm && await this.isSeq2SeqTranslateModelPath(modelPath);
    const isTranslateGemma = this.isTranslateGemmaModelRef(modelId, modelPath);
    const requestedDevice = await this.getLocalTranslateDevice();
    const properties = isTranslateGemma ? {} : this.getLocalTranslatePipelineProperties(modelId);
    const cacheDir = typeof properties.CACHE_DIR === 'string' ? properties.CACHE_DIR : null;
    const promptLookupEnabled = Boolean(properties.prompt_lookup);
    const schedulerConfig =
      properties.schedulerConfig && typeof properties.schedulerConfig === 'object'
        ? { ...(properties.schedulerConfig as Record<string, unknown>) }
        : null;
    const beforeSnapshotStartedAt = Date.now();
    const beforeSnapshot = await this.safeSystemSnapshot(true);
    this.traceLocalTranslate('getOrCreateTranslatePipeline:before-snapshot', {
      modelId,
      elapsedMs: Date.now() - beforeSnapshotStartedAt,
      ok: Boolean(beforeSnapshot),
    });
    if (isVlm) {
      const loadStartedAt = Date.now();
      const loaded = await this.loadGenaiTranslateModelWithFallback({
        modelPath,
        requestedDevice,
        runtimeKind: 'vlm',
        properties,
        timeoutMs: this.getTranslateLoadTimeoutMs('vlm'),
      });
      this.traceLocalTranslate('getOrCreateTranslatePipeline:vlm-load', {
        modelId,
        elapsedMs: Date.now() - loadStartedAt,
        effectiveDevice: loaded.effectiveDevice,
      });
      const loadResult = loaded.result;
      const pipeline = {
        device: String(loadResult?.device || loaded.effectiveDevice),
        async generate(input: { prompt?: string; messages?: Array<Record<string, unknown>> }, options?: Record<string, unknown>) {
          const generationConfig =
            options && typeof options.generationConfig === 'object'
              ? (options.generationConfig as Record<string, unknown>)
              : {};
          return OpenVinoGenaiTranslateHelperClient.generate(input, generationConfig);
        },
        async dispose() {
          await OpenVinoGenaiTranslateHelperClient.shutdown();
        },
      };
      const afterSnapshotStartedAt = Date.now();
      const afterSnapshot = await this.safeSystemSnapshot(true);
      this.traceLocalTranslate('getOrCreateTranslatePipeline:vlm-after-snapshot', {
        modelId,
        elapsedMs: Date.now() - afterSnapshotStartedAt,
        ok: Boolean(afterSnapshot),
      });
      const wrapped: CachedTranslatePipeline = {
        kind: 'vlm',
        pipeline,
        runtimeDebug: {
          modelId,
          modelPath,
          pipelineKind: 'vlm',
          requestedDevice,
          pipelineDevice: typeof pipeline?.device === 'string' ? pipeline.device : null,
          cacheDir,
          promptLookupEnabled,
          schedulerConfig,
          loadInference:
            this.inferAcceleratorByDelta(beforeSnapshot, afterSnapshot) ||
            this.pickLikelyActiveAccelerator(afterSnapshot, 'model-load-snapshot'),
          lastInference: null,
          lastPerfMetrics: null,
        },
      };
      this.translatePipelineCache.set(modelId, wrapped);
      this.activeTranslateModelId = modelId;
      return wrapped;
    }

    if (isSeq2Seq) {
      const loadStartedAt = Date.now();
      const loaded = await this.loadSeq2SeqTranslateModelWithFallback({
        modelPath,
        requestedDevice,
        timeoutMs: this.getTranslateLoadTimeoutMs('seq2seq'),
      });
      this.traceLocalTranslate('getOrCreateTranslatePipeline:seq2seq-load', {
        modelId,
        elapsedMs: Date.now() - loadStartedAt,
        effectiveDevice: loaded.effectiveDevice,
      });
      const loadResult = loaded.result;
      const pipeline = {
        device: String(loadResult?.device || loaded.effectiveDevice),
      async generate(input: { prompt?: string }, generationConfig?: Record<string, unknown>) {
        return OpenVinoTranslateHelperClient.generate(String(input.prompt || ''), generationConfig || {});
      },
        async dispose() {
          await OpenVinoTranslateHelperClient.shutdown();
        },
      };
      const afterSnapshotStartedAt = Date.now();
      const afterSnapshot = await this.safeSystemSnapshot(true);
      this.traceLocalTranslate('getOrCreateTranslatePipeline:seq2seq-after-snapshot', {
        modelId,
        elapsedMs: Date.now() - afterSnapshotStartedAt,
        ok: Boolean(afterSnapshot),
      });
      const wrapped: CachedTranslatePipeline = {
        kind: 'seq2seq',
        pipeline,
        runtimeDebug: {
          modelId,
          modelPath,
          pipelineKind: 'seq2seq',
          requestedDevice,
          pipelineDevice: pipeline.device,
          cacheDir,
          promptLookupEnabled: false,
          schedulerConfig: null,
          loadInference:
            this.inferAcceleratorByDelta(beforeSnapshot, afterSnapshot) ||
            this.pickLikelyActiveAccelerator(afterSnapshot, 'model-load-snapshot'),
          lastInference: null,
          lastPerfMetrics: null,
        },
      };
      this.translatePipelineCache.set(modelId, wrapped);
      this.activeTranslateModelId = modelId;
      return wrapped;
    }

    const loadStartedAt = Date.now();
    const loaded = await this.loadGenaiTranslateModelWithFallback({
      modelPath,
      requestedDevice,
      runtimeKind: 'llm',
      properties,
      timeoutMs: this.getTranslateLoadTimeoutMs('llm'),
    });
    this.traceLocalTranslate('getOrCreateTranslatePipeline:llm-load', {
      modelId,
      elapsedMs: Date.now() - loadStartedAt,
      effectiveDevice: loaded.effectiveDevice,
    });
    const loadResult = loaded.result;
    const pipeline = {
      device: String(loadResult?.device || loaded.effectiveDevice),
      async generate(input: { prompt?: string; messages?: Array<Record<string, unknown>> }, generationConfig?: Record<string, unknown>) {
        return OpenVinoGenaiTranslateHelperClient.generate(input, generationConfig || {});
      },
      async dispose() {
        await OpenVinoGenaiTranslateHelperClient.shutdown();
      },
    };
    const afterSnapshotStartedAt = Date.now();
    const afterSnapshot = await this.safeSystemSnapshot(true);
    this.traceLocalTranslate('getOrCreateTranslatePipeline:llm-after-snapshot', {
      modelId,
      elapsedMs: Date.now() - afterSnapshotStartedAt,
      ok: Boolean(afterSnapshot),
    });
    const wrapped: CachedTranslatePipeline = {
      kind: 'llm',
      pipeline,
      runtimeDebug: {
        modelId,
        modelPath,
        pipelineKind: 'llm',
        requestedDevice,
        pipelineDevice: typeof pipeline?.device === 'string' ? pipeline.device : null,
        cacheDir,
        promptLookupEnabled,
        schedulerConfig,
        loadInference:
          this.inferAcceleratorByDelta(beforeSnapshot, afterSnapshot) ||
          this.pickLikelyActiveAccelerator(afterSnapshot, 'model-load-snapshot'),
        lastInference: null,
        lastPerfMetrics: null,
      },
    };
    this.translatePipelineCache.set(modelId, wrapped);
    this.activeTranslateModelId = modelId;
    return wrapped;
  }

  private static async getLlmTranslatePipelineFactory(): Promise<OpenvinoPipelineFactory> {
    if (this.llmTranslatePipelineFactory) return this.llmTranslatePipelineFactory;

    const openvinoGenai = await import('openvino-genai-node');
    const factoryCandidate =
      (openvinoGenai as any)?.LLMPipeline ||
      (openvinoGenai as any)?.default?.LLMPipeline ||
      (openvinoGenai as any)?.default;

    if (typeof factoryCandidate !== 'function') {
      throw new Error('openvino-genai-node LLMPipeline is unavailable.');
    }

    this.llmTranslatePipelineFactory = factoryCandidate as OpenvinoPipelineFactory;
    return this.llmTranslatePipelineFactory;
  }

  private static async getVlmTranslatePipelineFactory(): Promise<OpenvinoPipelineFactory> {
    if (this.vlmTranslatePipelineFactory) return this.vlmTranslatePipelineFactory;

    const openvinoGenai = await import('openvino-genai-node');
    const factoryCandidate =
      (openvinoGenai as any)?.VLMPipeline ||
      (openvinoGenai as any)?.default?.VLMPipeline;

    if (typeof factoryCandidate !== 'function') {
      throw new Error('openvino-genai-node VLMPipeline is unavailable for this model.');
    }

    this.vlmTranslatePipelineFactory = factoryCandidate as OpenvinoPipelineFactory;
    return this.vlmTranslatePipelineFactory;
  }

  private static async getWhisperAsrPipelineFactory(): Promise<OpenvinoPipelineFactory> {
    if (this.whisperAsrPipelineFactory) return this.whisperAsrPipelineFactory;

    const openvinoGenai = await import('openvino-genai-node');
    const factoryCandidate =
      (openvinoGenai as any)?.WhisperPipeline ||
      (openvinoGenai as any)?.default?.WhisperPipeline;

    if (typeof factoryCandidate !== 'function') {
      throw new Error('openvino-genai-node WhisperPipeline is unavailable.');
    }

    this.whisperAsrPipelineFactory = factoryCandidate as OpenvinoPipelineFactory;
    return this.whisperAsrPipelineFactory;
  }

  private static normalizeWhisperLanguage(language?: string) {
    return buildWhisperLanguageToken(language);
  }

  private static getCompiledModelPortNames(compiledModel: any) {
    const inputPorts = Array.isArray(compiledModel?.inputs) ? compiledModel.inputs : [];
    const outputPorts = Array.isArray(compiledModel?.outputs) ? compiledModel.outputs : [];
    return {
      inputNames: inputPorts.map((port: any, index: number) => OpenvinoBackend.getPortName(port, `input_${index}`)),
      outputNames: outputPorts.map((port: any, index: number) => OpenvinoBackend.getPortName(port, `output_${index}`)),
    };
  }

  private static async loadJsonFile<T>(filePath: string): Promise<T> {
    return fs.readJson(filePath) as Promise<T>;
  }

  private static normalizeQwenAsrLanguage(language: string | undefined, runtime: Qwen3AsrRuntime) {
    const raw = String(language || '').trim();
    if (!raw) return undefined;

    const lower = raw.toLowerCase().replace(/_/g, '-');
    if (lower === 'auto' || lower === 'none' || lower === 'null') {
      return undefined;
    }

    const canonical =
      runtime.supportedLanguages.get(lower) ||
      runtime.supportedLanguages.get(raw) ||
      resolveQwenAsrLanguageName(lower);

    if (!canonical) {
      const supported = Array.from(runtime.supportedLanguages.values()).join(', ');
      throw new Error(`Qwen3-ASR does not support language "${raw}". Supported languages: ${supported}`);
    }

    return canonical;
  }

  private static getQwenAsrMaxNewTokens() {
    return Math.round(this.getEnvNumber('OPENVINO_QWEN3_ASR_MAX_NEW_TOKENS', 256, 16, 4096));
  }

  private static getQwenAsrConversionCompressionMode() {
    const normalized = String(process.env.OPENVINO_QWEN3_ASR_CONVERSION_COMPRESSION_MODE || 'int8')
      .trim()
      .toLowerCase();
    if (!normalized) return null;
    if (['none', 'off', 'false', '0'].includes(normalized)) return 'none';
    if (['int4', 'int4_asym', 'int4-sym', 'int4_sym'].includes(normalized)) return 'int4';
    if (['int8', 'int8_asym', 'int8-sym', 'int8_sym'].includes(normalized)) return 'int8';
    return null;
  }

  private static getQwenByteDecoderMap() {
    if (this.qwenByteDecoderMap) return this.qwenByteDecoderMap;

    const bytes: number[] = [];
    for (let i = 33; i <= 126; i += 1) bytes.push(i);
    for (let i = 161; i <= 172; i += 1) bytes.push(i);
    for (let i = 174; i <= 255; i += 1) bytes.push(i);

    const chars = [...bytes];
    let extra = 0;
    for (let i = 0; i < 256; i += 1) {
      if (bytes.includes(i)) continue;
      bytes.push(i);
      chars.push(256 + extra);
      extra += 1;
    }

    this.qwenByteDecoderMap = new Map(chars.map((codePoint, index) => [String.fromCharCode(codePoint), bytes[index]]));
    return this.qwenByteDecoderMap;
  }

  private static decodeQwenTokenString(token: string) {
    if (!token) return '';
    const byteDecoder = this.getQwenByteDecoderMap();
    const bytes: number[] = [];
    for (const symbol of token) {
      const value = byteDecoder.get(symbol);
      if (typeof value === 'number') {
        bytes.push(value);
      }
    }
    if (bytes.length === 0) return '';
    return this.utf8Decoder.decode(Uint8Array.from(bytes));
  }

  private static decodeQwenTokenIds(runtime: Qwen3AsrRuntime, tokenIds: number[], { keepSpecial = false } = {}) {
    const parts: string[] = [];
    for (const tokenId of tokenIds) {
      const tokenContent = runtime.tokenIdToContent.get(tokenId);
      if (!tokenContent) continue;
      const isSpecial = runtime.specialTokenIds.has(tokenId);
      if (isSpecial) {
        if (keepSpecial) parts.push(tokenContent);
        continue;
      }
      parts.push(this.decodeQwenTokenString(tokenContent));
    }
    return parts.join('');
  }

  private static parseQwenAsrTranscript(runtime: Qwen3AsrRuntime, generatedTokenIds: number[]) {
    const asrTextId = Number(runtime.promptTemplate.asr_text_id ?? Number.NaN);
    const markerIndex =
      Number.isFinite(asrTextId) && asrTextId >= 0 ? generatedTokenIds.indexOf(asrTextId) : -1;

    const prefixIds = markerIndex >= 0 ? generatedTokenIds.slice(0, markerIndex) : [];
    const transcriptIds = markerIndex >= 0 ? generatedTokenIds.slice(markerIndex + 1) : generatedTokenIds;

    const prefixText = this.decodeQwenTokenIds(runtime, prefixIds, { keepSpecial: true }).trim();
    const transcript = this.decodeQwenTokenIds(runtime, transcriptIds).trim();
    const languageMatch = prefixText.match(/language\s+([A-Za-z][A-Za-z\s-]+)/i);

    return {
      text: transcript,
      language: languageMatch ? languageMatch[1].trim() : undefined,
      rawPrefix: prefixText,
    };
  }

  private static getTensorByIndex(results: Record<string, any>, model: Qwen3AsrCompiledModel, index: number) {
    const outputName = model.outputNames[index];
    if (outputName && results[outputName]) return results[outputName];
    return Object.values(results)[index] ?? null;
  }

  private static inferQwenCompiledModel(model: Qwen3AsrCompiledModel, inputTensors: any[]) {
    const inferRequest = model.compiledModel.createInferRequest();
    for (let index = 0; index < inputTensors.length; index += 1) {
      inferRequest.setInputTensor(index, inputTensors[index]);
    }
    inferRequest.infer();
    return model.outputNames.map((_, index) => inferRequest.getOutputTensor(index));
  }

  private static getTensorShape(tensor: any) {
    if (!tensor) return [];
    if (typeof tensor.getShape === 'function') {
      return tensor.getShape().map((value: any) => Number(value));
    }
    return Array.isArray(tensor?.shape) ? tensor.shape.map((value: any) => Number(value)) : [];
  }

  private static async compileQwenAsrModel(modelPath: string, fileName: string): Promise<Qwen3AsrCompiledModel> {
    const compiledModel = await OpenvinoBackend.compileModel(path.join(modelPath, fileName), 'AUTO');
    const { inputNames, outputNames } = this.getCompiledModelPortNames(compiledModel);
    return {
      compiledModel,
      inputNames,
      outputNames,
    };
  }

  private static async getOrCreateQwenAsrRuntime(modelId: string, modelPath: string): Promise<Qwen3AsrRuntime> {
    const cached = this.qwenAsrRuntimeCache;
    if (cached && cached.modelId === modelId && cached.modelPath === modelPath) {
      return cached;
    }

    if (this.asrPipelineCache) {
      await this.disposeAsrPipeline().catch(() => {});
    }

    const generationConfigPath = path.join(modelPath, 'generation_config.json');
    const hasGenerationConfig = await fs.pathExists(generationConfigPath);
    const [promptTemplate, preprocessorConfigRaw, tokenizerConfig, generationConfig, vocab, audioEncoder, thinkerEmbeddings, decoderPrefill, decoderKv] =
      await Promise.all([
        this.loadJsonFile<Qwen3AsrPromptTemplate>(path.join(modelPath, 'prompt_template.json')),
        this.loadJsonFile<Qwen3AsrPreprocessorConfig>(path.join(modelPath, 'preprocessor_config.json')),
        this.loadJsonFile<Qwen3AsrTokenizerConfig>(path.join(modelPath, 'tokenizer_config.json')),
        hasGenerationConfig
          ? this.loadJsonFile<Qwen3AsrGenerationConfig>(generationConfigPath)
          : Promise.resolve(null),
        this.loadJsonFile<Record<string, number>>(path.join(modelPath, 'vocab.json')),
        this.compileQwenAsrModel(modelPath, 'audio_encoder_model.xml'),
        this.compileQwenAsrModel(modelPath, 'thinker_embeddings_model.xml'),
        this.compileQwenAsrModel(modelPath, 'decoder_prefill_kv_model.xml'),
        this.compileQwenAsrModel(modelPath, 'decoder_kv_model.xml'),
      ]);

    const preprocessorConfig: Qwen3AsrPreprocessorConfig = {
      sampling_rate: 16000,
      ...preprocessorConfigRaw,
    };
    const featureExtractor = new WhisperFeatureExtractor(preprocessorConfig as any);

    const tokenIdToContent = new Map<number, string>();
    for (const [token, tokenId] of Object.entries(vocab)) {
      tokenIdToContent.set(Number(tokenId), token);
    }

    const specialTokenIds = new Set<number>(promptTemplate.special_ids || []);
    for (const [tokenId, item] of Object.entries(tokenizerConfig?.added_tokens_decoder || {})) {
      const numericId = Number(tokenId);
      if (Number.isFinite(numericId) && typeof item?.content === 'string') {
        tokenIdToContent.set(numericId, item.content);
        if (item.special) {
          specialTokenIds.add(numericId);
        }
      }
    }

    if (Number.isFinite(Number(promptTemplate.asr_text_id))) {
      tokenIdToContent.set(Number(promptTemplate.asr_text_id), '<asr_text>');
    }

    const eosTokenIds = new Set<number>();
    const configuredEos = generationConfig?.eos_token_id;
    if (Array.isArray(configuredEos)) {
      for (const tokenId of configuredEos) {
        eosTokenIds.add(Number(tokenId));
      }
    } else if (Number.isFinite(Number(configuredEos))) {
      eosTokenIds.add(Number(configuredEos));
    }
    if (Number.isFinite(Number(promptTemplate.eos_id))) {
      eosTokenIds.add(Number(promptTemplate.eos_id));
    }
    if (Number.isFinite(Number(promptTemplate.eot_id))) {
      eosTokenIds.add(Number(promptTemplate.eot_id));
    }

    const supportedLanguages = new Map<string, string>();
    for (const item of promptTemplate.supported_languages || []) {
      const normalized = String(item || '').trim();
      if (!normalized) continue;
      supportedLanguages.set(normalized, normalized);
      supportedLanguages.set(normalized.toLowerCase(), normalized);
    }

    const vocabSize = Array.from(tokenIdToContent.keys()).reduce((maxTokenId, tokenId) => {
      return Math.max(maxTokenId, Number(tokenId));
    }, 151935) + 1;

    const runtime: Qwen3AsrRuntime = {
      modelId,
      modelPath,
      featureExtractor,
      audioEncoder,
      thinkerEmbeddings,
      decoderPrefill,
      decoderKv,
      promptTemplate,
      preprocessorConfig,
      tokenizerConfig,
      generationConfig,
      tokenIdToContent,
      vocabSize,
      specialTokenIds,
      eosTokenIds,
      supportedLanguages,
    };
    this.qwenAsrRuntimeCache = runtime;
    return runtime;
  }

  private static async embedQwenTokenIds(runtime: Qwen3AsrRuntime, tokenIds: number[]) {
    if (tokenIds.length === 0) {
      return new Float32Array(0);
    }

    const inputTensor = await OpenvinoBackend.createTensor(
      'i64',
      [1, tokenIds.length],
      BigInt64Array.from(tokenIds.map((tokenId) => BigInt(tokenId)))
    );
    const outputs = this.inferQwenCompiledModel(runtime.thinkerEmbeddings, [inputTensor]);
    const output = OpenvinoBackend.getTensorData(outputs[0]) as Float32Array | null;
    if (!output) {
      throw new Error('Qwen3-ASR thinker embeddings output is empty.');
    }
    return output;
  }

  private static async encodeQwenAudio(runtime: Qwen3AsrRuntime, audio: Float32Array) {
    const maxLength = Math.round(runtime.promptTemplate.n_samples || 160000);
    const extracted = await runtime.featureExtractor(audio, { max_length: maxLength });
    const inputFeatures = extracted?.input_features;
    const dims = Array.isArray(inputFeatures?.dims) ? inputFeatures.dims.map((value: any) => Number(value)) : [];
    const featureData = inputFeatures?.data as Float32Array | undefined;
    if (!featureData || dims.length < 3) {
      throw new Error('Qwen3-ASR feature extractor returned invalid mel features.');
    }

    const melBins = dims[dims.length - 2];
    const frames = dims[dims.length - 1];
    const inputTensor = await OpenvinoBackend.createTensor('f32', [melBins, frames], featureData);
    const outputs = this.inferQwenCompiledModel(runtime.audioEncoder, [inputTensor]);
    const outputTensor = outputs[0];
    const output = OpenvinoBackend.getTensorData(outputTensor) as Float32Array | null;
    const outputShape = this.getTensorShape(outputTensor);
    if (!output || outputShape.length === 0) {
      throw new Error('Qwen3-ASR audio encoder output is empty.');
    }

    const hiddenSize = outputShape[outputShape.length - 1];
    const sequenceLength = output.length / hiddenSize;
    return {
      data: output,
      sequenceLength,
      hiddenSize,
    };
  }

  private static concatQwenEmbeddings(parts: Float32Array[], hiddenSize: number) {
    const totalLength = parts.reduce((sum, part) => sum + part.length, 0);
    if (totalLength === 0) {
      return {
        data: new Float32Array(0),
        sequenceLength: 0,
      };
    }
    const output = new Float32Array(totalLength);
    let offset = 0;
    for (const part of parts) {
      output.set(part, offset);
      offset += part.length;
    }
    const sequenceLength = output.length / hiddenSize;
    return { data: output, sequenceLength };
  }

  private static pickGreedyToken(logits: Float32Array, vocabSize: number) {
    let bestToken = 0;
    let bestScore = Number.NEGATIVE_INFINITY;
    const start = Math.max(0, logits.length - vocabSize);
    for (let index = start; index < logits.length; index += 1) {
      const score = logits[index];
      if (score > bestScore) {
        bestScore = score;
        bestToken = index - start;
      }
    }
    return bestToken;
  }

  private static joinTranscriptTexts(chunks: LocalAsrChunk[]) {
    return chunks
      .map((chunk) => String(chunk.text || '').trim())
      .filter(Boolean)
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  private static normalizeWhisperWordTimings(rawWords: any[]): LocalAsrWordTiming[] {
    return rawWords
      .map((word: any) => {
        const start = Number(word?.start_ts ?? word?.start ?? word?.startTs ?? Number.NaN);
        const end = Number(word?.end_ts ?? word?.end ?? word?.endTs ?? Number.NaN);
        const text = String(word?.word || word?.text || '').trim();
        if (!Number.isFinite(start) || !text) return null;
        return {
          text,
          start_ts: start,
          end_ts: Number.isFinite(end) && end > start ? end : undefined,
        } satisfies LocalAsrWordTiming;
      })
      .filter(Boolean) as LocalAsrWordTiming[];
  }

  private static async isWhisperMultilingualModel(modelPath: string) {
    const cacheKey = path.resolve(modelPath).toLowerCase();
    const cached = this.whisperMultilingualCache.get(cacheKey);
    if (typeof cached === 'boolean') {
      return cached;
    }

    let isMultilingual = true;
    const generationConfigPath = path.join(modelPath, 'generation_config.json');
    const configPath = path.join(modelPath, 'config.json');

    try {
      if (await fs.pathExists(generationConfigPath)) {
        const parsed = await fs.readJson(generationConfigPath);
        if (typeof parsed?.is_multilingual === 'boolean') {
          isMultilingual = parsed.is_multilingual;
        }
      } else if (await fs.pathExists(configPath)) {
        const parsed = await fs.readJson(configPath);
        const forcedDecoderIds = Array.isArray(parsed?.forced_decoder_ids) ? parsed.forced_decoder_ids : [];
        isMultilingual = forcedDecoderIds.length === 0;
      }
    } catch {
      isMultilingual = !/whisper[-_.](?:tiny|base|small|medium)[-_.]en(?:[\\/._-]|$)/i.test(modelPath);
    }

    this.whisperMultilingualCache.set(cacheKey, isMultilingual);
    return isMultilingual;
  }

  private static async getLocalWhisperGenerationConfig(input: {
    modelPath: string;
    language?: string;
    segmentation?: boolean;
    wordAlignment?: boolean;
    prompt?: string;
  }) {
    const normalizedLanguage = this.normalizeWhisperLanguage(input.language);
    const isMultilingual = await this.isWhisperMultilingualModel(input.modelPath);
    const effectiveLanguage = isMultilingual ? normalizedLanguage : undefined;
    const generationConfig: Record<string, unknown> = {
      return_timestamps: input.segmentation !== false,
      word_timestamps: input.segmentation !== false && input.wordAlignment !== false,
    };

    if (isMultilingual) {
      generationConfig.task = 'transcribe';
    }

    if (effectiveLanguage) {
      generationConfig.language = effectiveLanguage;
    }
    if (input.prompt && input.prompt.trim()) {
      generationConfig.initial_prompt = input.prompt.trim();
    }

    return generationConfig;
  }

  private static attachWordTimingsToChunks(chunks: LocalAsrChunk[], words: LocalAsrWordTiming[]): LocalAsrSegment[] {
    if (chunks.length === 0) return [];
    if (words.length === 0) {
      return chunks.map((chunk) => ({ ...chunk }));
    }

    const lastWordEnd = words.reduce((max, word) => {
      const end = Number(word.end_ts);
      return Number.isFinite(end) ? Math.max(max, end) : max;
    }, 0);
    const segments = chunks.map((chunk, index) => {
      const nextStart = index + 1 < chunks.length ? Number(chunks[index + 1]?.start_ts) : Number.NaN;
      const resolvedEnd =
        Number.isFinite(Number(chunk.end_ts))
          ? Number(chunk.end_ts)
          : Number.isFinite(nextStart) && nextStart > chunk.start_ts
            ? nextStart
            : lastWordEnd > chunk.start_ts
              ? lastWordEnd
              : undefined;
      return {
        ...chunk,
        end_ts: resolvedEnd,
        words: [] as LocalAsrWordTiming[],
      };
    });

    const epsilon = 0.05;
    let segmentIndex = 0;
    for (const word of words) {
      while (
        segmentIndex + 1 < segments.length &&
        Number.isFinite(Number(segments[segmentIndex].end_ts)) &&
        word.start_ts >= Number(segments[segmentIndex].end_ts) - epsilon
      ) {
        segmentIndex += 1;
      }

      let assignedIndex = -1;
      for (let index = Math.max(0, segmentIndex - 1); index < Math.min(segments.length, segmentIndex + 2); index += 1) {
        const segment = segments[index];
        const segmentEnd = Number(segment.end_ts);
        if (!Number.isFinite(segmentEnd)) continue;
        const wordEnd = Number.isFinite(Number(word.end_ts)) ? Number(word.end_ts) : word.start_ts;
        if (word.start_ts < segmentEnd + epsilon && wordEnd > segment.start_ts - epsilon) {
          assignedIndex = index;
          break;
        }
      }

      if (assignedIndex < 0) {
        assignedIndex = Math.min(segmentIndex, segments.length - 1);
      }
      segments[assignedIndex].words?.push(word);
    }

    return segments.map((segment) => ({
      ...segment,
      words: segment.words && segment.words.length > 0 ? segment.words : undefined,
    }));
  }

  private static async transcribeQwenAsrChunk(runtime: Qwen3AsrRuntime, audio: Float32Array, language?: string) {
    const normalizedLanguage = this.normalizeQwenAsrLanguage(language, runtime);
    const prefixIds = runtime.promptTemplate.prefix_ids || [];
    const suffixIds = [
      ...(runtime.promptTemplate.suffix_ids || []),
      ...((normalizedLanguage && runtime.promptTemplate.language_suffix_ids?.[normalizedLanguage]) || []),
    ];

    const [prefixEmbeddings, suffixEmbeddings, audioEmbeddings] = await Promise.all([
      this.embedQwenTokenIds(runtime, prefixIds),
      this.embedQwenTokenIds(runtime, suffixIds),
      this.encodeQwenAudio(runtime, audio),
    ]);

    const hiddenSize = audioEmbeddings.hiddenSize;
    const prefillEmbeddings = this.concatQwenEmbeddings([prefixEmbeddings, audioEmbeddings.data, suffixEmbeddings], hiddenSize);
    const prefillTensor = await OpenvinoBackend.createTensor(
      'f32',
      [1, prefillEmbeddings.sequenceLength, hiddenSize],
      prefillEmbeddings.data
    );
    const positionIds = await OpenvinoBackend.createTensor(
      'i64',
      [1, prefillEmbeddings.sequenceLength],
      BigInt64Array.from(Array.from({ length: prefillEmbeddings.sequenceLength }, (_, index) => BigInt(index)))
    );

    const prefillOutputs = this.inferQwenCompiledModel(runtime.decoderPrefill, [prefillTensor, positionIds]);

    let logitsTensor = prefillOutputs[0];
    let pastKeysTensor = prefillOutputs[1];
    let pastValuesTensor = prefillOutputs[2];

    const generatedTokenIds: number[] = [];
    const vocabSize = runtime.vocabSize;
    const maxNewTokens = this.getQwenAsrMaxNewTokens();
    let currentPosition = prefillEmbeddings.sequenceLength;
    let consecutiveSpecialTokens = 0;

    for (let step = 0; step < maxNewTokens; step += 1) {
      const logits = OpenvinoBackend.getTensorData(logitsTensor) as Float32Array | null;
      if (!logits) {
        throw new Error('Qwen3-ASR decoder returned empty logits.');
      }

      const nextTokenId = this.pickGreedyToken(logits, vocabSize);
      if (runtime.eosTokenIds.has(nextTokenId)) {
        break;
      }

      generatedTokenIds.push(nextTokenId);
      if (runtime.specialTokenIds.has(nextTokenId)) {
        consecutiveSpecialTokens += 1;
        if (consecutiveSpecialTokens >= 24) {
          break;
        }
      } else {
        consecutiveSpecialTokens = 0;
      }

      const tokenEmbeddings = await this.embedQwenTokenIds(runtime, [nextTokenId]);
      const newEmbedTensor = await OpenvinoBackend.createTensor('f32', [1, 1, hiddenSize], tokenEmbeddings);
      const newPosTensor = await OpenvinoBackend.createTensor(
        'i64',
        [1, 1],
        BigInt64Array.from([BigInt(currentPosition)])
      );

      const decodeOutputs = this.inferQwenCompiledModel(runtime.decoderKv, [
        newEmbedTensor,
        newPosTensor,
        pastKeysTensor,
        pastValuesTensor,
      ]);

      logitsTensor = decodeOutputs[0];
      pastKeysTensor = decodeOutputs[1];
      pastValuesTensor = decodeOutputs[2];
      currentPosition += 1;
    }

    return this.parseQwenAsrTranscript(runtime, generatedTokenIds);
  }

  private static async transcribeWithQwenAsr(input: {
    modelId: string;
    modelPath: string;
    audioPath: string;
    language?: string;
    prompt?: string;
    segmentation?: boolean;
  }) {
    const runtime = await this.getOrCreateQwenAsrRuntime(input.modelId, input.modelPath);
    const decodedAudio = await this.extractAudioForWhisper(input.audioPath);
    const audio = decodedAudio.audio;
    const chunkSamples = Math.max(16000, Math.round(runtime.promptTemplate.n_samples || 160000));
    const chunks: LocalAsrChunk[] = [];
    const totalSamples = audio.length;

    for (let offset = 0; offset < totalSamples; offset += chunkSamples) {
      const end = Math.min(totalSamples, offset + chunkSamples);
      const audioChunk = audio.slice(offset, end);
      const result = await this.transcribeQwenAsrChunk(runtime, audioChunk, input.language);
      const text = String(result.text || '').trim();
      if (!text) continue;

      chunks.push({
        start_ts: Number((offset / 16000).toFixed(3)),
        end_ts: Number((end / 16000).toFixed(3)),
        text,
      });
    }

    if (chunks.length === 0) {
      throw new Error('Local Qwen3-ASR model returned empty transcription.');
    }

    const finalChunks =
      input.segmentation === false
        ? [{ start_ts: 0, end_ts: chunks[chunks.length - 1]?.end_ts, text: this.joinTranscriptTexts(chunks) }]
        : chunks;

    return {
      chunks: finalChunks,
      meta: {
        providerType: 'openvino-local',
        backend: 'qwen3-asr-node',
        device: await this.getLocalAsrDevice(),
        callCount: chunks.length,
        rawSegmentCount: finalChunks.length,
        rawHasTimestamps: finalChunks.some((item) => Number.isFinite(item.end_ts as number)),
        autoLanguageFallbackUsed: false,
        fileLimitFallbackUsed: false,
        fileLimitChunkCount: 0,
        promptIgnored: Boolean(String(input.prompt || '').trim()),
        timing: {
          audioDecodeMs: decodedAudio.decodeMs,
          audioDecodeCacheHit: decodedAudio.cacheHit,
          audioSampleCount: decodedAudio.sampleCount,
        },
      },
    };
  }

  private static async disposeTranslatePipeline(pipeline: any) {
    if (!pipeline) return;
    const disposer = pipeline.dispose || pipeline.release || pipeline.close || pipeline.delete;
    if (typeof disposer === 'function') {
      await Promise.resolve(disposer.call(pipeline));
    }
  }

  private static async transcribeWithAsrHelper(input: {
    modelPath: string;
    runtime?: string;
    audioPath: string;
    language?: string;
    prompt?: string;
    segmentation?: boolean;
    wordAlignment?: boolean;
  }) {
    const requestedDevice = await this.getLocalAsrDevice();
    const cacheDir = this.getLocalAsrCacheDir(input.modelPath);
    const helperHealth = await OpenVinoAsrHelperClient.healthCheck(5000).catch(() => null);
    const expectedRuntimeKind = await this.detectHelperAsrRuntimeKind(input.modelPath);
    const isQwenRuntime = expectedRuntimeKind === 'qwen3_asr_official';
    let effectiveDevice = requestedDevice;
    let fallbackToCpu = false;
    const helperModelLoaded =
      Boolean(helperHealth?.modelLoaded) &&
      String(helperHealth?.modelPath || '').toLowerCase() === String(input.modelPath).toLowerCase() &&
      (
        this.normalizeRequestedDevice(helperHealth?.device, requestedDevice) === requestedDevice ||
        (isQwenRuntime && this.normalizeRequestedDevice(helperHealth?.device, requestedDevice) === 'CPU')
      ) &&
      (!expectedRuntimeKind || String(helperHealth?.runtimeKind || '').trim().toLowerCase() === expectedRuntimeKind);

    if (helperModelLoaded) {
      effectiveDevice = this.normalizeRequestedDevice(helperHealth?.device, requestedDevice);
      fallbackToCpu = effectiveDevice === 'CPU' && requestedDevice !== 'CPU';
    }

    if (!helperModelLoaded) {
      const loaded = await this.loadAsrHelperModelWithFallback({
        modelPath: input.modelPath,
        requestedDevice,
        cacheDir,
        helperRuntimeKind: expectedRuntimeKind,
      });
      effectiveDevice = loaded.effectiveDevice;
      fallbackToCpu = loaded.fallbackToCpu;
    }

    const startedAt = Date.now();
    const transcribePayload = {
      audioPath: input.audioPath,
      language: input.language || '',
      prompt: input.prompt || '',
      returnTimestamps: input.segmentation !== false,
      wordTimestamps: input.segmentation !== false && input.wordAlignment !== false,
      task: 'transcribe',
    };
    let result: any = null;

    const buildHelperResponse = (resultPayload: any, fallbackLanguage?: string) => {
      const runtimeKind = String(resultPayload?.runtimeKind || '').trim().toLowerCase();
      const resolvedRuntimeKind = runtimeKind || expectedRuntimeKind || 'whisper';
      const backend =
        resolvedRuntimeKind === 'qwen3_asr_official'
          ? 'qwen3-asr-official-helper'
          : resolvedRuntimeKind === 'ctc_asr'
            ? 'ctc-asr-helper'
            : 'whisper-helper';
      const rawChunks = Array.isArray(resultPayload?.chunks) ? resultPayload.chunks : [];
      const normalizedWords = this.normalizeWhisperWordTimings(Array.isArray(resultPayload?.words) ? resultPayload.words : []);
      const normalizedChunks = rawChunks
        .map((chunk: any) => {
          const start = Number(chunk?.start_ts ?? chunk?.start ?? chunk?.startTime ?? Number.NaN);
          const end = Number(chunk?.end_ts ?? chunk?.end ?? chunk?.endTime ?? Number.NaN);
          const text = String(chunk?.text || '').trim();
          if (!Number.isFinite(start) || !text) return null;
          return {
            start_ts: start,
            end_ts: Number.isFinite(end) && end > start ? end : undefined,
            text,
          };
        })
        .filter(Boolean) as Array<{ start_ts: number; end_ts?: number; text: string }>;
      const segments = this.attachWordTimingsToChunks(normalizedChunks, normalizedWords);
      const autoLanguageFallbackUsed = Boolean(fallbackLanguage);

      if (normalizedChunks.length > 0) {
        return {
          text: this.joinTranscriptTexts(normalizedChunks),
          chunks: normalizedChunks,
          segments,
          meta: {
            providerType: 'openvino-local',
            backend,
            runtimeKind: resolvedRuntimeKind,
            device: effectiveDevice,
            requestedDevice,
            fallbackToCpu,
            callCount: 1,
            rawSegmentCount: normalizedChunks.length,
            rawWordCount: normalizedWords.length,
            rawHasTimestamps: normalizedChunks.some((item) => Number.isFinite(item.end_ts as number)),
            nativeWordTimestamps: normalizedWords.length > 0,
            autoLanguageFallbackUsed,
            autoLanguageFallbackLanguage: fallbackLanguage || null,
            fileLimitFallbackUsed: false,
            fileLimitChunkCount: 0,
            cacheDir,
            helperChunking: resultPayload?.chunking || null,
            forcedAlignment: resultPayload?.forcedAlignment || null,
            timing: {
              providerMs: Math.max(0, Date.now() - startedAt),
            },
            promptIgnored:
              runtimeKind === 'qwen3_asr_official' || runtimeKind === 'ctc_asr'
                ? Boolean(String(input.prompt || '').trim())
                : undefined,
          },
        };
      }

      const plainText = this.normalizeLocalAsrText(
        Array.isArray(resultPayload?.texts) && resultPayload.texts.length > 0
          ? resultPayload.texts
          : resultPayload?.text ?? resultPayload?.transcript ?? resultPayload?.transcription ?? resultPayload?.toString?.()
      );
      const synthesizedText =
        plainText ||
        normalizedWords
          .map((word) => String(word.text || '').trim())
          .filter(Boolean)
          .join(' ')
          .replace(/\s+/g, ' ')
          .trim();
      if (!synthesizedText) {
        this.log('asr helper returned empty transcription', {
          modelPath: input.modelPath,
          runtimeKind: resolvedRuntimeKind,
          rawRuntimeKind: runtimeKind || null,
          expectedRuntimeKind: expectedRuntimeKind || null,
          resultType: resultPayload == null ? String(resultPayload) : Array.isArray(resultPayload) ? 'array' : typeof resultPayload,
          resultKeys: resultPayload && typeof resultPayload === 'object' ? Object.keys(resultPayload) : [],
          rawChunkCount: rawChunks.length,
          normalizedChunkCount: normalizedChunks.length,
          rawWordCount: Array.isArray(resultPayload?.words) ? resultPayload.words.length : 0,
          normalizedWordCount: normalizedWords.length,
          fallbackLanguage: fallbackLanguage || null,
        });
        return null;
      }

      return {
        text: synthesizedText,
        chunks: [{ start_ts: 0, text: synthesizedText }],
        segments: [{ start_ts: 0, text: synthesizedText }],
        meta: {
          providerType: 'openvino-local',
          backend,
          runtimeKind: resolvedRuntimeKind,
          device: effectiveDevice,
          requestedDevice,
          fallbackToCpu,
          callCount: 1,
          rawSegmentCount: 1,
          rawHasTimestamps: false,
          autoLanguageFallbackUsed,
          autoLanguageFallbackLanguage: fallbackLanguage || null,
          fileLimitFallbackUsed: false,
          fileLimitChunkCount: 0,
          cacheDir,
          helperChunking: resultPayload?.chunking || null,
          forcedAlignment: resultPayload?.forcedAlignment || null,
          timing: {
            providerMs: Math.max(0, Date.now() - startedAt),
          },
          promptIgnored:
            runtimeKind === 'qwen3_asr_official' || runtimeKind === 'ctc_asr'
              ? Boolean(String(input.prompt || '').trim())
              : undefined,
        },
      };
    };

    try {
      result = await OpenVinoAsrHelperClient.transcribe(transcribePayload);
    } catch (error: any) {
      const message = String(error?.message || error || 'ASR helper transcription failed.');
      const canFallbackToCpuOnInference =
        isQwenRuntime &&
        effectiveDevice !== 'CPU' &&
        this.shouldFallbackQwenAsrToCpu(effectiveDevice, error);
      if (!canFallbackToCpuOnInference) {
        throw error;
      }

      this.log('qwen asr helper gpu inference allocation failed, retrying on cpu', {
        runtimeKind: expectedRuntimeKind,
        modelPath: input.modelPath,
        requestedDevice,
        effectiveDevice,
        message,
      });
      OpenVinoAsrHelperClient.forceShutdownNow('Retrying Qwen3-ASR helper on CPU after GPU inference allocation failure.');
      await OpenVinoAsrHelperClient.loadModel(input.modelPath, 'CPU', cacheDir);
      effectiveDevice = 'CPU';
      fallbackToCpu = true;
      result = await OpenVinoAsrHelperClient.transcribe(transcribePayload);
    }

    const directResponse = buildHelperResponse(result);
    if (directResponse) {
      return directResponse;
    }

    const runtimeKind = String(result?.runtimeKind || '').trim().toLowerCase();
    const helperShouldRetryAutoLanguage =
      this.isAutoLanguageRequest(input.language) && (runtimeKind === 'whisper' || !expectedRuntimeKind);
    if (helperShouldRetryAutoLanguage) {
      for (const fallbackLanguage of this.getWhisperAutoFallbackLanguages()) {
        this.log('asr helper auto language returned empty transcription, retrying with explicit language', {
          modelPath: input.modelPath,
          runtimeKind: runtimeKind || expectedRuntimeKind || 'whisper',
          rawRuntimeKind: runtimeKind || null,
          expectedRuntimeKind: expectedRuntimeKind || null,
          fallbackLanguage,
          requestedDevice,
        });
        try {
          const retryResult = await OpenVinoAsrHelperClient.transcribe({
            ...transcribePayload,
            language: fallbackLanguage,
          });
          const retryResponse = buildHelperResponse(retryResult, fallbackLanguage);
          if (retryResponse) {
            return retryResponse;
          }
        } catch (retryError: any) {
          this.log('asr helper explicit language retry failed', {
            modelPath: input.modelPath,
            fallbackLanguage,
            message: String(retryError?.message || retryError || 'ASR helper explicit language retry failed.'),
          });
        }
      }
    }

    this.log('asr helper exhausted transcription fallbacks', {
      modelPath: input.modelPath,
      runtimeKind: runtimeKind || expectedRuntimeKind || 'whisper',
      rawRuntimeKind: runtimeKind || null,
      expectedRuntimeKind: expectedRuntimeKind || null,
      requestedDevice,
      attemptedAutoFallback: helperShouldRetryAutoLanguage,
    });
    throw new Error('Local ASR model returned empty transcription.');
  }

  private static async disposeAsrPipeline() {
    const cached = this.asrPipelineCache;
    this.asrPipelineCache = null;
    if (!cached?.pipeline) return;
    await this.disposeTranslatePipeline(cached.pipeline);
  }

  private static async createAsrPipeline(
    factory: OpenvinoPipelineFactory,
    modelId: string,
    modelPath: string,
    requestedDevice: string,
    wordTimestampsEnabled = true
  ): Promise<CachedAsrPipeline> {
    const configuredDevice = this.getConfiguredLocalAsrDevice();
    const candidates = [requestedDevice];
    if (!configuredDevice && requestedDevice !== 'AUTO') {
      candidates.push('AUTO');
    }

    let lastError: unknown = null;
    for (const device of candidates) {
      for (const useCache of [true, false]) {
        const properties = this.getAsrPipelineProperties(modelId, wordTimestampsEnabled, useCache);
        const timeoutMs = this.getAsrTimeoutMs('load');
        const startedAt = Date.now();
        try {
          const pipeline = await this.withTimeout(
            Promise.resolve((factory as any)(modelPath, device, properties)),
            timeoutMs,
            `Local ASR model load timed out (${timeoutMs} ms).`
          );
          return {
            modelId,
            modelPath,
            pipeline,
            requestedDevice: device,
            cacheDir: typeof properties.CACHE_DIR === 'string' ? properties.CACHE_DIR : '',
            loadMs: Math.max(0, Date.now() - startedAt),
            wordTimestampsEnabled,
          };
        } catch (error) {
          lastError = error;
          const cacheDir = typeof properties.CACHE_DIR === 'string' ? properties.CACHE_DIR : '';
          const serializationCacheError = useCache && cacheDir && this.isOpenvinoSerializationCacheError(error);
          this.log('local asr pipeline load failed', {
            modelId,
            requestedDevice: device,
            useCache,
            serializationCacheError,
            fallbackPending:
              serializationCacheError || device !== candidates[candidates.length - 1],
            message: error instanceof Error ? error.message : String(error),
          });
          if (serializationCacheError) {
            try {
              await fs.remove(cacheDir);
            } catch {
              // Ignore cache cleanup failures and still retry without CACHE_DIR.
            }
            continue;
          }
          break;
        }
      }
    }

    throw lastError instanceof Error ? lastError : new Error(String(lastError || 'Failed to load local ASR pipeline.'));
  }

  private static async getOrCreateAsrPipeline(
    modelId: string,
    modelPath: string,
    wordTimestampsEnabled = true,
    requestedDeviceOverride?: string
  ) {
    if (this.qwenAsrRuntimeCache) {
      this.qwenAsrRuntimeCache = null;
    }

    const factory = await this.getWhisperAsrPipelineFactory();
    const requestedDevice = requestedDeviceOverride
      ? this.normalizeRequestedDevice(requestedDeviceOverride, 'AUTO')
      : await this.getLocalAsrDevice();
    const cached = this.asrPipelineCache;
    if (
      cached &&
      cached.modelId === modelId &&
      cached.modelPath === modelPath &&
      cached.wordTimestampsEnabled === wordTimestampsEnabled &&
      cached.requestedDevice === requestedDevice
    ) {
      const nativeGenerate = cached.pipeline?.pipeline?.generate;
      if (typeof nativeGenerate === 'function') {
        return cached;
      }
      this.log('cached asr pipeline lost native generate binding, rebuilding wrapper', {
        modelId,
        requestedDevice,
      });
      const refreshed = await this.createAsrPipeline(
        factory,
        modelId,
        modelPath,
        requestedDevice,
        wordTimestampsEnabled
      );
      this.asrPipelineCache = refreshed;
      return refreshed;
    }

    if (cached) {
      this.log('switching asr model, releasing previous pipeline', {
        fromModelId: cached.modelId,
        toModelId: modelId,
      });
      await this.disposeAsrPipeline();
    }

    this.asrPipelineCache = await this.createAsrPipeline(factory, modelId, modelPath, requestedDevice, wordTimestampsEnabled);
    return this.asrPipelineCache;
  }

  private static async extractAudioForWhisper(filePath: string): Promise<LocalAudioDecodeResult> {
    const stats = await fs.stat(filePath);
    const cacheKey = this.buildAudioDecodeCacheKey(filePath, { size: stats.size, mtimeMs: stats.mtimeMs });
    const cached = this.audioDecodeCache.get(cacheKey);
    if (cached) {
      this.audioDecodeCache.delete(cacheKey);
      this.audioDecodeCache.set(cacheKey, {
        ...cached,
        cachedAt: Date.now(),
      });
      return {
        audio: cached.audio,
        decodeMs: 0,
        cacheHit: true,
        sampleCount: cached.sampleCount,
        cacheKey,
      };
    }

    const startedAt = Date.now();
    const audio = await new Promise<Float32Array>((resolve, reject) => {
      const ffmpeg = spawn(resolveToolCommand('ffmpeg'), [
        '-i',
        filePath,
        '-f',
        'f32le',
        '-ac',
        '1',
        '-ar',
        '16000',
        'pipe:1',
      ], {
        windowsHide: true,
      });

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

        const stderrText = Buffer.concat(stderrChunks).toString('utf8').trim();
        reject(new Error(`FFmpeg audio decode failed (${code}): ${stderrText}`));
      });
      ffmpeg.on('error', (error) => {
        reject(new Error(`Failed to start FFmpeg for local ASR: ${error.message}`));
      });
    });

    const result = {
      audio,
      decodeMs: Math.max(0, Date.now() - startedAt),
      cacheHit: false,
      sampleCount: audio.length,
      cacheKey,
    } satisfies LocalAudioDecodeResult;
    if (this.getAudioDecodeCacheMaxItems() > 0) {
      this.audioDecodeCache.set(cacheKey, {
        cacheKey,
        audio,
        sampleCount: audio.length,
        cachedAt: Date.now(),
      });
      this.trimAudioDecodeCache();
    }
    return result;
  }

  private static async generateWithWhisperPipeline(
    pipeline: any,
    audio: Float32Array,
    generationConfig: Record<string, unknown>
  ) {
    const nativeGenerate = pipeline?.pipeline?.generate;
    if (typeof nativeGenerate === 'function') {
      return await new Promise<any>((resolve, reject) => {
        nativeGenerate.call(
          pipeline.pipeline,
          audio,
          generationConfig,
          undefined,
          (error: Error | null, result: any) => {
            if (error) {
              reject(error);
              return;
            }
            resolve(result);
          }
        );
      });
    }

    return pipeline.generate(audio, { generationConfig });
  }

  static async translateWithLocalModel(input: {
    modelId: string;
    modelPath: string;
    prompt: string;
    messages?: Array<Record<string, unknown>>;
    maxNewTokens?: number;
    generation?: LocalTranslateGenerationOptions;
  }) {
    this.traceLocalTranslate('translateWithLocalModel:start', {
      modelId: input.modelId,
      modelPath: input.modelPath,
      hasMessages: Array.isArray(input.messages) && input.messages.length > 0,
      promptLength: String(input.prompt || '').length,
    });
    const generation = input.generation || {};
    const maxNewTokens = Math.max(32, Number(input.maxNewTokens || 1024));
    const primaryConfig: Record<string, unknown> = {
      max_new_tokens: maxNewTokens,
      do_sample: generation.doSample ?? false,
      temperature: typeof generation.temperature === 'number' ? generation.temperature : 0.1,
      top_p: typeof generation.topP === 'number' ? generation.topP : 0.9,
    };

    if (typeof generation.topK === 'number') {
      primaryConfig.top_k = Math.max(1, Math.floor(generation.topK));
    }
    if (typeof generation.minP === 'number') {
      primaryConfig.min_p = Math.max(0, generation.minP);
    }
    if (typeof generation.maxNgramSize === 'number') {
      primaryConfig.max_ngram_size = Math.max(1, Math.floor(generation.maxNgramSize));
    }
    if (typeof generation.numAssistantTokens === 'number') {
      primaryConfig.num_assistant_tokens = Math.max(1, Math.floor(generation.numAssistantTokens));
    }
    if (typeof generation.assistantConfidenceThreshold === 'number') {
      primaryConfig.assistant_confidence_threshold = Math.max(0, generation.assistantConfidenceThreshold);
    }
    if (typeof generation.repetitionPenalty === 'number') {
      primaryConfig.repetition_penalty = generation.repetitionPenalty;
    }
    if (typeof generation.noRepeatNgramSize === 'number') {
      primaryConfig.no_repeat_ngram_size = Math.max(0, Math.floor(generation.noRepeatNgramSize));
    }
    if (typeof generation.applyChatTemplate === 'boolean') {
      primaryConfig.apply_chat_template = generation.applyChatTemplate;
    }
    if (typeof generation.presencePenalty === 'number') {
      primaryConfig.presence_penalty = generation.presencePenalty;
    }
    if (typeof generation.frequencyPenalty === 'number') {
      primaryConfig.frequency_penalty = generation.frequencyPenalty;
    }

    let result: any;
    let generationInput: {
      prompt: string;
      messages?: Array<Record<string, unknown>>;
    } = {
      prompt: input.prompt,
      messages: Array.isArray(input.messages) ? input.messages : undefined,
    };
    const isTranslateGemma = this.isTranslateGemmaModelRef(input.modelId, input.modelPath);

    if (isTranslateGemma && Array.isArray(generationInput.messages) && generationInput.messages.length > 0) {
      const renderStartedAt = Date.now();
      generationInput = {
        prompt: await this.renderHfChatTemplate({
          modelPath: input.modelPath,
          messages: generationInput.messages,
        }),
      };
      this.traceLocalTranslate('translateWithLocalModel:render-chat-template', {
        modelId: input.modelId,
        elapsedMs: Date.now() - renderStartedAt,
        promptLength: generationInput.prompt.length,
      });
      primaryConfig.apply_chat_template = false;
    }

    const cachedPipeline = await this.getOrCreateTranslatePipeline(input.modelId, input.modelPath);
    const pipeline = cachedPipeline.pipeline;

    const runGenerate = async (
      targetPipeline: any,
      targetKind: TranslatePipelineKind,
      generationConfig: Record<string, unknown>,
      currentInput = generationInput
    ) =>
      targetKind === 'vlm'
        ? await targetPipeline.generate(currentInput, { generationConfig })
        : await targetPipeline.generate(currentInput, generationConfig);

    try {
      const generateStartedAt = Date.now();
      result = await runGenerate(pipeline, cachedPipeline.kind, primaryConfig);
      this.traceLocalTranslate('translateWithLocalModel:generate-primary', {
        modelId: input.modelId,
        elapsedMs: Date.now() - generateStartedAt,
      });
    } catch (primaryError) {
      const fallbackConfig: Record<string, unknown> = {
        max_new_tokens: maxNewTokens,
        do_sample: false,
        temperature: 0.2,
        top_p: 0.95,
      };
      const primaryMessage = String(primaryError instanceof Error ? primaryError.message : primaryError || '');
      const helperStateError =
        /another generation is already in progress|translation pipeline is not loaded|translate helper request timed out/i.test(
          primaryMessage
        );
      const templateParseError =
        Array.isArray(input.messages) &&
        input.messages.length > 0 &&
        /expected value expression|chat template|apply_chat_template|jinja/i.test(primaryMessage);

      if (templateParseError && generationInput.messages) {
        generationInput = {
          prompt: await this.renderHfChatTemplate({
            modelPath: input.modelPath,
            messages: generationInput.messages,
          }),
        };
        fallbackConfig.apply_chat_template = false;
      }

      if (helperStateError) {
        await this.releaseTranslateRuntime().catch(() => {});
        const refreshedPipeline = await this.getOrCreateTranslatePipeline(input.modelId, input.modelPath);
        const generateStartedAt = Date.now();
        result = await runGenerate(refreshedPipeline.pipeline, refreshedPipeline.kind, fallbackConfig);
        this.traceLocalTranslate('translateWithLocalModel:generate-fallback-refreshed', {
          modelId: input.modelId,
          elapsedMs: Date.now() - generateStartedAt,
        });
      } else {
        const generateStartedAt = Date.now();
        result = await runGenerate(pipeline, cachedPipeline.kind, fallbackConfig);
        this.traceLocalTranslate('translateWithLocalModel:generate-fallback', {
          modelId: input.modelId,
          elapsedMs: Date.now() - generateStartedAt,
        });
      }
    }

    const asTexts = Array.isArray(result?.texts) ? result.texts : [];
    const translated = this.normalizeTranslatedText(
      asTexts.length > 0 ? asTexts[0] : result?.text ?? result?.translatedText ?? result?.toString?.()
    );
    if (!translated) {
      throw new Error('Local translation model returned empty output.');
    }
    const postSnapshotStartedAt = Date.now();
    const postSnapshot = await this.safeSystemSnapshot(true);
    this.traceLocalTranslate('translateWithLocalModel:post-snapshot', {
      modelId: input.modelId,
      elapsedMs: Date.now() - postSnapshotStartedAt,
      ok: Boolean(postSnapshot),
    });
    cachedPipeline.runtimeDebug = {
      ...cachedPipeline.runtimeDebug,
      pipelineDevice:
        typeof result?.device === 'string' && result.device.trim()
          ? result.device
          : cachedPipeline.runtimeDebug.pipelineDevice,
      lastInference:
        this.pickLikelyActiveAccelerator(postSnapshot, 'post-generate-snapshot') ||
        cachedPipeline.runtimeDebug.lastInference,
      lastPerfMetrics: this.summarizePerfMetrics(result?.perfMetrics),
    };
    return translated;
  }

  static async transcribeWithLocalAsr(input: {
    modelId: string;
    modelPath: string;
    runtime?: string;
    audioPath: string;
    language?: string;
    prompt?: string;
    segmentation?: boolean;
    wordAlignment?: boolean;
  }) {
    const normalizedRuntime = this.normalizeAsrRuntimeValue(input.runtime);
    const detectedHelperRuntimeKind =
      normalizedRuntime !== 'openvino-whisper-node' ? await this.detectHelperAsrRuntimeKind(input.modelPath) : null;
    const shouldUseHelperRuntime =
      normalizedRuntime === 'openvino-ctc-asr' ||
      detectedHelperRuntimeKind === 'qwen3_asr_official' ||
      detectedHelperRuntimeKind === 'ctc_asr';
    if (shouldUseHelperRuntime) {
      return this.transcribeWithAsrHelper(input);
    }

    try {
      const decodedAudio = await this.extractAudioForWhisper(input.audioPath);
      const audio = decodedAudio.audio;
      const generationConfig = await this.getLocalWhisperGenerationConfig({
        modelPath: input.modelPath,
        language: input.language,
        segmentation: input.segmentation,
        wordAlignment: input.wordAlignment,
        prompt: input.prompt,
      });
      const runNativeWhisper = async (cachedPipeline: CachedAsrPipeline) => {
        const pipeline = cachedPipeline.pipeline;
        const timeoutMs = this.getAsrTimeoutMs('generate');
        const generateStartedAt = Date.now();
        const result = await this.withTimeout(
          Promise.resolve(this.generateWithWhisperPipeline(pipeline, audio, generationConfig)),
          timeoutMs,
          `Local ASR transcription timed out (${timeoutMs} ms).`
        );
        const generateMs = Math.max(0, Date.now() - generateStartedAt);
        return { cachedPipeline, result, generateMs };
      };

      let nativeRun: { cachedPipeline: CachedAsrPipeline; result: any; generateMs: number };
      let whisperFallbackToCpu = false;
      const originallyRequestedDevice = await this.getLocalAsrDevice();
      const configuredAsrDevice = this.getConfiguredLocalAsrDevice();
      try {
        const cachedPipeline = await this.getOrCreateAsrPipeline(
          input.modelId,
          input.modelPath,
          input.segmentation !== false && input.wordAlignment !== false
        );
        nativeRun = await runNativeWhisper(cachedPipeline);
      } catch (nativeError: any) {
        const message = String(nativeError?.message || nativeError || '');
        const canRetryWithoutNativeWordTimestamps =
          /cross_attention_qk_scaled_scores|Port for tensor name .*cross_attention/i.test(message);
        const canFallbackToCpu =
          !configuredAsrDevice &&
          this.shouldFallbackWhisperAsrToCpu(originallyRequestedDevice, nativeError);
        if (!canRetryWithoutNativeWordTimestamps && !canFallbackToCpu) {
          throw nativeError;
        }

        if (canRetryWithoutNativeWordTimestamps) {
          this.log('local whisper runtime lacks native word timestamp support, retrying without it', {
            modelId: input.modelId,
            modelPath: input.modelPath,
          });
          await this.disposeAsrPipeline().catch(() => {});
          const fallbackPipeline = await this.getOrCreateAsrPipeline(input.modelId, input.modelPath, false);
          nativeRun = await runNativeWhisper(fallbackPipeline);
        } else {
          this.log('local whisper runtime hit device limitation, retrying on cpu', {
            modelId: input.modelId,
            modelPath: input.modelPath,
            message,
          });
          await this.disposeAsrPipeline().catch(() => {});
          const cpuPipeline = await this.getOrCreateAsrPipeline(
            input.modelId,
            input.modelPath,
            input.segmentation !== false && input.wordAlignment !== false,
            'CPU'
          );
          nativeRun = await runNativeWhisper(cpuPipeline);
          whisperFallbackToCpu = true;
        }
      }

      const { cachedPipeline, result, generateMs } = nativeRun;

      const rawChunks = Array.isArray(result?.chunks) ? result.chunks : [];
      const normalizedWords = this.normalizeWhisperWordTimings(Array.isArray(result?.words) ? result.words : []);
      const normalizedChunks = rawChunks
        .map((chunk: any) => {
          const start = Number(chunk?.start_ts ?? chunk?.start ?? chunk?.startTs ?? Number.NaN);
          const end = Number(chunk?.end_ts ?? chunk?.end ?? chunk?.endTs ?? Number.NaN);
          const text = String(chunk?.text || '').trim();
          if (!Number.isFinite(start) || !text) return null;
          return {
            start_ts: start,
            end_ts: Number.isFinite(end) && end > start ? end : undefined,
            text,
          };
        })
        .filter(Boolean) as Array<{ start_ts: number; end_ts?: number; text: string }>;
      const segments = this.attachWordTimingsToChunks(normalizedChunks, normalizedWords);

      if (normalizedChunks.length > 0) {
        return {
          text: this.joinTranscriptTexts(normalizedChunks),
          chunks: normalizedChunks,
          segments,
          meta: {
            providerType: 'openvino-local',
            backend: 'whisper-node',
            runtimeKind: 'whisper',
            device: cachedPipeline.requestedDevice,
            requestedDevice: originallyRequestedDevice,
            fallbackToCpu: whisperFallbackToCpu,
            callCount: 1,
            rawSegmentCount: normalizedChunks.length,
            rawWordCount: normalizedWords.length,
            rawHasTimestamps: normalizedChunks.some((item) => Number.isFinite(item.end_ts as number)),
            nativeWordTimestamps: normalizedWords.length > 0,
            autoLanguageFallbackUsed: false,
            fileLimitFallbackUsed: false,
            fileLimitChunkCount: 0,
            cacheDir: cachedPipeline.cacheDir,
            timing: {
              pipelineLoadMs: cachedPipeline.loadMs,
              audioDecodeMs: decodedAudio.decodeMs,
              audioDecodeCacheHit: decodedAudio.cacheHit,
              audioSampleCount: decodedAudio.sampleCount,
              asrGenerateMs: generateMs,
              providerMs: decodedAudio.decodeMs + generateMs,
            },
          },
        };
      }

      const plainText = this.normalizeLocalAsrText(
        Array.isArray(result?.texts) && result.texts.length > 0
          ? result.texts
          : result?.text ?? result?.transcript ?? result?.transcription ?? result?.toString?.()
      );
      if (!plainText) {
        this.log('local whisper native runtime returned empty transcription, retrying with helper', {
          modelId: input.modelId,
          modelPath: input.modelPath,
          requestedDevice: originallyRequestedDevice,
          resultKeys: result && typeof result === 'object' ? Object.keys(result) : [],
        });
        return this.transcribeWithAsrHelper(input);
      }

      return {
        text: plainText,
        chunks: [{ start_ts: 0, text: plainText }],
        segments: [{ start_ts: 0, text: plainText }],
        meta: {
          providerType: 'openvino-local',
          backend: 'whisper-node',
          runtimeKind: 'whisper',
          device: cachedPipeline.requestedDevice,
          requestedDevice: originallyRequestedDevice,
          fallbackToCpu: whisperFallbackToCpu,
          callCount: 1,
          rawSegmentCount: 1,
          rawWordCount: 0,
          rawHasTimestamps: false,
          nativeWordTimestamps: false,
          autoLanguageFallbackUsed: false,
          fileLimitFallbackUsed: false,
          fileLimitChunkCount: 0,
          cacheDir: cachedPipeline.cacheDir,
          timing: {
            pipelineLoadMs: cachedPipeline.loadMs,
            audioDecodeMs: decodedAudio.decodeMs,
            audioDecodeCacheHit: decodedAudio.cacheHit,
            audioSampleCount: decodedAudio.sampleCount,
            asrGenerateMs: generateMs,
            providerMs: decodedAudio.decodeMs + generateMs,
          },
        },
      };
    } catch (nativeError: any) {
      const canFallbackToHelper = /WhisperPipeline is unavailable/i.test(String(nativeError?.message || nativeError || ''));
      if (!canFallbackToHelper) {
        throw nativeError;
      }
      return this.transcribeWithAsrHelper(input);
    }
  }

  static async convertOfficialQwenAsrModel(input: { repoId: string; outputDir: string; useLocalDir?: boolean }) {
    const scriptPath = this.getOfficialQwenConvertScriptPath();
    if (!(await fs.pathExists(scriptPath))) {
      throw new Error(`Official Qwen3-ASR convert script not found: ${scriptPath}`);
    }

    const args = [scriptPath, '--repo-id', input.repoId, '--output-dir', input.outputDir];
    if (input.useLocalDir) {
      args.push('--use-local-dir');
    }
    const compressionMode = this.getQwenAsrConversionCompressionMode();
    if (compressionMode && compressionMode !== 'none') {
      args.push('--compression', compressionMode);
    }

    return new Promise((resolve, reject) => {
      const child = spawn(this.getPythonCommand(), args, {
        windowsHide: true,
        env: {
          ...process.env,
          PYTHONIOENCODING: 'utf-8',
        },
        stdio: ['ignore', 'pipe', 'pipe'],
      });

      let stdout = '';
      let stderr = '';
      const timeoutMs = Math.max(
        60000,
        Math.floor(this.getEnvNumber('OPENVINO_HELPER_CONVERT_TIMEOUT_MS', 7200000, 60000, 43200000))
      );
      const timeout = setTimeout(() => {
        try {
          child.kill();
        } catch {}
        reject(new Error(`Official Qwen3-ASR conversion timed out (${timeoutMs} ms).`));
      }, timeoutMs);

      child.stdout?.on('data', (chunk) => {
        stdout += chunk.toString('utf8');
      });
      child.stderr?.on('data', (chunk) => {
        stderr += chunk.toString('utf8');
      });
      child.on('error', (error) => {
        clearTimeout(timeout);
        reject(new Error(`Failed to start official Qwen3-ASR conversion: ${error.message}`));
      });
      child.on('close', (code) => {
        clearTimeout(timeout);
        try {
          const parsed = this.parseJsonFromToolOutput(stdout);
          if (code === 0 && parsed?.converted) {
            resolve(parsed);
            return;
          }
          const errorMessage = String(parsed?.error || '').trim() || stderr.trim() || `exit code ${code ?? 'null'}`;
          reject(
            new Error(
              `Official Qwen3-ASR conversion failed: ${errorMessage}${this.formatToolFailureDetails(parsed, stderr)}`
            )
          );
        } catch (error: any) {
          reject(
            new Error(
              `Official Qwen3-ASR conversion returned invalid output: ${String(error?.message || error)}${
                stderr.trim() ? `\n${stderr.trim()}` : ''
              }`
            )
          );
        }
      });
    });
  }

  private static normalizeAsrRuntimeValue(runtime: string | undefined) {
    const normalized = String(runtime || '').trim().toLowerCase();
    if (normalized === 'openvino-qwen3-asr') {
      return 'openvino-qwen3-asr';
    }
    if (normalized === 'openvino-ctc-asr') {
      return 'openvino-ctc-asr';
    }
    if (normalized === 'openvino-whisper-node') {
      return 'openvino-whisper-node';
    }
    return normalized;
  }

  private static parseJsonFromToolOutput(stdout: string) {
    const raw = String(stdout || '').trim();
    if (!raw) return {};
    try {
      return JSON.parse(raw);
    } catch {
      const firstBrace = raw.indexOf('{');
      const lastBrace = raw.lastIndexOf('}');
      if (firstBrace >= 0 && lastBrace > firstBrace) {
        return JSON.parse(raw.slice(firstBrace, lastBrace + 1));
      }
      throw new Error('No JSON payload found in tool output.');
    }
  }

  private static async renderHfChatTemplate(input: { modelPath: string; messages: Array<Record<string, unknown>> }) {
    const scriptPath = this.getChatTemplateRenderScriptPath();
    if (!(await fs.pathExists(scriptPath))) {
      throw new Error(`Hugging Face chat template renderer not found: ${scriptPath}`);
    }

    return await new Promise<string>((resolve, reject) => {
      const child = spawn(this.getPythonCommand(), [scriptPath], {
        windowsHide: true,
        env: {
          ...process.env,
          PYTHONIOENCODING: 'utf-8',
          PYTHONUTF8: '1',
        },
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      let stdout = '';
      let stderr = '';
      const timeoutMs = Math.max(
        30000,
        Math.floor(this.getEnvNumber('OPENVINO_HELPER_LOAD_TIMEOUT_MS', 180000, 1000, 1800000))
      );
      const timeout = setTimeout(() => {
        try {
          child.kill();
        } catch {}
        reject(new Error(`Hugging Face chat template rendering timed out (${timeoutMs} ms).`));
      }, timeoutMs);

      child.stdout?.on('data', (chunk) => {
        stdout += chunk.toString('utf8');
      });
      child.stderr?.on('data', (chunk) => {
        stderr += chunk.toString('utf8');
      });
      child.on('error', (error) => {
        clearTimeout(timeout);
        reject(new Error(`Failed to start Hugging Face chat template renderer: ${error.message}`));
      });
      child.on('close', (code) => {
        clearTimeout(timeout);
        try {
          const parsed = this.parseJsonFromToolOutput(stdout);
          const prompt = String(parsed?.prompt || '').trim();
          if (code === 0 && prompt) {
            resolve(prompt);
            return;
          }
          const errorMessage = stderr.trim() || `exit code ${code ?? 'null'}`;
          reject(new Error(`Hugging Face chat template rendering failed: ${errorMessage}`));
        } catch (error: any) {
          reject(
            new Error(
              `Hugging Face chat template renderer returned invalid output: ${String(error?.message || error)}${
                stderr.trim() ? `\n${stderr.trim()}` : ''
              }`
            )
          );
        }
      });

      child.stdin?.end(
        JSON.stringify({
          modelDir: input.modelPath,
          messages: input.messages,
        })
      );
    });
  }

  static async countHfInputTokens(input: {
    modelPath: string;
    entries: Array<{
      prompt?: string;
      messages?: Array<Record<string, unknown>>;
    }>;
  }) {
    const scriptPath = this.getInputTokenCountScriptPath();
    if (!(await fs.pathExists(scriptPath))) {
      throw new Error(`Hugging Face input token counter not found: ${scriptPath}`);
    }

    return await new Promise<number[]>((resolve, reject) => {
      const child = spawn(this.getPythonCommand(), [scriptPath], {
        windowsHide: true,
        env: {
          ...process.env,
          PYTHONIOENCODING: 'utf-8',
          PYTHONUTF8: '1',
        },
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      let stdout = '';
      let stderr = '';
      const timeoutMs = Math.max(
        30000,
        Math.floor(this.getEnvNumber('OPENVINO_HELPER_LOAD_TIMEOUT_MS', 180000, 1000, 1800000))
      );
      const timeout = setTimeout(() => {
        try {
          child.kill();
        } catch {}
        reject(new Error(`Hugging Face input token counting timed out (${timeoutMs} ms).`));
      }, timeoutMs);

      child.stdout?.on('data', (chunk) => {
        stdout += chunk.toString('utf8');
      });
      child.stderr?.on('data', (chunk) => {
        stderr += chunk.toString('utf8');
      });
      child.on('error', (error) => {
        clearTimeout(timeout);
        reject(new Error(`Failed to start Hugging Face input token counter: ${error.message}`));
      });
      child.on('close', (code) => {
        clearTimeout(timeout);
        try {
          const parsed = this.parseJsonFromToolOutput(stdout);
          const counts = Array.isArray(parsed?.counts)
            ? parsed.counts.map((value: unknown) => Math.max(0, Math.floor(Number(value) || 0)))
            : [];
          if (code === 0 && counts.length === input.entries.length) {
            resolve(counts);
            return;
          }
          const errorMessage = stderr.trim() || `exit code ${code ?? 'null'}`;
          reject(new Error(`Hugging Face input token counting failed: ${errorMessage}`));
        } catch (error: any) {
          reject(
            new Error(
              `Hugging Face input token counter returned invalid output: ${String(error?.message || error)}${
                stderr.trim() ? `\n${stderr.trim()}` : ''
              }`
            )
          );
        }
      });

      child.stdin?.end(
        JSON.stringify({
          modelDir: input.modelPath,
          entries: input.entries,
        })
      );
    });
  }

  private static formatToolFailureDetails(parsed: any, stderr: string) {
    const detailParts: string[] = [];
    const stdoutTail = Array.isArray(parsed?.detail?.stdoutTail)
      ? parsed.detail.stdoutTail.map((item: any) => String(item || '').trim()).filter(Boolean)
      : [];
    const converterStderrTail = Array.isArray(parsed?.detail?.stderrTail)
      ? parsed.detail.stderrTail.map((item: any) => String(item || '').trim()).filter(Boolean)
      : [];
    const stderrTail = String(stderr || '')
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .slice(-40);

    if (stdoutTail.length > 0) {
      detailParts.push(`stdout tail:\n${stdoutTail.join('\n')}`);
    }
    if (converterStderrTail.length > 0) {
      detailParts.push(`converter stderr tail:\n${converterStderrTail.join('\n')}`);
    }
    if (stderrTail.length > 0) {
      detailParts.push(`stderr tail:\n${stderrTail.join('\n')}`);
    }

    return detailParts.length > 0 ? `\n${detailParts.join('\n\n')}` : '';
  }

  private static async detectHelperAsrRuntimeKind(modelPath: string): Promise<'qwen3_asr_official' | 'ctc_asr' | null> {
    const officialChecks = await Promise.all([
      fs.pathExists(path.join(modelPath, 'thinker', 'openvino_thinker_language_model.xml')),
      fs.pathExists(path.join(modelPath, 'thinker', 'openvino_thinker_audio_model.xml')),
      fs.pathExists(path.join(modelPath, 'thinker', 'openvino_thinker_audio_encoder_model.xml')),
      fs.pathExists(path.join(modelPath, 'thinker', 'openvino_thinker_embedding_model.xml')),
    ]);
    if (officialChecks.every(Boolean)) {
      return 'qwen3_asr_official';
    }

    const ctcChecks = await Promise.all([
      fs.pathExists(path.join(modelPath, 'openvino_model.xml')),
      fs.pathExists(path.join(modelPath, 'openvino_model.bin')),
      fs.pathExists(path.join(modelPath, 'preprocessor_config.json')),
      fs.pathExists(path.join(modelPath, 'tokenizer_config.json')),
    ]);
    const hasCtcTokenizer =
      (await fs.pathExists(path.join(modelPath, 'tokenizer.json'))) ||
      (await fs.pathExists(path.join(modelPath, 'vocab.json'))) ||
      (await fs.pathExists(path.join(modelPath, 'vocab.txt')));
    if (ctcChecks.every(Boolean) && hasCtcTokenizer) {
      return 'ctc_asr';
    }

    return null;
  }

  static async convertHuggingFaceModel(input: {
    repoId: string;
    outputDir: string;
    type: 'asr' | 'translate';
    sourceFormat: string;
    conversionMethod: string;
    runtimeLayout: string;
    hfTask?: string;
    envOverrides?: Record<string, string>;
  }) {
    const scriptPath = this.getGenericHfConvertScriptPath();
    if (!(await fs.pathExists(scriptPath))) {
      throw new Error(`Generic Hugging Face OpenVINO convert script not found: ${scriptPath}`);
    }

    const args = [
      scriptPath,
      '--repo-id',
      input.repoId,
      '--output-dir',
      input.outputDir,
      '--type',
      input.type,
      '--source-format',
      input.sourceFormat,
      '--conversion-method',
      input.conversionMethod,
      '--runtime-layout',
      input.runtimeLayout,
    ];
    if (String(input.hfTask || '').trim()) {
      args.push('--hf-task', String(input.hfTask).trim());
    }

    return new Promise((resolve, reject) => {
      const child = spawn(this.getPythonCommand(), args, {
        windowsHide: true,
        env: {
          ...process.env,
          ...(input.envOverrides || {}),
          PYTHONIOENCODING: 'utf-8',
        },
        stdio: ['ignore', 'pipe', 'pipe'],
      });

      let stdout = '';
      let stderr = '';
      const timeoutMs = Math.max(
        60000,
        Math.floor(this.getEnvNumber('OPENVINO_HELPER_CONVERT_TIMEOUT_MS', 7200000, 60000, 43200000))
      );
      const timeout = setTimeout(() => {
        try {
          child.kill();
        } catch {}
        reject(new Error(`Generic Hugging Face OpenVINO conversion timed out (${timeoutMs} ms).`));
      }, timeoutMs);

      child.stdout?.on('data', (chunk) => {
        stdout += chunk.toString('utf8');
      });
      child.stderr?.on('data', (chunk) => {
        stderr += chunk.toString('utf8');
      });
      child.on('error', (error) => {
        clearTimeout(timeout);
        reject(new Error(`Failed to start generic Hugging Face OpenVINO conversion: ${error.message}`));
      });
      child.on('close', (code) => {
        clearTimeout(timeout);
        try {
          const parsed = this.parseJsonFromToolOutput(stdout);
          if (code === 0 && parsed?.converted) {
            resolve(parsed);
            return;
          }
          const errorMessage = String(parsed?.error || '').trim() || stderr.trim() || `exit code ${code ?? 'null'}`;
          reject(
            new Error(
              `Generic Hugging Face OpenVINO conversion failed: ${errorMessage}${this.formatToolFailureDetails(parsed, stderr)}`
            )
          );
        } catch (error: any) {
          reject(
            new Error(
              `Generic Hugging Face OpenVINO conversion returned invalid output: ${String(error?.message || error)}${
                stderr.trim() ? `\n${stderr.trim()}` : ''
              }`
            )
          );
        }
      });
    });
  }

  static async getOpenvinoRuntimeStatus() {
    const cacheMs = Math.round(this.getEnvNumber('OPENVINO_STATUS_CACHE_MS', 10000, 0, 300000));
    const now = Date.now();
    if (cacheMs > 0 && this.openvinoStatusCache && now - this.openvinoStatusCache.at < cacheMs) {
      return this.openvinoStatusCache.value;
    }

    let nodeAvailable = false;
    let nodeError = '';
    let genaiAvailable = false;
    let genaiError = '';
    let whisperNodeAvailable = false;
    let llmNodeAvailable = false;
    let vlmNodeAvailable = false;
    let qwenAsrNodeAvailable = false;
    const helperPath = OpenVinoAsrHelperClient.getHelperPath();
    const convertScriptPath = this.getOfficialQwenConvertScriptPath();
    const [helperExists, convertScriptExists] = await Promise.all([
      fs.pathExists(helperPath),
      fs.pathExists(convertScriptPath),
    ]);
    const qwenOfficialAvailable = helperExists && convertScriptExists;
    const [openvinoNodeVersion, openvinoGenaiVersion] = await Promise.all([
      this.getInstalledNodePackageVersion('openvino-node'),
      this.getInstalledNodePackageVersion('openvino-genai-node'),
    ]);

    try {
      await import('openvino-node');
      nodeAvailable = true;
      qwenAsrNodeAvailable = true;
    } catch (error: any) {
      nodeAvailable = false;
      nodeError = String(error?.message || error || 'OpenVINO Node packages are unavailable.');
      qwenAsrNodeAvailable = false;
    }

    try {
      const openvinoGenai = await import('openvino-genai-node');
      const whisperFactory =
        (openvinoGenai as any)?.WhisperPipeline ||
        (openvinoGenai as any)?.default?.WhisperPipeline;
      const llmFactory =
        (openvinoGenai as any)?.LLMPipeline ||
        (openvinoGenai as any)?.default?.LLMPipeline ||
        (openvinoGenai as any)?.default;
      const vlmFactory =
        (openvinoGenai as any)?.VLMPipeline ||
        (openvinoGenai as any)?.default?.VLMPipeline;
      whisperNodeAvailable = typeof whisperFactory === 'function';
      llmNodeAvailable = typeof llmFactory === 'function';
      vlmNodeAvailable = typeof vlmFactory === 'function';
      genaiAvailable = true;
    } catch (error: any) {
      genaiAvailable = false;
      genaiError = String(error?.message || error || 'OpenVINO GenAI packages are unavailable.');
    }

    const status = {
      node: {
        available: nodeAvailable,
        version: openvinoNodeVersion || undefined,
        error: nodeError || undefined,
      },
      genai: {
        available: genaiAvailable,
        version: openvinoGenaiVersion || undefined,
        whisperPipelineAvailable: whisperNodeAvailable,
        llmPipelineAvailable: llmNodeAvailable,
        vlmPipelineAvailable: vlmNodeAvailable,
        error: genaiError || undefined,
      },
      asr: {
        ready: whisperNodeAvailable || qwenAsrNodeAvailable || qwenOfficialAvailable,
        whisperPipelineAvailable: whisperNodeAvailable,
        qwenExplicitKvAvailable: qwenAsrNodeAvailable,
        qwenOfficialAvailable,
        error:
          whisperNodeAvailable || qwenAsrNodeAvailable || qwenOfficialAvailable
            ? undefined
            : nodeError || genaiError || 'No local OpenVINO ASR runtime is available.',
      },
      helper: {
        path: helperPath,
        exists: helperExists,
        healthy: helperExists,
        error: helperExists ? undefined : `ASR helper script is unavailable: ${helperPath}`,
      },
    };

    if (cacheMs > 0) {
      this.openvinoStatusCache = { at: now, value: status };
    }

    return status;
  }

  static async releaseTranslateRuntime() {
    const pipelineCount = this.translatePipelineCache.size;
    this.log('releaseTranslateRuntime start', { pipelineCount });
    const pipelines = Array.from(this.translatePipelineCache.values()).map((item) => item.pipeline);
    this.translatePipelineCache.clear();
    this.activeTranslateModelId = null;
    const settled = await Promise.allSettled(pipelines.map((pipeline) => this.disposeTranslatePipeline(pipeline)));
    const failed = settled.filter((item) => item.status === 'rejected').length;
    OpenVinoTranslateHelperClient.forceShutdownNow();
    OpenVinoGenaiTranslateHelperClient.forceShutdownNow();
    this.log('releaseTranslateRuntime done', {
      pipelineCount,
      released: pipelineCount - failed,
      failed,
    });
  }

  static async preloadTranslateRuntime(input: { modelId: string; modelPath: string }) {
    const startedAt = Date.now();
    const cachedPipeline = await this.getOrCreateTranslatePipeline(input.modelId, input.modelPath);
    this.traceLocalTranslate('preloadTranslateRuntime:done', {
      modelId: input.modelId,
      elapsedMs: Date.now() - startedAt,
      pipelineKind: cachedPipeline.kind,
      device: cachedPipeline.runtimeDebug.pipelineDevice,
    });
    return {
      success: true,
      runtimeDebug: { ...cachedPipeline.runtimeDebug },
    };
  }

  static async releaseAsrRuntime() {
    this.log('releaseAsrRuntime start', {
      hadLoadedModel: Boolean(this.asrPipelineCache),
    });
    await this.disposeAsrPipeline().catch(() => {});
    this.qwenAsrRuntimeCache = null;
    OpenVinoAsrHelperClient.forceShutdownNow();
    this.log('releaseAsrRuntime done');
  }

  static async releaseAllRuntimes() {
    this.log('releaseAllRuntimes start');
    await Promise.allSettled([this.releaseAsrRuntime(), this.releaseTranslateRuntime()]);
    this.log('releaseAllRuntimes done');
  }
}
