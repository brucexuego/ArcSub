import fs from 'fs-extra';
import path from 'path';
import { PathManager } from './path_manager.js';

type OpenvinoAddon = any;

export class OpenvinoBackend {
  private static addonPromise: Promise<OpenvinoAddon> | null = null;
  private static corePromise: Promise<any> | null = null;
  private static envPrepared = false;
  private static readonly isWindows = process.platform === 'win32';
  private static readonly isMac = process.platform === 'darwin';

  private static mergePathEnv(key: string, values: string[]) {
    const delimiter = path.delimiter;
    const current = String(process.env[key] || '');
    const existing = current.split(delimiter).filter(Boolean);
    const merged: string[] = [];
    const seen = new Set<string>();

    for (const value of [...values, ...existing]) {
      if (!value) continue;
      const normalized = path.normalize(value).toLowerCase();
      if (seen.has(normalized)) continue;
      seen.add(normalized);
      merged.push(value);
    }

    process.env[key] = merged.join(delimiter);
  }

  private static uniquePaths(values: string[]) {
    const unique: string[] = [];
    const seen = new Set<string>();

    for (const value of values) {
      if (!value) continue;
      const normalized = path.normalize(value);
      const dedupeKey = this.isWindows ? normalized.toLowerCase() : normalized;
      if (seen.has(dedupeKey)) continue;
      seen.add(dedupeKey);
      unique.push(normalized);
    }

    return unique;
  }

  private static collectExistingDirs(candidates: string[]) {
    return this.uniquePaths(candidates.filter((candidate) => fs.existsSync(candidate)));
  }

  private static getConfiguredRoots() {
    const configuredRoots = [
      process.env.OPENVINO_HELPER_INTEL_ROOT,
      process.env.INTEL_OPENVINO_DIR,
      process.env.OPENVINO_INSTALL_DIR,
      process.env.OPENVINO_DIR,
      process.env.OPENVINO_GENAI_DIR,
    ]
      .map((value) => String(value || '').trim())
      .filter(Boolean);

    const defaultParents = this.isWindows
      ? ['C:\\Program Files (x86)\\Intel', 'C:\\Program Files\\Intel']
      : ['/opt/intel'];

    const discoveredRoots: string[] = [];
    for (const parent of defaultParents) {
      try {
        if (!fs.existsSync(parent)) continue;
        for (const entry of fs.readdirSync(parent, { withFileTypes: true })) {
          if (!entry.isDirectory()) continue;
          if (!entry.name.toLowerCase().startsWith('openvino')) continue;
          discoveredRoots.push(path.join(parent, entry.name));
        }
      } catch {
        continue;
      }
    }

    return this.uniquePaths([...configuredRoots, ...discoveredRoots]).map((value) => path.resolve(value));
  }

  private static getRuntimeCandidateDirs(root: string) {
    const resolvedRoot = path.resolve(root);
    const runtimeRoots = new Set<string>();
    runtimeRoots.add(resolvedRoot);

    if (path.basename(resolvedRoot).toLowerCase() === 'runtime') {
      runtimeRoots.add(path.dirname(resolvedRoot));
    } else {
      runtimeRoots.add(path.join(resolvedRoot, 'runtime'));
    }

    const libDirs: string[] = [];
    for (const candidateRoot of runtimeRoots) {
      libDirs.push(
        path.join(candidateRoot, 'bin', 'intel64', 'Release'),
        path.join(candidateRoot, 'bin', 'intel64', 'Debug'),
        path.join(candidateRoot, '3rdparty', 'tbb', 'bin'),
        path.join(candidateRoot, 'lib'),
        path.join(candidateRoot, 'lib', 'intel64'),
        path.join(candidateRoot, 'lib', 'intel64', 'Release'),
        path.join(candidateRoot, 'lib', 'aarch64'),
        path.join(candidateRoot, 'lib', 'aarch64', 'Release'),
        path.join(candidateRoot, '3rdparty', 'tbb', 'lib'),
      );
    }

    return this.collectExistingDirs(libDirs);
  }

  private static mergeNativeLibraryEnv(values: string[]) {
    if (values.length === 0) return;

    this.mergePathEnv('OPENVINO_LIB_PATHS', values);
    if (this.isWindows) {
      this.mergePathEnv('PATH', values);
      return;
    }

    this.mergePathEnv('LD_LIBRARY_PATH', values);
    if (this.isMac) {
      this.mergePathEnv('DYLD_LIBRARY_PATH', values);
    }
  }

  private static prepareNativeEnv() {
    if (this.envPrepared) return;

    const repoBinDirs = [
      PathManager.resolveNodeModulesPath('openvino-node', 'bin'),
      PathManager.resolveNodeModulesPath('openvino-genai-node', 'bin'),
    ].filter((candidate) => fs.existsSync(candidate));

    const configuredRoots = this.getConfiguredRoots();
    const intelBinDirs = configuredRoots.flatMap((root) => this.getRuntimeCandidateDirs(root));

    const allBinDirs = this.uniquePaths([...repoBinDirs, ...intelBinDirs]);
    this.mergeNativeLibraryEnv(allBinDirs);

    this.envPrepared = true;
  }

  static async getAddon(): Promise<OpenvinoAddon> {
    if (!this.addonPromise) {
      this.prepareNativeEnv();
      this.addonPromise = import('openvino-node').then((module: any) => {
        if (!module?.addon) {
          throw new Error('openvino-node addon is unavailable.');
        }
        return module.addon;
      });
    }
    return this.addonPromise;
  }

  static async getCore() {
    if (!this.corePromise) {
      this.corePromise = this.getAddon().then((ov) => new ov.Core());
    }
    return this.corePromise;
  }

  static getBaselineModelDevice() {
    const configured = String(process.env.OPENVINO_BASELINE_DEVICE || '').trim().toUpperCase();
    if (configured) return configured;
    return 'CPU';
  }

  static async compileModel(modelPath: string, device = 'AUTO', config: Record<string, string | number | boolean> = {}) {
    const core = await this.getCore();
    return core.compileModel(modelPath, device, config);
  }

  static async convertModelToIr(sourcePath: string, outputXmlPath: string, options: { compressToFp16?: boolean } = {}) {
    const core = await this.getCore();
    const ov = await this.getAddon();
    const outputDir = path.dirname(outputXmlPath);
    await fs.ensureDir(outputDir);
    const model = await core.readModel(sourcePath);
    ov.saveModelSync(model, outputXmlPath, options.compressToFp16 ?? false);
  }

  static async createTensor(type: string, shape: number[], data: any) {
    const ov = await this.getAddon();
    return new ov.Tensor(type, shape, data);
  }

  static getPortName(port: any, fallback: string) {
    try {
      const name = String(port?.getAnyName?.() || '').trim();
      return name || fallback;
    } catch {
      return fallback;
    }
  }

  static getTensorData(tensor: any) {
    if (!tensor) return null;
    if (typeof tensor.getData === 'function') return tensor.getData();
    return tensor.data ?? null;
  }
}
