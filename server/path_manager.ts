import path from 'path';
import fs from 'fs-extra';

export class PathManager {
  private static rootPath = process.cwd();
  private static readonly PROJECT_ID_RE = /^[a-zA-Z0-9_-]{1,64}$/;
  private static readonly PROJECT_SUB_DIRS = ['assets', 'subtitles'] as const;

  private static getEnv(name: string) {
    const value = String(process.env[name] || '').trim();
    return value || null;
  }

  private static resolveFromRoot(targetPath: string) {
    return path.isAbsolute(targetPath) ? path.normalize(targetPath) : path.resolve(this.rootPath, targetPath);
  }

  private static getRuntimeRoot() {
    const configured = this.getEnv('APP_RUNTIME_DIR');
    return configured ? this.resolveFromRoot(configured) : null;
  }

  private static resolveConfiguredDir(explicitEnv: string, runtimeSubdir: string) {
    const explicit = this.getEnv(explicitEnv);
    if (explicit) {
      return this.resolveFromRoot(explicit);
    }

    return path.join(this.getRuntimePath(), runtimeSubdir);
  }

  static getRoot() {
    return this.rootPath;
  }

  static getRuntimePath() {
    const runtimeRoot = this.getRuntimeRoot();
    const p = runtimeRoot || path.join(this.rootPath, 'runtime');
    fs.ensureDirSync(p);
    return p;
  }

  static getModelsPath() {
    const p = this.resolveConfiguredDir('APP_MODELS_DIR', 'models');
    fs.ensureDirSync(p);
    return p;
  }

  static getTransformersCachePath() {
    const p = path.join(this.getModelsPath(), 'transformersjs-cache');
    fs.ensureDirSync(p);
    return p;
  }

  static getToolsPath() {
    const p = this.resolveConfiguredDir('APP_TOOLS_DIR', 'tools');
    fs.ensureDirSync(p);
    return p;
  }

  static getProjectsPath() {
    const p = this.resolveConfiguredDir('APP_PROJECTS_DIR', 'projects');
    fs.ensureDirSync(p);
    return p;
  }

  static getLogsPath() {
    const p = this.resolveConfiguredDir('APP_LOGS_DIR', 'logs');
    fs.ensureDirSync(p);
    return p;
  }

  static getTmpPath() {
    const p = this.resolveConfiguredDir('APP_TMP_DIR', 'tmp');
    fs.ensureDirSync(p);
    return p;
  }

  static getRuntimeTempEnv() {
    const runtimePath = this.getRuntimePath();
    const tmpPath = this.getTmpPath();
    const huggingFaceCachePath = path.join(tmpPath, 'huggingface');
    const pipCachePath = path.join(tmpPath, 'pip-cache');
    const torchCachePath = path.join(tmpPath, 'torch-cache');
    fs.ensureDirSync(huggingFaceCachePath);
    fs.ensureDirSync(pipCachePath);
    fs.ensureDirSync(torchCachePath);
    return {
      APP_RUNTIME_DIR: runtimePath,
      APP_TMP_DIR: tmpPath,
      ARCSUB_RUNTIME_DIR: runtimePath,
      ARCSUB_RUNTIME_TMP_DIR: tmpPath,
      HF_HOME: huggingFaceCachePath,
      HF_HUB_CACHE: path.join(huggingFaceCachePath, 'hub'),
      HF_XET_CACHE: path.join(huggingFaceCachePath, 'xet'),
      HF_DATASETS_CACHE: path.join(huggingFaceCachePath, 'datasets'),
      PIP_CACHE_DIR: pipCachePath,
      TORCH_HOME: torchCachePath,
      TMPDIR: tmpPath,
      TEMP: tmpPath,
      TMP: tmpPath,
    };
  }

  static getLocalPath() {
    const p = this.resolveConfiguredDir('APP_LOCAL_DIR', 'local');
    fs.ensureDirSync(p);
    return p;
  }

  static getPrivateVideoSitesPath() {
    const p = path.join(this.getLocalPath(), 'video_sites');
    fs.ensureDirSync(p);
    return p;
  }

  static getDbPath() {
    const explicit = this.getEnv('APP_DB_PATH');
    const p = explicit
      ? this.resolveFromRoot(explicit)
      : path.join(this.getRuntimePath(), 'db', 'db.json');
    fs.ensureDirSync(path.dirname(p));
    return p;
  }

  static getBuiltInGlossaryRootPath() {
    return path.resolve(this.rootPath, 'server', 'glossaries', 'translation');
  }

  static getPublicPath() {
    return path.join(this.rootPath, 'public');
  }

  static getDistPath() {
    return path.join(this.rootPath, 'dist');
  }

  static getRelativeToRoot(targetPath: string) {
    return path.relative(this.rootPath, targetPath);
  }

  static toClientPath(targetPath: string) {
    const absolutePath = path.resolve(targetPath);
    const projectsRoot = this.getProjectsPath();
    const projectRelative = path.relative(projectsRoot, absolutePath);
    if (!projectRelative.startsWith('..') && !path.isAbsolute(projectRelative)) {
      return `/Projects/${projectRelative.replace(/\\/g, '/')}`;
    }

    const rootRelative = path.relative(this.rootPath, absolutePath);
    if (!rootRelative.startsWith('..') && !path.isAbsolute(rootRelative)) {
      return `/${rootRelative.replace(/\\/g, '/')}`;
    }

    return absolutePath;
  }

  static resolveRootPath(...parts: string[]) {
    return path.join(this.rootPath, ...parts);
  }

  static resolveToolsSourcePath(...parts: string[]) {
    return this.resolveRootPath('tools_src', ...parts);
  }

  static resolveNodeModulesPath(...parts: string[]) {
    return this.resolveRootPath('node_modules', ...parts);
  }

  static resolveDotEnvPath() {
    return this.resolveRootPath('.env');
  }

  static getAsrLogPath() {
    return path.join(this.getLogsPath(), 'asr.log');
  }

  static getSettingsProjectsPathValue() {
    const projectsPath = this.getProjectsPath();
    const relative = path.relative(this.rootPath, projectsPath);
    return relative && !relative.startsWith('..') && !path.isAbsolute(relative)
      ? relative.replace(/\\/g, '/')
      : projectsPath;
  }

  static getOpenvinoLocalTranslateCachePath() {
    const p = path.join(this.getModelsPath(), 'openvino-cache', 'translate');
    fs.ensureDirSync(p);
    return p;
  }

  static getOpenvinoLocalAsrCachePath() {
    const p = path.join(this.getModelsPath(), 'openvino-cache', 'asr');
    fs.ensureDirSync(p);
    return p;
  }

  static getOpenvinoAlignmentModelsPath() {
    const p = path.join(this.getModelsPath(), 'openvino-alignment');
    fs.ensureDirSync(p);
    return p;
  }

  static describeResolvedPaths() {
    return {
      root: this.getRoot(),
      runtime: this.getRuntimePath(),
      db: this.getDbPath(),
      projects: this.getProjectsPath(),
      models: this.getModelsPath(),
      openvinoAlignmentModels: this.getOpenvinoAlignmentModelsPath(),
      tools: this.getToolsPath(),
      logs: this.getLogsPath(),
      tmp: this.getTmpPath(),
      local: this.getLocalPath(),
      privateVideoSites: this.getPrivateVideoSitesPath(),
      glossaryRoot: this.getBuiltInGlossaryRootPath(),
    };
  }

  static assertValidProjectId(projectId: string) {
    if (!this.PROJECT_ID_RE.test(projectId)) {
      throw new Error('Invalid project id');
    }
  }

  private static isSubPath(parent: string, child: string) {
    const rel = path.relative(parent, child);
    return rel !== '' && !rel.startsWith('..') && !path.isAbsolute(rel);
  }

  static getProjectPath(projectId: string, options: { create?: boolean } = {}) {
    this.assertValidProjectId(projectId);
    const create = options.create ?? true;
    const projectsRoot = this.getProjectsPath();
    const projectPath = path.resolve(projectsRoot, projectId);

    if (!this.isSubPath(projectsRoot, projectPath)) {
      throw new Error('Project path escapes workspace');
    }

    if (create) {
      fs.ensureDirSync(projectPath);
      for (const subDir of this.PROJECT_SUB_DIRS) {
        fs.ensureDirSync(path.join(projectPath, subDir));
      }
    }

    return projectPath;
  }

  static resolveProjectFile(
    projectId: string,
    subDir: (typeof this.PROJECT_SUB_DIRS)[number],
    fileName: string,
    options: { createProject?: boolean } = {}
  ) {
    if (!fileName || fileName.includes('\0')) {
      throw new Error('Invalid file name');
    }

    const normalizedName = path.basename(fileName);
    if (normalizedName !== fileName || normalizedName === '.' || normalizedName === '..') {
      throw new Error('Invalid file path');
    }

    const projectPath = this.getProjectPath(projectId, { create: options.createProject ?? false });
    const targetDir = path.resolve(projectPath, subDir);

    if (!this.isSubPath(projectPath, targetDir)) {
      throw new Error('Invalid sub directory');
    }

    const fullPath = path.resolve(targetDir, normalizedName);
    if (!this.isSubPath(targetDir, fullPath)) {
      throw new Error('File path escapes project directory');
    }

    return fullPath;
  }

  static async ensureBaseDirs() {
    await fs.ensureDir(this.getModelsPath());
    await fs.ensureDir(this.getToolsPath());
    await fs.ensureDir(this.getProjectsPath());
    await fs.ensureDir(this.getLogsPath());
    await fs.ensureDir(this.getTmpPath());
    await fs.ensureDir(this.getLocalPath());
    await fs.ensureDir(path.dirname(this.getDbPath()));
  }
}
