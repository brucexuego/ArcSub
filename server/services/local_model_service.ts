import fs from 'fs-extra';
import https from 'https';
import path from 'path';
import { SettingsManager, AppSettings } from './settings_manager.js';
import {
  buildLocalModelId,
  BUILTIN_LOCAL_MODELS,
  HuggingFaceModelMetadata,
  inferLocalModelDefinitionFromHf,
  inferLocalModelDefinitionFromInstalledDir,
  LocalModelDefinition,
  LocalModelSelection,
  LocalModelType,
  sanitizeLocalModelDefinition,
  getLocalModelInstallDir,
  getDefaultLocalModelSelection,
} from '../local_model_catalog.js';
import { OpenvinoRuntimeManager } from '../openvino_runtime_manager.js';

interface InstallState {
  installing: boolean;
  startedAt: number;
}

interface LocalModelOverviewEntry {
  id: string;
  type: LocalModelType;
  name: string;
  repoId: string;
  sourceFormat: LocalModelDefinition['sourceFormat'];
  conversionMethod: LocalModelDefinition['conversionMethod'];
  runtimeLayout: LocalModelDefinition['runtimeLayout'];
  runtime: LocalModelDefinition['runtime'];
  selected: boolean;
  installError?: string | null;
}

export class LocalModelService {
  private static readonly HF_BASE = 'https://huggingface.co';
  private static readonly OVERVIEW_CACHE_MS = 1500;
  private static installTasks = new Map<string, Promise<void>>();
  private static installStates = new Map<string, InstallState>();
  private static installErrors = new Map<string, string>();
  private static overviewCache: { value: Awaited<ReturnType<typeof LocalModelService['getCatalogView']>>; at: number } | null = null;

  private static invalidateOverviewCache() {
    this.overviewCache = null;
  }

  private static async buildOverview() {
    const settings = await SettingsManager.getSettings({ mask: false });
    return this.getCatalogView(settings);
  }

  private static getHfToken() {
    const token = String(process.env.HF_TOKEN || '').trim();
    return token || null;
  }

  private static shouldAttachHfAuth(rawUrl: string) {
    try {
      const hostname = new URL(rawUrl).hostname.toLowerCase();
      return hostname === 'huggingface.co' || hostname.endsWith('.huggingface.co');
    } catch {
      return false;
    }
  }

  private static buildDownloadHeaders(rawUrl: string) {
    const headers: Record<string, string> = {
      'User-Agent': 'ArcSub-LocalModelInstaller/2.0',
      Accept: 'application/json, text/plain, */*',
    };
    const token = this.getHfToken();
    if (token && this.shouldAttachHfAuth(rawUrl)) {
      headers.Authorization = `Bearer ${token}`;
    }
    return headers;
  }

  private static normalizeRepoIdInput(rawValue: string) {
    const raw = String(rawValue || '').trim();
    if (!raw) {
      throw new Error('Hugging Face model id is required.');
    }

    if (/^https?:\/\//i.test(raw)) {
      let url: URL;
      try {
        url = new URL(raw);
      } catch {
        throw new Error('Invalid Hugging Face model URL.');
      }
      const hostname = url.hostname.toLowerCase();
      if (hostname !== 'huggingface.co' && !hostname.endsWith('.huggingface.co')) {
        throw new Error('Only Hugging Face model URLs are supported.');
      }

      const parts = url.pathname
        .split('/')
        .map((part) => String(part || '').trim())
        .filter(Boolean);

      const filtered = parts[0] === 'models' ? parts.slice(1) : parts;
      if (filtered.length < 2) {
        throw new Error('Invalid Hugging Face model URL.');
      }
      return `${filtered[0]}/${filtered[1]}`;
    }

    const normalized = raw.replace(/^\/+|\/+$/g, '');
    const parts = normalized.split('/').filter(Boolean);
    if (parts.length !== 2) {
      throw new Error('Hugging Face model id must be in the form owner/model-name.');
    }
    return `${parts[0]}/${parts[1]}`;
  }

  private static getHfApiUrl(repoId: string) {
    const [owner, name] = repoId.split('/');
    return `${this.HF_BASE}/api/models/${encodeURIComponent(owner)}/${encodeURIComponent(name)}`;
  }

  private static getHfFileUrl(repoId: string, fileName: string) {
    const encodedPath = fileName
      .split('/')
      .map((segment) => encodeURIComponent(segment))
      .join('/');
    return `${this.HF_BASE}/${repoId}/resolve/main/${encodedPath}`;
  }

  private static async fetchJsonWithRedirect(url: string): Promise<{ status: number; body: any }> {
    return new Promise((resolve, reject) => {
      const request = https.get(url, { headers: this.buildDownloadHeaders(url) }, (response) => {
        const status = response.statusCode || 0;
        if ([301, 302, 303, 307, 308].includes(status)) {
          const redirectUrl = response.headers.location;
          response.resume();
          if (!redirectUrl) {
            reject(new Error(`Request redirect without location: ${url}`));
            return;
          }
          const nextUrl = new URL(redirectUrl, url).toString();
          this.fetchJsonWithRedirect(nextUrl).then(resolve).catch(reject);
          return;
        }

        let raw = '';
        response.setEncoding('utf8');
        response.on('data', (chunk) => {
          raw += chunk;
        });
        response.on('end', () => {
          if (status < 200 || status >= 300) {
            const errorCode = String(response.headers['x-error-code'] || '').trim();
            const errorMessage = String(response.headers['x-error-message'] || '').trim();
            if (
              (status === 401 || status === 403) &&
              (errorCode.toLowerCase() === 'gatedrepo' ||
                /gated|restricted|authenticated|access to model/i.test(errorMessage))
            ) {
              reject(
                new Error(
                  `Unable to access Hugging Face model metadata (${status}). ` +
                    `This repository requires approval/token access. Please accept the model license and set HF_TOKEN in .env.`
                )
              );
              return;
            }
            if (status === 404) {
              reject(new Error(`Hugging Face model not found: ${url}`));
              return;
            }
            reject(new Error(`Failed to fetch Hugging Face model metadata (${status}).`));
            return;
          }

          try {
            resolve({
              status,
              body: raw ? JSON.parse(raw) : {},
            });
          } catch (error: any) {
            reject(new Error(`Failed to parse Hugging Face model metadata: ${String(error?.message || error)}`));
          }
        });
      });

      request.on('error', (error) => {
        reject(error);
      });
      request.setTimeout(30000, () => {
        request.destroy(new Error(`Metadata request timeout for ${url}`));
      });
    });
  }

  private static async fetchHfModelMetadata(repoId: string): Promise<HuggingFaceModelMetadata> {
    const { body } = await this.fetchJsonWithRedirect(this.getHfApiUrl(repoId));
    return body || {};
  }

  private static async downloadWithRedirect(url: string, destination: string): Promise<void> {
    await fs.ensureDir(path.dirname(destination));
    const tempPath = `${destination}.part`;

    await new Promise<void>((resolve, reject) => {
      const request = https.get(url, { headers: this.buildDownloadHeaders(url) }, (response) => {
        const status = response.statusCode || 0;
        if ([301, 302, 303, 307, 308].includes(status)) {
          const redirectUrl = response.headers.location;
          response.resume();
          if (!redirectUrl) {
            reject(new Error(`Download redirect without location: ${url}`));
            return;
          }
          const nextUrl = new URL(redirectUrl, url).toString();
          this.downloadWithRedirect(nextUrl, destination).then(resolve).catch(reject);
          return;
        }

        if (status !== 200) {
          const errorCode = String(response.headers['x-error-code'] || '').trim();
          const errorMessage = String(response.headers['x-error-message'] || '').trim();
          response.resume();
          if (
            (status === 401 || status === 403) &&
            (errorCode.toLowerCase() === 'gatedrepo' ||
              /gated|restricted|authenticated|access to model/i.test(errorMessage))
          ) {
            reject(
              new Error(
                `Failed to download model file (${status}): ${url}. ` +
                  `Hugging Face reports gated access (${errorCode || 'GatedRepo'}). ` +
                  `Please request/accept access on the model page and set HF_TOKEN in .env.`
              )
            );
            return;
          }
          reject(new Error(`Failed to download model file (${status}): ${url}`));
          return;
        }

        const writer = fs.createWriteStream(tempPath);
        response.pipe(writer);
        writer.on('finish', async () => {
          try {
            await fs.move(tempPath, destination, { overwrite: true });
            resolve();
          } catch (error) {
            await fs.remove(tempPath).catch(() => {});
            reject(error);
          }
        });
        writer.on('error', async (error) => {
          await fs.remove(tempPath).catch(() => {});
          reject(error);
        });
      });

      request.on('error', async (error) => {
        await fs.remove(tempPath).catch(() => {});
        reject(error);
      });
      request.setTimeout(120000, () => {
        request.destroy(new Error(`Download timeout for ${url}`));
      });
    });
  }

  private static async ensureModelFiles(model: LocalModelDefinition) {
    const modelDir = getLocalModelInstallDir(model);
    await fs.ensureDir(modelDir);
    const downloadRepoId = String(model.downloadRepoId || model.repoId).trim() || model.repoId;

    for (const fileName of model.requiredFiles) {
      const destination = path.join(modelDir, fileName);
      if (await fs.pathExists(destination)) continue;
      const fileUrl = this.getHfFileUrl(downloadRepoId, fileName);
      await this.downloadWithRedirect(fileUrl, destination);
    }
  }

  private static async ensureConvertedQwen3AsrModel(model: LocalModelDefinition) {
    const modelDir = getLocalModelInstallDir(model);
    const checks = await Promise.all(
      model.requiredFiles.map((fileName) => fs.pathExists(path.join(modelDir, fileName)))
    );
    if (checks.every(Boolean)) {
      return;
    }

    await fs.remove(modelDir);
    await fs.ensureDir(modelDir);
    await OpenvinoRuntimeManager.convertOfficialQwenAsrModel({
      repoId: model.repoId,
      outputDir: modelDir,
    });
  }

  private static async ensureAutoConvertedModel(model: LocalModelDefinition) {
    const modelDir = getLocalModelInstallDir(model);
    const probed = inferLocalModelDefinitionFromInstalledDir(model.type, model.repoId, modelDir, {
      id: model.id,
      displayName: model.displayName,
      localSubdir: model.localSubdir,
      downloadRepoId: model.downloadRepoId,
      installMode: model.installMode,
      sourceFormat: model.sourceFormat,
      conversionMethod: model.conversionMethod,
      source: model.source,
    });
    if (probed && (await this.isModelInstalled(probed))) {
      return;
    }

    await fs.remove(modelDir);
    await fs.ensureDir(modelDir);
    await OpenvinoRuntimeManager.convertHuggingFaceModel({
      repoId: String(model.downloadRepoId || model.repoId).trim() || model.repoId,
      outputDir: modelDir,
      type: model.type,
      sourceFormat: model.sourceFormat,
      conversionMethod: model.conversionMethod,
      runtimeLayout: model.runtimeLayout,
    });
  }

  private static async ensureModelInstalledArtifacts(model: LocalModelDefinition) {
    if (model.installMode === 'hf-qwen3-asr-convert') {
      await this.ensureConvertedQwen3AsrModel(model);
      return;
    }
    if (model.installMode === 'hf-auto-convert') {
      await this.ensureAutoConvertedModel(model);
      return;
    }
    await this.ensureModelFiles(model);
  }

  static async isModelInstalled(model: LocalModelDefinition) {
    const modelDir = getLocalModelInstallDir(model);
    const checks = await Promise.all(
      model.requiredFiles.map((fileName) => fs.pathExists(path.join(modelDir, fileName)))
    );
    return checks.every(Boolean);
  }

  private static dedupeModels(models: LocalModelDefinition[]) {
    const byId = new Map<string, LocalModelDefinition>();
    for (const candidate of models) {
      const normalized = sanitizeLocalModelDefinition(candidate);
      if (!normalized) continue;
      const existing = byId.get(normalized.id);
      if (!existing || (existing.source !== 'builtin' && normalized.source === 'builtin')) {
        byId.set(normalized.id, normalized);
      }
    }
    return Array.from(byId.values());
  }

  private static async getRegisteredModels(settings: AppSettings | null | undefined) {
    const configured = Array.isArray(settings?.localModels?.installed)
      ? settings!.localModels.installed
          .map((item) => sanitizeLocalModelDefinition(item))
          .filter(Boolean) as LocalModelDefinition[]
      : [];

    const merged: LocalModelDefinition[] = [...configured];
    const builtinInstallStates = await Promise.all(
      BUILTIN_LOCAL_MODELS.map(async (builtin) => ({
        builtin,
        installed: await this.isModelInstalled(builtin),
      }))
    );
    for (const { builtin, installed } of builtinInstallStates) {
      if (merged.some((item) => item.id === builtin.id)) continue;
      if (!installed) continue;
      merged.push({ ...builtin, requiredFiles: [...builtin.requiredFiles] });
    }

    return this.dedupeModels(merged);
  }

  private static async filterInstalledModels(models: LocalModelDefinition[]) {
    const checks = await Promise.all(
      models.map(async (model) => ({
        model,
        installed: await this.isModelInstalled(model),
      }))
    );
    return checks.filter((item) => item.installed).map((item) => item.model);
  }

  private static normalizeSelection(
    settings: AppSettings | null | undefined,
    availableModels: LocalModelDefinition[]
  ) {
    const defaults = getDefaultLocalModelSelection();
    const current = settings?.localModels || defaults;
    const asrModels = availableModels.filter((model) => model.type === 'asr');
    const translateModels = availableModels.filter((model) => model.type === 'translate');

    return {
      asrSelectedId:
        typeof current.asrSelectedId === 'string' && asrModels.some((model) => model.id === current.asrSelectedId)
          ? current.asrSelectedId
          : asrModels[0]?.id || defaults.asrSelectedId,
      translateSelectedId:
        typeof current.translateSelectedId === 'string' &&
        translateModels.some((model) => model.id === current.translateSelectedId)
          ? current.translateSelectedId
          : translateModels[0]?.id || defaults.translateSelectedId,
    };
  }

  private static buildLocalModelState(
    settings: AppSettings | null | undefined,
    installedModels: LocalModelDefinition[]
  ): LocalModelSelection {
    const selection = this.normalizeSelection(settings, installedModels);
    return {
      ...selection,
      installed: installedModels.map((model) => ({
        ...model,
        requiredFiles: [...model.requiredFiles],
      })),
    };
  }

  private static sortOverviewEntries(
    items: LocalModelOverviewEntry[],
    selection: { asrSelectedId: string; translateSelectedId: string }
  ) {
    const score = (item: LocalModelOverviewEntry) => {
      if (item.type === 'asr') return item.id === selection.asrSelectedId ? 0 : 1;
      return item.id === selection.translateSelectedId ? 0 : 1;
    };

    return [...items].sort((a, b) => {
      const scoreDelta = score(a) - score(b);
      if (scoreDelta !== 0) return scoreDelta;
      if (a.type !== b.type) return a.type.localeCompare(b.type);
      return a.name.localeCompare(b.name, undefined, { sensitivity: 'base' });
    });
  }

  private static async getCatalogView(settings: AppSettings) {
    const registered = await this.getRegisteredModels(settings);
    const installedModels = await this.filterInstalledModels(registered);

    const selection = this.normalizeSelection(settings, installedModels);
    const entries = installedModels.map((model) => ({
      id: model.id,
      type: model.type,
      name: model.displayName,
      repoId: model.repoId,
      sourceFormat: model.sourceFormat,
      conversionMethod: model.conversionMethod,
      runtimeLayout: model.runtimeLayout,
      runtime: model.runtime,
      selected: model.type === 'asr' ? selection.asrSelectedId === model.id : selection.translateSelectedId === model.id,
      installError: this.installErrors.get(model.id) || null,
    }));

    return {
      catalog: this.sortOverviewEntries(entries, selection),
      selection,
    };
  }

  private static async persistInstalledModels(
    settings: AppSettings,
    installedModels: LocalModelDefinition[],
    selectionOverrides?: Partial<Pick<LocalModelSelection, 'asrSelectedId' | 'translateSelectedId'>>
  ) {
    const localModels = this.buildLocalModelState(settings, this.dedupeModels(installedModels));
    if (selectionOverrides?.asrSelectedId) {
      localModels.asrSelectedId = selectionOverrides.asrSelectedId;
    }
    if (selectionOverrides?.translateSelectedId) {
      localModels.translateSelectedId = selectionOverrides.translateSelectedId;
    }
    await SettingsManager.updateSettings({ localModels });
  }

  private static async resolveRegisteredModel(settings: AppSettings, modelId: string) {
    const registered = await this.getRegisteredModels(settings);
    return registered.find((model) => model.id === modelId) || null;
  }

  private static async resolveInstalledModels(settings: AppSettings) {
    const registered = await this.getRegisteredModels(settings);
    return this.filterInstalledModels(registered);
  }

  static async getLocalModelsOverview(options: { forceFresh?: boolean } = {}) {
    if (!options.forceFresh && this.overviewCache && Date.now() - this.overviewCache.at < this.OVERVIEW_CACHE_MS) {
      return this.overviewCache.value;
    }

    const overview = await this.buildOverview();
    this.overviewCache = { value: overview, at: Date.now() };
    return overview;
  }

  static async selectModel(type: LocalModelType, modelId: string) {
    const settings = await SettingsManager.getSettings({ mask: false });
    const installedModels = await this.resolveInstalledModels(settings);
    const model = installedModels.find((item) => item.id === modelId);
    if (!model || model.type !== type) {
      throw new Error('Invalid local model selection.');
    }

    const currentState = this.buildLocalModelState(settings, installedModels);
    const nextState =
      type === 'asr'
        ? { ...currentState, asrSelectedId: modelId }
        : { ...currentState, translateSelectedId: modelId };

    await SettingsManager.updateSettings({ localModels: nextState });
    this.invalidateOverviewCache();
    return this.getLocalModelsOverview({ forceFresh: true });
  }

  static async installModel(type: LocalModelType, repoIdOrUrl: string) {
    const normalizedRepoId = this.normalizeRepoIdInput(repoIdOrUrl);
    const metadata = await this.fetchHfModelMetadata(normalizedRepoId);
    const canonicalRepoId = String(metadata?.id || normalizedRepoId).trim() || normalizedRepoId;
    const inferredModel = inferLocalModelDefinitionFromHf(type, canonicalRepoId, metadata);
    const modelId = inferredModel.id || buildLocalModelId(type, canonicalRepoId);

    const existingTask = this.installTasks.get(modelId);
    if (existingTask) {
      await existingTask;
      this.invalidateOverviewCache();
      return this.getLocalModelsOverview({ forceFresh: true });
    }

    const settings = await SettingsManager.getSettings({ mask: false });
    const registered = await this.getRegisteredModels(settings);
    const existing = registered.find((model) => model.id === modelId);
    const model: LocalModelDefinition = existing
      ? {
          ...existing,
          ...inferredModel,
          source: existing.source === 'builtin' ? 'builtin' : inferredModel.source,
        }
      : inferredModel;

    const latestTask = this.installTasks.get(modelId);
    if (latestTask) {
      await latestTask;
      this.invalidateOverviewCache();
      return this.getLocalModelsOverview({ forceFresh: true });
    }

    this.installErrors.delete(modelId);
    this.installStates.set(modelId, { installing: true, startedAt: Date.now() });

    const installTask = (async () => {
      try {
        await this.ensureModelInstalledArtifacts(model);
        const finalizedModel =
          inferLocalModelDefinitionFromInstalledDir(model.type, model.repoId, getLocalModelInstallDir(model), {
            id: model.id,
            displayName: model.displayName,
            localSubdir: model.localSubdir,
            downloadRepoId: model.downloadRepoId,
            installMode: model.installMode,
            sourceFormat: model.sourceFormat,
            conversionMethod: model.conversionMethod,
            source: model.source,
          }) || null;
        if (!finalizedModel) {
          throw new Error(
            'Model conversion completed, but the resulting OpenVINO files do not match a supported ArcSub runtime layout.'
          );
        }
        const nextInstalled = this.dedupeModels([...registered.filter((item) => item.id !== model.id), finalizedModel]);
        await this.persistInstalledModels(settings, nextInstalled, {
          asrSelectedId: finalizedModel.type === 'asr' ? finalizedModel.id : undefined,
          translateSelectedId: finalizedModel.type === 'translate' ? finalizedModel.id : undefined,
        });
      } catch (error: any) {
        const message = String(error?.message || error || 'Model installation failed.');
        this.installErrors.set(modelId, message);
        throw new Error(message);
      } finally {
        this.installStates.delete(modelId);
      }
    })();

    this.installTasks.set(modelId, installTask);
    try {
      await installTask;
    } finally {
      this.installTasks.delete(modelId);
    }

    this.invalidateOverviewCache();
    return this.getLocalModelsOverview({ forceFresh: true });
  }

  static async removeModel(modelId: string) {
    const settings = await SettingsManager.getSettings({ mask: false });
    const model = await this.resolveRegisteredModel(settings, modelId);
    if (!model) {
      throw new Error('Local model not found.');
    }

    if (this.installStates.get(modelId)?.installing || this.installTasks.has(modelId)) {
      throw new Error('Model installation is still in progress. Please wait before removing it.');
    }

    try {
      if (model.type === 'asr') {
        await OpenvinoRuntimeManager.releaseAsrRuntime();
      } else {
        await OpenvinoRuntimeManager.releaseTranslateRuntime();
      }
    } catch (error: any) {
      throw new Error(
        `Failed to release local ${model.type} runtime before remove: ${String(
          error?.message || error || 'unknown error'
        )}`
      );
    }

    const modelDir = getLocalModelInstallDir(model);
    await fs.remove(modelDir);
    this.installErrors.delete(modelId);

    const registered = await this.getRegisteredModels(settings);
    const nextInstalled = registered.filter((item) => item.id !== modelId);
    await this.persistInstalledModels(settings, nextInstalled);

    this.invalidateOverviewCache();
    return this.getLocalModelsOverview({ forceFresh: true });
  }

  static async getRuntimeModels() {
    const settings = await SettingsManager.getSettings();
    const installedLocalModels = await this.resolveInstalledModels(settings as any);
    const selection = this.normalizeSelection(settings as any, installedLocalModels);

    const asrModels = Array.isArray(settings.asrModels) ? [...settings.asrModels] : [];
    const translateModels = Array.isArray(settings.translateModels) ? [...settings.translateModels] : [];

    const localAsrModels = installedLocalModels
      .filter((model) => model.type === 'asr')
      .sort((a, b) => {
        const aSelected = a.id === selection.asrSelectedId ? 0 : 1;
        const bSelected = b.id === selection.asrSelectedId ? 0 : 1;
        if (aSelected !== bSelected) return aSelected - bSelected;
        return a.displayName.localeCompare(b.displayName, undefined, { sensitivity: 'base' });
      });

    for (const local of localAsrModels) {
      if (asrModels.some((model: any) => model?.id === local.id)) continue;
      asrModels.push({
        id: local.id,
        name: `${local.displayName} (OpenVINO Local)`,
        url: 'local://openvino/asr',
        key: '',
        model: local.repoId,
        isLocal: true,
        provider: 'local-openvino',
      });
    }

    const localTranslateModels = installedLocalModels
      .filter((model) => model.type === 'translate')
      .sort((a, b) => {
        const aSelected = a.id === selection.translateSelectedId ? 0 : 1;
        const bSelected = b.id === selection.translateSelectedId ? 0 : 1;
        if (aSelected !== bSelected) return aSelected - bSelected;
        return a.displayName.localeCompare(b.displayName, undefined, { sensitivity: 'base' });
      });

    for (const local of localTranslateModels) {
      if (translateModels.some((model: any) => model?.id === local.id)) continue;
      translateModels.push({
        id: local.id,
        name: `${local.displayName} (OpenVINO Local)`,
        url: 'local://openvino/translate',
        key: '',
        model: local.repoId,
        isLocal: true,
        provider: 'local-openvino',
      });
    }

    return {
      asrModels,
      translateModels,
      localSelection: selection,
    };
  }

  static async getOpenvinoStatus() {
    return OpenvinoRuntimeManager.getOpenvinoRuntimeStatus();
  }

  static async resolveLocalModelForRequest(
    type: LocalModelType,
    modelId: string | undefined,
    settings?: AppSettings
  ) {
    if (!modelId) return null;

    const resolvedSettings = settings || (await SettingsManager.getSettings({ mask: false }));
    const localModel = await this.resolveRegisteredModel(resolvedSettings, modelId);
    if (!localModel) {
      if (String(modelId).startsWith('local_')) {
        throw new Error('Invalid local model for this request.');
      }
      return null;
    }

    if (localModel.type !== type) {
      throw new Error('Invalid local model for this request.');
    }

    if (!(await this.isModelInstalled(localModel))) {
      throw new Error(
        type === 'asr'
          ? 'Local ASR model is not installed. Please install it in Settings first.'
          : 'Local translation model is not installed. Please install it in Settings first.'
      );
    }

    return localModel;
  }

  static async releaseLocalRuntimes(target: 'asr' | 'translate' | 'all') {
    console.log(
      `[${new Date().toISOString()}] [LocalModelService] releaseLocalRuntimes start ${JSON.stringify({ target })}`
    );
    const errors: string[] = [];
    if (target === 'asr' || target === 'all') {
      try {
        await OpenvinoRuntimeManager.releaseAsrRuntime();
      } catch (error: any) {
        errors.push(String(error?.message || error || 'Failed to release local ASR runtime.'));
      }
    }
    if (target === 'translate' || target === 'all') {
      try {
        await OpenvinoRuntimeManager.releaseTranslateRuntime();
      } catch (error: any) {
        errors.push(String(error?.message || error || 'Failed to release local translation runtime.'));
      }
    }

    const result = {
      success: errors.length === 0,
      released: {
        asr: target === 'asr' || target === 'all',
        translate: target === 'translate' || target === 'all',
      },
      errors,
    };

    console.log(
      `[${new Date().toISOString()}] [LocalModelService] releaseLocalRuntimes done ${JSON.stringify({
        target,
        success: result.success,
        released: result.released,
        errorCount: result.errors.length,
      })}`
    );

    return result;
  }

  static async preloadLocalRuntime(target: 'asr' | 'translate', modelId: string) {
    const settings = await SettingsManager.getSettings({ mask: false });
    const localModel = await this.resolveLocalModelForRequest(target, modelId, settings);
    if (!localModel) {
      throw new Error('Only installed local models can be preloaded.');
    }

    if (target === 'asr') {
      throw new Error('ASR preload is not implemented for this route.');
    }

    const modelPath = getLocalModelInstallDir(localModel);
    const result = await OpenvinoRuntimeManager.preloadTranslateRuntime({
      modelId: localModel.id,
      modelPath,
    });

    return {
      success: true,
      target,
      modelId: localModel.id,
      runtimeDebug: result.runtimeDebug,
    };
  }

  static listModelsByType(type: LocalModelType) {
    return BUILTIN_LOCAL_MODELS.filter((model) => model.type === type);
  }
}
