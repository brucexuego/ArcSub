import fs from 'fs-extra';
import path from 'path';
import { EnvFileService } from './env_file_service.js';
import { PathManager } from '../path_manager.js';

export interface PyannoteSetupStatus {
  tokenConfigured: boolean;
  ready: boolean;
  state: 'ready' | 'partial' | 'missing';
  installing: boolean;
  lastError: string | null;
  paths: {
    segmentation: string;
    embedding: string;
    pldaConfig: string;
  };
}

export class PyannoteSetupService {
  private static installTask: Promise<PyannoteSetupStatus> | null = null;
  private static lastError: string | null = null;

  private static getAssetPaths() {
    return {
      segmentation: path.join(PathManager.getModelsPath(), 'pyannote', 'segmentation', 'model.xml'),
      embedding: path.join(PathManager.getModelsPath(), 'pyannote', 'embedding', 'model.xml'),
      pldaConfig: path.join(PathManager.getModelsPath(), 'pyannote', 'plda', 'vbx.json'),
    };
  }

  static async hasTokenConfigured() {
    const runtime = String(process.env.HF_TOKEN || '').trim();
    if (runtime) return true;
    return Boolean(await EnvFileService.getValue('HF_TOKEN'));
  }

  static async getStatus(): Promise<PyannoteSetupStatus> {
    const paths = this.getAssetPaths();
    const [segmentationExists, embeddingExists, pldaExists, tokenConfigured] = await Promise.all([
      fs.pathExists(paths.segmentation),
      fs.pathExists(paths.embedding),
      fs.pathExists(paths.pldaConfig),
      this.hasTokenConfigured(),
    ]);

    const ready = segmentationExists && embeddingExists && pldaExists;
    const partial = !ready && (segmentationExists || embeddingExists || pldaExists);
    return {
      tokenConfigured,
      ready,
      state: ready ? 'ready' : partial ? 'partial' : 'missing',
      installing: Boolean(this.installTask),
      lastError: this.lastError,
      paths,
    };
  }

  static async configureToken(token: string) {
    const normalized = String(token || '').trim();
    if (!normalized) {
      throw new Error('HF token is required.');
    }
    await EnvFileService.setValue('HF_TOKEN', normalized);
    this.lastError = null;
    return this.getStatus();
  }

  static async clearError() {
    this.lastError = null;
    return this.getStatus();
  }

  static async ensureInstalled(input?: { token?: string }) {
    const token = String(input?.token || '').trim();
    if (token) {
      await this.configureToken(token);
    } else if (!(await this.hasTokenConfigured())) {
      throw new Error('HF token is required before installing pyannote assets.');
    }

    if (this.installTask) {
      return this.installTask;
    }

    this.lastError = null;
    this.installTask = (async () => {
      try {
        const { PyannoteDiarizationService } = await import('../pyannote_diarization_service.js');
        await PyannoteDiarizationService.ensureDeploymentAssets();
        this.lastError = null;
        return await this.getStatus();
      } catch (error: any) {
        this.lastError = String(error?.message || error || 'Pyannote asset installation failed.');
        throw new Error(this.lastError);
      } finally {
        this.installTask = null;
      }
    })();

    return this.installTask;
  }
}
