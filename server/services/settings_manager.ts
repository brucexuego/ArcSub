import { getDb } from '../db.js';
import { ApiConfig } from '../../src/types.js';
import { encrypt, decrypt } from '../crypto_utils.js';
import {
  getDefaultLocalModelSelection,
  LocalModelSelection,
  sanitizeLocalModelDefinition,
} from '../local_model_catalog.js';

export interface AppSettings {
  asrModels: ApiConfig[];
  translateModels: ApiConfig[];
  localModels: LocalModelSelection;
  interfaceLanguage: string;
  projectsPath: string;
}

export class SettingsManager {
  private static readonly WRITE_DEBOUNCE_MS = 120;
  private static pendingSettingsUpdates: Partial<AppSettings> | null = null;
  private static pendingResolvers: Array<{
    resolve: (value: AppSettings) => void;
    reject: (error: unknown) => void;
  }> = [];
  private static flushTimer: NodeJS.Timeout | null = null;
  private static writeQueue: Promise<void> = Promise.resolve();

  private static normalizeLocalModels(input: any): LocalModelSelection {
    const defaults = getDefaultLocalModelSelection();
    const current = input || {};
    const installed = Array.isArray(current.installed)
      ? current.installed
          .map((item: any) => sanitizeLocalModelDefinition(item))
          .filter(Boolean)
      : defaults.installed;
    const getValidSelection = (type: 'asr' | 'translate', selectedId: unknown) => {
      const normalizedId = String(selectedId || '').trim();
      if (!normalizedId) return '';
      return installed.some((item: any) => item?.type === type && item?.id === normalizedId) ? normalizedId : '';
    };
    const firstAsr = installed.find((item: any) => item?.type === 'asr');
    const firstTranslate = installed.find((item: any) => item?.type === 'translate');
    const asrSelectedId = getValidSelection('asr', current.asrSelectedId) || firstAsr?.id || defaults.asrSelectedId;
    const translateSelectedId =
      getValidSelection('translate', current.translateSelectedId) || firstTranslate?.id || defaults.translateSelectedId;
    const moveSelectedToTypeFront = (models: NonNullable<typeof installed>, type: 'asr' | 'translate', selectedId: string) => {
      if (!selectedId) return models;
      const index = models.findIndex((item: any) => item?.type === type && item?.id === selectedId);
      if (index <= 0) return models;
      const next = [...models];
      const [selected] = next.splice(index, 1);
      const insertAt = next.findIndex((item: any) => item?.type === type);
      next.splice(insertAt >= 0 ? insertAt : 0, 0, selected);
      return next;
    };
    const orderedInstalled = moveSelectedToTypeFront(
      moveSelectedToTypeFront(installed, 'asr', asrSelectedId),
      'translate',
      translateSelectedId
    );

    return {
      asrSelectedId,
      translateSelectedId,
      installed: orderedInstalled,
    };
  }

  private static isMaskedKeyValue(value: string): boolean {
    const v = String(value || '').trim();
    if (!v) return false;

    if (v.includes('****') || v.includes('•')) return true;
    if (/^[A-Za-z0-9._-]{2,12}[*•?]{3,}[A-Za-z0-9._-]{0,12}$/.test(v)) return true;
    if (/[^\x20-\x7E]/.test(v) && v.length <= 64) return true;
    return false;
  }

  private static maskKey(key: string): string {
    if (!key) return '';
    if (key.length <= 8) return '****';
    return `${key.slice(0, 4)}****${key.slice(-4)}`;
  }

  private static mergeQueuedUpdates(
    base: Partial<AppSettings> | null,
    next: Partial<AppSettings>
  ): Partial<AppSettings> {
    const merged: Partial<AppSettings> = { ...(base || {}), ...next };
    if (base?.localModels || next.localModels) {
      merged.localModels = {
        ...(base?.localModels || {}),
        ...(next.localModels || {}),
      } as LocalModelSelection;
    }
    return merged;
  }

  private static scheduleFlush(delayMs = this.WRITE_DEBOUNCE_MS) {
    if (this.flushTimer) return;
    this.flushTimer = setTimeout(() => {
      this.flushTimer = null;
      this.writeQueue = this.writeQueue
        .catch(() => {})
        .then(() => this.flushQueuedSettingsUpdates());
    }, delayMs);
  }

  private static async flushQueuedSettingsUpdates() {
    const queuedUpdates = this.pendingSettingsUpdates;
    const queuedResolvers = this.pendingResolvers.splice(0);
    this.pendingSettingsUpdates = null;

    if (!queuedUpdates) {
      for (const { resolve } of queuedResolvers) {
        resolve(await this.getSettings());
      }
      return;
    }

    try {
      const saved = await this.applySettingsUpdate(queuedUpdates);
      for (const { resolve } of queuedResolvers) {
        resolve(saved);
      }
    } catch (error) {
      for (const { reject } of queuedResolvers) {
        reject(error);
      }
    }

    if (this.pendingSettingsUpdates) {
      this.scheduleFlush(0);
    }
  }

  static async getSettings(options = { mask: true }): Promise<AppSettings> {
    const db = await getDb();
    const settings = JSON.parse(JSON.stringify(db.data.settings || {}));

    const mapModels = (models: ApiConfig[] = [], mask: boolean) =>
      models.map((m: ApiConfig) => {
        const plain = m.key ? decrypt(m.key) : '';
        return { ...m, key: mask ? this.maskKey(plain) : plain };
      });

    settings.asrModels = mapModels(settings.asrModels || [], options.mask);
    settings.translateModels = mapModels(settings.translateModels || [], options.mask);
    settings.localModels = this.normalizeLocalModels(settings.localModels);

    return settings as AppSettings;
  }

  private static async applySettingsUpdate(updates: Partial<AppSettings>): Promise<AppSettings> {
    const db = await getDb();
    const currentSettings = db.data.settings;

    if (updates.asrModels) {
      updates.asrModels = updates.asrModels.map((newModel) => {
        const oldModel = currentSettings.asrModels.find((m: any) => m.id === newModel.id);
        if (newModel.key && this.isMaskedKeyValue(newModel.key)) {
          return { ...newModel, key: oldModel ? oldModel.key : '' };
        }
        return { ...newModel, key: newModel.key ? encrypt(newModel.key) : '' };
      });
    }

    if (updates.translateModels) {
      updates.translateModels = updates.translateModels.map((newModel) => {
        const oldModel = currentSettings.translateModels.find((m: any) => m.id === newModel.id);
        if (newModel.key && this.isMaskedKeyValue(newModel.key)) {
          return { ...newModel, key: oldModel ? oldModel.key : '' };
        }
        return { ...newModel, key: newModel.key ? encrypt(newModel.key) : '' };
      });
    }

    if (updates.localModels) {
      const normalizedCurrent = this.normalizeLocalModels(currentSettings.localModels);
      updates.localModels = this.normalizeLocalModels({
        ...normalizedCurrent,
        ...updates.localModels,
      });
    }

    db.data.settings = {
      ...db.data.settings,
      ...updates,
      localModels: updates.localModels
        ? this.normalizeLocalModels(updates.localModels)
        : this.normalizeLocalModels(db.data.settings.localModels),
    };

    await db.write();
    return this.getSettings();
  }

  static async updateSettings(updates: Partial<AppSettings>): Promise<AppSettings> {
    return new Promise<AppSettings>((resolve, reject) => {
      this.pendingSettingsUpdates = this.mergeQueuedUpdates(this.pendingSettingsUpdates, updates);
      this.pendingResolvers.push({ resolve, reject });
      this.scheduleFlush();
    });
  }
}
