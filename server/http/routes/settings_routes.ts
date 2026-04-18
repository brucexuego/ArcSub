import express from 'express';
import { createHash } from 'crypto';
import { SettingsManager } from '../../services/settings_manager.js';

const SETTINGS_CACHE_MS = 1500;
let cachedAt = 0;
let cachedSettings: any | null = null;

function toEtag(value: unknown) {
  const serialized = JSON.stringify(value);
  const hash = createHash('sha1').update(serialized).digest('hex');
  return `W/"${hash}"`;
}

function sendJsonWithEtag(req: express.Request, res: express.Response, payload: unknown) {
  const etag = toEtag(payload);
  res.setHeader('ETag', etag);
  res.setHeader('Cache-Control', 'private, no-cache');
  if (req.headers['if-none-match'] === etag) {
    return res.status(304).end();
  }
  return res.json(payload);
}

async function getCachedSettings() {
  const now = Date.now();
  if (cachedSettings && now - cachedAt < SETTINGS_CACHE_MS) {
    return cachedSettings;
  }
  const settings = await SettingsManager.getSettings();
  cachedSettings = settings;
  cachedAt = now;
  return settings;
}

function buildModelSetupStatus(settings: any) {
  const installedLocalModels = Array.isArray(settings?.localModels?.installed) ? settings.localModels.installed : [];
  const selectedLocalAsrId =
    typeof settings?.localModels?.asrSelectedId === 'string' ? settings.localModels.asrSelectedId : '';
  const selectedLocalTranslateId =
    typeof settings?.localModels?.translateSelectedId === 'string' ? settings.localModels.translateSelectedId : '';

  const cloudAsrReady = Array.isArray(settings?.asrModels) && settings.asrModels.length > 0;
  const cloudTranslateReady = Array.isArray(settings?.translateModels) && settings.translateModels.length > 0;
  const localAsrReady = installedLocalModels.some((model: any) => model?.type === 'asr' && model?.id === selectedLocalAsrId);
  const localTranslateReady = installedLocalModels.some(
    (model: any) => model?.type === 'translate' && model?.id === selectedLocalTranslateId
  );

  const hasAsr = cloudAsrReady || localAsrReady;
  const hasTranslate = cloudTranslateReady || localTranslateReady;

  return {
    hasAsr,
    hasTranslate,
    ready: hasAsr && hasTranslate,
  };
}

export function registerSettingsCrudRoutes(app: express.Express) {
  app.get('/api/settings', async (req, res) => {
    try {
      const settings = await getCachedSettings();
      return sendJsonWithEtag(req, res, settings);
    } catch {
      return res.status(500).json({ error: 'Failed to fetch settings' });
    }
  });

  app.get('/api/settings/model-setup', async (req, res) => {
    try {
      const settings = await getCachedSettings();
      return sendJsonWithEtag(req, res, buildModelSetupStatus(settings));
    } catch {
      return res.status(500).json({ error: 'Failed to fetch model setup status' });
    }
  });

  app.post('/api/settings', async (req, res) => {
    try {
      const settings = await SettingsManager.updateSettings(req.body);
      cachedSettings = settings;
      cachedAt = Date.now();
      res.json(settings);
    } catch (error: any) {
      res.status(400).json({ error: error.message || 'Failed to save settings' });
    }
  });
}
