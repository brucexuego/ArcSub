import express from 'express';
import { resolveCloudAsrProvider } from '../../services/cloud_asr_provider.js';

export interface SettingsTestConnectionRouteDeps {
  parseHttpUrl: (raw: string) => URL | null;
  isAllowedTestUrl: (parsed: URL) => boolean;
  isMaskedKey: (value: string) => boolean;
}

let settingsManagerTask: Promise<typeof import('../../services/settings_manager.js').SettingsManager> | null = null;
let translationServiceTask: Promise<typeof import('../../services/translation_service.js').TranslationService> | null = null;
let asrServiceTask: Promise<typeof import('../../services/asr_service.js').AsrService> | null = null;

function getSettingsManager() {
  if (!settingsManagerTask) {
    settingsManagerTask = import('../../services/settings_manager.js').then((module) => module.SettingsManager);
  }
  return settingsManagerTask;
}

function getTranslationService() {
  if (!translationServiceTask) {
    translationServiceTask = import('../../services/translation_service.js').then((module) => module.TranslationService);
  }
  return translationServiceTask;
}

function getAsrService() {
  if (!asrServiceTask) {
    asrServiceTask = import('../../services/asr_service.js').then((module) => module.AsrService);
  }
  return asrServiceTask;
}

export function registerSettingsTestConnectionRoute(
  app: express.Express,
  deps: SettingsTestConnectionRouteDeps
) {
  const { parseHttpUrl, isAllowedTestUrl, isMaskedKey } = deps;

  app.post('/api/settings/test-connection', async (req, res) => {
    try {
      const type = req.body?.type === 'translate' ? 'translate' : 'asr';
      const modelId = typeof req.body?.modelId === 'string' ? req.body.modelId : '';
      const requestedModel = typeof req.body?.model === 'string' ? req.body.model.trim() : '';
      const requestedName = typeof req.body?.name === 'string' ? req.body.name.trim() : '';
      const requestedUrl = typeof req.body?.url === 'string' ? req.body.url.trim() : '';
      const providedKey = typeof req.body?.key === 'string' ? req.body.key : '';

      let finalUrl = requestedUrl;
      let finalKey = providedKey;
      let finalModel = requestedModel;
      let finalName = requestedName;

      if (isMaskedKey(providedKey)) {
        if (!modelId) {
          return res.status(400).json({ success: false, error: 'Model ID is required for masked key test.' });
        }
        const SettingsManager = await getSettingsManager();
        const settings = await SettingsManager.getSettings({ mask: false });
        const models = type === 'asr' ? settings.asrModels : settings.translateModels;
        const found = models.find((m: any) => m.id === modelId);

        if (!found) {
          return res.status(404).json({ success: false, error: 'Model not found.' });
        }

        finalUrl = String(found.url || '');
        finalKey = String(found.key || '');
        finalModel = String(found.model || requestedModel || '');
        finalName = String(found.name || requestedName || '');
      }

      if (finalKey && /[^\x20-\x7E]/.test(finalKey)) {
        return res.status(400).json({
          success: false,
          error: 'API key appears masked or malformed. Please re-enter the key and test again.',
        });
      }

      const rawParsedUrl = parseHttpUrl(finalUrl);
      if (!rawParsedUrl) {
        return res.status(400).json({ success: false, error: 'Invalid URL. Only http/https are allowed.' });
      }

      const normalizedUrl =
        type === 'asr'
          ? resolveCloudAsrProvider({
              url: finalUrl,
              modelName: finalName,
              model: finalModel,
            }).endpointUrl
          : finalUrl;

      const parsedUrl = parseHttpUrl(normalizedUrl);
      if (!parsedUrl) {
        return res.status(400).json({ success: false, error: 'Invalid URL. Only http/https are allowed.' });
      }
      if (!isAllowedTestUrl(parsedUrl)) {
        return res.status(400).json({
          success: false,
          error: 'Endpoint blocked: only localhost/private LAN IP (or LAN hostname over HTTP) is allowed. Public IP addresses are not allowed.',
        });
      }

      if (type === 'translate') {
        const TranslationService = await getTranslationService();
        const translatedConnection = await TranslationService.testConnection({
          url: finalUrl,
          key: finalKey,
          model: finalModel,
          name: finalName,
        });
        if (translatedConnection.success) {
          return res.json({ success: true, message: translatedConnection.message || 'Connection succeeded.' });
        }
        return res.json({
          success: false,
          error: translatedConnection.error || 'Connection failed.',
        });
      }

      const AsrService = await getAsrService();
      const asrConnection = await AsrService.testConnection({
        url: finalUrl,
        key: finalKey,
        model: finalModel,
        name: finalName,
      });
      if (asrConnection.success) {
        return res.json({ success: true, message: asrConnection.message || 'Connection succeeded.' });
      }
      return res.json({
        success: false,
        error: asrConnection.error || 'Connection failed.',
      });
    } catch (error: any) {
      res.status(500).json({ success: false, error: error.message });
    }
  });
}
