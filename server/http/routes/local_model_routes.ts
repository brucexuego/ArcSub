import express from 'express';
import { createHash } from 'crypto';
import { PyannoteSetupService } from '../../services/pyannote_setup_service.js';
import { RuntimeReadinessService } from '../../services/runtime_readiness_service.js';

let localModelServiceTask: Promise<typeof import('../../services/local_model_service.js').LocalModelService> | null = null;
let systemMonitorTask: Promise<typeof import('../../services/system_monitor.js').SystemMonitor> | null = null;

function getLocalModelService() {
  if (!localModelServiceTask) {
    localModelServiceTask = import('../../services/local_model_service.js').then((module) => module.LocalModelService);
  }
  return localModelServiceTask;
}

function getSystemMonitor() {
  if (!systemMonitorTask) {
    systemMonitorTask = import('../../services/system_monitor.js').then((module) => module.SystemMonitor);
  }
  return systemMonitorTask;
}

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

export function registerSystemAndLocalModelRoutes(app: express.Express) {
  app.get('/api/system/resources', async (req, res) => {
    try {
      const SystemMonitor = await getSystemMonitor();
      const fastMode = String(req.query?.fast || '').trim() === '1';
      const snapshot = await SystemMonitor.getSnapshot(false, fastMode);
      return sendJsonWithEtag(req, res, snapshot);
    } catch (error: any) {
      return res.status(500).json({ error: error?.message || 'Failed to fetch system resources' });
    }
  });

  app.get('/api/openvino/status', async (req, res) => {
    try {
      const LocalModelService = await getLocalModelService();
      const status = await LocalModelService.getOpenvinoStatus();
      res.json(status);
    } catch (error: any) {
      res.status(500).json({ error: error?.message || 'Failed to check OpenVINO status.' });
    }
  });

  app.get('/api/runtime/readiness', async (req, res) => {
    try {
      const snapshot = await RuntimeReadinessService.getSnapshot();
      return sendJsonWithEtag(req, res, snapshot);
    } catch (error: any) {
      return res.status(500).json({ error: error?.message || 'Failed to inspect runtime readiness.' });
    }
  });

  app.get('/api/runtime/pyannote/status', async (req, res) => {
    try {
      const status = await PyannoteSetupService.getStatus();
      return sendJsonWithEtag(req, res, status);
    } catch (error: any) {
      return res.status(500).json({ error: error?.message || 'Failed to inspect pyannote status.' });
    }
  });

  app.post('/api/runtime/pyannote/install', async (req, res) => {
    try {
      const token = typeof req.body?.token === 'string' ? req.body.token : '';
      const status = await PyannoteSetupService.ensureInstalled({ token });
      return res.json({ success: true, status });
    } catch (error: any) {
      const status = await PyannoteSetupService.getStatus().catch(() => null);
      return res.status(400).json({
        error: error?.message || 'Failed to install pyannote assets.',
        status,
      });
    }
  });

  app.post('/api/runtime/pyannote/token', async (req, res) => {
    try {
      const token = typeof req.body?.token === 'string' ? req.body.token : '';
      const status = await PyannoteSetupService.configureToken(token);
      return res.json({ success: true, status });
    } catch (error: any) {
      return res.status(400).json({
        error: error?.message || 'Failed to save HF token.',
      });
    }
  });

  app.get('/api/local-models', async (req, res) => {
    try {
      const LocalModelService = await getLocalModelService();
      const localModels = await LocalModelService.getLocalModelsOverview();
      return sendJsonWithEtag(req, res, localModels);
    } catch (error: any) {
      return res.status(500).json({ error: error?.message || 'Failed to fetch local models.' });
    }
  });

  app.post('/api/local-models/select', async (req, res) => {
    try {
      const LocalModelService = await getLocalModelService();
      const rawType = typeof req.body?.type === 'string' ? req.body.type.trim() : '';
      if (rawType !== 'asr' && rawType !== 'translate') {
        return res.status(400).json({ error: 'type must be asr or translate.' });
      }
      const type = rawType;
      const modelId = typeof req.body?.modelId === 'string' ? req.body.modelId.trim() : '';
      if (!modelId) {
        return res.status(400).json({ error: 'modelId is required.' });
      }

      const data = await LocalModelService.selectModel(type, modelId);
      return res.json(data);
    } catch (error: any) {
      return res.status(400).json({ error: error?.message || 'Failed to update local model selection.' });
    }
  });

  app.post('/api/local-models/install', async (req, res) => {
    try {
      const LocalModelService = await getLocalModelService();
      const rawType = typeof req.body?.type === 'string' ? req.body.type.trim() : '';
      if (rawType !== 'asr' && rawType !== 'translate') {
        return res.status(400).json({ error: 'type must be asr or translate.' });
      }
      const repoId = typeof req.body?.repoId === 'string' ? req.body.repoId.trim() : '';
      if (!repoId) {
        return res.status(400).json({ error: 'repoId is required.' });
      }

      const openvinoStatus = await LocalModelService.getOpenvinoStatus();
      const asrInstallReady = Boolean(openvinoStatus?.asr?.ready);
      const translateInstallReady = Boolean(
        openvinoStatus?.node?.available &&
          openvinoStatus?.genai?.available &&
          (openvinoStatus?.genai?.llmPipelineAvailable || openvinoStatus?.genai?.vlmPipelineAvailable)
      );
      if (rawType === 'asr' && !asrInstallReady) {
        return res.status(400).json({
          error: 'ASR runtime is unavailable. Please make sure OpenVINO local runtime is ready first.',
          capability: {
            type: 'asr',
            local_model_install_available: false,
            openvino_detected: Boolean(openvinoStatus?.node?.available || openvinoStatus?.genai?.available),
          },
        });
      }
      if (rawType === 'translate' && !translateInstallReady) {
        return res.status(400).json({
          error: 'Translation runtime is unavailable. Please make sure OpenVINO GenAI runtime is ready first.',
          capability: {
            type: 'translate',
            local_model_install_available: false,
            openvino_detected: Boolean(openvinoStatus?.node?.available || openvinoStatus?.genai?.available),
          },
        });
      }

      const data = await LocalModelService.installModel(rawType, repoId);
      return res.json(data);
    } catch (error: any) {
      return res.status(400).json({ error: error?.message || 'Failed to install local model.' });
    }
  });

  app.post('/api/local-models/remove', async (req, res) => {
    try {
      const LocalModelService = await getLocalModelService();
      const modelId = typeof req.body?.modelId === 'string' ? req.body.modelId.trim() : '';
      if (!modelId) {
        return res.status(400).json({ error: 'modelId is required.' });
      }
      const data = await LocalModelService.removeModel(modelId);
      return res.json(data);
    } catch (error: any) {
      return res.status(400).json({ error: error?.message || 'Failed to remove local model.' });
    }
  });

  app.get('/api/runtime-models', async (req, res) => {
    try {
      const LocalModelService = await getLocalModelService();
      const runtimeModels = await LocalModelService.getRuntimeModels();
      res.json(runtimeModels);
    } catch (error: any) {
      res.status(500).json({ error: error?.message || 'Failed to fetch runtime models.' });
    }
  });

  app.post('/api/local-models/release', async (req, res) => {
    try {
      const LocalModelService = await getLocalModelService();
      const target = req.body?.target === 'asr' || req.body?.target === 'translate' ? req.body.target : 'all';
      console.log(
        `[${new Date().toISOString()}] [API local-models/release] request ${JSON.stringify({
          target,
          ip: req.ip,
        })}`
      );
      const result = await LocalModelService.releaseLocalRuntimes(target);
      if (result.success) {
        console.log(
          `[${new Date().toISOString()}] [API local-models/release] success ${JSON.stringify({
            target,
            released: result.released,
          })}`
        );
        return res.json(result);
      }
      console.warn(
        `[${new Date().toISOString()}] [API local-models/release] failed ${JSON.stringify({
          target,
          released: result.released,
          errors: result.errors,
        })}`
      );
      return res.status(500).json(result);
    } catch (error: any) {
      console.error(
        `[${new Date().toISOString()}] [API local-models/release] exception ${JSON.stringify({
          error: String(error?.message || error || 'Failed to release local runtimes.'),
        })}`
      );
      return res.status(500).json({ error: error?.message || 'Failed to release local runtimes.' });
    }
  });

  app.post('/api/local-models/preload', async (req, res) => {
    try {
      const LocalModelService = await getLocalModelService();
      const rawTarget = typeof req.body?.target === 'string' ? req.body.target.trim() : '';
      if (rawTarget !== 'translate') {
        return res.status(400).json({ error: 'target must be translate.' });
      }
      const modelId = typeof req.body?.modelId === 'string' ? req.body.modelId.trim() : '';
      if (!modelId) {
        return res.status(400).json({ error: 'modelId is required.' });
      }

      const result = await LocalModelService.preloadLocalRuntime('translate', modelId);
      return res.json(result);
    } catch (error: any) {
      return res.status(400).json({ error: error?.message || 'Failed to preload local runtime.' });
    }
  });
}
