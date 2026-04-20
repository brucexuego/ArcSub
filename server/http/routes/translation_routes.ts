import express from 'express';
import { persistTranslationArtifacts, readProjectSourceAsset } from '../text_utils.js';
import { PROJECT_STATUS } from '../../../src/project_status.js';

export interface TranslationRouteDeps {
  writeAsrLog: (message: string, payload?: unknown, level?: 'log' | 'error') => void;
  isAbortLikeError: (error: unknown) => boolean;
}

let projectManagerTask: Promise<typeof import('../../services/project_manager.js').ProjectManager> | null = null;
let translationServiceTask: Promise<typeof import('../../services/translation_service.js').TranslationService> | null = null;

function getProjectManager() {
  if (!projectManagerTask) {
    projectManagerTask = import('../../services/project_manager.js').then((module) => module.ProjectManager);
  }
  return projectManagerTask;
}

function getTranslationService() {
  if (!translationServiceTask) {
    translationServiceTask = import('../../services/translation_service.js').then((module) => module.TranslationService);
  }
  return translationServiceTask;
}

export function registerTranslationRoutes(app: express.Express, deps: TranslationRouteDeps) {
  const { writeAsrLog, isAbortLikeError } = deps;

  app.get('/api/translate/:projectId', async (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    const { projectId } = req.params;
    const targetLang = typeof req.query.targetLang === 'string' ? req.query.targetLang : 'en';
    const sourceLang = typeof req.query.sourceLang === 'string' ? req.query.sourceLang : '';
    const glossary = typeof req.query.glossary === 'string' ? req.query.glossary : '';
    const prompt = typeof req.query.prompt === 'string' ? req.query.prompt : '';
    const promptTemplateId = typeof req.query.promptTemplateId === 'string' ? req.query.promptTemplateId : '';
    const strictJsonLineRepairRaw = req.query.strictJsonLineRepair;
    const strictJsonLineRepair =
      strictJsonLineRepairRaw === undefined
        ? true
        : !['0', 'false', 'off', 'no'].includes(String(strictJsonLineRepairRaw).trim().toLowerCase());
    const modelId = typeof req.query.modelId === 'string' ? req.query.modelId : undefined;
    const assetName = typeof req.query.assetName === 'string' ? req.query.assetName.trim() : '';
    const requestId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
    let clientDisconnected = false;
    const abortController = new AbortController();
    req.on('close', () => {
      clientDisconnected = true;
      abortController.abort();
      writeAsrLog(`[TR ${requestId}] client disconnected`);
    });

    try {
      const [ProjectManager, TranslationService] = await Promise.all([
        getProjectManager(),
        getTranslationService(),
      ]);
      const projects = await ProjectManager.getAllProjects();
      const project = projects.find((p) => p.id === projectId);
      if (!project) {
        res.write(`data: ${JSON.stringify({ error: 'Project not found.' })}\n\n`);
        return res.end();
      }

      let sourceText = '';
      if (assetName) {
        sourceText =
          (await readProjectSourceAsset(projectId, 'subtitles', assetName)) ??
          (await readProjectSourceAsset(projectId, 'assets', assetName)) ??
          '';
      } else {
        sourceText = String(project.originalSubtitles || '').trim();
      }

      if (!sourceText) {
        res.write(`data: ${JSON.stringify({ error: 'No source text available for translation.' })}\n\n`);
        return res.end();
      }

      writeAsrLog(`[TR ${requestId}] start`, {
        projectId,
        modelId: modelId || '',
        assetName,
        sourceLang: sourceLang || '',
        targetLang,
        hasGlossary: Boolean(glossary),
        hasPrompt: Boolean(String(prompt || '').trim()),
        promptTemplateId: promptTemplateId || null,
        strictJsonLineRepair,
      });

      const result = await TranslationService.translateTextDetailed(
        {
          text: sourceText,
          targetLang,
          sourceLang: sourceLang || undefined,
          glossary,
          prompt,
          promptTemplateId,
          modelId,
          enableJsonLineRepair: strictJsonLineRepair,
          signal: abortController.signal,
        },
        (msg) => {
          if (clientDisconnected || res.writableEnded) return;
          writeAsrLog(`[TR ${requestId}] ${msg}`);
          res.write(`data: ${JSON.stringify({ status: 'processing', message: msg })}\n\n`);
        }
      );

      if (clientDisconnected || res.writableEnded) {
        writeAsrLog(`[TR ${requestId}] skipped completion because client disconnected`);
        return;
      }
      const artifacts = await persistTranslationArtifacts(projectId, String(result.translatedText || ''));
      const actualModelId =
        typeof result.debug?.provider?.modelId === 'string' && result.debug.provider.modelId.trim()
          ? result.debug.provider.modelId.trim()
          : typeof modelId === 'string' && modelId.trim()
            ? modelId.trim()
            : null;
      await ProjectManager.updateProject(projectId, {
        originalSubtitles: String(project.originalSubtitles || sourceText || '').trim(),
        translatedSubtitles: String(result.translatedText || '').trim(),
        status: PROJECT_STATUS.COMPLETED,
        translationMetadata: {
          lastModelId: actualModelId,
          lastProviderModel:
            typeof result.debug?.provider?.model === 'string' ? String(result.debug.provider.model || '').trim() : null,
          lastProviderName:
            typeof result.debug?.provider?.name === 'string' ? String(result.debug.provider.name || '').trim() : null,
          lastSourceType: assetName ? 'project' : 'transcription',
          lastSourceLang: sourceLang || null,
          lastTargetLang: targetLang || null,
          lastTranslatedAt: new Date().toISOString(),
        },
      });

      if (clientDisconnected || res.writableEnded) {
        writeAsrLog(`[TR ${requestId}] skipped response after artifacts because client disconnected`);
        return;
      }
      writeAsrLog(`[TR ${requestId}] completed`, result.debug || {});
      res.write(`data: ${JSON.stringify({ status: 'completed', result: { ...result, exports: { hasTimecodes: artifacts.hasTimecodes } } })}\n\n`);
      return res.end();
    } catch (error: any) {
      if (clientDisconnected && isAbortLikeError(error)) {
        writeAsrLog(`[TR ${requestId}] aborted after client stop/disconnect`);
        return;
      }
      if (clientDisconnected || res.writableEnded) return;
      writeAsrLog(`[TR ${requestId}] failed`, error?.message || error, 'error');
      res.write(`data: ${JSON.stringify({ error: error?.message || 'Translation failed.' })}\n\n`);
      return res.end();
    } finally {
      if (clientDisconnected && modelId && modelId.startsWith('local_translate_')) {
        writeAsrLog(`[TR ${requestId}] client disconnected; keeping local translate runtime warm`);
      }
    }
  });

  app.post('/api/translate', async (req, res) => {
    try {
      const TranslationService = await getTranslationService();
      const text = typeof req.body?.text === 'string' ? req.body.text : '';
      const targetLang = typeof req.body?.targetLang === 'string' ? req.body.targetLang : 'en';
      const sourceLang = typeof req.body?.sourceLang === 'string' ? req.body.sourceLang : '';
      const glossary = typeof req.body?.glossary === 'string' ? req.body.glossary : '';
      const prompt = typeof req.body?.prompt === 'string' ? req.body.prompt : '';
      const promptTemplateId = typeof req.body?.promptTemplateId === 'string' ? req.body.promptTemplateId : '';
      const strictJsonLineRepairRaw = req.body?.strictJsonLineRepair;
      const strictJsonLineRepair =
        typeof strictJsonLineRepairRaw === 'boolean'
          ? strictJsonLineRepairRaw
          : strictJsonLineRepairRaw === undefined
            ? true
            : !['0', 'false', 'off', 'no'].includes(String(strictJsonLineRepairRaw).trim().toLowerCase());
      const modelId = typeof req.body?.modelId === 'string' ? req.body.modelId : undefined;

      if (!text.trim()) {
        return res.status(400).json({ error: 'Text is required' });
      }

      const result = await TranslationService.translateTextDetailed({
        text,
        targetLang,
        sourceLang: sourceLang || undefined,
        glossary,
        prompt,
        promptTemplateId,
        modelId,
        enableJsonLineRepair: strictJsonLineRepair,
      });

      res.json({ translatedText: result.translatedText, debug: result.debug });
    } catch (error: any) {
      res.status(400).json({ error: error.message || 'Translation failed' });
    }
  });
}
