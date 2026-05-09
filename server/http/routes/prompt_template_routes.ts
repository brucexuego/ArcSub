import express from 'express';
import { PromptTemplateService } from '../../services/prompt_template_service.js';

function readTargetLang(req: express.Request) {
  const value = req.method === 'GET' ? req.query.targetLang : req.body?.targetLang;
  return typeof value === 'string' && value.trim() ? value.trim() : 'en';
}

function readSourceLang(req: express.Request) {
  const value = req.method === 'GET' ? req.query.sourceLang : req.body?.sourceLang;
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
}

function sendError(res: express.Response, error: unknown) {
  const message = error instanceof Error ? error.message : String(error || 'Prompt template request failed.');
  return res.status(400).json({ error: message });
}

export function registerPromptTemplateRoutes(app: express.Express) {
  app.get('/api/translation/prompt-templates', async (req, res) => {
    try {
      const templates = await PromptTemplateService.listTemplates({
        targetLang: readTargetLang(req),
        sourceLang: readSourceLang(req),
      });
      res.json({ templates });
    } catch (error) {
      sendError(res, error);
    }
  });

  app.get('/api/translation/prompt-templates/:templateId', async (req, res) => {
    try {
      const template = await PromptTemplateService.resolveTemplate({
        templateId: req.params.templateId,
        targetLang: readTargetLang(req),
        sourceLang: readSourceLang(req),
      });
      if (!template) return res.status(404).json({ error: 'Prompt template not found.' });
      return res.json({ template });
    } catch (error) {
      return sendError(res, error);
    }
  });

  app.post('/api/translation/prompt-templates/:templateId', async (req, res) => {
    try {
      const template = await PromptTemplateService.saveTemplateOverride({
        templateId: req.params.templateId,
        targetLang: readTargetLang(req),
        sourceLang: readSourceLang(req),
        content: typeof req.body?.content === 'string' ? req.body.content : '',
        fallbackLabel: typeof req.body?.fallbackLabel === 'string' ? req.body.fallbackLabel : undefined,
      });
      return res.json({ template });
    } catch (error) {
      return sendError(res, error);
    }
  });

  app.post('/api/translation/prompt-templates/:templateId/reset', async (req, res) => {
    try {
      const template = await PromptTemplateService.resetTemplateOverride({
        templateId: req.params.templateId,
        targetLang: readTargetLang(req),
        sourceLang: readSourceLang(req),
      });
      return res.json({ template });
    } catch (error) {
      return sendError(res, error);
    }
  });
}
