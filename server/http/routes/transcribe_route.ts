import express from 'express';
import { buildLegacyProcessingEvent, buildRunFailureIssue } from './run_progress_events.js';

export interface TranscribeRouteDeps {
  writeAsrLog: (message: string, payload?: unknown, level?: 'log' | 'error') => void;
  isAbortLikeError: (error: unknown) => boolean;
}

let asrServiceTask: Promise<typeof import('../../services/asr_service.js').AsrService> | null = null;

function getAsrService() {
  if (!asrServiceTask) {
    asrServiceTask = import('../../services/asr_service.js').then((module) => module.AsrService);
  }
  return asrServiceTask;
}

export function registerTranscribeRoute(app: express.Express, deps: TranscribeRouteDeps) {
  const { writeAsrLog, isAbortLikeError } = deps;

  app.get('/api/transcribe/:projectId', (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    res.flushHeaders();
    const keepAlive = setInterval(() => {
      if (res.writableEnded) return;
      res.write(': keepalive\n\n');
    }, 10000);

    const { projectId } = req.params;
    const { modelId, assetName, language, prompt, segmentation, wordAlignment, vad, diarization, pipelineMode } = req.query;
    const diarizationProvider = typeof req.query.diarizationProvider === 'string' ? req.query.diarizationProvider : undefined;
    const diarizationMode = typeof req.query.diarizationMode === 'string' ? req.query.diarizationMode : undefined;
    const diarizationScenePreset = typeof req.query.diarizationScenePreset === 'string' ? req.query.diarizationScenePreset : undefined;
    const diarizationExactSpeakerCount = Number(req.query.diarizationExactSpeakerCount);
    const diarizationMinSpeakers = Number(req.query.diarizationMinSpeakers);
    const diarizationMaxSpeakers = Number(req.query.diarizationMaxSpeakers);
    const parseOptionalBoolean = (value: unknown) => {
      if (value == null) return undefined;
      const normalized = String(value).trim().toLowerCase();
      if (['1', 'true', 'on', 'yes'].includes(normalized)) return true;
      if (['0', 'false', 'off', 'no'].includes(normalized)) return false;
      return undefined;
    };
    const diarizationOptions = {
      provider: diarizationProvider as any,
      mode: diarizationMode as any,
      scenePreset: diarizationScenePreset as any,
      exactSpeakerCount: Number.isFinite(diarizationExactSpeakerCount) ? diarizationExactSpeakerCount : undefined,
      minSpeakers: Number.isFinite(diarizationMinSpeakers) ? diarizationMinSpeakers : undefined,
      maxSpeakers: Number.isFinite(diarizationMaxSpeakers) ? diarizationMaxSpeakers : undefined,
      preferStablePrimarySpeaker: parseOptionalBoolean(req.query.preferStablePrimarySpeaker),
      allowShortInterjectionSpeaker: parseOptionalBoolean(req.query.allowShortInterjectionSpeaker),
      preferVadBoundedRegions: parseOptionalBoolean(req.query.preferVadBoundedRegions),
      forceMergeTinyClustersInTwoSpeakerMode: parseOptionalBoolean(req.query.forceMergeTinyClustersInTwoSpeakerMode),
      semanticFallbackEnabled: parseOptionalBoolean(req.query.semanticFallbackEnabled),
    };
    const requestId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
    let clientDisconnected = false;
    let cleanedUp = false;
    const abortController = new AbortController();
    const cleanup = () => {
      if (cleanedUp) return;
      cleanedUp = true;
      clearInterval(keepAlive);
      req.off('close', handleClientClose);
      req.off('aborted', handleClientAborted);
      res.off('close', handleResponseClose);
      res.off('error', handleResponseError);
    };
    const handleClientClose = () => {
      clientDisconnected = true;
      abortController.abort();
      cleanup();
      writeAsrLog(`[ASR ${requestId}] client disconnected`);
    };
    const handleClientAborted = () => {
      clientDisconnected = true;
      abortController.abort();
      cleanup();
      writeAsrLog(`[ASR ${requestId}] client aborted`);
    };
    const handleResponseClose = () => {
      clientDisconnected = true;
      abortController.abort();
      cleanup();
    };
    const handleResponseError = () => {
      clientDisconnected = true;
      abortController.abort();
      cleanup();
    };
    req.on('close', handleClientClose);
    req.on('aborted', handleClientAborted);
    res.on('close', handleResponseClose);
    res.on('error', handleResponseError);

    writeAsrLog(`[ASR ${requestId}] start`, {
      projectId,
      modelId: modelId || '',
      assetName: assetName || '',
      language: language || '',
      segmentation: segmentation === 'true',
      wordAlignment: wordAlignment !== 'false',
      vad: vad === 'true',
      diarization: diarization === 'true',
      pipelineMode: typeof pipelineMode === 'string' ? pipelineMode : '',
      diarizationOptions: diarization === 'true' ? diarizationOptions : null,
      hasPrompt: Boolean(prompt),
    });

    void getAsrService()
      .then((AsrService) => AsrService.transcribe(
      projectId,
      {
        modelId: modelId as string,
        assetName: assetName as string,
        language: language as string,
        prompt: prompt as string,
        segmentation: segmentation === 'true',
        wordAlignment: wordAlignment !== 'false',
        vad: vad === 'true',
        diarization: diarization === 'true',
        pipelineMode: typeof pipelineMode === 'string' ? (pipelineMode as 'stable' | 'throughput') : undefined,
        diarizationOptions: diarization === 'true' ? diarizationOptions : undefined,
      },
      (msg) => {
        if (clientDisconnected || res.writableEnded) return;
        writeAsrLog(`[ASR ${requestId}] ${msg}`);
        res.write(`data: ${JSON.stringify({ status: 'processing', message: msg, event: buildLegacyProcessingEvent('asr', msg) })}\n\n`);
      },
      abortController.signal
      ))
      .then((result) => {
        cleanup();
        if (clientDisconnected || res.writableEnded) return;
        writeAsrLog(`[ASR ${requestId}] completed`, result?.debug || {});
        res.write(`data: ${JSON.stringify({
          status: 'completed',
          event: {
            status: 'completed',
            code: 'run.completed',
            stage: 'complete',
            progressHint: 100,
            message: 'Transcription completed.',
            data: { kind: 'asr' },
          },
          result,
        })}\n\n`);
        res.end();
      })
      .catch((error) => {
        cleanup();
        if (clientDisconnected && isAbortLikeError(error)) {
          writeAsrLog(`[ASR ${requestId}] aborted after client stop/disconnect`);
          return;
        }
        if (clientDisconnected || res.writableEnded) return;
        writeAsrLog(`[ASR ${requestId}] failed`, error?.message || error, 'error');
        const errorMessage = error?.message || 'Transcription failed.';
        res.write(`data: ${JSON.stringify({
          error: errorMessage,
          errorIssue: buildRunFailureIssue('asr.request.failed', errorMessage),
        })}\n\n`);
        res.end();
      });
  });
}
