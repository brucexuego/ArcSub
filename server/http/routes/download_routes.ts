import express from 'express';
import { PathManager } from '../../path_manager.js';

export interface DownloadProgressRouteDeps {
  parseHttpUrl: (raw: string) => URL | null;
  toClientPath: (value: string) => string;
}

let videoServiceTask: Promise<typeof import('../../services/video_service.js').VideoService> | null = null;
let projectManagerTask: Promise<typeof import('../../services/project_manager.js').ProjectManager> | null = null;

function getVideoService() {
  if (!videoServiceTask) {
    videoServiceTask = import('../../services/video_service.js').then((module) => module.VideoService);
  }
  return videoServiceTask;
}

function getProjectManager() {
  if (!projectManagerTask) {
    projectManagerTask = import('../../services/project_manager.js').then((module) => module.ProjectManager);
  }
  return projectManagerTask;
}

function inferProjectSourceMode(project: any): 'online' | 'upload' | null {
  const explicit = String(project?.mediaSourceType || '').trim().toLowerCase();
  if (explicit === 'online' || explicit === 'upload') {
    return explicit;
  }

  const metadataMode = String(project?.videoMetadata?.sourceMode || '').trim().toLowerCase();
  if (metadataMode === 'online' || metadataMode === 'upload') {
    return metadataMode;
  }

  const hasOnlineLastUrl = String(project?.videoMetadata?.lastUrl || '').trim();
  if (hasOnlineLastUrl) {
    return 'online';
  }
  return null;
}

export function registerDownloadProgressRoute(app: express.Express, deps: DownloadProgressRouteDeps) {
  const { parseHttpUrl, toClientPath } = deps;

  app.get('/api/download-progress/:projectId', (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    const { projectId } = req.params;
    const url = String(req.query.url || '').trim();
    const formatId = String(req.query.formatId || '').trim();
    const downloadType = req.query.type === 'audio' ? 'audio' : 'video';

    if (!parseHttpUrl(url) || !formatId) {
      res.write(`data: ${JSON.stringify({ error: 'Missing or invalid parameters' })}\n\n`);
      return res.end();
    }

    try {
      PathManager.assertValidProjectId(projectId);
    } catch (error: any) {
      res.write(`data: ${JSON.stringify({ error: error.message || 'Invalid project id' })}\n\n`);
      return res.end();
    }

    void (async () => {
      const ProjectManager = await getProjectManager();
      const projects = await ProjectManager.getAllProjects();
      const project = projects.find((item) => item.id === projectId);
      if (!project) {
        throw new Error('Project not found');
      }

      const sourceMode = inferProjectSourceMode(project);
      if (sourceMode === 'upload') {
        throw new Error('This project is locked to uploaded media source mode');
      }

      const VideoService = await getVideoService();
      const videoPath = await VideoService.downloadVideo(url, projectId, formatId, downloadType, (progress) => {
        res.write(`data: ${JSON.stringify(progress)}\n\n`);
      });
      res.write(`data: ${JSON.stringify({ status: 'extracting', msg: 'Extracting audio (16kHz)...' })}\n\n`);
      const audioPath = await VideoService.extractAudio(videoPath, projectId);
      res.write(`data: ${JSON.stringify({ status: 'finished', videoPath: toClientPath(videoPath), audioPath: toClientPath(audioPath) })}\n\n`);
      res.end();
    })().catch((error) => {
      res.write(`data: ${JSON.stringify({ error: error?.message || 'Download failed' })}\n\n`);
      res.end();
    });
  });
}
