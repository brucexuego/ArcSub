import express from 'express';
import path from 'path';
import fs from 'fs-extra';
import { PathManager } from '../../path_manager.js';

let projectManagerTask: Promise<typeof import('../../services/project_manager.js').ProjectManager> | null = null;
let videoServiceTask: Promise<typeof import('../../services/video_service.js').VideoService> | null = null;

const UPLOAD_VIDEO_EXTENSIONS = new Set([
  '.mp4',
  '.mkv',
  '.mov',
  '.avi',
  '.wmv',
  '.webm',
  '.m4v',
  '.flv',
  '.ts',
  '.mpeg',
  '.mpg',
  '.m2ts',
  '.mts',
  '.3gp',
  '.ogv',
  '.vob',
]);

function getProjectManager() {
  if (!projectManagerTask) {
    projectManagerTask = import('../../services/project_manager.js').then((module) => module.ProjectManager);
  }
  return projectManagerTask;
}

function getVideoService() {
  if (!videoServiceTask) {
    videoServiceTask = import('../../services/video_service.js').then((module) => module.VideoService);
  }
  return videoServiceTask;
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

export interface ProjectRouteHandlers {
  uploadSingle: express.RequestHandler;
  uploadTextSingle: express.RequestHandler;
}

export function registerProjectRoutes(app: express.Express, handlers: ProjectRouteHandlers) {
  const { uploadSingle, uploadTextSingle } = handlers;

  app.get('/api/projects', async (req, res) => {
    try {
      const ProjectManager = await getProjectManager();
      const projects = await ProjectManager.getAllProjects();
      res.json(projects);
    } catch {
      res.status(500).json({ error: 'Failed to fetch projects' });
    }
  });

  app.post('/api/projects', async (req, res) => {
    try {
      const ProjectManager = await getProjectManager();
      const { name, notes } = req.body || {};
      const project = await ProjectManager.createProject(name, notes);
      res.json(project);
    } catch (error: any) {
      res.status(400).json({ error: error.message || 'Failed to create project' });
    }
  });

  app.get('/api/projects/:id', async (req, res) => {
    try {
      const ProjectManager = await getProjectManager();
      const projectId = String(req.params.id || '').trim();
      const projects = await ProjectManager.getAllProjects();
      const project = projects.find((item) => item.id === projectId);
      if (!project) {
        return res.status(404).json({ error: 'Project not found' });
      }
      return res.json(project);
    } catch (error: any) {
      return res.status(400).json({ error: error.message || 'Failed to fetch project' });
    }
  });

  app.patch('/api/projects/:id', async (req, res) => {
    try {
      const ProjectManager = await getProjectManager();
      const project = await ProjectManager.updateProject(req.params.id, req.body);
      if (!project) return res.status(404).json({ error: 'Project not found' });
      res.json(project);
    } catch (error: any) {
      res.status(400).json({ error: error.message || 'Failed to update project' });
    }
  });

  app.delete('/api/projects/:id', async (req, res) => {
    try {
      const ProjectManager = await getProjectManager();
      const success = await ProjectManager.deleteProject(req.params.id);
      res.json({ success });
    } catch (error: any) {
      res.status(400).json({ error: error.message || 'Failed to delete project' });
    }
  });

  app.get('/api/projects/:id/materials', async (req, res) => {
    try {
      const ProjectManager = await getProjectManager();
      const materials = await ProjectManager.getMaterials(req.params.id);
      res.json(materials);
    } catch (error: any) {
      res.status(400).json({ error: error.message || 'Failed to fetch materials' });
    }
  });

  app.post('/api/projects/:id/materials/upload', uploadSingle, async (req, res) => {
    try {
      res.json({ success: true, file: req.file });
    } catch (error: any) {
      res.status(500).json({ error: error.message || 'Failed to upload material' });
    }
  });

  app.post('/api/projects/:id/upload-video', uploadSingle, async (req, res) => {
    const file = req.file;
    if (!file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const ext = path.extname(String(file.filename || file.originalname || '')).toLowerCase();
    if (!UPLOAD_VIDEO_EXTENSIONS.has(ext)) {
      await fs.remove(file.path).catch(() => {});
      return res.status(400).json({ error: 'Unsupported video file type' });
    }

    try {
      const ProjectManager = await getProjectManager();
      const projects = await ProjectManager.getAllProjects();
      const project = projects.find((item) => item.id === req.params.id);

      if (!project) {
        await fs.remove(file.path).catch(() => {});
        return res.status(404).json({ error: 'Project not found' });
      }

      const lockedSourceMode = inferProjectSourceMode(project);
      if (lockedSourceMode === 'online') {
        await fs.remove(file.path).catch(() => {});
        return res.status(409).json({ error: 'This project is locked to online source mode' });
      }

      const VideoService = await getVideoService();
      const playbackVideoPath = await VideoService.ensureBrowserPlayableVideo(file.path, project.id);
      const audioPath = await VideoService.extractAudio(playbackVideoPath, project.id);
      const inferredTitle = path.parse(String(file.originalname || file.filename || project.name || 'uploaded_video')).name.trim();

      return res.json({
        success: true,
        file: {
          filename: file.filename,
          originalname: file.originalname,
          size: file.size,
        },
        videoPath: PathManager.toClientPath(playbackVideoPath),
        audioPath: PathManager.toClientPath(audioPath),
        videoTitle: inferredTitle || project.name || 'uploaded_video',
        sourceMode: 'upload',
      });
    } catch (error: any) {
      return res.status(500).json({ error: error?.message || 'Failed to process uploaded video' });
    }
  });

  app.post('/api/projects/:id/materials/upload-text', uploadTextSingle, async (req, res) => {
    try {
      res.json({ success: true, file: req.file });
    } catch (error: any) {
      res.status(500).json({ error: error.message || 'Failed to upload material' });
    }
  });

  app.delete('/api/projects/:id/materials/:category/:filename', async (req, res) => {
    try {
      const ProjectManager = await getProjectManager();
      const { id, category, filename } = req.params;
      const success = await ProjectManager.deleteMaterial(id, category, filename);
      res.json({ success });
    } catch (error: any) {
      res.status(400).json({ error: error.message || 'Failed to delete material' });
    }
  });
}
