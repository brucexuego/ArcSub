import express from 'express';
import fs from 'fs-extra';
import { PathManager } from '../../path_manager.js';

let projectManagerTask: Promise<typeof import('../../services/project_manager.js').ProjectManager> | null = null;
let textUtilsTask: Promise<typeof import('../text_utils.js')> | null = null;

function getProjectManager() {
  if (!projectManagerTask) {
    projectManagerTask = import('../../services/project_manager.js').then((module) => module.ProjectManager);
  }
  return projectManagerTask;
}

async function persistTranslationArtifactsLazy(projectId: string, translated: string) {
  if (!textUtilsTask) {
    textUtilsTask = import('../text_utils.js');
  }
  const { persistTranslationArtifacts } = await textUtilsTask;
  return persistTranslationArtifacts(projectId, translated);
}

export function registerExportRoutes(app: express.Express) {
  app.get('/api/projects/:projectId/translation/download', async (req, res) => {
    const { projectId } = req.params;
    const format = String(req.query.format || 'txt').toLowerCase();

    try {
      const ProjectManager = await getProjectManager();
      const projects = await ProjectManager.getAllProjects();
      const project = projects.find((p) => p.id === projectId);
      const translated = String(project?.translatedSubtitles || '').trim();
      if (!project || !translated) {
        return res.status(404).json({ error: 'Translated subtitles not found.' });
      }

      const artifacts = await persistTranslationArtifactsLazy(projectId, translated);

      if (format === 'txt') {
        const txtPath = PathManager.resolveProjectFile(projectId, 'subtitles', 'translation.txt', { createProject: false });
        return res.download(txtPath, `${project.name || 'translation'}.txt`);
      }

      if (format === 'srt') {
        if (!artifacts.hasTimecodes) {
          return res.status(400).json({ error: 'SRT export requires timecoded subtitles.' });
        }
        const srtPath = PathManager.resolveProjectFile(projectId, 'subtitles', 'translation.srt', { createProject: false });
        return res.download(srtPath, `${project.name || 'translation'}.srt`);
      }

      if (format === 'vtt') {
        if (!artifacts.hasTimecodes) {
          return res.status(400).json({ error: 'VTT export requires timecoded subtitles.' });
        }
        const vttPath = PathManager.resolveProjectFile(projectId, 'subtitles', 'translation.vtt', { createProject: false });
        return res.download(vttPath, `${project.name || 'translation'}.vtt`);
      }

      return res.status(400).json({ error: 'Unsupported format' });
    } catch (error: any) {
      return res.status(500).json({ error: error.message || 'Failed to export translation.' });
    }
  });

  app.get('/api/projects/:projectId/transcript/download', async (req, res) => {
    const { projectId } = req.params;
    const format = String(req.query.format || 'json');

    let filePath = '';
    try {
      filePath = PathManager.resolveProjectFile(projectId, 'assets', 'transcription.json', { createProject: false });
    } catch (error: any) {
      return res.status(400).json({ error: error.message || 'Invalid project id' });
    }

    if (!(await fs.pathExists(filePath))) {
      return res.status(404).json({ error: 'Transcription file not found' });
    }

    try {
      const data = await fs.readJson(filePath);
      const chunks = data.segments || data.chunks || [];

      const formatTime = (seconds: number, type: 'srt' | 'vtt') => {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        const ms = Math.floor((seconds % 1) * 1000);
        const msSep = type === 'srt' ? ',' : '.';
        return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}${msSep}${ms.toString().padStart(3, '0')}`;
      };

      if (format === 'txt') {
        const text = chunks.map((c: any) => c.text).join('\n');
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        res.setHeader('Content-Disposition', 'attachment; filename="transcript.txt"');
        return res.send(text);
      }

      if (format === 'srt') {
        let srt = '';
        chunks.forEach((c: any, i: number) => {
          const start = formatTime(c.start || c.start_ts || 0, 'srt');
          const end = formatTime(c.end || (c.start_ts + 2) || 2, 'srt');
          srt += `${i + 1}\n${start} --> ${end}\n${String(c.text || '').trim()}\n\n`;
        });
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        res.setHeader('Content-Disposition', 'attachment; filename="transcript.srt"');
        return res.send(srt);
      }

      if (format === 'vtt') {
        let vtt = 'WEBVTT\n\n';
        chunks.forEach((c: any) => {
          const start = formatTime(c.start || c.start_ts || 0, 'vtt');
          const end = formatTime(c.end || (c.start_ts + 2) || 2, 'vtt');
          vtt += `${start} --> ${end}\n${String(c.text || '').trim()}\n\n`;
        });
        res.setHeader('Content-Type', 'text/vtt; charset=utf-8');
        res.setHeader('Content-Disposition', 'attachment; filename="transcript.vtt"');
        return res.send(vtt);
      }

      return res.download(filePath, 'transcript.json');
    } catch (error: any) {
      res.status(500).json({ error: `Failed to export transcript: ${error.message}` });
    }
  });
}
