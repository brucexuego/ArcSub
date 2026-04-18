import express from 'express';
import { Readable } from 'node:stream';
import { extractErrorMessage } from '../text_utils.js';

export interface VideoSourceRouteDeps {
  parseHttpUrl: (raw: string) => URL | null;
  isBlockedPlaybackProxyUrl: (parsed: URL) => boolean;
  isRedirectStatus: (status: number) => boolean;
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

export function registerVideoSourceRoutes(app: express.Express, deps: VideoSourceRouteDeps) {
  const { parseHttpUrl, isBlockedPlaybackProxyUrl, isRedirectStatus } = deps;

  app.post('/api/parse-video', async (req, res) => {
    const rawUrl = String(req.body?.url || '').trim();
    const parsed = parseHttpUrl(rawUrl);
    if (!parsed) {
      return res.status(400).json({ error: 'Valid URL is required' });
    }

    const projectId = String(req.body?.projectId || '').trim();

    try {
      if (projectId) {
        const ProjectManager = await getProjectManager();
        const projects = await ProjectManager.getAllProjects();
        const project = projects.find((item) => item.id === projectId);
        if (!project) {
          return res.status(404).json({ error: 'Project not found' });
        }
        const sourceMode = inferProjectSourceMode(project);
        if (sourceMode === 'upload') {
          return res.status(409).json({ error: 'This project is locked to uploaded media source mode' });
        }
      }

      const VideoService = await getVideoService();
      const metadata = await VideoService.parseMetadata(parsed.toString());
      res.json(metadata);
    } catch (error: any) {
      res.status(500).json({ error: error.message || 'Failed to parse video' });
    }
  });

  app.get('/api/playback-proxy', async (req, res) => {
    const rawUrl = String(req.query.url || '').trim();
    const parsed = parseHttpUrl(rawUrl);
    if (!parsed) {
      return res.status(400).json({ error: 'Valid playback URL is required' });
    }
    if (isBlockedPlaybackProxyUrl(parsed)) {
      return res.status(400).json({ error: 'Blocked playback source host' });
    }

    const rangeHeader = req.header('range');
    const ifRangeHeader = req.header('if-range');

    const VideoService = await getVideoService();
    const buildUpstreamHeaders = (targetUrl: string) => {
      const headers = new Headers(VideoService.buildPlaybackRequestHeaders(targetUrl));
      if (rangeHeader) headers.set('Range', rangeHeader);
      if (ifRangeHeader) headers.set('If-Range', ifRangeHeader);
      return headers;
    };

    try {
      const maxRedirects = 5;
      const visitedUrls = new Set<string>();
      let targetUrl = parsed;
      let upstream: Response | null = null;

      for (let hop = 0; hop <= maxRedirects; hop += 1) {
        const targetHref = targetUrl.toString();
        if (visitedUrls.has(targetHref)) {
          return res.status(400).json({ error: 'Blocked playback redirect loop' });
        }
        visitedUrls.add(targetHref);

        upstream = await fetch(targetHref, {
          headers: buildUpstreamHeaders(targetHref),
          redirect: 'manual',
        });

        if (!isRedirectStatus(upstream.status)) {
          break;
        }

        const location = String(upstream.headers.get('location') || '').trim();
        if (!location) {
          return res.status(502).json({ error: 'Playback redirect missing location header' });
        }

        const nextTarget = parseHttpUrl(new URL(location, targetUrl).toString());
        if (!nextTarget) {
          return res.status(400).json({ error: 'Blocked playback redirect protocol' });
        }
        if (isBlockedPlaybackProxyUrl(nextTarget)) {
          return res.status(400).json({ error: 'Blocked playback redirect host' });
        }
        targetUrl = nextTarget;
      }

      if (!upstream || isRedirectStatus(upstream.status)) {
        return res.status(502).json({ error: 'Playback redirect limit exceeded' });
      }

      if (!upstream.ok && upstream.status !== 206) {
        const contentType = String(upstream.headers.get('content-type') || '').toLowerCase();
        const bodyText = await upstream.text().catch(() => '');
        const detail = extractErrorMessage(bodyText, contentType);
        return res
          .status(upstream.status || 502)
          .json({ error: detail || `Playback source request failed (${upstream.status})` });
      }

      res.status(upstream.status === 206 ? 206 : 200);
      res.setHeader('Access-Control-Allow-Origin', '*');

      const passthroughHeaders = [
        'content-type',
        'content-length',
        'content-range',
        'accept-ranges',
        'cache-control',
        'etag',
        'last-modified',
      ];

      passthroughHeaders.forEach((headerName) => {
        const headerValue = upstream.headers.get(headerName);
        if (headerValue) {
          res.setHeader(headerName, headerValue);
        }
      });

      if (!res.hasHeader('accept-ranges')) {
        res.setHeader('accept-ranges', 'bytes');
      }

      if (!upstream.body) {
        return res.end();
      }

      const upstreamStream = Readable.fromWeb(upstream.body as any);
      req.on('close', () => {
        upstreamStream.destroy();
      });
      upstreamStream.on('error', (error) => {
        if (!res.headersSent) {
          res.status(502).json({ error: error instanceof Error ? error.message : 'Playback proxy stream failed' });
          return;
        }
        res.destroy(error instanceof Error ? error : undefined);
      });
      upstreamStream.pipe(res);
    } catch (error: any) {
      res.status(502).json({ error: error?.message || 'Playback proxy failed' });
    }
  });
}
