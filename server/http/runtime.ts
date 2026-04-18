import express from 'express';
import path from 'path';
import { PathManager } from '../path_manager.js';
import { ResourceManager } from '../services/resource_manager.js';

export interface HttpListenConfig {
  host: string;
  port: number;
}

export async function initializeRuntimeResources() {
  console.log('Initializing ArcSub resources...');
  await PathManager.ensureBaseDirs();
  console.log('[Paths]', PathManager.describeResolvedPaths());
  await ResourceManager.ensureTools();
  console.log('[ResourceWarmup] Tool check completed.');
}

export function mountCoreStaticAssets(app: express.Express) {
  app.use(express.static(PathManager.getPublicPath()));
  app.use('/Projects', express.static(PathManager.getProjectsPath()));
}

export async function mountFrontendRuntime(app: express.Express) {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
    return;
  }

  const distPath = PathManager.getDistPath();
  app.use(express.static(distPath, {
    index: false,
    maxAge: '1y',
    immutable: true,
    setHeaders: (res, filePath) => {
      if (filePath.endsWith('.html')) {
        res.setHeader('Cache-Control', 'no-cache');
      }
    },
  }));
  app.get('*', (req, res) => {
    res.setHeader('Cache-Control', 'no-cache');
    res.sendFile(path.join(distPath, 'index.html'));
  });
}

export function mountGlobalHttpErrorHandler(app: express.Express) {
  app.use((error: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
    const message = String(error?.message || error || 'Unknown server error');
    console.error(
      `[${new Date().toISOString()}] [HTTP unhandled] ${JSON.stringify({
        method: req.method,
        url: req.originalUrl,
        message,
      })}`
    );

    if (res.headersSent) {
      return next(error);
    }

    if (error instanceof URIError || /failed to decode param|uri malformed/i.test(message)) {
      return res.status(400).json({ error: 'Malformed request URL.' });
    }

    return res.status(500).json({ error: 'Internal server error.' });
  });
}

export function listenHttp(
  app: express.Express,
  config: HttpListenConfig,
  onError: (error: unknown) => void
) {
  const httpServer = app.listen(config.port, config.host, () => {
    console.log(`Server running on http://${config.host}:${config.port}`);
  });

  httpServer.on('error', onError);
  return httpServer;
}
