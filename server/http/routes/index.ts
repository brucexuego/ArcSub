import express from 'express';
import { registerDownloadProgressRoute } from './download_routes.js';
import { registerExportRoutes } from './export_routes.js';
import { registerSystemAndLocalModelRoutes } from './local_model_routes.js';
import { registerProjectRoutes } from './project_routes.js';
import { registerSettingsCrudRoutes } from './settings_routes.js';
import { registerSettingsTestConnectionRoute } from './settings_test_connection_route.js';
import { registerTranscribeRoute } from './transcribe_route.js';
import { registerTranslationRoutes } from './translation_routes.js';
import { registerVideoSourceRoutes } from './video_routes.js';

export interface RegisterHttpRoutesDeps {
  uploadSingle: express.RequestHandler;
  uploadTextSingle: express.RequestHandler;
  parseHttpUrl: (raw: string) => URL | null;
  isAllowedTestUrl: (parsed: URL) => boolean;
  isMaskedKey: (value: string) => boolean;
  isBlockedPlaybackProxyUrl: (parsed: URL) => boolean;
  isRedirectStatus: (status: number) => boolean;
  toClientPath: (value: string) => string;
  writeAsrLog: (message: string, payload?: unknown, level?: 'log' | 'error') => void;
  isAbortLikeError: (error: unknown) => boolean;
}

export function registerHttpRoutes(app: express.Express, deps: RegisterHttpRoutesDeps) {
  registerSystemAndLocalModelRoutes(app);
  registerProjectRoutes(app, { uploadSingle: deps.uploadSingle, uploadTextSingle: deps.uploadTextSingle });
  registerSettingsTestConnectionRoute(app, {
    parseHttpUrl: deps.parseHttpUrl,
    isAllowedTestUrl: deps.isAllowedTestUrl,
    isMaskedKey: deps.isMaskedKey,
  });
  registerSettingsCrudRoutes(app);
  registerVideoSourceRoutes(app, {
    parseHttpUrl: deps.parseHttpUrl,
    isBlockedPlaybackProxyUrl: deps.isBlockedPlaybackProxyUrl,
    isRedirectStatus: deps.isRedirectStatus,
  });
  registerDownloadProgressRoute(app, { parseHttpUrl: deps.parseHttpUrl, toClientPath: deps.toClientPath });
  registerTranscribeRoute(app, { writeAsrLog: deps.writeAsrLog, isAbortLikeError: deps.isAbortLikeError });
  registerTranslationRoutes(app, { writeAsrLog: deps.writeAsrLog, isAbortLikeError: deps.isAbortLikeError });
  registerExportRoutes(app);
}
