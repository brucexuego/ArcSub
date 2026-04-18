import express from 'express';
import fs from 'fs-extra';
import path from 'path';
import { PathManager } from './server/path_manager.js';
import {
  initializeRuntimeResources,
  listenHttp,
  mountCoreStaticAssets,
  mountFrontendRuntime,
  mountGlobalHttpErrorHandler,
} from './server/http/runtime.js';
import { registerHttpRoutes } from './server/http/routes/index.js';
import {
  isAllowedTestUrl,
  isBlockedPlaybackProxyUrl,
  isMaskedKey,
  isRedirectStatus,
  parseHttpUrl,
} from './server/http/network_guards.js';
import { createUploadHandlers } from './server/http/upload.js';
import { toClientPath } from './server/http/text_utils.js';

const ASR_LOG_PATH = PathManager.getAsrLogPath();

export interface ServerListenConfig {
  host: string;
  port: number;
}

let processHandlersRegistered = false;

export function writeAsrLog(message: string, payload?: unknown, level: 'log' | 'error' = 'log') {
  const payloadText =
    payload === undefined
      ? ''
      : ` ${typeof payload === 'string' ? payload : JSON.stringify(payload)}`;
  const line = `[${new Date().toISOString()}] ${message}${payloadText}`;

  if (level === 'error') {
    console.error(line);
  } else {
    console.log(line);
  }

  try {
    fs.ensureDirSync(path.dirname(ASR_LOG_PATH));
    fs.appendFileSync(ASR_LOG_PATH, `${line}\n`, 'utf8');
  } catch {
    // Keep request path healthy even if log file cannot be written.
  }
}

export function resolveServerListenConfig(): ServerListenConfig {
  return {
    host: process.env.HOST || '127.0.0.1',
    port: Number(process.env.PORT || 3000),
  };
}

function isAbortLikeError(error: unknown) {
  const message = error instanceof Error ? error.message : String(error || '');
  return /aborted/i.test(message);
}

function parseEnvInt(name: string, fallback: number, min = 0) {
  const raw = Number(String(process.env[name] || '').trim());
  if (!Number.isFinite(raw)) return fallback;
  return Math.max(min, Math.round(raw));
}

function parseEnvBoolean(name: string, fallback = false) {
  const raw = String(process.env[name] || '').trim();
  if (!raw) return fallback;
  if (/^(1|true|yes|on)$/i.test(raw)) return true;
  if (/^(0|false|no|off)$/i.test(raw)) return false;
  return fallback;
}

function normalizeTimingPath(rawUrl: string) {
  const withoutQuery = String(rawUrl || '/').split('?')[0] || '/';
  return withoutQuery
    .replace(/\/[0-9a-f]{24}(?=\/|$)/gi, '/:id')
    .replace(/\/[0-9a-f]{8}-[0-9a-f-]{27,}(?=\/|$)/gi, '/:id')
    .replace(/\/\d+(?=\/|$)/g, '/:n');
}

function percentile(values: number[], p: number) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[index];
}

function mountHttpTimingMiddleware(app: express.Express) {
  const slowMs = parseEnvInt('APP_HTTP_SLOW_MS', 800, 50);
  const logAll = parseEnvBoolean('APP_LOG_ALL_HTTP_TIMING', false);
  const logSystemResourcesPolling = parseEnvBoolean('APP_LOG_SYSTEM_RESOURCES_POLLING', false);
  const summaryWindowSec = parseEnvInt('APP_HTTP_TIMING_SUMMARY_SEC', 60, 0);
  const summaryTopN = parseEnvInt('APP_HTTP_TIMING_TOPN', 8, 1);
  const sampleCap = parseEnvInt('APP_HTTP_TIMING_SAMPLE_CAP', 60, 10);
  const routeStats = new Map<
    string,
    {
      count: number;
      totalMs: number;
      maxMs: number;
      samples: number[];
    }
  >();

  if (summaryWindowSec > 0) {
    const summaryTimer = setInterval(() => {
      if (routeStats.size === 0) return;
      const summary = Array.from(routeStats.entries())
        .map(([route, stat]) => ({
          route,
          count: stat.count,
          avgMs: Number((stat.totalMs / Math.max(1, stat.count)).toFixed(1)),
          p50Ms: Number(percentile(stat.samples, 50).toFixed(1)),
          p95Ms: Number(percentile(stat.samples, 95).toFixed(1)),
          maxMs: Number(stat.maxMs.toFixed(1)),
        }))
        .sort((a, b) => b.p95Ms - a.p95Ms)
        .slice(0, summaryTopN);

      writeAsrLog('[HTTP timing summary]', {
        windowSec: summaryWindowSec,
        topRoutes: summary,
      });

      routeStats.clear();
    }, summaryWindowSec * 1000);
    summaryTimer.unref?.();
  }

  app.use((req, res, next) => {
    const start = process.hrtime.bigint();
    res.on('finish', () => {
      const durationMs = Number(process.hrtime.bigint() - start) / 1_000_000;
      const normalizedPath = normalizeTimingPath(req.originalUrl || req.url || '/');
      const isSystemResourcesPolling = normalizedPath === '/api/system/resources';

      // Sidebar polls this endpoint every few seconds; suppress by default to reduce log noise.
      // Set APP_LOG_SYSTEM_RESOURCES_POLLING=1 when actively debugging this route.
      if (isSystemResourcesPolling && !logSystemResourcesPolling) {
        return;
      }

      const routeKey = `${req.method} ${normalizedPath}`;
      const stat = routeStats.get(routeKey) || { count: 0, totalMs: 0, maxMs: 0, samples: [] as number[] };
      stat.count += 1;
      stat.totalMs += durationMs;
      stat.maxMs = Math.max(stat.maxMs, durationMs);
      stat.samples.push(durationMs);
      if (stat.samples.length > sampleCap) {
        stat.samples.shift();
      }
      routeStats.set(routeKey, stat);

      if (!logAll && durationMs < slowMs) return;
      writeAsrLog('[HTTP timing]', {
        method: req.method,
        url: req.originalUrl,
        status: res.statusCode,
        durationMs: Number(durationMs.toFixed(1)),
      });
    });
    next();
  });
}

export function registerProcessErrorLogging() {
  if (processHandlersRegistered) return;
  processHandlersRegistered = true;

  process.on('uncaughtException', (error) => {
    const detail = error?.stack || error?.message || String(error || 'Unknown uncaught exception');
    writeAsrLog('[Process uncaughtException]', detail, 'error');
  });

  process.on('unhandledRejection', (reason) => {
    const detail =
      reason instanceof Error
        ? reason.stack || reason.message
        : typeof reason === 'string'
          ? reason
          : JSON.stringify(reason);
    writeAsrLog('[Process unhandledRejection]', detail, 'error');
  });
}

export async function startServer(config: ServerListenConfig = resolveServerListenConfig()) {
  const { host, port } = config;
  const app = express();
  const startupBeginAt = Date.now();
  app.disable('x-powered-by');
  app.use(express.json({ limit: '1mb' }));
  mountHttpTimingMiddleware(app);

  const resourceInitStartAt = Date.now();
  await initializeRuntimeResources();
  writeAsrLog('[Startup phase]', {
    phase: 'initializeRuntimeResources',
    durationMs: Date.now() - resourceInitStartAt,
  });

  const staticMountStartAt = Date.now();
  mountCoreStaticAssets(app);
  writeAsrLog('[Startup phase]', {
    phase: 'mountCoreStaticAssets',
    durationMs: Date.now() - staticMountStartAt,
  });

  const { uploadSingle, uploadTextSingle } = createUploadHandlers();

  const routeRegisterStartAt = Date.now();
  registerHttpRoutes(app, {
    uploadSingle,
    uploadTextSingle,
    parseHttpUrl,
    isAllowedTestUrl,
    isMaskedKey,
    isBlockedPlaybackProxyUrl,
    isRedirectStatus,
    toClientPath,
    writeAsrLog,
    isAbortLikeError,
  });
  writeAsrLog('[Startup phase]', {
    phase: 'registerHttpRoutes',
    durationMs: Date.now() - routeRegisterStartAt,
  });

  const frontendMountStartAt = Date.now();
  await mountFrontendRuntime(app);
  writeAsrLog('[Startup phase]', {
    phase: 'mountFrontendRuntime',
    durationMs: Date.now() - frontendMountStartAt,
  });

  mountGlobalHttpErrorHandler(app);
  writeAsrLog('[Startup phase]', {
    phase: 'total',
    durationMs: Date.now() - startupBeginAt,
  });

  listenHttp(app, { host, port }, (error: any) => {
    writeAsrLog('[HTTP listen error]', error?.stack || error?.message || String(error || 'Unknown listen error'), 'error');
  });
}
