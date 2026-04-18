import path from 'path';
import fs from 'fs-extra';
import { fileURLToPath, pathToFileURL } from 'url';
import { PathManager } from '../path_manager.js';
import { VIDEO_WEB_UA, buildBaseYtDlpArgs } from './runtime.js';
import type {
  LoadedVideoSiteManifest,
  ParseFailureTrigger,
  SiteUrlMatchRule,
  VideoSiteHandlerModule,
  VideoSiteManifest,
} from './types.js';

const PUBLIC_VIDEO_SITES_ROOT = path.dirname(fileURLToPath(import.meta.url));
const VALID_PARSE_TRIGGERS = new Set<ParseFailureTrigger>([
  'no_formats',
  'unsupported_url',
  'cloudflare',
  'bot_detection',
]);

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === 'string' && item.trim());
}

function isStringRecord(value: unknown): value is Record<string, string> {
  return Boolean(value) &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    Object.values(value as Record<string, unknown>).every((item) => typeof item === 'string');
}

function isStringOrStringArrayRecord(value: unknown): value is Record<string, string | string[]> {
  return Boolean(value) &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    Object.values(value as Record<string, unknown>).every((item) => typeof item === 'string' || isStringArray(item));
}

function hasMatchFields(match: SiteUrlMatchRule) {
  return Boolean(
    (match.hosts && match.hosts.length > 0) ||
    (match.hostSuffixes && match.hostSuffixes.length > 0) ||
    (match.urlIncludes && match.urlIncludes.length > 0) ||
    (match.pathnameRegexes && match.pathnameRegexes.length > 0) ||
    (match.queryParamEquals && Object.keys(match.queryParamEquals).length > 0)
  );
}

function validateMatchRule(rule: unknown, label: string): SiteUrlMatchRule {
  if (!rule || typeof rule !== 'object' || Array.isArray(rule)) {
    throw new Error(`${label} must be an object`);
  }
  const candidate = rule as Record<string, unknown>;
  const match: SiteUrlMatchRule = {};

  if (candidate.hosts !== undefined) {
    if (!isStringArray(candidate.hosts)) throw new Error(`${label}.hosts must be a string array`);
    match.hosts = candidate.hosts;
  }
  if (candidate.hostSuffixes !== undefined) {
    if (!isStringArray(candidate.hostSuffixes)) throw new Error(`${label}.hostSuffixes must be a string array`);
    match.hostSuffixes = candidate.hostSuffixes;
  }
  if (candidate.urlIncludes !== undefined) {
    if (!isStringArray(candidate.urlIncludes)) throw new Error(`${label}.urlIncludes must be a string array`);
    match.urlIncludes = candidate.urlIncludes;
  }
  if (candidate.pathnameRegexes !== undefined) {
    if (!isStringArray(candidate.pathnameRegexes)) throw new Error(`${label}.pathnameRegexes must be a string array`);
    candidate.pathnameRegexes.forEach((pattern) => {
      try {
        new RegExp(pattern);
      } catch (error: any) {
        throw new Error(`${label}.pathnameRegexes contains an invalid regex: ${error?.message || pattern}`);
      }
    });
    match.pathnameRegexes = candidate.pathnameRegexes;
  }
  if (candidate.queryParamEquals !== undefined) {
    if (!isStringOrStringArrayRecord(candidate.queryParamEquals)) {
      throw new Error(`${label}.queryParamEquals must be a record of strings or string arrays`);
    }
    match.queryParamEquals = candidate.queryParamEquals;
  }

  if (!hasMatchFields(match)) {
    throw new Error(`${label} must define at least one matcher field`);
  }

  return match;
}

function validateManifest(raw: unknown): VideoSiteManifest {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('Manifest root must be an object');
  }

  const candidate = raw as Record<string, unknown>;
  if (candidate.schemaVersion !== 1) {
    throw new Error('schemaVersion must be 1');
  }

  const id = typeof candidate.id === 'string' ? candidate.id.trim() : '';
  if (!/^[a-z0-9_-]{1,64}$/.test(id)) {
    throw new Error('id must match ^[a-z0-9_-]{1,64}$');
  }

  const manifest: VideoSiteManifest = {
    schemaVersion: 1,
    id,
    match: validateMatchRule(candidate.match, 'match'),
  };

  if (candidate.description !== undefined) {
    if (typeof candidate.description !== 'string') throw new Error('description must be a string');
    manifest.description = candidate.description;
  }
  if (candidate.priority !== undefined) {
    if (!Number.isFinite(candidate.priority)) throw new Error('priority must be a number');
    manifest.priority = Number(candidate.priority);
  }
  if (candidate.enabled !== undefined) {
    if (typeof candidate.enabled !== 'boolean') throw new Error('enabled must be a boolean');
    manifest.enabled = candidate.enabled;
  }

  if (candidate.ytDlp !== undefined) {
    if (!candidate.ytDlp || typeof candidate.ytDlp !== 'object' || Array.isArray(candidate.ytDlp)) {
      throw new Error('ytDlp must be an object');
    }
    const ytDlp = candidate.ytDlp as Record<string, unknown>;
    manifest.ytDlp = {};
    if (ytDlp.impersonate !== undefined) {
      if (typeof ytDlp.impersonate !== 'string' || !ytDlp.impersonate.trim()) {
        throw new Error('ytDlp.impersonate must be a string');
      }
      manifest.ytDlp.impersonate = ytDlp.impersonate;
    }
    if (ytDlp.extractorArgs !== undefined) {
      if (!isStringArray(ytDlp.extractorArgs)) throw new Error('ytDlp.extractorArgs must be a string array');
      manifest.ytDlp.extractorArgs = ytDlp.extractorArgs;
    }
    if (ytDlp.extraArgs !== undefined) {
      if (!isStringArray(ytDlp.extraArgs)) throw new Error('ytDlp.extraArgs must be a string array');
      manifest.ytDlp.extraArgs = ytDlp.extraArgs;
    }
    if (ytDlp.metadataExtraArgs !== undefined) {
      if (!isStringArray(ytDlp.metadataExtraArgs)) throw new Error('ytDlp.metadataExtraArgs must be a string array');
      manifest.ytDlp.metadataExtraArgs = ytDlp.metadataExtraArgs;
    }
    if (ytDlp.downloadExtraArgs !== undefined) {
      if (!isStringArray(ytDlp.downloadExtraArgs)) throw new Error('ytDlp.downloadExtraArgs must be a string array');
      manifest.ytDlp.downloadExtraArgs = ytDlp.downloadExtraArgs;
    }
  }

  if (candidate.playback !== undefined) {
    if (!candidate.playback || typeof candidate.playback !== 'object' || Array.isArray(candidate.playback)) {
      throw new Error('playback must be an object');
    }
    const playback = candidate.playback as Record<string, unknown>;
    manifest.playback = {};
    if (playback.match !== undefined) {
      manifest.playback.match = validateMatchRule(playback.match, 'playback.match');
    }
    if (playback.headers !== undefined) {
      if (!isStringRecord(playback.headers)) throw new Error('playback.headers must be a string record');
      manifest.playback.headers = playback.headers;
    }
  }

  if (candidate.parse !== undefined) {
    if (!candidate.parse || typeof candidate.parse !== 'object' || Array.isArray(candidate.parse)) {
      throw new Error('parse must be an object');
    }
    const parse = candidate.parse as Record<string, unknown>;
    manifest.parse = {};
    if (parse.fallbackOn !== undefined) {
      if (!isStringArray(parse.fallbackOn)) throw new Error('parse.fallbackOn must be a string array');
      const invalid = parse.fallbackOn.filter((item) => !VALID_PARSE_TRIGGERS.has(item as ParseFailureTrigger));
      if (invalid.length > 0) {
        throw new Error(`parse.fallbackOn contains invalid triggers: ${invalid.join(', ')}`);
      }
      manifest.parse.fallbackOn = parse.fallbackOn as ParseFailureTrigger[];
    }
  }

  if (candidate.download !== undefined) {
    if (!candidate.download || typeof candidate.download !== 'object' || Array.isArray(candidate.download)) {
      throw new Error('download must be an object');
    }
    const download = candidate.download as Record<string, unknown>;
    manifest.download = {};
    if (download.referer !== undefined) {
      if (typeof download.referer !== 'string') throw new Error('download.referer must be a string');
      manifest.download.referer = download.referer;
    }
    if (download.headers !== undefined) {
      if (!isStringRecord(download.headers)) throw new Error('download.headers must be a string record');
      manifest.download.headers = download.headers;
    }
    if (download.customFormatPrefixes !== undefined) {
      if (!isStringArray(download.customFormatPrefixes)) {
        throw new Error('download.customFormatPrefixes must be a string array');
      }
      manifest.download.customFormatPrefixes = download.customFormatPrefixes;
    }
    if (download.alwaysUseHandler !== undefined) {
      if (typeof download.alwaysUseHandler !== 'boolean') {
        throw new Error('download.alwaysUseHandler must be a boolean');
      }
      manifest.download.alwaysUseHandler = download.alwaysUseHandler;
    }
  }

  if (candidate.handlers !== undefined) {
    if (!candidate.handlers || typeof candidate.handlers !== 'object' || Array.isArray(candidate.handlers)) {
      throw new Error('handlers must be an object');
    }
    const handlers = candidate.handlers as Record<string, unknown>;
    manifest.handlers = {};
    if (handlers.parse !== undefined) {
      if (typeof handlers.parse !== 'string' || !handlers.parse.trim()) throw new Error('handlers.parse must be a string');
      manifest.handlers.parse = handlers.parse;
    }
    if (handlers.download !== undefined) {
      if (typeof handlers.download !== 'string' || !handlers.download.trim()) throw new Error('handlers.download must be a string');
      manifest.handlers.download = handlers.download;
    }
  }

  if ((manifest.parse?.fallbackOn?.length || 0) > 0 && !manifest.handlers?.parse) {
    throw new Error('parse.fallbackOn requires handlers.parse');
  }
  if ((manifest.download?.alwaysUseHandler || (manifest.download?.customFormatPrefixes?.length || 0) > 0) && !manifest.handlers?.download) {
    throw new Error('download handler settings require handlers.download');
  }

  return manifest;
}

function patternSpecificity(pattern: SiteUrlMatchRule) {
  return (
    (pattern.hosts?.length || 0) * 4 +
    (pattern.hostSuffixes?.length || 0) * 3 +
    (pattern.queryParamEquals ? Object.keys(pattern.queryParamEquals).length * 2 : 0) +
    (pattern.pathnameRegexes?.length || 0) * 2 +
    (pattern.urlIncludes?.length || 0)
  );
}

function normalizeExpectedValues(value: string | string[]) {
  return Array.isArray(value) ? value : [value];
}

function scoreMatch(rawUrl: string, pattern: SiteUrlMatchRule): number {
  let parsed: URL;
  try {
    parsed = new URL(rawUrl);
  } catch {
    return -1;
  }

  const host = parsed.hostname.toLowerCase();
  const fullUrl = parsed.toString().toLowerCase();
  let score = 0;

  if (pattern.hosts?.length) {
    const normalized = pattern.hosts.map((item) => item.toLowerCase());
    if (!normalized.includes(host)) return -1;
    score += normalized.length * 4;
  }

  if (pattern.hostSuffixes?.length) {
    const normalized = pattern.hostSuffixes.map((item) => item.toLowerCase());
    const matched = normalized.some((suffix) => host === suffix || host.endsWith(`.${suffix}`));
    if (!matched) return -1;
    score += normalized.length * 3;
  }

  if (pattern.urlIncludes?.length) {
    const normalized = pattern.urlIncludes.map((item) => item.toLowerCase());
    if (!normalized.every((snippet) => fullUrl.includes(snippet))) return -1;
    score += normalized.length;
  }

  if (pattern.pathnameRegexes?.length) {
    if (!pattern.pathnameRegexes.every((snippet) => new RegExp(snippet).test(parsed.pathname))) return -1;
    score += pattern.pathnameRegexes.length * 2;
  }

  if (pattern.queryParamEquals) {
    for (const [key, expectedRaw] of Object.entries(pattern.queryParamEquals)) {
      const actual = parsed.searchParams.get(key);
      const expectedValues = normalizeExpectedValues(expectedRaw);
      if (!actual || !expectedValues.includes(actual)) return -1;
      score += 2;
    }
  }

  return score;
}

function getVideoSiteRoots() {
  return [PUBLIC_VIDEO_SITES_ROOT, PathManager.getPrivateVideoSitesPath()];
}

function loadAllSiteManifests(): LoadedVideoSiteManifest[] {
  const loadedById = new Map<string, LoadedVideoSiteManifest>();

  for (const rootPath of getVideoSiteRoots()) {
    const manifestDir = path.join(rootPath, 'sites');
    if (!fs.existsSync(manifestDir)) continue;

    const files = fs.readdirSync(manifestDir)
      .filter((file) => file.toLowerCase().endsWith('.json'))
      .sort((a, b) => a.localeCompare(b));

    for (const file of files) {
      const manifestPath = path.join(manifestDir, file);
      try {
        const manifest = validateManifest(fs.readJsonSync(manifestPath));
        if (manifest.enabled === false) continue;
        loadedById.set(manifest.id, {
          rootPath,
          manifestPath,
          manifest,
        });
      } catch (error: any) {
        console.warn(`[VideoSiteRegistry] Failed to load ${manifestPath}: ${error?.message || error}`);
      }
    }
  }

  return Array.from(loadedById.values());
}

function resolvePattern(rule: LoadedVideoSiteManifest, target: 'site' | 'playback') {
  if (target === 'playback') {
    return rule.manifest.playback?.match ?? rule.manifest.match;
  }
  return rule.manifest.match;
}

export function findMatchingSiteRule(
  rawUrl: string,
  target: 'site' | 'playback' = 'site'
): LoadedVideoSiteManifest | null {
  const candidates = loadAllSiteManifests()
    .map((rule) => {
      const pattern = resolvePattern(rule, target);
      return {
        rule,
        score: scoreMatch(rawUrl, pattern),
        priority: Number(rule.manifest.priority || 0),
        specificity: patternSpecificity(pattern),
      };
    })
    .filter((candidate) => candidate.score >= 0)
    .sort((a, b) => {
      return (
        b.priority - a.priority ||
        b.score - a.score ||
        b.specificity - a.specificity ||
        a.rule.manifest.id.localeCompare(b.rule.manifest.id)
      );
    });

  return candidates[0]?.rule || null;
}

export function buildYtDlpArgsForUrl(
  rawUrl: string,
  usage: 'metadata' | 'download',
  trailingArgs: string[] = []
) {
  const rule = findMatchingSiteRule(rawUrl, 'site');
  const args = buildBaseYtDlpArgs();
  const ytDlp = rule?.manifest.ytDlp;

  if (ytDlp?.impersonate) {
    args.push('--impersonate', ytDlp.impersonate);
  }
  for (const extractorArg of ytDlp?.extractorArgs || []) {
    args.push('--extractor-args', extractorArg);
  }
  args.push(...(ytDlp?.extraArgs || []));
  args.push(...(usage === 'metadata' ? ytDlp?.metadataExtraArgs || [] : ytDlp?.downloadExtraArgs || []));
  args.push(...trailingArgs);

  return { args, rule };
}

export function buildPlaybackHeadersForUrl(rawUrl: string) {
  const headers: Record<string, string> = {
    'User-Agent': VIDEO_WEB_UA,
    Accept: '*/*',
  };
  const rule = findMatchingSiteRule(rawUrl, 'playback');
  Object.assign(headers, rule?.manifest.playback?.headers || {});
  return headers;
}

export function buildDownloadHeadersForUrl(rawUrl: string) {
  const headers: Record<string, string> = {};
  const rule = findMatchingSiteRule(rawUrl, 'site');
  const referer = rule?.manifest.download?.referer || rawUrl;
  if (referer) {
    headers.Referer = referer;
  }
  Object.assign(headers, rule?.manifest.download?.headers || {});
  return { headers, rule };
}

export function detectParseFailureTriggers(logText: string): ParseFailureTrigger[] {
  const lower = String(logText || '').toLowerCase();
  const triggers: ParseFailureTrigger[] = [];

  if (lower.includes('no video formats found')) triggers.push('no_formats');
  if (lower.includes('unsupported url')) triggers.push('unsupported_url');
  if (lower.includes('403') || lower.includes('cloudflare')) triggers.push('cloudflare');
  if (
    lower.includes('sign in to confirm you') ||
    lower.includes('cookies-from-browser') ||
    lower.includes('not a bot')
  ) {
    triggers.push('bot_detection');
  }

  return Array.from(new Set(triggers));
}

export function shouldUseParseHandler(rule: LoadedVideoSiteManifest | null, triggers: ParseFailureTrigger[]) {
  if (!rule?.manifest.handlers?.parse) return false;
  const requiredTriggers = rule.manifest.parse?.fallbackOn || [];
  if (requiredTriggers.length === 0) return false;
  return requiredTriggers.some((trigger) => triggers.includes(trigger));
}

export function shouldUseDownloadHandler(rule: LoadedVideoSiteManifest | null, formatId: string) {
  if (!rule?.manifest.handlers?.download) return false;
  if (rule.manifest.download?.alwaysUseHandler) return true;
  const prefixes = rule.manifest.download?.customFormatPrefixes || [];
  return prefixes.some((prefix) => formatId.startsWith(prefix));
}

export async function loadSiteHandlerModule(
  rule: LoadedVideoSiteManifest,
  hook: 'parse' | 'download'
): Promise<VideoSiteHandlerModule | null> {
  const relativePath = rule.manifest.handlers?.[hook];
  if (!relativePath) return null;

  const modulePath = path.resolve(path.dirname(rule.manifestPath), relativePath);
  const relativeToRoot = path.relative(rule.rootPath, modulePath);
  if (relativeToRoot.startsWith('..') || path.isAbsolute(relativeToRoot)) {
    throw new Error(`Handler path escapes video_sites root: ${relativePath}`);
  }

  const stat = await fs.stat(modulePath);
  const specifier = `${pathToFileURL(modulePath).href}?v=${stat.mtimeMs}`;
  const imported = await import(specifier);
  const handler = imported.default ?? imported.handler ?? imported;

  if (!handler || typeof handler !== 'object') {
    throw new Error(`Handler module must export an object: ${modulePath}`);
  }

  return handler as VideoSiteHandlerModule;
}
