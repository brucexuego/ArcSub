import fs from 'node:fs';
import path from 'node:path';
import { PathManager } from '../../path_manager.js';
import type { ApiModelRequestOptions } from '../../../src/types.js';
import type {
  CloudTranslateProvider,
  CloudTranslateQuotaProfile,
  ResolvedCloudTranslateProvider,
} from './types.js';

interface QuotaBucket {
  minuteStartedAt: number;
  dayStartedAt: number;
  minuteRequests: number;
  minuteTokens: number;
  dayRequests: number;
  active: number;
  blockedUntilMs?: number;
  lastRateLimitReason?: string | null;
}

interface QuotaPersistedState {
  version: 1;
  updatedAt: string;
  buckets: Record<string, Omit<QuotaBucket, 'active'>>;
}

const quotaBuckets = new Map<string, QuotaBucket>();
let loadedFromDisk = false;

function nowMs() {
  return Date.now();
}

function isPersistenceEnabled() {
  const raw = String(process.env.TRANSLATE_CLOUD_QUOTA_PERSIST || '').trim();
  if (!raw) return true;
  return /^(1|true|yes|on)$/i.test(raw);
}

function getStatePath() {
  return path.join(PathManager.getRuntimePath(), 'cache', 'cloud_translate_quota_state.json');
}

function loadState() {
  if (loadedFromDisk || !isPersistenceEnabled()) return;
  loadedFromDisk = true;
  try {
    const statePath = getStatePath();
    if (!fs.existsSync(statePath)) return;
    const parsed = JSON.parse(fs.readFileSync(statePath, 'utf8')) as QuotaPersistedState;
    if (!parsed || parsed.version !== 1 || !parsed.buckets) return;
    for (const [key, persisted] of Object.entries(parsed.buckets)) {
      quotaBuckets.set(key, {
        ...persisted,
        active: 0,
      });
    }
  } catch {
    // Quota persistence is best-effort. A corrupt state file should not block translation.
  }
}

function persistState() {
  if (!isPersistenceEnabled()) return;
  try {
    const statePath = getStatePath();
    fs.mkdirSync(path.dirname(statePath), { recursive: true });
    const buckets: QuotaPersistedState['buckets'] = {};
    for (const [key, bucket] of quotaBuckets.entries()) {
      const { active: _active, ...persisted } = bucket;
      buckets[key] = persisted;
    }
    fs.writeFileSync(
      statePath,
      JSON.stringify(
        {
          version: 1,
          updatedAt: new Date().toISOString(),
          buckets,
        } satisfies QuotaPersistedState,
        null,
        2
      )
    );
  } catch {
    // Best-effort only.
  }
}

function getBucket(key: string) {
  loadState();
  const now = nowMs();
  let bucket = quotaBuckets.get(key);
  if (!bucket) {
    bucket = {
      minuteStartedAt: now,
      dayStartedAt: now,
      minuteRequests: 0,
      minuteTokens: 0,
      dayRequests: 0,
      active: 0,
      blockedUntilMs: 0,
      lastRateLimitReason: null,
    };
    quotaBuckets.set(key, bucket);
  }
  return bucket;
}

function resetWindows(bucket: QuotaBucket) {
  const now = nowMs();
  if (now - bucket.minuteStartedAt >= 60_000) {
    bucket.minuteStartedAt = now;
    bucket.minuteRequests = 0;
    bucket.minuteTokens = 0;
  }
  if (now - bucket.dayStartedAt >= 86_400_000) {
    bucket.dayStartedAt = now;
    bucket.dayRequests = 0;
  }
  if (bucket.blockedUntilMs && bucket.blockedUntilMs <= now) {
    bucket.blockedUntilMs = 0;
    bucket.lastRateLimitReason = null;
  }
}

function estimateTokens(text: string, profile: CloudTranslateQuotaProfile, maxOutputTokens?: number | null) {
  const value = String(text || '');
  const cjkChars = (value.match(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu) || [])
    .length;
  const latinLikeChars = Math.max(0, value.replace(/\s+/g, ' ').length - cjkChars);
  const estimator = profile.tokenEstimator || 'chars_heuristic';
  const inputTokens =
    estimator === 'openai_like'
      ? Math.max(1, Math.ceil(value.replace(/\s+/g, ' ').length / 4))
      : estimator === 'gemini_like'
        ? Math.max(1, Math.ceil(cjkChars + latinLikeChars / 4))
        : Math.max(1, Math.ceil(value.length / 4));
  const reserve = Number(profile.safetyReserveTokens);
  const outputTokens =
    Number.isFinite(Number(maxOutputTokens)) && Number(maxOutputTokens) > 0
      ? Math.round(Number(maxOutputTokens))
      : Number.isFinite(reserve) && reserve > 0
        ? Math.round(reserve)
        : Math.ceil(inputTokens * 1.35) + 64;
  return { inputTokens, estimatedTotalTokens: inputTokens + outputTokens };
}

function delay(ms: number, signal?: AbortSignal) {
  if (ms <= 0) return Promise.resolve();
  return new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    const onAbort = () => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
      reject(new Error('Translation request aborted while waiting for provider quota window.'));
    };
    if (signal) {
      if (signal.aborted) {
        onAbort();
        return;
      }
      signal.addEventListener('abort', onAbort, { once: true });
    }
  });
}

function quotaKey(provider: CloudTranslateProvider, model: string, key?: string) {
  const keyTail = String(key || '').slice(-8);
  return `${provider}:${String(model || '').trim() || 'default'}:${keyTail}`;
}

function getHeader(headers: Record<string, string> | undefined, names: string[]) {
  if (!headers) return '';
  const entries = Object.entries(headers);
  for (const name of names) {
    const found = entries.find(([key]) => key.toLowerCase() === name.toLowerCase());
    if (found) return String(found[1] || '').trim();
  }
  return '';
}

function parseResetMs(value: string) {
  const raw = String(value || '').trim();
  if (!raw) return null;
  const numeric = Number(raw);
  if (!Number.isFinite(numeric)) return null;
  if (numeric > 1_000_000_000_000) return Math.max(0, Math.round(numeric - Date.now()));
  if (numeric > 1_000_000_000) return Math.max(0, Math.round(numeric * 1000 - Date.now()));
  return Math.max(0, Math.round(numeric * 1000));
}

function getBucketForResolved(input: {
  resolvedProvider: ResolvedCloudTranslateProvider;
  key?: string;
}) {
  return getBucket(quotaKey(input.resolvedProvider.provider, input.resolvedProvider.effectiveModel, input.key));
}

export function recordCloudTranslateQuotaBackoff(input: {
  resolvedProvider: ResolvedCloudTranslateProvider;
  key?: string;
  retryAfterMs?: number | null;
  fallbackMs?: number;
  reason?: string;
}) {
  const waitMs = Math.max(0, Math.round(input.retryAfterMs || input.fallbackMs || 0));
  if (!waitMs) return;
  const bucket = getBucketForResolved(input);
  bucket.blockedUntilMs = Math.max(bucket.blockedUntilMs || 0, nowMs() + waitMs);
  bucket.lastRateLimitReason = input.reason || 'provider_429';
  persistState();
}

export function recordCloudTranslateRateLimitHeaders(input: {
  resolvedProvider: ResolvedCloudTranslateProvider;
  key?: string;
  headers?: Record<string, string>;
}) {
  const remaining = Number(
    getHeader(input.headers, [
      'x-ratelimit-remaining-requests',
      'x-ratelimit-remaining',
      'ratelimit-remaining',
    ])
  );
  const resetMs = parseResetMs(
    getHeader(input.headers, [
      'x-ratelimit-reset-requests',
      'x-ratelimit-reset',
      'ratelimit-reset',
    ])
  );
  if (!Number.isFinite(remaining) || remaining > 0 || !resetMs) return;
  const bucket = getBucketForResolved(input);
  bucket.blockedUntilMs = Math.max(bucket.blockedUntilMs || 0, nowMs() + resetMs);
  bucket.lastRateLimitReason = 'provider_rate_limit_headers';
  persistState();
}

export async function enforceCloudTranslateQuotaLimit(input: {
  resolvedProvider: ResolvedCloudTranslateProvider;
  key?: string;
  text: string;
  modelOptions?: ApiModelRequestOptions;
  signal?: AbortSignal;
  onProgress?: (message: string) => void;
}) {
  const profile: CloudTranslateQuotaProfile | null = input.resolvedProvider.quotaProfile;
  if (!profile) {
    return {
      estimatedInputTokens: null,
      estimatedTotalTokens: null,
      waitedMs: 0,
      waitReason: null,
      waitEvents: [],
      applied: false,
      profileId: null,
      tokenEstimator: null,
    };
  }

  const samplingMaxOutputTokens = input.modelOptions?.sampling?.maxOutputTokens;
  const { inputTokens, estimatedTotalTokens } = estimateTokens(input.text, profile, samplingMaxOutputTokens || null);
  const key = quotaKey(input.resolvedProvider.provider, input.resolvedProvider.effectiveModel, input.key);
  const bucket = getBucket(key);
  let waitedMs = 0;
  let waitReason: string | null = null;
  const waitEvents: Array<{ reason: string; waitedMs: number }> = [];

  while (true) {
    resetWindows(bucket);
    const now = nowMs();
    const blockedWaitMs = bucket.blockedUntilMs && bucket.blockedUntilMs > now ? bucket.blockedUntilMs - now : 0;
    const minuteWaitMs =
      (profile.rpm && bucket.minuteRequests + 1 > profile.rpm) ||
      (profile.tpm && bucket.minuteTokens + estimatedTotalTokens > profile.tpm)
        ? Math.max(0, 60_000 - (now - bucket.minuteStartedAt))
        : 0;
    const concurrencyWaitMs =
      profile.maxConcurrency && bucket.active >= profile.maxConcurrency ? 250 : 0;
    const waitMs = Math.max(blockedWaitMs, minuteWaitMs, concurrencyWaitMs);
    if (!waitMs) break;
    const reason =
      blockedWaitMs === waitMs
        ? bucket.lastRateLimitReason || 'provider_rate_limit'
        : minuteWaitMs === waitMs
          ? 'local_rpm_tpm_window'
          : 'local_max_concurrency';
    waitReason = reason;
    waitedMs += waitMs;
    waitEvents.push({ reason, waitedMs: waitMs });
    input.onProgress?.(`Waiting for cloud translation quota window (${Math.ceil(waitMs / 1000)}s, ${reason})...`);
    await delay(waitMs, input.signal);
  }

  resetWindows(bucket);
  if (profile.rpd && bucket.dayRequests + 1 > profile.rpd) {
    throw new Error(
      `Cloud translation daily request limit reached locally (${profile.rpd} RPD for ${input.resolvedProvider.effectiveModel}).`
    );
  }

  bucket.minuteRequests += 1;
  bucket.minuteTokens += estimatedTotalTokens;
  bucket.dayRequests += 1;
  bucket.active += 1;
  persistState();

  return {
    estimatedInputTokens: inputTokens,
    estimatedTotalTokens,
    waitedMs,
    waitReason,
    waitEvents,
    applied: true,
    profileId: profile.profileId || null,
    tokenEstimator: profile.tokenEstimator || null,
    release: () => {
      bucket.active = Math.max(0, bucket.active - 1);
      persistState();
    },
  };
}
