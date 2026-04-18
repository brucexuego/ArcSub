import { spawn } from 'child_process';
import fs from 'fs-extra';
import { pipeline } from 'stream/promises';
import { Readable } from 'stream';
import { resolveToolCommand } from '../runtime_tools.js';
import type { VideoMetadataResult } from './types.js';

export const VIDEO_WEB_UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 ArcSub/1.0';
export const YOUTUBE_FALLBACK_AUDIO_FORMAT_ID = '__youtube_fallback_audio__';
let ytDlpSupportsJsRuntimes: boolean | null = null;
let ytDlpSupportsChromeImpersonation: boolean | null = null;

export function getVideoTools() {
  return {
    ytdlp: resolveToolCommand('yt-dlp'),
    ffmpeg: resolveToolCommand('ffmpeg'),
  };
}

export function buildBaseYtDlpArgs(extraArgs: string[] = []) {
  const args: string[] = [
    '--no-check-certificate',
    '--no-warnings',
    '--geo-bypass',
  ];
  if (process.execPath && ytDlpSupportsJsRuntimes !== false) {
    args.push('--js-runtimes', `node:${process.execPath}`);
  }
  return [...args, ...extraArgs];
}

function stripUnsupportedJsRuntimeArgs(args: string[]) {
  const sanitized: string[] = [];
  for (let index = 0; index < args.length; index += 1) {
    const current = String(args[index] || '');
    if (current === '--js-runtimes') {
      index += 1;
      continue;
    }
    sanitized.push(current);
  }
  return sanitized;
}

function stripUnsupportedImpersonationArgs(args: string[]) {
  const sanitized: string[] = [];
  for (let index = 0; index < args.length; index += 1) {
    const current = String(args[index] || '');
    if (current === '--impersonate') {
      index += 1;
      continue;
    }
    sanitized.push(current);
  }
  return sanitized;
}

function hasJsRuntimeArgs(args: string[]) {
  return args.some((item) => String(item || '').trim() === '--js-runtimes');
}

function hasChromeImpersonationArgs(args: string[]) {
  for (let index = 0; index < args.length; index += 1) {
    if (String(args[index] || '').trim() !== '--impersonate') continue;
    const target = String(args[index + 1] || '').trim().toLowerCase();
    if (target === 'chrome') return true;
  }
  return false;
}

function isUnsupportedJsRuntimeError(stdout: string, stderr: string) {
  const combined = `${stdout || ''}\n${stderr || ''}`.toLowerCase();
  return combined.includes('no such option: --js-runtimes');
}

function isUnsupportedChromeImpersonationError(stdout: string, stderr: string) {
  const combined = `${stdout || ''}\n${stderr || ''}`.toLowerCase();
  return combined.includes('impersonate target "chrome" is not available') ||
    combined.includes("no such option: --impersonate");
}

function prepareYtDlpArgs(args: string[]) {
  let effectiveArgs = [...args];
  if (ytDlpSupportsJsRuntimes === false) {
    effectiveArgs = stripUnsupportedJsRuntimeArgs(effectiveArgs);
  }
  if (ytDlpSupportsChromeImpersonation === false) {
    effectiveArgs = stripUnsupportedImpersonationArgs(effectiveArgs);
  }
  return effectiveArgs;
}

export function getCompatibleYtDlpArgs(args: string[]) {
  return prepareYtDlpArgs(args);
}

export function shouldRetryWithoutJsRuntimes(args: string[], stdout: string, stderr: string) {
  return ytDlpSupportsJsRuntimes !== false && hasJsRuntimeArgs(args) && isUnsupportedJsRuntimeError(stdout, stderr);
}

export function markYtDlpJsRuntimesUnsupported() {
  ytDlpSupportsJsRuntimes = false;
}

export function stripUnsupportedYtDlpArgs(args: string[]) {
  return stripUnsupportedJsRuntimeArgs(args);
}

export function shouldRetryWithoutChromeImpersonation(args: string[], stdout: string, stderr: string) {
  return ytDlpSupportsChromeImpersonation !== false &&
    hasChromeImpersonationArgs(args) &&
    isUnsupportedChromeImpersonationError(stdout, stderr);
}

export function markYtDlpChromeImpersonationUnsupported() {
  ytDlpSupportsChromeImpersonation = false;
}

export function stripUnsupportedImpersonationYtDlpArgs(args: string[]) {
  return stripUnsupportedImpersonationArgs(args);
}

export async function runYtDlp(
  args: string[],
  retryCount = 0
): Promise<{ code: number; stdout: string; stderr: string }> {
  const { ytdlp } = getVideoTools();
  const jitter = Math.floor(Math.random() * 600) + 200;
  await new Promise((resolve) => setTimeout(resolve, jitter));

  return new Promise<{ code: number; stdout: string; stderr: string }>((resolve) => {
    const effectiveArgs = prepareYtDlpArgs(args);
    const proc = spawn(ytdlp, effectiveArgs);
    let stdout = '';
    let stderr = '';
    proc.stdout.on('data', (d) => (stdout += d.toString()));
    proc.stderr.on('data', (d) => (stderr += d.toString()));
    proc.on('close', async (code) => {
      if (
        code !== 0 &&
        ytDlpSupportsJsRuntimes !== false &&
        hasJsRuntimeArgs(effectiveArgs) &&
        isUnsupportedJsRuntimeError(stdout, stderr)
      ) {
        ytDlpSupportsJsRuntimes = false;
        console.warn('[VideoService] yt-dlp does not support --js-runtimes. Retrying without it.');
        resolve(await runYtDlp(stripUnsupportedJsRuntimeArgs(effectiveArgs), retryCount));
        return;
      }
      if (
        code !== 0 &&
        ytDlpSupportsChromeImpersonation !== false &&
        hasChromeImpersonationArgs(effectiveArgs) &&
        isUnsupportedChromeImpersonationError(stdout, stderr)
      ) {
        ytDlpSupportsChromeImpersonation = false;
        console.warn('[VideoService] yt-dlp does not support --impersonate chrome. Retrying without it.');
        resolve(await runYtDlp(stripUnsupportedImpersonationArgs(effectiveArgs), retryCount));
        return;
      }
      const fullLog = `${stdout}\n${stderr}`.toLowerCase();
      if ((fullLog.includes('403') || fullLog.includes('429')) && retryCount < 2 && code !== 0) {
        const waitTime = (retryCount + 1) * 2000;
        console.warn(`[VideoService] Site restriction detected, retrying in ${waitTime}ms... (Attempt ${retryCount + 1})`);
        await new Promise((resolveDelay) => setTimeout(resolveDelay, waitTime));
        resolve(await runYtDlp(args, retryCount + 1));
        return;
      }
      resolve({ code: code ?? 1, stdout, stderr });
    });
  });
}

export function parseYtDlpMetadataOutput(stdout: string): VideoMetadataResult {
  const jsonStart = stdout.indexOf('{');
  if (jsonStart === -1) throw new Error('No JSON found in yt-dlp output');
  const metadata = JSON.parse(stdout.substring(jsonStart).trim());
  const allFormats = (metadata.formats || []).map((f: any) => ({
    id: String(f.format_id ?? ''),
    quality: f.format_note || f.resolution || 'Unknown',
    size: Number(f.filesize || f.filesize_approx || 0),
    ext: f.ext,
    url: String(f.url || ''),
    vcodec: f.vcodec,
    acodec: f.acodec,
  }));

  return {
    title: metadata.title,
    description: metadata.description || metadata.info_dict?.description || '',
    uploader: metadata.uploader || metadata.channel || '',
    viewCount: metadata.view_count || metadata.views || 0,
    uploadDate: metadata.upload_date || '',
    duration: metadata.duration_string,
    thumbnail: metadata.thumbnail,
    formats: allFormats
      .filter((f: any) => f.vcodec !== 'none')
      .sort((a: any, b: any) => (b.size || 0) - (a.size || 0)),
    audioFormats: allFormats
      .filter((f: any) => f.vcodec === 'none' && f.acodec !== 'none')
      .sort((a: any, b: any) => (b.size || 0) - (a.size || 0)),
  };
}

export function secondsToDurationString(totalSeconds: number) {
  const safe = Math.max(0, Math.floor(Number(totalSeconds || 0)));
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const seconds = safe % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

export async function downloadFileWithFetch(
  resourceUrl: string,
  outPath: string,
  headers: Record<string, string>,
  onProgress?: (ratio: number) => void
) {
  const response = await fetch(resourceUrl, { headers, redirect: 'follow' });
  if (!response.ok || !response.body) {
    throw new Error(`Resource download failed (${response.status})`);
  }

  const total = Number(response.headers.get('content-length') || 0);
  const nodeReadable = Readable.fromWeb(response.body as any);
  let downloaded = 0;
  nodeReadable.on('data', (chunk: Buffer) => {
    downloaded += chunk.length;
    if (onProgress && total > 0) {
      onProgress(Math.min(1, downloaded / total));
    }
  });

  await pipeline(nodeReadable, fs.createWriteStream(outPath));
  if (onProgress) onProgress(1);
}

export async function runFfmpeg(args: string[]) {
  const { ffmpeg } = getVideoTools();
  return new Promise<void>((resolve, reject) => {
    const proc = spawn(ffmpeg, args);
    let stderr = '';
    proc.stderr.on('data', (d) => (stderr += d.toString()));
    proc.on('close', (code) => {
      if (code === 0) return resolve();
      reject(new Error(`ffmpeg failed (${code}): ${stderr || 'unknown error'}`));
    });
  });
}
