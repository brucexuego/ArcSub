import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs-extra';
import { PathManager } from '../path_manager.js';
import {
  buildDownloadHeadersForUrl,
  buildPlaybackHeadersForUrl,
  buildYtDlpArgsForUrl,
  detectParseFailureTriggers,
  loadSiteHandlerModule,
  shouldUseDownloadHandler,
  shouldUseParseHandler,
} from '../video_sites/registry.js';
import {
  YOUTUBE_FALLBACK_AUDIO_FORMAT_ID,
  getVideoTools,
  getCompatibleYtDlpArgs,
  markYtDlpChromeImpersonationUnsupported,
  parseYtDlpMetadataOutput,
  runYtDlp,
  shouldRetryWithoutChromeImpersonation,
  shouldRetryWithoutJsRuntimes,
  markYtDlpJsRuntimesUnsupported,
  stripUnsupportedImpersonationYtDlpArgs,
  stripUnsupportedYtDlpArgs,
} from '../video_sites/runtime.js';
import { ResourceManager } from './resource_manager.js';
import type {
  DownloadProgress,
  VideoDownloadType,
  VideoMetadataResult,
} from '../video_sites/types.js';

export type { DownloadProgress } from '../video_sites/types.js';

const DIRECT_MP4_VIDEO_CODECS = new Set(['h264', 'av1']);
const DIRECT_MP4_AUDIO_CODECS = new Set(['aac', 'mp3']);
const DIRECT_WEBM_VIDEO_CODECS = new Set(['vp8', 'vp9', 'av1']);
const DIRECT_WEBM_AUDIO_CODECS = new Set(['opus', 'vorbis']);

type BrowserPlaybackProbe = {
  formatName: string;
  videoCodec: string;
  audioCodec: string;
};

function attachMetadataDebug(
  metadata: VideoMetadataResult,
  siteRuleId: string | null,
  parseStrategy: string,
  extraDebug: Record<string, unknown> = {}
) {
  return {
    ...metadata,
    siteRuleId: siteRuleId || metadata.siteRuleId,
    debug: {
      ...(metadata.debug || {}),
      ...(siteRuleId ? { matchedSiteRule: siteRuleId } : {}),
      parseStrategy,
      ...extraDebug,
    },
  };
}

export class VideoService {
  private static videoToolsReadyTask: Promise<void> | null = null;

  private static ensureVideoToolsReady() {
    if (!this.videoToolsReadyTask) {
      this.videoToolsReadyTask = ResourceManager.ensureTools().catch((error) => {
        this.videoToolsReadyTask = null;
        throw error;
      });
    }
    return this.videoToolsReadyTask;
  }

  static buildPlaybackRequestHeaders(rawUrl: string) {
    return buildPlaybackHeadersForUrl(rawUrl);
  }

  static async parseMetadata(url: string) {
    await this.ensureVideoToolsReady();
    const { args, rule } = buildYtDlpArgsForUrl(url, 'metadata', ['--dump-single-json', url]);
    const result = await runYtDlp(args);
    if (result.code === 0) {
      return attachMetadataDebug(
        parseYtDlpMetadataOutput(result.stdout),
        rule?.manifest.id || null,
        'yt-dlp'
      );
    }

    const failureTriggers = detectParseFailureTriggers(`${result.stderr}\n${result.stdout}`);
    if (shouldUseParseHandler(rule, failureTriggers)) {
      const handler = await loadSiteHandlerModule(rule!, 'parse');
      if (handler?.parseMetadataFallback) {
        const metadata = await handler.parseMetadataFallback({
          url,
          rule: rule!,
          ytDlpResult: result,
          failureTriggers,
        });
        return attachMetadataDebug(
          metadata,
          rule!.manifest.id,
          `handler:${rule!.manifest.id}`,
          { failureTriggers }
        );
      }
    }

    throw new Error(`yt-dlp error: ${result.stderr || result.code}`);
  }

  static async downloadVideo(
    url: string,
    projectId: string,
    formatId: string,
    downloadType: VideoDownloadType,
    onProgress: (progress: DownloadProgress) => void
  ) {
    await this.ensureVideoToolsReady();
    const { ytdlp, ffmpeg } = getVideoTools();
    const assetsDir = path.join(PathManager.getProjectPath(projectId), 'assets');
    fs.ensureDirSync(assetsDir);

    try {
      const existingFiles = fs.readdirSync(assetsDir);
      for (const file of existingFiles) {
        const shouldRemove =
          downloadType === 'video'
            ? file.startsWith('source.')
            : file.startsWith('source_audio.') || file === 'audio.wav';

        if (shouldRemove) {
          fs.removeSync(path.join(assetsDir, file));
        }
      }
      console.log(`[VideoService] Cleaned up ${downloadType} assets in ${projectId}`);
    } catch (error) {
      console.warn(`[VideoService] Cleanup failed: ${error}`);
    }

    const normalizedFormatId = String(formatId || '').trim().replace(/[\r\n\\]+$/, '');
    const { headers: downloadHeaders, rule } = buildDownloadHeadersForUrl(url);

    if (shouldUseDownloadHandler(rule, normalizedFormatId)) {
      const handler = await loadSiteHandlerModule(rule!, 'download');
      if (!handler?.download) {
        throw new Error(`Download handler missing for site rule ${rule!.manifest.id}`);
      }
      return handler.download({
        url,
        projectId,
        assetsDir,
        formatId: normalizedFormatId,
        downloadType,
        onProgress,
        rule: rule!,
      });
    }

    const isDirectUrl = normalizedFormatId.startsWith('http');
    const { args } = buildYtDlpArgsForUrl(url, 'download');

    if (!isDirectUrl) {
      const wantsAudioFallbackVideo =
        rule?.manifest.id === 'youtube' &&
        downloadType === 'audio' &&
        normalizedFormatId === YOUTUBE_FALLBACK_AUDIO_FORMAT_ID;
      const isExplicitMergedFormat =
        normalizedFormatId.includes('+') || normalizedFormatId === '18' || normalizedFormatId.startsWith('best');
      const requestedFormat = wantsAudioFallbackVideo
        ? '18'
        : (normalizedFormatId.toLowerCase().includes('audio') || isExplicitMergedFormat
            ? normalizedFormatId
            : `${normalizedFormatId}+bestaudio/best`);
      args.push('-f', requestedFormat);
    }

    const outputPrefix = downloadType === 'audio' ? 'source_audio' : 'source';
    let downloadArgs = [
      ...getCompatibleYtDlpArgs(args),
      '--ffmpeg-location', ffmpeg,
      ...Object.entries(downloadHeaders).flatMap(([key, value]) => ['--add-header', `${key}:${value}`]),
      '-o', path.join(assetsDir, `${outputPrefix}.%(ext)s`),
      '--newline',
      isDirectUrl ? normalizedFormatId : url,
    ];

    const executeDownload = (currentArgs: string[]) =>
      new Promise<string>((resolve, reject) => {
        console.log(`[VideoService] Executing download: ${ytdlp} ${currentArgs.join(' ')}`);
        const proc = spawn(ytdlp, currentArgs);
        let stdout = '';
        let lastError = '';

        proc.stdout.on('data', (data) => {
          const line = data.toString();
          stdout += line;
          const match = line.match(/\[download\]\s+([\d.]+)%\s+of\s+[\d.A-Za-z]+\s+at\s+([\d.A-Za-z/]+)\s+ETA\s+([\d:]+)/);
          if (match) {
            onProgress({
              status: 'downloading',
              progress: parseFloat(match[1]),
              speed: match[2],
              eta: match[3],
            });
          } else {
            const simpleMatch = line.match(/\[download\]\s+([\d.]+)%/);
            if (simpleMatch) {
              onProgress({ status: 'downloading', progress: parseFloat(simpleMatch[1]) });
            }
          }
        });

        proc.stderr.on('data', (data) => {
          lastError += data.toString();
        });

        proc.on('close', async (code) => {
          if (code !== 0) {
            if (shouldRetryWithoutJsRuntimes(currentArgs, stdout, lastError)) {
              markYtDlpJsRuntimesUnsupported();
              console.warn('[VideoService] yt-dlp download does not support --js-runtimes. Retrying without it.');
              try {
                resolve(await executeDownload(stripUnsupportedYtDlpArgs(currentArgs)));
                return;
              } catch (retryError) {
                reject(retryError as Error);
                return;
              }
            }
            if (shouldRetryWithoutChromeImpersonation(currentArgs, stdout, lastError)) {
              markYtDlpChromeImpersonationUnsupported();
              console.warn('[VideoService] yt-dlp download does not support --impersonate chrome. Retrying without it.');
              try {
                resolve(await executeDownload(stripUnsupportedImpersonationYtDlpArgs(currentArgs)));
                return;
              } catch (retryError) {
                reject(retryError as Error);
                return;
              }
            }
            console.error(`[VideoService] Download failed. Stderr: ${lastError}`);
            return reject(new Error(`Download failed with code ${code}. ${lastError.split('\n').pop()}`));
          }
          const files = await fs.readdir(assetsDir);
          const videoFile = files.find((file) => file.startsWith(`${outputPrefix}.`) && !file.endsWith('.part'));
          if (!videoFile) return reject(new Error('Downloaded file not found'));
          resolve(path.join(assetsDir, videoFile));
        });
      });

    return executeDownload(downloadArgs);
  }

  static async extractAudio(videoPath: string, projectId: string): Promise<string> {
    await this.ensureVideoToolsReady();
    const { ffmpeg } = getVideoTools();
    const audioPath = path.join(PathManager.getProjectPath(projectId), 'assets', 'audio.wav');

    return new Promise((resolve, reject) => {
      const proc = spawn(ffmpeg, [
        '-i', videoPath,
        '-ar', '16000',
        '-ac', '1',
        '-c:a', 'pcm_s16le',
        '-y',
        audioPath,
      ]);

      proc.on('close', (code) => {
        if (code !== 0) return reject(new Error('Audio extraction failed'));
        resolve(audioPath);
      });
    });
  }

  private static async inspectBrowserPlaybackCompatibility(videoPath: string): Promise<BrowserPlaybackProbe> {
    await this.ensureVideoToolsReady();
    const { ffmpeg } = getVideoTools();

    return new Promise((resolve, reject) => {
      const proc = spawn(ffmpeg, ['-hide_banner', '-i', videoPath]);
      let stderr = '';

      proc.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      proc.on('close', () => {
        const formatMatch = stderr.match(/Input #0,\s*([^,]+(?:,[^,]+)*)\s*,\s*from\b/i);
        const videoCodecMatch = stderr.match(/Stream #\d+:\d+(?:\[[^\]]+\])?(?:\([^)]+\))?: Video:\s*([^,\s]+)/i);
        const audioCodecMatch = stderr.match(/Stream #\d+:\d+(?:\[[^\]]+\])?(?:\([^)]+\))?: Audio:\s*([^,\s]+)/i);

        const formatName = String(formatMatch?.[1] || '').trim().toLowerCase();
        const videoCodec = String(videoCodecMatch?.[1] || '').trim().toLowerCase();
        const audioCodec = String(audioCodecMatch?.[1] || '').trim().toLowerCase();

        if (!formatName || !videoCodec) {
          reject(new Error(`Unable to inspect media format for ${path.basename(videoPath)}`));
          return;
        }

        resolve({ formatName, videoCodec, audioCodec });
      });
    });
  }

  private static isDirectBrowserPlayable(probe: BrowserPlaybackProbe) {
    const format = probe.formatName;
    const videoCodec = probe.videoCodec;
    const audioCodec = probe.audioCodec;

    const isMp4Family = /(mov|mp4|m4a|3gp|3g2|mj2)/i.test(format);
    if (isMp4Family) {
      const videoOk = DIRECT_MP4_VIDEO_CODECS.has(videoCodec);
      const audioOk = !audioCodec || DIRECT_MP4_AUDIO_CODECS.has(audioCodec);
      return videoOk && audioOk;
    }

    const isWebm = /\bwebm\b/i.test(format);
    if (isWebm) {
      const videoOk = DIRECT_WEBM_VIDEO_CODECS.has(videoCodec);
      const audioOk = !audioCodec || DIRECT_WEBM_AUDIO_CODECS.has(audioCodec);
      return videoOk && audioOk;
    }

    return false;
  }

  private static shouldAttemptMp4Remux(probe: BrowserPlaybackProbe) {
    const videoCodec = probe.videoCodec;
    const audioCodec = probe.audioCodec;
    const videoOk = DIRECT_MP4_VIDEO_CODECS.has(videoCodec);
    const audioOk = !audioCodec || DIRECT_MP4_AUDIO_CODECS.has(audioCodec);
    return videoOk && audioOk;
  }

  static async ensureBrowserPlayableVideo(videoPath: string, projectId: string): Promise<string> {
    await this.ensureVideoToolsReady();
    const { ffmpeg } = getVideoTools();
    const assetsDir = path.join(PathManager.getProjectPath(projectId), 'assets');
    const outputPath = path.join(assetsDir, 'source.mp4');

    await fs.ensureDir(assetsDir);

    const probe = await this.inspectBrowserPlaybackCompatibility(videoPath);
    if (this.isDirectBrowserPlayable(probe)) {
      return videoPath;
    }

    const runFfmpeg = (args: string[]) =>
      new Promise<void>((resolve, reject) => {
        const proc = spawn(ffmpeg, args);
        let stderr = '';

        proc.stderr.on('data', (data) => {
          stderr += data.toString();
        });

        proc.on('close', (code) => {
          if (code === 0) {
            resolve();
            return;
          }
          reject(new Error(stderr.trim() || `ffmpeg exited with code ${code}`));
        });
      });

    const remuxArgs = [
      '-hide_banner',
      '-i', videoPath,
      '-map', '0:v:0',
      '-map', '0:a?',
      '-c', 'copy',
      '-movflags', '+faststart',
      '-y',
      outputPath,
    ];

    if (this.shouldAttemptMp4Remux(probe)) {
      try {
        await runFfmpeg(remuxArgs);
        return outputPath;
      } catch (remuxError) {
        console.warn(`[VideoService] MP4 remux failed for ${path.basename(videoPath)}. Falling back to transcode.`, remuxError);
      }
    }

    const transcodeArgs = [
      '-hide_banner',
      '-i', videoPath,
      '-map', '0:v:0',
      '-map', '0:a?',
      '-c:v', 'libx264',
      '-preset', 'veryfast',
      '-crf', '23',
      '-pix_fmt', 'yuv420p',
      '-c:a', 'aac',
      '-b:a', '192k',
      '-movflags', '+faststart',
      '-y',
      outputPath,
    ];

    await runFfmpeg(transcodeArgs);
    return outputPath;
  }
}
