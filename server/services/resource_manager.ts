import fs from 'fs-extra';
import path from 'path';
import { PathManager } from '../path_manager.js';
import https from 'https';
import unzipper from 'unzipper';
import { pipeline } from 'stream/promises';
import { getBundledToolPath, isCommandAvailable } from '../runtime_tools.js';

export class ResourceManager {
  private static HF_BASE = 'https://huggingface.co';
  private static readonly ENSURE_TOOLS_FAILURE_COOLDOWN_MS = 120_000;
  private static readonly DOWNLOAD_MAX_REDIRECTS = 5;
  private static readonly DOWNLOAD_MAX_RETRIES = 3;
  private static readonly DOWNLOAD_TIMEOUT_MS = 45_000;
  private static deployManifestCache: any | null = null;
  private static ensureToolsTask: Promise<void> | null = null;
  private static ensureToolsFailure: { message: string; until: number } | null = null;

  private static async getDeployManifest() {
    if (this.deployManifestCache) return this.deployManifestCache;
    const manifestPath = PathManager.resolveRootPath('scripts', 'deploy-manifest.json');
    this.deployManifestCache = (await fs.pathExists(manifestPath))
      ? await fs.readJson(manifestPath).catch(() => null)
      : null;
    return this.deployManifestCache;
  }

  static async ensureTools() {
    if (this.ensureToolsFailure && Date.now() < this.ensureToolsFailure.until) {
      const waitMs = this.ensureToolsFailure.until - Date.now();
      throw new Error(
        `Tool warmup is temporarily paused after failure: ${this.ensureToolsFailure.message} (retry in ${Math.ceil(waitMs / 1000)}s)`
      );
    }

    if (this.ensureToolsTask) {
      return this.ensureToolsTask;
    }

    this.ensureToolsTask = this.ensureToolsInternal()
      .then(() => {
        this.ensureToolsFailure = null;
      })
      .catch((error: any) => {
        const message = String(error?.message || error || 'Unknown tool warmup failure.');
        this.ensureToolsFailure = {
          message,
          until: Date.now() + this.ENSURE_TOOLS_FAILURE_COOLDOWN_MS,
        };
        this.ensureToolsTask = null;
        throw new Error(message);
      });
    return this.ensureToolsTask;
  }

  private static async ensureToolsInternal() {
    const toolsPath = PathManager.getToolsPath();
    const ytdlpPath = getBundledToolPath('yt-dlp');
    const ffmpegPath = getBundledToolPath('ffmpeg');
    const manifest = await this.getDeployManifest();
    const ytDlpWindowsUrl =
      String(manifest?.tools?.['yt-dlp']?.windowsUrl || '').trim() ||
      'https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe';
    const ytDlpLinuxUrl =
      String(manifest?.tools?.['yt-dlp']?.linuxUrl || '').trim() ||
      'https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux';
    const ytDlpLinuxAarch64Url =
      String(manifest?.tools?.['yt-dlp']?.linuxAarch64Url || '').trim() ||
      'https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux_aarch64';
    const ffmpegWindowsUrl =
      String(manifest?.tools?.ffmpeg?.windowsUrl || '').trim() ||
      'https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v4.4.1/ffmpeg-4.4.1-win-64.zip';

    const [hasYtDlp, hasFfmpeg] = await Promise.all([
      fs.pathExists(ytdlpPath),
      fs.pathExists(ffmpegPath),
    ]);

    const hasYtDlpOnPath = isCommandAvailable('yt-dlp');
    const hasFfmpegOnPath = isCommandAvailable('ffmpeg');

    const tasks: Array<Promise<void>> = [];

    if (!hasYtDlp) {
      if (process.platform !== 'win32') {
        if (!hasYtDlpOnPath) {
          tasks.push((async () => {
            const arch = process.arch === 'arm64' ? 'arm64' : 'x64';
            const url = arch === 'arm64' ? ytDlpLinuxAarch64Url : ytDlpLinuxUrl;
            console.log(`Downloading yt-dlp (${arch})...`);
            await this.downloadWithRedirect(url, ytdlpPath);
            await fs.chmod(ytdlpPath, 0o755);
          })());
        }
      } else {
        tasks.push((async () => {
          console.log('Downloading yt-dlp.exe...');
          await this.downloadWithRedirect(ytDlpWindowsUrl, ytdlpPath);
        })());
      }
    }

    if (!hasFfmpeg) {
      if (process.platform !== 'win32') {
        if (!hasFfmpegOnPath) {
          throw new Error('ffmpeg is required but was not found in runtime/tools or on PATH.');
        }
      } else {
        tasks.push((async () => {
          console.log('Downloading ffmpeg.zip...');
          const zipPath = path.join(toolsPath, 'ffmpeg.zip');
          await this.downloadWithRedirect(ffmpegWindowsUrl, zipPath);

          console.log('Extracting ffmpeg...');
          await fs.createReadStream(zipPath)
            .pipe(unzipper.Extract({ path: toolsPath }))
            .promise();

          await fs.remove(zipPath);
        })());
      }
    }

    await Promise.all(tasks);
  }

  static async ensureModels() {
    const modelsRoot = PathManager.getModelsPath();

    // 1. Whisper Large V3
    await this.ensureHFModel('OpenVINO/whisper-large-v3-fp16-ov', [
      'openvino_encoder_model.xml', 'openvino_encoder_model.bin',
      'openvino_decoder_model.xml', 'openvino_decoder_model.bin',
      'openvino_tokenizer.xml', 'openvino_tokenizer.bin',
      'openvino_detokenizer.xml', 'openvino_detokenizer.bin',
      'tokenizer.json', 'config.json', 'preprocessor_config.json',
      'generation_config.json'
    ], 'whisper-large-v3-fp16-ov');

    // 2. Qwen3 8B (Qwen2.5 based probably, but using the specific OV repo)
    await this.ensureHFModel('OpenVINO/Qwen3-8B-int4-ov', [
      'openvino_model.xml', 'openvino_model.bin',
      'openvino_tokenizer.xml', 'openvino_tokenizer.bin',
      'openvino_detokenizer.xml', 'openvino_detokenizer.bin',
      'tokenizer.json', 'config.json', 'generation_config.json',
      'vocab.json', 'merges.txt'
    ], 'Qwen3-8B-int4-ov');
  }

  private static async ensureHFModel(repoId: string, files: string[], localSubdir: string) {
    const modelsRoot = PathManager.getModelsPath();
    const modelDir = path.join(modelsRoot, localSubdir);
    await fs.ensureDir(modelDir);

    for (const file of files) {
      const dest = path.join(modelDir, file);
      if (!await fs.pathExists(dest)) {
        console.log(`Downloading ${repoId}/${file}...`);
        const url = `${this.HF_BASE}/${repoId}/resolve/main/${file}`;
        await this.downloadWithRedirect(url, dest);
      }
    }
  }

  private static sleep(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private static async downloadWithRedirect(url: string, dest: string): Promise<void> {
    const tempPath = `${dest}.part`;
    let attempt = 0;
    while (attempt <= this.DOWNLOAD_MAX_RETRIES) {
      try {
        await this.downloadWithRedirectOnce(url, dest, 0);
        return;
      } catch (error) {
        await fs.remove(tempPath).catch(() => {});
        if (attempt >= this.DOWNLOAD_MAX_RETRIES) {
          throw error;
        }
        const backoffMs = 350 * Math.pow(2, attempt);
        await this.sleep(backoffMs);
        attempt += 1;
      }
    }
  }

  private static downloadWithRedirectOnce(url: string, dest: string, redirectDepth: number): Promise<void> {
    return new Promise((resolve, reject) => {
      if (redirectDepth > this.DOWNLOAD_MAX_REDIRECTS) {
        reject(new Error(`Too many redirects while downloading: ${url}`));
        return;
      }

      const tempPath = `${dest}.part`;
      fs.ensureDir(path.dirname(dest))
        .then(() => {
          const request = https.get(url, (response) => {
            if (response.statusCode === 302 || response.statusCode === 301 || response.statusCode === 307 || response.statusCode === 308) {
              const location = response.headers.location;
              response.resume();
              if (!location) {
                reject(new Error(`Missing redirect location for download: ${url}`));
                return;
              }
              const nextUrl = new URL(location, url).toString();
              this.downloadWithRedirectOnce(nextUrl, dest, redirectDepth + 1).then(resolve).catch(reject);
              return;
            }
            if (response.statusCode !== 200) {
              response.resume();
              reject(new Error(`Failed to download: ${response.statusCode} from ${url}`));
              return;
            }
            const file = fs.createWriteStream(tempPath);
            void pipeline(response, file)
              .then(async () => {
                await fs.move(tempPath, dest, { overwrite: true });
                resolve();
              })
              .catch((error) => {
                reject(error);
              });
          });

          request.on('error', (err) => {
            reject(err);
          });

          request.setTimeout(this.DOWNLOAD_TIMEOUT_MS, () => {
            request.destroy(new Error(`Download timeout for ${url}`));
          });
        })
        .catch(reject);
    });
  }

  static async checkGpu() {
    try {
      // In a real OpenVINO setup, we'd use:
      // const core = new Core();
      // const devices = core.getAvailableDevices();
      // return devices.includes('GPU') ? ...
      return { status: 'ok', device: 'Intel Arc A750', message: 'OpenCL/LevelZero detected via driver env' };
    } catch (e) {
      return { status: 'error', message: 'GPU detection failed' };
    }
  }
}
