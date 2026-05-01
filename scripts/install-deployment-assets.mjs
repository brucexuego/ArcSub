import fs from 'fs-extra';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

function parseArgs(argv) {
  const result = {
    manifestPath: null,
    readinessOut: null,
    json: false,
    skipPyannote: false,
    failOnBaseline: true,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    switch (token) {
      case '--manifest':
        result.manifestPath = argv[++index] || null;
        break;
      case '--readiness-out':
        result.readinessOut = argv[++index] || null;
        break;
      case '--json':
        result.json = true;
        break;
      case '--skip-pyannote':
        result.skipPyannote = true;
        break;
      case '--no-fail-on-baseline':
        result.failOnBaseline = false;
        break;
      default:
        throw new Error(`Unknown argument: ${token}`);
    }
  }

  return result;
}

async function loadBuiltModule(rootDir, relativePath) {
  const absolutePath = path.join(rootDir, 'build', 'server', relativePath);
  if (!(await fs.pathExists(absolutePath))) {
    throw new Error(`Missing built module: ${absolutePath}`);
  }
  return import(pathToFileURL(absolutePath).href);
}

async function hasConfiguredEnvValue(rootDir, key) {
  if (String(process.env[key] || '').trim()) return true;
  const envPath = path.join(rootDir, '.env');
  if (!(await fs.pathExists(envPath))) return false;
  const content = await fs.readFile(envPath, 'utf8');
  const matched = content.match(new RegExp(`^${key}=(.+)$`, 'm'));
  return Boolean(String(matched?.[1] || '').trim());
}

function printSummary(snapshot, assetSummary, pyannoteSummary) {
  const lines = [
    `[deploy-assets] baseline_ready=${snapshot.baseline_ready}`,
    `[deploy-assets] local_runtime_ready=${snapshot.local_runtime_ready}`,
    `[deploy-assets] openvino_detected=${snapshot.openvino_detected}`,
    `[deploy-assets] python_ready=${snapshot.python_ready} python=${snapshot.python?.path || snapshot.python?.command || 'missing'}`,
    `[deploy-assets] local_model_install_available=${snapshot.local_model_install_available}`,
    `[deploy-assets] ffmpeg=${snapshot.tools.ffmpeg.ready} yt-dlp=${snapshot.tools.ytDlp.ready}`,
    `[deploy-assets] vad_ready=${snapshot.vad_ready} speaker_embedding_ready=${snapshot.speaker_embedding_ready}`,
    `[deploy-assets] pyannote_ready=${snapshot.pyannote_ready} pyannote_state=${snapshot.pyannote_state}`,
    `[deploy-assets] required_assets_installed=${assetSummary.installed.length} skipped=${assetSummary.skipped.length}`,
    `[deploy-assets] required_assets_converted=${assetSummary.converted?.length || 0}`,
    `[deploy-assets] pyannote_step=${pyannoteSummary.status}`,
  ];
  for (const line of lines) {
    console.log(line);
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const rootDir = process.cwd();
  const manifestPath = path.resolve(rootDir, options.manifestPath || path.join('scripts', 'deploy-manifest.json'));
  const manifest = await fs.readJson(manifestPath);

  const [{ PathManager }, { ResourceManager }, { RuntimeReadinessService }] = await Promise.all([
    loadBuiltModule(rootDir, 'path_manager.js'),
    loadBuiltModule(rootDir, 'services/resource_manager.js'),
    loadBuiltModule(rootDir, 'services/runtime_readiness_service.js'),
  ]);

  await PathManager.ensureBaseDirs();
  await ResourceManager.ensureTools();

  const assetSummary = await ResourceManager.ensureRequiredBaselineAssets(manifest);

  const pyannoteSummary = {
    status: 'skipped',
    reason: 'HF_TOKEN not configured.',
  };
  if (!options.skipPyannote && (await hasConfiguredEnvValue(rootDir, 'HF_TOKEN'))) {
    try {
      const { PyannoteDiarizationService } = await loadBuiltModule(rootDir, 'pyannote_diarization_service.js');
      await PyannoteDiarizationService.ensureDeploymentAssets();
      pyannoteSummary.status = 'installed';
      pyannoteSummary.reason = '';
    } catch (error) {
      pyannoteSummary.status = 'skipped';
      pyannoteSummary.reason = String(error?.message || error || 'Pyannote asset installation failed.');
      console.warn(`[deploy-assets] WARNING: ${pyannoteSummary.reason}`);
    }
  }

  const snapshot = await RuntimeReadinessService.getSnapshot();
  if (pyannoteSummary.status === 'skipped' && snapshot.pyannote_ready) {
    pyannoteSummary.status = 'already-ready';
    pyannoteSummary.reason = '';
  }
  const payload = {
    snapshot,
    assets: assetSummary,
    pyannote: pyannoteSummary,
  };

  if (options.readinessOut) {
    const outputPath = path.resolve(rootDir, options.readinessOut);
    await fs.ensureDir(path.dirname(outputPath));
    await fs.writeJson(outputPath, payload, { spaces: 2 });
  }

  if (options.json) {
    console.log(JSON.stringify(payload, null, 2));
  } else {
    printSummary(snapshot, assetSummary, pyannoteSummary);
  }

  if (options.failOnBaseline && !snapshot.baseline_ready) {
    throw new Error('Baseline runtime is not ready after deployment asset install.');
  }
}

main().catch((error) => {
  console.error(`[deploy-assets] Failed: ${error?.message || error}`);
  process.exitCode = 1;
});
