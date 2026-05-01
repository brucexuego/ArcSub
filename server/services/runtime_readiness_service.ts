import fs from 'fs-extra';
import path from 'path';
import { spawnSync } from 'child_process';
import { PathManager } from '../path_manager.js';
import { getBundledToolPathIfExists, isCommandAvailable } from '../runtime_tools.js';
import { LocalModelService } from './local_model_service.js';
import { PyannoteSetupService } from './pyannote_setup_service.js';
import { SettingsManager } from './settings_manager.js';

type ToolReadiness = {
  bundledPath: string | null;
  pathAvailable: boolean;
  ready: boolean;
  command: string;
};

type FileReadiness = {
  path: string;
  exists: boolean;
};

type PythonReadiness = {
  command: string;
  configured: boolean;
  path: string | null;
  version: string | null;
  ready: boolean;
  meetsMinimum: boolean;
};

type PythonCommandCandidate = {
  command: string;
  args: string[];
  display: string;
};

type ReleaseReadiness = {
  buildId: string | null;
  version: string | null;
  target: string | null;
  createdAt: string | null;
  gitHead: string | null;
  gitBranch: string | null;
  manifestPath: string | null;
};

export type RuntimeReadinessSnapshot = {
  checkedAt: string;
  release: ReleaseReadiness;
  baseline_ready: boolean;
  local_runtime_ready: boolean;
  cloud_ready: boolean;
  cloud_capable: boolean;
  openvino_detected: boolean;
  python_ready: boolean;
  python: PythonReadiness;
  local_model_install_available: boolean;
  local_model_install: {
    asr: boolean;
    translate: boolean;
  };
  tools: {
    ffmpeg: ToolReadiness;
    ytDlp: ToolReadiness;
  };
  vad_ready: boolean;
  vad: FileReadiness;
  ten_vad_ready: boolean;
  ten_vad: FileReadiness;
  speaker_embedding_ready: boolean;
  speaker_embedding: FileReadiness;
  pyannote_ready: boolean;
  pyannote_state: 'ready' | 'partial' | 'missing';
  pyannote: {
    segmentation: FileReadiness;
    embedding: FileReadiness;
    pldaConfig: FileReadiness;
    gatedAssetsExpected: boolean;
    tokenConfigured: boolean;
    installing: boolean;
    lastError: string | null;
  };
  model_setup: {
    hasAsr: boolean;
    hasTranslate: boolean;
    ready: boolean;
  };
  openvino: Awaited<ReturnType<typeof LocalModelService.getOpenvinoStatus>>;
  paths: ReturnType<typeof PathManager.describeResolvedPaths>;
};

function getReleaseReadiness(): ReleaseReadiness {
  const manifestPath = path.resolve(process.cwd(), 'release-manifest.json');
  if (!fs.existsSync(manifestPath)) {
    return {
      buildId: null,
      version: null,
      target: null,
      createdAt: null,
      gitHead: null,
      gitBranch: null,
      manifestPath: null,
    };
  }

  try {
    const manifest = fs.readJsonSync(manifestPath);
    return {
      buildId: typeof manifest?.buildId === 'string' ? manifest.buildId : null,
      version: typeof manifest?.version === 'string' ? manifest.version : null,
      target: typeof manifest?.target === 'string' ? manifest.target : null,
      createdAt: typeof manifest?.createdAt === 'string' ? manifest.createdAt : null,
      gitHead: typeof manifest?.gitHead === 'string' ? manifest.gitHead : null,
      gitBranch: typeof manifest?.gitBranch === 'string' ? manifest.gitBranch : null,
      manifestPath,
    };
  } catch {
    return {
      buildId: null,
      version: null,
      target: null,
      createdAt: null,
      gitHead: null,
      gitBranch: null,
      manifestPath,
    };
  }
}

const PYTHON_MINIMUM = {
  major: 3,
  minor: 12,
};

function getToolReadiness(command: string): ToolReadiness {
  const bundledPath = getBundledToolPathIfExists(command);
  const pathAvailable = isCommandAvailable(command);
  return {
    bundledPath,
    pathAvailable,
    ready: Boolean(bundledPath || pathAvailable),
    command,
  };
}

async function getFileReadiness(targetPath: string): Promise<FileReadiness> {
  return {
    path: targetPath,
    exists: await fs.pathExists(targetPath),
  };
}

async function getPreferredFileReadiness(targetPaths: string[]): Promise<FileReadiness> {
  let fallback: FileReadiness | null = null;
  for (const targetPath of targetPaths) {
    const readiness = await getFileReadiness(targetPath);
    if (!fallback) fallback = readiness;
    if (readiness.exists) return readiness;
  }
  return fallback ?? { path: targetPaths[0] ?? '', exists: false };
}

async function hasConfiguredEnvValue(filePath: string, key: string) {
  if (!(await fs.pathExists(filePath))) return false;
  const content = await fs.readFile(filePath, 'utf8');
  const matched = content.match(new RegExp(`^${key}=(.+)$`, 'm'));
  return Boolean(String(matched?.[1] || '').trim());
}

function parsePythonVersion(version: string | null) {
  const matched = String(version || '').trim().match(/^(\d+)\.(\d+)\.(\d+)/);
  if (!matched) return null;
  return {
    major: Number(matched[1]),
    minor: Number(matched[2]),
    patch: Number(matched[3]),
  };
}

function getPythonCommandCandidates(): {
  configured: string;
  candidates: PythonCommandCandidate[];
} {
  const configured = String(process.env.OPENVINO_HELPER_PYTHON || '').trim();
  const candidates: PythonCommandCandidate[] = [];

  if (configured) {
    candidates.push({
      command: configured,
      args: [],
      display: configured,
    });
  }

  candidates.push({
    command: process.platform === 'win32' ? 'python' : 'python3',
    args: [],
    display: process.platform === 'win32' ? 'python' : 'python3',
  });

  if (process.platform === 'win32') {
    candidates.push({
      command: 'py',
      args: ['-3.12'],
      display: 'py -3.12',
    });
  } else {
    candidates.push({
      command: 'python',
      args: [],
      display: 'python',
    });
  }

  return {
    configured,
    candidates: candidates.filter(
      (candidate, index, all) =>
        all.findIndex(
          (item) =>
            item.command === candidate.command &&
            item.display === candidate.display &&
            item.args.join('\0') === candidate.args.join('\0')
        ) === index
    ),
  };
}

function inspectPythonCommand(candidate: PythonCommandCandidate) {
  const result = spawnSync(
    candidate.command,
    [
      ...candidate.args,
      '-c',
      'import os,sys; print(sys.executable); print(f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}")',
    ],
    {
    encoding: 'utf8',
    env: {
      ...process.env,
      ...PathManager.getRuntimeTempEnv(),
      PYTHONUTF8: '1',
      PYTHONIOENCODING: 'utf-8',
    },
  });

  if (result.status !== 0) {
    return null;
  }

  const lines = String(result.stdout || '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length < 2) {
    return null;
  }

  const executablePath = lines[0];
  const version = lines[1];
  const parsed = parsePythonVersion(version);
  const meetsMinimum = Boolean(
    parsed &&
      (parsed.major > PYTHON_MINIMUM.major ||
        (parsed.major === PYTHON_MINIMUM.major && parsed.minor >= PYTHON_MINIMUM.minor))
  );

  return {
    path: executablePath || null,
    version: version || null,
    ready: meetsMinimum,
    meetsMinimum,
  };
}

function getPythonReadiness(): PythonReadiness {
  const { configured, candidates } = getPythonCommandCandidates();

  for (const candidate of candidates) {
    const inspected = inspectPythonCommand(candidate);
    if (!inspected) continue;
    return {
      command: candidate.display,
      configured: Boolean(configured),
      path: inspected.path,
      version: inspected.version,
      ready: inspected.ready,
      meetsMinimum: inspected.meetsMinimum,
    };
  }

  return {
    command: configured || (process.platform === 'win32' ? 'python' : 'python3'),
    configured: Boolean(configured),
    path: null,
    version: null,
    ready: false,
    meetsMinimum: false,
  };
}

function buildModelSetupStatus(settings: any) {
  const installedLocalModels = Array.isArray(settings?.localModels?.installed) ? settings.localModels.installed : [];
  const selectedLocalAsrId =
    typeof settings?.localModels?.asrSelectedId === 'string' ? settings.localModels.asrSelectedId : '';
  const selectedLocalTranslateId =
    typeof settings?.localModels?.translateSelectedId === 'string' ? settings.localModels.translateSelectedId : '';

  const cloudAsrReady = Array.isArray(settings?.asrModels) && settings.asrModels.length > 0;
  const cloudTranslateReady = Array.isArray(settings?.translateModels) && settings.translateModels.length > 0;
  const localAsrReady = installedLocalModels.some((model: any) => model?.type === 'asr' && model?.id === selectedLocalAsrId);
  const localTranslateReady = installedLocalModels.some(
    (model: any) => model?.type === 'translate' && model?.id === selectedLocalTranslateId
  );

  const hasAsr = cloudAsrReady || localAsrReady;
  const hasTranslate = cloudTranslateReady || localTranslateReady;

  return {
    hasAsr,
    hasTranslate,
    ready: hasAsr && hasTranslate,
  };
}

export class RuntimeReadinessService {
  static async getSnapshot(): Promise<RuntimeReadinessSnapshot> {
    const [settings, openvinoStatus] = await Promise.all([
      SettingsManager.getSettings(),
      LocalModelService.getOpenvinoStatus(),
    ]);

    const python = getPythonReadiness();
    const ffmpeg = getToolReadiness('ffmpeg');
    const ytDlp = getToolReadiness('yt-dlp');
    const vad = await getPreferredFileReadiness([
      path.join(PathManager.getModelsPath(), 'silero_vad.xml'),
      path.join(PathManager.getModelsPath(), 'silero_vad.onnx'),
    ]);
    const tenVad = await getFileReadiness(path.join(PathManager.getModelsPath(), 'ten-vad.onnx'));
    const speakerEmbedding = await getPreferredFileReadiness([
      path.join(PathManager.getModelsPath(), 'ecapa-tdnn.xml'),
      path.join(PathManager.getModelsPath(), 'ecapa-tdnn.onnx'),
    ]);
    const pyannoteSegmentation = await getFileReadiness(path.join(PathManager.getModelsPath(), 'pyannote', 'segmentation', 'model.xml'));
    const pyannoteEmbedding = await getFileReadiness(path.join(PathManager.getModelsPath(), 'pyannote', 'embedding', 'model.xml'));
    const pyannotePlda = await getFileReadiness(path.join(PathManager.getModelsPath(), 'pyannote', 'plda', 'vbx.json'));

    const pyannoteSetupStatus = await PyannoteSetupService.getStatus();
    const hfTokenConfigured =
      pyannoteSetupStatus.tokenConfigured ||
      (await hasConfiguredEnvValue(PathManager.resolveDotEnvPath(), 'HF_TOKEN'));

    const pyannoteReady = pyannoteSegmentation.exists && pyannoteEmbedding.exists && pyannotePlda.exists;
    const pyannotePartial = !pyannoteReady && (pyannoteSegmentation.exists || pyannoteEmbedding.exists || pyannotePlda.exists);
    const pyannoteState: RuntimeReadinessSnapshot['pyannote_state'] = pyannoteReady
      ? 'ready'
      : pyannotePartial
        ? 'partial'
        : 'missing';

    const localAsrInstallAvailable = Boolean(openvinoStatus?.asr?.ready);
    const localTranslateInstallAvailable = Boolean(
      openvinoStatus?.node?.available &&
        openvinoStatus?.genai?.available &&
        (openvinoStatus?.genai?.llmPipelineAvailable || openvinoStatus?.genai?.vlmPipelineAvailable)
    );
    const localModelInstallAvailable = localAsrInstallAvailable && localTranslateInstallAvailable;

    const modelSetup = buildModelSetupStatus(settings);
    const openvinoDetected = Boolean(openvinoStatus?.node?.available || openvinoStatus?.genai?.available);
    const baselineReady = ffmpeg.ready && ytDlp.ready && vad.exists && speakerEmbedding.exists;
    const release = getReleaseReadiness();

    return {
      checkedAt: new Date().toISOString(),
      release,
      baseline_ready: baselineReady,
      local_runtime_ready: baselineReady && localModelInstallAvailable,
      cloud_ready: modelSetup.ready,
      cloud_capable: true,
      openvino_detected: openvinoDetected,
      python_ready: python.ready,
      python,
      local_model_install_available: localModelInstallAvailable,
      local_model_install: {
        asr: localAsrInstallAvailable,
        translate: localTranslateInstallAvailable,
      },
      tools: {
        ffmpeg,
        ytDlp,
      },
      vad_ready: vad.exists,
      vad,
      ten_vad_ready: tenVad.exists,
      ten_vad: tenVad,
      speaker_embedding_ready: speakerEmbedding.exists,
      speaker_embedding: speakerEmbedding,
      pyannote_ready: pyannoteReady,
      pyannote_state: pyannoteState,
      pyannote: {
        segmentation: pyannoteSegmentation,
        embedding: pyannoteEmbedding,
        pldaConfig: pyannotePlda,
        gatedAssetsExpected: hfTokenConfigured,
        tokenConfigured: pyannoteSetupStatus.tokenConfigured,
        installing: pyannoteSetupStatus.installing,
        lastError: pyannoteSetupStatus.lastError,
      },
      model_setup: modelSetup,
      openvino: openvinoStatus,
      paths: PathManager.describeResolvedPaths(),
    };
  }
}
