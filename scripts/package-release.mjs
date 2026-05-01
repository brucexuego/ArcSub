import fs from 'fs-extra';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = process.cwd();
const target = String(process.env.ARCSUB_RELEASE_TARGET || 'linux-x64').trim() || 'linux-x64';
const releaseRoot = path.resolve(root, process.env.ARCSUB_RELEASE_DIR || `.release/${target}`);
const isWindowsHost = process.platform === 'win32';
const bundleRuntime = !['0', 'false', 'no'].includes(
  String(process.env.ARCSUB_RELEASE_BUNDLE_RUNTIME || '1').trim().toLowerCase()
);
const runtimeDependencyNames = [
  '@google/genai',
  '@huggingface/transformers',
  '@pinyin-pro/data',
  'dotenv',
  'express',
  'fs-extra',
  'hangul-romanization',
  'kuromoji',
  'lowdb',
  'multer',
  'onnxruntime-node',
  'opencc-js',
  'openvino-genai-node',
  'openvino-node',
  'pinyin-pro',
  'sherpa-onnx',
  'unzipper',
];

async function readPackageLock() {
  const lockPath = path.join(root, 'package-lock.json');
  if (!(await fs.pathExists(lockPath))) {
    return null;
  }

  return fs.readJson(lockPath);
}

const releaseCopies = [
  { source: 'build', destination: 'build' },
  { source: 'dist', destination: 'dist' },
  { source: 'public', destination: 'public' },
  { source: 'server/glossaries', destination: 'server/glossaries' },
  { source: 'tools_src/openvino_asr_env.py', destination: 'tools_src/openvino_asr_env.py' },
  { source: 'tools_src/python_runtime_bootstrap.py', destination: 'tools_src/python_runtime_bootstrap.py' },
  { source: 'tools_src/openvino_genai_translate_helper.mjs', destination: 'tools_src/openvino_genai_translate_helper.mjs' },
  { source: 'tools_src/openvino_translate_helper.py', destination: 'tools_src/openvino_translate_helper.py' },
  { source: 'tools_src/openvino_whisper_helper.py', destination: 'tools_src/openvino_whisper_helper.py' },
  { source: 'tools_src/convert_hf_model_to_openvino.py', destination: 'tools_src/convert_hf_model_to_openvino.py' },
  { source: 'tools_src/convert_official_qwen3_asr.py', destination: 'tools_src/convert_official_qwen3_asr.py' },
  { source: 'tools_src/qwen3_asr_official_support.py', destination: 'tools_src/qwen3_asr_official_support.py' },
  { source: 'tools_src/qwen_asr_runtime.py', destination: 'tools_src/qwen_asr_runtime.py' },
  { source: 'tools_src/prepare_pyannote_vbx.py', destination: 'tools_src/prepare_pyannote_vbx.py' },
  { source: 'tools_src/export_pyannote.py', destination: 'tools_src/export_pyannote.py' },
  { source: '.env.example', destination: '.env.example' },
  { source: 'README.md', destination: 'README.md' },
  { source: 'README.zh-TW.md', destination: 'README.zh-TW.md' },
  { source: 'README.ja.md', destination: 'README.ja.md' },
  { source: 'docs', destination: 'docs' },
  { source: 'collect-diagnostics.ps1', destination: 'collect-diagnostics.ps1' },
  { source: 'collect-diagnostics.sh', destination: 'collect-diagnostics.sh' },
  { source: 'deploy.ps1', destination: 'deploy.ps1' },
  { source: 'deploy.sh', destination: 'deploy.sh' },
  { source: 'start.production.ps1', destination: 'start.production.ps1' },
  { source: 'start.production.sh', destination: 'start.production.sh' },
  { source: 'install-linux-system-deps.sh', destination: 'install-linux-system-deps.sh' },
  { source: 'install-linux-release-deps.sh', destination: 'install-linux-release-deps.sh' },
  { source: 'scripts/preflight-linux-runtime.sh', destination: 'scripts/preflight-linux-runtime.sh' },
  { source: 'scripts/install-deployment-assets.mjs', destination: 'scripts/install-deployment-assets.mjs' },
  { source: 'scripts/install-pyannote-assets.mjs', destination: 'scripts/install-pyannote-assets.mjs' },
  { source: 'scripts/finalize-runtime-install.mjs', destination: 'scripts/finalize-runtime-install.mjs' },
  { source: 'scripts/runtime-smoke.mjs', destination: 'scripts/runtime-smoke.mjs' },
  { source: 'scripts/deploy-manifest.json', destination: 'scripts/deploy-manifest.json' },
];

const linuxExecutableCopies = [
  'deploy.sh',
  'collect-diagnostics.sh',
  'start.production.sh',
  'install-linux-system-deps.sh',
  'install-linux-release-deps.sh',
  'scripts/preflight-linux-runtime.sh',
];

async function assertExists(relativePath) {
  const absolutePath = path.join(root, relativePath);
  if (!(await fs.pathExists(absolutePath))) {
    throw new Error(`Required release input is missing: ${relativePath}`);
  }
}

async function copyReleaseEntry(sourceRelativePath, destinationRelativePath) {
  const source = path.join(root, sourceRelativePath);
  const destination = path.join(releaseRoot, destinationRelativePath);
  if (!(await fs.pathExists(source))) return;

  const stats = await fs.stat(source);
  if (stats.isDirectory()) {
    await fs.copy(source, destination, { overwrite: true });
    return;
  }

  if (destinationRelativePath.endsWith('.sh')) {
    const content = await fs.readFile(source, 'utf8');
    await fs.ensureDir(path.dirname(destination));
    await fs.writeFile(destination, content.replace(/\r\n/g, '\n'), 'utf8');
    return;
  }

  await fs.copy(source, destination, { overwrite: true });
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd || root,
    encoding: 'utf8',
    stdio: options.stdio || 'inherit',
    shell: false,
    env: options.env || process.env,
  });
  if (result.status !== 0) {
    throw new Error(`Command failed: ${command} ${args.join(' ')}`.trim());
  }
  return result;
}

function resolveWslPath(absolutePath) {
  const escaped = absolutePath.replace(/\\/g, '/').replace(/'/g, `'\\''`);
  const result = run('wsl', ['bash', '-lc', `wslpath -a '${escaped}'`], {
    stdio: 'pipe',
  });
  const resolved = String(result.stdout || '').trim();
  if (!resolved) {
    throw new Error(`Failed to resolve WSL path for ${absolutePath}`);
  }
  return resolved;
}

function runBash(command, options = {}) {
  if (isWindowsHost) {
    return run('wsl', ['bash', '-lc', command], options);
  }
  return run('bash', ['-lc', command], options);
}

function resolveBashPath(absolutePath) {
  if (isWindowsHost) {
    return resolveWslPath(absolutePath);
  }
  return absolutePath.replace(/\\/g, '/');
}

async function removeReleaseSensitiveState() {
  const cleanupTargets = [
    '.env',
    'runtime/db',
    'runtime/deploy',
    'runtime/local',
    'runtime/logs',
    'runtime/projects',
    'runtime/tmp',
  ];

  for (const relativePath of cleanupTargets) {
    await fs.remove(path.join(releaseRoot, relativePath));
  }
}

async function bundleReleaseRuntime() {
  if (!bundleRuntime) {
    return;
  }

  console.log(`[package:release] Bundling runtime dependencies for ${target}...`);

  if (target.startsWith('windows')) {
    run('powershell', [
      '-NoProfile',
      '-ExecutionPolicy',
      'Bypass',
      '-File',
      path.join(releaseRoot, 'deploy.ps1'),
      '-SkipBuild',
      '-SkipPyannote',
    ]);
  } else if (target.startsWith('linux')) {
    const releaseRootBash = resolveBashPath(releaseRoot).replace(/'/g, `'\\''`);
    runBash(`cd '${releaseRootBash}' && bash ./deploy.sh --skip-build --skip-pyannote`);
  } else {
    throw new Error(`Unsupported release target for bundled runtime: ${target}`);
  }

  await removeReleaseSensitiveState();
}

async function main() {
  await assertExists('build/server/index.js');
  await assertExists('dist/index.html');
  for (const entry of releaseCopies) {
    await assertExists(entry.source);
  }

  await fs.remove(releaseRoot);
  await fs.ensureDir(releaseRoot);

  for (const entry of releaseCopies) {
    await copyReleaseEntry(entry.source, entry.destination);
  }

  if (target.startsWith('linux')) {
    for (const relativePath of linuxExecutableCopies) {
      const absolutePath = path.join(releaseRoot, relativePath);
      if (!(await fs.pathExists(absolutePath))) continue;
      await fs.chmod(absolutePath, 0o755);
    }
  }

  const packageJsonPath = path.join(root, 'package.json');
  const packageJson = await fs.readJson(packageJsonPath);
  const packageLock = await readPackageLock();
  const gitHead = spawnSync('git', ['rev-parse', '--short', 'HEAD'], {
    cwd: root,
    encoding: 'utf8',
  });
  const gitBranch = spawnSync('git', ['rev-parse', '--abbrev-ref', 'HEAD'], {
    cwd: root,
    encoding: 'utf8',
  });
  const createdAt = new Date().toISOString();
  const buildId = [
    packageJson.version,
    target,
    createdAt.replace(/[:.]/g, '-'),
  ].join('__');
  const runtimeDependencies = Object.fromEntries(
    runtimeDependencyNames
      .filter((name) => packageJson.dependencies?.[name])
      .map((name) => [
        name,
        packageLock?.packages?.[`node_modules/${name}`]?.version || packageJson.dependencies[name],
      ])
  );

  const releasePackageJson = {
    name: packageJson.name,
    private: packageJson.private,
    version: packageJson.version,
    type: packageJson.type,
    license: packageJson.license,
    scripts: {
      'deploy:windows': 'powershell -ExecutionPolicy Bypass -File ./deploy.ps1',
      'deploy:linux': 'bash ./deploy.sh',
      'start:prod:windows': 'powershell -ExecutionPolicy Bypass -File ./start.production.ps1',
      'start:prod:linux': 'bash ./start.production.sh',
    },
    dependencies: runtimeDependencies,
  };

  await fs.writeJson(path.join(releaseRoot, 'package.json'), releasePackageJson, { spaces: 2 });

  await bundleReleaseRuntime();

  const manifest = {
    buildId,
    createdAt,
    target,
    version: packageJson.version,
    gitHead: gitHead.status === 0 ? String(gitHead.stdout || '').trim() || null : null,
    gitBranch: gitBranch.status === 0 ? String(gitBranch.stdout || '').trim() || null : null,
    releaseRoot: path.relative(root, releaseRoot).replace(/\\/g, '/'),
    includes: releaseCopies.map((entry) => entry.destination.replace(/\\/g, '/')),
    excludes: [
      'src/',
      '.git/',
      'workspace-only files and local caches',
    ],
    notes: [
      bundleRuntime
        ? 'Runtime dependencies and baseline assets are prebundled into this release.'
        : 'Run ./deploy.sh or ./deploy.ps1 inside the release directory to install runtime dependencies and baseline assets.',
      'Local ASR and local translation models are intentionally not preinstalled.',
    ],
  };

  await fs.writeJson(path.join(releaseRoot, 'release-manifest.json'), manifest, { spaces: 2 });
  console.log(`[package:release] ${target} assembled at ${releaseRoot}`);
}

main().catch((error) => {
  console.error('[package:release] Failed to assemble release:', error);
  process.exitCode = 1;
});
