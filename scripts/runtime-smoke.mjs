import fs from 'fs-extra';
import path from 'node:path';

const root = process.cwd();

const requiredFiles = [
  'package.json',
  '.env.example',
  'README.md',
  'README.zh-TW.md',
  'README.ja.md',
  'build/server/index.js',
  'dist/index.html',
  'deploy.ps1',
  'deploy.sh',
  'start.production.ps1',
  'start.production.sh',
  'collect-diagnostics.ps1',
  'collect-diagnostics.sh',
  'scripts/deploy-manifest.json',
  'scripts/install-deployment-assets.mjs',
  'scripts/install-pyannote-assets.mjs',
  'scripts/finalize-runtime-install.mjs',
];

const requiredDirectories = [
  'build/server',
  'dist',
  'public',
  'docs',
  'tools_src',
  'server/glossaries',
];

async function assertFile(relativePath) {
  const absolutePath = path.join(root, relativePath);
  if (!(await fs.pathExists(absolutePath))) {
    throw new Error(`Missing required runtime file: ${relativePath}`);
  }
  const stats = await fs.stat(absolutePath);
  if (!stats.isFile()) {
    throw new Error(`Expected runtime file but found another type: ${relativePath}`);
  }
}

async function assertDirectory(relativePath) {
  const absolutePath = path.join(root, relativePath);
  if (!(await fs.pathExists(absolutePath))) {
    throw new Error(`Missing required runtime directory: ${relativePath}`);
  }
  const stats = await fs.stat(absolutePath);
  if (!stats.isDirectory()) {
    throw new Error(`Expected runtime directory but found another type: ${relativePath}`);
  }
}

async function main() {
  for (const relativePath of requiredFiles) {
    await assertFile(relativePath);
  }
  for (const relativePath of requiredDirectories) {
    await assertDirectory(relativePath);
  }

  const packageJson = await fs.readJson(path.join(root, 'package.json'));
  const dependencies = packageJson.dependencies || {};
  for (const packageName of ['express', 'fs-extra', 'dotenv', 'openvino-node', 'openvino-genai-node']) {
    if (!dependencies[packageName]) {
      throw new Error(`Missing runtime dependency in package.json: ${packageName}`);
    }
  }

  console.log('[runtime-smoke] Runtime structure check passed.');
}

main().catch((error) => {
  console.error('[runtime-smoke] Failed:', error?.message || error);
  process.exitCode = 1;
});
