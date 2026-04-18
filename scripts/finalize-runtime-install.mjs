import fs from 'fs-extra';
import path from 'node:path';
import { spawn } from 'node:child_process';

const root = process.cwd();
const timeoutMs = Number.parseInt(process.env.ARCSUB_OPENVINO_DOWNLOAD_TIMEOUT_MS || '300000', 10);

function log(message) {
  console.log(`[finalize-runtime-install] ${message}`);
}

async function pathExists(relativePath) {
  return fs.pathExists(path.join(root, relativePath));
}

async function patchOpenvinoTimeout() {
  const utilsPath = path.join(root, 'node_modules/openvino-node/scripts/lib/utils.js');
  if (!(await fs.pathExists(utilsPath))) {
    return false;
  }

  const original = await fs.readFile(utilsPath, 'utf8');
  const patched = original.replace(
    /const timeout = \d+;/,
    `const timeout = Number.parseInt(process.env.ARCSUB_OPENVINO_DOWNLOAD_TIMEOUT_MS || "${timeoutMs}", 10);`
  );

  if (patched !== original) {
    await fs.writeFile(utilsPath, patched, 'utf8');
    log(`Patched OpenVINO download timeout to ${timeoutMs} ms.`);
  }

  return true;
}

async function runNodeScript(scriptRelativePath, args = []) {
  const nodeExe = process.execPath;
  const nodeDir = path.dirname(nodeExe);
  const scriptPath = path.join(root, scriptRelativePath);
  if (!(await fs.pathExists(scriptPath))) {
    log(`Skipping missing script: ${scriptRelativePath}`);
    return;
  }

  await new Promise((resolve, reject) => {
    const child = spawn(nodeExe, [scriptPath, ...args], {
      cwd: root,
      stdio: 'inherit',
      env: {
        ...process.env,
        PATH: `${nodeDir}${path.delimiter}${process.env.PATH || ''}`,
        npm_node_execpath: nodeExe,
        npm_execpath: process.env.npm_execpath || '',
        ARCSUB_OPENVINO_DOWNLOAD_TIMEOUT_MS: String(timeoutMs),
      },
      windowsHide: true,
    });

    child.on('error', reject);
    child.on('exit', (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(`Command failed (${code}): ${scriptRelativePath} ${args.join(' ')}`.trim()));
    });
  });
}

async function main() {
  log(`Working directory: ${root}`);

  const maybeOnnx = [
    'node_modules/onnxruntime-node/script/install',
    'node_modules/onnxruntime-node/script/install.js',
  ];
  const onnxInstallScript = [];
  for (const candidate of maybeOnnx) {
    if (await pathExists(candidate)) {
      onnxInstallScript.push(candidate);
      break;
    }
  }
  if (onnxInstallScript.length > 0) {
    log('Running onnxruntime-node install script...');
    await runNodeScript(onnxInstallScript[0]);
  }

  const hasOpenvinoNode = await patchOpenvinoTimeout();
  if (hasOpenvinoNode && (await pathExists('node_modules/openvino-node/scripts/download-runtime.js'))) {
    log('Running openvino-node runtime installer...');
    await runNodeScript('node_modules/openvino-node/scripts/download-runtime.js', ['--ignore-if-exists']);
  }

  if (await pathExists('node_modules/openvino-genai-node/scripts/download-runtime.cjs')) {
    log('Running openvino-genai-node runtime installer...');
    await runNodeScript('node_modules/openvino-genai-node/scripts/download-runtime.cjs', ['--ignore-if-exists']);
  }

  log('Runtime package finalization completed.');
}

main().catch((error) => {
  console.error('[finalize-runtime-install] Failed:', error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
