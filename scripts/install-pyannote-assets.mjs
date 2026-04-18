import fs from 'fs-extra';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

function parseArgs(argv) {
  const result = {
    readinessOut: null,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    switch (token) {
      case '--readiness-out':
        result.readinessOut = argv[++index] || null;
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

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const rootDir = process.cwd();
  const [{ PyannoteSetupService }, { RuntimeReadinessService }] = await Promise.all([
    loadBuiltModule(rootDir, 'services/pyannote_setup_service.js'),
    loadBuiltModule(rootDir, 'services/runtime_readiness_service.js'),
  ]);

  const status = await PyannoteSetupService.ensureInstalled();
  const snapshot = await RuntimeReadinessService.getSnapshot();
  const payload = { success: true, status, snapshot };

  if (options.readinessOut) {
    const outputPath = path.resolve(rootDir, options.readinessOut);
    await fs.ensureDir(path.dirname(outputPath));
    await fs.writeJson(outputPath, payload, { spaces: 2 });
  }

  console.log(JSON.stringify(payload, null, 2));
}

main().catch((error) => {
  console.error(`[install-pyannote-assets] Failed: ${error?.message || error}`);
  process.exitCode = 1;
});
