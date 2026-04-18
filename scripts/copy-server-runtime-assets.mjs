import fs from 'fs-extra';
import path from 'node:path';

const root = process.cwd();
const buildRoot = path.join(root, 'build');

const directoryCopies = [
  {
    source: path.join(root, 'server', 'language_alignment'),
    destination: path.join(buildRoot, 'server', 'language_alignment'),
    filter: (sourcePath) => sourcePath.endsWith('.json'),
  },
  {
    source: path.join(root, 'server', 'video_sites', 'sites'),
    destination: path.join(buildRoot, 'server', 'video_sites', 'sites'),
    filter: (sourcePath) => sourcePath.endsWith('.json'),
  },
];

async function main() {
  for (const entry of directoryCopies) {
    if (!(await fs.pathExists(entry.source))) continue;
    await fs.ensureDir(entry.destination);
    await fs.copy(entry.source, entry.destination, {
      overwrite: true,
      filter: (sourcePath) => {
        const stat = fs.statSync(sourcePath);
        if (stat.isDirectory()) return true;
        return entry.filter(sourcePath);
      },
    });
  }

  console.log('[build:server] Runtime assets copied into build/.');
}

main().catch((error) => {
  console.error('[build:server] Failed to copy runtime assets:', error);
  process.exitCode = 1;
});
