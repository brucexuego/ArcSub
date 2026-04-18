import { createHash } from 'node:crypto';
import { createReadStream, existsSync } from 'node:fs';
import path from 'node:path';
import fs from 'fs-extra';
import { getReleaseAssetNames } from './release-file-names.mjs';

const repoRoot = process.cwd();
const releaseRoot = path.join(repoRoot, '.release');
const assetNames = getReleaseAssetNames(repoRoot);
const releaseFiles = [
  assetNames.windowsArchive,
  assetNames.linuxArchive,
];

async function hashFile(filePath) {
  return await new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    const stream = createReadStream(filePath);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

async function main() {
  await fs.ensureDir(releaseRoot);
  const entries = [];

  for (const fileName of releaseFiles) {
    const absolutePath = path.join(releaseRoot, fileName);
    if (!existsSync(absolutePath)) continue;
    const digest = await hashFile(absolutePath);
    entries.push(`${digest}  ${fileName}`);
  }

  const outputPath = path.join(releaseRoot, assetNames.checksums);
  const legacyOutputPath = path.join(releaseRoot, assetNames.legacyChecksums);
  const body = entries.length > 0 ? `${entries.join('\n')}\n` : '';
  await fs.writeFile(outputPath, body, 'utf8');
  await fs.writeFile(legacyOutputPath, body, 'utf8');
  console.log(`[release:checksums] wrote ${outputPath}`);
}

main().catch((error) => {
  console.error('[release:checksums] Failed:', error?.message || error);
  process.exitCode = 1;
});
