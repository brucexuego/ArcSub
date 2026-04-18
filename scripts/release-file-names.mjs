import { readFileSync } from 'node:fs';
import path from 'node:path';

export function getReleaseAssetNames(repoRoot = process.cwd()) {
  const packageJsonPath = path.join(repoRoot, 'package.json');
  const packageJson = JSON.parse(readFileSync(packageJsonPath, 'utf8'));
  const version = String(packageJson.version || '').trim();
  if (!version) {
    throw new Error(`Missing package version in ${packageJsonPath}`);
  }

  return {
    version,
    windowsArchive: `ArcSub-v${version}-windows-x64.zip`,
    linuxArchive: `ArcSub-v${version}-linux-x64.tar.gz`,
    checksums: `ArcSub-v${version}-SHA256SUMS.txt`,
    legacyWindowsArchive: 'ArcSub-windows-x64.zip',
    legacyLinuxArchive: 'ArcSub-linux-x64.tar.gz',
    legacyChecksums: 'SHA256SUMS.txt',
  };
}
