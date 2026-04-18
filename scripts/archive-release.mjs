import { existsSync, rmSync } from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { getReleaseAssetNames } from "./release-file-names.mjs";

const target = process.argv[2] ?? "all";
const repoRoot = process.cwd();
const releaseRoot = path.join(repoRoot, ".release");
const isWindowsHost = process.platform === "win32";
const assetNames = getReleaseAssetNames(repoRoot);

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd ?? repoRoot,
    stdio: options.stdio ?? "inherit",
    encoding: options.encoding ?? "utf8",
    shell: false,
  });
  if (result.status !== 0) {
    throw new Error(`Command failed: ${command} ${args.join(" ")}`);
  }
  return result;
}

function runCapture(command, args) {
  const result = run(command, args, { stdio: "pipe" });
  return String(result.stdout || "").trim();
}

function ensureReleaseDir(name) {
  const dir = path.join(releaseRoot, name);
  if (!existsSync(dir)) {
    throw new Error(`Release directory not found: ${dir}`);
  }
  return dir;
}

function resolveWslPath(targetPath) {
  const escaped = targetPath.replace(/\\/g, "/").replace(/'/g, `'\\''`);
  return runCapture("wsl", ["bash", "-lc", `wslpath -a '${escaped}'`]);
}

function resolveBashPath(targetPath) {
  if (isWindowsHost) {
    return resolveWslPath(targetPath);
  }
  return targetPath.replace(/\\/g, "/");
}

function removeIfExists(filePath) {
  if (existsSync(filePath)) {
    rmSync(filePath, { force: true });
  }
}

function archiveWindows() {
  const dir = ensureReleaseDir("windows-x64");
  const output = path.join(releaseRoot, assetNames.windowsArchive);
  const legacyOutput = path.join(releaseRoot, assetNames.legacyWindowsArchive);
  removeIfExists(output);
  removeIfExists(legacyOutput);

  const escape = (value) => value.replace(/'/g, "''");
  run("powershell", [
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-Command",
    `$min=[datetime]'1980-01-01T00:00:00';` +
      `Get-ChildItem -LiteralPath '${escape(dir)}' -Recurse -Force | ` +
      `Where-Object { $_.LastWriteTime -lt $min } | ` +
      `ForEach-Object { $_.LastWriteTime = $min }; ` +
      `Compress-Archive -Path '${escape(path.join(dir, "*"))}' -DestinationPath '${escape(output)}' -CompressionLevel Optimal`,
  ]);
  console.log(`[archive:release] windows archive created at ${output}`);
}

function archiveLinux() {
  ensureReleaseDir("linux-x64");
  const output = path.join(releaseRoot, assetNames.linuxArchive);
  const legacyOutput = path.join(releaseRoot, assetNames.legacyLinuxArchive);
  removeIfExists(output);
  removeIfExists(legacyOutput);

  if (isWindowsHost) {
    const releaseRootBash = resolveBashPath(releaseRoot).replace(/'/g, `'\\''`);
    const outputBash = resolveBashPath(output).replace(/'/g, `'\\''`);
    run("wsl", [
      "bash",
      "-lc",
      `cd '${releaseRootBash}' && tar -czf '${outputBash}' linux-x64`,
    ]);
  } else {
    run("tar", ["-czf", output, "-C", releaseRoot, "linux-x64"]);
  }

  console.log(`[archive:release] linux archive created at ${output}`);
}

function refreshChecksums() {
  run(process.execPath, [path.join(repoRoot, "scripts", "generate-release-checksums.mjs")]);
}

switch (target) {
  case "windows-x64":
    archiveWindows();
    refreshChecksums();
    break;
  case "linux-x64":
    archiveLinux();
    refreshChecksums();
    break;
  case "all":
    archiveWindows();
    archiveLinux();
    refreshChecksums();
    break;
  default:
    throw new Error(`Unknown archive target: ${target}`);
}
