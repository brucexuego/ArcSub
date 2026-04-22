import fs from 'fs-extra';
import path from 'path';
import { PathManager } from './path_manager.js';

function getExecutableFileName(baseName: string) {
  return process.platform === 'win32' ? `${baseName}.exe` : baseName;
}

function isFile(candidatePath: string) {
  try {
    return fs.statSync(candidatePath).isFile();
  } catch {
    return false;
  }
}

function getPathCandidates(command: string) {
  const pathValue = String(process.env.PATH || '');
  if (!pathValue) return [];

  const dirs = pathValue.split(path.delimiter).filter(Boolean);
  const extensions =
    process.platform === 'win32'
      ? String(process.env.PATHEXT || '.EXE;.CMD;.BAT;.COM')
          .split(';')
          .map((value) => value.trim())
          .filter(Boolean)
      : [''];

  const hasExplicitExt = path.extname(command).length > 0;
  const commandNames = hasExplicitExt ? [command] : extensions.map((ext) => `${command}${ext.toLowerCase()}`);
  return dirs.flatMap((dir) => commandNames.map((name) => path.join(dir, name)));
}

function resolveCommandOnPath(command: string) {
  const candidates = getPathCandidates(command);
  return candidates.find((candidatePath) => isFile(candidatePath)) || null;
}

export function getBundledToolPath(baseName: string) {
  return path.join(PathManager.getToolsPath(), getExecutableFileName(baseName));
}

export function getBundledToolPathIfExists(baseName: string) {
  const bundledPath = getBundledToolPath(baseName);
  return isFile(bundledPath) ? bundledPath : null;
}

export function isCommandAvailable(command: string) {
  return getPathCandidates(command).some((candidatePath) => isFile(candidatePath));
}

export function resolveToolCommand(baseName: string, fallbackCommand = baseName) {
  return getBundledToolPathIfExists(baseName) || resolveCommandOnPath(fallbackCommand) || fallbackCommand;
}

