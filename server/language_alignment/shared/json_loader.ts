import fs from 'fs-extra';
import path from 'path';
import { fileURLToPath } from 'url';

export function loadJsonFromModuleDir<T>(moduleUrl: string, filename: string): T {
  const moduleDir = path.dirname(fileURLToPath(moduleUrl));
  return fs.readJsonSync(path.join(moduleDir, filename)) as T;
}

export function tryLoadJsonFromModuleDir<T>(moduleUrl: string, filename: string): T | null {
  const moduleDir = path.dirname(fileURLToPath(moduleUrl));
  const filePath = path.join(moduleDir, filename);
  if (!fs.existsSync(filePath)) {
    return null;
  }
  return fs.readJsonSync(filePath) as T;
}

export function normalizeLanguageCode(language?: string) {
  return String(language || '').trim().toLowerCase();
}
