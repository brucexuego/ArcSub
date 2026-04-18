import fs from 'fs-extra';
import { PathManager } from '../path_manager.js';

export class EnvFileService {
  private static getEnvPath() {
    return PathManager.resolveDotEnvPath();
  }

  static async getValue(key: string) {
    const envPath = this.getEnvPath();
    if (!(await fs.pathExists(envPath))) return '';
    const content = await fs.readFile(envPath, 'utf8');
    const matched = content.match(new RegExp(`^${key}=(.+)$`, 'm'));
    return String(matched?.[1] || '').trim();
  }

  static async setValue(key: string, value: string) {
    const envPath = this.getEnvPath();
    const trimmedValue = String(value || '').trim();
    const lines = (await fs.pathExists(envPath))
      ? (await fs.readFile(envPath, 'utf8')).split(/\r?\n/)
      : [];
    const pattern = new RegExp(`^${key}=`);
    let replaced = false;
    const nextLines = lines.map((line) => {
      if (pattern.test(line)) {
        replaced = true;
        return `${key}=${trimmedValue}`;
      }
      return line;
    });

    if (!replaced) {
      if (nextLines.length > 0 && nextLines[nextLines.length - 1] !== '') {
        nextLines.push('');
      }
      nextLines.push(`${key}=${trimmedValue}`);
    }

    await fs.writeFile(envPath, `${nextLines.join('\n').replace(/\n+$/g, '')}\n`, 'utf8');
    process.env[key] = trimmedValue;
    return envPath;
  }
}
