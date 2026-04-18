import fs from 'fs-extra';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';
import { matchesLanguageAlias } from '../language/resolver.js';
import { createSimpleNoSpaceLanguagePackFromDir } from './shared/simple_no_space_language.js';
import type { ForcedAlignmentLanguageModule, LanguageAlignmentPack, LocalAsrPromptModule, NoSpaceLanguageModule } from './shared/types.js';

const registryDir = path.dirname(fileURLToPath(import.meta.url));

async function discoverLanguagePacks(): Promise<LanguageAlignmentPack[]> {
  const entries = fs
    .readdirSync(registryDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name !== 'shared')
    .sort((a, b) => a.name.localeCompare(b.name));

  const packs: LanguageAlignmentPack[] = [];
  for (const entry of entries) {
    const dirPath = path.join(registryDir, entry.name);
    const configPath = path.join(dirPath, 'config.json');
    if (!fs.existsSync(configPath)) continue;

    const indexTsPath = path.join(dirPath, 'index.ts');
    const indexJsPath = path.join(dirPath, 'index.js');
    const modulePath = fs.existsSync(indexTsPath) ? indexTsPath : fs.existsSync(indexJsPath) ? indexJsPath : null;

    if (modulePath) {
      try {
        const imported = await import(pathToFileURL(modulePath).href);
        const pack = resolvePackFromModule(imported);
        if (pack) {
          packs.push(pack);
          continue;
        }
      } catch (error) {
        console.warn(`[LANG-ALIGN] Failed to load language module "${entry.name}", falling back to config-only pack:`, error);
      }
    }

    packs.push(createSimpleNoSpaceLanguagePackFromDir(dirPath));
  }

  return dedupePacks(packs);
}

function resolvePackFromModule(imported: Record<string, unknown>): LanguageAlignmentPack | null {
  const direct = imported.languagePack;
  if (isLanguagePack(direct)) return direct;

  for (const value of Object.values(imported)) {
    if (isLanguagePack(value)) return value;
  }
  return null;
}

function isLanguagePack(value: unknown): value is LanguageAlignmentPack {
  return Boolean(
    value &&
      typeof value === 'object' &&
      typeof (value as LanguageAlignmentPack).key === 'string' &&
      Array.isArray((value as LanguageAlignmentPack).aliases)
  );
}

function dedupePacks(packs: LanguageAlignmentPack[]) {
  const seen = new Set<string>();
  const deduped: LanguageAlignmentPack[] = [];
  for (const pack of packs) {
    const key = String(pack.key || '').trim().toLowerCase();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    deduped.push(pack);
  }
  return deduped;
}

const registeredPacks = await discoverLanguagePacks();

export class LanguageAlignmentRegistry {
  static list() {
    return [...registeredPacks];
  }

  static getPack(language?: string) {
    return registeredPacks.find((pack) => pack.aliases.some((alias) => this.matchesLanguageAlias(language, alias))) || null;
  }

  static getNoSpaceModule(language?: string, sampleText?: string): NoSpaceLanguageModule | null {
    const normalized = String(language || '').trim().toLowerCase();
    const byLanguage = registeredPacks.find((pack) => pack.noSpace?.matchesLanguage(language))?.noSpace || null;
    if (byLanguage) return byLanguage;
    if (sampleText && (!normalized || normalized === 'auto')) {
      return registeredPacks.find((pack) => pack.noSpace?.matchesText(sampleText))?.noSpace || null;
    }
    return null;
  }

  static getForcedAlignmentModule(language?: string, sampleText?: string): ForcedAlignmentLanguageModule | null {
    const normalized = String(language || '').trim().toLowerCase();
    const byLanguage = registeredPacks.find((pack) => pack.forcedAlignment?.matchesLanguage(language))?.forcedAlignment || null;
    if (byLanguage) return byLanguage;
    if (sampleText && (!normalized || normalized === 'auto')) {
      return this.getBestForcedAlignmentTextMatch(sampleText)?.module || null;
    }
    return null;
  }

  static getLocalAsrPromptModule(language?: string, sampleText?: string): LocalAsrPromptModule | null {
    const normalized = String(language || '').trim().toLowerCase();
    const byLanguage = registeredPacks.find((pack) => pack.localAsrPrompt?.matchesLanguage(language))?.localAsrPrompt || null;
    if (byLanguage) return byLanguage;
    if (sampleText && (!normalized || normalized === 'auto')) {
      return this.getBestLocalAsrPromptTextMatch(sampleText)?.module || null;
    }
    return null;
  }

  static getBestForcedAlignmentTextMatch(sampleText?: string) {
    const text = String(sampleText || '').trim();
    if (!text) return null;

    let best: { module: ForcedAlignmentLanguageModule; score: number; key: string } | null = null;
    for (const pack of registeredPacks) {
      const module = pack.forcedAlignment;
      if (!module) continue;
      const score =
        typeof module.scoreText === 'function'
          ? Number(module.scoreText(text)) || 0
          : module.matchesText?.(text)
            ? 1
            : 0;
      if (!(score > 0)) continue;
      if (!best || score > best.score) {
        best = {
          module,
          score,
          key: pack.key,
        };
      }
    }
    return best;
  }

  static getBestLocalAsrPromptTextMatch(sampleText?: string) {
    const text = String(sampleText || '').trim();
    if (!text) return null;

    let best: { module: LocalAsrPromptModule; score: number; key: string } | null = null;
    for (const pack of registeredPacks) {
      const module = pack.localAsrPrompt;
      if (!module) continue;
      const score =
        typeof module.scoreText === 'function'
          ? Number(module.scoreText(text)) || 0
          : module.matchesText?.(text)
            ? 1
            : 0;
      if (!(score > 0)) continue;
      if (!best || score > best.score) {
        best = {
          module,
          score,
          key: pack.key,
        };
      }
    }
    return best;
  }

  private static matchesLanguageAlias(language: string | undefined, alias: string) {
    return matchesLanguageAlias(language, alias);
  }
}
