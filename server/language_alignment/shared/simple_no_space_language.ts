import fs from 'fs-extra';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadJsonFromModuleDir, normalizeLanguageCode } from './json_loader.js';
import { genericFallbackSegmentNoSpaceLexicalUnits, normalizeNoSpaceAlignmentText } from './no_space_utils.js';
import type { LanguageAlignmentPack, NoSpaceLanguageConfig, NoSpaceLanguageModule } from './types.js';

type SimpleNoSpaceConfig = NoSpaceLanguageConfig & {
  scriptPattern: string;
};

export function createSimpleNoSpaceLanguagePack(moduleUrl: string): LanguageAlignmentPack {
  const config = loadJsonFromModuleDir<SimpleNoSpaceConfig>(moduleUrl, 'config.json');
  return createSimpleNoSpaceLanguagePackFromConfig(config);
}

export function createSimpleNoSpaceLanguagePackFromDir(dirPath: string): LanguageAlignmentPack {
  const config = fs.readJsonSync(path.join(dirPath, 'config.json')) as SimpleNoSpaceConfig;
  return createSimpleNoSpaceLanguagePackFromConfig(config);
}

export function createSimpleNoSpaceLanguagePackFromModuleDir(moduleUrl: string): LanguageAlignmentPack {
  const moduleDir = path.dirname(fileURLToPath(moduleUrl));
  return createSimpleNoSpaceLanguagePackFromDir(moduleDir);
}

function createSimpleNoSpaceLanguagePackFromConfig(config: SimpleNoSpaceConfig): LanguageAlignmentPack {
  const aliasSet = new Set(config.aliases.map((alias) => normalizeLanguageCode(alias)));
  const scriptRegex = new RegExp(config.scriptPattern, 'u');

  const noSpace: NoSpaceLanguageModule = {
    config,
    matchesLanguage(language?: string) {
      const normalized = normalizeLanguageCode(language);
      if (!normalized) return false;
      if (aliasSet.has(normalized)) return true;
      return [...aliasSet].some((alias) => normalized.startsWith(`${alias}-`));
    },
    matchesText(text?: string) {
      return scriptRegex.test(String(text || ''));
    },
    resolveVariant() {
      return null;
    },
    getVariantDebug() {
      return null;
    },
    normalizeAlignmentText: normalizeNoSpaceAlignmentText,
    fallbackSegmentLexicalUnits: genericFallbackSegmentNoSpaceLexicalUnits,
    shouldPreferStandaloneToken() {
      return false;
    },
    startsWithStandaloneToken() {
      return false;
    },
    isShortContinuationToken() {
      return false;
    },
    shouldMergeContinuation() {
      return false;
    },
    isShortTailToken() {
      return false;
    },
  };

  return {
    key: config.key,
    aliases: config.aliases,
    noSpace,
  };
}
