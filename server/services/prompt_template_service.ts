import path from 'path';
import fs from 'fs-extra';
import { buildTargetLanguageDescriptor, normalizeLanguageKey } from '../language/resolver.js';
import { PathManager } from '../path_manager.js';

type PromptTemplateSource = 'built-in' | 'local';

interface PromptTemplateManifestEntry {
  id: string;
  labelKey?: string;
  fallbackLabel?: string;
  order?: number;
  enabled?: boolean;
}

export interface PromptTemplateListItem {
  id: string;
  labelKey: string | null;
  fallbackLabel: string;
  order: number;
  enabled: boolean;
  source: PromptTemplateSource;
  contentSource: PromptTemplateSource;
  hasLocalOverride: boolean;
  targetLang: string;
  language: string;
  content: string;
}

const TEMPLATE_ID_RE = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/;
const MAX_TEMPLATE_BYTES = 20 * 1024;

function isSafeTemplateId(id: string) {
  return TEMPLATE_ID_RE.test(id);
}

function isInside(parent: string, child: string) {
  const rel = path.relative(parent, child);
  return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel));
}

function normalizeWritableLanguage(targetLang: string) {
  const normalized = normalizeLanguageKey(targetLang || 'en');
  if (!normalized) return 'en';
  if (normalized === 'ja') return 'jp';
  return normalized.replace(/[^a-z0-9-]/gi, '').toLowerCase() || 'en';
}

function languageCandidates(targetLang: string) {
  const normalized = normalizeWritableLanguage(targetLang);
  const base = normalized.split('-')[0];
  const candidates = [
    normalized,
    normalized === 'zh-tw' ? 'zh-hant' : '',
    normalized === 'zh-cn' ? 'zh-hans' : '',
    normalized === 'jp' ? 'ja' : '',
    base && base !== normalized ? base : '',
    'en',
  ].filter(Boolean);
  return Array.from(new Set(candidates));
}

function renderTemplateVariables(
  template: string,
  input: { targetLang: string; sourceLang?: string; language: string }
) {
  const targetLanguage = buildTargetLanguageDescriptor(input.targetLang || input.language || 'en') || 'the requested target language';
  const sourceLanguage = input.sourceLang
    ? buildTargetLanguageDescriptor(input.sourceLang)
    : 'the source language';
  return template
    .replace(/\{target_language\}/g, targetLanguage)
    .replace(/\{target_lang\}/g, targetLanguage)
    .replace(/\{lang\}/g, targetLanguage)
    .replace(/\{target_locale\}/g, input.language)
    .replace(/\{source_language\}/g, sourceLanguage)
    .replace(/\{source_lang\}/g, sourceLanguage);
}

async function readJsonFile<T>(filePath: string): Promise<T | null> {
  try {
    if (!(await fs.pathExists(filePath))) return null;
    return await fs.readJson(filePath);
  } catch {
    return null;
  }
}

async function readManifest(root: string): Promise<PromptTemplateManifestEntry[]> {
  const manifest = await readJsonFile<{ templates?: PromptTemplateManifestEntry[] }>(path.join(root, 'manifest.json'));
  if (!manifest || !Array.isArray(manifest.templates)) return [];
  return manifest.templates
    .map((entry) => ({
      id: String(entry?.id || '').trim(),
      labelKey: typeof entry?.labelKey === 'string' ? entry.labelKey.trim() : undefined,
      fallbackLabel: typeof entry?.fallbackLabel === 'string' ? entry.fallbackLabel.trim() : undefined,
      order: Number.isFinite(Number(entry?.order)) ? Math.round(Number(entry.order)) : undefined,
      enabled: entry?.enabled !== false,
    }))
    .filter((entry) => isSafeTemplateId(entry.id));
}

async function readTemplateContent(input: {
  root: string;
  id: string;
  targetLang: string;
}): Promise<{ content: string; language: string; filePath: string } | null> {
  for (const language of languageCandidates(input.targetLang)) {
    const filePath = path.resolve(input.root, input.id, `${language}.txt`);
    if (!isInside(input.root, filePath)) continue;
    if (!(await fs.pathExists(filePath))) continue;
    const stats = await fs.stat(filePath);
    if (!stats.isFile() || stats.size > MAX_TEMPLATE_BYTES) continue;
    const raw = await fs.readFile(filePath, 'utf8');
    const content = raw.replace(/\u0000/g, '').trim();
    if (!content) continue;
    return { content, language, filePath };
  }
  return null;
}

export class PromptTemplateService {
  private static getBuiltInRoot() {
    return PathManager.getBuiltInTranslationPromptRootPath();
  }

  private static getLocalRoot() {
    return PathManager.getLocalTranslationPromptRootPath();
  }

  private static async getMergedManifest() {
    const builtInRoot = this.getBuiltInRoot();
    const builtIn = await readManifest(builtInRoot);
    const map = new Map<string, PromptTemplateManifestEntry & { source: PromptTemplateSource }>();

    for (const entry of builtIn) {
      map.set(entry.id, {
        ...entry,
        fallbackLabel: entry.fallbackLabel || entry.id,
        order: entry.order ?? 100,
        enabled: entry.enabled !== false,
        source: 'built-in',
      });
    }
    return Array.from(map.values()).sort((a, b) => (a.order ?? 0) - (b.order ?? 0) || a.id.localeCompare(b.id));
  }

  static async resolveTemplate(input: {
    templateId?: string;
    targetLang: string;
    sourceLang?: string;
  }): Promise<PromptTemplateListItem | null> {
    const id = String(input.templateId || '').trim();
    if (!id || !isSafeTemplateId(id)) return null;

    const manifest = await this.getMergedManifest();
    const metadata = manifest.find((entry) => entry.id === id);
    if (!metadata) return null;

    const localRoot = this.getLocalRoot();
    const builtInRoot = this.getBuiltInRoot();
    const local = await readTemplateContent({ root: localRoot, id, targetLang: input.targetLang });
    const builtIn = local ? null : await readTemplateContent({ root: builtInRoot, id, targetLang: input.targetLang });
    const resolved = local || builtIn;
    if (!resolved) return null;

    return {
      id,
      labelKey: metadata.labelKey || null,
      fallbackLabel: metadata.fallbackLabel || id,
      order: metadata.order ?? 500,
      enabled: metadata.enabled !== false,
      source: metadata.source || 'built-in',
      contentSource: local ? 'local' : 'built-in',
      hasLocalOverride: Boolean(local),
      targetLang: input.targetLang,
      language: resolved.language,
      content: renderTemplateVariables(resolved.content, {
        targetLang: input.targetLang,
        sourceLang: input.sourceLang,
        language: resolved.language,
      }),
    };
  }

  static async listTemplates(input: { targetLang: string; sourceLang?: string; includeDisabled?: boolean }) {
    const manifest = await this.getMergedManifest();
    const resolved = await Promise.all(
      manifest
        .filter((entry) => input.includeDisabled || entry.enabled !== false)
        .map((entry) =>
          this.resolveTemplate({
            templateId: entry.id,
            targetLang: input.targetLang,
            sourceLang: input.sourceLang,
          })
        )
    );
    return resolved.filter(Boolean) as PromptTemplateListItem[];
  }

  static async saveTemplateOverride(input: {
    templateId: string;
    targetLang: string;
    sourceLang?: string;
    content: string;
    fallbackLabel?: string;
  }) {
    const id = String(input.templateId || '').trim();
    if (!isSafeTemplateId(id)) throw new Error('Invalid prompt template id.');

    const content = String(input.content || '').replace(/\u0000/g, '').trim();
    if (!content) throw new Error('Prompt template content is required.');
    if (Buffer.byteLength(content, 'utf8') > MAX_TEMPLATE_BYTES) {
      throw new Error('Prompt template content is too large.');
    }

    const manifest = await this.getMergedManifest();
    const existing = manifest.find((entry) => entry.id === id);
    if (!existing) throw new Error('Prompt template not found.');

    const localRoot = this.getLocalRoot();
    const language = normalizeWritableLanguage(input.targetLang);
    const targetPath = path.resolve(localRoot, id, `${language}.txt`);
    if (!isInside(localRoot, targetPath)) throw new Error('Prompt template path is invalid.');

    await fs.ensureDir(path.dirname(targetPath));
    await fs.writeFile(targetPath, `${content}\n`, 'utf8');

    return this.resolveTemplate({ templateId: id, targetLang: input.targetLang, sourceLang: input.sourceLang });
  }

  static async resetTemplateOverride(input: { templateId: string; targetLang: string; sourceLang?: string }) {
    const id = String(input.templateId || '').trim();
    if (!isSafeTemplateId(id)) throw new Error('Invalid prompt template id.');
    const manifest = await this.getMergedManifest();
    if (!manifest.some((entry) => entry.id === id)) throw new Error('Prompt template not found.');

    const localRoot = this.getLocalRoot();
    const language = normalizeWritableLanguage(input.targetLang);
    const targetPath = path.resolve(localRoot, id, `${language}.txt`);
    if (!isInside(localRoot, targetPath)) throw new Error('Prompt template path is invalid.');
    await fs.remove(targetPath);
    return this.resolveTemplate({ templateId: id, targetLang: input.targetLang, sourceLang: input.sourceLang });
  }
}
