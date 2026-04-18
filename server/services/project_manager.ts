import { getDb, DatabaseSchema } from '../db.js';
import type { Project } from '../../src/types.js';
import { PathManager } from '../path_manager.js';
import { normalizeProjectStatus, PROJECT_STATUS } from '../../src/project_status.js';
import fs from 'fs-extra';
import path from 'path';

export class ProjectManager {
  private static readonly CATEGORY_TO_SUBDIR: Record<string, 'assets' | 'subtitles'> = {
    video: 'assets',
    audio: 'assets',
    other: 'assets',
    subtitle: 'subtitles',
  };

  private static sanitizeProjectName(name: string) {
    const trimmed = String(name ?? '').trim();
    if (!trimmed || trimmed.length > 100) {
      throw new Error('Invalid project name');
    }
    if (/[\\/:*?"<>|]/.test(trimmed)) {
      throw new Error('Invalid project name');
    }
    return trimmed;
  }

  private static normalizeProjectStatuses(projects: Project[]) {
    let changed = false;
    const normalized = projects.map((project) => {
      const nextStatus = normalizeProjectStatus((project as any).status);
      if (project.status !== nextStatus) {
        changed = true;
        return { ...project, status: nextStatus };
      }
      return project;
    });
    return { normalized, changed };
  }

  static async getAllProjects(): Promise<Project[]> {
    const db = await getDb();
    const { normalized, changed } = this.normalizeProjectStatuses(db.data.projects);
    if (changed) {
      db.data.projects = normalized;
      await db.write();
    }
    return db.data.projects;
  }

  static async createProject(name: string, notes?: string): Promise<Project> {
    const db = await getDb();
    const id = Math.random().toString(36).slice(2, 11);
    const sanitizedName = this.sanitizeProjectName(name);
    const sanitizedNotes = typeof notes === 'string' ? notes.slice(0, 5000) : undefined;
    
    // 1. Create project metadata
    const newProject: Project = {
      id,
      name: sanitizedName,
      notes: sanitizedNotes,
      status: PROJECT_STATUS.VIDEO_FETCHING,
      lastUpdated: new Date().toLocaleString(),
    };

    // 2. Create physical folder
    PathManager.getProjectPath(id);

    // 3. Update DB
    db.data.projects.unshift(newProject);
    await db.write();

    return newProject;
  }

  static async updateProject(id: string, updates: Partial<Project>): Promise<Project | null> {
    const db = await getDb();
    const { normalized, changed } = this.normalizeProjectStatuses(db.data.projects);
    if (changed) {
      db.data.projects = normalized;
    }

    const index = db.data.projects.findIndex(p => p.id === id);
    if (index === -1) {
      if (changed) {
        await db.write();
      }
      return null;
    }

    const { id: _ignoredId, ...safeUpdates } = updates as Partial<Project> & { id?: string };
    if (typeof safeUpdates.name === 'string') {
      safeUpdates.name = this.sanitizeProjectName(safeUpdates.name);
    }
    if (typeof safeUpdates.notes === 'string') {
      safeUpdates.notes = safeUpdates.notes.slice(0, 5000);
    }
    if (typeof (safeUpdates as any).status === 'string') {
      (safeUpdates as any).status = normalizeProjectStatus((safeUpdates as any).status);
    }

    db.data.projects[index] = { 
      ...db.data.projects[index], 
      ...safeUpdates, 
      lastUpdated: new Date().toLocaleString() 
    } as Project;
    
    await db.write();
    return db.data.projects[index];
  }

  static async deleteProject(id: string): Promise<boolean> {
    const db = await getDb();
    const index = db.data.projects.findIndex(p => p.id === id);
    if (index === -1) return false;

    // 1. Remove files
    const projectDir = PathManager.getProjectPath(id, { create: false });
    if (await fs.pathExists(projectDir)) {
      await fs.remove(projectDir);
    }

    // 2. Update DB
    db.data.projects.splice(index, 1);
    await db.write();

    return true;
  }

  static async getMaterials(projectId: string): Promise<any[]> {
    const projectPath = PathManager.getProjectPath(projectId, { create: false });
    const materials: any[] = [];

    const scanDir = async (dirName: string, category: string) => {
      const dirPath = path.join(projectPath, dirName);
      if (!(await fs.pathExists(dirPath))) return;

      const files = await fs.readdir(dirPath);
      for (const file of files) {
        const filePath = path.join(dirPath, file);
        const stats = await fs.stat(filePath);
        if (stats.isFile()) {
          materials.push({
            name: file,
            category,
            size: this.formatBytes(stats.size),
            date: stats.mtime.toLocaleString(),
          });
        }
      }
    };

    await scanDir('assets', 'video'); // Defaulting assets to video/audio based on extension later if needed
    // More precise categorization
    for (const m of materials) {
      if (m.category === 'video') {
        const ext = path.extname(m.name).toLowerCase();
        const lowerName = String(m.name || '').toLowerCase();
        if (
          lowerName.startsWith('source_audio.') ||
          ['.mp3', '.wav', '.aac', '.m4a', '.flac', '.ogg', '.opus', '.wma'].includes(ext)
        ) {
          m.category = 'audio';
        } else if (['.srt', '.vtt', '.ass', '.ssa'].includes(ext)) {
          m.category = 'subtitle';
        }
      }
    }

    await scanDir('subtitles', 'subtitle');
    
    // Deduplicate and filter out system files if any
    return materials;
  }

  static async deleteMaterial(projectId: string, category: string, filename: string): Promise<boolean> {
    const subDir = this.CATEGORY_TO_SUBDIR[category];
    if (!subDir) {
      throw new Error('Invalid material category');
    }

    const filePath = PathManager.resolveProjectFile(projectId, subDir, filename, { createProject: false });

    if (await fs.pathExists(filePath)) {
      await fs.remove(filePath);
      return true;
    }
    return false;
  }

  private static formatBytes(bytes: number, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }
}
