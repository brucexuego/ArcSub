import { JSONFilePreset } from 'lowdb/node';
import { Project } from '../src/types';
import { PathManager } from './path_manager.js';

export interface DatabaseSchema {
  projects: Project[];
  settings: {
    projectsPath: string;
    lastProject?: string;
    asrModels: any[];
    translateModels: any[];
    localModels: {
      asrSelectedId: string;
      translateSelectedId: string;
      installed: any[];
    };
    interfaceLanguage: string;
  };
}

const defaultData: DatabaseSchema = {
  projects: [],
  settings: {
    projectsPath: PathManager.getSettingsProjectsPathValue(),
    asrModels: [],
    translateModels: [],
    localModels: {
      asrSelectedId: '',
      translateSelectedId: '',
      installed: [],
    },
    interfaceLanguage: 'zh-tw'
  },
};

let dbPromise: ReturnType<typeof JSONFilePreset<DatabaseSchema>> | null = null;

export async function getDb() {
  if (!dbPromise) {
    dbPromise = JSONFilePreset<DatabaseSchema>(PathManager.getDbPath(), defaultData);
  }
  const db = await dbPromise;
  const expectedProjectsPath = PathManager.getSettingsProjectsPathValue();
  if (db.data.settings.projectsPath !== expectedProjectsPath) {
    db.data.settings.projectsPath = expectedProjectsPath;
    await db.write();
  }
  return db;
}
