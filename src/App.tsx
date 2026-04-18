import React from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import { useLanguage } from './i18n/LanguageContext';
import { Project } from './types';
import { getProjectStatusTab, normalizeProjectStatus, PROJECT_STATUS } from './project_status';
import { deleteJson, getJson, patchJson, postJson } from './utils/http_client';

type TaskLockOwner = 'asr' | 'translate';

const loadVideoDownloader = () => import('./components/VideoDownloader');
const loadSpeechToText = () => import('./components/SpeechToText');
const loadTextTranslation = () => import('./components/TextTranslation');
const loadVideoPlayer = () => import('./components/VideoPlayer');
const loadSettings = () => import('./components/Settings');
const loadMaterialModal = () => import('./components/MaterialModal');
const loadNewProjectModal = () => import('./components/NewProjectModal');

const VideoDownloader = React.lazy(loadVideoDownloader);
const SpeechToText = React.lazy(loadSpeechToText);
const TextTranslation = React.lazy(loadTextTranslation);
const VideoPlayer = React.lazy(loadVideoPlayer);
const Settings = React.lazy(loadSettings);
const MaterialModal = React.lazy(loadMaterialModal);
const NewProjectModal = React.lazy(loadNewProjectModal);

export default function App() {
  const { t } = useLanguage();
  const [activeTab, setActiveTab] = React.useState('dashboard');
  const [projects, setProjects] = React.useState<Project[]>([]);
  const initialDataLoadedRef = React.useRef(false);
  const modelSetupRequestRef = React.useRef<Promise<void> | null>(null);
  const [modelSetupStatus, setModelSetupStatus] = React.useState({
    loading: true,
    hasAsr: false,
    hasTranslate: false,
    ready: false,
  });

  const loadProjects = React.useCallback(async () => {
    try {
      const data = await getJson<Project[]>('/api/projects', {
        timeoutMs: 10_000,
        retries: 1,
        dedupeKey: 'projects:list',
      });
      setProjects(data);
    } catch (err) {
      console.error('Failed to fetch projects', err);
    }
  }, []);

  React.useEffect(() => {
    const preload = () => {
      void loadVideoDownloader();
      void loadSpeechToText();
      void loadTextTranslation();
      void loadVideoPlayer();
      void loadSettings();
      void loadMaterialModal();
      void loadNewProjectModal();
    };

    if (typeof window === 'undefined') return;
    const idle = (window as any).requestIdleCallback as ((callback: () => void, options?: { timeout: number }) => number) | undefined;
    const cancelIdle = (window as any).cancelIdleCallback as ((id: number) => void) | undefined;

    if (idle && cancelIdle) {
      const idleId = idle(preload, { timeout: 2000 });
      return () => cancelIdle(idleId);
    }

    const timeoutId = window.setTimeout(preload, 800);
    return () => window.clearTimeout(timeoutId);
  }, []);

  const refreshModelSetupStatus = React.useCallback(async () => {
    if (modelSetupRequestRef.current) {
      return modelSetupRequestRef.current;
    }

    modelSetupRequestRef.current = (async () => {
      try {
        const data = await getJson<{ hasAsr?: boolean; hasTranslate?: boolean; ready?: boolean }>(
          '/api/settings/model-setup',
          {
            timeoutMs: 8_000,
            retries: 1,
            dedupe: false,
            cancelPreviousKey: 'model-setup-status',
          }
        );
        const hasAsr = Boolean(data?.hasAsr);
        const hasTranslate = Boolean(data?.hasTranslate);
        const ready = typeof data?.ready === 'boolean' ? data.ready : hasAsr && hasTranslate;

        setModelSetupStatus({
          loading: false,
          hasAsr,
          hasTranslate,
          ready,
        });
      } catch (err) {
        console.error('Failed to fetch model setup status', err);
        setModelSetupStatus((prev) => ({ ...prev, loading: false }));
      } finally {
        modelSetupRequestRef.current = null;
      }
    })();

    return modelSetupRequestRef.current;
  }, []);

  React.useEffect(() => {
    if (initialDataLoadedRef.current) return;
    initialDataLoadedRef.current = true;
    void loadProjects();
    void refreshModelSetupStatus();
  }, [loadProjects, refreshModelSetupStatus]);

  React.useEffect(() => {
    if (activeTab === 'dashboard' || activeTab === 'settings') {
      void refreshModelSetupStatus();
    }
  }, [activeTab, refreshModelSetupStatus]);

  const [selectedProject, setSelectedProject] = React.useState<Project | null>(null);
  const [isMaterialModalOpen, setIsMaterialModalOpen] = React.useState(false);
  const [isProjectModalOpen, setIsProjectModalOpen] = React.useState(false);
  const [projectEditorTarget, setProjectEditorTarget] = React.useState<Project | null>(null);

  const refreshSelectedProject = React.useCallback(async (projectId: string) => {
    try {
      const latest = await getJson<Project>(`/api/projects/${encodeURIComponent(projectId)}`, {
        cache: 'no-store',
        timeoutMs: 8_000,
        retries: 1,
        dedupe: false,
        cancelPreviousKey: `project:${projectId}`,
      });
      if (!latest || typeof latest.id !== 'string') return;
      setSelectedProject(latest);
      setProjects((prev) => prev.map((project) => (project.id === latest.id ? latest : project)));
    } catch (err) {
      console.error('Failed to refresh selected project', err);
    }
  }, []);

  const selectedProjectHasMedia = Boolean(String(selectedProject?.videoUrl || '').trim() || String(selectedProject?.audioUrl || '').trim());
  const selectedProjectStatus = normalizeProjectStatus(selectedProject?.status);
  const canOpenPlayer = Boolean(
    selectedProject &&
    selectedProjectHasMedia &&
    selectedProjectStatus !== PROJECT_STATUS.VIDEO_FETCHING
  );

  const canOpenPlayerForProject = React.useCallback((project: Project | null | undefined) => {
    if (!project) return false;
    const hasMedia = Boolean(String(project.videoUrl || '').trim() || String(project.audioUrl || '').trim());
    const status = normalizeProjectStatus(project.status);
    return hasMedia && status !== PROJECT_STATUS.VIDEO_FETCHING;
  }, []);
  const [taskLock, setTaskLock] = React.useState<{ active: boolean; owner: TaskLockOwner | null }>({
    active: false,
    owner: null,
  });

  const handleTaskLockChange = React.useCallback((owner: TaskLockOwner, locked: boolean) => {
    setTaskLock((prev) => {
      if (locked) return { active: true, owner };
      if (prev.owner !== owner) return prev;
      return { active: false, owner: null };
    });
  }, []);

  const handleAsrTaskLockChange = React.useCallback(
    (locked: boolean) => {
      handleTaskLockChange('asr', locked);
    },
    [handleTaskLockChange]
  );

  const handleTranslateTaskLockChange = React.useCallback(
    (locked: boolean) => {
      handleTaskLockChange('translate', locked);
    },
    [handleTaskLockChange]
  );

  const releaseLocalRuntime = React.useCallback(async (target: 'asr' | 'translate' | 'all') => {
    try {
      await postJson('/api/local-models/release', { target }, {
        timeoutMs: 12_000,
        retries: 0,
      });
    } catch {
      // Keep tab navigation resilient even if runtime release fails.
    }
  }, []);

  const handleTabChange = React.useCallback((tab: string, targetProject?: Project | null) => {
    void (async () => {
      const projectForNavigation = targetProject ?? selectedProject;
      if (tab === 'player' && !canOpenPlayerForProject(projectForNavigation)) return;
      if (taskLock.active && tab !== activeTab) return;

      if (activeTab === 'asr' && tab !== 'asr') {
        await releaseLocalRuntime('asr');
      }
      if (activeTab === 'translate' && tab !== 'translate') {
        await releaseLocalRuntime('translate');
      }

      React.startTransition(() => {
        if (targetProject) {
          setSelectedProject(targetProject);
        }
      });

      if (tab === 'player' && projectForNavigation?.id) {
        void refreshSelectedProject(projectForNavigation.id);
      }
      React.startTransition(() => {
        setActiveTab(tab);
      });
    })();
  }, [activeTab, canOpenPlayerForProject, refreshSelectedProject, releaseLocalRuntime, selectedProject, taskLock.active]);

  const handleNewProject = React.useCallback(async (name: string, notes: string) => {
    try {
      const newProject = await postJson<Project>('/api/projects', { name, notes }, {
        timeoutMs: 10_000,
      });
      setProjects((prev) => [newProject, ...prev]);
      setSelectedProject(newProject);
      setIsProjectModalOpen(false);
      setProjectEditorTarget(null);
      handleTabChange('downloader');
    } catch (err) {
      console.error('Failed to create project', err);
    }
  }, [handleTabChange]);

  const handleUpdateProjectMetadata = React.useCallback(async (projectId: string, name: string, notes: string) => {
    try {
      const updatedProject = await patchJson<Project>(`/api/projects/${projectId}`, { name, notes }, {
        timeoutMs: 10_000,
      });
      setProjects((prev) => prev.map((project) => (project.id === updatedProject.id ? updatedProject : project)));
      setSelectedProject((prev) => (prev?.id === updatedProject.id ? updatedProject : prev));
      setIsProjectModalOpen(false);
      setProjectEditorTarget(null);
    } catch (err) {
      console.error('Failed to update project metadata', err);
    }
  }, []);

  const handleUpdateProject = async (updates: Partial<Project>) => {
    if (!selectedProject) return null;
    try {
      const updatedProject = await patchJson<Project>(`/api/projects/${selectedProject.id}`, updates, {
        timeoutMs: 10_000,
      });
      setSelectedProject(updatedProject);
      setProjects((prev) => prev.map((p) => (p.id === updatedProject.id ? updatedProject : p)));
      return updatedProject;
    } catch (err) {
      console.error('Failed to update project', err);
      return null;
    }
  };

  const handleDeleteProject = async (id: string) => {
    try {
      const data = await deleteJson<{ success?: boolean }>(`/api/projects/${id}`, {
        timeoutMs: 10_000,
      });
      if (data.success) {
        setProjects((prev) => prev.filter((p) => p.id !== id));
        if (selectedProject?.id === id) {
          setSelectedProject(null);
        }
      }
    } catch (err) {
      console.error('Failed to delete project', err);
    }
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return (
          <Dashboard 
            projects={projects} 
            modelSetupStatus={modelSetupStatus}
            onNewProject={() => {
              setProjectEditorTarget(null);
              setIsProjectModalOpen(true);
            }}
            onOpenSettings={() => handleTabChange('settings')}
            onEditProject={(project) => {
              setProjectEditorTarget(project);
              setSelectedProject(project);
              setIsProjectModalOpen(true);
            }}
            onDeleteProject={handleDeleteProject}
            onManageMaterials={(p) => {
              setSelectedProject(p);
              setIsMaterialModalOpen(true);
            }}
            onNextStep={(p) => {
              handleTabChange(getProjectStatusTab(p.status), p);
            }}
            onQuickJump={(p, tab) => {
              handleTabChange(tab, p);
            }}
          />
        );
      case 'downloader': 
        return <VideoDownloader project={selectedProject} onUpdateProject={handleUpdateProject} onNext={() => handleTabChange('asr')} />;
      case 'asr': 
        return (
          <SpeechToText
            project={selectedProject}
            onUpdateProject={handleUpdateProject}
            onNext={() => handleTabChange('translate')}
            onBack={() => handleTabChange('downloader')}
            onTaskLockChange={handleAsrTaskLockChange}
          />
        );
      case 'translate': 
        return (
          <TextTranslation
            project={selectedProject}
            onUpdateProject={handleUpdateProject}
            onNext={() => handleTabChange('player')}
            onBack={() => handleTabChange('asr')}
            onTaskLockChange={handleTranslateTaskLockChange}
          />
        );
      case 'player': 
        return <VideoPlayer project={selectedProject} />;
      case 'settings': 
        return <Settings project={selectedProject} />;
      default: 
        return (
          <Dashboard
            projects={projects}
            modelSetupStatus={modelSetupStatus}
            onNewProject={() => {
              setProjectEditorTarget(null);
              setIsProjectModalOpen(true);
            }}
            onOpenSettings={() => handleTabChange('settings')}
            onEditProject={() => {}}
            onDeleteProject={handleDeleteProject}
            onManageMaterials={() => {}}
            onNextStep={() => {}}
            onQuickJump={() => {}}
          />
        );
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-surface">
      <Sidebar
        activeTab={activeTab}
        setActiveTab={handleTabChange}
        canOpenPlayer={canOpenPlayer}
        navigationLocked={taskLock.active}
      />
      
      <main className="flex-1 overflow-y-auto custom-scrollbar p-8 lg:p-12">
        <div className="max-w-7xl mx-auto">
          <React.Suspense fallback={<PageLoadingFallback />}>
            {renderContent()}
          </React.Suspense>
        </div>
      </main>

      {isMaterialModalOpen && selectedProject && (
        <React.Suspense fallback={null}>
          <MaterialModal 
            project={selectedProject} 
            onClose={() => setIsMaterialModalOpen(false)} 
          />
        </React.Suspense>
      )}

      {isProjectModalOpen && (
        <React.Suspense fallback={null}>
          <NewProjectModal 
            mode={projectEditorTarget ? 'edit' : 'create'}
            initialName={projectEditorTarget?.name || ''}
            initialNotes={projectEditorTarget?.notes || ''}
            onClose={() => {
              setIsProjectModalOpen(false);
              setProjectEditorTarget(null);
            }}
            onSubmit={(name, notes) => (
              projectEditorTarget
                ? handleUpdateProjectMetadata(projectEditorTarget.id, name, notes)
                : handleNewProject(name, notes)
            )}
          />
        </React.Suspense>
      )}
    </div>
  );

  function PageLoadingFallback() {
    return (
      <div className="relative min-h-[360px] overflow-hidden rounded-[28px] border border-white/6 bg-surface-container/80 px-8 py-16 shadow-[0_24px_80px_rgba(6,10,24,0.34)]">
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute -left-16 top-10 h-40 w-40 rounded-full bg-primary-container/12 blur-3xl" />
          <div className="absolute right-0 top-0 h-48 w-48 rounded-full bg-tertiary/10 blur-3xl" />
          <div className="absolute inset-x-10 bottom-8 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
        </div>
        <div className="relative mx-auto flex max-w-md flex-col items-center justify-center text-center">
          <div className="relative">
            <div className="absolute inset-0 rounded-full bg-primary-container/25 blur-xl" />
            <div className="relative flex h-16 w-16 items-center justify-center rounded-full border border-white/10 bg-surface-container-high/80 shadow-[0_18px_40px_rgba(79,70,229,0.22)]">
              <img src="/logo.png" alt="ArcSub" className="h-10 w-10 animate-pulse pointer-events-none select-none" />
            </div>
            <div className="absolute inset-[-10px] animate-spin rounded-full border border-primary/20 border-t-primary/85" />
          </div>
          <div className="mt-6 text-base font-semibold tracking-[-0.02em] text-secondary">{t('common.loadingWorkspace')}</div>
          <div className="mt-2 max-w-xs text-sm leading-6 text-outline">{t('common.preparingNextPanel')}</div>
        </div>
      </div>
    );
  }
}
