import React from 'react';
import { Plus, ArrowRight, FolderOpen, Trash2, Activity, CheckCircle2, LayoutGrid, PencilLine, AlertCircle, Settings as SettingsIcon, ChevronDown } from 'lucide-react';
import { Project } from '../types';
import { useLanguage } from '../i18n/LanguageContext';
import {
  PROJECT_STATUS,
  getProjectStatusColorClass,
  getProjectStatusTab,
  getProjectStatusTranslationKey,
  normalizeProjectStatus,
} from '../project_status';

type ProjectRouteTab = 'downloader' | 'asr' | 'translate' | 'player';

interface DashboardProps {
  projects: Project[];
  modelSetupStatus: {
    loading: boolean;
    hasAsr: boolean;
    hasTranslate: boolean;
    ready: boolean;
  };
  onNewProject: () => void;
  onOpenSettings: () => void;
  onEditProject: (project: Project) => void;
  onDeleteProject: (id: string) => void;
  onManageMaterials: (project: Project) => void;
  onNextStep: (project: Project) => void;
  onQuickJump: (project: Project, tab: ProjectRouteTab) => void;
}

export default function Dashboard({
  projects,
  modelSetupStatus,
  onNewProject,
  onOpenSettings,
  onEditProject,
  onDeleteProject,
  onManageMaterials,
  onNextStep,
  onQuickJump,
}: DashboardProps) {
  const INITIAL_RENDER_BATCH = 24;
  const PROGRESSIVE_RENDER_BATCH = 48;
  const { t } = useLanguage();
  const [activeFilter, setActiveFilter] = React.useState<'all' | 'progress' | 'completed'>('all');
  const [openQuickJumpProjectId, setOpenQuickJumpProjectId] = React.useState<string | null>(null);
  const deferredProjects = React.useDeferredValue(projects);
  const deferredFilter = React.useDeferredValue(activeFilter);
  const [renderCount, setRenderCount] = React.useState(INITIAL_RENDER_BATCH);
  const totalProjects = deferredProjects.length;
  const completedProjects = React.useMemo(
    () =>
      deferredProjects.filter(
        (project) => normalizeProjectStatus(project.status) === PROJECT_STATUS.COMPLETED
      ).length,
    [deferredProjects]
  );
  const inProgressProjects = totalProjects - completedProjects;

  const summaryCards = [
    {
      id: 'all',
      icon: LayoutGrid,
      label: t('dashboard.totalProjects'),
      value: totalProjects,
      accent: 'text-primary',
      surface: 'bg-primary/10 border-primary/15',
      isActive: activeFilter === 'all',
    },
    {
      id: 'progress',
      icon: Activity,
      label: t('dashboard.inProgress'),
      value: inProgressProjects,
      accent: 'text-secondary',
      surface: 'bg-secondary/10 border-secondary/12',
      isActive: activeFilter === 'progress',
    },
    {
      id: 'completed',
      icon: CheckCircle2,
      label: t('dashboard.completedProjects'),
      value: completedProjects,
      accent: 'text-tertiary',
      surface: 'bg-tertiary/10 border-tertiary/15',
      isActive: activeFilter === 'completed',
    },
  ] as const;

  const visibleProjects = React.useMemo(
    () =>
      deferredProjects.filter((project) => {
        const status = normalizeProjectStatus(project.status);
        if (deferredFilter === 'progress') return status !== PROJECT_STATUS.COMPLETED;
        if (deferredFilter === 'completed') return status === PROJECT_STATUS.COMPLETED;
        return true;
      }),
    [deferredFilter, deferredProjects]
  );
  const renderedProjects = React.useMemo(
    () => visibleProjects.slice(0, renderCount),
    [renderCount, visibleProjects]
  );

  React.useEffect(() => {
    setRenderCount(Math.min(INITIAL_RENDER_BATCH, visibleProjects.length));
  }, [deferredFilter, visibleProjects.length]);

  React.useEffect(() => {
    if (!openQuickJumpProjectId) return;
    if (!visibleProjects.some((project) => project.id === openQuickJumpProjectId)) {
      setOpenQuickJumpProjectId(null);
    }
  }, [openQuickJumpProjectId, visibleProjects]);

  React.useEffect(() => {
    if (renderCount >= visibleProjects.length) return;

    let cancelled = false;
    const schedule =
      typeof window !== 'undefined' && typeof (window as any).requestIdleCallback === 'function'
        ? (callback: () => void) => {
            const id = (window as any).requestIdleCallback(callback, { timeout: 300 });
            return () => (window as any).cancelIdleCallback?.(id);
          }
        : (callback: () => void) => {
            const id = window.setTimeout(callback, 50);
            return () => window.clearTimeout(id);
          };

    const cancel = schedule(() => {
      if (cancelled) return;
      setRenderCount((current) =>
        Math.min(visibleProjects.length, current + PROGRESSIVE_RENDER_BATCH)
      );
    });

    return () => {
      cancelled = true;
      cancel();
    };
  }, [renderCount, visibleProjects.length]);

  const currentFilterLabel =
    summaryCards.find((card) => card.id === activeFilter)?.label ?? t('dashboard.totalProjects');
  const missingSetupItems = [
    !modelSetupStatus.hasAsr ? t('dashboard.setupMissingAsr') : null,
    !modelSetupStatus.hasTranslate ? t('dashboard.setupMissingTranslate') : null,
  ].filter(Boolean) as string[];
  const showSetupCallout = !modelSetupStatus.loading && !modelSetupStatus.ready && projects.length > 0;

  const handleDelete = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (window.confirm(t('dashboard.deleteConfirm'))) {
      onDeleteProject(id);
    }
  };

  const stopCardClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  const getActionLabel = (project: Project) => {
    switch (normalizeProjectStatus(project.status)) {
      case PROJECT_STATUS.VIDEO_FETCHING:
        return t('dashboard.ctaFetcher');
      case PROJECT_STATUS.SPEECH_TO_TEXT:
        return t('dashboard.ctaAsr');
      case PROJECT_STATUS.TEXT_TRANSLATION:
        return t('dashboard.ctaTranslate');
      case PROJECT_STATUS.VIDEO_PLAYER:
      case PROJECT_STATUS.COMPLETED:
        return t('dashboard.ctaReview');
      default:
        return t('dashboard.ctaFetcher');
    }
  };

  const getUnlockedTabs = React.useCallback((project: Project): ProjectRouteTab[] => {
    const status = normalizeProjectStatus(project.status);
    const hasMedia = Boolean(String(project.videoUrl || '').trim() || String(project.audioUrl || '').trim());
    switch (status) {
      case PROJECT_STATUS.VIDEO_FETCHING:
        return ['downloader'];
      case PROJECT_STATUS.SPEECH_TO_TEXT:
        return ['downloader', 'asr'];
      case PROJECT_STATUS.TEXT_TRANSLATION:
        return ['downloader', 'asr', 'translate'];
      case PROJECT_STATUS.VIDEO_PLAYER:
      case PROJECT_STATUS.COMPLETED: {
        const base: ProjectRouteTab[] = ['downloader', 'asr', 'translate'];
        if (hasMedia) base.push('player');
        return base;
      }
      default:
        return ['downloader'];
    }
  }, []);

  const tabToLabel = React.useCallback((tab: ProjectRouteTab) => {
    switch (tab) {
      case 'downloader':
        return t('nav.videoFetcher');
      case 'asr':
        return t('nav.speechToText');
      case 'translate':
        return t('nav.textTranslation');
      case 'player':
        return t('nav.videoPlayer');
      default:
        return t('nav.videoFetcher');
    }
  }, [t]);

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="flex flex-col gap-5 px-2 xl:flex-row xl:items-end xl:justify-between">
        <div className="space-y-4">
          <h2 className="text-4xl font-black text-white tracking-tight uppercase italic">{t('dashboard.title')}</h2>
          <p className="max-w-2xl text-sm font-medium leading-6 text-outline/80 md:text-[15px]">{t('dashboard.subtitle')}</p>
          <div className="grid gap-3 sm:grid-cols-3">
            {summaryCards.map((card) => (
              <button
                key={card.id}
                onClick={() => setActiveFilter(card.id)}
                className={`min-w-[172px] rounded-2xl border px-4 py-3 text-left shadow-[inset_0_1px_0_rgba(255,255,255,0.03)] transition-all ${
                  card.isActive
                    ? `${card.surface} ring-1 ring-white/12 shadow-[0_20px_45px_rgba(12,18,36,0.2)]`
                    : 'border-white/6 bg-white/[0.02] hover:border-white/10 hover:bg-white/[0.04]'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className={`text-[11px] font-semibold uppercase tracking-[0.18em] ${card.isActive ? 'text-white/88' : 'text-outline/80'}`}>{card.label}</span>
                  <card.icon className={`h-4 w-4 ${card.accent}`} />
                </div>
                <div className="mt-2 text-[1.75rem] font-black tracking-[-0.04em] text-white">{card.value}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-col items-start gap-3 xl:items-end">
          {!modelSetupStatus.loading && !modelSetupStatus.ready && (
            <button
              onClick={onOpenSettings}
              className="group inline-flex items-center gap-3 rounded-2xl border border-primary/16 bg-white/5 px-6 py-3 font-bold text-secondary transition-all hover:border-primary/30 hover:bg-primary/10 hover:text-white"
            >
              <SettingsIcon className="h-4 w-4 text-primary" />
              {t('dashboard.setupOpenSettings')}
            </button>
          )}
          <button
            onClick={onNewProject}
            disabled={!modelSetupStatus.loading && !modelSetupStatus.ready}
            title={!modelSetupStatus.loading && !modelSetupStatus.ready ? t('dashboard.setupBeforeCreate') : t('dashboard.newProject')}
            className="group inline-flex items-center gap-3 self-start rounded-2xl border border-white/5 bg-primary-container px-7 py-3.5 font-black text-white shadow-2xl shadow-primary-container/20 transition-all active:scale-95 hover:bg-white hover:text-surface disabled:cursor-not-allowed disabled:opacity-45 disabled:hover:bg-primary-container disabled:hover:text-white xl:self-auto"
          >
            <div className="flex h-6 w-6 items-center justify-center rounded-lg bg-white/10 group-hover:bg-surface/10">
              <Plus className="h-4 w-4" />
            </div>
            {t('dashboard.newProject')}
          </button>
          {!modelSetupStatus.loading && !modelSetupStatus.ready && (
            <p className="max-w-xs text-right text-xs leading-5 text-outline/72">
              {t('dashboard.setupBeforeCreate')}
            </p>
          )}
        </div>
      </div>

      {showSetupCallout && (
        <div className="rounded-[28px] border border-primary/12 bg-primary/6 px-5 py-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-start gap-3.5">
              <div className="mt-0.5 flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl border border-primary/18 bg-primary-container/14 text-primary">
                <AlertCircle className="h-5 w-5" />
              </div>
              <div className="space-y-2">
                <div className="text-lg font-bold tracking-tight text-white">{t('dashboard.setupRequiredTitle')}</div>
                <p className="max-w-2xl text-sm leading-6 text-outline/84">{t('dashboard.setupRequiredSubtitle')}</p>
                <div className="flex flex-wrap gap-2">
                  {missingSetupItems.map((item) => (
                    <span
                      key={item}
                      className="rounded-full border border-white/8 bg-white/5 px-3 py-1.5 text-[11px] font-semibold text-white/86"
                    >
                      {item}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <button
              onClick={onOpenSettings}
              className="inline-flex items-center justify-center gap-2 rounded-2xl border border-primary/16 bg-primary-container px-5 py-3 font-bold text-white transition-all hover:brightness-110"
            >
              <SettingsIcon className="h-4 w-4" />
              {t('dashboard.setupOpenSettings')}
            </button>
          </div>
        </div>
      )}

      <div className="relative overflow-hidden rounded-[32px] border border-white/5 bg-surface-container shadow-2xl">
        <div className="pointer-events-none absolute right-0 top-0 h-64 w-64 bg-primary/5 blur-[100px]" />

        <div className="relative z-10 space-y-4 p-5 md:p-6">
          <div className="flex flex-col gap-2 rounded-2xl border border-white/5 bg-white/[0.02] px-4 py-3 md:flex-row md:items-center md:justify-between">
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-full border border-white/6 bg-surface-container-highest px-3 py-1.5 text-[11px] font-semibold text-white/84">
                {currentFilterLabel}
              </span>
              <span className="text-sm font-medium text-outline/76">
                {visibleProjects.length} / {totalProjects}
              </span>
            </div>
            {activeFilter !== 'all' && (
              <button
                onClick={() => setActiveFilter('all')}
                className="inline-flex w-fit items-center whitespace-nowrap rounded-xl border border-white/6 bg-surface-container-highest px-4 py-2 text-sm font-semibold text-secondary transition-all hover:bg-white/10 hover:text-white"
              >
                {t('dashboard.totalProjects')}
              </button>
            )}
          </div>

          {renderedProjects.map((project) => {
            const description = String(project.notes || '').trim() || t('dashboard.noProjectDescription');
            const statusLabel = t(getProjectStatusTranslationKey(project.status));
            const actionLabel = getActionLabel(project);
            const descriptionIsEmpty = !String(project.notes || '').trim();
            const quickJumpTabs = getUnlockedTabs(project);
            const quickJumpOptions = quickJumpTabs.map((tab) => ({
              tab,
              label: tabToLabel(tab),
            }));
            const isQuickJumpOpen = openQuickJumpProjectId === project.id;
            const recommendedTab = getProjectStatusTab(project.status);

            return (
              <div
                key={project.id}
                onClick={() => {
                  setOpenQuickJumpProjectId(null);
                  onNextStep(project);
                }}
                className="group cursor-pointer rounded-[26px] border border-white/6 bg-white/[0.02] px-4 py-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.02)] transition-all hover:border-primary/18 hover:bg-white/[0.035] hover:shadow-[0_24px_50px_rgba(12,18,36,0.22)] md:px-5"
              >
                <div className="grid gap-4 2xl:grid-cols-[minmax(0,1fr)_320px] 2xl:gap-5">
                  <div className="min-w-0">
                    <div className="flex items-start gap-3.5">
                      <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl border border-white/8 bg-gradient-to-br from-primary-container/25 via-primary/12 to-transparent text-lg font-black text-white shadow-[0_14px_30px_rgba(79,70,229,0.16)]">
                        {String(project.name || '?').trim().charAt(0).toUpperCase()}
                      </div>

                      <div className="min-w-0 flex-1">
                        <div className="flex flex-col gap-2 xl:flex-row xl:items-center xl:justify-between">
                          <div className="min-w-0 truncate text-lg font-extrabold tracking-tight text-white transition-colors group-hover:text-primary md:text-[1.15rem]">
                            {project.name}
                          </div>
                          <div className="flex shrink-0 flex-wrap items-center gap-2 xl:justify-end">
                            <div className="flex w-fit items-center gap-2 rounded-full border border-white/6 bg-white/5 px-3 py-1.5 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
                              <div className={`h-2 w-2 rounded-full ${getProjectStatusColorClass(project.status)} shadow-[0_0_10px_currentColor]`} />
                              <span className="text-[11px] font-semibold tracking-[0.08em] text-white/82">
                                {statusLabel}
                              </span>
                            </div>
                            <span className="rounded-full border border-white/6 bg-surface-container-highest px-3 py-1.5 text-[11px] font-semibold text-outline/84">
                              {t('dashboard.lastUpdated')}: {project.lastUpdated}
                            </span>
                          </div>
                        </div>

                        <div className="mt-3 rounded-2xl border border-white/5 bg-surface-container-lowest/70 px-4 py-3.5">
                          <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-outline/62">
                            {t('dashboard.description')}
                          </div>
                          <div className={`mt-1.5 min-h-[40px] line-clamp-2 text-sm leading-5.5 ${descriptionIsEmpty ? 'text-outline/62' : 'text-outline/86'}`}>
                            {description}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-col gap-2.5 2xl:justify-between">
                    <div className="relative">
                      <button
                        onClick={(e) => {
                          stopCardClick(e);
                          setOpenQuickJumpProjectId((current) => (current === project.id ? null : project.id));
                        }}
                        title={t('dashboard.quickJump')}
                        className="inline-flex w-full items-center justify-between gap-3 rounded-2xl border border-primary/18 bg-primary-container px-4 py-3 text-left text-white shadow-xl shadow-primary-container/12 transition-all hover:brightness-110"
                      >
                        <div className="min-w-0">
                          <div className="truncate text-sm font-bold tracking-[-0.01em]">
                            {t('dashboard.quickJump')}
                          </div>
                          <div className="mt-0.5 truncate text-[11px] text-white/72">
                            {actionLabel}
                          </div>
                        </div>
                        <ChevronDown className={`h-4 w-4 shrink-0 transition-transform ${isQuickJumpOpen ? 'rotate-180' : ''}`} />
                      </button>

                      {isQuickJumpOpen && (
                        <div
                          onClick={stopCardClick}
                          className="mt-2 rounded-2xl border border-white/10 bg-surface-container-high p-2 shadow-2xl"
                        >
                          {quickJumpOptions.map((option) => (
                            <button
                              key={`${project.id}-${option.tab}`}
                              type="button"
                              onClick={(e) => {
                                stopCardClick(e);
                                setOpenQuickJumpProjectId(null);
                                onQuickJump(project, option.tab);
                              }}
                              className="flex w-full items-center justify-between rounded-xl px-3 py-2.5 text-left text-sm font-semibold text-secondary transition-colors hover:bg-white/10 hover:text-white"
                            >
                              <span>{option.label}</span>
                              {option.tab === recommendedTab ? (
                                <span className="rounded-full border border-white/10 bg-white/10 px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest text-white/90">
                                  {t('dashboard.quickJumpRecommended')}
                                </span>
                              ) : (
                                <ArrowRight className="h-3.5 w-3.5 text-outline/70" />
                              )}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>

                    <div className="flex flex-wrap items-center justify-end gap-2.5">
                      <button
                        onClick={(e) => {
                          stopCardClick(e);
                          setOpenQuickJumpProjectId(null);
                          onEditProject(project);
                        }}
                        aria-label={t('dashboard.editProject')}
                        title={t('dashboard.editProject')}
                        className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-white/6 bg-surface-container-highest text-secondary transition-all hover:bg-white/10 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/55"
                      >
                        <PencilLine className="h-4 w-4" />
                      </button>
                      <button
                        onClick={(e) => {
                          stopCardClick(e);
                          setOpenQuickJumpProjectId(null);
                          onManageMaterials(project);
                        }}
                        aria-label={t('dashboard.manageMaterials')}
                        title={t('dashboard.manageMaterials')}
                        className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-white/6 bg-surface-container-highest text-secondary transition-all hover:bg-white/10 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/55"
                      >
                        <FolderOpen className="h-4 w-4" />
                      </button>
                      <button
                        onClick={(e) => {
                          setOpenQuickJumpProjectId(null);
                          handleDelete(e, project.id);
                        }}
                        aria-label={t('dashboard.delete')}
                        title={t('dashboard.delete')}
                        className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-white/6 bg-surface-container-highest text-outline/45 transition-all hover:bg-red-400/10 hover:text-red-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-400/45"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {visibleProjects.length === 0 && (
          <div className="relative z-10 p-24 text-center">
            <div className="mx-auto mb-6 flex h-24 w-24 items-center justify-center rounded-[32px] border border-white/5 bg-white/5 shadow-[0_18px_40px_rgba(12,18,36,0.3)]">
              <FolderOpen className="h-10 w-10 text-outline/30" />
            </div>
            <p className="text-sm font-semibold uppercase tracking-[0.18em] text-outline/52">
              {totalProjects === 0
                ? (!modelSetupStatus.loading && !modelSetupStatus.ready ? t('dashboard.setupRequiredTitle') : t('dashboard.emptyState'))
                : `${currentFilterLabel}: 0`}
            </p>
            {totalProjects === 0 && !modelSetupStatus.loading && !modelSetupStatus.ready && (
              <p className="mx-auto mt-4 max-w-xl text-sm leading-6 text-outline/72">
                {t('dashboard.setupRequiredSubtitle')}
              </p>
            )}
            {totalProjects === 0 && !modelSetupStatus.loading && !modelSetupStatus.ready ? (
              <button
                onClick={onOpenSettings}
                className="mt-6 inline-flex items-center gap-2 rounded-xl border border-primary/16 bg-primary-container px-5 py-3 text-sm font-semibold text-white transition-all hover:brightness-110"
              >
                <SettingsIcon className="h-4 w-4" />
                {t('dashboard.setupOpenSettings')}
              </button>
            ) : totalProjects > 0 && activeFilter !== 'all' && (
              <button
                onClick={() => setActiveFilter('all')}
                className="mt-6 inline-flex items-center whitespace-nowrap rounded-xl border border-white/6 bg-surface-container-highest px-4 py-3 text-sm font-semibold text-secondary transition-all hover:bg-white/10 hover:text-white"
              >
                {t('dashboard.totalProjects')}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
