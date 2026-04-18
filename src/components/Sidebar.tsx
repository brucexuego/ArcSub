import React from 'react';
import { 
  LayoutDashboard, 
  Download, 
  Mic, 
  Languages, 
  PlayCircle, 
  Settings,
  Cpu,
  HardDrive,
  Zap,
  Layers,
  Microchip
} from 'lucide-react';
import { AcceleratorStats, ResourceStats } from '../types';
import { useLanguage } from '../i18n/LanguageContext';
import { getJson } from '../utils/http_client';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  canOpenPlayer: boolean;
  navigationLocked?: boolean;
}

export default function Sidebar({ activeTab, setActiveTab, canOpenPlayer, navigationLocked = false }: SidebarProps) {
  const { t } = useLanguage();
  const [stats, setStats] = React.useState<ResourceStats>({
    cpu: 0,
    ram: 0,
    gpu: 0,
    vram: 0,
    ramUsedGB: 0,
    ramTotalGB: 0,
    cpuModel: '',
    gpuModel: '',
    accelerators: [],
  });
  const systemStatsInFlightRef = React.useRef(false);

  React.useEffect(() => {
    let cancelled = false;
    const highFrequencyMode = navigationLocked;
    const pollIntervalMs = highFrequencyMode ? 1000 : 4000;
    const resourcesApiPath = highFrequencyMode ? '/api/system/resources?fast=1' : '/api/system/resources';

    const loadStats = async () => {
      if (systemStatsInFlightRef.current) return;
      systemStatsInFlightRef.current = true;
      try {
        const data = await getJson<Partial<ResourceStats>>(resourcesApiPath, {
          timeoutMs: highFrequencyMode ? 20000 : 15000,
          retries: highFrequencyMode ? 0 : 1,
          dedupe: false,
        });
        if (!cancelled && data && typeof data === 'object') {
          setStats((prev) => ({ ...prev, ...data }));
        }
      } catch {
        // Keep previous values on transient errors.
      } finally {
        systemStatsInFlightRef.current = false;
      }
    };

    const triggerLoadStats = () => {
      if (typeof document !== 'undefined' && document.visibilityState !== 'visible') return;
      void loadStats();
    };

    triggerLoadStats();
    const interval = setInterval(() => {
      triggerLoadStats();
    }, pollIntervalMs);

    const handleVisibilityChange = () => {
      if (typeof document !== 'undefined' && document.visibilityState === 'visible') {
        triggerLoadStats();
      }
    };
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', handleVisibilityChange);
    }

    return () => {
      cancelled = true;
      clearInterval(interval);
      if (typeof document !== 'undefined') {
        document.removeEventListener('visibilitychange', handleVisibilityChange);
      }
    };
  }, [navigationLocked]);

  const ramDisplay = formatMemoryDisplay(stats.ramUsedGB, stats.ramTotalGB, stats.ram);

  const sourceAccelerators = Array.isArray(stats.accelerators) ? stats.accelerators : [];
  const gpuAccelerators = sourceAccelerators.filter((item) => item.kind === 'gpu');
  const npuAccelerators = sourceAccelerators.filter((item) => item.kind === 'npu');

  const fallbackGpu: AcceleratorStats[] =
    gpuAccelerators.length > 0
      ? []
      : [
          {
            id: 'fallback-gpu-0',
            kind: 'gpu',
            vendor: 'intel',
            model: stats.gpuModel || 'Intel Graphics',
            utilization: stats.gpu,
            vramUsedGB: stats.vramUsedGB,
            vramTotalGB: stats.vramTotalGB,
            taskManagerIndex: 0,
          },
        ];
  const renderedGpuAccelerators = gpuAccelerators.length > 0 ? gpuAccelerators : fallbackGpu;

  const menuItems = [
    { id: 'dashboard', label: t('nav.dashboard'), icon: LayoutDashboard },
    { id: 'downloader', label: t('nav.videoFetcher'), icon: Download },
    { id: 'asr', label: t('nav.speechToText'), icon: Mic },
    { id: 'translate', label: t('nav.textTranslation'), icon: Languages },
    { id: 'player', label: t('nav.videoPlayer'), icon: PlayCircle },
    { id: 'settings', label: t('nav.settings'), icon: Settings },
  ];

  return (
    <aside className="w-64 bg-surface-container flex flex-col h-screen border-r border-white/5 shrink-0">
      <div className="px-5 pt-5 pb-4 mb-3 flex items-center gap-3.5 text-primary">
        <div className="relative shrink-0">
          <div className="absolute inset-0 rounded-[22px] bg-primary-container/25 blur-lg" />
          <img src="/logo.png" alt="Logo" className="relative w-[72px] h-[72px] pointer-events-none drop-shadow-[0_8px_22px_rgba(79,70,229,0.32)]" />
        </div>
        <div>
          <h1 className="text-[30px] font-bold tracking-[-0.04em] leading-none text-white">{t('app.title')}</h1>
          <p className="mt-1 text-xs font-medium leading-snug text-secondary/82">{t('app.subtitle')}</p>
        </div>
      </div>

      <nav className="flex-1 px-4 space-y-2">
        {menuItems.map((item) => {
          const disabled = (item.id === 'player' && !canOpenPlayer) || (navigationLocked && item.id !== activeTab);
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              disabled={disabled}
              className={`w-full flex items-center px-4 py-3.5 rounded-xl transition-all duration-200 group ${
                activeTab === item.id 
                  ? 'bg-primary-container/32 text-white border border-primary/35 shadow-[0_14px_34px_rgba(79,70,229,0.24)]' 
                  : 'text-secondary/85 hover:text-white hover:bg-white/7'
              } ${disabled ? 'opacity-45 cursor-not-allowed border-transparent hover:text-secondary/78 hover:bg-transparent' : ''}`}
            >
              <item.icon className={`w-[18px] h-[18px] mr-3.5 ${activeTab === item.id ? 'text-primary' : 'text-secondary/65 group-hover:text-white'}`} />
              <span className="text-[15px] font-semibold tracking-[-0.01em]">{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="px-4 pt-4 pb-5 border-t border-white/5 space-y-3 overflow-y-auto">
        <div className="px-1">
          <div className="text-[11px] font-semibold tracking-[0.12em] uppercase text-outline/70">{t('system.resources')}</div>
        </div>
        <ResourceGroup
          title="CPU"
          model={stats.cpuModel || 'Unknown CPU'}
          items={[
            { icon: Cpu, label: 'CPU', value: stats.cpu },
            { icon: HardDrive, label: 'RAM', value: stats.ram, displayValue: ramDisplay },
          ]}
        />

        {renderedGpuAccelerators.map((accelerator, index) => {
          const vramDisplay = formatMemoryDisplay(
            accelerator.vramUsedGB,
            accelerator.vramTotalGB,
            accelerator.vramTotalGB && accelerator.vramTotalGB > 0
              ? (Math.max(0, accelerator.vramUsedGB || 0) / accelerator.vramTotalGB) * 100
              : 0
          );
          const vramPercent =
            accelerator.vramTotalGB && accelerator.vramTotalGB > 0
              ? Math.max(0, Math.min(100, (Math.max(0, accelerator.vramUsedGB || 0) / accelerator.vramTotalGB) * 100))
              : 0;
          const displayIndex =
            typeof accelerator.taskManagerIndex === 'number' && Number.isFinite(accelerator.taskManagerIndex)
              ? accelerator.taskManagerIndex
              : index;
          const title = `GPU ${displayIndex}`;

          return (
            <div key={accelerator.id || `gpu-${index}`}>
              <ResourceGroup
                title={title}
                model={accelerator.model || 'Intel Graphics'}
                items={[
                  { icon: Zap, label: 'GPU', value: accelerator.utilization },
                  { icon: Layers, label: 'VRAM', value: vramPercent, displayValue: vramDisplay },
                ]}
              />
            </div>
          );
        })}

        {npuAccelerators.map((accelerator, index) => {
          const title = npuAccelerators.length > 1 ? `NPU ${index + 1}` : 'NPU';
          return (
            <div key={accelerator.id || `npu-${index}`}>
              <ResourceGroup
                title={title}
                model={accelerator.model || 'Intel NPU'}
                items={[
                  { icon: Microchip, label: 'NPU', value: accelerator.utilization },
                ]}
              />
            </div>
          );
        })}
      </div>
    </aside>
  );
}

function formatMemoryDisplay(used?: number, total?: number, fallbackPercent = 0) {
  if (Number.isFinite(total) && (total || 0) > 0) {
    return `${(used || 0).toFixed(1)}/${(total || 0).toFixed(1)}GB`;
  }
  return `${fallbackPercent.toFixed(1)}%`;
}

function ResourceGroup({
  title,
  model,
  items,
}: {
  title: string;
  model: string;
  items: Array<{ icon: any; label: string; value: number; displayValue?: string }>;
}) {
  return (
    <div className="rounded-xl border border-white/6 bg-surface-container-high/45 px-3.5 py-3.5 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)] space-y-3">
      <div className="space-y-1.5">
        <div className="text-[11px] font-semibold text-outline/80 uppercase tracking-[0.18em]">{title}</div>
        <div className="text-[12px] text-secondary/92 leading-snug break-words">{model}</div>
      </div>
      <div className="space-y-3">
        {items.map((item) => (
          <div key={item.label}>
            <StatItem icon={item.icon} label={item.label} value={item.value} displayValue={item.displayValue} />
          </div>
        ))}
      </div>
    </div>
  );
}

function StatItem({ icon: Icon, label, value, displayValue }: { icon: any, label: string, value: number, displayValue?: string }) {
  const safeValue = Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0;
  const percentText = safeValue >= 10 ? `${Math.round(safeValue)}%` : `${safeValue.toFixed(1)}%`;
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center text-[11px] font-semibold tracking-[0.08em]">
        <span className="flex items-center gap-1.5 text-outline/82 uppercase">
          <Icon className="w-3.5 h-3.5" />
          {label}
        </span>
        <span className="text-tertiary font-mono text-[11px]">{displayValue || percentText}</span>
      </div>
      <div className="h-1.5 w-full bg-surface-container-highest/85 rounded-full overflow-hidden">
        <div 
          className="h-full bg-tertiary transition-all duration-500 ease-out" 
          style={{ width: `${safeValue}%` }}
        />
      </div>
    </div>
  );
}
