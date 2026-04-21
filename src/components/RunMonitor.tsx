import React from 'react';
import { AlertCircle, ChevronDown, ChevronUp, CheckCircle2, Loader2 } from 'lucide-react';

export interface RunMonitorProgressItem {
  label: string;
  progress: number;
  status?: string;
  tone?: 'success' | 'error' | 'normal';
}

export interface RunMonitorField {
  label: string;
  value: React.ReactNode;
  breakAll?: boolean;
}

export interface RunMonitorSection {
  key: string;
  title: string;
  tone?: 'default' | 'error';
  fields?: RunMonitorField[];
  content?: React.ReactNode;
}

export interface RunMonitorBadge {
  label: string;
  tone?: 'default' | 'info' | 'success' | 'warning' | 'error';
}

function RunStatusItem({
  label,
  progress,
  status,
  tone = 'normal',
}: RunMonitorProgressItem) {
  const textClass = tone === 'success' ? 'text-tertiary' : tone === 'error' ? 'text-error' : 'text-primary';
  const barClass = tone === 'success' ? 'bg-tertiary' : tone === 'error' ? 'bg-error' : 'bg-primary';

  return (
    <div className="space-y-3">
      <div className="flex flex-col gap-2 text-[11px] font-bold tracking-wider sm:flex-row sm:items-start sm:justify-between sm:gap-4">
        <span className="text-outline uppercase shrink-0">{label}</span>
        <span className={`${textClass} break-words leading-relaxed sm:max-w-[70%] sm:text-right`}>
          {status || `${Math.round(progress)}%`}
        </span>
      </div>
      <div className="h-1.5 bg-surface-container-lowest rounded-full overflow-hidden">
        <div className={`h-full transition-all duration-500 ${barClass}`} style={{ width: `${progress}%` }} />
      </div>
    </div>
  );
}

export default function RunMonitor({
  title,
  isRunning,
  standbyLabel,
  statusLabel,
  progressItems,
  badges = [],
  message,
  error,
  detailsTitle,
  detailsSummary,
  showDetails,
  onToggleDetails,
  sections,
  compact = false,
  showStatusPillWhenCompact = false,
}: {
  title: string;
  isRunning: boolean;
  standbyLabel: string;
  statusLabel?: string | null;
  progressItems: RunMonitorProgressItem[];
  badges?: RunMonitorBadge[];
  message?: string | null;
  error?: string | null;
  detailsTitle: string;
  detailsSummary?: string | null;
  showDetails: boolean;
  onToggleDetails: () => void;
  sections: RunMonitorSection[];
  compact?: boolean;
  showStatusPillWhenCompact?: boolean;
}) {
  const hasSections = sections.length > 0;
  const maxVisibleBadges = 4;
  const normalizedStatusLabel = String(statusLabel || '').trim();
  const normalizedMessage = String(message || '').trim();
  const shouldRenderMessage =
    normalizedMessage.length > 0 &&
    normalizedMessage !== normalizedStatusLabel &&
    normalizedMessage !== String(standbyLabel || '').trim();
  const visibleBadges = badges.slice(0, maxVisibleBadges);
  const hiddenBadges = badges.slice(maxVisibleBadges);
  const containerClass = compact
    ? 'bg-surface-container-low p-6 rounded-2xl border border-white/5'
    : 'bg-surface-container-high p-6 rounded-[28px] border border-primary/20 shadow-xl';
  const badgeToneClass = (tone: RunMonitorBadge['tone']) => {
    switch (tone) {
      case 'info':
        return 'border-primary/20 bg-primary/10 text-primary';
      case 'success':
        return 'border-tertiary/20 bg-tertiary/10 text-tertiary';
      case 'warning':
        return 'border-yellow-400/25 bg-yellow-400/10 text-yellow-200';
      case 'error':
        return 'border-error/20 bg-error/10 text-error';
      default:
        return 'border-white/10 bg-white/[0.03] text-outline';
    }
  };

  return (
    <section className={containerClass}>
      <div className="mb-6 flex flex-col items-start gap-3 sm:flex-row sm:items-center sm:justify-between">
        <h3 className="flex shrink-0 items-center gap-2 text-sm font-bold text-secondary">
          {isRunning ? <Loader2 className="w-4 h-4 animate-spin text-tertiary" /> : <CheckCircle2 className="w-4 h-4 text-tertiary" />}
          {title}
        </h3>
        {(!compact || showStatusPillWhenCompact) && (
          <div
            className={`inline-flex w-full max-w-full items-start gap-2 rounded-2xl px-3 py-2 text-[11px] font-bold sm:w-auto sm:max-w-[75%] ${
              isRunning ? 'bg-primary/10 text-primary border border-primary/20' : 'bg-white/[0.04] text-outline border border-white/10'
            }`}
          >
            <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${isRunning ? 'bg-primary animate-pulse' : 'bg-outline/40'}`} />
            <span className="min-w-0 break-words leading-5 whitespace-normal">
              {isRunning ? (statusLabel || standbyLabel) : standbyLabel}
            </span>
          </div>
        )}
      </div>

      <div className="space-y-4">
        {badges.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {visibleBadges.map((badge) => (
              <span
                key={`${badge.label}-${badge.tone || 'default'}`}
                className={`inline-flex max-w-full items-center rounded-2xl border px-3 py-1 text-[10px] font-bold leading-4 tracking-wide whitespace-normal break-words ${badgeToneClass(badge.tone)}`}
              >
                {badge.label}
              </span>
            ))}
            {hiddenBadges.length > 0 && (
              <span
                title={hiddenBadges.map((badge) => badge.label).join('\n')}
                className={`inline-flex items-center rounded-2xl border px-3 py-1 text-[10px] font-bold leading-4 tracking-wide ${badgeToneClass('default')}`}
              >
                +{hiddenBadges.length}
              </span>
            )}
          </div>
        )}
        {progressItems.map((item) => (
          <div key={item.label}>
            <RunStatusItem
              label={item.label}
              progress={item.progress}
              status={item.status}
              tone={item.tone}
            />
          </div>
        ))}
        {shouldRenderMessage && (
          <div className="max-h-24 overflow-y-auto custom-scrollbar text-[11px] leading-5 text-outline bg-surface-container-lowest/70 border border-white/5 rounded-lg p-3">
            {normalizedMessage}
          </div>
        )}
        {error && (
          <div className="break-words text-[11px] text-error bg-error/10 border border-error/20 rounded-lg p-3">
            {error}
          </div>
        )}
        {hasSections && (
          <div className="rounded-2xl border border-white/8 bg-surface-container-lowest/50 overflow-hidden">
            <button
              type="button"
              onClick={onToggleDetails}
              className="w-full px-4 py-3 flex flex-col items-start gap-3 text-left transition-colors hover:bg-white/5 sm:flex-row sm:items-center sm:justify-between sm:gap-4"
            >
              <div className="min-w-0">
                <div className="text-xs font-bold text-secondary">{detailsTitle}</div>
                {detailsSummary ? (
                  <div className="mt-1 max-h-10 overflow-hidden break-words text-[11px] leading-5 text-outline">
                    {detailsSummary}
                  </div>
                ) : null}
              </div>
              <div className="shrink-0 inline-flex h-9 w-9 items-center justify-center rounded-full border border-white/10 bg-white/[0.03] text-outline">
                {showDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </div>
            </button>

            {showDetails && (
              <div className="px-4 pb-4 space-y-4 border-t border-white/5 max-h-[420px] overflow-y-auto custom-scrollbar">
                {sections.map((section, index) => {
                  const toneClass =
                    section.tone === 'error'
                      ? 'text-[11px] text-error bg-error/10 border border-error/20 rounded-lg p-3 space-y-2'
                      : 'text-[11px] text-outline bg-surface-container-lowest/70 border border-white/5 rounded-lg p-3 space-y-2';
                  return (
                    <div key={section.key} className={`${toneClass}${index === 0 ? ' mt-4' : ''}`}>
                      <div className={`font-bold flex items-center gap-2 ${section.tone === 'error' ? '' : 'text-secondary'}`}>
                        {section.tone === 'error' ? <AlertCircle className="w-3.5 h-3.5" /> : null}
                        {section.title}
                      </div>
                      {section.fields?.map((field) => (
                        <div
                          key={`${section.key}-${field.label}`}
                          className={`flex flex-col gap-1 sm:grid sm:grid-cols-[minmax(0,140px)_minmax(0,1fr)] sm:gap-3 ${field.breakAll ? 'break-all' : 'break-words'}`}
                        >
                          <span className={`${section.tone === 'error' ? 'text-error' : 'text-secondary'} font-bold`}>
                            {field.label}:
                          </span>
                          <span className="min-w-0 break-words">{field.value}</span>
                        </div>
                      ))}
                      {section.content ? <div className="space-y-2 break-words">{section.content}</div> : null}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
