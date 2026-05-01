import React from 'react';
import { createPortal } from 'react-dom';
import { HelpCircle } from 'lucide-react';

interface FieldHelpProps {
  ariaLabel: string;
  title: string;
  body: string;
}

interface TooltipPosition {
  left: number;
  top: number;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

export default function FieldHelp({ ariaLabel, title, body }: FieldHelpProps) {
  const buttonRef = React.useRef<HTMLButtonElement | null>(null);
  const tooltipRef = React.useRef<HTMLSpanElement | null>(null);
  const [open, setOpen] = React.useState(false);
  const [position, setPosition] = React.useState<TooltipPosition | null>(null);

  const updatePosition = React.useCallback(() => {
    const button = buttonRef.current;
    if (!button || typeof window === 'undefined') return;

    const rect = button.getBoundingClientRect();
    const gap = 8;
    const viewportPadding = 12;
    const fallbackWidth = 320;
    const tooltipWidth = tooltipRef.current?.offsetWidth || fallbackWidth;
    const tooltipHeight = tooltipRef.current?.offsetHeight || 120;
    const maxLeft = Math.max(viewportPadding, window.innerWidth - tooltipWidth - viewportPadding);
    const maxTop = Math.max(viewportPadding, window.innerHeight - tooltipHeight - viewportPadding);
    const preferredLeft = rect.right + gap;
    const preferredTop = rect.top + rect.height / 2 - tooltipHeight / 2;

    setPosition({
      left: clamp(preferredLeft, viewportPadding, maxLeft),
      top: clamp(preferredTop, viewportPadding, maxTop),
    });
  }, []);

  React.useLayoutEffect(() => {
    if (!open) return;
    updatePosition();
  }, [open, updatePosition]);

  React.useEffect(() => {
    if (!open) return;

    const handleLayoutChange = () => updatePosition();
    window.addEventListener('scroll', handleLayoutChange, true);
    window.addEventListener('resize', handleLayoutChange);
    return () => {
      window.removeEventListener('scroll', handleLayoutChange, true);
      window.removeEventListener('resize', handleLayoutChange);
    };
  }, [open, updatePosition]);

  const tooltip =
    open && typeof document !== 'undefined'
      ? createPortal(
          <span
            ref={tooltipRef}
            role="tooltip"
            style={{
              left: position?.left ?? -9999,
              top: position?.top ?? -9999,
            }}
            className="pointer-events-none fixed z-[9999] w-80 max-w-[min(20rem,calc(100vw-1.5rem))] max-h-[70vh] overflow-y-auto rounded-2xl border border-white/10 bg-surface-container-high px-4 py-3 text-left shadow-2xl shadow-black/35"
          >
            <span className="block text-[11px] font-black uppercase tracking-[0.16em] text-primary">{title}</span>
            <span className="mt-1.5 block text-xs leading-relaxed text-outline">{body}</span>
          </span>,
          document.body
        )
      : null;

  return (
    <span className="relative inline-flex shrink-0">
      <button
        ref={buttonRef}
        type="button"
        aria-label={ariaLabel}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={(event) => event.stopPropagation()}
        className="inline-flex h-6 w-6 items-center justify-center rounded-full border border-white/12 bg-white/[0.04] text-outline transition-colors hover:border-primary/45 hover:bg-primary/12 hover:text-primary focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/55"
      >
        <HelpCircle className="h-3.5 w-3.5" />
      </button>
      {tooltip}
    </span>
  );
}
