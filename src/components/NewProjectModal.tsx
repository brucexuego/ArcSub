import React from 'react';
import { X, Plus, FilePlus, AlertCircle, PencilLine } from 'lucide-react';
import { useLanguage } from '../i18n/LanguageContext';
import { sanitizeInput, isValidProjectName } from '../utils/security';

interface NewProjectModalProps {
  onClose: () => void;
  onSubmit: (name: string, notes: string) => void | Promise<void>;
  initialName?: string;
  initialNotes?: string;
  mode?: 'create' | 'edit';
}

export default function NewProjectModal({
  onClose,
  onSubmit,
  initialName = '',
  initialNotes = '',
  mode = 'create',
}: NewProjectModalProps) {
  const { t } = useLanguage();
  const isEdit = mode === 'edit';
  const [name, setName] = React.useState(initialName);
  const [notes, setNotes] = React.useState(initialNotes);
  const [error, setError] = React.useState<string | null>(null);

  const handleSubmit = () => {
    const sanitizedName = sanitizeInput(name);
    if (!isValidProjectName(sanitizedName)) {
      setError(t('dashboard.invalidProjectName'));
      return;
    }
    void onSubmit(sanitizedName, notes);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-6">
      <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" onClick={onClose} />

      <div className="relative w-full max-w-[920px] overflow-hidden rounded-[28px] border border-white/10 bg-surface-container-high shadow-2xl animate-in zoom-in-95 duration-200">
        <div className="border-b border-white/5 bg-white/5 px-7 py-6">
          <div className="flex items-start justify-between gap-6">
            <div className="flex items-start gap-4">
              <div className="mt-1 flex h-14 w-14 items-center justify-center rounded-2xl border border-primary/20 bg-primary-container/18 shadow-[0_16px_30px_rgba(79,70,229,0.18)]">
                {isEdit ? <PencilLine className="h-7 w-7 text-primary" /> : <FilePlus className="h-7 w-7 text-primary" />}
              </div>
              <div className="max-w-2xl">
                <h2 className="text-[1.75rem] font-bold tracking-tight text-secondary">
                  {isEdit ? t('dashboard.editProject') : t('dashboard.newProject')}
                </h2>
                <p className="mt-2 text-sm leading-6 text-outline">
                  {isEdit ? t('dashboard.editProjectSubtitle') : t('dashboard.startNewProject')}
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="rounded-xl p-2 text-outline transition-colors hover:bg-white/8 hover:text-white"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        <div className="grid gap-5 px-7 py-6 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
          <div className="rounded-2xl border border-white/6 bg-surface-container-lowest/70 p-5">
            <label className="mb-3 block text-sm font-semibold text-white/88">
              {t('dashboard.projectName')}
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => {
                setName(e.target.value);
                if (error) setError(null);
              }}
              placeholder={t('dashboard.projectNamePlaceholder')}
              className={`w-full rounded-xl border bg-surface-container-lowest px-4 py-3.5 text-secondary outline-none transition-all placeholder:text-outline/30 focus:ring-2 focus:ring-primary-container ${
                error ? 'border-error/50' : 'border-white/10'
              }`}
              autoFocus
            />
            <p className="mt-3 text-xs leading-5 text-outline/78">{t('dashboard.projectNameHint')}</p>
            {error && (
              <div className="mt-3 flex items-center gap-2 text-xs font-bold text-error animate-in slide-in-from-top-1 duration-200">
                <AlertCircle className="h-3.5 w-3.5" />
                {error}
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-white/6 bg-surface-container-lowest/70 p-5">
            <label className="mb-3 block text-sm font-semibold text-white/88">
              {t('dashboard.projectNotes')}
            </label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(sanitizeInput(e.target.value))}
              placeholder={t('dashboard.projectNotesPlaceholder')}
              className="min-h-[220px] w-full resize-none rounded-xl border border-white/10 bg-surface-container-lowest px-4 py-3.5 text-secondary outline-none transition-all placeholder:text-outline/30 focus:ring-2 focus:ring-primary-container"
              rows={7}
            />
            <p className="mt-3 text-xs leading-5 text-outline/78">{t('dashboard.projectNotesHint')}</p>
          </div>
        </div>

        <div className="flex items-center justify-end gap-3 border-t border-white/5 bg-white/5 px-7 py-5">
          <button
            onClick={onClose}
            className="rounded-xl px-5 py-3 text-sm font-bold text-outline transition-all hover:bg-white/8 hover:text-white"
          >
            {t('dashboard.cancel')}
          </button>
          <button
            onClick={handleSubmit}
            disabled={!name.trim()}
            className="flex min-w-[168px] items-center justify-center gap-2 rounded-xl bg-gradient-to-br from-primary-container to-primary px-7 py-3 font-bold text-white shadow-lg shadow-primary-container/20 transition-all hover:scale-[1.02] active:scale-[0.98] disabled:scale-100 disabled:opacity-50"
          >
            {isEdit ? <PencilLine className="h-5 w-5" /> : <Plus className="h-5 w-5" />}
            {isEdit ? t('dashboard.saveProject') : t('dashboard.createProject')}
          </button>
        </div>
      </div>
    </div>
  );
}
