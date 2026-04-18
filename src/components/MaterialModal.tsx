import React, { useState, useEffect, useRef } from 'react';
import { X, FileVideo, FileAudio, FileText, File as FileIcon, UploadCloud, Trash2, Loader2 } from 'lucide-react';
import { Project, Material } from '../types';
import { useLanguage } from '../i18n/LanguageContext';

interface MaterialModalProps {
  project: Project;
  onClose: () => void;
}

export default function MaterialModal({ project, onClose }: MaterialModalProps) {
  const { t } = useLanguage();
  const [materials, setMaterials] = useState<Material[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    void fetchMaterials();
  }, [project.id]);

  const fetchMaterials = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/projects/${project.id}/materials`);
      const data = await res.json();
      setMaterials(data);
    } catch (error) {
      console.error('Failed to fetch materials:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`/api/projects/${project.id}/materials/upload`, {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        await fetchMaterials();
      } else {
        const error = await res.json();
        alert(error.error || t('stt.uploadError'));
      }
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  const handleDelete = async (material: Material) => {
    if (!window.confirm(t('dashboard.deleteConfirm'))) return;

    try {
      const res = await fetch(`/api/projects/${project.id}/materials/${material.category}/${encodeURIComponent(material.name)}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        setMaterials((prev) => prev.filter((m) => !(m.category === material.category && m.name === material.name)));
      }
    } catch (error) {
      console.error('Delete failed:', error);
    }
  };

  const groupedMaterials = {
    video: materials.filter((m) => m.category === 'video'),
    audio: materials.filter((m) => m.category === 'audio'),
    subtitle: materials.filter((m) => m.category === 'subtitle'),
    other: materials.filter((m) => m.category === 'other'),
  };

  const totalMaterialCount = materials.length;
  const categorySummary = [
    { key: 'video', label: t('dashboard.categoryVideo'), count: groupedMaterials.video.length },
    { key: 'audio', label: t('dashboard.categoryAudio'), count: groupedMaterials.audio.length },
    { key: 'subtitle', label: t('dashboard.categorySubtitle'), count: groupedMaterials.subtitle.length },
    { key: 'other', label: t('dashboard.categoryOther'), count: groupedMaterials.other.length },
  ].filter((item) => item.count > 0);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-6">
      <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" onClick={onClose} />

      <div className="relative w-full max-w-[1080px] overflow-hidden rounded-[30px] border border-white/10 bg-surface-container-high shadow-2xl animate-in zoom-in-95 duration-200">
        <div className="flex items-start justify-between border-b border-white/5 bg-white/5 px-7 py-6">
          <div className="space-y-3">
            <div className="space-y-1.5">
              <h3 className="text-[1.75rem] font-bold tracking-tight text-secondary">{t('dashboard.manageMaterials')}</h3>
              <p className="text-sm font-semibold text-primary">{project.name}</p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-full border border-white/8 bg-white/5 px-3 py-1.5 text-[11px] font-semibold text-outline/85">
                {t('stt.totalFiles').replace('{count}', totalMaterialCount.toString())}
              </span>
              {categorySummary.map((item) => (
                <span
                  key={item.key}
                  className="rounded-full border border-white/8 bg-surface-container-highest px-3 py-1.5 text-[11px] font-semibold text-outline/78"
                >
                  {item.label} {item.count}
                </span>
              ))}
            </div>
          </div>
          <button onClick={onClose} className="rounded-xl p-2 text-outline transition-colors hover:bg-white/8 hover:text-white">
            <X className="h-6 w-6" />
          </button>
        </div>

        <div className="max-h-[74vh] overflow-y-auto p-7 custom-scrollbar">
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_320px]">
            <div className="space-y-5">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <h4 className="text-sm font-semibold text-white/90">{t('dashboard.uploadedMaterials')}</h4>
                  <p className="mt-1 text-xs leading-5 text-outline/72">
                    {totalMaterialCount === 0 ? t('dashboard.noMaterials') : t('dashboard.preview')}
                  </p>
                </div>
                <button
                  onClick={handleUploadClick}
                  className="inline-flex shrink-0 items-center gap-2 rounded-xl border border-primary/20 bg-primary/10 px-4 py-2 text-xs font-bold text-primary transition-all hover:bg-primary/15"
                >
                  <UploadCloud className="h-4 w-4" />
                  {t('dashboard.addMaterial')}
                </button>
              </div>

              {loading ? (
                <div className="flex min-h-[320px] flex-col items-center justify-center rounded-[24px] border border-white/6 bg-white/[0.02] text-outline/40">
                  <Loader2 className="mb-4 h-8 w-8 animate-spin" />
                  <p className="text-xs font-bold uppercase tracking-widest">{t('common.loading')}</p>
                </div>
              ) : totalMaterialCount === 0 ? (
                <div className="rounded-[24px] border border-dashed border-white/8 bg-white/[0.02] px-8 py-14 text-center">
                  <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                    <UploadCloud className="h-6 w-6" />
                  </div>
                  <p className="mx-auto max-w-md text-sm leading-6 text-outline/65">{t('dashboard.noMaterials')}</p>
                  <button
                    onClick={handleUploadClick}
                    className="mt-5 inline-flex items-center gap-2 rounded-xl border border-primary/20 bg-primary/10 px-4 py-2 text-xs font-bold text-primary transition-all hover:bg-primary/15"
                  >
                    <UploadCloud className="h-4 w-4" />
                    {t('dashboard.addMaterial')}
                  </button>
                </div>
              ) : (
                <div className="space-y-5">
                  {Object.entries(groupedMaterials).map(([category, items]) => {
                    if (items.length === 0) return null;
                    const label =
                      category === 'video'
                        ? t('dashboard.categoryVideo')
                        : category === 'audio'
                          ? t('dashboard.categoryAudio')
                          : category === 'subtitle'
                            ? t('dashboard.categorySubtitle')
                            : t('dashboard.categoryOther');

                    return (
                      <div key={category} className="space-y-3 rounded-[24px] border border-white/5 bg-white/[0.02] p-4">
                        <div className="flex items-center gap-3 px-1">
                          <h5 className="text-[10px] font-black uppercase tracking-wider text-primary/60">{label}</h5>
                          <span className="rounded-full bg-white/5 px-2 py-0.5 text-[10px] font-semibold text-outline/80">{items.length}</span>
                        </div>
                        <div className="space-y-2">
                          {items.map((material) => (
                            <div key={`${material.category}-${material.name}`}>
                              <MaterialItem
                                material={material}
                                onDelete={() => handleDelete(material)}
                              />
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            <div className="space-y-5 lg:sticky lg:top-0">
              <div className="space-y-4 rounded-[24px] border border-white/6 bg-white/[0.02] p-5">
                <div>
                  <h4 className="text-sm font-semibold text-white/90">{t('dashboard.addMaterial')}</h4>
                  <p className="mt-1 text-xs leading-5 text-outline/72">{t('dashboard.supportedFormats')}</p>
                </div>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  className="hidden"
                  accept=".mp4,.mkv,.mov,.avi,.wmv,.mp3,.wav,.aac,.m4a,.flac,.srt,.vtt,.ass,.ssa"
                />
                <div
                  onClick={handleUploadClick}
                  className={`group flex cursor-pointer flex-col items-center justify-center rounded-[24px] border-2 border-dashed border-white/10 px-6 py-10 text-center transition-all ${
                    uploading ? 'pointer-events-none opacity-50' : 'hover:border-primary/40 hover:bg-primary/5'
                  }`}
                >
                  <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-surface-container-highest transition-transform group-hover:scale-110">
                    {uploading ? (
                      <Loader2 className="h-7 w-7 animate-spin text-primary" />
                    ) : (
                      <UploadCloud className="h-7 w-7 text-primary" />
                    )}
                  </div>
                  <p className="text-sm font-bold leading-6 text-secondary">
                    {uploading ? (
                      t('dashboard.uploading')
                    ) : (
                      <>
                        {t('dashboard.dragDrop')} <span className="text-primary">{t('dashboard.clickUpload')}</span>
                      </>
                    )}
                  </p>
                  <p className="mt-3 text-[10px] font-bold uppercase tracking-widest text-outline">
                    {t('dashboard.supportedFormats')}
                  </p>
                </div>
              </div>
              <div className="rounded-[24px] border border-white/6 bg-surface-container-lowest/70 p-5">
                <div className="text-sm font-semibold text-white/90">{t('dashboard.projectNotes')}</div>
                <div className="mt-3 rounded-2xl border border-white/5 bg-white/[0.02] px-4 py-4 text-sm leading-6 text-outline/80">
                  {project.notes ? (
                    <div className="max-h-[180px] overflow-y-auto custom-scrollbar">{project.notes}</div>
                  ) : (
                    t('dashboard.noProjectDescription')
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="flex justify-end bg-white/5 px-7 py-5">
          <button
            onClick={onClose}
            className="rounded-xl bg-surface-container-highest px-7 py-3 font-bold text-secondary transition-all hover:bg-white/12 hover:text-white"
          >
            {t('dashboard.close')}
          </button>
        </div>
      </div>
    </div>
  );
}

function MaterialItem({
  material,
  onDelete,
}: {
  material: Material;
  onDelete: () => void | Promise<void>;
}) {
  const { t } = useLanguage();
  const getIcon = () => {
    switch (material.category) {
      case 'video':
        return <FileVideo className="h-5 w-5" />;
      case 'audio':
        return <FileAudio className="h-5 w-5" />;
      case 'subtitle':
        return <FileText className="h-5 w-5" />;
      default:
        return <FileIcon className="h-5 w-5" />;
    }
  };

  const getIconBg = () => {
    switch (material.category) {
      case 'video':
        return 'bg-primary/10 text-primary';
      case 'audio':
        return 'bg-tertiary/10 text-tertiary';
      case 'subtitle':
        return 'bg-secondary/10 text-secondary';
      default:
        return 'bg-outline/10 text-outline';
    }
  };

  return (
    <div className="group flex items-center justify-between gap-4 rounded-2xl border border-white/5 bg-surface-container-lowest px-4 py-3.5 transition-all hover:bg-white/5">
      <div className="flex items-center gap-4 overflow-hidden">
        <div className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl ${getIconBg()}`}>
          {getIcon()}
        </div>
        <div className="overflow-hidden">
          <div className="flex flex-wrap items-center gap-2">
            <div className="truncate text-sm font-bold text-secondary" title={material.name}>
              {material.name}
            </div>
            <span className="rounded-full border border-white/8 bg-white/[0.04] px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest text-outline/75">
              {material.category === 'video'
                ? t('dashboard.categoryVideo')
                : material.category === 'audio'
                  ? t('dashboard.categoryAudio')
                  : material.category === 'subtitle'
                    ? t('dashboard.categorySubtitle')
                    : t('dashboard.categoryOther')}
            </span>
          </div>
          <div className="mt-1 flex flex-wrap gap-2 text-[10px] font-bold uppercase tracking-widest text-outline">
            <span className="rounded bg-white/5 px-1.5 py-0.5">{material.size}</span>
            <span>{material.date}</span>
          </div>
        </div>
      </div>
      <div className="ml-4 flex shrink-0 items-center gap-2 opacity-60 transition-opacity group-hover:opacity-100">
        <button
          onClick={onDelete}
          className="inline-flex items-center gap-2 rounded-lg border border-red-400/10 px-3 py-2 text-xs font-bold text-red-300 transition-all hover:bg-red-400/10"
        >
          <Trash2 className="h-4 w-4" />
          <span className="hidden sm:inline">{t('dashboard.delete')}</span>
        </button>
      </div>
    </div>
  );
}
