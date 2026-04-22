import React from 'react';
import { FileText, FolderOpen, Loader2, MonitorPlay, SlidersHorizontal, Upload, X } from 'lucide-react';
import Artplayer from 'artplayer';
import type { Material, Project } from '../types';
import { useLanguage } from '../i18n/LanguageContext';

interface VideoPlayerProps {
  project: Project | null;
}

type SubtitleMode = 'ai' | 'original' | 'project' | 'none';
type VideoSourceProvider = 'none' | 'local' | 'online';
type SubtitleInteractionMode = 'none' | 'move' | 'resize-width' | 'resize-height';

interface SubtitleCue {
  start: number;
  end: number;
  text: string;
}

interface VideoSourceOption {
  id: string;
  label: string;
  src: string;
  provider: VideoSourceProvider;
  detail?: string;
  resolution?: string;
  playable: boolean;
}

const SUBTITLE_EXTENSIONS = new Set(['.srt', '.vtt', '.ass', '.ssa', '.txt', '.json']);
const VIDEO_EXTENSIONS = new Set(['.mp4', '.webm', '.mkv', '.mov', '.m4v', '.avi', '.flv', '.wmv', '.ts', '.m3u8']);
const AUDIO_EXTENSIONS = new Set(['.m4a', '.mp3', '.wav', '.aac', '.flac', '.ogg', '.opus', '.wma']);
const SUBTITLE_WIDTH_MIN_PCT = 28;
const SUBTITLE_WIDTH_MAX_PCT = 92;
const SUBTITLE_HEIGHT_MIN_PCT = 6;
const SUBTITLE_HEIGHT_MAX_PCT = 36;
const SUBTITLE_POSITION_MIN_PCT = 5;
const SUBTITLE_POSITION_MAX_PCT = 95;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function formatVttTime(time: number) {
  const h = Math.floor(time / 3600);
  const m = Math.floor((time % 3600) / 60);
  const s = (time % 60).toFixed(3);
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(6, '0')}`;
}

function generateVttBlob(cues: SubtitleCue[]) {
  let vtt = 'WEBVTT\n\n';
  cues.forEach((cue) => {
    vtt += `${formatVttTime(cue.start)} --> ${formatVttTime(cue.end)}\n${cue.text}\n\n`;
  });
  return URL.createObjectURL(new Blob([vtt], { type: 'text/vtt' }));
}

function safeDecodeUriComponent(raw: string) {
  const value = String(raw || '');
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function getFileExt(fileName: string) {
  const normalized = String(fileName || '')
    .replace(/\\/g, '/')
    .split('?')[0]
    .split('#')[0]
    .trim();
  const baseName = normalized.slice(normalized.lastIndexOf('/') + 1);
  const idx = baseName.lastIndexOf('.');
  if (idx < 0) return '';
  return baseName.slice(idx).toLowerCase();
}

function formatBytes(value: number) {
  if (!Number.isFinite(value) || value <= 0) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = value;
  let idx = 0;
  while (size >= 1024 && idx < units.length - 1) {
    size /= 1024;
    idx += 1;
  }
  return `${size.toFixed(size >= 100 || idx === 0 ? 0 : 1)} ${units[idx]}`;
}

function formatTime(time: number) {
  const safe = Number.isFinite(time) ? Math.max(0, time) : 0;
  const h = Math.floor(safe / 3600);
  const m = Math.floor((safe % 3600) / 60);
  const s = Math.floor(safe % 60);
  if (h > 0) {
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function formatDelayLabel(delayMs: number) {
  const seconds = delayMs / 1000;
  return `${seconds >= 0 ? '+' : ''}${seconds.toFixed(1)}s`;
}

function toRgbaFromHex(rawHex: string, opacityPct: number) {
  const value = String(rawHex || '').trim();
  const match = value.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  const normalized = match ? value.toUpperCase() : '#000000';
  const hex = normalized.length === 4
    ? normalized.split('').map((ch, idx) => (idx === 0 ? ch : ch + ch)).join('')
    : normalized;
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const alpha = clamp(opacityPct, 0, 100) / 100;
  return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(2)})`;
}

function formatUploadDate(raw: string) {
  const value = String(raw || '').trim();
  if (!value) return '';
  const compact = value.match(/^(\d{4})(\d{2})(\d{2})$/);
  if (compact) {
    return `${compact[1]}-${compact[2]}-${compact[3]}`;
  }
  const unixTs = Number(value);
  if (Number.isFinite(unixTs) && unixTs > 0) {
    const asDate = new Date(unixTs > 1e12 ? unixTs : unixTs * 1000);
    if (!Number.isNaN(asDate.getTime())) {
      return `${asDate.getFullYear()}-${String(asDate.getMonth() + 1).padStart(2, '0')}-${String(asDate.getDate()).padStart(2, '0')}`;
    }
  }
  return value;
}

function truncateText(value: string, maxLength: number) {
  const clean = String(value || '').replace(/\s+/g, ' ').trim();
  if (!clean) return '';
  if (clean.length <= maxLength) return clean;
  return `${clean.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…`;
}

function collectResolutionLabels(formats: any[]) {
  const seen = new Set<string>();
  const labels: string[] = [];
  formats.forEach((format) => {
    const label =
      toResolutionLabel(String(format?.quality || '')) ||
      toResolutionLabel(String(format?.format_note || '')) ||
      toResolutionLabel(String(format?.resolution || '')) ||
      toResolutionLabel(String(format?.id || ''));
    if (!label || seen.has(label)) return;
    seen.add(label);
    labels.push(label);
  });
  return labels.sort((a, b) => Number(b.replace(/\D/g, '')) - Number(a.replace(/\D/g, '')));
}

function toResolutionLabel(raw: string) {
  const text = String(raw || '').trim();
  if (!text) return '';
  const pMatch = text.match(/(\d{3,4})\s*p/i);
  if (pMatch) return `${pMatch[1]}P`;
  const resMatch = text.match(/(?:^|[^\d])(\d{3,4})(?:[^\d]|$)/);
  if (resMatch) return `${resMatch[1]}P`;
  return '';
}

function extractPathname(raw: string) {
  const value = String(raw || '').trim();
  if (!value) return '';
  if (!/^https?:\/\//i.test(value)) return value;
  try {
    return new URL(value).pathname || value;
  } catch {
    return value;
  }
}

function toProjectLocalPath(raw: string) {
  const pathname = extractPathname(raw).replace(/\\/g, '/');
  if (!pathname) return '';
  if (pathname.startsWith('/Projects/')) return pathname;
  if (pathname.startsWith('Projects/')) return `/${pathname}`;
  return pathname;
}

function isExternalMediaUrl(raw: string) {
  const value = String(raw || '').trim();
  if (!/^https?:\/\//i.test(value)) return false;
  const pathname = toProjectLocalPath(value);
  return !pathname.startsWith('/Projects/');
}

function isAudioOnlyProjectAssetName(fileName: string) {
  const normalized = safeDecodeUriComponent(String(fileName || ''))
    .replace(/\\/g, '/')
    .split('?')[0]
    .split('#')[0]
    .trim()
    .toLowerCase();
  const baseName = normalized.slice(normalized.lastIndexOf('/') + 1);
  if (!baseName) return false;
  if (baseName.startsWith('source_audio.')) return true;
  return AUDIO_EXTENSIONS.has(getFileExt(baseName));
}

function isSubtitleAsset(asset: Material) {
  if (asset.category === 'subtitle') return true;
  return SUBTITLE_EXTENSIONS.has(getFileExt(String(asset.name || '')));
}

function isVideoAsset(asset: Material) {
  if (asset.category === 'audio') return false;
  const fileName = String(asset.name || '');
  if (isAudioOnlyProjectAssetName(fileName)) return false;
  return VIDEO_EXTENSIONS.has(getFileExt(fileName));
}

function isAllowedSubtitleFileName(fileName: string) {
  return SUBTITLE_EXTENSIONS.has(getFileExt(fileName));
}

function scoreVideoAsset(assetName: string) {
  const fileExt = getFileExt(assetName);
  const extPriority: Record<string, number> = {
    '.mp4': 90,
    '.webm': 80,
    '.mkv': 70,
    '.mov': 60,
    '.m4v': 55,
    '.avi': 50,
    '.flv': 40,
    '.wmv': 35,
    '.ts': 30,
  };
  let total = extPriority[fileExt] || 0;
  const lower = String(assetName || '').toLowerCase();
  if (lower.startsWith('source.')) total += 100;
  return total;
}

function parseTimestamp(raw: string) {
  const value = String(raw || '').trim().replace(',', '.');
  const parts = value.split(':');
  if (parts.length < 2 || parts.length > 3) return null;
  const sec = Number(parts[parts.length - 1]);
  const min = Number(parts[parts.length - 2]);
  const hour = parts.length === 3 ? Number(parts[0]) : 0;
  if (!Number.isFinite(sec) || !Number.isFinite(min) || !Number.isFinite(hour)) return null;
  return hour * 3600 + min * 60 + sec;
}

function cleanSubtitleText(raw: string) {
  return String(raw || '')
    .replace(/\{\\[^}]*\}/g, '')
    .replace(/<[^>]*>/g, '')
    .replace(/\\N/g, '\n')
    .replace(/\s+/g, ' ')
    .trim();
}

function parseBracketTimedText(raw: string) {
  const lines = String(raw || '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  const rows: Array<{ start: number; text: string }> = [];

  lines.forEach((line) => {
    const matched = line.match(/^((?:\[[^\]]+\]\s*)+)(.*)$/);
    if (!matched) return;
    const tags = Array.from(matched[1].matchAll(/\[([^\]]+)\]/g)).map((m) => String(m[1] || '').trim());
    const timeTag = tags.find((tag) => /^\d{2}:\d{2}:\d{2}(?:[.,]\d{1,3})?$/.test(tag));
    if (!timeTag) return;
    const start = parseTimestamp(timeTag);
    if (start == null) return;
    rows.push({ start, text: cleanSubtitleText(matched[2]) });
  });

  if (rows.length === 0) return [] as SubtitleCue[];

  return rows.map((row, idx) => {
    const nextStart = idx + 1 < rows.length ? rows[idx + 1].start : row.start + 2;
    const end = nextStart > row.start ? nextStart : row.start + 2;
    return { start: row.start, end, text: row.text };
  });
}

function parseSrt(raw: string) {
  const blocks = String(raw || '').replace(/\r\n/g, '\n').split(/\n{2,}/);
  const cues: SubtitleCue[] = [];
  blocks.forEach((block) => {
    const lines = block.split('\n').map((line) => line.trim()).filter(Boolean);
    const timingLineIndex = lines.findIndex((line) => line.includes('-->'));
    if (timingLineIndex < 0) return;
    const timing = lines[timingLineIndex].split('-->');
    if (timing.length !== 2) return;
    const start = parseTimestamp(timing[0]);
    const end = parseTimestamp(timing[1]);
    if (start == null || end == null || end <= start) return;
    const text = cleanSubtitleText(lines.slice(timingLineIndex + 1).join('\n'));
    if (!text) return;
    cues.push({ start, end, text });
  });
  return cues;
}

function parseVtt(raw: string) {
  const lines = String(raw || '').replace(/\r\n/g, '\n').split('\n');
  const cues: SubtitleCue[] = [];

  for (let i = 0; i < lines.length; i += 1) {
    const current = lines[i].trim();
    if (!current || current.startsWith('WEBVTT') || current.startsWith('NOTE')) continue;

    let timingLine = current;
    let textStart = i + 1;
    if (!current.includes('-->') && i + 1 < lines.length && lines[i + 1].includes('-->')) {
      timingLine = lines[i + 1].trim();
      textStart = i + 2;
    } else if (!current.includes('-->')) {
      continue;
    }

    const timing = timingLine.split('-->');
    if (timing.length !== 2) continue;
    const start = parseTimestamp(timing[0]);
    const end = parseTimestamp(timing[1]);
    if (start == null || end == null || end <= start) continue;

    const textLines: string[] = [];
    let j = textStart;
    while (j < lines.length && lines[j].trim() !== '') {
      textLines.push(lines[j]);
      j += 1;
    }
    const text = cleanSubtitleText(textLines.join('\n'));
    if (text) cues.push({ start, end, text });
    i = j;
  }

  return cues;
}

function parseAss(raw: string) {
  const lines = String(raw || '').replace(/\r\n/g, '\n').split('\n');
  const cues: SubtitleCue[] = [];

  lines.forEach((line) => {
    if (!/^Dialogue:/i.test(line)) return;
    const payload = line.replace(/^Dialogue:\s*/i, '');
    const parts = payload.split(',');
    if (parts.length < 10) return;
    const start = parseTimestamp(parts[1]);
    const end = parseTimestamp(parts[2]);
    if (start == null || end == null || end <= start) return;
    const text = cleanSubtitleText(parts.slice(9).join(','));
    if (!text) return;
    cues.push({ start, end, text });
  });

  return cues;
}

function normalizeSubtitleInput(fileName: string, rawContent: string) {
  const clean = String(rawContent || '').replace(/^\uFEFF/, '').trim();
  if (!clean) return '';
  if (getFileExt(fileName) !== '.json') return clean;

  try {
    const parsed = JSON.parse(clean);
    if (typeof parsed === 'string') return parsed.trim();
    if (typeof parsed?.translatedSubtitles === 'string') return parsed.translatedSubtitles.trim();
    if (typeof parsed?.originalSubtitles === 'string') return parsed.originalSubtitles.trim();
    if (typeof parsed?.text === 'string') return parsed.text.trim();
    if (Array.isArray(parsed?.segments)) {
      const merged = parsed.segments.map((seg: any) => String(seg?.text || '').trim()).filter(Boolean).join('\n');
      if (merged) return merged;
    }
    if (Array.isArray(parsed?.chunks)) {
      const merged = parsed.chunks.map((chunk: any) => String(chunk?.text || '').trim()).filter(Boolean).join('\n');
      if (merged) return merged;
    }
  } catch {
    // Fallback to raw text.
  }

  return clean;
}

function parseSubtitleContent(fileName: string, rawContent: string) {
  const normalized = normalizeSubtitleInput(fileName, rawContent);
  if (!normalized) return [] as SubtitleCue[];
  const ext = getFileExt(fileName);

  if (ext === '.srt') return parseSrt(normalized);
  if (ext === '.vtt') return parseVtt(normalized);
  if (ext === '.ass' || ext === '.ssa') return parseAss(normalized);

  const bracketParsed = parseBracketTimedText(normalized);
  if (bracketParsed.length > 0) return bracketParsed;
  return [];
}

function getCueAtTime(cues: SubtitleCue[], time: number) {
  for (let i = 0; i < cues.length; i += 1) {
    const cue = cues[i];
    if (time >= cue.start && time < cue.end) return cue;
  }
  return null;
}

function ArcFallbackIcon() {
  return (
    <div className="w-full h-full bg-black relative flex items-center justify-center overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-primary/35 via-transparent to-tertiary/40 blur-md" />
      <div className="relative w-8 h-8 rounded-lg border border-primary/40 bg-primary/20 flex items-center justify-center shadow-[0_0_20px_rgba(99,102,241,0.45)]">
        <span className="text-primary font-black text-sm leading-none">A</span>
      </div>
    </div>
  );
}

function isDirectPlayableFormat(format: any) {
  const url = String(format?.url || '').trim();
  if (!url || !isExternalMediaUrl(url)) return false;
  const vcodec = String(format?.vcodec || '').toLowerCase();
  const acodec = String(format?.acodec || '').toLowerCase();
  if (vcodec === 'none' || acodec === 'none') return false;
  const ext = getFileExt(String(format?.ext || ''));
  if (ext && !VIDEO_EXTENSIONS.has(ext)) return false;
  return true;
}

function isLikelyDirectMediaUrl(raw: string) {
  const url = String(raw || '').trim();
  if (!url || !isExternalMediaUrl(url)) return false;

  const ext = getFileExt(url);
  if (VIDEO_EXTENSIONS.has(ext) && !AUDIO_EXTENSIONS.has(ext) && !isAudioOnlyProjectAssetName(url)) {
    return true;
  }

  try {
    const parsed = new URL(url);
    const host = parsed.hostname.toLowerCase();
    const pathname = parsed.pathname.toLowerCase();
    const mime = String(parsed.searchParams.get('mime') || parsed.searchParams.get('mime_type') || '').toLowerCase();

    if (mime.includes('video/')) return true;
    if (pathname.includes('videoplayback') && parsed.searchParams.get('source') === 'youtube') return true;
    if (host.includes('googlevideo.com') && pathname.includes('videoplayback')) return true;
    if (host.includes('akamaized.net')) return true;
    if (pathname.endsWith('.m4s') || pathname.endsWith('.mpd') || pathname.endsWith('.ts')) return true;
  } catch {
    return false;
  }

  return false;
}

function toPlaybackProxyUrl(raw: string) {
  const url = String(raw || '').trim();
  if (!url) return '';
  return `/api/playback-proxy?url=${encodeURIComponent(url)}`;
}

function buildVideoSourceOptions(
  project: Project | null,
  materials: Material[],
  playbackMetadata: any,
  t: (key: string) => string
) {
  if (!project?.id) return [] as VideoSourceOption[];

  const options: VideoSourceOption[] = [];
  const seenIds = new Set<string>();
  const seenSrc = new Set<string>();
  const push = (option: VideoSourceOption | null) => {
    if (!option?.id || (option.playable && !option.src)) return;
    if (seenIds.has(option.id) || seenSrc.has(option.src)) return;
    seenIds.add(option.id);
    if (option.src) seenSrc.add(option.src);
    options.push(option);
  };

  const localVideos = [...materials]
    .filter((item) => isVideoAsset(item))
    .sort((a, b) => scoreVideoAsset(String(b.name || '')) - scoreVideoAsset(String(a.name || '')));

  localVideos.forEach((asset) => {
    const safeName = String(asset.name || '').trim();
    if (!safeName) return;
    push({
      id: `local:${safeName}`,
      label: `${t('player.sourceLocal')}: ${safeName}`,
      src: `/Projects/${project.id}/assets/${encodeURIComponent(safeName)}`,
      provider: 'local',
      detail: t('player.sourceLocal'),
      resolution: toResolutionLabel(safeName) || toResolutionLabel(String(project.resolution || '')),
      playable: true,
    });
  });

  const metadataFormats = Array.isArray(playbackMetadata?.formats) ? playbackMetadata.formats : [];
  metadataFormats
    .sort((a: any, b: any) => Number(b?.size || 0) - Number(a?.size || 0))
    .forEach((format: any) => {
      const formatId = String(format?.id || format?.quality || format?.url || '');
      const quality = String(format?.quality || format?.format_note || formatId || 'Auto');
      const ext = String(format?.ext || '').trim().toUpperCase();
      const sizeText = formatBytes(Number(format?.size || 0));
      const rawUrl = String(format?.url || '').trim();
      const playable = isDirectPlayableFormat(format);
      push({
        id: `online:format:${formatId}`,
        label: `${t('player.sourceOnline')}: ${quality}`,
        src: playable ? toPlaybackProxyUrl(rawUrl) : '',
        provider: 'online',
        detail: playable
          ? [quality, ext, sizeText].filter(Boolean).join(' | ')
          : [quality, ext, sizeText, t('player.requiresDownloadPlayback')].filter(Boolean).join(' | '),
        resolution:
          toResolutionLabel(quality) ||
          toResolutionLabel(formatId) ||
          toResolutionLabel(String(project.resolution || '')),
        playable,
      });
    });

  const directCandidates = [project.videoUrl, playbackMetadata?.lastUrl].filter((value, index, arr) => {
    const normalized = String(value || '').trim();
    return normalized && arr.indexOf(value) === index;
  });

  directCandidates.forEach((candidate, index) => {
    const url = String(candidate || '').trim();
    if (!isLikelyDirectMediaUrl(url)) return;
    push({
      id: `online:auto:${index}`,
      label: t(index === 0 ? 'player.sourceOnlineAuto' : 'player.sourceOnline'),
      src: toPlaybackProxyUrl(url),
      provider: 'online',
      detail: t('player.sourceOnline'),
      resolution: toResolutionLabel(url) || toResolutionLabel(String(project.resolution || '')),
      playable: true,
    });
  });

  return options;
}

function chooseDefaultVideoSource(options: VideoSourceOption[]) {
  return options.find((option) => option.provider === 'local' && option.playable)
    || options.find((option) => option.playable)
    || null;
}

function hasRequiredPlaybackMetadata(metadata: any, expectedUrl: string) {
  const currentUrl = String(expectedUrl || '').trim();
  const metadataUrl = String(metadata?.lastUrl || '').trim();
  if (!currentUrl || !metadata || !metadataUrl || metadataUrl !== currentUrl) {
    return false;
  }

  const formats = Array.isArray(metadata?.formats) ? metadata.formats : [];
  const hasPlayableFormats = formats.some((format) => isDirectPlayableFormat(format));
  return hasPlayableFormats;
}

function inferProjectSourceMode(project: Project | null): 'online' | 'upload' | null {
  const explicit = String(project?.mediaSourceType || '').trim().toLowerCase();
  if (explicit === 'online' || explicit === 'upload') {
    return explicit;
  }

  const metadataMode = String(project?.videoMetadata?.sourceMode || '').trim().toLowerCase();
  if (metadataMode === 'online' || metadataMode === 'upload') {
    return metadataMode;
  }

  const hasOnlineLastUrl = String(project?.videoMetadata?.lastUrl || '').trim();
  if (hasOnlineLastUrl) {
    return 'online';
  }
  return null;
}

export default function VideoPlayer({ project }: VideoPlayerProps) {
  const { t } = useLanguage();
  const artContainerRef = React.useRef<HTMLDivElement | null>(null);
  const artPlayerRef = React.useRef<any>(null);
  const playerContainerRef = React.useRef<HTMLDivElement | null>(null);
  const subtitleFileInputRef = React.useRef<HTMLInputElement | null>(null);
  const playbackRefreshKeyRef = React.useRef('');
  const hasManualVideoSourceSelectionRef = React.useRef(false);
  const lastPlaybackTimeRef = React.useRef(0);
  const selectedVideoSourceIdRef = React.useRef('');
  const selectedVideoSourceUrlRef = React.useRef('');
  const videoSourceOptionsRef = React.useRef<VideoSourceOption[]>([]);
  const currentSubtitleBlobUrlRef = React.useRef('');
  const pendingSwitchTimeRef = React.useRef(0);
  const pendingSwitchPlaybackRef = React.useRef(false);
  const pendingSwitchSourceRef = React.useRef('');
  const isSwitchingSourceRef = React.useRef(false);
  const subtitleInteractionRef = React.useRef({
    mode: 'none' as SubtitleInteractionMode,
    pointerId: null as number | null,
    startClientX: 0,
    startClientY: 0,
    startPosition: { xPct: 50, yPct: 82 },
    startWidthPct: 56,
    startHeightPct: 10,
    containerWidth: 0,
    containerHeight: 0,
  });
  const subtitleTrackStateRef = React.useRef<{
    activeCues: SubtitleCue[];
    subtitleMode: SubtitleMode;
    subtitleDelayMs: number;
  }>({
    activeCues: [],
    subtitleMode: 'none',
    subtitleDelayMs: 0,
  });
  const subtitleStyleStateRef = React.useRef({
    subtitleColor: '#FFFFFF',
    subtitleBackgroundColor: '#000000',
    subtitleBackgroundOpacityPct: 0,
    subtitleFontSizePx: 30,
    subtitlePosition: { xPct: 50, yPct: 82 },
    subtitleWidthPct: 56,
    subtitleHeightPct: 10,
  });

  const [materials, setMaterials] = React.useState<Material[]>([]);
  const [isLoadingMaterials, setIsLoadingMaterials] = React.useState(false);
  const [playbackMetadata, setPlaybackMetadata] = React.useState<any>(project?.videoMetadata || null);
  const [isRefreshingPlaybackMetadata, setIsRefreshingPlaybackMetadata] = React.useState(false);
  const [selectedVideoSourceId, setSelectedVideoSourceId] = React.useState('');
  const [videoError, setVideoError] = React.useState<string | null>(null);
  const [currentTime, setCurrentTime] = React.useState(0);
  const [duration, setDuration] = React.useState(0);

  const [subtitleMode, setSubtitleMode] = React.useState<SubtitleMode>('none');
  const [aiCues, setAiCues] = React.useState<SubtitleCue[]>([]);
  const [originalCues, setOriginalCues] = React.useState<SubtitleCue[]>([]);
  const [projectCues, setProjectCues] = React.useState<SubtitleCue[]>([]);
  const [loadedProjectSubtitleName, setLoadedProjectSubtitleName] = React.useState('');
  const [subtitleError, setSubtitleError] = React.useState<string | null>(null);
  const [showSubtitleModal, setShowSubtitleModal] = React.useState(false);
  const [selectedSubtitleAssetName, setSelectedSubtitleAssetName] = React.useState<string | null>(null);
  const [isLoadingSubtitleContent, setIsLoadingSubtitleContent] = React.useState(false);
  const [isUploadingSubtitle, setIsUploadingSubtitle] = React.useState(false);

  const [subtitlePosition, setSubtitlePosition] = React.useState({ xPct: 50, yPct: 82 });
  const [subtitleWidthPct, setSubtitleWidthPct] = React.useState(56);
  const [subtitleHeightPct, setSubtitleHeightPct] = React.useState(10);
  const [subtitleFontSizePx, setSubtitleFontSizePx] = React.useState(30);
  const [subtitleColor, setSubtitleColor] = React.useState('#FFFFFF');
  const [subtitleBackgroundColor, setSubtitleBackgroundColor] = React.useState('#000000');
  const [subtitleBackgroundOpacityPct, setSubtitleBackgroundOpacityPct] = React.useState(0);
  const [subtitleDelayMs, setSubtitleDelayMs] = React.useState(0);
  const [subtitleInteractionMode, setSubtitleInteractionMode] = React.useState<SubtitleInteractionMode>('none');

  const destroyArtplayer = React.useCallback(() => {
    if (currentSubtitleBlobUrlRef.current) {
      URL.revokeObjectURL(currentSubtitleBlobUrlRef.current);
      currentSubtitleBlobUrlRef.current = '';
    }

    if (artPlayerRef.current) {
      try {
        artPlayerRef.current.destroy(false);
      } catch {
        // no-op
      }
      artPlayerRef.current = null;
    }
  }, []);

  const applySubtitleTrackToArt = React.useCallback((artInstance?: any | null) => {
    const art = artInstance || artPlayerRef.current;
    if (!art) return;

    if (currentSubtitleBlobUrlRef.current) {
      URL.revokeObjectURL(currentSubtitleBlobUrlRef.current);
      currentSubtitleBlobUrlRef.current = '';
    }

    const { activeCues, subtitleMode, subtitleDelayMs } = subtitleTrackStateRef.current;
    if (activeCues.length > 0 && subtitleMode !== 'none') {
      const blobUrl = generateVttBlob(activeCues);
      currentSubtitleBlobUrlRef.current = blobUrl;
      art.subtitle.switch(blobUrl, { name: subtitleMode });
      art.subtitle.offset = subtitleDelayMs / 1000;
      art.subtitle.show = true;
      return;
    }

    art.subtitle.show = false;
  }, []);

  const applySubtitleStyleToArt = React.useCallback((artInstance?: any | null) => {
    const art = artInstance || artPlayerRef.current;
    if (!art) return;

    const {
      subtitleColor,
      subtitleBackgroundColor,
      subtitleBackgroundOpacityPct,
      subtitleFontSizePx,
      subtitlePosition,
      subtitleWidthPct,
      subtitleHeightPct,
    } = subtitleStyleStateRef.current;
    const subtitleBackgroundRgba = toRgbaFromHex(subtitleBackgroundColor, subtitleBackgroundOpacityPct);
    art.subtitle.style({
      color: subtitleColor,
      fontSize: `${subtitleFontSizePx}px`,
      backgroundColor: subtitleBackgroundRgba,
    });

    const subtitleEl = art.template?.$subtitle;
    if (!subtitleEl) return;

    subtitleEl.style.width = `${subtitleWidthPct}%`;
    subtitleEl.style.minHeight = `${subtitleHeightPct}%`;
    subtitleEl.style.height = 'auto';
    subtitleEl.style.left = `${subtitlePosition.xPct}%`;
    subtitleEl.style.top = `${subtitlePosition.yPct}%`;
    subtitleEl.style.bottom = 'auto';
    subtitleEl.style.transform = 'translate(-50%, -50%)';
    subtitleEl.style.display = 'flex';
    subtitleEl.style.alignItems = 'center';
    subtitleEl.style.justifyContent = 'center';
    subtitleEl.style.boxSizing = 'border-box';
    subtitleEl.style.backgroundColor = subtitleBackgroundRgba;
    subtitleEl.style.padding = subtitleBackgroundOpacityPct > 0 ? '0.08em 0.55em' : '0';
    subtitleEl.style.borderRadius = subtitleBackgroundOpacityPct > 0 ? '0.45rem' : '0';
  }, []);

  const endSubtitleInteraction = React.useCallback(() => {
    subtitleInteractionRef.current.mode = 'none';
    subtitleInteractionRef.current.pointerId = null;
    setSubtitleInteractionMode('none');
  }, []);

  const beginSubtitleInteraction = React.useCallback(
    (event: React.PointerEvent<HTMLElement>, mode: Exclude<SubtitleInteractionMode, 'none'>) => {
      const container = playerContainerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      if (!Number.isFinite(rect.width) || rect.width <= 0 || !Number.isFinite(rect.height) || rect.height <= 0) return;

      event.preventDefault();
      event.stopPropagation();
      try {
        event.currentTarget.setPointerCapture(event.pointerId);
      } catch {
        // Ignore pointer capture failures.
      }

      subtitleInteractionRef.current.mode = mode;
      subtitleInteractionRef.current.pointerId = event.pointerId;
      subtitleInteractionRef.current.startClientX = event.clientX;
      subtitleInteractionRef.current.startClientY = event.clientY;
      subtitleInteractionRef.current.startPosition = { ...subtitlePosition };
      subtitleInteractionRef.current.startWidthPct = subtitleWidthPct;
      subtitleInteractionRef.current.startHeightPct = subtitleHeightPct;
      subtitleInteractionRef.current.containerWidth = rect.width;
      subtitleInteractionRef.current.containerHeight = rect.height;
      setSubtitleInteractionMode(mode);
    },
    [subtitleHeightPct, subtitlePosition, subtitleWidthPct]
  );

  React.useEffect(() => {
    const handlePointerMove = (event: PointerEvent) => {
      const session = subtitleInteractionRef.current;
      if (session.mode === 'none') return;
      if (session.pointerId !== null && event.pointerId !== session.pointerId) return;

      const deltaXPct = ((event.clientX - session.startClientX) / Math.max(1, session.containerWidth)) * 100;
      const deltaYPct = ((event.clientY - session.startClientY) / Math.max(1, session.containerHeight)) * 100;

      if (session.mode === 'move') {
        const nextX = clamp(
          session.startPosition.xPct + deltaXPct,
          SUBTITLE_POSITION_MIN_PCT,
          SUBTITLE_POSITION_MAX_PCT
        );
        const nextY = clamp(
          session.startPosition.yPct + deltaYPct,
          SUBTITLE_POSITION_MIN_PCT,
          SUBTITLE_POSITION_MAX_PCT
        );
        setSubtitlePosition({ xPct: nextX, yPct: nextY });
        return;
      }

      if (session.mode === 'resize-width') {
        const nextWidth = clamp(
          session.startWidthPct + (deltaXPct * 2),
          SUBTITLE_WIDTH_MIN_PCT,
          SUBTITLE_WIDTH_MAX_PCT
        );
        setSubtitleWidthPct(nextWidth);
        return;
      }

      if (session.mode === 'resize-height') {
        const nextHeight = clamp(
          session.startHeightPct + (deltaYPct * 2),
          SUBTITLE_HEIGHT_MIN_PCT,
          SUBTITLE_HEIGHT_MAX_PCT
        );
        setSubtitleHeightPct(nextHeight);
      }
    };

    const handlePointerUp = (event: PointerEvent) => {
      const session = subtitleInteractionRef.current;
      if (session.mode === 'none') return;
      if (session.pointerId !== null && event.pointerId !== session.pointerId) return;
      endSubtitleInteraction();
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    window.addEventListener('pointercancel', handlePointerUp);
    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
      window.removeEventListener('pointercancel', handlePointerUp);
    };
  }, [endSubtitleInteraction]);

  const restorePlaybackAfterSwitch = React.useCallback((artInstance?: any | null) => {
    const art = artInstance || artPlayerRef.current;
    if (!art || !isSwitchingSourceRef.current) return;

    const resumeTime = pendingSwitchTimeRef.current;
    const shouldResume = pendingSwitchPlaybackRef.current;

    pendingSwitchSourceRef.current = '';
    isSwitchingSourceRef.current = false;
    pendingSwitchTimeRef.current = 0;
    pendingSwitchPlaybackRef.current = false;

    if (resumeTime > 0) {
      try {
        art.currentTime = resumeTime;
      } catch {
        // Ignore seeks rejected by the browser.
      }
    }

    lastPlaybackTimeRef.current = resumeTime;
    applySubtitleTrackToArt(art);
    applySubtitleStyleToArt(art);

    if (shouldResume) {
      const maybePromise = art.play?.();
      if (maybePromise && typeof maybePromise.catch === 'function') {
        maybePromise.catch(() => {});
      }
    }
  }, [applySubtitleStyleToArt, applySubtitleTrackToArt]);

  const persistProjectUpdates = React.useCallback(
    async (updates: Partial<Project>) => {
      if (!project?.id) return;
      try {
        await fetch(`/api/projects/${project.id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(updates),
        });
      } catch (err) {
        console.error('Failed to persist player project updates', err);
      }
    },
    [project?.id]
  );

  React.useEffect(() => {
    setPlaybackMetadata(project?.videoMetadata || null);
  }, [project?.id, project?.videoMetadata]);

  const loadMaterials = React.useCallback(async () => {
    if (!project?.id) {
      setMaterials([]);
      return;
    }

    setIsLoadingMaterials(true);
    try {
      const response = await fetch(`/api/projects/${project.id}/materials`, { cache: 'no-store' });
      const data = await response.json();
      setMaterials(Array.isArray(data) ? (data as Material[]) : []);
    } catch (err) {
      console.error('Failed to load project materials', err);
      setMaterials([]);
    } finally {
      setIsLoadingMaterials(false);
    }
  }, [project?.id]);

  React.useEffect(() => {
    void loadMaterials();
  }, [loadMaterials]);

  const localVideoAssets = React.useMemo(() => materials.filter((asset) => isVideoAsset(asset)), [materials]);
  const subtitleAssets = React.useMemo(() => materials.filter((asset) => isSubtitleAsset(asset)), [materials]);
  const lastParsedUrl = String(project?.videoMetadata?.lastUrl || project?.videoUrl || '').trim();
  const hasSufficientPlaybackMetadata = React.useMemo(
    () => hasRequiredPlaybackMetadata(playbackMetadata, lastParsedUrl),
    [lastParsedUrl, playbackMetadata]
  );

  React.useEffect(() => {
    playbackRefreshKeyRef.current = '';
    hasManualVideoSourceSelectionRef.current = false;
  }, [project?.id]);

  React.useEffect(() => {
    let cancelled = false;

    if (
      !project?.id ||
      !lastParsedUrl ||
      !isExternalMediaUrl(lastParsedUrl) ||
      localVideoAssets.length > 0 ||
      hasSufficientPlaybackMetadata
    ) {
      setIsRefreshingPlaybackMetadata(false);
      return;
    }

    const refreshKey = `${project.id}|${lastParsedUrl}`;
    if (playbackRefreshKeyRef.current === refreshKey) {
      return;
    }

    playbackRefreshKeyRef.current = refreshKey;
    setIsRefreshingPlaybackMetadata(true);

    const run = async () => {
      try {
        const response = await fetch('/api/parse-video', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: lastParsedUrl }),
        });
        const data = await response.json();
        if (!response.ok || data?.error) {
          throw new Error(String(data?.error || t('player.videoLoadFailed')));
        }
        if (!cancelled) {
          const nextMetadata = { ...data, lastUrl: lastParsedUrl };
          setPlaybackMetadata(nextMetadata);
          void persistProjectUpdates({ videoMetadata: nextMetadata });
        }
      } catch (err) {
        console.error('Failed to refresh playback metadata', err);
      } finally {
        if (!cancelled) {
          setIsRefreshingPlaybackMetadata(false);
        }
      }
    };

    void run();
    return () => {
      cancelled = true;
    };
  }, [hasSufficientPlaybackMetadata, localVideoAssets.length, lastParsedUrl, persistProjectUpdates, project?.id, t]);

  const videoSourceOptions = React.useMemo(
    () => buildVideoSourceOptions(project, materials, playbackMetadata, t),
    [materials, playbackMetadata, project, t]
  );

  React.useEffect(() => {
    setSelectedVideoSourceId((prev) => {
      const preferred = chooseDefaultVideoSource(videoSourceOptions);
      const prevOption = videoSourceOptions.find((option) => option.id === prev) || null;
      const currentUrl = selectedVideoSourceUrlRef.current || artPlayerRef.current?.url || '';

      if (!prevOption) {
        const sameSourceOption = currentUrl
          ? videoSourceOptions.find((option) => option.src === currentUrl) || null
          : null;
        if (sameSourceOption) {
          return sameSourceOption.id;
        }
        return preferred?.id || '';
      }

      if (!hasManualVideoSourceSelectionRef.current && preferred?.provider === 'local' && prevOption.provider !== 'local') {
        return preferred.id;
      }

      return prev;
    });
  }, [project?.id, videoSourceOptions]);

  const selectedVideoSourceOption = React.useMemo(
    () => videoSourceOptions.find((option) => option.id === selectedVideoSourceId) || null,
    [selectedVideoSourceId, videoSourceOptions]
  );
  const showSubtitleDragOverlay = Boolean(selectedVideoSourceOption) && subtitleMode !== 'none';
  const subtitleDragOverlayStyle = React.useMemo(
    () => ({
      width: `${subtitleWidthPct}%`,
      height: `${subtitleHeightPct}%`,
      left: `${subtitlePosition.xPct}%`,
      top: `${subtitlePosition.yPct}%`,
      transform: 'translate(-50%, -50%)',
    }),
    [subtitleHeightPct, subtitlePosition, subtitleWidthPct]
  );
  const subtitleResizeHandleVisibilityClass = subtitleInteractionMode === 'none'
    ? 'opacity-0 pointer-events-none group-hover/subtitle-frame:opacity-100 group-hover/subtitle-frame:pointer-events-auto'
    : 'opacity-100 pointer-events-auto';

  React.useEffect(() => {
    selectedVideoSourceIdRef.current = selectedVideoSourceId;
  }, [selectedVideoSourceId]);

  React.useEffect(() => {
    selectedVideoSourceUrlRef.current = selectedVideoSourceOption?.src || '';
  }, [selectedVideoSourceOption?.src]);

  React.useEffect(() => {
    videoSourceOptionsRef.current = videoSourceOptions;
  }, [videoSourceOptions]);

  const resolveProjectSubtitleText = React.useCallback(async (projectId: string, fileName: string) => {
    const encodedName = encodeURIComponent(fileName);
    const fetchBySubDir = async (subDir: 'subtitles' | 'assets') => {
      const response = await fetch(`/Projects/${projectId}/${subDir}/${encodedName}`, { cache: 'no-store' });
      if (!response.ok) return null;
      return response.text();
    };

    const subtitleText = await fetchBySubDir('subtitles');
    if (subtitleText != null) return subtitleText;
    const assetText = await fetchBySubDir('assets');
    if (assetText != null) return assetText;
    return null;
  }, []);

  React.useEffect(() => {
    let cancelled = false;

    setSubtitleError(null);
    setLoadedProjectSubtitleName('');
    setProjectCues([]);
    setSelectedSubtitleAssetName(null);

    if (!project?.id) {
      setAiCues([]);
      setOriginalCues([]);
      setSubtitleMode('none');
      return;
    }

    const inlineAiCues = parseSubtitleContent('inline_translation.txt', String(project.translatedSubtitles || ''));
    const inlineOriginalCues = parseSubtitleContent('inline_original.txt', String(project.originalSubtitles || ''));
    setAiCues(inlineAiCues);
    setOriginalCues(inlineOriginalCues);
    setSubtitleMode(inlineAiCues.length > 0 ? 'ai' : inlineOriginalCues.length > 0 ? 'original' : 'none');

    const tryLoadPersistedTranslation = async () => {
      const candidates = ['translation.vtt', 'translation.srt', 'translation.txt'];
      for (const fileName of candidates) {
        const content = await resolveProjectSubtitleText(project.id, fileName);
        if (cancelled || !content) continue;
        const parsed = parseSubtitleContent(fileName, content);
        if (parsed.length > 0) {
          setAiCues(parsed);
          setSubtitleMode((prev) => (prev === 'project' ? prev : 'ai'));
          return;
        }
      }
    };

    // Prefer in-project edited translation text/timecodes when available.
    // Persisted export files are only fallback for projects that have no inline translated subtitles.
    if (inlineAiCues.length === 0) {
      void tryLoadPersistedTranslation();
    }

    return () => {
      cancelled = true;
    };
  }, [project, resolveProjectSubtitleText]);

  const hasAiSubtitles = aiCues.length > 0;
  const hasOriginalSubtitles = originalCues.length > 0;
  const hasProjectSubtitles = projectCues.length > 0;
  const activeCues = React.useMemo(() => {
    if (subtitleMode === 'ai') return aiCues;
    if (subtitleMode === 'original') return originalCues;
    if (subtitleMode === 'project') return projectCues;
    return [];
  }, [aiCues, originalCues, projectCues, subtitleMode]);

  const delayedSubtitleTime = currentTime - subtitleDelayMs / 1000;
  const activeSubtitleCue = React.useMemo(
    () => getCueAtTime(activeCues, delayedSubtitleTime),
    [activeCues, delayedSubtitleTime]
  );
  const currentSubtitle = activeSubtitleCue?.text || '';
  React.useEffect(() => {
    setCurrentTime(0);
    setDuration(0);
    setVideoError(null);
  }, [selectedVideoSourceId, project?.id]);

  React.useEffect(() => {
    pendingSwitchSourceRef.current = '';
    pendingSwitchTimeRef.current = 0;
    pendingSwitchPlaybackRef.current = false;
    isSwitchingSourceRef.current = false;
    destroyArtplayer();
  }, [destroyArtplayer, project?.id]);

  React.useEffect(() => () => {
    destroyArtplayer();
  }, [destroyArtplayer]);

  React.useEffect(() => {
    subtitleTrackStateRef.current = {
      activeCues,
      subtitleMode,
      subtitleDelayMs,
    };
    applySubtitleTrackToArt();
  }, [activeCues, applySubtitleTrackToArt, subtitleDelayMs, subtitleMode]);

  React.useEffect(() => {
    subtitleStyleStateRef.current = {
      subtitleColor,
      subtitleBackgroundColor,
      subtitleBackgroundOpacityPct,
      subtitleFontSizePx,
      subtitlePosition,
      subtitleWidthPct,
      subtitleHeightPct,
    };
    applySubtitleStyleToArt();
  }, [
    applySubtitleStyleToArt,
    subtitleBackgroundColor,
    subtitleBackgroundOpacityPct,
    subtitleColor,
    subtitleFontSizePx,
    subtitleHeightPct,
    subtitlePosition,
    subtitleWidthPct,
  ]);

  React.useEffect(() => {
    if (showSubtitleDragOverlay) return;
    endSubtitleInteraction();
  }, [endSubtitleInteraction, showSubtitleDragOverlay]);

  // Build the player once the initial source becomes available. Later source changes use Artplayer.switch.
  React.useEffect(() => {
    const container = artContainerRef.current;
    const initialSource = selectedVideoSourceOption?.src || '';

    if (artPlayerRef.current || !container || !initialSource) return;

    const art = new Artplayer({
      container,
      url: initialSource,
      autoplay: false,
      autoSize: true,
      autoMini: true,
      pip: true,
      fullscreen: true,
      fullscreenWeb: true,
      setting: true,
      playbackRate: true,
      mutex: true,
      screenshot: true,
      moreVideoAttr: {
        crossOrigin: 'anonymous',
        playsInline: true,
      },
      subtitle: {
        url: '',
        type: 'vtt',
        style: {
          color: subtitleColor,
          fontSize: `${subtitleFontSizePx}px`,
          backgroundColor: toRgbaFromHex(subtitleBackgroundColor, subtitleBackgroundOpacityPct),
        },
      },
    });

    art.on('ready', () => {
      if (playerContainerRef.current) {
        if (art.fullscreen && typeof art.fullscreen === 'object') {
          (art.fullscreen as any).target = playerContainerRef.current;
        }
        if (art.fullscreenWeb && typeof art.fullscreenWeb === 'object') {
          (art.fullscreenWeb as any).target = playerContainerRef.current;
        }
      }
      if (lastPlaybackTimeRef.current > 0) {
        art.currentTime = lastPlaybackTimeRef.current;
        art.play();
      }
      applySubtitleTrackToArt(art);
      applySubtitleStyleToArt(art);
    });

    // Handle source switching back to React state
    art.on('video:url', (url: string) => {
      const opt = videoSourceOptionsRef.current.find((o) => o.src === url);
      if (opt && opt.id !== selectedVideoSourceIdRef.current) {
        setSelectedVideoSourceId(opt.id);
      }
    });

    art.on('video:loadedmetadata', () => {
      restorePlaybackAfterSwitch(art);
    });

    art.on('video:canplay', () => {
      restorePlaybackAfterSwitch(art);
    });

    art.on('video:error', () => {
      isSwitchingSourceRef.current = false;
      pendingSwitchSourceRef.current = '';
      pendingSwitchTimeRef.current = 0;
      pendingSwitchPlaybackRef.current = false;
      setVideoError(t('player.videoLoadFailed'));
    });

    artPlayerRef.current = art;
    setVideoError(null);

    const syncStatus = () => {
      const now = Number(art.currentTime || 0);
      lastPlaybackTimeRef.current = now;
      const total = Number(art.duration || 0);
      if (Number.isFinite(now)) setCurrentTime(now);
      if (Number.isFinite(total) && total > 0) setDuration(total);
    };

    art.on('video:timeupdate', syncStatus);
  }, [
    applySubtitleStyleToArt,
    applySubtitleTrackToArt,
    destroyArtplayer,
    project?.id,
    restorePlaybackAfterSwitch,
    selectedVideoSourceOption?.src,
    subtitleColor,
    subtitleBackgroundColor,
    subtitleBackgroundOpacityPct,
    subtitleFontSizePx,
    t,
  ]);

  // Handle Video Source Switching (Seamless)
  React.useEffect(() => {
    const art = artPlayerRef.current;
    const nextUrl = selectedVideoSourceOption?.src || '';
    if (art && nextUrl && art.url !== nextUrl) {
      const currentTime = Number(art.currentTime || 0);
      pendingSwitchTimeRef.current = currentTime;
      pendingSwitchPlaybackRef.current = !art.paused && currentTime > 0.75;
      pendingSwitchSourceRef.current = nextUrl;
      isSwitchingSourceRef.current = true;
      lastPlaybackTimeRef.current = pendingSwitchTimeRef.current;
      setVideoError(null);

      if (typeof art.switch === 'function') {
        art.switch(nextUrl);
      } else {
        art.switch = nextUrl;
      }
    }
  }, [selectedVideoSourceOption?.src]);

  const sourceAvailabilityError =
    !selectedVideoSourceOption && !isLoadingMaterials && !isRefreshingPlaybackMetadata && project?.id
      ? t('player.videoUnavailable')
      : null;

  const handleOpenSubtitleModal = () => {
    setSubtitleError(null);
    setSelectedSubtitleAssetName(loadedProjectSubtitleName || null);
    setShowSubtitleModal(true);
    void loadMaterials();
  };

  const handleSubtitleFileUpload = async (file?: File | null) => {
    if (!project?.id || !file) return;
    if (!isAllowedSubtitleFileName(file.name)) {
      setSubtitleError(t('player.onlySubtitleFiles'));
      return;
    }

    const ext = getFileExt(file.name);
    const endpoint =
      ext === '.txt' || ext === '.json'
        ? `/api/projects/${project.id}/materials/upload-text`
        : `/api/projects/${project.id}/materials/upload`;

    setIsUploadingSubtitle(true);
    setSubtitleError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok || data?.error) {
        throw new Error(String(data?.error || t('translation.errorFailed')));
      }

      await loadMaterials();
      const uploadedName = String(data?.file?.filename || file.name || '').trim();
      if (uploadedName) {
        setSelectedSubtitleAssetName(uploadedName);
      }
    } catch (err: any) {
      setSubtitleError(String(err?.message || t('translation.errorFailed')));
    } finally {
      setIsUploadingSubtitle(false);
      if (subtitleFileInputRef.current) {
        subtitleFileInputRef.current.value = '';
      }
    }
  };

  const handleSubtitleFileDrop = async (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (!file) return;
    await handleSubtitleFileUpload(file);
  };

  const handleConfirmSubtitleSelection = async () => {
    if (!project?.id || !selectedSubtitleAssetName) return;
    setIsLoadingSubtitleContent(true);
    setSubtitleError(null);

    try {
      const content = await resolveProjectSubtitleText(project.id, selectedSubtitleAssetName);
      if (!content) {
        throw new Error(t('player.subtitleLoadFailed'));
      }
      const cues = parseSubtitleContent(selectedSubtitleAssetName, content);
      if (cues.length === 0) {
        throw new Error(t('player.subtitleNoTimecode'));
      }
      setProjectCues(cues);
      setLoadedProjectSubtitleName(selectedSubtitleAssetName);
      setSubtitleMode('project');
      setShowSubtitleModal(false);
    } catch (err: any) {
      setSubtitleError(String(err?.message || t('player.subtitleLoadFailed')));
    } finally {
      setIsLoadingSubtitleContent(false);
    }
  };

  React.useEffect(() => {
    // Positioning and styles are handled via sidebar sliders and Artplayer synchronization
  }, []);

  const resetSubtitleLayout = () => {
    setSubtitlePosition({ xPct: 50, yPct: 82 });
    setSubtitleWidthPct(56);
    setSubtitleHeightPct(10);
    setSubtitleFontSizePx(30);
    setSubtitleColor('#FFFFFF');
    setSubtitleBackgroundColor('#000000');
    setSubtitleBackgroundOpacityPct(0);
    setSubtitleDelayMs(0);
  };

  const selectedResolutionLabel =
    selectedVideoSourceOption?.resolution ||
    toResolutionLabel(selectedVideoSourceOption?.label || '') ||
    toResolutionLabel(selectedVideoSourceOption?.detail || '') ||
    toResolutionLabel(String(project?.resolution || '')) ||
    t('player.autoResolution');

  const selectedSourceSummary =
    selectedVideoSourceOption?.detail ||
    (selectedVideoSourceOption?.provider === 'local'
      ? t('player.sourceLocal')
      : selectedVideoSourceOption?.provider === 'online'
        ? t('player.sourceOnline')
        : '');

  const availableResolutionLabels = React.useMemo(
    () => collectResolutionLabels(Array.isArray(playbackMetadata?.formats) ? playbackMetadata.formats : []),
    [playbackMetadata]
  );

  const projectSourceMode = React.useMemo(() => inferProjectSourceMode(project), [project]);
  const showOnlineMetadata = projectSourceMode !== 'upload';
  const visibleDescription = truncateText(String(playbackMetadata?.description || ''), 260);
  const formattedUploadDate = formatUploadDate(String(playbackMetadata?.uploadDate || ''));
  const formattedViewCount = Number.isFinite(Number(playbackMetadata?.viewCount || 0))
    ? Number(playbackMetadata?.viewCount || 0).toLocaleString()
    : '';
  const showPlayerLoadingOverlay = !showSubtitleModal && !selectedVideoSourceOption && (isRefreshingPlaybackMetadata || isLoadingMaterials);

  return (
    <>
      <div className="h-full flex flex-col space-y-8 animate-in fade-in duration-500">
        <div className="space-y-1">
          <div className="flex items-center gap-4">
            <h2 className="text-3xl font-bold text-secondary tracking-tight">{t('player.title')}</h2>
            {project && (
              <div className="px-3 py-1 bg-primary-container/10 border border-primary-container/20 rounded-full flex items-center gap-2 animate-in fade-in zoom-in duration-500">
                <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse shadow-[0_0_8px_rgba(var(--primary),0.8)]" />
                <span className="text-[10px] font-bold text-primary uppercase tracking-widest">{project.name}</span>
              </div>
            )}
          </div>
          <p className="text-outline">{t('player.subtitle')}</p>
        </div>

        <div className="space-y-4">
          <div
            ref={playerContainerRef}
            className="bg-black rounded-3xl relative overflow-hidden shadow-2xl border border-white/5 aspect-video [&.art-fullscreen-web]:rounded-none [&.art-fullscreen]:rounded-none"
          >
            {selectedVideoSourceOption ? (
              <div ref={artContainerRef} className="w-full h-full bg-black" />
            ) : (
              <div className="absolute inset-0 bg-black">
                {project?.thumbnail ? (
                  <>
                    <img
                      src={project.thumbnail}
                      alt={project.videoTitle || project.name || t('player.title')}
                      className="w-full h-full object-cover opacity-60"
                      referrerPolicy="no-referrer"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-black/40" />
                  </>
                ) : (
                  <div className="w-full h-full">
                    <ArcFallbackIcon />
                  </div>
                )}
              </div>
            )}

            {showPlayerLoadingOverlay && (
              <div className="absolute inset-0 bg-black/35 backdrop-blur-[2px] flex items-center justify-center z-20">
                <div className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-surface-container-high border border-white/10 text-sm text-secondary">
                  <Loader2 className="w-4 h-4 animate-spin text-primary" />
                  <span>{t('common.loading')}</span>
                </div>
              </div>
            )}

            {showSubtitleDragOverlay && (
              <div className="absolute inset-0 z-30 pointer-events-none">
                <div
                  style={subtitleDragOverlayStyle}
                  className="absolute group/subtitle-frame"
                >
                  <button
                    type="button"
                    onPointerDown={(event) => beginSubtitleInteraction(event, 'move')}
                    className="pointer-events-auto absolute inset-0 cursor-move bg-transparent"
                    aria-label={t('player.subtitleDragHint')}
                  />
                  <button
                    type="button"
                    onPointerDown={(event) => beginSubtitleInteraction(event, 'resize-width')}
                    className={`${subtitleResizeHandleVisibilityClass} absolute -right-2 top-1/2 h-5 w-5 -translate-y-1/2 cursor-ew-resize rounded border border-white/30 bg-surface-container-high shadow-lg transition-opacity`}
                    title={t('player.subtitleWidth')}
                  />
                  <button
                    type="button"
                    onPointerDown={(event) => beginSubtitleInteraction(event, 'resize-height')}
                    className={`${subtitleResizeHandleVisibilityClass} absolute left-1/2 -bottom-2 h-5 w-5 -translate-x-1/2 cursor-ns-resize rounded border border-white/30 bg-surface-container-high shadow-lg transition-opacity`}
                    title={t('player.subtitleHeight')}
                  />
                </div>
              </div>
            )}
          </div>

          {(sourceAvailabilityError || videoError) && (
            <div className="text-xs text-error bg-error/10 border border-error/20 rounded-lg p-3">
              {videoError || sourceAvailabilityError}
            </div>
          )}

          {subtitleError && (
            <div className="text-xs text-error bg-error/10 border border-error/20 rounded-lg p-3">{subtitleError}</div>
          )}
        </div>

        <div className="bg-surface-container p-6 rounded-2xl border border-white/5">
          <div className="grid gap-5 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,0.95fr)_minmax(0,1fr)]">
            <div className="rounded-2xl border border-white/5 bg-surface-container-highest/40 p-5 h-full flex flex-col gap-5">
              <div className="flex items-start gap-4">
                <div className="w-16 h-16 rounded-xl bg-surface-container-highest flex items-center justify-center overflow-hidden border border-white/5 shrink-0">
                  {project?.thumbnail ? (
                    <img
                      src={project.thumbnail}
                      alt={project?.videoTitle || project?.name || t('player.unnamedProject')}
                      className="w-full h-full object-cover"
                      referrerPolicy="no-referrer"
                    />
                  ) : (
                    <ArcFallbackIcon />
                  )}
                </div>
                <div className="min-w-0 flex-1">
                  <h3 className="text-lg font-bold text-secondary truncate">{project?.videoTitle || project?.name || t('player.unnamedProject')}</h3>
                  <p className="text-xs text-outline mt-1 truncate">{selectedSourceSummary || t('player.sourceMenuEmpty')}</p>
                  <div className="flex gap-2 mt-3 flex-wrap">
                    <span className="text-[10px] font-bold bg-white/5 text-outline px-2 py-1 rounded-full uppercase tracking-widest">
                      {selectedResolutionLabel}
                    </span>
                    <span className="text-[10px] font-bold bg-white/5 text-outline px-2 py-1 rounded-full uppercase tracking-widest">
                      {subtitleMode === 'none' ? t('player.noSubtitles') : t('player.subtitleLoaded')}
                    </span>
                    {hasAiSubtitles && (
                      <span className="text-[10px] font-bold bg-tertiary/10 text-tertiary px-2 py-1 rounded-full uppercase tracking-widest">
                        {t('player.aiTranslationComplete')}
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {showOnlineMetadata ? (
                <>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className="rounded-xl border border-white/5 bg-white/5 px-4 py-3">
                      <div className="text-[11px] uppercase tracking-widest text-outline/70">{t('fetcher.channel')}</div>
                      <div className="text-secondary font-semibold mt-1 truncate">{String(playbackMetadata?.uploader || '').trim() || '-'}</div>
                    </div>
                    <div className="rounded-xl border border-white/5 bg-white/5 px-4 py-3">
                      <div className="text-[11px] uppercase tracking-widest text-outline/70">{t('status.views')}</div>
                      <div className="text-secondary font-semibold mt-1">{formattedViewCount || '-'}</div>
                    </div>
                    <div className="rounded-xl border border-white/5 bg-white/5 px-4 py-3">
                      <div className="text-[11px] uppercase tracking-widest text-outline/70">{t('player.publishDate')}</div>
                      <div className="text-secondary font-semibold mt-1">{formattedUploadDate || '-'}</div>
                    </div>
                    <div className="rounded-xl border border-white/5 bg-white/5 px-4 py-3">
                      <div className="text-[11px] uppercase tracking-widest text-outline/70">{t('fetcher.duration')}</div>
                      <div className="text-secondary font-semibold mt-1">
                        {formatTime(currentTime)} / {formatTime(duration)}
                      </div>
                    </div>
                  </div>

                  <div className="rounded-xl border border-white/5 bg-white/5 px-4 py-4 flex-1 min-h-0 flex flex-col">
                    <div className="text-[11px] uppercase tracking-widest text-outline/70">{t('player.videoDescription')}</div>
                    <div className="mt-2 flex-1 min-h-0 overflow-y-auto custom-scrollbar pr-1">
                      <p className="text-sm text-secondary leading-6 whitespace-pre-wrap">
                        {visibleDescription || t('fetcher.noDescription')}
                      </p>
                    </div>
                  </div>
                </>
              ) : (
                <div className="rounded-xl border border-white/5 bg-white/5 px-4 py-4 text-sm">
                  <div className="text-[11px] uppercase tracking-widest text-outline/70">{t('fetcher.duration')}</div>
                  <div className="text-secondary font-semibold mt-1">
                    {formatTime(currentTime)} / {formatTime(duration)}
                  </div>
                </div>
              )}
            </div>

            <div className="rounded-2xl border border-white/5 bg-surface-container-highest/40 p-5 space-y-5">
              <div className="flex items-center gap-2 text-secondary">
                <MonitorPlay className="w-4 h-4" />
                <h3 className="text-sm font-bold uppercase tracking-widest">{t('player.videoSource')}</h3>
              </div>

              <label className="block text-xs text-outline space-y-2">
                <span className="block">{t('player.sourceButton')}</span>
                <select
                  value={selectedVideoSourceId}
                  onChange={(event) => {
                    hasManualVideoSourceSelectionRef.current = true;
                    setSelectedVideoSourceId(event.target.value);
                    setVideoError(null);
                  }}
                  className="w-full bg-surface-container-high border border-white/10 text-sm text-white rounded-xl px-4 py-3 focus:ring-2 focus:ring-primary-container outline-none appearance-none cursor-pointer [&>option]:bg-surface-container-high [&>option]:text-white"
                >
                  {videoSourceOptions.length === 0 && (
                    <option value="">{t('player.sourceMenuEmpty')}</option>
                  )}
                  {videoSourceOptions.map((option) => (
                    <option key={option.id} value={option.id} disabled={!option.playable}>
                      {option.playable ? option.label : `${option.label} · ${t('player.requiresDownloadPlayback')}`}
                    </option>
                  ))}
                </select>
              </label>

              <label className="block text-xs text-outline space-y-2">
                <span className="block">{t('player.subtitleSource')}</span>
                <select
                  value={subtitleMode}
                  onChange={(event) => setSubtitleMode(event.target.value as SubtitleMode)}
                  className="w-full bg-surface-container-high border border-white/10 text-sm text-white rounded-xl px-4 py-3 focus:ring-2 focus:ring-primary-container outline-none appearance-none cursor-pointer [&>option]:bg-surface-container-high [&>option]:text-white"
                >
                  <option value="ai" disabled={!hasAiSubtitles}>
                    {t('player.aiTranslation')}
                  </option>
                  <option value="original" disabled={!hasOriginalSubtitles}>
                    {t('player.originalSubtitles')}
                  </option>
                  <option value="project" disabled={!hasProjectSubtitles}>
                    {loadedProjectSubtitleName ? `${t('player.loadedSubtitle')}: ${loadedProjectSubtitleName}` : t('player.loadedSubtitle')}
                  </option>
                  <option value="none">{t('player.noSubtitles')}</option>
                </select>
              </label>

              <button
                onClick={handleOpenSubtitleModal}
                className="w-full bg-primary-container text-white px-4 py-3 rounded-xl text-sm font-bold flex items-center justify-center gap-2 hover:brightness-110 transition-all shadow-lg shadow-primary-container/20"
              >
                <Upload className="w-4 h-4" />
                {t('player.loadSubtitles')}
              </button>
            </div>

            <div className="rounded-2xl border border-white/5 bg-surface-container-highest/40 p-5 space-y-5">
              <div className="flex items-center gap-2 text-secondary">
                <SlidersHorizontal className="w-4 h-4" />
                <h3 className="text-sm font-bold uppercase tracking-widest text-secondary">{t('player.subtitleStyle')}</h3>
              </div>

              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                <label className="block rounded-xl border border-white/5 bg-white/[0.04] px-4 py-3 text-xs text-outline space-y-2">
                  <span className="block">{t('player.subtitleSize')}: {Math.round(subtitleFontSizePx)}px</span>
                  <input
                    type="range"
                    min={18}
                    max={60}
                    step={1}
                    value={subtitleFontSizePx}
                    onChange={(event) => setSubtitleFontSizePx(Number(event.target.value))}
                    className="w-full accent-primary"
                  />
                </label>

                <label className="block rounded-xl border border-white/5 bg-white/[0.04] px-4 py-3 text-xs text-outline space-y-2">
                  <span className="block">{t('player.subtitleWidth')}: {Math.round(subtitleWidthPct)}%</span>
                  <input
                    type="range"
                    min={SUBTITLE_WIDTH_MIN_PCT}
                    max={SUBTITLE_WIDTH_MAX_PCT}
                    step={1}
                    value={subtitleWidthPct}
                    onChange={(event) => setSubtitleWidthPct(Number(event.target.value))}
                    className="w-full accent-primary"
                  />
                </label>
                <label className="block rounded-xl border border-white/5 bg-white/[0.04] px-4 py-3 text-xs text-outline space-y-2">
                  <span className="block">{t('player.subtitleHeight')}: {Math.round(subtitleHeightPct)}%</span>
                  <input
                    type="range"
                    min={SUBTITLE_HEIGHT_MIN_PCT}
                    max={SUBTITLE_HEIGHT_MAX_PCT}
                    step={1}
                    value={subtitleHeightPct}
                    onChange={(event) => setSubtitleHeightPct(Number(event.target.value))}
                    className="w-full accent-primary"
                  />
                </label>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <label className="block rounded-xl border border-white/5 bg-white/[0.04] px-4 py-3 text-xs text-outline space-y-2">
                  <span className="block">{t('player.subtitlePositionX') || 'Horizontal'}: {Math.round(subtitlePosition.xPct)}%</span>
                  <input
                    type="range"
                    min={SUBTITLE_POSITION_MIN_PCT}
                    max={SUBTITLE_POSITION_MAX_PCT}
                    step={1}
                    value={subtitlePosition.xPct}
                    onChange={(event) => setSubtitlePosition(prev => ({ ...prev, xPct: Number(event.target.value) }))}
                    className="w-full accent-primary"
                  />
                </label>

                <label className="block rounded-xl border border-white/5 bg-white/[0.04] px-4 py-3 text-xs text-outline space-y-2">
                  <span className="block">{t('player.subtitlePositionY') || 'Vertical'}: {Math.round(subtitlePosition.yPct)}%</span>
                  <input
                    type="range"
                    min={SUBTITLE_POSITION_MIN_PCT}
                    max={SUBTITLE_POSITION_MAX_PCT}
                    step={1}
                    value={subtitlePosition.yPct}
                    onChange={(event) => setSubtitlePosition(prev => ({ ...prev, yPct: Number(event.target.value) }))}
                    className="w-full accent-primary"
                  />
                </label>
              </div>

              <label className="block rounded-xl border border-white/5 bg-white/[0.04] px-4 py-3 text-xs text-outline space-y-2">
                <span className="block">{t('player.subtitleDelay')}: {formatDelayLabel(subtitleDelayMs)}</span>
                <input
                  type="range"
                  min={-3000}
                  max={3000}
                  step={100}
                  value={subtitleDelayMs}
                  onChange={(event) => setSubtitleDelayMs(Number(event.target.value))}
                  className="w-full accent-primary"
                />
              </label>

              <div className="rounded-xl border border-white/5 bg-white/[0.04] px-4 py-4 space-y-3">
                <div className="space-y-2">
                  <div className="flex items-center gap-3 rounded-lg border border-white/5 bg-black/20 px-3 py-2">
                    <span className="w-24 shrink-0 text-xs text-outline">{t('player.subtitleColor')}</span>
                    <input
                      type="color"
                      value={subtitleColor}
                      onChange={(event) => setSubtitleColor(event.target.value)}
                      className="h-9 w-12 rounded-md border border-white/10 bg-transparent cursor-pointer"
                    />
                    <div className="ml-auto min-w-[90px] text-right text-sm font-mono text-secondary">{subtitleColor.toUpperCase()}</div>
                  </div>
                  <div className="flex items-center gap-3 rounded-lg border border-white/5 bg-black/20 px-3 py-2">
                    <span className="w-24 shrink-0 text-xs text-outline">{t('player.subtitleBoxColor')}</span>
                    <input
                      type="color"
                      value={subtitleBackgroundColor}
                      onChange={(event) => setSubtitleBackgroundColor(event.target.value)}
                      className="h-9 w-12 rounded-md border border-white/10 bg-transparent cursor-pointer"
                    />
                    <div className="ml-auto min-w-[90px] text-right text-sm font-mono text-secondary">{subtitleBackgroundColor.toUpperCase()}</div>
                  </div>
                </div>
                <label className="block rounded-lg border border-white/5 bg-black/20 px-3 py-2 text-xs text-outline space-y-2">
                  <span className="block">{t('player.subtitleBoxOpacity')}: {Math.round(subtitleBackgroundOpacityPct)}%</span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={1}
                    value={subtitleBackgroundOpacityPct}
                    onChange={(event) => setSubtitleBackgroundOpacityPct(Number(event.target.value))}
                    className="w-full accent-primary"
                  />
                </label>
                <div className="flex justify-end">
                  <button
                    onClick={resetSubtitleLayout}
                    className="shrink-0 rounded-xl border border-white/10 px-4 py-2.5 text-sm font-semibold text-outline hover:text-white hover:bg-white/8 transition-colors"
                  >
                    {t('player.subtitleReset')}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {showSubtitleModal && (
        <div className="fixed inset-0 z-[2000] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200">
          <div className="bg-[#13151A] border border-white/10 rounded-[30px] w-full max-w-[1080px] shadow-2xl overflow-hidden flex flex-col max-h-[90vh] animate-in zoom-in-95 duration-200">
            <div className="flex items-start justify-between border-b border-white/5 bg-white/5 px-8 py-6">
              <div className="space-y-3">
                <div className="space-y-1.5">
                  <h2 className="text-[1.75rem] font-bold tracking-tight text-white">{t('player.subtitleAssetModalTitle')}</h2>
                  <p className="text-sm text-outline">{t('player.subtitleAssetModalDesc')}</p>
                  <p className="text-sm font-semibold text-primary">{project?.name}</p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full border border-white/8 bg-white/5 px-3 py-1.5 text-[11px] font-semibold text-outline/85">
                    {t('stt.totalFiles').replace('{count}', subtitleAssets.length.toString())}
                  </span>
                  {project?.notes && (
                    <span className="rounded-full border border-primary/15 bg-primary/10 px-3 py-1.5 text-[11px] font-semibold text-primary/85">
                      {t('dashboard.projectNotes')}
                    </span>
                  )}
                </div>
              </div>
              <button onClick={() => setShowSubtitleModal(false)} className="rounded-xl p-2 text-outline transition-colors hover:bg-white/8 hover:text-white">
                <X className="h-6 w-6" />
              </button>
            </div>

            <div className="px-8 py-7 overflow-y-auto custom-scrollbar flex-1">
              <div className="grid gap-6 xl:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.95fr)]">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-medium text-outline">{t('translation.existingAssets')}</h3>
                    <span className="text-xs font-medium text-outline bg-white/5 px-3 py-1 rounded-md">
                      {t('stt.totalFiles').replace('{count}', subtitleAssets.length.toString())}
                    </span>
                  </div>

                  <div className="rounded-[24px] border border-white/6 bg-white/[0.02] min-h-[320px] relative overflow-hidden">
                    {isLoadingMaterials && (
                      <div className="absolute inset-0 bg-black/20 backdrop-blur-[1px] flex items-center justify-center z-10">
                        <Loader2 className="w-8 h-8 animate-spin text-primary" />
                      </div>
                    )}
                    <div className="p-5 space-y-3">
                      {subtitleAssets.length === 0 && !isLoadingMaterials && (
                        <div className="rounded-2xl border border-dashed border-white/8 bg-black/10 px-6 py-12 text-center text-outline/40 italic text-sm">
                          {t('player.noSubtitleAssets')}
                        </div>
                      )}

                      {subtitleAssets.map((asset) => (
                        <label
                          key={asset.name}
                          className={`flex items-center gap-4 rounded-2xl border px-4 py-4 cursor-pointer transition-all group ${
                            selectedSubtitleAssetName === asset.name
                              ? 'border-primary/35 bg-primary/10 shadow-[0_10px_30px_rgba(99,102,241,0.14)]'
                              : 'border-white/5 bg-white/[0.03] hover:bg-white/[0.05]'
                          }`}
                        >
                          <div className="w-11 h-11 rounded-xl bg-tertiary/12 border border-tertiary/10 flex items-center justify-center shrink-0">
                            <FileText className="w-5 h-5 text-tertiary" />
                          </div>
                          <div className="min-w-0 flex-1">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="text-sm font-semibold text-secondary truncate">{asset.name}</span>
                              <span className="rounded-full border border-white/8 bg-white/5 px-2 py-1 text-[10px] font-bold uppercase tracking-widest text-outline">
                                {getFileExt(asset.name).replace('.', '') || 'SUB'}
                              </span>
                            </div>
                            <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-outline">
                              <span className="rounded-full bg-white/5 px-2.5 py-1">{asset.size}</span>
                              <span className="rounded-full bg-white/5 px-2.5 py-1">{asset.date}</span>
                            </div>
                          </div>
                          <div
                            className={`w-5 h-5 rounded-full border-2 flex items-center justify-center transition-colors ${
                              selectedSubtitleAssetName === asset.name ? 'border-primary bg-primary' : 'border-outline/50 group-hover:border-outline'
                            }`}
                          >
                            {selectedSubtitleAssetName === asset.name && <div className="w-2.5 h-2.5 bg-white rounded-full" />}
                          </div>
                          <input
                            type="radio"
                            name="subtitleSelection"
                            className="hidden"
                            checked={selectedSubtitleAssetName === asset.name}
                            onChange={() => setSelectedSubtitleAssetName(asset.name)}
                          />
                        </label>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="space-y-5">
                  <div className="space-y-4">
                    <h3 className="text-sm font-medium text-outline">{t('translation.uploadNew')}</h3>
                <input
                  ref={subtitleFileInputRef}
                  type="file"
                  accept=".srt,.vtt,.ass,.ssa,.txt,.json,text/plain,application/json"
                  className="hidden"
                  onChange={(event) => {
                    void handleSubtitleFileUpload(event.target.files?.[0] || null);
                  }}
                />
                    <div
                      onClick={() => subtitleFileInputRef.current?.click()}
                      onDragOver={(event) => {
                        event.preventDefault();
                        event.currentTarget.classList.add('border-primary/50', 'bg-primary/5');
                      }}
                      onDragLeave={(event) => {
                        event.preventDefault();
                        event.currentTarget.classList.remove('border-primary/50', 'bg-primary/5');
                      }}
                      onDrop={(event) => {
                        event.currentTarget.classList.remove('border-primary/50', 'bg-primary/5');
                        void handleSubtitleFileDrop(event);
                      }}
                      className={`border-2 border-dashed border-white/10 rounded-[24px] p-10 flex flex-col items-center justify-center hover:border-primary/50 hover:bg-primary/5 transition-all cursor-pointer group min-h-[260px] ${
                        isUploadingSubtitle ? 'pointer-events-none opacity-50' : ''
                      }`}
                    >
                      {isUploadingSubtitle ? (
                        <div className="flex flex-col items-center">
                          <Loader2 className="w-10 h-10 text-primary animate-spin mb-3" />
                          <p className="text-sm font-bold text-primary">{t('dashboard.uploading')}</p>
                        </div>
                      ) : (
                        <>
                          <div className="w-16 h-16 bg-primary-container/20 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                            <FolderOpen className="w-8 h-8 text-primary" />
                          </div>
                          <p className="text-base font-bold text-secondary mb-1">{t('translation.dragDrop')}</p>
                          <p className="text-xs text-outline text-center leading-6">{t('player.subtitleSupportedFormats')}</p>
                        </>
                      )}
                    </div>
                  </div>

                  {project?.notes && (
                    <div className="rounded-[24px] border border-white/6 bg-white/[0.02] p-5 space-y-3">
                      <div className="text-[11px] uppercase tracking-widest text-outline/70">{t('dashboard.projectNotes')}</div>
                      <div className="max-h-[180px] overflow-y-auto custom-scrollbar text-sm leading-7 text-secondary">
                        {project.notes}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="p-6 border-t border-white/5 bg-[#13151A] flex justify-end gap-4">
              <button
                onClick={() => setShowSubtitleModal(false)}
                className="px-6 py-3 text-sm font-bold text-outline transition-colors hover:bg-white/8 hover:text-white rounded-lg"
              >
                {t('translation.cancel')}
              </button>
              <button
                onClick={handleConfirmSubtitleSelection}
                disabled={!selectedSubtitleAssetName || isLoadingSubtitleContent}
                className="px-8 py-3 bg-primary hover:bg-primary/90 text-white text-sm font-bold rounded-lg transition-colors shadow-lg shadow-primary/20 disabled:opacity-50 flex items-center gap-2"
              >
                {isLoadingSubtitleContent && <Loader2 className="w-4 h-4 animate-spin" />}
                {t('translation.confirm')}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
