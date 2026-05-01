import React from 'react';
import { Search, Download, Film, Music, ArrowRight, CheckCircle2, Loader2, AlertCircle, Upload } from 'lucide-react';
import { Project } from '../types';
import { useLanguage } from '../i18n/LanguageContext';
import type { Language } from '../i18n/translations';
import { isValidUrl, sanitizeInput } from '../utils/security';
import { PROJECT_STATUS } from '../project_status';
import RunMonitor, { type RunMonitorBadge } from './RunMonitor';
import FieldHelp from './FieldHelp';

type FetcherSourceMode = 'online' | 'upload';

interface VideoDownloaderProps {
  project: Project | null;
  onUpdateProject: (updates: Partial<Project>) => void | Promise<Project | null>;
  onNext: () => void;
}

const VIDEO_UPLOAD_EXTENSIONS = new Set(['mp4', 'mkv', 'mov', 'avi', 'wmv', 'webm', 'm4v', 'flv', 'ts', 'mpeg', 'mpg', 'm2ts', 'mts', '3gp', 'ogv', 'vob']);
const VIDEO_UPLOAD_ACCEPT = '.mp4,.mkv,.mov,.avi,.wmv,.webm,.m4v,.flv,.ts,.mpeg,.mpg,.m2ts,.mts,.3gp,.ogv,.vob';

function getSourceModeCopy(language: Language) {
  const maps = {
    'zh-tw': {
      modeLabel: '素材來源',
      modeHint: '同一個專案只建議使用一種來源，避免後續轉錄與翻譯流程混用。',
      onlineTitle: '線上連結',
      onlineHint: '先解析網址，再下載影片或音檔格式。',
      uploadTitle: '本地上傳',
      uploadHint: '直接上傳影片，系統會同步轉出標準音檔給 ASR 使用。',
      modeOnlineName: '線上連結',
      modeUploadName: '本地上傳',
      lockHint: '此專案已鎖定為「{mode}」模式。如要切換來源，請建立新專案。',
      unlockHint: '來源尚未鎖定。完成網址解析或上傳影片後會自動鎖定。',
      disabledUrlPlaceholder: '本地上傳模式啟用中，線上解析已停用。',
      uploadPanelTitle: '本地影片上傳',
      uploadPanelHint: '支援 mp4, mkv, mov, avi, wmv, webm, m4v, flv, ts。音檔會自動轉為 16kHz WAV。',
      uploadBtn: '上傳影片',
      uploadingBtn: '處理中...',
      currentAssetPrefix: '目前素材',
      noUploadYet: '尚未上傳影片。',
      switchToUploadHint: '切換到本地上傳模式後才可使用此功能。',
      workflowTitle: '本地上傳流程',
      workflowHint: '此模式不顯示線上解析與格式下載。',
      workflowVideoLabel: '影片',
      workflowAudioLabel: '標準音檔',
      workflowNoVideo: '尚未上傳',
      workflowNoAudio: '尚未產生',
      uploadedVideoLabel: '上傳影片',
      msgUploading: '正在上傳影片並建立標準音檔...',
      msgReady: '影片與標準音檔已就緒。',
      errModeLockedOnline: '此專案已鎖定線上來源，無法使用本地上傳。',
      errUnsupportedFormat: '僅支援上傳：mp4, mkv, mov, avi, wmv, webm, m4v, flv, ts, mpeg, mpg, m2ts, mts, 3gp, ogv, vob。',
      errUploadFailed: '影片上傳失敗。',
      errParseDisabled: '目前是本地上傳模式，無法解析線上網址。',
      errDownloadDisabled: '此專案已鎖定本地上傳模式，無法進行線上下載。',
    },
    'zh-cn': {
      modeLabel: '素材来源',
      modeHint: '同一个项目建议只使用一种来源，避免后续转录与翻译流程混用。',
      onlineTitle: '线上链接',
      onlineHint: '先解析网址，再下载视频或音频格式。',
      uploadTitle: '本地上传',
      uploadHint: '直接上传视频，系统会同步转出标准音频给 ASR 使用。',
      modeOnlineName: '线上链接',
      modeUploadName: '本地上传',
      lockHint: '此项目已锁定为“{mode}”模式。如需切换来源，请新建项目。',
      unlockHint: '来源尚未锁定。完成网址解析或上传视频后会自动锁定。',
      disabledUrlPlaceholder: '本地上传模式启用中，线上解析已停用。',
      uploadPanelTitle: '本地视频上传',
      uploadPanelHint: '支持 mp4, mkv, mov, avi, wmv, webm, m4v, flv, ts。音频会自动转换为 16kHz WAV。',
      uploadBtn: '上传视频',
      uploadingBtn: '处理中...',
      currentAssetPrefix: '当前素材',
      noUploadYet: '尚未上传视频。',
      switchToUploadHint: '切换到本地上传模式后才可使用此功能。',
      workflowTitle: '本地上传流程',
      workflowHint: '此模式不显示线上解析与格式下载。',
      workflowVideoLabel: '视频',
      workflowAudioLabel: '标准音频',
      workflowNoVideo: '尚未上传',
      workflowNoAudio: '尚未生成',
      uploadedVideoLabel: '上传视频',
      msgUploading: '正在上传视频并生成标准音频...',
      msgReady: '视频与标准音频已就绪。',
      errModeLockedOnline: '此项目已锁定线上来源，无法使用本地上传。',
      errUnsupportedFormat: '仅支持上传：mp4, mkv, mov, avi, wmv, webm, m4v, flv, ts, mpeg, mpg, m2ts, mts, 3gp, ogv, vob。',
      errUploadFailed: '视频上传失败。',
      errParseDisabled: '当前为本地上传模式，无法解析线上网址。',
      errDownloadDisabled: '此项目已锁定本地上传模式，无法进行线上下载。',
    },
    en: {
      modeLabel: 'Source Mode',
      modeHint: 'Use one source path per project to keep downstream transcription and translation stable.',
      onlineTitle: 'Online Link',
      onlineHint: 'Parse URL metadata and download selected video/audio formats.',
      uploadTitle: 'Local Upload',
      uploadHint: 'Upload your own video and auto-create standardized audio for ASR.',
      modeOnlineName: 'Online Link',
      modeUploadName: 'Local Upload',
      lockHint: 'This project is locked to "{mode}" mode. Create a new project to switch source path.',
      unlockHint: 'Source mode is not locked yet. It will lock after URL parsing or local upload.',
      disabledUrlPlaceholder: 'Local upload mode is active. Online parsing is disabled.',
      uploadPanelTitle: 'Local Video Upload',
      uploadPanelHint: 'Supported: mp4, mkv, mov, avi, wmv, webm, m4v, flv, ts. Audio is auto-converted to 16kHz WAV.',
      uploadBtn: 'Upload Video',
      uploadingBtn: 'Processing...',
      currentAssetPrefix: 'Current asset',
      noUploadYet: 'No video uploaded yet.',
      switchToUploadHint: 'Switch to Local Upload mode to use this action.',
      workflowTitle: 'Local Upload Workflow',
      workflowHint: 'Online parsing and format selection are disabled in this source mode.',
      workflowVideoLabel: 'Video',
      workflowAudioLabel: 'Standard Audio',
      workflowNoVideo: 'Not uploaded yet',
      workflowNoAudio: 'Not generated yet',
      uploadedVideoLabel: 'Uploaded video',
      msgUploading: 'Uploading video and creating standardized audio...',
      msgReady: 'Video and standardized audio are ready.',
      errModeLockedOnline: 'This project is locked to online source mode. Local upload is disabled.',
      errUnsupportedFormat: 'Supported upload formats: mp4, mkv, mov, avi, wmv, webm, m4v, flv, ts, mpeg, mpg, m2ts, mts, 3gp, ogv, vob.',
      errUploadFailed: 'Video upload failed.',
      errParseDisabled: 'Local upload mode is active. Online URL parsing is disabled.',
      errDownloadDisabled: 'This project is locked to local upload mode. Online download is disabled.',
    },
    jp: {
      modeLabel: '素材ソース',
      modeHint: '後続の文字起こしと翻訳を安定させるため、1プロジェクトにつき1つのソース方式を推奨します。',
      onlineTitle: 'オンラインリンク',
      onlineHint: 'URLを解析して、必要な動画/音声形式をダウンロードします。',
      uploadTitle: 'ローカルアップロード',
      uploadHint: '動画を直接アップロードすると、ASR用の標準音声が自動生成されます。',
      modeOnlineName: 'オンラインリンク',
      modeUploadName: 'ローカルアップロード',
      lockHint: 'このプロジェクトは「{mode}」モードに固定されています。切り替えるには新しいプロジェクトを作成してください。',
      unlockHint: 'ソースモードは未固定です。URL解析または動画アップロード後に自動固定されます。',
      disabledUrlPlaceholder: 'ローカルアップロードモードが有効なため、オンライン解析は無効です。',
      uploadPanelTitle: 'ローカル動画アップロード',
      uploadPanelHint: '対応形式: mp4, mkv, mov, avi, wmv, webm, m4v, flv, ts。音声は16kHz WAVへ自動変換されます。',
      uploadBtn: '動画をアップロード',
      uploadingBtn: '処理中...',
      currentAssetPrefix: '現在の素材',
      noUploadYet: 'まだ動画はアップロードされていません。',
      switchToUploadHint: 'この機能を使うにはローカルアップロードモードに切り替えてください。',
      workflowTitle: 'ローカルアップロードフロー',
      workflowHint: 'このモードではオンライン解析と形式選択は表示されません。',
      workflowVideoLabel: '動画',
      workflowAudioLabel: '標準音声',
      workflowNoVideo: '未アップロード',
      workflowNoAudio: '未生成',
      uploadedVideoLabel: 'アップロード動画',
      msgUploading: '動画をアップロードし、標準音声を生成しています...',
      msgReady: '動画と標準音声の準備が完了しました。',
      errModeLockedOnline: 'このプロジェクトはオンラインソースに固定されているため、ローカルアップロードは無効です。',
      errUnsupportedFormat: 'アップロード対応形式: mp4, mkv, mov, avi, wmv, webm, m4v, flv, ts, mpeg, mpg, m2ts, mts, 3gp, ogv, vob。',
      errUploadFailed: '動画アップロードに失敗しました。',
      errParseDisabled: '現在はローカルアップロードモードのため、オンラインURL解析は無効です。',
      errDownloadDisabled: 'このプロジェクトはローカルアップロードモードに固定されているため、オンラインダウンロードは無効です。',
    },
    de: {
      modeLabel: 'Quellmodus',
      modeHint: 'Verwenden Sie pro Projekt nur einen Quellenpfad, damit Transkription und Übersetzung stabil bleiben.',
      onlineTitle: 'Online-Link',
      onlineHint: 'URL-Metadaten analysieren und gewünschte Video-/Audioformate herunterladen.',
      uploadTitle: 'Lokaler Upload',
      uploadHint: 'Eigenes Video hochladen und standardisiertes Audio für ASR automatisch erzeugen.',
      modeOnlineName: 'Online-Link',
      modeUploadName: 'Lokaler Upload',
      lockHint: 'Dieses Projekt ist auf den Modus "{mode}" festgelegt. Zum Wechsel bitte ein neues Projekt erstellen.',
      unlockHint: 'Der Quellmodus ist noch nicht gesperrt. Nach URL-Analyse oder lokalem Upload wird er automatisch gesperrt.',
      disabledUrlPlaceholder: 'Lokaler Upload-Modus ist aktiv. Online-Analyse ist deaktiviert.',
      uploadPanelTitle: 'Lokaler Video-Upload',
      uploadPanelHint: 'Unterstützt: mp4, mkv, mov, avi, wmv, webm, m4v, flv, ts. Audio wird automatisch in 16kHz WAV umgewandelt.',
      uploadBtn: 'Video hochladen',
      uploadingBtn: 'Verarbeitung...',
      currentAssetPrefix: 'Aktuelles Asset',
      noUploadYet: 'Noch kein Video hochgeladen.',
      switchToUploadHint: 'Wechseln Sie zum lokalen Upload-Modus, um diese Aktion zu nutzen.',
      workflowTitle: 'Lokaler Upload-Workflow',
      workflowHint: 'Online-Analyse und Formatauswahl sind in diesem Modus deaktiviert.',
      workflowVideoLabel: 'Video',
      workflowAudioLabel: 'Standard-Audio',
      workflowNoVideo: 'Noch nicht hochgeladen',
      workflowNoAudio: 'Noch nicht erzeugt',
      uploadedVideoLabel: 'Hochgeladenes Video',
      msgUploading: 'Video wird hochgeladen und standardisiertes Audio wird erzeugt...',
      msgReady: 'Video und standardisiertes Audio sind bereit.',
      errModeLockedOnline: 'Dieses Projekt ist auf Online-Quelle gesperrt. Lokaler Upload ist deaktiviert.',
      errUnsupportedFormat: 'Unterstützte Upload-Formate: mp4, mkv, mov, avi, wmv, webm, m4v, flv, ts, mpeg, mpg, m2ts, mts, 3gp, ogv, vob.',
      errUploadFailed: 'Video-Upload fehlgeschlagen.',
      errParseDisabled: 'Lokaler Upload-Modus ist aktiv. Online-URL-Analyse ist deaktiviert.',
      errDownloadDisabled: 'Dieses Projekt ist auf lokalen Upload-Modus gesperrt. Online-Download ist deaktiviert.',
    },
  } as const;

  return maps[language];
}

function inferProjectSourceMode(project: Project | null): FetcherSourceMode | null {
  const explicit = String(project?.mediaSourceType || '').trim().toLowerCase();
  if (explicit === 'online' || explicit === 'upload') return explicit;

  const metadataMode = String(project?.videoMetadata?.sourceMode || '').trim().toLowerCase();
  if (metadataMode === 'online' || metadataMode === 'upload') return metadataMode;

  if (String(project?.videoMetadata?.lastUrl || '').trim()) return 'online';
  return null;
}

function getFileNameFromClientPath(rawPath: string) {
  const value = String(rawPath || '').trim();
  if (!value) return '';
  const clean = value.split('?')[0].split('#')[0];
  const segments = clean.split('/').filter(Boolean);
  const fileName = segments.length > 0 ? segments[segments.length - 1] : '';
  try {
    return decodeURIComponent(fileName);
  } catch {
    return fileName;
  }
}

export default function VideoDownloader({ project, onUpdateProject, onNext }: VideoDownloaderProps) {
  const { t, language } = useLanguage();
  const sourceCopy = React.useMemo(() => getSourceModeCopy(language), [language]);
  const [url, setUrl] = React.useState('');
  const [sourceMode, setSourceMode] = React.useState<FetcherSourceMode>('online');
  const [isParsing, setIsParsing] = React.useState(false);
  const [isUploadingVideo, setIsUploadingVideo] = React.useState(false);
  const [videoData, setVideoData] = React.useState<any>(null);
  const [selectedVideoId, setSelectedVideoId] = React.useState<string>('');
  const [selectedAudioId, setSelectedAudioId] = React.useState<string>('');
  const [downloadProgress, setDownloadProgress] = React.useState(0);
  const [downloadSpeed, setDownloadSpeed] = React.useState<string | null>(null);
  const [downloadEta, setDownloadEta] = React.useState<string | null>(null);
  const [isDownloading, setIsDownloading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [downloadMsg, setDownloadMsg] = React.useState<string | null>(null);
  const [parseProgress, setParseProgress] = React.useState(0);
  const [lastSyncedProjectId, setLastSyncedProjectId] = React.useState<string | null>(null);
  const videoUploadInputRef = React.useRef<HTMLInputElement | null>(null);

  const projectSourceMode = inferProjectSourceMode(project);
  const effectiveSourceMode: FetcherSourceMode = projectSourceMode || sourceMode;
  const isSourceModeLocked = Boolean(projectSourceMode);
  const hasDownloadedMedia = Boolean(String(project?.videoUrl || '').trim() || String(project?.audioUrl || '').trim());
  const canProceedToAsr = !isDownloading && !isUploadingVideo && (downloadProgress >= 100 || hasDownloadedMedia);
  const selectedVideoFormat = videoData?.formats?.find((format: any) => format.id === selectedVideoId) ?? null;
  const selectedAudioFormat = videoData?.audioFormats?.find((format: any) => format.id === selectedAudioId) ?? null;
  const hasProject = Boolean(project);
  const parseReady = Boolean(videoData) && parseProgress === 100;
  const videoReady = Boolean(project?.videoUrl);
  const audioReady = Boolean(project?.audioUrl);
  const onlineModeEnabled = effectiveSourceMode === 'online';
  const uploadModeEnabled = effectiveSourceMode === 'upload';
  const uploadedFileName = getFileNameFromClientPath(String(project?.videoUrl || ''));
  const uploadPanelReady = uploadModeEnabled && hasDownloadedMedia;
  const stepTwoReady = onlineModeEnabled ? parseReady : uploadPanelReady;
  const videoActionLabel = uploadModeEnabled ? sourceCopy.uploadedVideoLabel : t('fetcher.downloadBtn');
  const downloaderStatusLabel =
    downloadMsg || ((isDownloading || isUploadingVideo)
      ? t('fetcher.downloading')
      : (downloadProgress === 100 ? t('fetcher.downloadComplete') : t('fetcher.pending')));
  const transferStatusDetails = [downloadSpeed, downloadEta].filter(Boolean).join(' - ');
  const downloadProgressStatusLabel = transferStatusDetails
    ? `${downloaderStatusLabel} - ${transferStatusDetails}`
    : downloaderStatusLabel;
  const downloaderMonitorBadges = React.useMemo<RunMonitorBadge[]>(() => {
    const badges: RunMonitorBadge[] = [
      {
        label: onlineModeEnabled ? sourceCopy.modeOnlineName : sourceCopy.modeUploadName,
        tone: 'info',
      },
    ];
    if (videoReady && audioReady) {
      badges.push({ label: t('fetcher.mediaReadyTitle'), tone: 'success' });
    } else if (stepTwoReady) {
      badges.push({ label: t('fetcher.ready'), tone: 'success' });
    }
    return badges;
  }, [audioReady, onlineModeEnabled, sourceCopy.modeOnlineName, sourceCopy.modeUploadName, stepTwoReady, t, videoReady]);

  const handleProceedToAsr = () => {
    if (!canProceedToAsr) return;
    onUpdateProject({ status: PROJECT_STATUS.SPEECH_TO_TEXT });
    onNext();
  };

  React.useEffect(() => {
    if (project && project.id !== lastSyncedProjectId) {
      const inferredSourceMode = inferProjectSourceMode(project);
      setSourceMode(inferredSourceMode || 'online');

      if (project.videoMetadata) {
        setVideoData(project.videoMetadata);
        setUrl(project.videoMetadata.lastUrl || '');
        if (project.videoMetadata.formats?.length > 0) setSelectedVideoId(project.videoMetadata.formats[0].id);
        if (project.videoMetadata.audioFormats?.length > 0) setSelectedAudioId(project.videoMetadata.audioFormats[0].id);
        setParseProgress(inferredSourceMode === 'online' ? 100 : 0);
      } else {
        setVideoData(null);
        setUrl('');
        setParseProgress(0);
      }

      if (inferredSourceMode === 'upload') {
        setVideoData(null);
        setUrl('');
      }

      setError(null);
      setLastSyncedProjectId(project.id);
    } else if (!project && lastSyncedProjectId !== null) {
      setSourceMode('online');
      setVideoData(null);
      setUrl('');
      setParseProgress(0);
      setError(null);
      setLastSyncedProjectId(null);
    }
  }, [project, lastSyncedProjectId]);

  const formatFileSize = (bytes: number) => {
    if (!bytes) return t('fetcher.unknownSize');
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  const handleSourceModeChange = (nextMode: FetcherSourceMode) => {
    if (isSourceModeLocked || sourceMode === nextMode) return;
    setSourceMode(nextMode);
    setError(null);
    if (nextMode === 'upload') {
      setVideoData(null);
      setParseProgress(0);
      setUrl('');
    }
  };

  const handleVideoUpload = async (file: File | null) => {
    if (!project) {
      setError(t('fetcher.errorNoProject'));
      return;
    }
    if (!file) return;

    if (effectiveSourceMode !== 'upload') {
      setError(sourceCopy.errModeLockedOnline);
      return;
    }

    const extension = String(file.name || '')
      .split('.')
      .pop()
      ?.trim()
      .toLowerCase();

    if (!extension || !VIDEO_UPLOAD_EXTENSIONS.has(extension)) {
      setError(sourceCopy.errUnsupportedFormat);
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setIsUploadingVideo(true);
    setDownloadProgress(12);
    setDownloadSpeed(null);
    setDownloadEta(null);
    setDownloadMsg(sourceCopy.msgUploading);
    setError(null);
    setVideoData(null);
    setParseProgress(0);

    try {
      const response = await fetch(`/api/projects/${encodeURIComponent(project.id)}/upload-video`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data?.error) {
        throw new Error(String(data?.error || sourceCopy.errUploadFailed));
      }

      setDownloadProgress(100);
      setDownloadMsg(sourceCopy.msgReady);

      const nextMetadata = {
        sourceMode: 'upload',
        localFileName: String(data?.file?.filename || file.name || '').trim(),
      };

      await Promise.resolve(
        onUpdateProject({
          mediaSourceType: 'upload',
          videoTitle: String(data?.videoTitle || file.name || '').trim() || project.name,
          videoUrl: String(data?.videoPath || '').trim(),
          audioUrl: String(data?.audioPath || '').trim(),
          videoMetadata: nextMetadata,
        })
      );
    } catch (err: any) {
      setDownloadProgress(0);
      setDownloadMsg(null);
      setError(String(err?.message || sourceCopy.errUploadFailed));
    } finally {
      setIsUploadingVideo(false);
      if (videoUploadInputRef.current) {
        videoUploadInputRef.current.value = '';
      }
    }
  };

  const handleParse = async () => {
    if (!project) {
      setError(t('fetcher.errorNoProject'));
      return;
    }

    if (effectiveSourceMode !== 'online') {
      setError(sourceCopy.errParseDisabled);
      return;
    }

    const trimmedUrl = sanitizeInput(url).trim();
    if (!trimmedUrl) return;

    if (!isValidUrl(trimmedUrl)) {
      setError(t('fetcher.errorInvalidUrl'));
      return;
    }

    setIsParsing(true);
    setParseProgress(30);
    setError(null);
    setVideoData(null);

    try {
      const response = await fetch('/api/parse-video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: trimmedUrl, projectId: project.id })
      });
      setParseProgress(70);
      const data = await response.json();
      if (data.error) throw new Error(data.error);

      setVideoData(data);
      if (data.formats?.length > 0) setSelectedVideoId(data.formats[0].id);
      if (data.audioFormats?.length > 0) setSelectedAudioId(data.audioFormats[0].id);
      setParseProgress(100);

      onUpdateProject({
        mediaSourceType: 'online',
        videoTitle: data.title,
        videoMetadata: { ...data, lastUrl: trimmedUrl, sourceMode: 'online' }
      });
    } catch (err: any) {
      setError(err.message || t('fetcher.errorParseFailed'));
      setParseProgress(0);
    } finally {
      setIsParsing(false);
    }
  };

  const handleDownload = (type: 'video' | 'audio') => {
    if (!project || !videoData) return;
    if (effectiveSourceMode !== 'online') {
      setError(sourceCopy.errDownloadDisabled);
      return;
    }

    const formatId = type === 'video' ? selectedVideoId : selectedAudioId;
    if (!formatId) return;

    setIsDownloading(true);
    setDownloadProgress(0);
    setDownloadSpeed(null);
    setDownloadEta(null);
    setError(null);
    setDownloadMsg(null);

    const urlParams = new URLSearchParams({
      url: sanitizeInput(url).trim(),
      formatId,
      type,
    });

    const eventSource = new EventSource(`/api/download-progress/${project.id}?${urlParams.toString()}`);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.error) {
        setError(data.error);
        setIsDownloading(false);
        eventSource.close();
        return;
      }

      if (data.status === 'downloading') {
        setDownloadProgress(data.progress || 0);
        setDownloadSpeed(data.speed);
        setDownloadEta(data.eta);
      } else if (data.status === 'extracting') {
        setDownloadMsg(data.msg);
        setDownloadProgress(100);
        setDownloadSpeed(null);
        setDownloadEta(null);
      } else if (data.status === 'finished') {
        setIsDownloading(false);
        setDownloadMsg(t('status.completed'));
        const nextUpdates: Partial<Project> = {
          mediaSourceType: 'online',
          videoTitle: videoData.title,
          videoMetadata: { ...videoData, lastUrl: sanitizeInput(url).trim(), sourceMode: 'online' },
        };

        if (type === 'video') {
          nextUpdates.videoUrl = data.videoPath;
          nextUpdates.audioUrl = data.audioPath;
        } else {
          nextUpdates.audioUrl = data.audioPath || data.videoPath;
        }

        onUpdateProject(nextUpdates);
        eventSource.close();
      }
    };

    eventSource.onerror = () => {
      setError(t('fetcher.errorConnectionInterrupted'));
      setIsDownloading(false);
      eventSource.close();
    };
  };

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex flex-col gap-3 xl:flex-row xl:items-end xl:justify-between">
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="text-3xl font-bold text-secondary tracking-tight">{t('fetcher.title')}</h2>
            {project && (
              <div className="px-3 py-1 bg-primary-container/10 border border-primary-container/20 rounded-full flex items-center gap-2 animate-in fade-in zoom-in duration-500">
                <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse shadow-[0_0_8px_rgba(var(--primary),0.8)]" />
                <span className="text-[10px] font-bold text-primary uppercase tracking-widest">{project.name}</span>
              </div>
            )}
          </div>
          <p className="text-outline">{t('fetcher.subtitle')}</p>
        </div>
      </div>

      {!hasProject && (
        <section className="rounded-[28px] border border-warning/25 bg-warning/8 px-5 py-4 shadow-lg">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-warning/12 text-warning">
              <AlertCircle className="h-5 w-5" />
            </div>
            <div className="space-y-1">
              <p className="text-sm font-bold text-secondary">{t('fetcher.projectMissingTitle')}</p>
              <p className="text-sm text-outline leading-relaxed">{t('fetcher.projectMissingSubtitle')}</p>
            </div>
          </div>
        </section>
      )}

      <section className="bg-surface-container rounded-[30px] p-6 border border-white/5 shadow-2xl relative overflow-hidden">
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
        <div className="relative z-10 space-y-5">
          <div className="space-y-3">
            <span className="text-[11px] font-black text-primary tracking-[0.22em] uppercase">{t('fetcher.stepUrl')}</span>
            <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <label className="block text-lg font-semibold text-secondary">{sourceCopy.modeLabel}</label>
                  <FieldHelp
                    ariaLabel={t('fetcher.help.sourceModeAria')}
                    title={sourceCopy.modeLabel}
                    body={t('fetcher.help.sourceModeBody')}
                  />
                </div>
              </div>
              {project && (
                <div className="inline-flex items-center gap-2 rounded-full border border-white/8 bg-white/[0.03] px-3 py-1.5 text-[11px] font-semibold text-outline">
                  <span className="h-2 w-2 rounded-full bg-primary shadow-[0_0_8px_rgba(var(--primary),0.7)]" />
                  {t('fetcher.selectedProject')}: {project.name}
                </div>
              )}
            </div>

            <div className="grid gap-3 md:grid-cols-2">
              <button
                type="button"
                onClick={() => handleSourceModeChange('online')}
                disabled={isSourceModeLocked || isUploadingVideo || isDownloading}
                className={`rounded-2xl border px-4 py-4 text-left transition-all disabled:opacity-60 disabled:cursor-not-allowed ${
                  onlineModeEnabled
                    ? 'border-primary/40 bg-primary/12 shadow-[0_8px_30px_rgba(var(--primary),0.14)]'
                    : 'border-white/8 bg-white/[0.02] hover:bg-white/[0.04]'
                }`}
              >
                <p className="text-sm font-bold text-secondary">{sourceCopy.onlineTitle}</p>
                <p className="mt-1 text-xs text-outline">{sourceCopy.onlineHint}</p>
              </button>
              <button
                type="button"
                onClick={() => handleSourceModeChange('upload')}
                disabled={isSourceModeLocked || isUploadingVideo || isDownloading}
                className={`rounded-2xl border px-4 py-4 text-left transition-all disabled:opacity-60 disabled:cursor-not-allowed ${
                  uploadModeEnabled
                    ? 'border-tertiary/45 bg-tertiary/12 shadow-[0_8px_30px_rgba(107,255,193,0.18)]'
                    : 'border-white/8 bg-white/[0.02] hover:bg-white/[0.04]'
                }`}
              >
                <p className="text-sm font-bold text-secondary">{sourceCopy.uploadTitle}</p>
                <p className="mt-1 text-xs text-outline">{sourceCopy.uploadHint}</p>
              </button>
            </div>

            <div className="rounded-xl border border-white/8 bg-white/[0.02] px-4 py-3 text-xs text-outline">
              {isSourceModeLocked
                ? sourceCopy.lockHint.replace('{mode}', onlineModeEnabled ? sourceCopy.modeOnlineName : sourceCopy.modeUploadName)
                : sourceCopy.unlockHint}
            </div>
          </div>

          {onlineModeEnabled && (
            <div className="space-y-3">
              <div>
                <div className="flex items-center gap-2">
                  <label className="block text-base font-semibold text-secondary">{t('fetcher.urlLabel')}</label>
                  <FieldHelp
                    ariaLabel={t('fetcher.help.urlAria')}
                    title={t('fetcher.help.urlTitle')}
                    body={t('fetcher.help.urlBody')}
                  />
                </div>
              </div>
              <div className="flex flex-col gap-4 xl:flex-row">
                <div className="flex-1 relative">
                  <input
                    type="text"
                    value={url}
                    onChange={(e) => {
                      setUrl(sanitizeInput(e.target.value));
                      if (error) setError(null);
                    }}
                    disabled={!hasProject || isUploadingVideo}
                    placeholder={t('fetcher.urlPlaceholder')}
                    className={`w-full bg-surface-container-lowest border rounded-2xl px-5 py-4 text-secondary focus:ring-1 focus:ring-primary-container focus:border-transparent outline-none transition-all placeholder:text-outline/40 disabled:opacity-60 disabled:cursor-not-allowed ${
                      error ? 'border-error/50' : 'border-white/5'
                    }`}
                  />
                  {error && (
                    <div className="flex items-center gap-2 mt-2 text-error text-xs font-bold animate-in slide-in-from-top-1 duration-200">
                      <AlertCircle className="w-3.5 h-3.5" />
                      {error}
                    </div>
                  )}
                </div>
                <button
                  onClick={handleParse}
                  disabled={isParsing || !url.trim() || !hasProject || isUploadingVideo}
                  className="min-w-[176px] bg-primary-container hover:bg-primary-container/90 text-white px-8 py-4 rounded-2xl font-bold transition-all active:scale-95 flex items-center justify-center gap-3 shadow-lg shadow-primary-container/20 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isParsing ? <Loader2 className="w-5 h-5 animate-spin" /> : <Search className="w-5 h-5" />}
                  <span className="whitespace-nowrap">{t('fetcher.parseBtn')}</span>
                </button>
              </div>
            </div>
          )}

          {uploadModeEnabled && (
            <div className="rounded-2xl border border-white/8 bg-white/[0.02] px-4 py-4">
              <input
                ref={videoUploadInputRef}
                type="file"
                className="hidden"
                accept={VIDEO_UPLOAD_ACCEPT}
                onChange={(event) => {
                  void handleVideoUpload(event.target.files?.[0] || null);
                }}
              />
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className="text-sm font-semibold text-secondary">{sourceCopy.uploadPanelTitle}</p>
                  <p className="text-xs text-outline">{sourceCopy.uploadPanelHint}</p>
                </div>
                <button
                  type="button"
                  onClick={() => videoUploadInputRef.current?.click()}
                  disabled={!hasProject || isUploadingVideo || isParsing || isDownloading}
                  className="min-w-[176px] rounded-xl border border-tertiary/35 bg-tertiary/10 px-5 py-3 text-sm font-bold text-tertiary transition-all hover:bg-tertiary/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isUploadingVideo ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                  <span>{isUploadingVideo ? sourceCopy.uploadingBtn : sourceCopy.uploadBtn}</span>
                </button>
              </div>
              <p className="mt-3 text-xs text-outline">
                {uploadedFileName
                  ? `${sourceCopy.currentAssetPrefix}: ${uploadedFileName}`
                  : sourceCopy.noUploadYet}
              </p>
            </div>
          )}
        </div>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.65fr)_minmax(320px,0.9fr)] gap-6 items-start">
        <section className="bg-surface-container rounded-[30px] border border-white/5 overflow-hidden shadow-2xl">
          <div className="border-b border-white/5 px-6 py-5">
            <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
              <div className="space-y-1">
                <span className="text-[11px] font-black text-primary tracking-[0.22em] uppercase">{t('fetcher.stepPrepare')}</span>
                <div className="flex items-center gap-2">
                  <h3 className="text-xl font-semibold text-secondary">{t('fetcher.selectionTitle')}</h3>
                  <FieldHelp
                    ariaLabel={t('fetcher.help.formatAria')}
                    title={t('fetcher.help.formatTitle')}
                    body={t('fetcher.help.formatBody')}
                  />
                </div>
              </div>
              {stepTwoReady && (
                <div className="inline-flex items-center gap-2 rounded-full border border-primary/15 bg-primary/8 px-3 py-1.5 text-[11px] font-semibold text-primary">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  {t('fetcher.readyToDownload')}
                </div>
              )}
            </div>
          </div>

          {onlineModeEnabled ? (videoData ? (
            <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.15fr)_minmax(280px,0.85fr)]">
              <div className="space-y-5 p-6 border-b border-white/5 xl:border-b-0 xl:border-r">
                <div className="aspect-video overflow-hidden rounded-[26px] border border-white/5 bg-black/25">
                  <img
                    src={videoData.thumbnail}
                    alt={videoData.title}
                    className="h-full w-full object-cover transition-transform duration-700 hover:scale-[1.03]"
                    referrerPolicy="no-referrer"
                  />
                </div>

                <div className="space-y-4">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                    <div className="space-y-2">
                      <h4 className="text-2xl font-bold text-secondary leading-snug">{videoData.title}</h4>
                      <div className="flex flex-wrap items-center gap-2 text-xs font-semibold text-outline">
                        {videoData.uploader && <span>{videoData.uploader}</span>}
                        {videoData.uploader && videoData.uploadDate && <span className="opacity-35">•</span>}
                        {videoData.uploadDate && <span>{videoData.uploadDate}</span>}
                        {(videoData.uploader || videoData.uploadDate) && videoData.viewCount && <span className="opacity-35">•</span>}
                        {videoData.viewCount && <span>{videoData.viewCount.toLocaleString()} {t('status.views')}</span>}
                      </div>
                    </div>
                    <div className="inline-flex items-center rounded-full border border-primary/20 bg-primary/10 px-3 py-1.5 text-[11px] font-bold text-primary shrink-0">
                      {videoData.duration}
                    </div>
                  </div>

                  <div className="rounded-2xl border border-white/5 bg-white/[0.03] p-4">
                    <p className="text-xs font-bold uppercase tracking-[0.16em] text-outline/70 mb-2">{t('fetcher.videoTitle')}</p>
                    <p className="text-sm leading-relaxed text-outline max-h-[120px] overflow-y-auto pr-1 custom-scrollbar">
                      {videoData.description ? videoData.description : <span className="opacity-55 italic">{t('fetcher.noDescription')}</span>}
                    </p>
                  </div>
                </div>
              </div>

              <div className="space-y-4 p-6 bg-white/[0.02]">
                <div className="rounded-[24px] border border-white/5 bg-surface-container-high p-4 space-y-3">
                  <div className="flex items-center gap-3">
                    <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-primary/12 text-primary">
                      <Film className="h-5 w-5" />
                    </div>
                    <div>
                      <p className="text-sm font-bold text-secondary">{t('fetcher.downloadBtn')}</p>
                      <p className="text-xs text-outline">{selectedVideoFormat ? `${selectedVideoFormat.quality} (${selectedVideoFormat.ext})` : t('fetcher.noSelection')}</p>
                    </div>
                  </div>

                  <div className="relative">
                    <select
                      value={selectedVideoId}
                      onChange={(e) => setSelectedVideoId(e.target.value)}
                      className="w-full bg-surface-container-highest text-white text-sm py-3 px-4 rounded-xl border border-white/5 focus:ring-1 focus:ring-primary-container outline-none cursor-pointer appearance-none transition-all hover:bg-surface-container-highest/80"
                    >
                      {videoData.formats.map((f: any) => (
                        <option key={f.id} value={f.id} className="bg-surface-container-high text-white">
                          {f.quality} ({f.ext}) - {formatFileSize(f.size)}
                        </option>
                      ))}
                    </select>
                    <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-outline/50">
                      <Film className="w-4 h-4" />
                    </div>
                  </div>

                  <button
                    onClick={() => handleDownload('video')}
                    disabled={isDownloading || isUploadingVideo}
                    className="w-full bg-primary-container hover:bg-primary-container/90 text-white py-3 rounded-xl flex items-center justify-center gap-2 transition-all font-bold text-sm shadow-lg shadow-primary-container/10 active:scale-95 disabled:opacity-50"
                  >
                    <Download className="w-4 h-4" />
                    <span className="whitespace-nowrap">{t('fetcher.downloadBtn')}</span>
                  </button>
                </div>

                <div className="rounded-[24px] border border-white/5 bg-surface-container-high p-4 space-y-3">
                  <div className="flex items-center gap-3">
                    <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-tertiary/12 text-tertiary">
                      <Music className="h-5 w-5" />
                    </div>
                    <div>
                      <p className="text-sm font-bold text-secondary">{t('fetcher.downloadAudio')}</p>
                      <p className="text-xs text-outline">{selectedAudioFormat ? `${selectedAudioFormat.quality} (${selectedAudioFormat.ext})` : t('fetcher.noSelection')}</p>
                    </div>
                  </div>

                  <div className="relative">
                    <select
                      value={selectedAudioId}
                      onChange={(e) => setSelectedAudioId(e.target.value)}
                      className="w-full bg-surface-container-highest text-white text-sm py-3 px-4 rounded-xl border border-white/5 focus:ring-1 focus:ring-tertiary outline-none cursor-pointer appearance-none transition-all hover:bg-surface-container-highest/80"
                    >
                      {videoData.audioFormats.map((f: any) => (
                        <option key={f.id} value={f.id} className="bg-surface-container-high text-white">
                          {f.quality} ({f.ext}) - {formatFileSize(f.size)}
                        </option>
                      ))}
                    </select>
                    <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-outline/50">
                      <Music className="w-4 h-4" />
                    </div>
                  </div>

                  <button
                    onClick={() => handleDownload('audio')}
                    disabled={isDownloading || isUploadingVideo}
                    className="w-full bg-tertiary/10 hover:bg-tertiary/20 text-tertiary border border-tertiary/30 py-3 rounded-xl flex items-center justify-center gap-2 transition-all font-bold text-sm active:scale-95 disabled:opacity-50"
                  >
                    <Music className="w-4 h-4" />
                    <span className="whitespace-nowrap">{t('fetcher.downloadAudio')}</span>
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="px-6 py-10">
              <div className="rounded-[28px] border border-dashed border-white/10 bg-white/[0.02] px-6 py-12 text-center">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-[22px] bg-white/[0.04] text-outline/40">
                  {isParsing ? <Loader2 className="h-8 w-8 animate-spin" /> : <Film className="h-8 w-8" />}
                </div>
                <div className="mt-5 space-y-2">
                  <h4 className="text-lg font-semibold text-secondary">{isParsing ? t('status.processing') : t('fetcher.emptyTitle')}</h4>
                  <p className="mx-auto max-w-xl text-sm leading-relaxed text-outline">
                    {isParsing ? t('fetcher.parseProgress') : t('fetcher.emptySubtitle')}
                  </p>
                </div>
              </div>
            </div>
          )) : (
            <div className="px-6 py-10">
              <div className="rounded-[28px] border border-white/10 bg-white/[0.02] px-6 py-10">
                <div className="flex flex-col gap-4">
                  <div className="flex items-center gap-3">
                    <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-tertiary/15 text-tertiary">
                      {isUploadingVideo ? <Loader2 className="h-6 w-6 animate-spin" /> : <Upload className="h-6 w-6" />}
                    </div>
                    <div>
                      <h4 className="text-lg font-semibold text-secondary">{sourceCopy.workflowTitle}</h4>
                      <p className="text-sm text-outline">{sourceCopy.workflowHint}</p>
                    </div>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="rounded-2xl border border-white/8 bg-surface-container-high px-4 py-3">
                      <div className="text-[11px] uppercase tracking-widest text-outline/70">{sourceCopy.workflowVideoLabel}</div>
                      <div className="mt-1 text-sm font-semibold text-secondary truncate">{uploadedFileName || sourceCopy.workflowNoVideo}</div>
                    </div>
                    <div className="rounded-2xl border border-white/8 bg-surface-container-high px-4 py-3">
                      <div className="text-[11px] uppercase tracking-widest text-outline/70">{sourceCopy.workflowAudioLabel}</div>
                      <div className="mt-1 text-sm font-semibold text-secondary truncate">
                        {audioReady ? getFileNameFromClientPath(String(project?.audioUrl || '')) || 'audio.wav' : sourceCopy.workflowNoAudio}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>

        <aside className="bg-surface-container rounded-[30px] p-6 border border-white/5 relative overflow-hidden flex flex-col gap-5 shadow-xl">
          <div className="absolute -top-24 -right-24 w-48 h-48 bg-primary/10 blur-[100px] pointer-events-none" />
          <div className="absolute -bottom-24 -left-24 w-48 h-48 bg-secondary/10 blur-[100px] pointer-events-none" />

          <div className="relative z-10 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-secondary">{t('fetcher.stepControl')}</h3>
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse shadow-[0_0_10px_rgba(var(--primary),0.8)]" />
          </div>

          <div className="relative z-10 rounded-[24px] border border-white/5 bg-white/[0.03] p-4 space-y-3">
            <div>
              <div className="flex items-center gap-2">
                <p className="text-xs font-black uppercase tracking-[0.18em] text-outline/65">{t('fetcher.mediaReadyTitle')}</p>
                <FieldHelp
                  ariaLabel={t('fetcher.help.readyAria')}
                  title={t('fetcher.help.readyTitle')}
                  body={t('fetcher.help.readyBody')}
                />
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between rounded-2xl border border-white/5 bg-surface-container-high px-3 py-3">
                <div className="flex items-center gap-3">
                  <div className={`h-2.5 w-2.5 rounded-full ${videoReady ? 'bg-tertiary shadow-[0_0_10px_rgba(107,255,193,0.75)]' : 'bg-outline/35'}`} />
                  <span className="text-sm font-semibold text-secondary">{videoActionLabel}</span>
                </div>
                <span className={`text-xs font-bold ${videoReady ? 'text-tertiary' : 'text-outline'}`}>
                  {videoReady ? t('fetcher.ready') : t('fetcher.pending')}
                </span>
              </div>
              <div className="flex items-center justify-between rounded-2xl border border-white/5 bg-surface-container-high px-3 py-3">
                <div className="flex items-center gap-3">
                  <div className={`h-2.5 w-2.5 rounded-full ${audioReady ? 'bg-tertiary shadow-[0_0_10px_rgba(107,255,193,0.75)]' : 'bg-outline/35'}`} />
                  <span className="text-sm font-semibold text-secondary">{t('fetcher.downloadAudio')}</span>
                </div>
                <span className={`text-xs font-bold ${audioReady ? 'text-tertiary' : 'text-outline'}`}>
                  {audioReady ? t('fetcher.ready') : t('fetcher.pending')}
                </span>
              </div>
            </div>
          </div>

          <div className="relative z-10">
            <RunMonitor
              title={t('fetcher.statusTitle')}
              isRunning={isParsing || isDownloading || isUploadingVideo}
              standbyLabel={t('common.standby')}
              statusLabel={downloaderStatusLabel}
              badges={downloaderMonitorBadges}
              compact
              progressItems={[
                ...(onlineModeEnabled ? [{
                  label: t('fetcher.parseProgress'),
                  progress: parseProgress,
                  status: parseReady ? t('status.completed') : isParsing ? `${Math.round(parseProgress)}%` : t('fetcher.pending'),
                  tone: parseReady ? 'success' as const : 'normal' as const,
                }] : []),
                {
                  label: t('fetcher.downloadProgressLabel'),
                  progress: downloadProgress,
                  status: downloadProgressStatusLabel,
                  tone: downloadProgress >= 100 ? 'success' : 'normal',
                },
              ]}
              message={downloaderStatusLabel}
              detailsTitle=""
              showDetails={false}
              onToggleDetails={() => undefined}
              sections={[]}
            />
          </div>

          <button
            onClick={handleProceedToAsr}
            disabled={!canProceedToAsr}
            className={`relative z-10 mt-2 w-full rounded-2xl border py-4 font-bold transition-all flex items-center justify-center gap-3 group disabled:opacity-50 ${
              canProceedToAsr
                ? 'border-tertiary/30 bg-gradient-to-r from-tertiary/85 via-primary/85 to-primary text-white shadow-[0_18px_45px_rgba(95,224,183,0.22)] hover:brightness-110'
                : 'border-white/10 bg-white/5 text-secondary hover:bg-white/10'
            }`}
          >
            {`${t('dashboard.goToStep').replace('{step}', '').trim()} ${t('stt.title')}`}
            <ArrowRight className={`w-5 h-5 transition-transform ${canProceedToAsr ? 'group-hover:translate-x-1' : ''}`} />
          </button>
        </aside>
      </div>
    </div>
  );
}
