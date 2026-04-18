export const PROJECT_STATUS = {
  VIDEO_FETCHING: 'video_fetching',
  SPEECH_TO_TEXT: 'speech_to_text',
  TEXT_TRANSLATION: 'text_translation',
  VIDEO_PLAYER: 'video_player',
  COMPLETED: 'completed',
} as const;

export type ProjectStatus = (typeof PROJECT_STATUS)[keyof typeof PROJECT_STATUS];

const LEGACY_STATUS_TO_CODE: Record<string, ProjectStatus> = {
  影片獲取器: PROJECT_STATUS.VIDEO_FETCHING,
  語音轉文字: PROJECT_STATUS.SPEECH_TO_TEXT,
  文字精準翻譯: PROJECT_STATUS.TEXT_TRANSLATION,
  影片播放器: PROJECT_STATUS.VIDEO_PLAYER,
  已完成: PROJECT_STATUS.COMPLETED,
};

export function normalizeProjectStatus(status: unknown): ProjectStatus {
  if (typeof status !== 'string') {
    return PROJECT_STATUS.VIDEO_FETCHING;
  }

  const trimmed = status.trim();
  if (!trimmed) {
    return PROJECT_STATUS.VIDEO_FETCHING;
  }

  if (Object.values(PROJECT_STATUS).includes(trimmed as ProjectStatus)) {
    return trimmed as ProjectStatus;
  }

  return LEGACY_STATUS_TO_CODE[trimmed] || PROJECT_STATUS.VIDEO_FETCHING;
}

export function getProjectStatusTranslationKey(status: unknown): string {
  switch (normalizeProjectStatus(status)) {
    case PROJECT_STATUS.VIDEO_FETCHING:
      return 'nav.videoFetcher';
    case PROJECT_STATUS.SPEECH_TO_TEXT:
      return 'nav.speechToText';
    case PROJECT_STATUS.TEXT_TRANSLATION:
      return 'nav.textTranslation';
    case PROJECT_STATUS.VIDEO_PLAYER:
      return 'nav.videoPlayer';
    case PROJECT_STATUS.COMPLETED:
      return 'status.completed';
    default:
      return 'nav.videoFetcher';
  }
}

export function getProjectStatusColorClass(status: unknown): string {
  switch (normalizeProjectStatus(status)) {
    case PROJECT_STATUS.VIDEO_FETCHING:
      return 'bg-outline';
    case PROJECT_STATUS.SPEECH_TO_TEXT:
    case PROJECT_STATUS.TEXT_TRANSLATION:
      return 'bg-primary';
    case PROJECT_STATUS.VIDEO_PLAYER:
    case PROJECT_STATUS.COMPLETED:
      return 'bg-tertiary';
    default:
      return 'bg-outline';
  }
}

export function getProjectStatusTab(
  status: unknown
): 'downloader' | 'asr' | 'translate' | 'player' {
  switch (normalizeProjectStatus(status)) {
    case PROJECT_STATUS.VIDEO_FETCHING:
      return 'downloader';
    case PROJECT_STATUS.SPEECH_TO_TEXT:
      return 'asr';
    case PROJECT_STATUS.TEXT_TRANSLATION:
      return 'translate';
    case PROJECT_STATUS.VIDEO_PLAYER:
    case PROJECT_STATUS.COMPLETED:
      return 'player';
    default:
      return 'downloader';
  }
}
