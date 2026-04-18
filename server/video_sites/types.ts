export type VideoDownloadType = 'video' | 'audio';

export interface DownloadProgress {
  status: 'downloading' | 'finished';
  progress: number;
  speed?: string;
  eta?: string;
  filename?: string;
}

export interface VideoFormat {
  id: string;
  quality: string;
  size: number;
  ext: string;
  url: string;
  vcodec: string;
  acodec: string;
  header?: Record<string, string>;
}

export interface VideoMetadataResult {
  title: string;
  description: string;
  uploader: string;
  viewCount: number;
  uploadDate: string;
  duration: string;
  thumbnail: string;
  parseWarning?: string;
  siteRuleId?: string;
  debug?: Record<string, unknown>;
  formats: VideoFormat[];
  audioFormats: VideoFormat[];
}

export interface SiteUrlMatchRule {
  hosts?: string[];
  hostSuffixes?: string[];
  urlIncludes?: string[];
  pathnameRegexes?: string[];
  queryParamEquals?: Record<string, string | string[]>;
}

export type ParseFailureTrigger =
  | 'no_formats'
  | 'unsupported_url'
  | 'cloudflare'
  | 'bot_detection';

export interface VideoSiteYtDlpConfig {
  impersonate?: string;
  extractorArgs?: string[];
  extraArgs?: string[];
  metadataExtraArgs?: string[];
  downloadExtraArgs?: string[];
}

export interface VideoSitePlaybackConfig {
  match?: SiteUrlMatchRule;
  headers?: Record<string, string>;
}

export interface VideoSiteParseConfig {
  fallbackOn?: ParseFailureTrigger[];
}

export interface VideoSiteDownloadConfig {
  referer?: string;
  headers?: Record<string, string>;
  customFormatPrefixes?: string[];
  alwaysUseHandler?: boolean;
}

export interface VideoSiteManifest {
  schemaVersion: 1;
  id: string;
  description?: string;
  priority?: number;
  enabled?: boolean;
  match: SiteUrlMatchRule;
  ytDlp?: VideoSiteYtDlpConfig;
  playback?: VideoSitePlaybackConfig;
  parse?: VideoSiteParseConfig;
  download?: VideoSiteDownloadConfig;
  handlers?: {
    parse?: string;
    download?: string;
  };
}

export interface LoadedVideoSiteManifest {
  rootPath: string;
  manifestPath: string;
  manifest: VideoSiteManifest;
}

export interface VideoSiteParseContext {
  url: string;
  rule: LoadedVideoSiteManifest;
  ytDlpResult: {
    code: number;
    stdout: string;
    stderr: string;
  };
  failureTriggers: ParseFailureTrigger[];
}

export interface VideoSiteDownloadContext {
  url: string;
  projectId: string;
  assetsDir: string;
  formatId: string;
  downloadType: VideoDownloadType;
  onProgress: (progress: DownloadProgress) => void;
  rule: LoadedVideoSiteManifest;
}

export interface VideoSiteHandlerModule {
  parseMetadataFallback?: (context: VideoSiteParseContext) => Promise<VideoMetadataResult>;
  download?: (context: VideoSiteDownloadContext) => Promise<string>;
}
