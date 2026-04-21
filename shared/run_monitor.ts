export type RunStage =
  | 'prepare'
  | 'load_model'
  | 'preprocess'
  | 'provider'
  | 'repair'
  | 'quality_check'
  | 'postprocess'
  | 'persist'
  | 'complete';

export type RunSeverity = 'info' | 'warning' | 'error';

export type RunIssueArea =
  | 'request'
  | 'provider'
  | 'runtime'
  | 'quality'
  | 'alignment'
  | 'diarization'
  | 'io';

export interface RunIssue {
  code: string;
  severity: RunSeverity;
  area?: RunIssueArea;
  userMessage?: string | null;
  technicalMessage?: string | null;
  data?: Record<string, unknown>;
}

export interface RunProgressEvent {
  status: 'processing' | 'completed' | 'failed';
  code: string;
  stage: RunStage;
  progressHint?: number | null;
  message?: string | null;
  data?: Record<string, unknown>;
}

export interface RunDebugInfo {
  requested?: Record<string, unknown>;
  provider?: Record<string, unknown>;
  applied?: Record<string, unknown>;
  quality?: Record<string, unknown> | null;
  stats?: Record<string, unknown>;
  timing?: Record<string, unknown> | null;
  runtime?: Record<string, unknown> | null;
  diagnostics?: Record<string, unknown> | null;
  warnings?: string[];
  warningIssues?: RunIssue[];
  errors?: Record<string, unknown>;
  errorIssues?: RunIssue[];
  artifacts?: Record<string, unknown> | null;
}
