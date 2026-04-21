import type { RunIssue, RunProgressEvent, RunStage } from '../../../shared/run_monitor.js';

const DEFAULT_STAGE_BY_KIND: Record<'asr' | 'translation', RunStage> = {
  asr: 'provider',
  translation: 'provider',
};

function inferStageFromMessage(message: string): RunStage {
  const normalized = String(message || '').trim().toLowerCase();
  if (!normalized) return 'provider';
  if (normalized.includes('loading') || normalized.includes('warm') || normalized.includes('configuration')) {
    return 'load_model';
  }
  if (normalized.includes('voice activity detection') || normalized.includes('diarization')) {
    return 'preprocess';
  }
  if (normalized.includes('repair')) return 'repair';
  if (normalized.includes('quality') || normalized.includes('stricter prompt')) return 'quality_check';
  if (normalized.includes('persist')) return 'persist';
  if (normalized.includes('completed') || normalized.includes('complete')) return 'complete';
  return 'provider';
}

function inferCodeFromMessage(kind: 'asr' | 'translation', message: string): string {
  const normalized = String(message || '').trim().toLowerCase();
  if (!normalized) return `${kind}.progress.unknown`;

  if (normalized.includes('loading translation model configuration')) return 'model.load.configuration';
  if (normalized.includes('loading local asr model')) return 'model.load.asr';
  if (normalized.includes('loading')) return 'model.load.started';
  if (normalized.includes('voice activity detection')) return 'vad.started';
  if (normalized.includes('vad detected')) return 'vad.detected';
  if (normalized.includes('vad windows produced no transcript text')) return 'vad.fallback.full_audio';
  if (normalized.includes('local whisper asr keeps full-audio transcription')) return 'vad.windowing.disabled';
  if (normalized.includes('diarization')) return 'diarization.started';
  if (normalized.includes('repairing subtitle alignment')) return 'repair.alignment.started';
  if (normalized.includes('provider returned no timestamps')) return 'timestamps.synthesized';
  if (normalized.includes('forced alignment unavailable')) return 'alignment.unavailable';
  if (normalized.includes('provider-native forced alignment applied')) return 'alignment.provider_native_applied';
  if (normalized.includes('forced alignment applied')) return 'alignment.applied';
  if (normalized.includes('forced alignment did not improve timing')) return 'alignment.kept_provider_timestamps';
  if (normalized.includes('retrying translation request')) return 'provider.retry.transient';
  if (normalized.includes('retrying translation provider')) return 'quality.retry.started';
  if (normalized.includes('detected untranslated') || normalized.includes('target-language-mismatched')) {
    return 'quality.retry.triggered';
  }
  if (normalized.includes('calling asr provider (segmentation: on)')) return 'provider.call.segmented';
  if (normalized.includes('calling asr provider (segmentation: off)')) return 'provider.call.full_audio';
  if (normalized.includes('calling asr provider (vad window')) return 'provider.call.vad_window';
  if (normalized.includes('calling asr provider (file-limit chunk')) return 'provider.call.file_limit_chunk';
  if (normalized.includes('calling translation provider') && normalized.includes('whole-document mode')) return 'provider.call.whole_document';
  if (normalized.includes('calling translation provider') && normalized.includes('cloud context window')) return 'provider.call.cloud_context';
  if (normalized.includes('calling translation provider')) return 'provider.call.started';
  if (normalized.includes('translating local subtitle batch')) return 'provider.translation.local_batch';
  if (normalized.includes('translating remote subtitle batch with cloud context')) return 'provider.translation.remote_context_batch';
  if (normalized.includes('translating remote subtitle batch')) return 'provider.translation.remote_batch';
  if (normalized.includes('retrying cloud translation as single line')) return 'provider.retry.single_line';
  if (normalized.includes('transcribing')) return 'provider.transcription.started';
  if (normalized.includes('completed')) return 'run.completed';
  return `${kind}.progress.message`;
}

function inferProgressHint(message: string): number | null {
  const matched = String(message || '').match(/\((\d+)\s*\/\s*(\d+)\)/);
  if (!matched) return null;
  const current = Number(matched[1]);
  const total = Number(matched[2]);
  if (!Number.isFinite(current) || !Number.isFinite(total) || total <= 0) return null;
  return Math.max(0, Math.min(100, Math.round((current / total) * 100)));
}

export function buildLegacyProcessingEvent(kind: 'asr' | 'translation', message: string): RunProgressEvent {
  const stage = inferStageFromMessage(message) || DEFAULT_STAGE_BY_KIND[kind];
  return {
    status: 'processing',
    code: inferCodeFromMessage(kind, message),
    stage,
    progressHint: inferProgressHint(message),
    message,
    data: {
      kind,
    },
  };
}

export function buildRunFailureIssue(code: string, technicalMessage: string, area: RunIssue['area'] = 'provider'): RunIssue {
  return {
    code,
    severity: 'error',
    area,
    technicalMessage,
  };
}
