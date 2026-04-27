import React from 'react';
import { ArrowRight, Download, Loader2, Play, CheckCircle2, FolderOpen, FileText, Upload, Square, X } from 'lucide-react';
import { Project, ApiConfig, Material } from '../types';
import { useLanguage } from '../i18n/LanguageContext';
import {
  getCustomTargetTranslationPromptTemplateText,
  getTranslationPromptTemplateText,
  type Language,
  type TranslationPromptTemplateId,
} from '../i18n/translations';
import { sanitizeInput } from '../utils/security';
import { PROJECT_STATUS } from '../project_status';
import { postJson } from '../utils/http_client';
import SubtitleRowsEditor from './SubtitleRowsEditor';
import RunMonitor, { type RunMonitorBadge, type RunMonitorSection } from './RunMonitor';
import {
  EditableSubtitleRow,
  extractLeadingTimeTag,
  hasStrictBracketTimecodes,
  parseTimedSubtitleLine,
  stripLeadingTimeTag,
  subtitleRowsFromText,
  subtitleRowsToLines,
  validateSubtitleRows,
} from '../utils/subtitle_editor';
import type { RunIssue, RunProgressEvent } from '../../shared/run_monitor';

interface TextTranslationProps {
  project: Project | null;
  onUpdateProject: (updates: Partial<Project>) => void | Promise<Project | null>;
  onNext: () => void;
  onBack: () => void;
  onTaskLockChange?: (locked: boolean) => void;
}

type PromptTemplateId = TranslationPromptTemplateId;
type DiarizationMode = 'auto' | 'fixed' | 'range' | 'many';
type DiarizationScenePreset = 'interview' | 'podcast' | 'meeting' | 'presentation_qa' | 'custom';
type DiarizationProvider = 'classic' | 'pyannote';

interface DiarizationDiagnostics {
  provider: 'acoustic' | 'semantic';
  selectedSource: 'speech_region' | 'vad_chunk' | 'chunk' | 'pyannote' | 'semantic';
  speechSegmentCount: number;
  vadWindowCount: number;
  options?: {
    provider: DiarizationProvider;
    mode: DiarizationMode;
    exactSpeakerCount: number | null;
    minSpeakers: number;
    maxSpeakers: number;
    scenePreset: DiarizationScenePreset;
    preferStablePrimarySpeaker: boolean;
    allowShortInterjectionSpeaker: boolean;
    preferVadBoundedRegions: boolean;
    forceMergeTinyClustersInTwoSpeakerMode: boolean;
    semanticFallbackEnabled: boolean;
  };
  selectedPass?: {
    source: string;
    regionCount: number;
    uniqueSpeakerCount: number;
    threshold: number;
    resegmentedChunkCount?: number;
    resegmentationPasses?: number;
  };
}

interface LocalTranslationBatchDebugInfo {
  source?: 'local' | 'translategemma';
  mode?: 'fixed_lines' | 'token_aware';
  batchCount?: number;
  lineCounts?: number[];
  charCounts?: number[];
  promptTokens?: number[];
  estimatedOutputTokens?: number[];
  durationsMs?: number[];
  inputTokenBudget?: number | null;
  outputTokenBudget?: number | null;
  contextWindow?: number | null;
  safetyReserveTokens?: number | null;
  maxLines?: number | null;
  charBudget?: number | null;
  fallbackReason?: string | null;
  totalDurationMs?: number | null;
  maxDurationMs?: number | null;
}

interface CloudTranslationBatchDebugInfo {
  source?: 'nvidia-hosted';
  mode?: 'line_safe' | 'plain_ordered';
  batchCount?: number;
  lineCounts?: number[];
  charCounts?: number[];
  estimatedOutputTokens?: number[];
  durationsMs?: number[];
  maxLines?: number | null;
  minLines?: number | null;
  charBudget?: number | null;
  maxOutputTokens?: number | null;
  timeoutMs?: number | null;
  stream?: boolean;
  splitCount?: number;
  totalDurationMs?: number | null;
  maxDurationMs?: number | null;
}

interface TranslationPipelineDebug {
  requested?: {
    targetLanguageDescriptor?: string;
    promptTemplateId?: string | null;
    effectiveGlossary?: string | null;
    jsonLineRepairEnabled?: boolean;
  };
  provider?: {
    name?: string;
    model?: string;
    profileId?: string | null;
    profileFamily?: string | null;
  };
  applied?: {
    retryCount?: number;
    fallback?: boolean;
    fallbackType?: string | null;
    translationQualityMode?: 'plain_probe' | 'template_validated' | 'json_strict';
    qualityRetryCount?: number;
    strictRetrySucceeded?: boolean;
    cloudStrategy?: 'plain' | 'forced_alignment' | 'context_window' | 'provider_batch';
    cloudContextChunkCount?: number;
    cloudContextFallbackCount?: number;
    cloudBatching?: CloudTranslationBatchDebugInfo | null;
    localModelFamily?: string;
    localModelProfileId?: string | null;
    localPromptStyle?: string;
    localGenerationStyle?: string;
    localBaselineConfidence?: string | null;
    localBaselineTaskFamily?: string | null;
    localFallbackBaseline?: boolean;
    localBatching?: LocalTranslationBatchDebugInfo | null;
    localBatchingMode?: string | null;
    localBatchCount?: number | null;
    localBatchLineCounts?: number[];
    localBatchPromptTokens?: number[];
  };
  quality?: {
    lineCountMatch?: boolean;
    targetLanguageMatch?: boolean;
    passThroughRisk?: 'low' | 'medium' | 'high';
    repetitionRisk?: 'low' | 'medium' | 'high';
    markerPreservation?: 'ok' | 'partial' | 'lost';
    strictRetryTriggered?: boolean;
  } | null;
  runtime?: {
    modelId?: string;
    modelPath?: string;
    pipelineKind?: string;
    vlmRuntimeMode?: string | null;
    requestedDevice?: string;
    pipelineDevice?: string | null;
    schedulerConfig?: {
      max_num_batched_tokens?: number;
    } | null;
    lastPerfMetrics?: {
      inputTokens?: number;
      generatedTokens?: number;
      ttftMs?: number;
      tpotMs?: number;
      throughputTokensPerSec?: number;
      generateDurationMs?: number;
    } | null;
    loadInference?: {
      observedAt?: string;
      source?: string;
      acceleratorModel?: string;
      luid?: string;
      memorySource?: 'dedicated' | 'shared';
      vramUsedGB?: number;
      vramTotalGB?: number;
      utilization?: number;
      physIndex?: number;
    } | null;
    lastInference?: {
      observedAt?: string;
      source?: string;
      acceleratorModel?: string;
      luid?: string;
      memorySource?: 'dedicated' | 'shared';
      vramUsedGB?: number;
      vramTotalGB?: number;
      utilization?: number;
      physIndex?: number;
    } | null;
  } | null;
  timing?: {
    elapsedMs?: number | null;
    providerMs?: number | null;
    qualityRetryMs?: number | null;
  } | null;
  warningIssues?: RunIssue[];
  errorIssues?: RunIssue[];
  artifacts?: {
    hasTimecodes?: boolean;
  } | null;
}

function getDiarizationSummaryCopy(language: Language) {
  const maps = {
    'zh-tw': {
      title: '字幕來源策略',
      loading: '正在讀取語者分離摘要...',
      unavailable: '這份字幕目前沒有可用的語者分離診斷資料。',
      provider: '執行方式',
      source: '實際來源',
      mode: '語者模式',
      scene: '場景預設',
      speakerTarget: '目標語者數',
      outputSpeakers: '輸出語者數',
      speechSegments: '語音段數',
      windows: 'VAD 視窗',
      regions: '聲紋區段',
      threshold: '分群閾值',
      resegmented: '邊界重整',
      providerAcoustic: '聲學分離',
      providerSemantic: '語義分離',
      sourceVad: 'VAD 收緊片段',
      sourceSpeechRegion: '語音區段',
      sourceChunk: '字幕區段',
      sourcePyannote: 'Pyannote turn segments',
      sourceSemantic: '語義回退',
      modeAuto: '自動',
      modeFixed: '固定人數',
      modeRange: '範圍人數',
      modeMany: '多人模式',
      sceneInterview: '訪談 / 對談',
      scenePodcast: 'Podcast',
      sceneMeeting: '會議',
      scenePresentationQa: '演講 + Q&A',
      sceneCustom: '其他 / 自訂',
      speakerExact: '{count} 人',
      speakerRange: '{min}-{max} 人',
      speakerMany: '{min}+ 人',
      resegmentedValue: '{count} 行 / {passes} 輪',
    },
    'zh-cn': {
      title: '字幕来源策略',
      loading: '正在读取说话人分离摘要...',
      unavailable: '这份字幕目前没有可用的说话人分离诊断数据。',
      provider: '执行方式',
      source: '实际来源',
      mode: '说话人模式',
      scene: '场景预设',
      speakerTarget: '目标说话人数',
      outputSpeakers: '输出说话人数',
      speechSegments: '语音段数',
      windows: 'VAD 窗口',
      regions: '声纹区段',
      threshold: '聚类阈值',
      resegmented: '边界重整',
      providerAcoustic: '声学分离',
      providerSemantic: '语义分离',
      sourceVad: 'VAD 收紧片段',
      sourceSpeechRegion: '语音区段',
      sourceChunk: '字幕区段',
      sourcePyannote: 'Pyannote turn segments',
      sourceSemantic: '语义回退',
      modeAuto: '自动',
      modeFixed: '固定人数',
      modeRange: '范围人数',
      modeMany: '多人模式',
      sceneInterview: '访谈 / 对谈',
      scenePodcast: 'Podcast',
      sceneMeeting: '会议',
      scenePresentationQa: '演讲 + Q&A',
      sceneCustom: '其他 / 自定义',
      speakerExact: '{count} 人',
      speakerRange: '{min}-{max} 人',
      speakerMany: '{min}+ 人',
      resegmentedValue: '{count} 行 / {passes} 轮',
    },
    en: {
      title: 'Subtitle Source Strategy',
      loading: 'Loading diarization summary...',
      unavailable: 'No diarization diagnostics are available for this subtitle source.',
      provider: 'Provider',
      source: 'Selected Source',
      mode: 'Speaker Mode',
      scene: 'Scene Preset',
      speakerTarget: 'Target Speakers',
      outputSpeakers: 'Output Speakers',
      speechSegments: 'Speech Segments',
      windows: 'VAD Windows',
      regions: 'Embedding Regions',
      threshold: 'Cluster Threshold',
      resegmented: 'Resegmentation',
      providerAcoustic: 'Acoustic',
      providerSemantic: 'Semantic',
      sourceVad: 'VAD-bounded',
      sourceSpeechRegion: 'Speech regions',
      sourceChunk: 'Chunk-based',
      sourcePyannote: 'Pyannote turn segments',
      sourceSemantic: 'Semantic fallback',
      modeAuto: 'Auto',
      modeFixed: 'Fixed Count',
      modeRange: 'Range',
      modeMany: 'Many Speakers',
      sceneInterview: 'Interview / Dialogue',
      scenePodcast: 'Podcast',
      sceneMeeting: 'Meeting',
      scenePresentationQa: 'Presentation + Q&A',
      sceneCustom: 'Other / Custom',
      speakerExact: '{count} speakers',
      speakerRange: '{min}-{max} speakers',
      speakerMany: '{min}+ speakers',
      resegmentedValue: '{count} lines / {passes} passes',
    },
    jp: {
      title: '字幕ソース戦略',
      loading: '話者分離サマリーを読み込み中...',
      unavailable: 'この字幕ソースには利用可能な話者分離診断がありません。',
      provider: '方式',
      source: '実際の入力',
      mode: '話者モード',
      scene: 'シーンプリセット',
      speakerTarget: '目標話者数',
      outputSpeakers: '出力話者数',
      speechSegments: '音声区間数',
      windows: 'VADウィンドウ',
      regions: '埋め込み区間',
      threshold: 'クラスタ閾値',
      resegmented: '境界再整列',
      providerAcoustic: '音響分離',
      providerSemantic: '意味分離',
      sourceVad: 'VAD収束区間',
      sourceSpeechRegion: '音声区間',
      sourceChunk: '字幕チャンク',
      sourcePyannote: 'Pyannote turn segments',
      sourceSemantic: '意味フォールバック',
      modeAuto: '自動',
      modeFixed: '固定人数',
      modeRange: '人数レンジ',
      modeMany: '多人数モード',
      sceneInterview: '対談 / インタビュー',
      scenePodcast: 'Podcast',
      sceneMeeting: '会議',
      scenePresentationQa: '講演 + Q&A',
      sceneCustom: 'その他 / カスタム',
      speakerExact: '{count}人',
      speakerRange: '{min}-{max}人',
      speakerMany: '{min}+人',
      resegmentedValue: '{count}行 / {passes}回',
    },
    de: {
      title: 'Strategie der Untertitelquelle',
      loading: 'Diarisierungszusammenfassung wird geladen...',
      unavailable: 'Für diese Untertitelquelle sind keine Diarisierungsdiagnosen verfügbar.',
      provider: 'Verfahren',
      source: 'Gewählte Quelle',
      mode: 'Sprechermodus',
      scene: 'Szenenvorgabe',
      speakerTarget: 'Ziel-Sprecherzahl',
      outputSpeakers: 'Ausgegebene Sprecher',
      speechSegments: 'Sprachsegmente',
      windows: 'VAD-Fenster',
      regions: 'Embedding-Segmente',
      threshold: 'Cluster-Schwelle',
      resegmented: 'Neusegmentierung',
      providerAcoustic: 'Akustisch',
      providerSemantic: 'Semantisch',
      sourceVad: 'VAD-begrenzt',
      sourceSpeechRegion: 'Sprachsegmente',
      sourceChunk: 'Chunk-basiert',
      sourcePyannote: 'Pyannote turn segments',
      sourceSemantic: 'Semantischer Fallback',
      modeAuto: 'Automatisch',
      modeFixed: 'Feste Anzahl',
      modeRange: 'Bereich',
      modeMany: 'Viele Sprecher',
      sceneInterview: 'Interview / Dialog',
      scenePodcast: 'Podcast',
      sceneMeeting: 'Meeting',
      scenePresentationQa: 'Vortrag + Q&A',
      sceneCustom: 'Andere / Benutzerdefiniert',
      speakerExact: '{count} Sprecher',
      speakerRange: '{min}-{max} Sprecher',
      speakerMany: '{min}+ Sprecher',
      resegmentedValue: '{count} Zeilen / {passes} Durchläufe',
    },
  } as const;

  return maps[language];
}

const TEXT_SOURCE_EXTENSIONS = new Set(['.txt', '.srt', '.vtt', '.ass', '.ssa', '.json']);

function getFileExt(fileName: string) {
  const idx = fileName.lastIndexOf('.');
  if (idx < 0) return '';
  return fileName.slice(idx).toLowerCase();
}

function isTextAsset(asset: Material) {
  return TEXT_SOURCE_EXTENSIONS.has(getFileExt(String(asset.name || '')));
}

function isAllowedTextFileName(fileName: string) {
  return TEXT_SOURCE_EXTENSIONS.has(getFileExt(fileName));
}

function normalizeProjectTextSource(fileName: string, rawContent: string) {
  const clean = String(rawContent || '').replace(/^﻿/, '').trim();
  if (!clean) return '';
  if (getFileExt(fileName) !== '.json') return clean;

  try {
    const parsed = JSON.parse(clean);
    if (typeof parsed === 'string') return parsed.trim();
    if (typeof parsed?.originalSubtitles === 'string') return parsed.originalSubtitles.trim();
    if (typeof parsed?.translatedSubtitles === 'string') return parsed.translatedSubtitles.trim();
    if (typeof parsed?.text === 'string') return parsed.text.trim();
    if (Array.isArray(parsed?.segments)) {
      const merged = parsed.segments
        .map((seg: any) => String(seg?.text || '').trim())
        .filter(Boolean)
        .join('\n');
      if (merged) return merged;
    }
    if (Array.isArray(parsed?.chunks)) {
      const merged = parsed.chunks
        .map((chunk: any) => String(chunk?.text || '').trim())
        .filter(Boolean)
        .join('\n');
      if (merged) return merged;
    }
  } catch {
    // Fallback to raw text.
  }

  return clean;
}

function rebuildTranslationWithSourceTimecodes(sourceText: string, translatedText: string) {
  const sourceLines = String(sourceText || '').split('\n');
  const translatedLines = String(translatedText || '').split('\n');
  if (sourceLines.length === 0) return String(translatedText || '').trim();

  const allSourceTimed = sourceLines.every((line) => Boolean(extractLeadingTimeTag(line.trim())));
  if (!allSourceTimed) return String(translatedText || '').trim();

  const rebuilt = sourceLines.map((sourceLine, index) => {
    const parsed = extractLeadingTimeTag(sourceLine.trim());
    if (!parsed) return stripLeadingTimeTag(translatedLines[index] || '');
    const translatedPayload = stripLeadingTimeTag(translatedLines[index] || '');
    if (!translatedPayload) return `${parsed.tag} ${parsed.text}`.trim();
    return `${parsed.tag} ${translatedPayload}`.trim();
  });

  return rebuilt.join('\n').trim();
}

function rebuildSourceWithTranslatedTimecodes(sourceText: string, translatedRows: EditableSubtitleRow[]) {
  const sourceRows = subtitleRowsFromText(sourceText);
  if (sourceRows.length === 0) return String(sourceText || '').trim();

  const rebuiltRows = sourceRows.map((sourceRow, index) => {
    const sourceTextPayload = String(sourceRow.text || '').trim();
    if (!sourceTextPayload) return null;
    const translatedTimecode = String(translatedRows[index]?.timecode || '').trim();
    return {
      timecode: translatedTimecode,
      text: sourceTextPayload,
    };
  });

  return subtitleRowsToLines(rebuiltRows.filter((row): row is { timecode: string; text: string } => Boolean(row))).join('\n');
}

function hasSameSubtitleRowOrder(beforeIds: string[], afterRows: EditableSubtitleRow[]) {
  if (beforeIds.length !== afterRows.length) return false;
  for (let index = 0; index < beforeIds.length; index += 1) {
    if (String(beforeIds[index] || '') !== String(afterRows[index]?.id || '')) return false;
  }
  return true;
}

function normalizeTranscriptionSourceLanguage(language?: string | null) {
  const normalized = String(language || '').trim();
  if (!normalized) return '';
  if (normalized.toLowerCase() === 'auto') return '';
  return normalized;
}

function isTranslateGemmaModel(model?: ApiConfig | null) {
  const haystack = [String(model?.id || ''), String(model?.name || ''), String(model?.model || '')]
    .join(' ')
    .toLowerCase();
  return haystack.includes('translategemma');
}

export default function TextTranslation({ project, onUpdateProject, onNext, onTaskLockChange }: TextTranslationProps) {
  const { t, language } = useLanguage();
  const diarizationCopy = React.useMemo(() => getDiarizationSummaryCopy(language), [language]);
  const TARGET_LANG_OPTIONS = [
    { value: 'zh-TW', label: t('lang.zh-tw') },
    { value: 'zh-CN', label: t('lang.zh-cn') },
    { value: 'zh-HK', label: t('lang.zh-hk') },
    { value: 'yue', label: t('lang.yue') },
    { value: 'en', label: t('lang.en') },
    { value: 'fi', label: t('lang.fi') },
    { value: 'es', label: t('lang.es') },
    { value: 'de', label: t('lang.de') },
    { value: 'pt', label: t('lang.pt') },
    { value: 'it', label: t('lang.it') },
    { value: 'fr', label: t('lang.fr') },
    { value: 'ru', label: t('lang.ru') },
    { value: 'pl', label: t('lang.pl') },
    { value: 'ar', label: t('lang.ar') },
    { value: 'nl', label: t('lang.nl') },
    { value: 'el', label: t('lang.el') },
    { value: 'fa', label: t('lang.fa') },
    { value: 'hu', label: t('lang.hu') },
    { value: 'ja', label: t('lang.jp') },
    { value: 'ko', label: t('lang.kr') },
    { value: 'other', label: t('lang.other') },
  ];

  const [isTranslating, setIsTranslating] = React.useState(false);
  const [progress, setProgress] = React.useState(0);
  const [targetLang, setTargetLang] = React.useState('zh-TW');
  const [customTargetLanguage, setCustomTargetLanguage] = React.useState('');
  const [promptText, setPromptText] = React.useState('');
  const [glossaryText, setGlossaryText] = React.useState('');
  const [strictJsonLineRepairEnabled, setStrictJsonLineRepairEnabled] = React.useState(false);
  const [selectedPromptTemplateId, setSelectedPromptTemplateId] = React.useState<PromptTemplateId>('');
  const [translatedLines, setTranslatedLines] = React.useState<Array<{ original: string; translated: string }>>([]);
  const [translateModels, setTranslateModels] = React.useState<ApiConfig[]>([]);
  const [selectedModelId, setSelectedModelId] = React.useState('');
  const [error, setError] = React.useState<string | null>(null);
  const [translateMsg, setTranslateMsg] = React.useState<string | null>(null);
  const [pipelineMode, setPipelineMode] = React.useState<string | null>(null);
  const [translationDebug, setTranslationDebug] = React.useState<TranslationPipelineDebug | null>(null);
  const [pipelineWarnings, setPipelineWarnings] = React.useState<string[]>([]);
  const [pipelineErrors, setPipelineErrors] = React.useState<string[]>([]);
  const [hasTimecodes, setHasTimecodes] = React.useState(false);
  const [modelLoadStatus, setModelLoadStatus] = React.useState<'idle' | 'loading' | 'ok' | 'failed'>('idle');
  const [modelLoadError, setModelLoadError] = React.useState<string | null>(null);
  const [sourceType, setSourceType] = React.useState<'transcription' | 'project'>('transcription');
  const [showSourceModal, setShowSourceModal] = React.useState(false);
  const [sourceAssets, setSourceAssets] = React.useState<Material[]>([]);
  const [isLoadingSourceAssets, setIsLoadingSourceAssets] = React.useState(false);
  const [selectedSourceAssetName, setSelectedSourceAssetName] = React.useState<string | null>(null);
  const [selectedProjectSourceText, setSelectedProjectSourceText] = React.useState('');
  const [isLoadingSourceText, setIsLoadingSourceText] = React.useState(false);
  const [isUploadingSource, setIsUploadingSource] = React.useState(false);
  const [isPersistingResult, setIsPersistingResult] = React.useState(false);
  const [sourceDiarizationDiagnostics, setSourceDiarizationDiagnostics] = React.useState<DiarizationDiagnostics | null>(null);
  const [isLoadingSourceDiarization, setIsLoadingSourceDiarization] = React.useState(false);
  const [showDebugDetails, setShowDebugDetails] = React.useState(false);
  const [activePreviewLine, setActivePreviewLine] = React.useState<number | null>(null);
  const [isEditingTranslatedOutput, setIsEditingTranslatedOutput] = React.useState(false);
  const [editingTranslatedOutputRows, setEditingTranslatedOutputRows] = React.useState<EditableSubtitleRow[]>([]);
  const [editingTranslatedOutputBaselineIds, setEditingTranslatedOutputBaselineIds] = React.useState<string[]>([]);
  const [isSavingTranslatedOutput, setIsSavingTranslatedOutput] = React.useState(false);
  const sourceFileInputRef = React.useRef<HTMLInputElement | null>(null);
  const translateEventSourceRef = React.useRef<EventSource | null>(null);
  const preloadSeqRef = React.useRef(0);
  const previewAudioRef = React.useRef<HTMLAudioElement | null>(null);
  const translationScrollRef = React.useRef<HTMLDivElement | null>(null);
  const translationLineRefs = React.useRef<Record<number, HTMLDivElement | null>>({});
  const effectiveTargetLanguage = React.useMemo(
    () => (targetLang === 'other' ? String(customTargetLanguage || '').trim() : targetLang),
    [targetLang, customTargetLanguage]
  );
  const selectedTranslateModel = React.useMemo(
    () => translateModels.find((model) => model.id === selectedModelId) || null,
    [translateModels, selectedModelId]
  );
  const translateGemmaPromptControlsDisabled = isTranslateGemmaModel(selectedTranslateModel);
  const transcriptionSourceLanguage = React.useMemo(
    () => normalizeTranscriptionSourceLanguage(project?.transcriptionSourceLanguage),
    [project?.transcriptionSourceLanguage]
  );
  const effectiveSourceLanguage = React.useMemo(
    () => (sourceType === 'transcription' ? transcriptionSourceLanguage : ''),
    [sourceType, transcriptionSourceLanguage]
  );

  React.useEffect(() => {
    setSourceType('transcription');
    setShowSourceModal(false);
    setSourceAssets([]);
    setSelectedSourceAssetName(null);
    setSelectedProjectSourceText('');
    setHasTimecodes(false);
    setIsPersistingResult(false);
    setSourceDiarizationDiagnostics(null);
    setIsLoadingSourceDiarization(false);
    setActivePreviewLine(null);
    setIsEditingTranslatedOutput(false);
    setEditingTranslatedOutputRows([]);
    setEditingTranslatedOutputBaselineIds([]);
    setIsSavingTranslatedOutput(false);
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
    }
  }, [project?.id]);

  React.useEffect(() => {
    if (!project?.id) {
      setSourceDiarizationDiagnostics(null);
      setIsLoadingSourceDiarization(false);
      return;
    }

    let cancelled = false;
    setIsLoadingSourceDiarization(true);
    fetch(`/Projects/${project.id}/assets/transcription.json`, { cache: 'no-store' })
      .then((res) => {
        if (!res.ok) return null;
        return res.json();
      })
      .then((data) => {
        if (cancelled) return;
        setSourceDiarizationDiagnostics((data as any)?.debug?.diarization || null);
      })
      .catch(() => {
        if (cancelled) return;
        setSourceDiarizationDiagnostics(null);
      })
      .finally(() => {
        if (cancelled) return;
        setIsLoadingSourceDiarization(false);
      });

    return () => {
      cancelled = true;
    };
  }, [project?.id]);

  React.useEffect(() => {
    fetch('/api/runtime-models')
      .then((res) => res.json())
      .then((data) => {
        const models = (data.translateModels || []) as ApiConfig[];
        setTranslateModels(models);
        if (models.length > 0) {
          setSelectedModelId(models[0].id);
          setModelLoadStatus('idle');
          setModelLoadError(null);
        } else {
          setSelectedModelId('');
          setModelLoadStatus('failed');
          setModelLoadError(t('stt.noModels'));
        }
      })
      .catch((err) => {
        console.error('Failed to load translation models', err);
        setModelLoadStatus('failed');
        setModelLoadError(t('settings.testFailed'));
      });
  }, [t]);

  const releaseLocalRuntime = React.useCallback(async () => {
    try {
      await postJson('/api/local-models/release', { target: 'translate' }, { timeoutMs: 12000 });
    } catch {
      // Keep navigation resilient even if runtime release fails.
    }
  }, []);

  React.useEffect(() => {
    return () => {
      translateEventSourceRef.current?.close();
      translateEventSourceRef.current = null;
      onTaskLockChange?.(false);
      void releaseLocalRuntime();
    };
  }, [releaseLocalRuntime, onTaskLockChange]);

  React.useEffect(() => {
    const currentModel = selectedTranslateModel;
    const isLocalModel = Boolean(currentModel?.id && currentModel?.isLocal && currentModel?.provider === 'local-openvino');
    const loadSeq = ++preloadSeqRef.current;
    let cancelled = false;

    if (isTranslating) {
      return () => {
        cancelled = true;
      };
    }

    if (!isLocalModel) {
      setModelLoadStatus('idle');
      setModelLoadError(null);
      void releaseLocalRuntime();
      return () => {
        cancelled = true;
      };
    }

    setModelLoadStatus('loading');
    setModelLoadError(null);

    void postJson(
      '/api/local-models/preload',
      {
        target: 'translate',
        modelId: currentModel!.id,
      },
      { timeoutMs: 240000 }
    )
      .then(() => {
        if (cancelled || preloadSeqRef.current !== loadSeq) return;
        setModelLoadStatus('ok');
        setModelLoadError(null);
      })
      .catch((error) => {
        if (cancelled || preloadSeqRef.current !== loadSeq) return;
        setModelLoadStatus('failed');
        setModelLoadError(String((error as any)?.message || error || t('translation.errorFailed')));
      });

    return () => {
      cancelled = true;
    };
  }, [selectedTranslateModel, isTranslating, releaseLocalRuntime, t]);

  const handleNextStep = async () => {
    await releaseLocalRuntime();
    onNext();
  };

  const activeSourceText = React.useMemo(() => {
    if (sourceType === 'project') return String(selectedProjectSourceText || '').trim();
    return String(project?.originalSubtitles || '').trim();
  }, [sourceType, selectedProjectSourceText, project?.originalSubtitles]);

  const sourceLines = React.useMemo(() => activeSourceText.split('\n'), [activeSourceText]);
  const parsedTranslatedLines = React.useMemo(
    () =>
      translatedLines.map((line, index) => ({
        index,
        ...line,
        ...parseTimedSubtitleLine(line.translated),
      })),
    [translatedLines]
  );
  const translatedOutputEditIssues = React.useMemo(
    () => validateSubtitleRows(editingTranslatedOutputRows),
    [editingTranslatedOutputRows]
  );
  const canSaveEditedTranslatedOutput =
    editingTranslatedOutputRows.length > 0 &&
    translatedOutputEditIssues.length === 0;
  const subtitleEditorCopy = React.useMemo(
    () => ({
      timecodeLabel: t('subtitleEditor.timecode'),
      textLabel: t('subtitleEditor.text'),
      addRow: t('subtitleEditor.addRow'),
      deleteRow: t('subtitleEditor.deleteRow'),
      timecodePlaceholder: t('subtitleEditor.timecodePlaceholder'),
      textPlaceholder: t('subtitleEditor.textPlaceholder'),
      emptyHint: t('subtitleEditor.emptyHint'),
      timecodeFormatError: t('subtitleEditor.timecodeFormatError'),
      timecodeBeforePreviousError: t('subtitleEditor.timecodeBeforePreviousError'),
      timecodeAfterNextError: t('subtitleEditor.timecodeAfterNextError'),
      textRequiredError: t('subtitleEditor.textRequiredError'),
    }),
    [t]
  );
  const hasSourceText = activeSourceText.length > 0;
  const hasProjectSourceSelection = sourceType !== 'project' || Boolean(selectedSourceAssetName);
  const canStartTranslation =
    !isTranslating &&
    Boolean(selectedModelId) &&
    hasSourceText &&
    hasProjectSourceSelection &&
    Boolean(String(effectiveTargetLanguage || '').trim());
  const promptTemplateOptions = React.useMemo(
    () => [
      { id: '' as PromptTemplateId, label: t('translation.promptTemplateNone') },
      { id: 'subtitle_general' as PromptTemplateId, label: t('translation.promptTemplateGeneral') },
      { id: 'subtitle_concise_spoken' as PromptTemplateId, label: t('translation.promptTemplateConciseSpoken') },
      { id: 'subtitle_formal_precise' as PromptTemplateId, label: t('translation.promptTemplateFormalPrecise') },
      { id: 'subtitle_asr_recovery' as PromptTemplateId, label: t('translation.promptTemplateAsrRecovery') },
      { id: 'subtitle_technical_terms' as PromptTemplateId, label: t('translation.promptTemplateTechnicalTerms') },
    ],
    [t]
  );

  React.useEffect(() => {
    if (!selectedPromptTemplateId) return;
    setPromptText(
      targetLang === 'other'
        ? getCustomTargetTranslationPromptTemplateText(selectedPromptTemplateId, effectiveTargetLanguage)
        : getTranslationPromptTemplateText(selectedPromptTemplateId, effectiveTargetLanguage)
    );
  }, [selectedPromptTemplateId, targetLang, effectiveTargetLanguage]);

  const selectedSourceDescription = React.useMemo(() => {
    if (sourceType === 'project') {
      return selectedSourceAssetName || t('translation.notSelected');
    }
    if (!String(project?.originalSubtitles || '').trim()) {
      return t('translation.noTranscription');
    }
    const named = String(project?.videoTitle || project?.name || '').trim();
    if (named) return t('translation.transcriptionOf').replace('{name}', named);
    return t('translation.transcriptionResult');
  }, [sourceType, selectedSourceAssetName, project?.originalSubtitles, project?.videoTitle, project?.name, t]);
  const previewAudioUrl = React.useMemo(() => {
    const direct = String(project?.audioUrl || '').trim();
    if (direct) return direct;
    const fallback = String(project?.videoUrl || '').trim();
    return fallback;
  }, [project?.audioUrl, project?.videoUrl]);

  React.useEffect(() => {
    setActivePreviewLine(null);
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
    }
  }, [previewAudioUrl]);

  React.useEffect(() => {
    if (isTranslating || isEditingTranslatedOutput || isSavingTranslatedOutput) return;

    const storedTranslated = String(project?.translatedSubtitles || '')
      .replace(/\r\n/g, '\n')
      .replace(/\r/g, '\n')
      .trim();
    if (!storedTranslated) return;

    const restoredTranslatedLines = storedTranslated
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean);
    const storedSource = String(project?.originalSubtitles || '')
      .replace(/\r\n/g, '\n')
      .replace(/\r/g, '\n')
      .trim();
    const restoredSourceLines = storedSource
      ? storedSource.split('\n').map((line) => line.trim())
      : [];

    const maxLineCount = Math.max(restoredSourceLines.length, restoredTranslatedLines.length);
    const restored = Array.from({ length: maxLineCount }, (_, index) => ({
      original: restoredSourceLines[index] || '',
      translated: restoredTranslatedLines[index] || '',
    })).filter((line) => String(line.translated || '').trim().length > 0);

    const currentSnapshot = translatedLines
      .map((line) => `${line.original}\u0001${line.translated}`)
      .join('\n');
    const restoredSnapshot = restored
      .map((line) => `${line.original}\u0001${line.translated}`)
      .join('\n');
    if (restoredSnapshot && restoredSnapshot !== currentSnapshot) {
      setTranslatedLines(restored);
    }
    setHasTimecodes(hasStrictBracketTimecodes(storedTranslated));
    if (progress < 100) {
      setProgress(100);
    }
  }, [
    isEditingTranslatedOutput,
    isSavingTranslatedOutput,
    isTranslating,
    progress,
    project?.originalSubtitles,
    project?.translatedSubtitles,
    translatedLines,
  ]);

  const loadProjectTextAssets = async () => {
    if (!project?.id) return;
    setIsLoadingSourceAssets(true);
    try {
      const response = await fetch(`/api/projects/${project.id}/materials`);
      const data = await response.json();
      const textAssets = (Array.isArray(data) ? data : []).filter((item) => isTextAsset(item as Material)) as Material[];
      setSourceAssets(textAssets);

      if (selectedSourceAssetName && !textAssets.some((asset) => asset.name === selectedSourceAssetName)) {
        setSelectedSourceAssetName(null);
      }
    } catch (err) {
      console.error('Failed to load project text assets', err);
      setSourceAssets([]);
    } finally {
      setIsLoadingSourceAssets(false);
    }
  };

  const fetchAssetTextContent = async (asset: Material) => {
    if (!project?.id) throw new Error(t('fetcher.errorNoProject'));

    const encodedName = encodeURIComponent(asset.name);
    const primarySubDir = asset.category === 'subtitle' ? 'subtitles' : 'assets';
    const fallbackSubDir = primarySubDir === 'subtitles' ? 'assets' : 'subtitles';

    const requestBySubDir = async (subDir: 'assets' | 'subtitles') => {
      const response = await fetch(`/Projects/${project.id}/${subDir}/${encodedName}`, { cache: 'no-store' });
      if (!response.ok) return null;
      const raw = await response.text();
      return normalizeProjectTextSource(asset.name, raw);
    };

    const primary = await requestBySubDir(primarySubDir);
    if (primary !== null) return primary;

    const fallback = await requestBySubDir(fallbackSubDir);
    if (fallback !== null) return fallback;

    throw new Error(t('translation.errorFailed'));
  };

  const handleOpenProjectSourceModal = () => {
    setShowSourceModal(true);
    setError(null);
    void loadProjectTextAssets();
  };

  const handleConfirmProjectSource = async () => {
    if (!selectedSourceAssetName) return;
    const selectedAsset = sourceAssets.find((asset) => asset.name === selectedSourceAssetName);
    if (!selectedAsset) return;

    setIsLoadingSourceText(true);
    setError(null);
    try {
      const text = await fetchAssetTextContent(selectedAsset);
      if (!String(text || '').trim()) {
        throw new Error(t('translation.noTranscription'));
      }
      setSelectedProjectSourceText(text);
      setSourceType('project');
      setShowSourceModal(false);
    } catch (err: any) {
      setError(String(err?.message || t('translation.errorFailed')));
    } finally {
      setIsLoadingSourceText(false);
    }
  };

  const handleSourceFileUpload = async (file?: File | null) => {
    if (!project?.id || !file) return;

    if (!isAllowedTextFileName(file.name)) {
      setError(t('translation.onlyTextFiles'));
      return;
    }

    setIsUploadingSource(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`/api/projects/${project.id}/materials/upload-text`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok || data?.error) {
        throw new Error(String(data?.error || t('translation.errorFailed')));
      }

      await loadProjectTextAssets();
      const uploadedName = String(data?.file?.filename || file.name || '').trim();
      if (uploadedName) {
        setSelectedSourceAssetName(uploadedName);
      }
    } catch (err: any) {
      setError(String(err?.message || t('translation.errorFailed')));
    } finally {
      setIsUploadingSource(false);
      if (sourceFileInputRef.current) {
        sourceFileInputRef.current.value = '';
      }
    }
  };

  const handleSourceFileDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    await handleSourceFileUpload(file);
  };

  const localizeTranslationMessage = (message: string) => {
    if (!message) return '';

    if (message === 'Loading translation model configuration...') {
      return t('translation.msgLoadingConfig');
    }
    if (message === 'Translation completed.') {
      return t('status.completed');
    }
    if (message === 'Chat completions unavailable, retrying with Responses API fallback...') {
      return t('translation.msgFallbackResponses');
    }
    if (message === 'Ollama /api/chat unavailable, retrying with /api/generate fallback...') {
      return t('translation.msgFallbackOllamaGenerate');
    }
    if (message === 'Ollama /api/generate unavailable, retrying with /api/chat fallback...') {
      return t('translation.msgFallbackOllamaChat');
    }

    const providerCallMatch = message.match(/^Calling translation provider \(([^)]+)\)\.\.\.$/);
    if (providerCallMatch) {
      return t('translation.msgCallingProvider').replace('{provider}', providerLabel(providerCallMatch[1]));
    }

    const retryMatch = message.match(/^Retrying translation request \((\d+)\/(\d+)\) after transient error \(([^)]+)\)\.\.\.$/);
    if (retryMatch) {
      return t('translation.msgRetrying')
        .replace('{current}', retryMatch[1])
        .replace('{max}', retryMatch[2])
        .replace('{status}', retryMatch[3]);
    }

    const jsonRepairMatch = message.match(/^Repairing subtitle alignment with strict JSON mapping \((\d+)\/(\d+)\)\.\.\.$/);
    if (jsonRepairMatch) {
      return t('translation.msgJsonRepairing')
        .replace('{current}', jsonRepairMatch[1])
        .replace('{max}', jsonRepairMatch[2]);
    }

    if (message === 'Detected untranslated, repetitive, or target-language-mismatched output, retrying with stricter prompt...') {
      return t('translation.msgStrictRetry');
    }

    const strictProviderRetryMatch = message.match(/^Retrying translation provider \(([^)]+)\) with stricter prompt\.\.\.$/);
    if (strictProviderRetryMatch) {
      return t('translation.msgRetryingProviderStrict').replace('{provider}', providerLabel(strictProviderRetryMatch[1]));
    }

    const wholeDocumentProviderMatch = message.match(/^Calling translation provider \(([^)]+)\) with whole-document mode\.\.\.$/);
    if (wholeDocumentProviderMatch) {
      return t('translation.msgCallingProviderWholeDocument').replace('{provider}', providerLabel(wholeDocumentProviderMatch[1]));
    }

    const cloudContextProviderMatch = message.match(/^Calling translation provider \(([^)]+)\) with cloud context window\.\.\.$/);
    if (cloudContextProviderMatch) {
      return t('translation.msgCallingProviderCloudContext').replace('{provider}', providerLabel(cloudContextProviderMatch[1]));
    }

    const hostedBatchProviderMatch = message.match(/^Calling translation provider \(([^)]+)\) with hosted batch\.\.\.$/);
    if (hostedBatchProviderMatch) {
      return t('translation.msgCallingProviderHostedBatch').replace('{provider}', providerLabel(hostedBatchProviderMatch[1]));
    }

    const localBatchMatch = message.match(/^Translating local subtitle batch \((\d+)\/(\d+)\)\.\.\.$/);
    if (localBatchMatch) {
      return t('translation.msgTranslatingLocalBatch')
        .replace('{current}', localBatchMatch[1])
        .replace('{max}', localBatchMatch[2]);
    }

    const remoteBatchMatch = message.match(/^Translating remote subtitle batch \((\d+)\/(\d+)\)\.\.\.$/);
    if (remoteBatchMatch) {
      return t('translation.msgTranslatingRemoteBatch')
        .replace('{current}', remoteBatchMatch[1])
        .replace('{max}', remoteBatchMatch[2]);
    }

    const remoteContextBatchMatch = message.match(/^Translating remote subtitle batch with cloud context \((\d+)\/(\d+)\)\.\.\.$/);
    if (remoteContextBatchMatch) {
      return t('translation.msgTranslatingRemoteCloudContextBatch')
        .replace('{current}', remoteContextBatchMatch[1])
        .replace('{max}', remoteContextBatchMatch[2]);
    }

    const hostedBatchMatch = message.match(/^Translating hosted cloud subtitle batch \((\d+)\/(\d+)\)\.\.\.$/);
    if (hostedBatchMatch) {
      return t('translation.msgTranslatingHostedCloudBatch')
        .replace('{current}', hostedBatchMatch[1])
        .replace('{max}', hostedBatchMatch[2]);
    }

    const cloudSingleLineMatch = message.match(/^Retrying cloud translation as single line \((\d+)\)\.\.\.$/);
    if (cloudSingleLineMatch) {
      return t('translation.msgRetryingCloudSingleLine').replace('{line}', cloudSingleLineMatch[1]);
    }

    if (message === 'Whole-document cloud translation looked unreliable, retrying with subtitle chunking...') {
      return t('translation.msgWholeDocumentUnreliable');
    }

    if (message === 'Whole-document cloud translation was rejected, retrying with subtitle chunking...') {
      return t('translation.msgWholeDocumentRejected');
    }

    return message;
  };

  const localizeTranslationEvent = React.useCallback((event?: RunProgressEvent | null, fallbackMessage = '') => {
    if (!event) return localizeTranslationMessage(fallbackMessage);

    switch (event.code) {
      case 'model.load.configuration':
        return t('translation.msgLoadingConfig');
      case 'provider.retry.transient':
      case 'repair.alignment.started':
      case 'provider.call.whole_document':
      case 'provider.call.cloud_context':
      case 'provider.call.started':
      case 'provider.translation.local_batch':
      case 'provider.translation.remote_batch':
      case 'provider.translation.remote_context_batch':
      case 'provider.retry.single_line':
      case 'quality.retry.started':
        return localizeTranslationMessage(String(event.message || fallbackMessage || ''));
      case 'quality.retry.triggered':
        return t('translation.msgStrictRetry');
      case 'run.completed':
        return t('status.completed');
      default:
        return localizeTranslationMessage(String(event.message || fallbackMessage || ''));
    }
  }, [t]);

  const providerLabel = (provider: string) => {
    const key = String(provider || '').toLowerCase();
    if (key === 'openai-compatible') return 'OpenAI-compatible';
    if (key === 'gemini-native') return 'Gemini';
    if (key === 'anthropic') return 'Claude/Anthropic';
    if (key === 'ollama-chat' || key === 'ollama-generate') return 'Ollama';
    if (key === 'deepl') return 'DeepL';
    return provider;
  };

  const localizeCloudStrategy = React.useCallback((strategy: string | null | undefined) => {
    switch (String(strategy || '')) {
      case 'plain':
        return t('translation.strategyPlain');
      case 'forced_alignment':
        return t('translation.strategyForcedAlignment');
      case 'context_window':
        return t('translation.strategyContextWindow');
      case 'provider_batch':
        return t('translation.strategyProviderBatch');
      default:
        return '-';
    }
  }, [t]);

  const localizeTranslationQualityMode = React.useCallback((mode: string | null | undefined) => {
    switch (String(mode || '')) {
      case 'plain_probe':
        return t('translation.qualityModePlainProbe');
      case 'template_validated':
        return t('translation.qualityModeTemplateValidated');
      case 'json_strict':
        return t('translation.qualityModeJsonStrict');
      default:
        return '-';
    }
  }, [t]);

  const localizeQualityRisk = React.useCallback((risk: string | null | undefined) => {
    switch (String(risk || '')) {
      case 'low':
        return t('translation.riskLow');
      case 'medium':
        return t('translation.riskMedium');
      case 'high':
        return t('translation.riskHigh');
      default:
        return '-';
    }
  }, [t]);

  const localizeMarkerPreservation = React.useCallback((value: string | null | undefined) => {
    switch (String(value || '')) {
      case 'ok':
        return t('translation.markerOk');
      case 'partial':
        return t('translation.markerPartial');
      case 'lost':
        return t('translation.markerLost');
      default:
        return '-';
    }
  }, [t]);

  const localizeLocalModelFamily = React.useCallback((family: string | null | undefined) => {
    switch (String(family || '')) {
      case 'qwen2_5':
        return t('translation.localFamilyQwen25');
      case 'qwen3':
        return t('translation.localFamilyQwen3');
      case 'deepseek_r1_distill_qwen':
        return t('translation.localFamilyDeepSeekR1DistillQwen');
      case 'phi4':
        return t('translation.localFamilyPhi4');
      case 'gemma3':
        return t('translation.localFamilyGemma3');
      case 'generic':
        return t('translation.localFamilyGeneric');
      default:
        return family ? String(family) : '-';
    }
  }, [t]);

  const localizeLocalPromptStyle = React.useCallback((promptStyle: string | null | undefined) => {
    switch (String(promptStyle || '')) {
      case 'qwen_chatml':
        return t('translation.localPromptStyleQwenChatml');
      case 'qwen3_non_thinking':
        return t('translation.localPromptStyleQwen3NonThinking');
      case 'deepseek_r1_distill_qwen':
        return t('translation.localPromptStyleDeepSeekR1');
      case 'phi4_chat':
        return t('translation.localPromptStylePhi4Chat');
      case 'gemma_plain':
        return t('translation.localPromptStyleGemmaPlain');
      case 'generic':
        return t('translation.localPromptStyleGeneric');
      default:
        return promptStyle ? String(promptStyle) : '-';
    }
  }, [t]);

  const localizeLocalGenerationStyle = React.useCallback((generationStyle: string | null | undefined) => {
    switch (String(generationStyle || '')) {
      case 'qwen':
        return t('translation.localGenerationStyleQwen');
      case 'qwen3':
        return t('translation.localGenerationStyleQwen3');
      case 'deepseek_r1':
        return t('translation.localGenerationStyleDeepSeekR1');
      case 'generic':
        return t('translation.localGenerationStyleGeneric');
      default:
        return generationStyle ? String(generationStyle) : '-';
    }
  }, [t]);

  const formatTechnicalValue = React.useCallback((value: string | null | undefined) => {
    const normalized = String(value || '').trim();
    if (!normalized) return '-';
    return normalized.replace(/[_-]+/g, ' ');
  }, []);

  const formatElapsedTime = React.useCallback((elapsedMs: unknown, options?: { suffix?: string }) => {
    if (typeof elapsedMs !== 'number' || !Number.isFinite(elapsedMs)) return '-';
    const safeMs = Math.max(0, elapsedMs);
    const totalSeconds = safeMs / 1000;
    const suffix = options?.suffix || '';
    if (totalSeconds < 60) return `${totalSeconds.toFixed(1)} s${suffix}`;
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    if (hours > 0) {
      return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${seconds.toFixed(1).padStart(4, '0')}${suffix}`;
    }
    return `${String(minutes).padStart(2, '0')}:${seconds.toFixed(1).padStart(4, '0')}${suffix}`;
  }, []);

  const formatNumberList = React.useCallback((values: unknown, options?: { suffix?: string }) => {
    if (!Array.isArray(values) || values.length === 0) return '-';
    const normalized = values
      .map((value) => (typeof value === 'number' && Number.isFinite(value) ? Math.round(value) : null))
      .filter((value): value is number => value !== null);
    if (normalized.length === 0) return '-';
    const formatValue = (value: number) => `${value.toLocaleString()}${options?.suffix || ''}`;
    if (normalized.length <= 8) {
      return normalized.map(formatValue).join(' / ');
    }
    const head = normalized.slice(0, 6).map(formatValue).join(' / ');
    return `${head} / ... / ${formatValue(normalized[normalized.length - 1])} (${normalized.length})`;
  }, []);

  const formatDurationList = React.useCallback((values: unknown) => {
    if (!Array.isArray(values) || values.length === 0) return '-';
    const normalized = values.filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
    if (normalized.length === 0) return '-';
    if (normalized.length <= 8) return normalized.map((value) => formatElapsedTime(value)).join(' / ');
    const head = normalized.slice(0, 6).map((value) => formatElapsedTime(value)).join(' / ');
    return `${head} / ... / ${formatElapsedTime(normalized[normalized.length - 1])} (${normalized.length})`;
  }, [formatElapsedTime]);

  const localizeLocalBatchMode = React.useCallback((mode: string | null | undefined) => {
    switch (String(mode || '').trim()) {
      case 'token_aware':
        return t('translation.localBatchModeTokenAware');
      case 'fixed_lines':
        return t('translation.localBatchModeFixedLines');
      default:
        return mode ? String(mode) : '-';
    }
  }, [t]);

  const promptTemplateLabel = React.useCallback((templateId: string | null | undefined) => {
    switch (String(templateId || '')) {
      case 'subtitle_general':
        return t('translation.promptTemplateGeneral');
      case 'subtitle_concise_spoken':
        return t('translation.promptTemplateConciseSpoken');
      case 'subtitle_formal_precise':
        return t('translation.promptTemplateFormalPrecise');
      case 'subtitle_strict_alignment':
        return t('translation.promptTemplateStrictAlignment');
      case 'subtitle_asr_recovery':
        return t('translation.promptTemplateAsrRecovery');
      case 'subtitle_technical_terms':
        return t('translation.promptTemplateTechnicalTerms');
      default:
        return t('translation.promptTemplateNone');
    }
  }, [t]);

  const formatTranslationPipeline = (debug: any) => {
    if (!debug || typeof debug !== 'object') return null;
    const provider = debug.provider || {};
    const applied = debug.applied || {};
    const runtime = debug.runtime || {};
    const inference = runtime.lastInference || runtime.loadInference || null;
    const modelText = String(provider.model || '-');
    const retryText = String(applied.retryCount ?? 0);
    const qualityRetryText = String(applied.qualityRetryCount ?? 0);
    const fallbackText = applied.fallback
      ? (applied.fallbackType ? String(applied.fallbackType) : t('translation.monitorYes'))
      : t('translation.monitorNo');
    const strategyText = localizeCloudStrategy(applied.cloudStrategy);
    const parts = [
      `${t('translation.monitorProvider')}: ${providerLabel(String(provider.name || '-'))}`,
      `${t('translation.monitorModel')}: ${modelText}`,
      `${t('translation.monitorStrategy')}: ${strategyText}`,
      `${t('translation.monitorRetry')}: ${retryText}`,
      `${t('translation.monitorQualityRetry')}: ${qualityRetryText}`,
      `${t('translation.monitorStrictRetrySucceeded')}: ${applied.strictRetrySucceeded ? t('translation.monitorYes') : t('translation.monitorNo')}`,
      `${t('translation.monitorFallback')}: ${fallbackText}`,
    ];
    if (applied.localModelFamily) {
      parts.push(`${t('translation.monitorLocalModelFamily')}: ${localizeLocalModelFamily(applied.localModelFamily)}`);
    }
    if (applied.localModelProfileId) {
      parts.push(`${t('translation.monitorLocalModelProfileId')}: ${formatTechnicalValue(applied.localModelProfileId)}`);
    }
    if (applied.localPromptStyle) {
      parts.push(`${t('translation.monitorLocalPromptStyle')}: ${localizeLocalPromptStyle(applied.localPromptStyle)}`);
    }
    if (provider.profileId) {
      parts.push(`${t('translation.monitorProfileId')}: ${provider.profileId}`);
    }
    if (provider.profileFamily) {
      parts.push(`${t('translation.monitorProfileFamily')}: ${formatTechnicalValue(provider.profileFamily)}`);
    }
    if (applied.localBaselineConfidence) {
      parts.push(`${t('translation.monitorLocalBaselineConfidence')}: ${formatTechnicalValue(applied.localBaselineConfidence)}`);
    }
    if (applied.localBaselineTaskFamily) {
      parts.push(`${t('translation.monitorLocalBaselineTaskFamily')}: ${formatTechnicalValue(applied.localBaselineTaskFamily)}`);
    }
    if (typeof applied.localFallbackBaseline === 'boolean') {
      parts.push(`${t('translation.monitorLocalFallbackBaseline')}: ${applied.localFallbackBaseline ? t('translation.monitorYes') : t('translation.monitorNo')}`);
    }
    const localBatching = applied.localBatching || null;
    const localBatchCount =
      typeof localBatching?.batchCount === 'number'
        ? localBatching.batchCount
        : typeof applied.localBatchCount === 'number'
          ? applied.localBatchCount
          : null;
    if (localBatchCount != null) {
      parts.push(`${t('translation.monitorLocalBatchCount')}: ${localBatchCount}`);
    }
    if (typeof applied.cloudContextChunkCount === 'number' && applied.cloudContextChunkCount > 0) {
      parts.push(`${t('translation.monitorCloudContextChunks')}: ${applied.cloudContextChunkCount}`);
    }
    if (typeof applied.cloudBatching?.batchCount === 'number' && applied.cloudBatching.batchCount > 0) {
      parts.push(`${t('translation.monitorCloudBatchCount')}: ${applied.cloudBatching.batchCount}`);
    }
    if (inference?.acceleratorModel) {
      const memorySource = inference.memorySource ? ` / ${inference.memorySource}` : '';
      parts.push(`${t('monitor.runtime')}: ${inference.acceleratorModel}${memorySource}`);
    }
    return parts.join(' | ');
  };

  const localizePipelineWarnings = React.useCallback((warnings: unknown, issues?: RunIssue[] | null) => {
    const warningCodes = Array.isArray(issues) && issues.length > 0
      ? issues.map((issue) => String(issue?.code || '').trim())
      : Array.isArray(warnings)
        ? warnings.map((item) => String(item || '').trim())
        : [];

    return warningCodes
      .filter(Boolean)
      .map((code) => {
        switch (code) {
          case 'transient_retry_applied':
            return t('translation.warnTransientRetry');
          case 'provider_fallback_applied':
            return t('translation.warnProviderFallback');
          case 'line_safe_alignment_applied':
            return t('translation.warnLineSafeAlignment');
          case 'line_index_rebind_applied':
            return t('translation.warnLineIndexRebind');
          case 'line_json_map_repair_applied':
            return t('translation.warnLineJsonMapRepair');
          case 'line_json_map_pre_split_applied':
            return t('translation.warnLineJsonMapPreSplit');
          case 'line_json_map_partial_fallback':
            return t('translation.warnLineJsonMapPartial');
          case 'line_json_map_policy_split':
            return t('translation.warnLineJsonMapPolicySplit');
          case 'line_json_map_policy_single_line_fallback':
            return t('translation.warnLineJsonMapPolicySingleLineFallback');
          case 'line_json_map_policy_source_fallback':
            return t('translation.warnLineJsonMapPolicySourceFallback');
          case 'line_json_map_repair_disabled':
            return t('translation.warnLineJsonMapRepairDisabled');
          case 'line_alignment_repair_failed':
            return t('translation.warnLineAlignmentFailed');
          case 'local_repetition_loop_detected':
            return t('translation.warnLocalRepetitionLoop');
          case 'local_recursive_chunk_split_applied':
            return t('translation.warnLocalRecursiveChunkSplit');
          case 'local_single_line_retry_applied':
            return t('translation.warnLocalSingleLineRetry');
          case 'quality_retry_triggered':
            return t('translation.warnQualityRetry');
          case 'post_repair_quality_retry_triggered':
            return t('translation.warnPostRepairQualityRetry');
          case 'strict_retry_applied':
            return t('translation.warnStrictRetryApplied');
          case 'local_strict_retry_applied':
            return t('translation.warnLocalStrictRetry');
          case 'local_batch_translation_applied':
            return t('translation.warnLocalBatchTranslation');
          case 'translategemma_batch_translation_applied':
            return t('translation.warnTranslateGemmaBatchTranslation');
          case 'translategemma_recursive_chunk_split_applied':
            return t('translation.warnTranslateGemmaRecursiveChunkSplit');
          case 'translategemma_single_line_retry_applied':
            return t('translation.warnTranslateGemmaSingleLineRetry');
          case 'residual_line_retry_triggered':
            return t('translation.warnResidualLineRetryTriggered');
          case 'residual_line_retry_applied':
            return t('translation.warnResidualLineRetryApplied');
          case 'cloud_context_parse_failed':
            return t('translation.warnCloudContextParseFailed');
          case 'cloud_context_chunk_split':
            return t('translation.warnCloudContextChunkSplit');
          case 'cloud_context_split_depth_exhausted':
            return t('translation.warnCloudContextSplitDepthExhausted');
          case 'cloud_context_single_line_fallback':
            return t('translation.warnCloudContextSingleLineFallback');
          case 'cloud_provider_batch_translation_applied':
            return t('translation.warnCloudProviderBatchTranslation');
          case 'cloud_provider_batch_split_applied':
            return t('translation.warnCloudProviderBatchSplit');
          case 'quality_issue_empty_output':
            return t('translation.warnQualityIssueEmptyOutput');
          case 'quality_issue_repetition_loop':
            return t('translation.warnQualityIssueRepetitionLoop');
          case 'quality_issue_adjacent_duplicate':
            return t('translation.warnQualityIssueAdjacentDuplicate');
          case 'quality_issue_pass_through':
            return t('translation.warnQualityIssuePassThrough');
          case 'quality_issue_zh_tw_naturalization_needed':
            return t('translation.warnQualityIssueZhTwNaturalization');
          case 'quality_issue_target_lang_mismatch':
            return t('translation.warnQualityIssueTargetMismatch');
          case 'quality_issue_line_count_loss':
            return t('translation.warnQualityIssueLineCountLoss');
          case 'quality_issue_marker_loss':
            return t('translation.warnQualityIssueMarkerLoss');
          default:
            return code;
        }
      });
  }, [t]);

  const extractPipelineErrors = React.useCallback((debug: any, errorIssue?: RunIssue | null) => {
    const errorItems: string[] = [];
    if (errorIssue?.technicalMessage) {
      errorItems.push(String(errorIssue.technicalMessage).trim());
    }
    if (Array.isArray(debug?.errorIssues)) {
      for (const issue of debug.errorIssues) {
        const technicalMessage = String(issue?.technicalMessage || '').trim();
        if (technicalMessage) errorItems.push(technicalMessage);
      }
    }
    if (typeof debug?.errors?.request === 'string' && debug.errors.request.trim()) {
      errorItems.push(debug.errors.request.trim());
    }
    return Array.from(new Set(errorItems));
  }, []);

  const handleStopTranslation = React.useCallback(() => {
    const current = translateEventSourceRef.current;
    if (!current) return;
    current.close();
    translateEventSourceRef.current = null;
    setIsTranslating(false);
    setTranslateMsg(t('translation.translationStopped'));
    setModelLoadStatus(selectedTranslateModel?.isLocal ? 'ok' : 'idle');
    setModelLoadError(null);
    setIsPersistingResult(false);
    onTaskLockChange?.(false);
  }, [onTaskLockChange, selectedTranslateModel, t]);

  const handleSelectPromptTemplate = React.useCallback((templateId: PromptTemplateId) => {
    setSelectedPromptTemplateId(templateId);
    if (!templateId) {
      setPromptText('');
      return;
    }
    setPromptText(
      targetLang === 'other'
        ? getCustomTargetTranslationPromptTemplateText(templateId, effectiveTargetLanguage)
        : getTranslationPromptTemplateText(templateId, effectiveTargetLanguage)
    );
  }, [targetLang, effectiveTargetLanguage]);

  const handleStartTranslation = async () => {
    if (!project?.id || !canStartTranslation) return;
    if (!String(effectiveTargetLanguage || '').trim()) {
      setModelLoadStatus('failed');
      setModelLoadError(t('translation.errorCustomTargetRequired'));
      return;
    }
    if (!selectedModelId) {
      setModelLoadStatus('failed');
      setModelLoadError(t('stt.noModels'));
      return;
    }

    if (translateEventSourceRef.current) {
      translateEventSourceRef.current.close();
      translateEventSourceRef.current = null;
    }

    setIsTranslating(true);
    onTaskLockChange?.(true);
    setModelLoadStatus('loading');
    setModelLoadError(null);
    setProgress(0);
    setTranslatedLines([]);
    setError(null);
    setTranslateMsg(null);
    setPipelineMode(null);
    setTranslationDebug(null);
    setPipelineWarnings([]);
    setPipelineErrors([]);
    setHasTimecodes(false);
    setIsPersistingResult(false);
    setIsEditingTranslatedOutput(false);
    setEditingTranslatedOutputRows([]);
    setEditingTranslatedOutputBaselineIds([]);
    setIsSavingTranslatedOutput(false);

    const params = new URLSearchParams();
    params.set('targetLang', effectiveTargetLanguage);
    if (selectedModelId) params.set('modelId', selectedModelId);
    if (isTranslateGemmaModel(selectedTranslateModel) && effectiveSourceLanguage) {
      params.set('sourceLang', effectiveSourceLanguage);
    }
    if (!translateGemmaPromptControlsDisabled && selectedPromptTemplateId) {
      params.set('promptTemplateId', selectedPromptTemplateId);
    }
    if (sourceType === 'project' && selectedSourceAssetName) {
      params.set('assetName', selectedSourceAssetName);
    }
    const safePrompt = translateGemmaPromptControlsDisabled ? '' : sanitizeInput(promptText);
    const safeGlossary = translateGemmaPromptControlsDisabled
      ? ''
      : String(glossaryText || '').replace(/\u0000/g, '').trim().slice(0, 4000);
    if (safePrompt) params.set('prompt', safePrompt);
    if (safeGlossary) params.set('glossary', safeGlossary);
    params.set('strictJsonLineRepair', strictJsonLineRepairEnabled ? '1' : '0');

    let eventSource: EventSource;
    try {
      eventSource = new EventSource(`/api/translate/${project.id}?${params.toString()}`);
    } catch (err) {
      console.error('Failed to start translation stream', err);
      setModelLoadStatus('failed');
      setModelLoadError(t('translation.errorFailed'));
      setIsPersistingResult(false);
      setIsTranslating(false);
      onTaskLockChange?.(false);
      return;
    }
    translateEventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      if (translateEventSourceRef.current !== eventSource) return;
      const data = JSON.parse(event.data);
      const progressEvent = (data.event || null) as RunProgressEvent | null;
      const errorIssue = (data.errorIssue || null) as RunIssue | null;

      if (data.error) {
        const errorDetail = String(data.error || t('translation.errorFailed'));
        setModelLoadStatus('failed');
        setModelLoadError(errorDetail);
        setError(errorDetail);
        setPipelineErrors(extractPipelineErrors(null, errorIssue).length > 0 ? extractPipelineErrors(null, errorIssue) : [errorDetail]);
        setIsPersistingResult(false);
        setIsTranslating(false);
        setTranslateMsg(null);
        translateEventSourceRef.current = null;
        onTaskLockChange?.(false);
        eventSource.close();
        return;
      }

      if (data.status === 'processing') {
        setModelLoadStatus('ok');
        setModelLoadError(null);
        setTranslateMsg(localizeTranslationEvent(progressEvent, String(data.message || '')));
        setProgress((prev) => {
          const hintedProgress =
            typeof progressEvent?.progressHint === 'number' && Number.isFinite(progressEvent.progressHint)
              ? Math.max(prev, Math.min(progressEvent.progressHint, 90))
              : prev;
          return Math.min(hintedProgress + 8, 90);
        });
        return;
      }

      if (data.status === 'completed') {
        setModelLoadStatus('ok');
        setModelLoadError(null);
        const result = data.result || {};
        const translatedText = String(result.translatedText || '');
        const persistedTranslatedText = rebuildTranslationWithSourceTimecodes(activeSourceText, translatedText);
        const translatedArray = persistedTranslatedText.split('\n');
        const merged = sourceLines.map((line, i) => ({
          original: line,
          translated: translatedArray[i] || '',
        }));

        setTranslatedLines(merged);
        setProgress(100);
        setIsTranslating(false);
        setTranslateMsg(localizeTranslationEvent(progressEvent, 'Translation completed.'));
        setPipelineMode(formatTranslationPipeline(result.debug));
        setTranslationDebug(result.debug || null);
        setPipelineWarnings(localizePipelineWarnings(result.debug?.warnings, result.debug?.warningIssues));
        setPipelineErrors(extractPipelineErrors(result.debug));
        setHasTimecodes(
          typeof result?.exports?.hasTimecodes === 'boolean'
            ? result.exports.hasTimecodes
            : typeof result?.debug?.artifacts?.hasTimecodes === 'boolean'
              ? result.debug.artifacts.hasTimecodes
            : hasStrictBracketTimecodes(persistedTranslatedText)
        );

        setIsPersistingResult(true);
        Promise.resolve(
          onUpdateProject({
            originalSubtitles: activeSourceText,
            translatedSubtitles: persistedTranslatedText,
            status: PROJECT_STATUS.COMPLETED,
          })
        )
          .catch((persistErr) => {
            console.error('Failed to persist translated subtitles to project', persistErr);
            setError(t('translation.errorFailed'));
          })
          .finally(() => {
            setIsPersistingResult(false);
          });

        translateEventSourceRef.current = null;
        onTaskLockChange?.(false);
        eventSource.close();
      }
    };

    eventSource.onerror = () => {
      if (translateEventSourceRef.current !== eventSource) return;
      const errorDetail = t('translation.errorFailed');
      setModelLoadStatus('failed');
      setModelLoadError(errorDetail);
      setError(errorDetail);
      setPipelineErrors([errorDetail]);
      setIsPersistingResult(false);
      setIsTranslating(false);
      setTranslateMsg(null);
      translateEventSourceRef.current = null;
      onTaskLockChange?.(false);
      eventSource.close();
    };
  };

  const langLabel =
    targetLang === 'other'
      ? String(customTargetLanguage || '').trim() || t('lang.other')
      : TARGET_LANG_OPTIONS.find((l) => l.value === targetLang)?.label || targetLang;
  const hasTimecodedTranslation = hasTimecodes || hasStrictBracketTimecodes(String(project?.translatedSubtitles || ''));
  const canDownloadTranslation = progress === 100 && Boolean(project?.translatedSubtitles) && translatedLines.length > 0;
  const nextStepLabel = React.useMemo(
    () => t('dashboard.goToStep').replace('{step}', t('player.title')),
    [t]
  );
  const translationSummary = React.useMemo(() => {
    if (!translationDebug?.requested) return null;
    const localBatching = translationDebug.applied?.localBatching || null;
    const cloudBatching = translationDebug.applied?.cloudBatching || null;
    const localBatchCount =
      typeof localBatching?.batchCount === 'number'
        ? localBatching.batchCount
        : typeof translationDebug.applied?.localBatchCount === 'number'
          ? translationDebug.applied.localBatchCount
          : null;
    const localBatchLineCounts = localBatching?.lineCounts || translationDebug.applied?.localBatchLineCounts || [];
    const localBatchPromptTokens = localBatching?.promptTokens || translationDebug.applied?.localBatchPromptTokens || [];
    const localBatchBudgetParts = [
      typeof localBatching?.inputTokenBudget === 'number'
        ? `${t('translation.monitorLocalBatchInputBudget')}: ${localBatching.inputTokenBudget.toLocaleString()}`
        : null,
      typeof localBatching?.outputTokenBudget === 'number'
        ? `${t('translation.monitorLocalBatchOutputBudget')}: ${localBatching.outputTokenBudget.toLocaleString()}`
        : null,
      typeof localBatching?.contextWindow === 'number'
        ? `${t('translation.monitorLocalBatchContext')}: ${localBatching.contextWindow.toLocaleString()}`
        : null,
      typeof localBatching?.maxLines === 'number'
        ? `${t('translation.monitorLocalBatchMaxLines')}: ${localBatching.maxLines.toLocaleString()}`
        : null,
      typeof localBatching?.charBudget === 'number'
        ? `${t('translation.monitorLocalBatchCharBudget')}: ${localBatching.charBudget.toLocaleString()}`
        : null,
      localBatching?.fallbackReason ? `${t('translation.monitorLocalBatchFallback')}: ${formatTechnicalValue(localBatching.fallbackReason)}` : null,
    ].filter(Boolean);
    const cloudBatchBudgetParts = [
      typeof cloudBatching?.maxLines === 'number'
        ? `${t('translation.monitorLocalBatchMaxLines')}: ${cloudBatching.maxLines.toLocaleString()}`
        : null,
      typeof cloudBatching?.minLines === 'number'
        ? `${t('translation.monitorCloudBatchMinLines')}: ${cloudBatching.minLines.toLocaleString()}`
        : null,
      typeof cloudBatching?.charBudget === 'number'
        ? `${t('translation.monitorLocalBatchCharBudget')}: ${cloudBatching.charBudget.toLocaleString()}`
        : null,
      typeof cloudBatching?.maxOutputTokens === 'number'
        ? `${t('translation.monitorCloudBatchMaxOutputTokens')}: ${cloudBatching.maxOutputTokens.toLocaleString()}`
        : null,
      typeof cloudBatching?.timeoutMs === 'number'
        ? `${t('translation.monitorCloudBatchTimeout')}: ${formatElapsedTime(cloudBatching.timeoutMs)}`
        : null,
      typeof cloudBatching?.stream === 'boolean'
        ? `${t('translation.monitorCloudBatchStream')}: ${cloudBatching.stream ? t('translation.monitorYes') : t('translation.monitorNo')}`
        : null,
    ].filter(Boolean);
    return {
      targetLanguage: String(translationDebug.requested.targetLanguageDescriptor || langLabel || '').trim() || '-',
      promptTemplate: promptTemplateLabel(translationDebug.requested.promptTemplateId),
      effectiveGlossary: String(translationDebug.requested.effectiveGlossary || '').trim(),
      strategy: localizeCloudStrategy(translationDebug.applied?.cloudStrategy),
      qualityMode: localizeTranslationQualityMode(translationDebug.applied?.translationQualityMode),
      qualityRetryCount:
        typeof translationDebug.applied?.qualityRetryCount === 'number'
          ? String(translationDebug.applied.qualityRetryCount)
          : '0',
      strictRetrySucceeded:
        typeof translationDebug.applied?.strictRetrySucceeded === 'boolean'
          ? (translationDebug.applied.strictRetrySucceeded ? t('translation.monitorYes') : t('translation.monitorNo'))
          : '-',
      profileId: String(translationDebug.provider?.profileId || '').trim() || '-',
      profileFamily: formatTechnicalValue(translationDebug.provider?.profileFamily),
      localModelFamily: localizeLocalModelFamily(translationDebug.applied?.localModelFamily),
      localModelProfileId: formatTechnicalValue(translationDebug.applied?.localModelProfileId),
      localPromptStyle: localizeLocalPromptStyle(translationDebug.applied?.localPromptStyle),
      localGenerationStyle: localizeLocalGenerationStyle(translationDebug.applied?.localGenerationStyle),
      localBaselineConfidence: formatTechnicalValue(translationDebug.applied?.localBaselineConfidence),
      localBaselineTaskFamily: formatTechnicalValue(translationDebug.applied?.localBaselineTaskFamily),
      localFallbackBaseline:
        typeof translationDebug.applied?.localFallbackBaseline === 'boolean'
          ? (translationDebug.applied.localFallbackBaseline ? t('translation.monitorYes') : t('translation.monitorNo'))
          : '-',
      localBatchingMode: localizeLocalBatchMode(localBatching?.mode || translationDebug.applied?.localBatchingMode),
      localBatchCount: localBatchCount != null ? String(localBatchCount) : '-',
      localBatchLineCounts: formatNumberList(localBatchLineCounts),
      localBatchCharCounts: formatNumberList(localBatching?.charCounts),
      localBatchPromptTokens: formatNumberList(localBatchPromptTokens),
      localBatchEstimatedOutputTokens: formatNumberList(localBatching?.estimatedOutputTokens),
      localBatchDurations: formatDurationList(localBatching?.durationsMs),
      localBatchTotalDuration: formatElapsedTime(localBatching?.totalDurationMs),
      localBatchMaxDuration: formatElapsedTime(localBatching?.maxDurationMs),
      localBatchBudget: localBatchBudgetParts.length > 0 ? localBatchBudgetParts.join(' · ') : '-',
      cloudBatchCount:
        typeof cloudBatching?.batchCount === 'number' ? String(cloudBatching.batchCount) : '-',
      cloudBatchLineCounts: formatNumberList(cloudBatching?.lineCounts),
      cloudBatchCharCounts: formatNumberList(cloudBatching?.charCounts),
      cloudBatchEstimatedOutputTokens: formatNumberList(cloudBatching?.estimatedOutputTokens),
      cloudBatchDurations: formatDurationList(cloudBatching?.durationsMs),
      cloudBatchTotalDuration: formatElapsedTime(cloudBatching?.totalDurationMs),
      cloudBatchMaxDuration: formatElapsedTime(cloudBatching?.maxDurationMs),
      cloudBatchSplitCount:
        typeof cloudBatching?.splitCount === 'number' ? String(cloudBatching.splitCount) : '-',
      cloudBatchBudget: cloudBatchBudgetParts.length > 0 ? cloudBatchBudgetParts.join(' · ') : '-',
      strictJsonRepair:
        typeof translationDebug.requested.jsonLineRepairEnabled === 'boolean'
          ? (translationDebug.requested.jsonLineRepairEnabled ? t('translation.monitorYes') : t('translation.monitorNo'))
          : '-',
      cloudContextChunkCount:
        typeof translationDebug.applied?.cloudContextChunkCount === 'number'
          ? String(translationDebug.applied.cloudContextChunkCount)
          : '-',
      cloudContextFallbackCount:
        typeof translationDebug.applied?.cloudContextFallbackCount === 'number'
          ? String(translationDebug.applied.cloudContextFallbackCount)
          : '-',
    };
  }, [formatDurationList, formatElapsedTime, formatNumberList, formatTechnicalValue, langLabel, localizeCloudStrategy, localizeLocalBatchMode, localizeLocalGenerationStyle, localizeLocalModelFamily, localizeLocalPromptStyle, localizeTranslationQualityMode, promptTemplateLabel, t, translationDebug]);
  const translationRuntimeSummary = React.useMemo(() => {
    const runtime = translationDebug?.runtime;
    if (!runtime) return null;
    const inference = runtime.lastInference || runtime.loadInference || null;
    const perf = runtime.lastPerfMetrics || null;
    const formatMetric = (value: unknown, digits = 1, suffix = '') => {
      return typeof value === 'number' && Number.isFinite(value)
        ? `${Number(value).toFixed(digits)}${suffix}`
        : '-';
    };
    return {
      requestedDevice: String(runtime.requestedDevice || '-'),
      pipelineDevice: String(runtime.pipelineDevice || '-'),
      pipelineKind: String(runtime.pipelineKind || '-'),
      vlmRuntimeMode: String(runtime.vlmRuntimeMode || '-'),
      modelPath: String(runtime.modelPath || '-'),
      cacheDir: String(runtime.cacheDir || '-'),
      promptLookupEnabled: Boolean(runtime.promptLookupEnabled),
      schedulerTokenLimit:
        typeof runtime?.schedulerConfig?.max_num_batched_tokens === 'number' &&
        Number.isFinite(runtime.schedulerConfig.max_num_batched_tokens)
          ? String(runtime.schedulerConfig.max_num_batched_tokens)
          : '-',
      observedAt: String(inference?.observedAt || ''),
      source: String(inference?.source || ''),
      acceleratorModel: String(inference?.acceleratorModel || '-'),
      luid: String(inference?.luid || ''),
      memorySource: String(inference?.memorySource || '-'),
      vram:
        Number.isFinite(inference?.vramUsedGB) && Number.isFinite(inference?.vramTotalGB)
          ? `${Number(inference?.vramUsedGB || 0).toFixed(1)}/${Number(inference?.vramTotalGB || 0).toFixed(1)}GB`
          : '-',
      utilization:
        typeof inference?.utilization === 'number' && Number.isFinite(inference.utilization)
          ? `${Math.round(inference.utilization)}%`
          : '-',
      physIndex:
        typeof inference?.physIndex === 'number' && Number.isFinite(inference.physIndex)
          ? String(inference.physIndex)
          : '-',
      inputTokens:
        typeof perf?.inputTokens === 'number' && Number.isFinite(perf.inputTokens)
          ? String(perf.inputTokens)
          : '-',
      generatedTokens:
        typeof perf?.generatedTokens === 'number' && Number.isFinite(perf.generatedTokens)
          ? String(perf.generatedTokens)
          : '-',
      ttft: formatElapsedTime(perf?.ttftMs),
      tpot: formatElapsedTime(perf?.tpotMs, { suffix: '/token' }),
      throughput: formatMetric(perf?.throughputTokensPerSec, 2, ' tok/s'),
      generateDuration: formatElapsedTime(perf?.generateDurationMs),
    };
  }, [formatElapsedTime, translationDebug]);
  const diarizationSummary = React.useMemo(() => {
    if (sourceType !== 'transcription') return null;
    const diagnostics = sourceDiarizationDiagnostics;
    if (!diagnostics) return null;
    const options = diagnostics.options;
    const providerLabel =
      diagnostics.provider === 'semantic' ? diarizationCopy.providerSemantic : diarizationCopy.providerAcoustic;
    const sourceLabel =
      diagnostics.selectedSource === 'speech_region'
        ? diarizationCopy.sourceSpeechRegion
        : diagnostics.selectedSource === 'vad_chunk'
        ? diarizationCopy.sourceVad
        : diagnostics.selectedSource === 'pyannote'
          ? diarizationCopy.sourcePyannote
        : diagnostics.selectedSource === 'semantic'
          ? diarizationCopy.sourceSemantic
          : diarizationCopy.sourceChunk;
    const modeLabel = options
      ? ({
          auto: diarizationCopy.modeAuto,
          fixed: diarizationCopy.modeFixed,
          range: diarizationCopy.modeRange,
          many: diarizationCopy.modeMany,
        } as const)[options.mode]
      : null;
    const sceneLabel = options
      ? ({
          interview: diarizationCopy.sceneInterview,
          podcast: diarizationCopy.scenePodcast,
          meeting: diarizationCopy.sceneMeeting,
          presentation_qa: diarizationCopy.scenePresentationQa,
          custom: diarizationCopy.sceneCustom,
        } as const)[options.scenePreset]
      : null;

    let speakerTarget = '-';
    if (options) {
      if (options.exactSpeakerCount) {
        speakerTarget = diarizationCopy.speakerExact.replace('{count}', String(options.exactSpeakerCount));
      } else if (options.mode === 'many') {
        speakerTarget = diarizationCopy.speakerMany.replace('{min}', String(options.minSpeakers));
      } else {
        speakerTarget = diarizationCopy.speakerRange
          .replace('{min}', String(options.minSpeakers))
          .replace('{max}', String(options.maxSpeakers));
      }
    }

    const resegmentedValue =
      diagnostics.selectedPass?.resegmentedChunkCount && diagnostics.selectedPass?.resegmentationPasses
        ? diarizationCopy.resegmentedValue
            .replace('{count}', String(diagnostics.selectedPass.resegmentedChunkCount))
            .replace('{passes}', String(diagnostics.selectedPass.resegmentationPasses))
        : '0';

    return {
      providerLabel,
      sourceLabel,
      modeLabel,
      sceneLabel,
      speakerTarget,
      outputSpeakers: diagnostics.selectedPass?.uniqueSpeakerCount ?? null,
      speechSegments: diagnostics.speechSegmentCount,
      windows: diagnostics.vadWindowCount,
      regions: diagnostics.selectedPass?.regionCount ?? null,
      threshold: diagnostics.selectedPass?.threshold ?? null,
      resegmentedValue,
    };
  }, [sourceDiarizationDiagnostics, diarizationCopy, sourceType]);
  const debugSummary = React.useMemo(() => {
    const parts: string[] = [];
    if (translationDebug?.timing && typeof translationDebug.timing.elapsedMs === 'number') {
      parts.push(`${t('stt.elapsedTime')}: ${formatElapsedTime(translationDebug.timing.elapsedMs)}`);
    }
    if (translationSummary?.qualityMode && translationSummary.qualityMode !== '-') {
      parts.push(`${t('translation.monitorQualityMode')}: ${translationSummary.qualityMode}`);
    }
    if (translationSummary?.localBatchCount && translationSummary.localBatchCount !== '-') {
      parts.push(`${t('translation.monitorLocalBatchCount')}: ${translationSummary.localBatchCount}`);
    }
    if (translationSummary?.cloudBatchCount && translationSummary.cloudBatchCount !== '-') {
      parts.push(`${t('translation.monitorCloudBatchCount')}: ${translationSummary.cloudBatchCount}`);
    }
    if (pipelineWarnings.length > 0) parts.push(`${t('translation.pipelineWarnings')} ${pipelineWarnings.length}`);
    if (pipelineErrors.length > 0) parts.push(`${t('translation.pipelineErrors')} ${pipelineErrors.length}`);
    return parts.join(' · ');
  }, [formatElapsedTime, pipelineErrors.length, pipelineWarnings.length, t, translationDebug?.timing, translationSummary?.cloudBatchCount, translationSummary?.localBatchCount, translationSummary?.qualityMode]);
  const translationMonitorBadges = React.useMemo<RunMonitorBadge[]>(() => {
    const badges: RunMonitorBadge[] = [];
    if (pipelineMode) {
      for (const part of pipelineMode.split('|').map((item) => item.trim()).filter(Boolean)) {
        badges.push({ label: part, tone: 'info' });
      }
    }
    if (translationDebug?.quality?.passThroughRisk) {
      badges.push({
        label: `${t('translation.qualityPassThroughRisk')}: ${localizeQualityRisk(translationDebug.quality.passThroughRisk)}`,
        tone: translationDebug.quality.passThroughRisk === 'high' ? 'warning' : 'default',
      });
    }
    if (pipelineErrors.length > 0) {
      badges.push({ label: `${t('translation.pipelineErrors')}: ${pipelineErrors.length}`, tone: 'error' });
    } else if (pipelineWarnings.length > 0) {
      badges.push({ label: `${t('translation.pipelineWarnings')}: ${pipelineWarnings.length}`, tone: 'warning' });
    }
    return badges;
  }, [localizeQualityRisk, pipelineErrors.length, pipelineMode, pipelineWarnings.length, t, translationDebug?.quality?.passThroughRisk]);
  const translationMonitorSections = React.useMemo<RunMonitorSection[]>(() => {
    const sections: RunMonitorSection[] = [];

    if (translationSummary) {
      sections.push({
        key: 'execution',
        title: t('monitor.execution'),
        fields: [
          { label: t('translation.targetLanguage'), value: translationSummary.targetLanguage },
          { label: t('translation.promptLabel'), value: translationSummary.promptTemplate },
          { label: t('translation.monitorStrategy'), value: translationSummary.strategy },
          { label: t('translation.monitorQualityMode'), value: translationSummary.qualityMode },
          { label: t('translation.monitorQualityRetry'), value: translationSummary.qualityRetryCount },
          { label: t('translation.monitorStrictRetrySucceeded'), value: translationSummary.strictRetrySucceeded },
          { label: t('translation.monitorProfileId'), value: translationSummary.profileId },
          { label: t('translation.monitorProfileFamily'), value: translationSummary.profileFamily },
          { label: t('translation.monitorLocalModelFamily'), value: translationSummary.localModelFamily },
          { label: t('translation.monitorLocalModelProfileId'), value: translationSummary.localModelProfileId },
          { label: t('translation.monitorLocalPromptStyle'), value: translationSummary.localPromptStyle },
          { label: t('translation.monitorLocalGenerationStyle'), value: translationSummary.localGenerationStyle },
          { label: t('translation.monitorLocalBaselineConfidence'), value: translationSummary.localBaselineConfidence },
          { label: t('translation.monitorLocalBaselineTaskFamily'), value: translationSummary.localBaselineTaskFamily },
          { label: t('translation.monitorLocalFallbackBaseline'), value: translationSummary.localFallbackBaseline },
          ...(translationSummary.localBatchCount !== '-'
            ? [
                { label: t('translation.monitorLocalBatchingMode'), value: translationSummary.localBatchingMode },
                { label: t('translation.monitorLocalBatchCount'), value: translationSummary.localBatchCount },
                { label: t('translation.monitorLocalBatchLines'), value: translationSummary.localBatchLineCounts },
                { label: t('translation.monitorLocalBatchChars'), value: translationSummary.localBatchCharCounts },
                { label: t('translation.monitorLocalBatchPromptTokens'), value: translationSummary.localBatchPromptTokens },
                { label: t('translation.monitorLocalBatchEstimatedOutput'), value: translationSummary.localBatchEstimatedOutputTokens },
                { label: t('translation.monitorLocalBatchDurations'), value: translationSummary.localBatchDurations },
                { label: t('translation.monitorLocalBatchTotalDuration'), value: translationSummary.localBatchTotalDuration },
                { label: t('translation.monitorLocalBatchMaxDuration'), value: translationSummary.localBatchMaxDuration },
                { label: t('translation.monitorLocalBatchBudget'), value: translationSummary.localBatchBudget },
              ]
            : []),
          ...(translationSummary.cloudBatchCount !== '-'
            ? [
                { label: t('translation.monitorCloudBatchCount'), value: translationSummary.cloudBatchCount },
                { label: t('translation.monitorCloudBatchLines'), value: translationSummary.cloudBatchLineCounts },
                { label: t('translation.monitorCloudBatchChars'), value: translationSummary.cloudBatchCharCounts },
                { label: t('translation.monitorCloudBatchEstimatedOutput'), value: translationSummary.cloudBatchEstimatedOutputTokens },
                { label: t('translation.monitorCloudBatchDurations'), value: translationSummary.cloudBatchDurations },
                { label: t('translation.monitorCloudBatchTotalDuration'), value: translationSummary.cloudBatchTotalDuration },
                { label: t('translation.monitorCloudBatchMaxDuration'), value: translationSummary.cloudBatchMaxDuration },
                { label: t('translation.monitorCloudBatchSplitCount'), value: translationSummary.cloudBatchSplitCount },
                { label: t('translation.monitorLocalBatchBudget'), value: translationSummary.cloudBatchBudget },
              ]
            : []),
          { label: t('translation.strictJsonRepairToggle'), value: translationSummary.strictJsonRepair },
          { label: t('translation.monitorCloudContextChunks'), value: translationSummary.cloudContextChunkCount },
          { label: t('translation.monitorCloudContextFallbacks'), value: translationSummary.cloudContextFallbackCount },
          { label: t('translation.glossary'), value: translationSummary.effectiveGlossary || '-' },
        ],
      });
    }

    if (translationDebug?.quality) {
      sections.push({
        key: 'quality',
        title: t('monitor.quality'),
        fields: [
          {
            label: t('translation.monitorStrictRetrySucceeded'),
            value:
              typeof translationDebug.applied?.strictRetrySucceeded === 'boolean'
                ? (translationDebug.applied.strictRetrySucceeded ? t('translation.monitorYes') : t('translation.monitorNo'))
                : t('translation.monitorNo'),
          },
          { label: t('translation.qualityTargetLanguageMatch'), value: translationDebug.quality.targetLanguageMatch ? t('translation.monitorYes') : t('translation.monitorNo') },
          { label: t('translation.qualityLineCountMatch'), value: translationDebug.quality.lineCountMatch ? t('translation.monitorYes') : t('translation.monitorNo') },
          { label: t('translation.qualityPassThroughRisk'), value: localizeQualityRisk(translationDebug.quality.passThroughRisk) },
          { label: t('translation.qualityRepetitionRisk'), value: localizeQualityRisk(translationDebug.quality.repetitionRisk) },
          { label: t('translation.qualityMarkerPreservation'), value: localizeMarkerPreservation(translationDebug.quality.markerPreservation) },
        ],
      });
    }

    if (translationRuntimeSummary) {
      sections.push({
        key: 'runtime',
        title: t('monitor.runtime'),
        fields: [
          { label: t('translation.runtimeRequestedDevice'), value: translationRuntimeSummary.requestedDevice },
          { label: t('translation.runtimePipelineDevice'), value: translationRuntimeSummary.pipelineDevice },
          { label: t('translation.runtimePipelineKind'), value: translationRuntimeSummary.pipelineKind },
          { label: t('translation.runtimeVlmMode'), value: translationRuntimeSummary.vlmRuntimeMode },
          { label: t('translation.runtimeAccelerator'), value: translationRuntimeSummary.acceleratorModel },
          { label: t('translation.runtimeMemorySource'), value: translationRuntimeSummary.memorySource },
          { label: t('translation.runtimeObservedVram'), value: translationRuntimeSummary.vram },
          { label: t('translation.runtimeObservedUtilization'), value: translationRuntimeSummary.utilization },
          { label: t('translation.runtimeLuid'), value: translationRuntimeSummary.luid || '-' },
          { label: t('translation.runtimePhysIndex'), value: translationRuntimeSummary.physIndex },
          { label: t('translation.runtimeCacheDir'), value: translationRuntimeSummary.cacheDir },
          { label: t('translation.runtimePromptLookup'), value: translationRuntimeSummary.promptLookupEnabled ? t('translation.monitorYes') : t('translation.monitorNo') },
          { label: t('translation.runtimeSchedulerTokens'), value: translationRuntimeSummary.schedulerTokenLimit },
          { label: t('translation.runtimeInputTokens'), value: translationRuntimeSummary.inputTokens },
          { label: t('translation.runtimeGeneratedTokens'), value: translationRuntimeSummary.generatedTokens },
          { label: t('translation.runtimeTtft'), value: translationRuntimeSummary.ttft },
          { label: t('translation.runtimeTpot'), value: translationRuntimeSummary.tpot },
          { label: t('translation.runtimeThroughput'), value: translationRuntimeSummary.throughput },
          { label: t('translation.runtimeGenerateDuration'), value: translationRuntimeSummary.generateDuration },
          { label: t('translation.runtimeInferenceSource'), value: translationRuntimeSummary.source || '-' },
          { label: t('translation.runtimeObservedAt'), value: translationRuntimeSummary.observedAt || '-' },
          { label: t('translation.runtimeModelPath'), value: translationRuntimeSummary.modelPath, breakAll: true },
        ],
      });
    }

    if (translationDebug?.timing) {
      sections.push({
        key: 'timing',
        title: t('monitor.timing'),
        fields: [
          { label: t('stt.elapsedTime'), value: formatElapsedTime(translationDebug.timing.elapsedMs) },
          { label: t('translation.runtimeGenerateDuration'), value: formatElapsedTime(translationDebug.timing.providerMs) },
          { label: t('translation.monitorQualityRetry'), value: formatElapsedTime(translationDebug.timing.qualityRetryMs) },
        ],
      });
    }

    sections.push({
      key: 'source',
      title: t('monitor.diagnostics'),
      content: isLoadingSourceDiarization ? (
        <div>{diarizationCopy.loading}</div>
      ) : diarizationSummary ? (
        <>
          <div>{diarizationCopy.provider}: {diarizationSummary.providerLabel}</div>
          <div>{diarizationCopy.source}: {diarizationSummary.sourceLabel}</div>
          {diarizationSummary.modeLabel && <div>{diarizationCopy.mode}: {diarizationSummary.modeLabel}</div>}
          {diarizationSummary.sceneLabel && <div>{diarizationCopy.scene}: {diarizationSummary.sceneLabel}</div>}
          <div>{diarizationCopy.speakerTarget}: {diarizationSummary.speakerTarget}</div>
          {diarizationSummary.outputSpeakers != null && <div>{diarizationCopy.outputSpeakers}: {diarizationSummary.outputSpeakers}</div>}
          <div>{diarizationCopy.speechSegments}: {diarizationSummary.speechSegments}</div>
          <div>{diarizationCopy.windows}: {diarizationSummary.windows}</div>
          {diarizationSummary.regions != null && <div>{diarizationCopy.regions}: {diarizationSummary.regions}</div>}
          {diarizationSummary.threshold != null && <div>{diarizationCopy.threshold}: {Number(diarizationSummary.threshold).toFixed(2)}</div>}
          <div>{diarizationCopy.resegmented}: {diarizationSummary.resegmentedValue}</div>
        </>
      ) : (
        <div>{diarizationCopy.unavailable}</div>
      ),
    });

    if (pipelineWarnings.length > 0) {
      sections.push({
        key: 'warnings',
        title: t('translation.pipelineWarnings'),
        content: (
          <>
            {pipelineWarnings.map((warning, idx) => (
              <div key={`${warning}-${idx}`}>- {warning}</div>
            ))}
          </>
        ),
      });
    }

    if (pipelineErrors.length > 0) {
      sections.push({
        key: 'errors',
        title: t('translation.pipelineErrors'),
        tone: 'error',
        content: (
          <>
            {pipelineErrors.map((errItem, idx) => (
              <div key={`${errItem}-${idx}`}>- {errItem}</div>
            ))}
          </>
        ),
      });
    }

    return sections;
  }, [diarizationCopy, diarizationSummary, formatElapsedTime, isLoadingSourceDiarization, localizeMarkerPreservation, localizeQualityRisk, pipelineErrors, pipelineWarnings, t, translationDebug, translationRuntimeSummary, translationSummary]);

  React.useEffect(() => {
    if (pipelineErrors.length > 0) {
      setShowDebugDetails(true);
    }
  }, [pipelineErrors.length]);

  const findActiveLineIndexByTime = React.useCallback((timeSec: number): number | null => {
    if (!Number.isFinite(timeSec) || parsedTranslatedLines.length === 0) return null;
    let targetIndex: number | null = null;
    for (const item of parsedTranslatedLines) {
      if (item.startSeconds == null) continue;
      if (item.startSeconds <= timeSec) {
        targetIndex = item.index;
        continue;
      }
      break;
    }
    return targetIndex;
  }, [parsedTranslatedLines]);

  const stopPreviewPlayback = React.useCallback(() => {
    const audio = previewAudioRef.current;
    if (!audio) return;
    audio.pause();
    setActivePreviewLine(null);
  }, []);

  const handlePreviewAudioTimeUpdate = React.useCallback(() => {
    const audio = previewAudioRef.current;
    if (!audio) return;
    const nextIndex = findActiveLineIndexByTime(audio.currentTime);
    if (nextIndex == null) return;
    setActivePreviewLine((prev) => (prev === nextIndex ? prev : nextIndex));
  }, [findActiveLineIndexByTime]);

  const handlePreviewAudioEnded = React.useCallback(() => {
    setActivePreviewLine(null);
  }, []);

  React.useEffect(() => {
    if (activePreviewLine == null) return;
    const container = translationScrollRef.current;
    const row = translationLineRefs.current[activePreviewLine];
    if (!container || !row) return;

    const viewTop = container.scrollTop;
    const viewBottom = viewTop + container.clientHeight;
    const rowTop = row.offsetTop;
    const rowBottom = rowTop + row.offsetHeight;
    const padding = 24;
    const outOfView = rowTop < viewTop + padding || rowBottom > viewBottom - padding;
    if (outOfView) {
      row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }, [activePreviewLine]);

  const handlePreviewLineClick = React.useCallback(async (line: { original: string; index: number; startSeconds: number | null }) => {
    if (line.startSeconds == null || !previewAudioUrl || !previewAudioRef.current) return;
    const audio = previewAudioRef.current;
    try {
      const targetSrc = new URL(previewAudioUrl, window.location.origin).toString();
      if (!audio.src || audio.src !== targetSrc) {
        audio.src = previewAudioUrl;
      }
      audio.currentTime = Math.max(0, line.startSeconds);
      await audio.play();
      setActivePreviewLine(line.index);
    } catch (err) {
      console.error('Failed to play preview audio at timecode', err);
    }
  }, [previewAudioUrl]);

  const handleDownloadTranslation = (format: 'txt' | 'srt' | 'vtt') => {
    if (!project?.id) return;
    window.open(`/api/projects/${project.id}/translation/download?format=${format}`, '_blank');
  };

  const handleStartEditTranslatedOutput = React.useCallback(() => {
    if (isTranslating || translatedLines.length === 0) return;
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
    }
    setActivePreviewLine(null);
    const rows = subtitleRowsFromText(
      translatedLines
        .map((line) => String(line.translated || '').trim())
        .join('\n')
    );
    setEditingTranslatedOutputRows(rows);
    setEditingTranslatedOutputBaselineIds(rows.map((row) => row.id));
    setIsEditingTranslatedOutput(true);
  }, [isTranslating, translatedLines]);

  const handleSaveEditedTranslatedOutput = React.useCallback(async () => {
    if (!project || isSavingTranslatedOutput) return;
    const issues = validateSubtitleRows(editingTranslatedOutputRows);
    if (issues.length > 0) return;

    const nextTranslatedLines = subtitleRowsToLines(editingTranslatedOutputRows);
    const persistedTranslatedText = nextTranslatedLines.join('\n');
    const canSyncOriginalTimecodes = hasSameSubtitleRowOrder(
      editingTranslatedOutputBaselineIds,
      editingTranslatedOutputRows
    );
    const persistedOriginalText = canSyncOriginalTimecodes
      ? rebuildSourceWithTranslatedTimecodes(activeSourceText, editingTranslatedOutputRows)
      : activeSourceText;
    const syncedOriginalLines = subtitleRowsToLines(subtitleRowsFromText(persistedOriginalText));
    const currentTranslatedText = subtitleRowsToLines(
      subtitleRowsFromText(
        translatedLines
          .map((line) => String(line.translated || '').trim())
          .join('\n')
      )
    ).join('\n');

    if (persistedTranslatedText === currentTranslatedText) {
      setIsEditingTranslatedOutput(false);
      setEditingTranslatedOutputBaselineIds([]);
      return;
    }

    const maxLineCount = Math.max(syncedOriginalLines.length, nextTranslatedLines.length);
    const merged = Array.from({ length: maxLineCount }, (_, index) => ({
      original: syncedOriginalLines[index] || '',
      translated: nextTranslatedLines[index] || '',
    }));

    setIsSavingTranslatedOutput(true);
    setIsPersistingResult(true);
    try {
      const updateResult = await Promise.resolve(
        onUpdateProject({
          originalSubtitles: persistedOriginalText,
          translatedSubtitles: persistedTranslatedText,
          status: PROJECT_STATUS.COMPLETED,
        })
      );
      if (updateResult === null) {
        throw new Error('Failed to persist edited translated subtitles');
      }
      setTranslatedLines(merged);
      setHasTimecodes(hasStrictBracketTimecodes(persistedTranslatedText));
      setProgress(merged.length > 0 ? 100 : 0);
      setIsEditingTranslatedOutput(false);
      setEditingTranslatedOutputBaselineIds([]);
    } catch (err) {
      console.error('Failed to save edited translated subtitles', err);
      alert(t('settings.saveRetry'));
    } finally {
      setIsSavingTranslatedOutput(false);
      setIsPersistingResult(false);
    }
  }, [
    activeSourceText,
    editingTranslatedOutputBaselineIds,
    editingTranslatedOutputRows,
    isSavingTranslatedOutput,
    onUpdateProject,
    project,
    t,
    translatedLines,
  ]);

  return (
    <>
      <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex items-center gap-4 mb-1">
        <h2 className="text-3xl font-bold text-secondary tracking-tight">{t('translation.title')}</h2>
        {project && (
          <div className="px-3 py-1 bg-primary-container/10 border border-primary-container/20 rounded-full flex items-center gap-2 animate-in fade-in zoom-in duration-500">
            <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse shadow-[0_0_8px_rgba(var(--primary),0.8)]" />
            <span className="text-[10px] font-bold text-primary uppercase tracking-widest">{project.name}</span>
          </div>
        )}
      </div>
      <p className="text-outline mt-1">{t('translation.subtitle')}</p>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        <div className="xl:col-span-4 space-y-5">
          <section className="bg-surface-container p-6 rounded-2xl border border-white/5 space-y-5">
            <h3 className="text-sm font-bold text-primary uppercase tracking-widest">{t('translation.config')}</h3>

            <div>
              <label className="block text-xs font-bold text-outline mb-3 uppercase tracking-widest">{t('translation.sourceText')}</label>
              <div className="grid grid-cols-2 gap-3 p-1.5 bg-surface-container-lowest rounded-xl">
                <button
                  onClick={() => {
                    setSourceType('transcription');
                    setError(null);
                  }}
                  className={`py-3 px-4 text-sm font-bold rounded-lg flex items-center justify-center gap-2 transition-all ${
                    sourceType === 'transcription'
                      ? 'bg-primary-container text-white shadow-lg'
                      : 'text-outline hover:text-white hover:bg-white/8'
                  }`}
                >
                  <CheckCircle2 className="w-4 h-4" />
                  {t('translation.fromTranscription')}
                </button>
                <button
                  onClick={handleOpenProjectSourceModal}
                  className={`py-3 px-4 text-sm font-bold rounded-lg flex items-center justify-center gap-2 transition-all ${
                    sourceType === 'project'
                      ? 'bg-primary-container text-white shadow-lg'
                      : 'text-outline hover:text-white hover:bg-white/8'
                  }`}
                >
                  <FolderOpen className="w-4 h-4" />
                  {t('translation.selectFromProject')}
                </button>
              </div>
            </div>

            <div className="p-3.5 bg-surface-container-lowest rounded-xl border border-white/5 flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-primary-container/20 flex items-center justify-center shrink-0">
                {isLoadingSourceText ? <Loader2 className="w-5 h-5 text-primary animate-spin" /> : <FileText className="w-5 h-5 text-primary" />}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs text-outline font-medium mb-1">{t('translation.selectedText')}</p>
                <p className="text-sm text-secondary font-bold truncate">
                  {isLoadingSourceText ? t('common.loading') : selectedSourceDescription}
                </p>
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold text-outline mb-3 uppercase tracking-widest">{t('translation.modelSelection')}</label>
              <select
                value={selectedModelId}
                onChange={(e) => {
                  setSelectedModelId(e.target.value);
                  setModelLoadStatus('idle');
                  setModelLoadError(null);
                }}
                className="w-full bg-surface-container-high border border-white/10 rounded-xl px-5 py-3.5 text-sm text-white focus:ring-2 focus:ring-primary-container outline-none appearance-none"
              >
                {translateModels.length === 0 && <option value="">{t('stt.noModels')}</option>}
                {translateModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-bold text-outline mb-3 uppercase tracking-widest">{t('translation.targetLanguage')}</label>
              <select
                value={targetLang}
                onChange={(e) => setTargetLang(e.target.value)}
                className="w-full bg-surface-container-high border border-white/10 rounded-xl px-5 py-3.5 text-sm text-white focus:ring-2 focus:ring-primary-container outline-none appearance-none"
              >
                {TARGET_LANG_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
              {targetLang === 'other' && (
                <div className="mt-3 space-y-2">
                  <input
                    type="text"
                    value={customTargetLanguage}
                    onChange={(e) => setCustomTargetLanguage(e.target.value)}
                    className="w-full bg-surface-container-lowest border border-white/10 rounded-xl px-5 py-3 text-sm text-white placeholder:text-outline/30 focus:ring-2 focus:ring-primary-container outline-none"
                    placeholder={t('translation.customTargetPlaceholder')}
                  />
                  <p className="text-[11px] text-outline leading-relaxed">
                    {t('translation.customTargetHint')}
                  </p>
                </div>
              )}
            </div>

            <div className="space-y-3">
              <label className="block text-xs font-bold text-outline uppercase tracking-widest">{t('translation.promptLabel')}</label>
              <div className="rounded-xl border border-white/10 bg-surface-container-lowest/70 px-4 py-3 flex items-center justify-between gap-4">
                <div className="min-w-0">
                  <div className="text-[11px] font-bold text-secondary leading-5">{t('translation.strictJsonRepairToggle')}</div>
                  <div className="mt-1 text-[11px] leading-5 text-outline/72">{t('translation.strictJsonRepairHint')}</div>
                </div>
                <button
                  type="button"
                  onClick={() => setStrictJsonLineRepairEnabled((prev) => !prev)}
                  className="inline-flex shrink-0 items-center gap-2 text-[11px] text-outline hover:text-white transition-colors"
                  title={t('translation.strictJsonRepairToggle')}
                  aria-pressed={strictJsonLineRepairEnabled}
                >
                  <span
                    className={`w-9 h-5 rounded-full transition-colors ${strictJsonLineRepairEnabled ? 'bg-primary' : 'bg-white/20'}`}
                  >
                    <span
                      className={`block w-4 h-4 rounded-full bg-white mt-0.5 transition-transform ${strictJsonLineRepairEnabled ? 'translate-x-4' : 'translate-x-0.5'}`}
                    />
                  </span>
                </button>
              </div>
              <select
                value={translateGemmaPromptControlsDisabled ? '__translategemma_official__' : selectedPromptTemplateId}
                onChange={(e) => handleSelectPromptTemplate(e.target.value as PromptTemplateId)}
                disabled={translateGemmaPromptControlsDisabled}
                className="w-full bg-surface-container-high border border-white/10 rounded-xl px-4 py-3 text-sm text-white focus:ring-2 focus:ring-primary-container outline-none appearance-none disabled:opacity-55 disabled:cursor-not-allowed"
              >
                {translateGemmaPromptControlsDisabled ? (
                  <option value="__translategemma_official__">{t('translation.promptTemplateTranslateGemmaOfficial')}</option>
                ) : (
                  promptTemplateOptions.map((option) => (
                    <option key={option.id || 'none'} value={option.id}>
                      {option.label}
                    </option>
                  ))
                )}
              </select>
              <textarea
                value={translateGemmaPromptControlsDisabled ? '' : promptText}
                onChange={(e) => setPromptText(e.target.value)}
                disabled={translateGemmaPromptControlsDisabled}
                className="w-full h-32 bg-surface-container-lowest border border-white/10 rounded-xl px-5 py-4 text-sm text-secondary placeholder:text-outline/20 focus:ring-2 focus:ring-primary-container outline-none resize-none disabled:opacity-55 disabled:cursor-not-allowed"
                placeholder={
                  translateGemmaPromptControlsDisabled
                    ? t('translation.translateGemmaPromptDisabledPlaceholder')
                    : targetLang === 'other'
                    ? t('translation.promptPlaceholderOther')
                    : t('translation.promptPlaceholder')
                }
              />
              <div className="text-[11px] font-bold text-secondary">{t('translation.glossary')}</div>
              <textarea
                value={translateGemmaPromptControlsDisabled ? '' : glossaryText}
                onChange={(e) => setGlossaryText(e.target.value)}
                disabled={translateGemmaPromptControlsDisabled}
                className="w-full h-28 bg-surface-container-lowest border border-white/10 rounded-xl px-5 py-4 text-sm text-secondary placeholder:text-outline/20 focus:ring-2 focus:ring-primary-container outline-none resize-none disabled:opacity-55 disabled:cursor-not-allowed"
                placeholder={
                  translateGemmaPromptControlsDisabled
                    ? t('translation.translateGemmaGlossaryDisabledPlaceholder')
                    : t('translation.glossaryPlaceholder')
                }
              />
            </div>

            {error && <div className="text-xs text-error bg-error/10 border border-error/20 rounded-lg p-3">{error}</div>}

            <button
              onClick={isTranslating ? handleStopTranslation : handleStartTranslation}
              disabled={!isTranslating && (!canStartTranslation || isLoadingSourceText)}
              className={`w-full py-4 font-bold rounded-2xl border transition-all flex items-center justify-center gap-3 shadow-xl active:scale-[0.98] disabled:opacity-50 ${
                isTranslating
                  ? 'border-red-400/45 bg-gradient-to-r from-red-500 via-red-500 to-red-600 text-white shadow-[0_18px_45px_rgba(239,68,68,0.32)] hover:brightness-110'
                  : 'border-primary/25 bg-gradient-to-r from-primary-container to-primary text-white shadow-[0_18px_45px_rgba(99,102,241,0.22)] hover:brightness-110'
              }`}
            >
              {isTranslating ? <Square className="w-4 h-4" /> : <Play className="w-5 h-5" />}
              {isTranslating ? t('translation.stopTranslation') : t('translation.startTranslation')}
            </button>
          </section>

          <RunMonitor
            title={t('translation.statusMonitor')}
            isRunning={isTranslating}
            standbyLabel={t('common.standby')}
            statusLabel={translateMsg || t('translation.translating')}
            badges={translationMonitorBadges}
            progressItems={[
              {
                label: t('translation.modelLoad'),
                progress: modelLoadStatus === 'ok' ? 100 : modelLoadStatus === 'failed' ? 0 : modelLoadStatus === 'loading' ? 50 : 0,
                status:
                  modelLoadStatus === 'ok'
                    ? t('settings.testSuccess')
                    : modelLoadStatus === 'failed'
                      ? t('settings.testFailed')
                      : modelLoadStatus === 'loading'
                        ? t('settings.testing')
                        : t('common.standby'),
                tone: modelLoadStatus === 'ok' ? 'success' : modelLoadStatus === 'failed' ? 'error' : 'normal',
              },
              {
                label: t('translation.progress'),
                progress,
              },
            ]}
            message={translateMsg}
            error={modelLoadStatus === 'failed' ? modelLoadError : null}
            detailsTitle={t('monitor.diagnostics')}
            detailsSummary={debugSummary}
            showDetails={showDebugDetails}
            onToggleDetails={() => setShowDebugDetails((prev) => !prev)}
            sections={translationMonitorSections}
          />
        </div>

        <div className="xl:col-span-8 flex flex-col min-h-0 space-y-5">
          <div className="flex-1 h-[760px] max-h-[760px] min-h-0 bg-surface-container rounded-2xl overflow-hidden flex flex-col border border-white/5 shadow-2xl">
            <audio
              ref={previewAudioRef}
              src={previewAudioUrl || undefined}
              preload="auto"
              onTimeUpdate={handlePreviewAudioTimeUpdate}
              onEnded={handlePreviewAudioEnded}
              className="hidden"
            />
            <div className="flex flex-wrap items-center gap-2 border-b border-white/5 bg-white/[0.02] px-6 py-3">
              <span className="rounded-full border border-white/6 bg-white/[0.03] px-3 py-1.5 text-[11px] font-semibold text-outline/84">
                {t('translation.selectedText')}: {isLoadingSourceText ? t('common.loading') : selectedSourceDescription}
              </span>
              <span className="rounded-full border border-primary/15 bg-primary/8 px-3 py-1.5 text-[11px] font-semibold text-primary/90">
                {t('translation.targetLanguage')}: {langLabel}
              </span>
              {translatedLines.length > 0 && !isTranslating && (
                <div className="ml-auto flex items-center">
                  <button
                    type="button"
                    onClick={isEditingTranslatedOutput ? handleSaveEditedTranslatedOutput : handleStartEditTranslatedOutput}
                    disabled={isSavingTranslatedOutput || (isEditingTranslatedOutput && !canSaveEditedTranslatedOutput)}
                    className="flex items-center gap-2 rounded-lg border border-white/5 bg-white/5 px-3 py-1.5 text-[10px] font-bold uppercase tracking-widest text-outline transition-all hover:bg-white/10 hover:text-secondary disabled:opacity-60"
                  >
                    {isEditingTranslatedOutput
                      ? (isSavingTranslatedOutput ? t('settings.autoSaving') : t('dashboard.saveProject'))
                      : t('settings.edit')}
                  </button>
                </div>
              )}
            </div>
            <div className="grid grid-cols-2 bg-surface-container-high px-6 py-3.5 border-b border-white/5">
              <div className="text-[10px] font-bold text-outline uppercase tracking-widest">{t('translation.originalLanguage')}</div>
              <div className="text-[10px] font-bold text-outline uppercase tracking-widest pl-6 border-l border-white/5">
                {t('translation.translationResult').replace('{lang}', langLabel)}
              </div>
            </div>

            <div ref={translationScrollRef} className="flex-1 min-h-0 overflow-y-auto overscroll-contain custom-scrollbar divide-y divide-white/5">
              {isEditingTranslatedOutput ? (
                <div className="p-3">
                  <SubtitleRowsEditor
                    rows={editingTranslatedOutputRows}
                    issues={translatedOutputEditIssues}
                    onChangeRows={setEditingTranslatedOutputRows}
                    copy={subtitleEditorCopy}
                    referenceLines={translatedLines.map((line) => line.original)}
                  />
                </div>
              ) : (
                <>
                  {parsedTranslatedLines.map((line) => (
                    <TranslationRow
                      key={line.index}
                      row={line}
                      canSeekPlay={line.startSeconds != null && Boolean(previewAudioUrl)}
                      isActive={activePreviewLine === line.index}
                      onPreview={() => handlePreviewLineClick(line)}
                      onStopPreview={stopPreviewPlayback}
                      registerRow={(el) => { translationLineRefs.current[line.index] = el; }}
                    />
                  ))}

                  {isTranslating && translatedLines.length === 0 && (
                    <div className="p-6">
                      <div className="mx-auto max-w-4xl rounded-[28px] border border-primary/15 bg-primary/[0.04] px-6 py-6">
                        <div className="mb-5 flex justify-center">
                          <div className="relative h-20 w-20">
                            <div className="absolute inset-0 rounded-full bg-primary/10 blur-md animate-pulse" />
                            <div
                              className="absolute inset-1 rounded-full border border-primary/20 border-dashed animate-spin"
                              style={{ animationDuration: '9s' }}
                            />
                            <div className="absolute inset-[10px] rounded-[22px] border border-white/10 bg-surface-container-high flex items-center justify-center shadow-[0_12px_30px_rgba(79,70,229,0.18)]">
                              <img src="/logo.png" alt="ArcSub Gecko" className="h-10 w-10 animate-pulse select-none pointer-events-none" />
                            </div>
                          </div>
                        </div>
                        <div className="space-y-3">
                          {[
                            [68, 54],
                            [82, 77],
                            [59, 71],
                            [76, 63],
                            [51, 84],
                          ].map(([left, right], idx) => (
                            <div key={`${left}-${right}-${idx}`} className="grid grid-cols-2 divide-x divide-white/5 rounded-2xl overflow-hidden border border-white/5 bg-surface-container-lowest/75">
                              <div className="px-4 py-3.5">
                                <div className="h-3 rounded-full bg-white/6 overflow-hidden">
                                  <div
                                    className="h-full rounded-full bg-gradient-to-r from-white/5 via-white/20 to-white/5 animate-pulse"
                                    style={{ width: `${left}%`, animationDelay: `${idx * 120}ms` }}
                                  />
                                </div>
                              </div>
                              <div className="px-4 py-3.5">
                                <div className="h-3 rounded-full bg-primary/8 overflow-hidden">
                                  <div
                                    className="h-full rounded-full bg-gradient-to-r from-primary/10 via-primary/35 to-primary/10 animate-pulse"
                                    style={{ width: `${right}%`, animationDelay: `${idx * 170}ms` }}
                                  />
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {!isTranslating && translatedLines.length === 0 && (
                    <div className="p-16 text-center text-outline/30 italic">{t('translation.clickToStart')}</div>
                  )}
                </>
              )}
            </div>
          </div>

          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(320px,420px)] lg:items-center">
            {canDownloadTranslation && (
              <div className="flex flex-wrap gap-2 animate-in fade-in zoom-in duration-300">
                <button
                  onClick={() => handleDownloadTranslation('txt')}
                  disabled={isEditingTranslatedOutput}
                  className="flex items-center gap-2 px-3 py-1.5 bg-white/5 hover:bg-white/10 text-outline hover:text-secondary border border-white/5 rounded-lg text-[10px] font-bold transition-all uppercase tracking-widest"
                >
                  <Download className="w-3.5 h-3.5" />
                  TXT
                </button>
                {hasTimecodedTranslation && (
                  <>
                    <button
                      onClick={() => handleDownloadTranslation('srt')}
                      disabled={isEditingTranslatedOutput}
                      className="flex items-center gap-2 px-3 py-1.5 bg-white/5 hover:bg-white/10 text-outline hover:text-secondary border border-white/5 rounded-lg text-[10px] font-bold transition-all uppercase tracking-widest"
                    >
                      <Download className="w-3.5 h-3.5" />
                      SRT
                    </button>
                    <button
                      onClick={() => handleDownloadTranslation('vtt')}
                      disabled={isEditingTranslatedOutput}
                      className="flex items-center gap-2 px-3 py-1.5 bg-white/5 hover:bg-white/10 text-outline hover:text-secondary border border-white/5 rounded-lg text-[10px] font-bold transition-all uppercase tracking-widest"
                    >
                      <Download className="w-3.5 h-3.5" />
                      VTT
                    </button>
                  </>
                )}
              </div>
            )}
            <button
              onClick={handleNextStep}
              disabled={progress < 100 || isPersistingResult || isEditingTranslatedOutput || isSavingTranslatedOutput}
              className={`w-full py-4 font-bold rounded-2xl border transition-all flex items-center justify-center gap-3 group disabled:opacity-50 ${
                progress >= 100 && !isPersistingResult && !isEditingTranslatedOutput && !isSavingTranslatedOutput
                  ? 'border-tertiary/30 bg-gradient-to-r from-tertiary/85 via-primary/85 to-primary text-white shadow-[0_18px_45px_rgba(95,224,183,0.22)] hover:brightness-110'
                  : 'border-white/10 bg-white/5 text-secondary hover:bg-white/10'
              }`}
            >
              {nextStepLabel}
              <ArrowRight className={`w-5 h-5 transition-transform ${progress >= 100 && !isPersistingResult ? 'group-hover:translate-x-1' : ''}`} />
            </button>
          </div>
        </div>
      </div>
      </div>

      {showSourceModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200">
          <div className="bg-[#13151A] border border-white/10 rounded-[30px] w-full max-w-[1080px] shadow-2xl overflow-hidden flex flex-col max-h-[90vh] animate-in zoom-in-95 duration-200">
            <div className="flex items-start justify-between border-b border-white/5 bg-white/5 px-8 py-6">
              <div className="space-y-3">
                <div className="space-y-1.5">
                  <h2 className="text-[1.75rem] font-bold tracking-tight text-white">{t('translation.modalTitle')}</h2>
                  <p className="text-sm text-outline">{t('translation.modalSubtitle')}</p>
                  <p className="text-sm font-semibold text-primary">{project?.name}</p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full border border-white/8 bg-white/5 px-3 py-1.5 text-[11px] font-semibold text-outline/85">
                    {t('stt.totalFiles').replace('{count}', sourceAssets.length.toString())}
                  </span>
                  {project?.notes && (
                    <span className="rounded-full border border-primary/15 bg-primary/10 px-3 py-1.5 text-[11px] font-semibold text-primary/85">
                      {t('dashboard.projectNotes')}
                    </span>
                  )}
                </div>
              </div>
              <button onClick={() => setShowSourceModal(false)} className="rounded-xl p-2 text-outline transition-colors hover:bg-white/8 hover:text-white">
                <X className="h-6 w-6" />
              </button>
            </div>

            <div className="px-8 py-7 overflow-y-auto custom-scrollbar flex-1">
              <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_320px]">
                <div className="space-y-5">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <h3 className="text-sm font-semibold text-white/90">{t('translation.existingAssets')}</h3>
                      <p className="mt-1 text-xs leading-5 text-outline/72">
                        {sourceAssets.length === 0 ? t('dashboard.noMaterials') : t('dashboard.preview')}
                      </p>
                    </div>
                    <button
                      onClick={() => sourceFileInputRef.current?.click()}
                      className="inline-flex shrink-0 items-center gap-2 rounded-xl border border-primary/20 bg-primary/10 px-4 py-2 text-xs font-bold text-primary transition-all hover:bg-primary/15"
                    >
                      <Upload className="h-4 w-4" />
                      {t('translation.uploadNew')}
                    </button>
                  </div>

                  <div className="rounded-[24px] border border-white/6 bg-white/[0.02] min-h-[320px] relative overflow-hidden">
                    {isLoadingSourceAssets && (
                      <div className="absolute inset-0 bg-black/20 backdrop-blur-[1px] flex items-center justify-center z-10">
                        <Loader2 className="w-8 h-8 animate-spin text-primary" />
                      </div>
                    )}
                    <div className="p-4 space-y-3">
                      {sourceAssets.length === 0 && !isLoadingSourceAssets && (
                        <div className="px-8 py-14 text-center">
                          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                            <Upload className="h-6 w-6" />
                          </div>
                          <p className="mx-auto max-w-md text-sm leading-6 text-outline/65">{t('dashboard.noMaterials')}</p>
                        </div>
                      )}
                      {sourceAssets.map((asset) => {
                        const categoryText =
                          asset.category === 'subtitle'
                            ? t('dashboard.categorySubtitle')
                            : asset.category === 'audio'
                              ? t('dashboard.categoryAudio')
                              : asset.category === 'video'
                                ? t('dashboard.categoryVideo')
                                : t('dashboard.categoryOther');

                        return (
                          <label
                            key={asset.name}
                            className={`flex items-center justify-between gap-4 rounded-2xl border px-4 py-4 transition-all cursor-pointer ${
                              selectedSourceAssetName === asset.name
                                ? 'border-primary/30 bg-primary/8 shadow-[0_12px_30px_rgba(79,70,229,0.12)]'
                                : 'border-white/5 bg-surface-container-lowest hover:bg-white/5'
                            }`}
                          >
                            <div className="flex min-w-0 items-center gap-4 overflow-hidden">
                              <div className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl ${selectedSourceAssetName === asset.name ? 'bg-primary/15 text-primary' : 'bg-tertiary/10 text-tertiary'}`}>
                                <FileText className="w-5 h-5" />
                              </div>
                              <div className="min-w-0">
                                <div className="flex flex-wrap items-center gap-2">
                                  <span className="truncate text-sm font-bold text-secondary" title={asset.name}>{asset.name}</span>
                                  <span className="rounded-full border border-white/8 bg-white/[0.04] px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest text-outline/75">
                                    {categoryText}
                                  </span>
                                </div>
                                <div className="mt-1 flex flex-wrap gap-2 text-[10px] font-bold uppercase tracking-widest text-outline">
                                  <span className="rounded bg-white/5 px-1.5 py-0.5">{asset.size}</span>
                                  <span>{asset.date}</span>
                                </div>
                              </div>
                            </div>
                            <div className="flex shrink-0 items-center gap-3">
                              <div className={`w-6 h-6 rounded-full border-2 flex items-center justify-center transition-colors ${selectedSourceAssetName === asset.name ? 'border-primary bg-primary' : 'border-outline/50'}`}>
                                {selectedSourceAssetName === asset.name && <div className="w-3 h-3 bg-white rounded-full" />}
                              </div>
                            </div>
                            <input
                              type="radio"
                              name="translationSourceSelection"
                              className="hidden"
                              checked={selectedSourceAssetName === asset.name}
                              onChange={() => setSelectedSourceAssetName(asset.name)}
                            />
                          </label>
                        );
                      })}
                    </div>
                  </div>
                </div>

                <div className="space-y-5 lg:sticky lg:top-0">
                  <div className="space-y-4 rounded-[24px] border border-white/6 bg-white/[0.02] p-5">
                    <div>
                      <h4 className="text-sm font-semibold text-white/90">{t('translation.uploadNew')}</h4>
                      <p className="mt-1 text-xs leading-5 text-outline/72">{t('translation.onlyTextFiles')}</p>
                    </div>
                    <input
                      ref={sourceFileInputRef}
                      type="file"
                      accept=".txt,.srt,.vtt,.ass,.ssa,.json,text/plain,application/json"
                      className="hidden"
                      onChange={(e) => {
                        void handleSourceFileUpload(e.target.files?.[0] || null);
                      }}
                    />
                    <div
                      onClick={() => sourceFileInputRef.current?.click()}
                      onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add('border-primary/50', 'bg-primary/5'); }}
                      onDragLeave={(e) => { e.preventDefault(); e.currentTarget.classList.remove('border-primary/50', 'bg-primary/5'); }}
                      onDrop={(e) => { e.currentTarget.classList.remove('border-primary/50', 'bg-primary/5'); void handleSourceFileDrop(e); }}
                      className={`border-2 border-dashed border-white/10 rounded-[24px] px-6 py-10 flex flex-col items-center justify-center text-center hover:border-primary/50 hover:bg-primary/5 transition-all cursor-pointer group relative ${isUploadingSource ? 'pointer-events-none opacity-50' : ''}`}
                    >
                      {isUploadingSource ? (
                        <div className="flex flex-col items-center">
                          <Loader2 className="w-10 h-10 text-primary animate-spin mb-4" />
                          <p className="text-sm font-bold text-primary">{t('dashboard.uploading')}</p>
                        </div>
                      ) : (
                        <>
                          <div className="w-14 h-14 bg-primary-container/20 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                            <Upload className="w-7 h-7 text-primary" />
                          </div>
                          <p className="text-sm font-bold text-secondary leading-6">
                            {t('translation.dragDrop')}
                          </p>
                          <p className="mt-3 text-[10px] font-bold uppercase tracking-widest text-outline">
                            {t('translation.supportedFormats')}
                          </p>
                        </>
                      )}
                    </div>
                  </div>

                  <div className="rounded-[24px] border border-white/6 bg-surface-container-lowest/70 p-5">
                    <div className="text-sm font-semibold text-white/90">{t('dashboard.projectNotes')}</div>
                    <div className="mt-3 rounded-2xl border border-white/5 bg-white/[0.02] px-4 py-4 text-sm leading-6 text-outline/80">
                      {project?.notes ? (
                        <div className="max-h-[180px] overflow-y-auto custom-scrollbar">{project.notes}</div>
                      ) : (
                        t('dashboard.noProjectDescription')
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="p-6 border-t border-white/5 bg-[#13151A] flex justify-end gap-4">
              <button
                onClick={() => setShowSourceModal(false)}
                className="px-6 py-3 text-sm font-bold text-outline transition-colors hover:bg-white/8 hover:text-white rounded-lg"
              >
                {t('translation.cancel')}
              </button>
              <button
                onClick={handleConfirmProjectSource}
                disabled={!selectedSourceAssetName || isLoadingSourceText}
                className="px-8 py-3 bg-primary hover:bg-primary/90 text-white text-sm font-bold rounded-lg transition-colors shadow-lg shadow-primary/20 disabled:opacity-50 flex items-center gap-2"
              >
                {isLoadingSourceText && <Loader2 className="w-4 h-4 animate-spin" />}
                {t('translation.confirm')}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

const TranslationRow: React.FC<{
  row: {
    index: number;
    original: string;
    translated: string;
    timecode: string | null;
    startSeconds: number | null;
  };
  canSeekPlay: boolean;
  isActive: boolean;
  onPreview: () => void;
  onStopPreview: () => void;
  registerRow: (el: HTMLDivElement | null) => void;
}> = ({ row, canSeekPlay, isActive, onPreview, onStopPreview, registerRow }) => {
  return (
    <div
      ref={registerRow}
      className={`group relative grid grid-cols-2 transition-colors ${
        isActive ? 'bg-primary/8 ring-1 ring-inset ring-primary/20' : 'hover:bg-white/5'
      }`}
    >
      <button
        type="button"
        onClick={onPreview}
        disabled={!canSeekPlay}
        className={`px-6 py-5 text-left text-sm text-outline/80 leading-relaxed transition-colors ${
          canSeekPlay ? 'cursor-pointer hover:text-secondary' : 'cursor-default'
        }`}
      >
        {row.original}
      </button>
      <button
        type="button"
        onClick={onPreview}
        disabled={!canSeekPlay}
        className={`border-l border-white/5 bg-primary/5 px-6 py-5 text-left text-sm text-primary leading-relaxed transition-colors ${
          canSeekPlay ? 'cursor-pointer hover:brightness-110' : 'cursor-default'
        }`}
      >
        {row.translated}
      </button>
      {isActive && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onStopPreview();
          }}
          className="absolute right-3 top-3 inline-flex items-center gap-1 rounded-md border border-white/10 bg-surface-container-high px-2 py-1 text-[10px] font-bold text-secondary hover:bg-white/10 opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto focus:opacity-100 focus:pointer-events-auto transition-opacity"
        >
          <Square className="w-3 h-3" />
        </button>
      )}
    </div>
  );
};


