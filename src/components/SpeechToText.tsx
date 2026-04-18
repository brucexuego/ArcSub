import React from 'react';
import { Mic, Upload, FolderOpen, Play, Download, CheckCircle2, Loader2, ArrowRight, Music, Square, ChevronDown, ChevronUp, X } from 'lucide-react';
import { Project } from '../types';
import { useLanguage } from '../i18n/LanguageContext';
import type { Language } from '../i18n/translations';
import { sanitizeInput } from '../utils/security';
import { PROJECT_STATUS } from '../project_status';
import { getJson, HttpRequestError, postJson } from '../utils/http_client';
import SubtitleRowsEditor from './SubtitleRowsEditor';
import {
  EditableSubtitleRow,
  subtitleRowsFromText,
  subtitleRowsToLines,
  validateSubtitleRows,
} from '../utils/subtitle_editor';

interface SpeechToTextProps {
  project: Project | null;
  onUpdateProject: (updates: Partial<Project>) => void | Promise<Project | null>;
  onNext: () => void;
  onBack: () => void;
  onTaskLockChange?: (locked: boolean) => void;
}

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
  };
}

interface AlignmentDiagnostics {
  applied: boolean;
  modelId: string;
  language: string | null;
  attemptedSegmentCount: number;
  alignedSegmentCount: number;
  skippedSegments: number;
  failureCount: number;
  alignedWordCount: number;
  avgConfidence: number | null;
  elapsedMs: number;
}

interface CjkWordDiagnostics {
  rawWordCount: number;
  mergedWordCount: number;
  droppedWordCount: number;
  replacementCharCount: number;
  rawSingleCharCount: number;
  mergedSingleCharCount: number;
  punctuationOnlyCount: number;
  lexicalWordCount?: number;
  lexicalSingleCharCount?: number;
  lexicalMismatchCount?: number;
  lexicalProjectionAppliedSegments?: number;
  splitSegmentCount?: number;
  usedIntlSegmenter?: boolean;
  segmenterLocale?: string;
  mergeApplied: boolean;
  chunkSource: string;
}

interface PyannoteSetupStatus {
  tokenConfigured: boolean;
  ready: boolean;
  state: 'ready' | 'partial' | 'missing';
  installing: boolean;
  lastError: string | null;
}

function getDiarizationCopy(language: Language) {
  const maps = {
    'zh-tw': {
      modeLabel: '語者模式',
      providerLabel: '分離引擎',
      sceneLabel: '場景預設',
      advancedLabel: '進階約束',
      providerClassic: '經典聲學分離',
      providerPyannote: 'Pyannote',
      auto: '自動',
      fixed: '固定人數',
      range: '範圍人數',
      many: '多人模式',
      interview: '訪談 / 對談',
      podcast: 'Podcast',
      meeting: '會議',
      presentationQa: '演講 + Q&A',
      custom: '其他 / 自訂',
      exactCount: '固定語者數',
      minSpeakers: '最少語者數',
      maxSpeakers: '最多語者數',
      scenePresetHint: '選擇場景預設後，會套用並鎖定對應的語者模式與約束。',
      customSceneHint: '切換到其他 / 自訂後，才可手動調整語者模式與數量。',
      classicOnlyHint: '下列進階約束目前只作用於經典聲學分離；選擇 Pyannote 時會停用。',
      preferStablePrimary: '偏好主講者穩定',
      allowShortInterjection: '允許短暫插話獨立成新語者',
      preferVadBounded: '優先使用 VAD 收緊語音區段',
      forceMergeTiny: '雙人模式強制合併小群',
      semanticFallback: '低信心時改用語義分離',
      diagnostics: '語者分離診斷',
      provider: '執行方式',
      source: '實際來源',
      speechSegments: '語音段數',
      windows: 'VAD 視窗',
      speakers: '輸出語者數',
      regions: '聲紋區段',
      threshold: '分群閾值',
      providerAcoustic: '聲學分離',
      providerSemantic: '語義分離',
      sourceVad: 'VAD 收緊片段',
      sourceSpeechRegion: '語音區段',
      sourceChunk: '字幕區段',
      sourcePyannote: 'Pyannote turn segments',
      sourceSemantic: '語義回退',
    },
    'zh-cn': {
      modeLabel: '说话人模式',
      providerLabel: '分离引擎',
      sceneLabel: '场景预设',
      advancedLabel: '高级约束',
      providerClassic: '经典声学分离',
      providerPyannote: 'Pyannote',
      auto: '自动',
      fixed: '固定人数',
      range: '范围人数',
      many: '多人模式',
      interview: '访谈 / 对谈',
      podcast: 'Podcast',
      meeting: '会议',
      presentationQa: '演讲 + Q&A',
      custom: '其他 / 自定义',
      exactCount: '固定说话人数',
      minSpeakers: '最少说话人数',
      maxSpeakers: '最多说话人数',
      scenePresetHint: '选择场景预设后，会套用并锁定对应的说话人模式与约束。',
      customSceneHint: '切换到其他 / 自定义后，才可手动调整说话人模式与数量。',
      classicOnlyHint: '下列高级约束目前只作用于经典声学分离；选择 Pyannote 时会停用。',
      preferStablePrimary: '偏好主讲者稳定',
      allowShortInterjection: '允许短暂插话独立成新说话人',
      preferVadBounded: '优先使用 VAD 收紧语音区段',
      forceMergeTiny: '双人模式强制合并小群',
      semanticFallback: '低置信度时改用语义分离',
      diagnostics: '语者分离诊断',
      provider: '执行方式',
      source: '实际来源',
      speechSegments: '语音段数',
      windows: 'VAD 窗口',
      speakers: '输出说话人数',
      regions: '声纹区段',
      threshold: '聚类阈值',
      providerAcoustic: '声学分离',
      providerSemantic: '语义分离',
      sourceVad: 'VAD 收紧片段',
      sourceSpeechRegion: '语音区段',
      sourceChunk: '字幕区段',
      sourcePyannote: 'Pyannote turn segments',
      sourceSemantic: '语义回退',
    },
    en: {
      modeLabel: 'Speaker Mode',
      providerLabel: 'Engine',
      sceneLabel: 'Scene Preset',
      advancedLabel: 'Advanced Constraints',
      providerClassic: 'Classic Acoustic',
      providerPyannote: 'Pyannote',
      auto: 'Auto',
      fixed: 'Fixed Count',
      range: 'Range',
      many: 'Many Speakers',
      interview: 'Interview / Dialogue',
      podcast: 'Podcast',
      meeting: 'Meeting',
      presentationQa: 'Presentation + Q&A',
      custom: 'Other / Custom',
      exactCount: 'Exact Speaker Count',
      minSpeakers: 'Min Speakers',
      maxSpeakers: 'Max Speakers',
      scenePresetHint: 'Selecting a scene preset applies and locks its speaker mode and constraints.',
      customSceneHint: 'Switch to Other / Custom to edit speaker mode and count manually.',
      classicOnlyHint: 'The advanced constraints below currently affect only Classic Acoustic; they are disabled for Pyannote.',
      preferStablePrimary: 'Prefer stable primary speaker',
      allowShortInterjection: 'Allow short interjections as separate speakers',
      preferVadBounded: 'Prefer VAD-bounded speech regions',
      forceMergeTiny: 'Force tiny-cluster merge in 2-speaker mode',
      semanticFallback: 'Use semantic fallback on low confidence',
      diagnostics: 'Diarization Diagnostics',
      provider: 'Provider',
      source: 'Selected Source',
      speechSegments: 'Speech Segments',
      windows: 'VAD Windows',
      speakers: 'Output Speakers',
      regions: 'Embedding Regions',
      threshold: 'Cluster Threshold',
      providerAcoustic: 'Acoustic',
      providerSemantic: 'Semantic',
      sourceVad: 'VAD-bounded',
      sourceSpeechRegion: 'Speech regions',
      sourceChunk: 'Chunk-based',
      sourcePyannote: 'Pyannote turn segments',
      sourceSemantic: 'Semantic fallback',
    },
    jp: {
      modeLabel: '話者モード',
      providerLabel: '分離エンジン',
      sceneLabel: 'シーンプリセット',
      advancedLabel: '詳細制約',
      providerClassic: '従来の音響分離',
      providerPyannote: 'Pyannote',
      auto: '自動',
      fixed: '固定人数',
      range: '人数レンジ',
      many: '多人数モード',
      interview: '対談 / インタビュー',
      podcast: 'Podcast',
      meeting: '会議',
      presentationQa: '講演 + Q&A',
      custom: 'その他 / カスタム',
      exactCount: '固定話者数',
      minSpeakers: '最小話者数',
      maxSpeakers: '最大話者数',
      scenePresetHint: 'シーンプリセットを選ぶと、対応する話者モードと制約が適用され固定されます。',
      customSceneHint: '話者モードと人数を手動で調整するには、その他 / カスタムに切り替えてください。',
      classicOnlyHint: '以下の詳細制約は現在、従来の音響分離にのみ作用します。Pyannote では無効になります。',
      preferStablePrimary: '主話者の安定性を優先',
      allowShortInterjection: '短い割り込みを別話者として許可',
      preferVadBounded: 'VADで絞った音声区間を優先',
      forceMergeTiny: '2人モードで小クラスタを強制統合',
      semanticFallback: '低信頼時は意味ベースへフォールバック',
      diagnostics: '話者分離診断',
      provider: '方式',
      source: '実際の入力',
      speechSegments: '音声区間数',
      windows: 'VADウィンドウ',
      speakers: '出力話者数',
      regions: '埋め込み区間',
      threshold: 'クラスタ閾値',
      providerAcoustic: '音響分離',
      providerSemantic: '意味分離',
      sourceVad: 'VAD収束区間',
      sourceSpeechRegion: '音声区間',
      sourceChunk: '字幕チャンク',
      sourcePyannote: 'Pyannote turn segments',
      sourceSemantic: '意味フォールバック',
    },
    de: {
      modeLabel: 'Sprechermodus',
      providerLabel: 'Engine',
      sceneLabel: 'Szenenvorgabe',
      advancedLabel: 'Erweiterte Regeln',
      providerClassic: 'Klassisch akustisch',
      providerPyannote: 'Pyannote',
      auto: 'Automatisch',
      fixed: 'Feste Anzahl',
      range: 'Bereich',
      many: 'Viele Sprecher',
      interview: 'Interview / Dialog',
      podcast: 'Podcast',
      meeting: 'Meeting',
      presentationQa: 'Vortrag + Q&A',
      custom: 'Andere / Benutzerdefiniert',
      exactCount: 'Exakte Sprecherzahl',
      minSpeakers: 'Min. Sprecher',
      maxSpeakers: 'Max. Sprecher',
      scenePresetHint: 'Eine Szenenvorgabe uebernimmt und sperrt den passenden Sprechermodus samt Regeln.',
      customSceneHint: 'Wechseln Sie zu Andere / Benutzerdefiniert, um Sprechermodus und Anzahl manuell zu aendern.',
      classicOnlyHint: 'Die folgenden erweiterten Regeln wirken derzeit nur auf Klassisch akustisch; fuer Pyannote sind sie deaktiviert.',
      preferStablePrimary: 'Stabilen Hauptsprecher bevorzugen',
      allowShortInterjection: 'Kurze Einwürfe als eigene Sprecher erlauben',
      preferVadBounded: 'VAD-begrenzte Sprachsegmente bevorzugen',
      forceMergeTiny: 'Tiny-Cluster im 2-Sprecher-Modus zusammenführen',
      semanticFallback: 'Bei niedriger Sicherheit semantisch fallbacken',
      diagnostics: 'Diarisierungsdiagnose',
      provider: 'Verfahren',
      source: 'Gewählte Quelle',
      speechSegments: 'Sprachsegmente',
      windows: 'VAD-Fenster',
      speakers: 'Ausgegebene Sprecher',
      regions: 'Embedding-Segmente',
      threshold: 'Cluster-Schwelle',
      providerAcoustic: 'Akustisch',
      providerSemantic: 'Semantisch',
      sourceVad: 'VAD-begrenzt',
      sourceSpeechRegion: 'Sprachsegmente',
      sourceChunk: 'Chunk-basiert',
      sourcePyannote: 'Pyannote turn segments',
      sourceSemantic: 'Semantischer Fallback',
    },
  } as const;

  return maps[language];
}

function getPyannoteSetupCopy(language: Language) {
  const maps = {
    'zh-tw': {
      requiresTokenHint: 'Pyannote 需要 Hugging Face 金鑰與模型授權。選取後會要求輸入 HF_TOKEN。',
      autoInstallHint: '已偵測到 HF_TOKEN。選取 Pyannote 後會自動開始安裝所需資產。',
      promptTitle: '請輸入 Hugging Face HF_TOKEN。\n先在 Hugging Face 接受 pyannote 模型授權後再輸入。',
      installStarting: '正在安裝 Pyannote 資產，請稍候。',
      installFailed: 'Pyannote 安裝失敗。',
      retryQuestion: '要重新輸入 HF_TOKEN 再試一次嗎？',
      skipped: '已取消 Pyannote 安裝，將維持使用經典引擎。',
      installIncomplete: 'Pyannote 安裝尚未完成。',
    },
    'zh-cn': {
      requiresTokenHint: 'Pyannote 需要 Hugging Face 金钥与模型授权。选择后会要求输入 HF_TOKEN。',
      autoInstallHint: '已检测到 HF_TOKEN。选择 Pyannote 后会自动开始安装所需资源。',
      promptTitle: '请输入 Hugging Face HF_TOKEN。\n请先在 Hugging Face 接受 pyannote 模型授权。',
      installStarting: '正在安装 Pyannote 资源，请稍候。',
      installFailed: 'Pyannote 安装失败。',
      retryQuestion: '要重新输入 HF_TOKEN 再试一次吗？',
      skipped: '已取消 Pyannote 安装，将继续使用经典引擎。',
      installIncomplete: 'Pyannote 安装尚未完成。',
    },
    en: {
      requiresTokenHint: 'Pyannote requires a Hugging Face token and accepted model access. Selecting it will prompt for HF_TOKEN.',
      autoInstallHint: 'An HF_TOKEN is already configured. Selecting Pyannote will automatically install the required assets.',
      promptTitle: 'Enter your Hugging Face HF_TOKEN.\nMake sure you have already accepted the pyannote model access terms on Hugging Face.',
      installStarting: 'Installing Pyannote assets. Please wait.',
      installFailed: 'Pyannote installation failed.',
      retryQuestion: 'Do you want to retry with another HF_TOKEN?',
      skipped: 'Pyannote installation was skipped. Classic diarization will remain selected.',
      installIncomplete: 'Pyannote installation is not complete yet.',
    },
    jp: {
      requiresTokenHint: 'Pyannote には Hugging Face のトークンとモデル利用承認が必要です。選択時に HF_TOKEN の入力を求めます。',
      autoInstallHint: 'HF_TOKEN が設定されています。Pyannote を選ぶと必要なアセットを自動でインストールします。',
      promptTitle: 'Hugging Face の HF_TOKEN を入力してください。\n先に Hugging Face 上で pyannote モデルの利用承認を済ませてください。',
      installStarting: 'Pyannote アセットをインストールしています。しばらくお待ちください。',
      installFailed: 'Pyannote のインストールに失敗しました。',
      retryQuestion: '別の HF_TOKEN で再試行しますか？',
      skipped: 'Pyannote のインストールをスキップしました。従来エンジンを使います。',
      installIncomplete: 'Pyannote のインストールがまだ完了していません。',
    },
    de: {
      requiresTokenHint: 'Pyannote benoetigt ein Hugging-Face-Token und bestaetigten Modellzugriff. Beim Auswaehlen wird nach HF_TOKEN gefragt.',
      autoInstallHint: 'Ein HF_TOKEN ist bereits konfiguriert. Beim Auswaehlen von Pyannote werden die benoetigten Assets automatisch installiert.',
      promptTitle: 'Bitte geben Sie Ihr Hugging Face HF_TOKEN ein.\nAkzeptieren Sie zuvor die pyannote-Modellbedingungen auf Hugging Face.',
      installStarting: 'Pyannote-Assets werden installiert. Bitte warten.',
      installFailed: 'Die Pyannote-Installation ist fehlgeschlagen.',
      retryQuestion: 'Moechten Sie es mit einem anderen HF_TOKEN erneut versuchen?',
      skipped: 'Die Pyannote-Installation wurde uebersprungen. Die klassische Engine bleibt aktiv.',
      installIncomplete: 'Die Pyannote-Installation ist noch nicht abgeschlossen.',
    },
  } as const;

  return maps[language];
}

// Mock removed in favor of real API

export default function SpeechToText({ project, onUpdateProject, onNext, onBack, onTaskLockChange }: SpeechToTextProps) {
  const { t, language } = useLanguage();
  const diarizationCopy = React.useMemo(() => getDiarizationCopy(language), [language]);
  const pyannoteSetupCopy = React.useMemo(() => getPyannoteSetupCopy(language), [language]);
  const [isTranscribing, setIsTranscribing] = React.useState(false);
  const [progress, setProgress] = React.useState(0);
  const [transcription, setTranscription] = React.useState<string[]>([]);
  const [activePreviewLine, setActivePreviewLine] = React.useState<number | null>(null);
  const [hasTimecodes, setHasTimecodes] = React.useState(false);
  const [lastTranscribedAssetName, setLastTranscribedAssetName] = React.useState<string | null>(null);
  const [pipelineMode, setPipelineMode] = React.useState<string | null>(null);
  const [pipelineWarnings, setPipelineWarnings] = React.useState<string[]>([]);
  const [providerDebug, setProviderDebug] = React.useState<any | null>(null);
  const [appliedDebug, setAppliedDebug] = React.useState<any | null>(null);
  const [showAssetModal, setShowAssetModal] = React.useState(false);
  const [sourceType, setSourceType] = React.useState<'online' | 'project'>('online');
  const [assets, setAssets] = React.useState<any[]>([]);
  const [isLoadingAssets, setIsLoadingAssets] = React.useState(false);
  const [isUploading, setIsUploading] = React.useState(false);
  const [selectedAssetName, setSelectedAssetName] = React.useState<string | null>(null);
  const [prompt, setPrompt] = React.useState('');
  const [segmentation, setSegmentation] = React.useState(true);
  const [wordAlignment, setWordAlignment] = React.useState(true);
  const [vad, setVad] = React.useState(false);
  const [diarization, setDiarization] = React.useState(false);
  const [diarizationProvider, setDiarizationProvider] = React.useState<DiarizationProvider>('classic');
  const [diarizationMode, setDiarizationMode] = React.useState<DiarizationMode>('auto');
  const [diarizationScenePreset, setDiarizationScenePreset] = React.useState<DiarizationScenePreset>('interview');
  const [exactSpeakerCount, setExactSpeakerCount] = React.useState(2);
  const [minSpeakers, setMinSpeakers] = React.useState(2);
  const [maxSpeakers, setMaxSpeakers] = React.useState(4);
  const [preferStablePrimarySpeaker, setPreferStablePrimarySpeaker] = React.useState(true);
  const [allowShortInterjectionSpeaker, setAllowShortInterjectionSpeaker] = React.useState(false);
  const [preferVadBoundedRegions, setPreferVadBoundedRegions] = React.useState(true);
  const [forceMergeTinyClustersInTwoSpeakerMode, setForceMergeTinyClustersInTwoSpeakerMode] = React.useState(true);
  const [semanticFallbackEnabled, setSemanticFallbackEnabled] = React.useState(true);
  const [diarizationDiagnostics, setDiarizationDiagnostics] = React.useState<DiarizationDiagnostics | null>(null);
  const [alignmentDiagnostics, setAlignmentDiagnostics] = React.useState<AlignmentDiagnostics | null>(null);
  const [cjkWordDiagnostics, setCjkWordDiagnostics] = React.useState<CjkWordDiagnostics | null>(null);
  const [lastElapsedMs, setLastElapsedMs] = React.useState<number | null>(null);
  const [lastWordAlignmentState, setLastWordAlignmentState] = React.useState<'on' | 'off' | 'unavailable' | null>(null);
  const [showAdvanced, setShowAdvanced] = React.useState(false);
  const [showStatusDetails, setShowStatusDetails] = React.useState(false);
  const [isEditingTranscription, setIsEditingTranscription] = React.useState(false);
  const [editingTranscriptionRows, setEditingTranscriptionRows] = React.useState<EditableSubtitleRow[]>([]);
  const [isSavingTranscription, setIsSavingTranscription] = React.useState(false);
  const isCustomDiarizationScene = diarizationScenePreset === 'custom';
  const isPyannoteDiarization = diarizationProvider === 'pyannote';
  const areSceneManagedSpeakerControlsLocked = diarization && !isCustomDiarizationScene;
  const areClassicOnlyConstraintsDisabled = diarization && (isPyannoteDiarization || !isCustomDiarizationScene);

  const applyScenePreset = React.useCallback((preset: DiarizationScenePreset) => {
    setDiarizationScenePreset(preset);
    if (preset === 'custom') return;
    if (preset === 'interview') {
      setDiarizationMode('fixed');
      setExactSpeakerCount(2);
      setMinSpeakers(2);
      setMaxSpeakers(2);
      setPreferStablePrimarySpeaker(true);
      setAllowShortInterjectionSpeaker(false);
      setPreferVadBoundedRegions(true);
      setForceMergeTinyClustersInTwoSpeakerMode(true);
      setSemanticFallbackEnabled(true);
      return;
    }
    if (preset === 'podcast') {
      setDiarizationMode('range');
      setExactSpeakerCount(2);
      setMinSpeakers(2);
      setMaxSpeakers(4);
      setPreferStablePrimarySpeaker(true);
      setAllowShortInterjectionSpeaker(false);
      setPreferVadBoundedRegions(true);
      setForceMergeTinyClustersInTwoSpeakerMode(false);
      setSemanticFallbackEnabled(true);
      return;
    }
    if (preset === 'meeting') {
      setDiarizationMode('many');
      setMinSpeakers(2);
      setMaxSpeakers(8);
      setPreferStablePrimarySpeaker(false);
      setAllowShortInterjectionSpeaker(true);
      setPreferVadBoundedRegions(false);
      setForceMergeTinyClustersInTwoSpeakerMode(false);
      setSemanticFallbackEnabled(true);
      return;
    }
    if (preset === 'presentation_qa') {
      setDiarizationMode('range');
      setMinSpeakers(2);
      setMaxSpeakers(4);
      setPreferStablePrimarySpeaker(true);
      setAllowShortInterjectionSpeaker(true);
      setPreferVadBoundedRegions(true);
      setForceMergeTinyClustersInTwoSpeakerMode(false);
      setSemanticFallbackEnabled(true);
    }
  }, []);

  // Recommendation logic: Diarization requires VAD
  const handleDiarizationChange = (checked: boolean) => {
    setDiarization(checked);
    if (checked) {
      setVad(true); // Auto-enable VAD
      setSegmentation(true); // Keep timestamps for diarization alignment
      applyScenePreset(diarizationScenePreset);
    }
  };

  const handleSegmentationChange = (checked: boolean) => {
    setSegmentation(checked);
    if (!checked) {
      setWordAlignment(false);
    }
  };

  const handleWordAlignmentChange = (checked: boolean) => {
    setWordAlignment(checked);
    if (checked) {
      setSegmentation(true);
    }
  };

  const [asrModels, setAsrModels] = React.useState<any[]>([]);
  const [selectedModelId, setSelectedModelId] = React.useState<string>('');
  const [selectedLanguage, setSelectedLanguage] = React.useState<string>('auto');
  const [modelLoadStatus, setModelLoadStatus] = React.useState<'idle' | 'loading' | 'ok' | 'failed'>('idle');
  const [modelLoadError, setModelLoadError] = React.useState<string | null>(null);
  const [pyannoteStatus, setPyannoteStatus] = React.useState<PyannoteSetupStatus | null>(null);
  const [isPreparingPyannote, setIsPreparingPyannote] = React.useState(false);
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const previewAudioRef = React.useRef<HTMLAudioElement | null>(null);
  const transcriptionScrollRef = React.useRef<HTMLDivElement | null>(null);
  const transcriptLineRefs = React.useRef<Record<number, HTMLDivElement | null>>({});
  const transcribeEventSourceRef = React.useRef<EventSource | null>(null);

  React.useEffect(() => {
    // Reset selection and output when switching project
    if (project?.id) {
      setSelectedAssetName(null);
      setTranscription([]);
      setProgress(0);
      setAsrMsg(null);
      setActivePreviewLine(null);
      setHasTimecodes(false);
      setLastTranscribedAssetName(null);
      setPipelineMode(null);
      setPipelineWarnings([]);
      setProviderDebug(null);
      setAppliedDebug(null);
      setDiarizationDiagnostics(null);
      setAlignmentDiagnostics(null);
      setCjkWordDiagnostics(null);
      setLastElapsedMs(null);
      setLastWordAlignmentState(null);
      setIsEditingTranscription(false);
      setEditingTranscriptionRows([]);
    }
  }, [project?.id]);

  const previewAudioUrl = React.useMemo(() => {
    if (!project?.id) return '';
    if (lastTranscribedAssetName) {
      return `/Projects/${project.id}/assets/${encodeURIComponent(lastTranscribedAssetName)}`;
    }
    const fallbackAudio = String(project?.audioUrl || '').trim();
    if (fallbackAudio) return fallbackAudio;
    const fallbackVideo = String(project?.videoUrl || '').trim();
    return fallbackVideo;
  }, [project?.id, project?.audioUrl, project?.videoUrl, lastTranscribedAssetName]);

  React.useEffect(() => {
    setActivePreviewLine(null);
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
    }
  }, [previewAudioUrl]);

  React.useEffect(() => {
    fetch('/api/runtime-models')
      .then(res => res.json())
      .then(data => {
        if (data.asrModels && data.asrModels.length > 0) {
          setAsrModels(data.asrModels);
          // If the currently selected model isn't in the new list, select the first one
          setSelectedModelId((prev) => (
            data.asrModels.find((m: any) => m.id === prev) ? prev : data.asrModels[0].id
          ));
          setModelLoadStatus('idle');
          setModelLoadError(null);
        } else {
          setModelLoadStatus('failed');
          setModelLoadError(t('stt.noModels'));
        }
      })
      .catch(err => {
        console.error('Failed to load ASR models', err);
        setModelLoadStatus('failed');
        setModelLoadError(t('settings.testFailed'));
      });
  }, [t]);

  const loadPyannoteStatus = React.useCallback(async () => {
    try {
      const status = await getJson<PyannoteSetupStatus>('/api/runtime/pyannote/status', {
        dedupeKey: 'stt:pyannote-status',
        cancelPreviousKey: 'stt:pyannote-status',
      });
      setPyannoteStatus(status);
      return status;
    } catch (error) {
      console.error('Failed to load pyannote status', error);
      return null;
    }
  }, []);

  React.useEffect(() => {
    void loadPyannoteStatus();
  }, [loadPyannoteStatus]);

  const getRequestErrorMessage = React.useCallback((error: unknown) => {
    if (error instanceof HttpRequestError) {
      try {
        const parsed = JSON.parse(error.bodyText || '{}') as { error?: string };
        if (parsed?.error) return parsed.error;
      } catch {
        if (error.bodyText) return error.bodyText;
      }
      return error.message;
    }
    return error instanceof Error ? error.message : String(error || 'Unknown error');
  }, []);

  const installPyannoteAssets = React.useCallback(
    async (token?: string) => {
      setIsPreparingPyannote(true);
      try {
        const response = await postJson<{ success?: boolean; status?: PyannoteSetupStatus }>(
          '/api/runtime/pyannote/install',
          token ? { token } : {},
          { timeoutMs: 900_000 }
        );
        const nextStatus = response?.status || (await loadPyannoteStatus());
        setPyannoteStatus(nextStatus);
        if (!nextStatus?.ready) {
          throw new Error(pyannoteSetupCopy.installIncomplete);
        }
        return nextStatus;
      } finally {
        setIsPreparingPyannote(false);
      }
    },
    [loadPyannoteStatus, pyannoteSetupCopy.installIncomplete]
  );

  const ensurePyannoteProviderReady = React.useCallback(async () => {
    let tokenConfigured = Boolean(pyannoteStatus?.tokenConfigured);

    while (true) {
      let token: string | undefined;
      if (!tokenConfigured) {
        const input = window.prompt(pyannoteSetupCopy.promptTitle, '') || '';
        token = input.trim();
        if (!token) {
          window.alert(pyannoteSetupCopy.skipped);
          return false;
        }
      }

      try {
        const status = await installPyannoteAssets(token);
        if (status?.ready) {
          setDiarizationProvider('pyannote');
          await loadPyannoteStatus();
          return true;
        }
      } catch (error) {
        const message = getRequestErrorMessage(error);
        tokenConfigured = false;
        const shouldRetry = window.confirm(
          `${pyannoteSetupCopy.installFailed}\n\n${message}\n\n${pyannoteSetupCopy.retryQuestion}`
        );
        if (!shouldRetry) {
          setDiarizationProvider('classic');
          return false;
        }
      }
    }
  }, [
    getRequestErrorMessage,
    installPyannoteAssets,
    loadPyannoteStatus,
    pyannoteSetupCopy.installFailed,
    pyannoteSetupCopy.installStarting,
    pyannoteSetupCopy.promptTitle,
    pyannoteSetupCopy.retryQuestion,
    pyannoteSetupCopy.skipped,
    pyannoteStatus?.tokenConfigured,
  ]);

  const handleDiarizationProviderSelect = React.useCallback(
    (nextProvider: DiarizationProvider) => {
      if (nextProvider === 'classic') {
        setDiarizationProvider('classic');
        return;
      }

      if (pyannoteStatus?.ready) {
        setDiarizationProvider('pyannote');
        return;
      }

      setDiarizationProvider('classic');
      void ensurePyannoteProviderReady();
    },
    [ensurePyannoteProviderReady, pyannoteStatus?.ready]
  );

  const releaseLocalRuntime = React.useCallback(async () => {
    try {
      await postJson('/api/local-models/release', { target: 'asr' }, { timeoutMs: 12000 });
    } catch {
      // Keep navigation resilient even if runtime release fails.
    }
  }, []);

  React.useEffect(() => {
    return () => {
      transcribeEventSourceRef.current?.close();
      transcribeEventSourceRef.current = null;
      onTaskLockChange?.(false);
      void releaseLocalRuntime();
    };
  }, [releaseLocalRuntime, onTaskLockChange]);

  const handleNextStep = async () => {
    await releaseLocalRuntime();
    onNext();
  };

  const fetchAssets = async () => {
    if (!project) return;
    setIsLoadingAssets(true);
    try {
      const response = await fetch(`/api/projects/${project.id}/materials`);
      const data = await response.json();
      // Only show audio formats as requested
      const audioAssets = data.filter((m: any) => m.category === 'audio');
      setAssets(audioAssets);
    } catch (e) {
      console.error('Failed to fetch assets', e);
    } finally {
      setIsLoadingAssets(false);
    }
  };

  React.useEffect(() => {
    if (showAssetModal && project?.id) {
      fetchAssets();
    }
  }, [showAssetModal, project?.id]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !project) return;

    // Strict audio check
    const allowedExtensions = ['.mp3', '.wav', '.aac', '.m4a', '.flac'];
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    if (!allowedExtensions.includes(ext)) {
      alert(t('stt.onlyAudio'));
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`/api/projects/${project.id}/materials/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.success) {
        await fetchAssets();
      } else {
        alert(data.error || t('stt.uploadError'));
      }
    } catch (e) {
      alert(t('stt.uploadError'));
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      const input = fileInputRef.current;
      if (input) {
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        input.files = dataTransfer.files;
        handleFileUpload({ target: input } as any);
      }
    }
  };

  const handleOnlineFetchClick = () => {
    if (!project?.videoUrl && !project?.audioUrl) {
      onBack();
    } else {
      setSourceType('online');
    }
  };

  const handleSelectFromProjectClick = () => {
    setSourceType('project');
    setShowAssetModal(true);
  };

  const [asrMsg, setAsrMsg] = React.useState<string | null>(null);

  const handleStopTranscription = React.useCallback(() => {
    const current = transcribeEventSourceRef.current;
    if (!current) return;
    current.close();
    transcribeEventSourceRef.current = null;
    setIsTranscribing(false);
    setAsrMsg(t('stt.transcriptionStopped'));
    setModelLoadStatus('idle');
    setModelLoadError(null);
    setDiarizationDiagnostics(null);
    onTaskLockChange?.(false);
    void releaseLocalRuntime();
  }, [onTaskLockChange, releaseLocalRuntime, t]);

  const handleDownloadTranscript = (format: 'txt' | 'srt' | 'vtt') => {
    if (!project) return;
    window.open(`/api/projects/${project.id}/transcript/download?format=${format}`, '_blank');
  };

  const handleStartEditTranscription = React.useCallback(() => {
    if (isTranscribing || transcription.length === 0) return;
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
    }
    setActivePreviewLine(null);
    setEditingTranscriptionRows(subtitleRowsFromText(transcription.join('\n')));
    setIsEditingTranscription(true);
  }, [isTranscribing, transcription]);

  const handleSaveEditedTranscription = React.useCallback(async () => {
    if (!project || isSavingTranscription) return;
    const issues = validateSubtitleRows(editingTranscriptionRows);
    if (issues.length > 0) return;
    const nextLines = subtitleRowsToLines(editingTranscriptionRows);
    const nextText = nextLines.join('\n');

    if (nextText === transcription.join('\n')) {
      setIsEditingTranscription(false);
      return;
    }

    setIsSavingTranscription(true);
    try {
      const updateResult = await Promise.resolve(
        onUpdateProject({
          originalSubtitles: nextText,
          status: PROJECT_STATUS.TEXT_TRANSLATION,
        })
      );
      if (updateResult === null) {
        throw new Error('Failed to persist edited transcription');
      }
      setTranscription(nextLines);
      setHasTimecodes(nextLines.some((line) => parseTranscriptLine(line).timecode != null));
      setProgress(nextLines.length > 0 ? 100 : 0);
      setIsEditingTranscription(false);
    } catch (err) {
      console.error('Failed to save edited transcription', err);
      alert(t('settings.saveRetry'));
    } finally {
      setIsSavingTranscription(false);
    }
  }, [editingTranscriptionRows, isSavingTranscription, onUpdateProject, project, t, transcription]);

  const handleStartTranscription = async () => {
    if (!project) return;
    if (!selectedModelId) {
      setModelLoadStatus('failed');
      setModelLoadError(t('stt.noModels'));
      return;
    }

    if (transcribeEventSourceRef.current) {
      transcribeEventSourceRef.current.close();
      transcribeEventSourceRef.current = null;
    }

    setIsTranscribing(true);
    onTaskLockChange?.(true);
    setModelLoadStatus('loading');
    setModelLoadError(null);
    setProgress(0);
    setTranscription([]);
    setAsrMsg(null);
    setActivePreviewLine(null);
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
    }
    setHasTimecodes(false);
    setLastTranscribedAssetName(null);
    setPipelineMode(null);
    setPipelineWarnings([]);
    setProviderDebug(null);
    setAppliedDebug(null);
    setDiarizationDiagnostics(null);
    setAlignmentDiagnostics(null);
    setCjkWordDiagnostics(null);
    setLastElapsedMs(null);
    setLastWordAlignmentState(null);
    setIsEditingTranscription(false);
    setEditingTranscriptionRows([]);

    const params = new URLSearchParams();
    let transcriptionAssetName = 'audio.wav';
    if (selectedModelId) params.append('modelId', selectedModelId);
    if (sourceType === 'project' && selectedAssetName) {
      params.append('assetName', selectedAssetName);
      transcriptionAssetName = selectedAssetName;
    }
    if (selectedLanguage) params.append('language', selectedLanguage);
    if (prompt) params.append('prompt', prompt);
    if (segmentation) params.append('segmentation', 'true');
    if (!wordAlignment) params.append('wordAlignment', 'false');
    if (vad) params.append('vad', 'true');
    if (diarization) {
      params.append('diarization', 'true');
      params.append('diarizationProvider', diarizationProvider);
      params.append('diarizationMode', diarizationMode);
      params.append('diarizationScenePreset', diarizationScenePreset);
      if (diarizationMode === 'fixed') {
        params.append('diarizationExactSpeakerCount', String(Math.max(1, exactSpeakerCount)));
      } else if (diarizationMode === 'range' || diarizationMode === 'many') {
        params.append('diarizationMinSpeakers', String(Math.max(1, minSpeakers)));
        params.append('diarizationMaxSpeakers', String(Math.max(Math.max(1, minSpeakers), maxSpeakers)));
      }
      if (diarizationProvider === 'classic') {
        params.append('preferStablePrimarySpeaker', String(preferStablePrimarySpeaker));
        params.append('allowShortInterjectionSpeaker', String(allowShortInterjectionSpeaker));
        params.append('preferVadBoundedRegions', String(preferVadBoundedRegions));
        params.append('forceMergeTinyClustersInTwoSpeakerMode', String(forceMergeTinyClustersInTwoSpeakerMode));
        params.append('semanticFallbackEnabled', String(semanticFallbackEnabled));
      }
    }

    let eventSource: EventSource;
    try {
      eventSource = new EventSource(`/api/transcribe/${project.id}?${params.toString()}`);
    } catch (error) {
      console.error('Failed to start ASR stream', error);
      setModelLoadStatus('failed');
      setModelLoadError(t('settings.testFailed'));
      setIsTranscribing(false);
      onTaskLockChange?.(false);
      return;
    }
    transcribeEventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      if (transcribeEventSourceRef.current !== eventSource) return;
      const data = JSON.parse(event.data);

      if (data.error) {
        console.error('ASR Error:', data.error);
        alert(`${t('settings.testFailed')}: ${data.error}`);
        setModelLoadStatus('failed');
        setModelLoadError(String(data.error || t('settings.testFailed')));
        setIsTranscribing(false);
        setDiarizationDiagnostics(null);
        setAlignmentDiagnostics(null);
        setCjkWordDiagnostics(null);
        setProviderDebug(null);
        setAppliedDebug(null);
        setLastElapsedMs(null);
        setLastWordAlignmentState(null);
        transcribeEventSourceRef.current = null;
        onTaskLockChange?.(false);
        void releaseLocalRuntime();
        eventSource.close();
        return;
      }

      if (data.status === 'processing') {
        setModelLoadStatus('ok');
        setModelLoadError(null);
        setAsrMsg(localizeAsrMessage(String(data.message || '')));
        // Simulate minor progress for UX during processing
        setProgress(prev => Math.min(prev + 5, 90));
      } else if (data.status === 'completed') {
        setModelLoadStatus('ok');
        setModelLoadError(null);
        const result = data.result;
        const includeTimecodes = shouldIncludeTimecodes(result);
        const formattedChunks = formatResultToLines(result, includeTimecodes);
        setTranscription(formattedChunks);
        setHasTimecodes(includeTimecodes);
        setLastTranscribedAssetName(transcriptionAssetName);
        setPipelineMode(formatPipelineMode(result?.debug));
        setPipelineWarnings(localizePipelineWarnings(result?.debug?.warnings));
        setProviderDebug(result?.debug?.provider || null);
        setAppliedDebug(result?.debug?.applied || null);
        setDiarizationDiagnostics(result?.debug?.diarization || null);
        setAlignmentDiagnostics(result?.debug?.alignment || null);
        setCjkWordDiagnostics(result?.debug?.cjkWordDiagnostics || null);
        const elapsedMsRaw = Number(result?.debug?.timing?.elapsedMs);
        setLastElapsedMs(Number.isFinite(elapsedMsRaw) ? elapsedMsRaw : null);
        const requestedWordAlignment = Boolean(result?.debug?.requested?.wordAlignment);
        const appliedWordAlignment = Boolean(result?.debug?.applied?.wordAlignment);
        setLastWordAlignmentState(
          requestedWordAlignment
            ? (appliedWordAlignment ? 'on' : 'unavailable')
            : 'off'
        );
        onUpdateProject({ 
          originalSubtitles: formattedChunks.join('\n'),
          status: PROJECT_STATUS.TEXT_TRANSLATION,
        });
        setProgress(100);
        setIsTranscribing(false);
        setAsrMsg(t('status.completed'));
        setIsEditingTranscription(false);
        setEditingTranscriptionRows([]);
        transcribeEventSourceRef.current = null;
        onTaskLockChange?.(false);
        eventSource.close();
      }
    };

    eventSource.onerror = () => {
      if (transcribeEventSourceRef.current !== eventSource) return;
      console.error('ASR Connection error');
      setModelLoadStatus('failed');
      setModelLoadError(t('settings.testFailed'));
      setIsTranscribing(false);
      setDiarizationDiagnostics(null);
      setAlignmentDiagnostics(null);
      setCjkWordDiagnostics(null);
      setProviderDebug(null);
      setAppliedDebug(null);
      setLastElapsedMs(null);
      setLastWordAlignmentState(null);
      transcribeEventSourceRef.current = null;
      onTaskLockChange?.(false);
      void releaseLocalRuntime();
      eventSource.close();
    };
  };

  const localizeAsrMessage = (message: string) => {
    if (!message) return '';

    if (message === 'Running voice activity detection...') {
      return t('stt.msgRunningVad');
    }

    if (message === 'Running speaker diarization...') {
      return t('stt.msgRunningDiarization');
    }

    if (message === 'Segmentation requested but provider endpoint may ignore it.') {
      return t('stt.msgSegmentationIgnored');
    }

    if (message === 'Segmentation was enabled automatically for diarization.') {
      return t('stt.msgSegmentationForcedForDiarization');
    }

    if (message === 'Provider rejected language=auto, retried without language.') {
      return t('stt.msgAutoLanguageRetried');
    }

    if (message === 'VAD windows produced no transcript text, falling back to full-audio request.') {
      return t('stt.msgVadFallbackFullAudio');
    }

    if (message === 'Provider rejected audio size/duration limit, retrying with chunked upload...') {
      return t('stt.msgFileLimitRetryChunked');
    }

    if (message === 'Provider file size/duration fallback was applied (chunked upload).') {
      return t('stt.msgFileLimitFallbackApplied');
    }

    const fileLimitChunkSizeReducedMatch = message.match(/^Chunk uploads still exceed provider limit, reducing chunk size to (\d+)s and retrying\.\.\.$/);
    if (fileLimitChunkSizeReducedMatch) {
      return t('stt.msgFileLimitReducingChunkSize').replace('{seconds}', fileLimitChunkSizeReducedMatch[1]);
    }

    const callingProviderMatch = message.match(/^Calling ASR provider \(segmentation: (on|off)\)\.\.\.$/);
    if (callingProviderMatch) {
      return callingProviderMatch[1] === 'on'
        ? t('stt.msgCallingProviderSegOn')
        : t('stt.msgCallingProviderSegOff');
    }

    const callingProviderWindowMatch = message.match(/^Calling ASR provider \(VAD window (\d+)\/(\d+)\)\.\.\.$/);
    if (callingProviderWindowMatch) {
      return t('stt.msgCallingProviderVadWindow')
        .replace('{index}', callingProviderWindowMatch[1])
        .replace('{total}', callingProviderWindowMatch[2]);
    }

    const providerCompletedWindowMatch = message.match(/^ASR provider completed \((\d+) VAD windows\)\.$/);
    if (providerCompletedWindowMatch) {
      return t('stt.msgProviderCompletedVadWindows').replace('{count}', providerCompletedWindowMatch[1]);
    }

    const proactiveChunkingMatch = message.match(/^Audio file is ([\d.]+)MB \(> ([\d.]+)MB\), using chunked upload proactively\.\.\.$/);
    if (proactiveChunkingMatch) {
      return t('stt.msgAudioFileChunkedUpload')
        .replace('{size}', proactiveChunkingMatch[1])
        .replace('{threshold}', proactiveChunkingMatch[2]);
    }

    const callingProviderFileLimitChunkMatch = message.match(/^Calling ASR provider \(file-limit chunk (\d+)\)\.\.\.$/);
    if (callingProviderFileLimitChunkMatch) {
      return t('stt.msgCallingProviderFileLimitChunk').replace('{index}', callingProviderFileLimitChunkMatch[1]);
    }

    const providerCompletedFileLimitChunkMatch = message.match(/^ASR provider completed \((\d+) file-limit chunks\)\.$/);
    if (providerCompletedFileLimitChunkMatch) {
      return t('stt.msgProviderCompletedFileLimitChunks').replace('{count}', providerCompletedFileLimitChunkMatch[1]);
    }

    const vadDetectedMatch = message.match(/^VAD detected (\d+) speech segments \((\d+) windows\)\.$/);
    if (vadDetectedMatch) {
      return t('stt.msgVadDetectedWindows')
        .replace('{segments}', vadDetectedMatch[1])
        .replace('{windows}', vadDetectedMatch[2]);
    }

    const vadAppliedMatch = message.match(/^VAD applied \((\d+) speech segments\)\.$/);
    if (vadAppliedMatch) {
      return t('stt.msgVadApplied').replace('{count}', vadAppliedMatch[1]);
    }

    const forcedAlignmentAppliedMatch = message.match(/^Forced alignment applied \((\d+) spans\)\.$/);
    if (forcedAlignmentAppliedMatch) {
      return t('stt.msgForcedAlignmentApplied').replace('{count}', forcedAlignmentAppliedMatch[1]);
    }

    if (message === 'Forced alignment did not improve timing, keeping provider timestamps.') {
      return t('stt.msgForcedAlignmentKeptProviderTimestamps');
    }

    if (message === 'Preparing pyannote diarization...') {
      return t('stt.msgPreparingPyannote');
    }

    const acousticFallbackMatch = message.match(/^Acoustic diarization fallback triggered: (.+)$/);
    if (acousticFallbackMatch) {
      return t('stt.msgAcousticDiarizationFallback').replace('{detail}', acousticFallbackMatch[1]);
    }

    const diarizationCollapsedMatch = message.match(/^(.+) diarization collapsed to one speaker, retrying with fallback regions\.\.\.$/);
    if (diarizationCollapsedMatch) {
      return t('stt.msgDiarizationCollapsed').replace('{source}', diarizationCollapsedMatch[1]);
    }

    return message;
  };

  const formatTime = (seconds: number) => {
    if (seconds === undefined || Number.isNaN(seconds)) return '00:00:00';
    const date = new Date(0);
    date.setSeconds(seconds);
    return date.toISOString().substr(11, 8);
  };

  const parseTimeToSeconds = (time: string): number | null => {
    const m = String(time || '').trim().match(/^(\d{2}):(\d{2}):(\d{2})(?:[.,](\d{1,3}))?$/);
    if (!m) return null;
    const hh = Number(m[1]);
    const mm = Number(m[2]);
    const ss = Number(m[3]);
    const msRaw = m[4] ? Number(m[4].padEnd(3, '0')) : 0;
    if (![hh, mm, ss, msRaw].every(Number.isFinite)) return null;
    return hh * 3600 + mm * 60 + ss + msRaw / 1000;
  };

  const parseTranscriptLine = (line: string) => {
    const timeMatch = String(line || '').match(/^\[(\d{2}:\d{2}:\d{2}(?:[.,]\d{1,3})?)\]\s([\s\S]*)$/);
    if (!timeMatch) {
      return {
        timecode: null as string | null,
        startSeconds: null as number | null,
        speaker: null as string | null,
        text: String(line || ''),
      };
    }
    const timecode = timeMatch[1];
    const startSeconds = parseTimeToSeconds(timecode);
    const content = timeMatch[2];
    const speakerMatch = content.match(/^\[(.*?)\]\s([\s\S]*)$/);
    if (speakerMatch) {
      return {
        timecode,
        startSeconds,
        speaker: speakerMatch[1],
        text: speakerMatch[2],
      };
    }
    return {
      timecode,
      startSeconds,
      speaker: null as string | null,
      text: content,
    };
  };

  React.useEffect(() => {
    if (isTranscribing) return;

    const storedText = String(project?.originalSubtitles || '')
      .replace(/\r\n/g, '\n')
      .replace(/\r/g, '\n')
      .trim();

    if (!storedText) return;

    const restoredLines = storedText.split('\n').map((line) => line.trim()).filter(Boolean);
    const currentLines = transcription.map((line) => line.trim()).filter(Boolean);
    if (restoredLines.join('\n') === currentLines.join('\n')) {
      if (progress < 100) setProgress(100);
      return;
    }

    setTranscription(restoredLines);
    setHasTimecodes(restoredLines.some((line) => parseTranscriptLine(line).timecode != null));
    setProgress(100);
  }, [project?.originalSubtitles, isTranscribing, transcription, progress]);

  const parsedTranscription = React.useMemo(() => {
    return transcription.map((line, index) => ({
      index,
      line,
      ...parseTranscriptLine(line),
    }));
  }, [transcription]);

  const transcriptionEditIssues = React.useMemo(
    () => validateSubtitleRows(editingTranscriptionRows),
    [editingTranscriptionRows]
  );
  const canSaveEditedTranscription =
    editingTranscriptionRows.length > 0 &&
    transcriptionEditIssues.length === 0;
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

  const canProceedToNextStep = !isTranscribing && !isEditingTranscription && transcription.some((line) => String(line || '').trim().length > 0);
  const nextStepLabel = React.useMemo(
    () => t('dashboard.goToStep').replace('{step}', t('translation.title')),
    [t]
  );

  const findActiveLineIndexByTime = React.useCallback((timeSec: number): number | null => {
    if (!hasTimecodes || !Number.isFinite(timeSec) || parsedTranscription.length === 0) return null;
    let targetIndex: number | null = null;
    for (const item of parsedTranscription) {
      if (item.startSeconds == null) continue;
      if (item.startSeconds <= timeSec) {
        targetIndex = item.index;
        continue;
      }
      break;
    }
    return targetIndex;
  }, [hasTimecodes, parsedTranscription]);

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
    const container = transcriptionScrollRef.current;
    const row = transcriptLineRefs.current[activePreviewLine];
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

  const handlePreviewLineClick = async (line: string, lineIndex: number) => {
    const parsed = parseTranscriptLine(line);
    if (parsed.startSeconds == null || !previewAudioUrl || !previewAudioRef.current) return;
    const audio = previewAudioRef.current;
    try {
      const targetSrc = new URL(previewAudioUrl, window.location.origin).toString();
      if (!audio.src || audio.src !== targetSrc) {
        audio.src = previewAudioUrl;
      }
      audio.currentTime = Math.max(0, parsed.startSeconds);
      await audio.play();
      setActivePreviewLine(lineIndex);
    } catch (err) {
      console.error('Failed to play preview audio at timecode', err);
    }
  };

  const shouldIncludeTimecodes = (result: any): boolean => {
    const requestedSegmentation = Boolean(result?.debug?.requested?.segmentation);
    const requestedDiarization = Boolean(result?.debug?.requested?.diarization);
    if (!requestedSegmentation && !requestedDiarization) {
      // Respect user intent: when segmentation is off (and no diarization), do not render timecodes.
      return false;
    }

    const appliedSegmentation = Boolean(result?.debug?.applied?.segmentation);
    const chunks = Array.isArray(result?.chunks) ? result.chunks : [];
    let prevStart: number | null = null;
    let hasIncreasingStarts = false;
    const hasDurationalWindow = chunks.some((chunk: any) => {
      const start = Number(chunk?.start_ts ?? chunk?.startTs ?? chunk?.start);
      const end = Number(chunk?.end_ts ?? chunk?.endTs ?? chunk?.end);
      if (Number.isFinite(start)) {
        if (prevStart == null || start > prevStart + 0.01) {
          if (prevStart != null) hasIncreasingStarts = true;
          prevStart = start;
        }
      }
      return Number.isFinite(start) && Number.isFinite(end) && end > start + 0.01;
    });

    const chunkHasUsableTimestamps = hasDurationalWindow || hasIncreasingStarts;
    if (!appliedSegmentation && !chunkHasUsableTimestamps) return false;

    return chunkHasUsableTimestamps || Boolean(result?.debug?.provider?.rawHasTimestamps);
  };


  const formatResultToLines = (result: any, includeTimecodes: boolean): string[] => {
    const rawChunks = Array.isArray(result?.chunks) ? result.chunks : [];
    if (rawChunks.length > 0) {
      return rawChunks.flatMap((chunk: any) => {
        const start = Number(chunk?.start_ts ?? chunk?.startTs ?? chunk?.start);
        const hasValidStart = Number.isFinite(start);
        const safeStart = hasValidStart ? start : 0;
        const speaker = typeof chunk?.speaker === 'string' && chunk.speaker.trim() ? `[${chunk.speaker.trim()}] ` : '';
        const rawText = String(chunk?.text ?? chunk?.transcript ?? chunk?.content ?? '')
          .replace(/\r\n/g, '\n')
          .replace(/\r/g, '\n');
        const lines = rawText.split('\n').map((line) => line.trim()).filter(Boolean);
        return lines.map((line) => (includeTimecodes && hasValidStart) ? `[${formatTime(safeStart)}] ${speaker}${line}` : `${speaker}${line}`);
      });
    }

    const fallbackText = String(result?.text ?? (Array.isArray(result?.texts) ? result.texts.join('\n') : '') ?? '')
      .replace(/\r\n/g, '\n')
      .replace(/\r/g, '\n')
      .trim();
    if (!fallbackText) return [];

    return fallbackText
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => line);
  };

  const formatPipelineMode = (debug: any) => {
    if (!debug || typeof debug !== 'object') return null;
    const applied = debug.applied || {};
    const stats = debug.stats || {};
    const modeParts: string[] = [];

    modeParts.push(
      applied.segmentation ? t('stt.modeSegOn') : t('stt.modeSegOff')
    );
    modeParts.push(
      applied.vad
        ? t('stt.modeVadWindows').replace('{count}', String(stats.vadWindowCount ?? 0))
        : t('stt.modeVadOff')
    );
    modeParts.push(
      applied.diarization ? t('stt.modeDiarizationOn') : t('stt.modeDiarizationOff')
    );
    modeParts.push(
      applied.forcedAlignment ? t('stt.modeAlignOn') : t('stt.modeAlignOff')
    );

    return modeParts.join(' | ');
  };

  const formatTechnicalValue = React.useCallback((value: string | null | undefined) => {
    const normalized = String(value || '').trim();
    if (!normalized) return '-';
    return normalized.replace(/[_-]+/g, ' ');
  }, []);

  const formatElapsedTime = React.useCallback((elapsedMs: number | null) => {
    if (!Number.isFinite(elapsedMs == null ? Number.NaN : elapsedMs)) return null;
    const safeMs = Math.max(0, Number(elapsedMs));
    if (safeMs < 1000) return `${safeMs} ms`;
    const totalSeconds = safeMs / 1000;
    if (totalSeconds < 60) return `${totalSeconds.toFixed(1)} s`;
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    if (hours > 0) {
      return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${seconds.toFixed(1).padStart(4, '0')}`;
    }
    return `${String(minutes).padStart(2, '0')}:${seconds.toFixed(1).padStart(4, '0')}`;
  }, []);

  const localizePipelineWarnings = (warnings: unknown): string[] => {
    if (!Array.isArray(warnings)) return [];
    return warnings
      .map((item) => String(item || '').trim())
      .filter(Boolean)
      .map((code) => {
        switch (code) {
          case 'segmentation_forced_for_diarization':
            return t('stt.warnSegmentationForcedForDiarization');
          case 'segmentation_ignored_by_provider':
            return t('stt.warnSegmentationIgnoredByProvider');
          case 'auto_language_retried_without_language':
            return t('stt.warnAutoLanguageRetried');
          case 'vad_segmented_transcription':
            return t('stt.warnVadWindowedPipeline');
          case 'provider_file_limit_chunked':
            return t('stt.warnProviderFileLimitChunked');
          case 'segmentation_timestamp_synthesized':
            return t('stt.warnSegmentationTimestampSynthesized');
          case 'english_forced_alignment_applied':
          case 'forced_alignment_applied':
            return t('stt.warnForcedAlignmentApplied');
          default:
            return code;
        }
      });
  };

  const diarizationModeOptions = React.useMemo(
    () => [
      { value: 'auto' as DiarizationMode, label: diarizationCopy.auto },
      { value: 'fixed' as DiarizationMode, label: diarizationCopy.fixed },
      { value: 'range' as DiarizationMode, label: diarizationCopy.range },
      { value: 'many' as DiarizationMode, label: diarizationCopy.many },
    ],
    [diarizationCopy]
  );

  const diarizationProviderOptions = React.useMemo(
    () => [
      { value: 'classic' as DiarizationProvider, label: diarizationCopy.providerClassic },
      { value: 'pyannote' as DiarizationProvider, label: diarizationCopy.providerPyannote },
    ],
    [diarizationCopy]
  );

  const diarizationSceneOptions = React.useMemo(
    () => [
      { value: 'interview' as DiarizationScenePreset, label: diarizationCopy.interview },
      { value: 'podcast' as DiarizationScenePreset, label: diarizationCopy.podcast },
      { value: 'meeting' as DiarizationScenePreset, label: diarizationCopy.meeting },
      { value: 'presentation_qa' as DiarizationScenePreset, label: diarizationCopy.presentationQa },
      { value: 'custom' as DiarizationScenePreset, label: diarizationCopy.custom },
    ],
    [diarizationCopy]
  );

  const localizedDiarizationProvider =
    diarizationDiagnostics?.provider === 'semantic'
      ? diarizationCopy.providerSemantic
      : diarizationCopy.providerAcoustic;
  const localizedDiarizationSource =
    diarizationDiagnostics?.selectedSource === 'speech_region'
      ? diarizationCopy.sourceSpeechRegion
      : diarizationDiagnostics?.selectedSource === 'vad_chunk'
      ? diarizationCopy.sourceVad
      : diarizationDiagnostics?.selectedSource === 'pyannote'
        ? diarizationCopy.sourcePyannote
      : diarizationDiagnostics?.selectedSource === 'semantic'
        ? diarizationCopy.sourceSemantic
        : diarizationCopy.sourceChunk;
  const formattedElapsedTime = React.useMemo(() => formatElapsedTime(lastElapsedMs), [formatElapsedTime, lastElapsedMs]);
  const formattedAlignmentElapsedTime = React.useMemo(
    () => formatElapsedTime(alignmentDiagnostics?.elapsedMs ?? null),
    [alignmentDiagnostics?.elapsedMs, formatElapsedTime]
  );
  const localizedWordAlignmentState =
    lastWordAlignmentState === 'on'
      ? t('stt.wordAlignmentOn')
      : lastWordAlignmentState === 'off'
        ? t('stt.wordAlignmentOff')
        : lastWordAlignmentState === 'unavailable'
          ? t('stt.wordAlignmentUnavailable')
          : null;
  const localizedForcedAlignmentState =
    alignmentDiagnostics?.applied
      ? t('stt.forcedAlignmentOn')
      : alignmentDiagnostics
        ? t('stt.forcedAlignmentOff')
        : null;
  const providerProfileId = String(providerDebug?.profileId || '').trim() || '-';
  const providerProfileFamily = formatTechnicalValue(providerDebug?.profileFamily);
  const localModelProfileId = String(appliedDebug?.localModelProfileId || '').trim() || '-';
  const localBaselineConfidence = formatTechnicalValue(appliedDebug?.localBaselineConfidence);
  const localBaselineTaskFamily = formatTechnicalValue(appliedDebug?.localBaselineTaskFamily);
  const localFallbackBaseline =
    typeof appliedDebug?.localFallbackBaseline === 'boolean'
      ? (appliedDebug.localFallbackBaseline ? t('stt.valueYes') : t('stt.valueNo'))
      : '-';
  const providerForcedAlignment = providerDebug?.forcedAlignment || null;
  const providerHelperChunking = providerDebug?.helperChunking || null;
  const alignmentSourceLabel =
    providerForcedAlignment?.backend === 'qwen3-forced-aligner' && providerForcedAlignment?.applied
      ? t('stt.alignmentSourceQwenOfficial')
      : alignmentDiagnostics?.modelId
        ? t('stt.alignmentSourceGeneric')
        : providerDebug?.rawHasTimestamps
          ? t('stt.alignmentSourceProvider')
          : '-';
  const providerForcedAlignmentError =
    providerForcedAlignment && !providerForcedAlignment.applied && providerForcedAlignment.error
      ? String(providerForcedAlignment.error).trim()
      : '';
  const hasStatusDetails = Boolean(
    pipelineMode ||
    providerDebug ||
    formattedElapsedTime ||
    localizedWordAlignmentState ||
    alignmentDiagnostics ||
    cjkWordDiagnostics ||
    pipelineWarnings.length > 0 ||
    diarizationDiagnostics
  );

  const currentAudioLabel =
    sourceType === 'online'
      ? (project?.audioUrl || project?.videoUrl ? (project.videoTitle || project.name) : t('stt.noOnlineVideo'))
      : (selectedAssetName || t('stt.notSelected'));
  const hasSourceReady = sourceType === 'online'
    ? Boolean(project?.audioUrl || project?.videoUrl)
    : Boolean(selectedAssetName);
  const selectedModelName = asrModels.find((model) => model.id === selectedModelId)?.name || t('stt.noModels');
  const selectedLanguageLabel = selectedLanguage === 'auto'
    ? t('stt.autoDetect')
    : selectedLanguage === 'en'
      ? t('lang.en')
      : selectedLanguage === 'zh'
        ? t('lang.zh-audio')
        : selectedLanguage === 'fi'
          ? t('lang.fi')
          : selectedLanguage === 'es'
            ? t('lang.es')
            : selectedLanguage === 'de'
              ? t('lang.de')
              : selectedLanguage === 'pt'
                ? t('lang.pt')
                : selectedLanguage === 'it'
                  ? t('lang.it')
                  : selectedLanguage === 'fr'
                    ? t('lang.fr')
                    : selectedLanguage === 'ja'
                      ? t('lang.jp')
                      : selectedLanguage === 'ko'
                      ? t('lang.kr')
                        : selectedLanguage;
  React.useEffect(() => {
    if (diarization || vad || !segmentation || prompt.trim()) {
      setShowAdvanced(true);
    }
  }, [diarization, vad, segmentation, prompt]);

  return (
    <>
      <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-4 mb-1">
            <h2 className="text-3xl font-bold text-secondary tracking-tight">{t('stt.title')}</h2>
            {project && (
              <div className="px-3 py-1 bg-primary-container/10 border border-primary-container/20 rounded-full flex items-center gap-2 animate-in fade-in zoom-in duration-500">
                <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse shadow-[0_0_8px_rgba(var(--primary),0.8)]" />
                <span className="text-[10px] font-bold text-primary uppercase tracking-widest">{project.name}</span>
              </div>
            )}
          </div>
          <p className="text-outline">{t('stt.subtitle')}</p>
        </div>

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.42fr)_minmax(360px,0.92fr)] gap-6">
        <div className="space-y-6">
          <section className="bg-surface-container p-6 rounded-[28px] border border-white/5 shadow-xl">
            <div className="flex flex-col gap-5">
              <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <h3 className="text-sm font-bold text-primary uppercase tracking-widest flex items-center gap-2">
                    <Mic className="w-4 h-4" />
                    {t('stt.sourceSelection')}
                  </h3>
                  <p className="text-sm text-outline mt-2">{t('stt.selectedAudio')}</p>
                </div>
                <div className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-[11px] font-bold ${
                  hasSourceReady ? 'border-tertiary/20 bg-tertiary/10 text-tertiary' : 'border-white/8 bg-white/[0.03] text-outline'
                }`}>
                  <span className={`h-2 w-2 rounded-full ${hasSourceReady ? 'bg-tertiary' : 'bg-outline/40'}`} />
                  {hasSourceReady ? t('status.completed') : t('common.standby')}
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 p-1.5 bg-surface-container-lowest rounded-2xl">
                <button
                  onClick={handleOnlineFetchClick}
                  className={`py-3 px-4 text-sm font-bold rounded-xl flex items-center justify-center gap-2 transition-all ${sourceType === 'online' ? 'bg-primary-container text-white shadow-lg' : 'text-outline hover:text-secondary hover:bg-white/5'}`}
                >
                  <Download className="w-4 h-4" />
                  {t('stt.onlineFetch')}
                </button>
                <button
                  onClick={handleSelectFromProjectClick}
                  className={`py-3 px-4 text-sm font-bold rounded-xl flex items-center justify-center gap-2 transition-all ${sourceType === 'project' ? 'bg-primary-container text-white shadow-lg' : 'text-outline hover:text-secondary hover:bg-white/5'}`}
                >
                  <FolderOpen className="w-4 h-4" />
                  {t('stt.selectFromProject')}
                </button>
              </div>

              <div className="rounded-2xl border border-white/5 bg-surface-container-lowest p-4 flex items-center gap-4">
                <div className="w-11 h-11 rounded-2xl bg-primary-container/20 flex items-center justify-center shrink-0">
                  <Music className="w-5 h-5 text-primary" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-outline font-medium mb-1">{t('stt.selectedAudio')}</p>
                  <p className="text-sm text-secondary font-bold truncate">{currentAudioLabel}</p>
                </div>
              </div>

              {sourceType === 'online' && !hasSourceReady && (
                <div className="rounded-2xl border border-white/5 bg-white/[0.03] p-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <p className="text-sm text-outline leading-relaxed">{t('stt.noOnlineVideo')}</p>
                  <button
                    type="button"
                    onClick={onBack}
                    className="shrink-0 px-4 py-2 rounded-xl border border-white/10 bg-white/[0.04] text-secondary text-sm font-bold hover:bg-white/10 transition-all"
                  >
                    {t('fetcher.title')}
                  </button>
                </div>
              )}
            </div>
          </section>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <section className="bg-surface-container p-6 rounded-[28px] border border-white/5 shadow-xl">
              <label className="block text-sm font-bold text-primary uppercase tracking-widest mb-4">{t('stt.modelLabel')}</label>
              <select
                value={selectedModelId}
                onChange={(e) => {
                  setSelectedModelId(e.target.value);
                  setModelLoadStatus('idle');
                  setModelLoadError(null);
                }}
                className="w-full bg-surface-container-high border border-white/10 rounded-2xl text-white py-3.5 px-4 focus:ring-2 focus:ring-primary-container outline-none appearance-none cursor-pointer [&>option]:bg-surface-container-high [&>option]:text-white"
              >
                {asrModels.map(model => (
                  <option key={model.id} value={model.id} className="bg-surface-container-high text-white">
                    {model.name}
                  </option>
                ))}
                {asrModels.length === 0 && <option value="">{t('stt.noModels')}</option>}
              </select>
            </section>
            <section className="bg-surface-container p-6 rounded-[28px] border border-white/5 shadow-xl">
              <label className="block text-sm font-bold text-primary uppercase tracking-widest mb-4">{t('stt.languageLabel')}</label>
              <select
                value={selectedLanguage}
                onChange={(e) => setSelectedLanguage(e.target.value)}
                className="w-full bg-surface-container-high border border-white/10 rounded-2xl text-white py-3.5 px-4 focus:ring-2 focus:ring-primary-container outline-none appearance-none cursor-pointer [&>option]:bg-surface-container-high [&>option]:text-white"
              >
                <option value="auto" className="bg-surface-container-high text-white">{t('stt.autoDetect')}</option>
                <option value="en" className="bg-surface-container-high text-white">{t('lang.en')}</option>
                <option value="zh" className="bg-surface-container-high text-white">{t('lang.zh-audio')}</option>
                <option value="fi" className="bg-surface-container-high text-white">{t('lang.fi')}</option>
                <option value="es" className="bg-surface-container-high text-white">{t('lang.es')}</option>
                <option value="de" className="bg-surface-container-high text-white">{t('lang.de')}</option>
                <option value="pt" className="bg-surface-container-high text-white">{t('lang.pt')}</option>
                <option value="it" className="bg-surface-container-high text-white">{t('lang.it')}</option>
                <option value="fr" className="bg-surface-container-high text-white">{t('lang.fr')}</option>
                <option value="ja" className="bg-surface-container-high text-white">{t('lang.jp')}</option>
                <option value="ko" className="bg-surface-container-high text-white">{t('lang.kr')}</option>
              </select>
            </section>
          </div>

          <section className="bg-surface-container p-6 rounded-[28px] border border-white/5 shadow-xl">
            <button
              type="button"
              onClick={() => setShowAdvanced((prev) => !prev)}
              className="w-full flex items-center justify-between gap-4"
            >
              <h3 className="text-sm font-bold text-primary uppercase tracking-widest">{t('stt.advancedFeatures')}</h3>
              <div className="w-10 h-10 rounded-2xl border border-white/10 bg-white/[0.03] flex items-center justify-center text-outline">
                {showAdvanced ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
              </div>
            </button>

            <div className="mt-5 grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
              <label className={`rounded-2xl border px-4 py-4 transition-all ${diarization ? 'border-primary/15 bg-primary/5 opacity-70' : 'border-white/5 bg-surface-container-lowest cursor-pointer hover:bg-white/5'}`}>
                <div className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    checked={segmentation}
                    disabled={diarization}
                    onChange={(e) => handleSegmentationChange(e.target.checked)}
                    className={`mt-0.5 w-5 h-5 rounded border-white/10 bg-transparent text-primary focus:ring-primary ${diarization ? 'cursor-not-allowed text-primary/40' : ''}`}
                  />
                  <div className="min-w-0">
                    <div className="text-sm font-bold text-secondary">{t('stt.segmentation')}</div>
                    <div className="mt-1 text-[11px] leading-relaxed text-outline/80">{t('stt.segmentationHint')}</div>
                  </div>
                </div>
              </label>
              <label className="rounded-2xl border border-white/5 bg-surface-container-lowest px-4 py-4 cursor-pointer hover:bg-white/5 transition-all">
                <div className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    checked={wordAlignment}
                    onChange={(e) => handleWordAlignmentChange(e.target.checked)}
                    className="mt-0.5 w-5 h-5 rounded border-white/10 bg-transparent text-primary focus:ring-primary"
                  />
                  <div className="min-w-0">
                    <div className="text-sm font-bold text-secondary">{t('stt.wordAlignment')}</div>
                    <div className="mt-1 text-[11px] leading-relaxed text-outline/80">{t('stt.wordAlignmentHint')}</div>
                  </div>
                </div>
              </label>
              <label className={`rounded-2xl border px-4 py-4 transition-all ${diarization ? 'border-primary/15 bg-primary/5 opacity-80' : 'border-white/5 bg-surface-container-lowest cursor-pointer hover:bg-white/5'}`}>
                <div className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    checked={vad}
                    disabled={diarization}
                    onChange={(e) => setVad(e.target.checked)}
                    className={`mt-0.5 w-5 h-5 rounded border-white/10 bg-transparent text-primary focus:ring-primary ${diarization ? 'cursor-not-allowed text-primary/40' : ''}`}
                  />
                  <div className="min-w-0">
                    <div className="text-sm font-bold text-secondary">{t('stt.vad')}</div>
                    <div className="mt-1 text-[11px] leading-relaxed text-outline/80">{t('stt.vadHint')}</div>
                  </div>
                </div>
              </label>
              <label className="rounded-2xl border border-white/5 bg-surface-container-lowest px-4 py-4 cursor-pointer hover:bg-white/5 transition-all">
                <div className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    checked={diarization}
                    onChange={(e) => handleDiarizationChange(e.target.checked)}
                    className="mt-0.5 w-5 h-5 rounded border-white/10 bg-transparent text-primary focus:ring-primary"
                  />
                  <div className="min-w-0">
                    <div className="text-sm font-bold text-secondary">{t('stt.diarization')}</div>
                    <div className="mt-1 text-[11px] leading-relaxed text-outline/80">{t('stt.diarizationHint')}</div>
                  </div>
                </div>
              </label>
            </div>

            {showAdvanced && (
              <div className="mt-5 space-y-5 animate-in fade-in slide-in-from-top-2 duration-200">
                {isPreparingPyannote && (
                  <div className="rounded-2xl border border-primary/20 bg-primary/10 px-4 py-3 text-sm font-medium text-primary">
                    {pyannoteSetupCopy.installStarting}
                  </div>
                )}
                {diarization && (
                  <div className="rounded-2xl border border-primary/15 bg-surface-container-lowest p-5 space-y-4">
                    <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] gap-4">
                      <div className="rounded-2xl border border-white/5 bg-white/[0.03] p-4 space-y-4">
                        <div>
                          <div className="text-[11px] font-bold text-outline uppercase tracking-widest mb-3">{diarizationCopy.providerLabel}</div>
                          <select
                            value={diarizationProvider}
                            onChange={(e) => handleDiarizationProviderSelect(e.target.value as DiarizationProvider)}
                            disabled={isPreparingPyannote}
                            className="w-full bg-surface-container-high border border-white/10 rounded-xl text-white py-3 px-4 outline-none disabled:opacity-60 disabled:cursor-not-allowed"
                          >
                            {diarizationProviderOptions.map((option) => (
                              <option key={option.value} value={option.value}>{option.label}</option>
                            ))}
                          </select>
                          {!pyannoteStatus?.ready && (
                            <div className="mt-3 text-[11px] leading-relaxed text-outline/70">
                              {pyannoteStatus?.tokenConfigured ? pyannoteSetupCopy.autoInstallHint : pyannoteSetupCopy.requiresTokenHint}
                            </div>
                          )}
                          {pyannoteStatus?.lastError && !pyannoteStatus.ready && (
                            <div className="mt-2 text-[11px] leading-relaxed text-error/80">
                              {pyannoteStatus.lastError}
                            </div>
                          )}
                        </div>

                        <div>
                          <div className="text-[11px] font-bold text-outline uppercase tracking-widest mb-3">{diarizationCopy.modeLabel}</div>
                          <select
                            value={diarizationMode}
                            onChange={(e) => setDiarizationMode(e.target.value as DiarizationMode)}
                            disabled={areSceneManagedSpeakerControlsLocked}
                            className="w-full bg-surface-container-high border border-white/10 rounded-xl text-white py-3 px-4 outline-none disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            {diarizationModeOptions.map((option) => (
                              <option key={option.value} value={option.value}>{option.label}</option>
                            ))}
                          </select>
                        </div>
                      </div>

                      <div className="rounded-2xl border border-white/5 bg-white/[0.03] p-4 space-y-3">
                        <div className="text-[11px] font-bold text-outline uppercase tracking-widest">{diarizationCopy.sceneLabel}</div>
                        <select
                          value={diarizationScenePreset}
                          onChange={(e) => applyScenePreset(e.target.value as DiarizationScenePreset)}
                          className="w-full bg-surface-container-high border border-white/10 rounded-xl text-white py-3 px-4 outline-none"
                        >
                          {diarizationSceneOptions.map((option) => (
                            <option key={option.value} value={option.value}>{option.label}</option>
                          ))}
                        </select>
                        <div className="text-[11px] text-outline/70 leading-relaxed">
                          {isCustomDiarizationScene ? diarizationCopy.customSceneHint : diarizationCopy.scenePresetHint}
                        </div>
                      </div>
                    </div>

                    {(diarizationMode === 'fixed' || diarizationMode === 'range' || diarizationMode === 'many') && (
                      <div className="rounded-2xl border border-white/5 bg-white/[0.03] p-4">
                        <div className="text-[11px] font-bold text-outline uppercase tracking-widest mb-3">
                          {diarizationMode === 'fixed' ? diarizationCopy.exactCount : `${diarizationCopy.minSpeakers} / ${diarizationCopy.maxSpeakers}`}
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {diarizationMode === 'fixed' ? (
                            <div className="md:col-span-2">
                              <input
                                type="number"
                                min={1}
                                max={8}
                                value={exactSpeakerCount}
                                onChange={(e) => setExactSpeakerCount(Math.max(1, Math.min(8, Number(e.target.value) || 1)))}
                                disabled={areSceneManagedSpeakerControlsLocked}
                                className="w-full bg-surface-container-high border border-white/10 rounded-xl text-white py-3 px-4 outline-none disabled:opacity-50 disabled:cursor-not-allowed"
                              />
                            </div>
                          ) : (
                            <>
                              <div>
                                <label className="block text-[11px] font-bold text-outline uppercase tracking-widest mb-2">{diarizationCopy.minSpeakers}</label>
                                <input
                                  type="number"
                                  min={1}
                                  max={8}
                                  value={minSpeakers}
                                  onChange={(e) => {
                                    const next = Math.max(1, Math.min(8, Number(e.target.value) || 1));
                                    setMinSpeakers(next);
                                    setMaxSpeakers((prev) => Math.max(next, prev));
                                  }}
                                  disabled={areSceneManagedSpeakerControlsLocked}
                                  className="w-full bg-surface-container-high border border-white/10 rounded-xl text-white py-3 px-4 outline-none disabled:opacity-50 disabled:cursor-not-allowed"
                                />
                              </div>
                              <div>
                                <label className="block text-[11px] font-bold text-outline uppercase tracking-widest mb-2">{diarizationCopy.maxSpeakers}</label>
                                <input
                                  type="number"
                                  min={Math.max(1, minSpeakers)}
                                  max={10}
                                  value={maxSpeakers}
                                  onChange={(e) => setMaxSpeakers(Math.max(minSpeakers, Math.min(10, Number(e.target.value) || minSpeakers)))}
                                  disabled={areSceneManagedSpeakerControlsLocked}
                                  className="w-full bg-surface-container-high border border-white/10 rounded-xl text-white py-3 px-4 outline-none disabled:opacity-50 disabled:cursor-not-allowed"
                                />
                              </div>
                            </>
                          )}
                        </div>
                      </div>
                    )}

                    <div className="rounded-2xl border border-white/5 bg-white/[0.03] p-4">
                      <div className="text-[11px] font-bold text-outline uppercase tracking-widest mb-2">{diarizationCopy.advancedLabel}</div>
                      <div className="mb-4 text-[11px] text-outline/70 leading-relaxed">
                        {isPyannoteDiarization ? diarizationCopy.classicOnlyHint : isCustomDiarizationScene ? diarizationCopy.customSceneHint : diarizationCopy.scenePresetHint}
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {[
                          {
                            checked: preferStablePrimarySpeaker,
                            onChange: setPreferStablePrimarySpeaker,
                            label: diarizationCopy.preferStablePrimary,
                          },
                          {
                            checked: allowShortInterjectionSpeaker,
                            onChange: setAllowShortInterjectionSpeaker,
                            label: diarizationCopy.allowShortInterjection,
                          },
                          {
                            checked: forceMergeTinyClustersInTwoSpeakerMode,
                            onChange: setForceMergeTinyClustersInTwoSpeakerMode,
                            label: diarizationCopy.forceMergeTiny,
                          },
                          {
                            checked: semanticFallbackEnabled,
                            onChange: setSemanticFallbackEnabled,
                            label: diarizationCopy.semanticFallback,
                          },
                        ].map((item) => (
                          <label
                            key={item.label}
                            className={`flex items-center gap-3 rounded-xl border border-white/5 px-4 py-3 ${
                              areClassicOnlyConstraintsDisabled ? 'bg-white/5 opacity-55 cursor-not-allowed' : 'bg-surface-container-high cursor-pointer'
                            }`}
                          >
                            <input
                              type="checkbox"
                              checked={isPyannoteDiarization ? false : item.checked}
                              onChange={(e) => item.onChange(e.target.checked)}
                              disabled={areClassicOnlyConstraintsDisabled}
                              className="w-4 h-4 rounded border-white/10 bg-transparent text-primary focus:ring-primary"
                            />
                            <span className="text-xs text-secondary font-medium leading-relaxed">{item.label}</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                <div>
                  <label className="block text-xs font-bold text-outline uppercase tracking-widest mb-3">{t('stt.promptLabel')}</label>
                  <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    className="w-full bg-surface-container-lowest border border-white/10 rounded-2xl text-secondary p-5 text-sm focus:ring-2 focus:ring-primary-container outline-none placeholder:text-outline/30 resize-none"
                    placeholder={t('stt.promptPlaceholder')}
                    rows={4}
                  />
                  <div className="mt-3 text-[11px] leading-relaxed text-outline/78">{t('stt.promptHint')}</div>
                </div>
              </div>
            )}
          </section>

          <button
            onClick={isTranscribing ? handleStopTranscription : handleStartTranscription}
            disabled={!isTranscribing && (!project || !selectedModelId)}
            className={`w-full py-5 text-white font-bold rounded-[28px] border active:scale-[0.98] transition-all flex items-center justify-center gap-3 disabled:opacity-50 ${
              isTranscribing
                ? 'border-error/30 bg-gradient-to-r from-[#5b1620] via-error/90 to-[#ff5a6f] shadow-[0_20px_50px_rgba(239,68,68,0.28)] hover:brightness-110'
                : 'border-primary/20 bg-gradient-to-r from-primary-container to-primary shadow-xl hover:brightness-110'
            }`}
          >
            {isTranscribing ? <Square className="w-5 h-5" /> : <Play className="w-6 h-6" />}
            {isTranscribing ? t('stt.stopTranscription') : t('stt.startTranscription')}
          </button>
        </div>

        <div className="space-y-6">
          <section className="bg-surface-container-high p-6 rounded-[28px] border border-primary/20 shadow-xl">
            <div className="flex items-center justify-between gap-4 mb-6">
              <h3 className="text-sm font-bold text-secondary flex items-center gap-2">
                {isTranscribing ? <Loader2 className="w-4 h-4 animate-spin text-tertiary" /> : <CheckCircle2 className="w-4 h-4 text-tertiary" />}
                {t('stt.statusMonitor')}
              </h3>
              <div className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-[11px] font-bold max-w-full ${
                isTranscribing ? 'bg-primary/10 text-primary border border-primary/20' : 'bg-white/[0.04] text-outline border border-white/10'
              }`}>
                <span className={`h-2 w-2 rounded-full ${isTranscribing ? 'bg-primary animate-pulse' : 'bg-outline/40'}`} />
                <span className="truncate">{isTranscribing ? (asrMsg || t('stt.processing')) : t('common.standby')}</span>
              </div>
            </div>

            <div className="space-y-6">
              <StatusItem 
                label={t('stt.modelLoad')} 
                progress={modelLoadStatus === 'ok' ? 100 : (modelLoadStatus === 'failed' ? 0 : (modelLoadStatus === 'loading' ? 50 : 0))} 
                status={
                  modelLoadStatus === 'ok'
                    ? t('settings.testSuccess')
                    : modelLoadStatus === 'failed'
                      ? t('settings.testFailed')
                      : modelLoadStatus === 'loading'
                        ? t('settings.testing')
                        : t('common.standby')
                } 
                tone={modelLoadStatus === 'ok' ? 'success' : (modelLoadStatus === 'failed' ? 'error' : 'normal')}
              />
              {modelLoadStatus === 'failed' && (
                <div className="text-[10px] text-error/80 bg-error/5 p-3 rounded-lg border border-error/10 animate-in fade-in slide-in-from-top-1">
                  {modelLoadError || t('settings.testFailed')}
                </div>
              )}
              <StatusItem label={t('stt.transcriptionProgress')} progress={progress} />
              {hasStatusDetails && (
                <div className="rounded-2xl border border-white/8 bg-surface-container-lowest/50 overflow-hidden">
                  <button
                    type="button"
                    onClick={() => setShowStatusDetails((current) => !current)}
                    className="w-full px-4 py-3 flex items-center justify-between gap-4 text-left transition-colors hover:bg-white/5"
                  >
                    <div className="min-w-0">
                      <div className="text-xs font-bold text-secondary">{t('stt.statusDetails')}</div>
                      <div className="text-[11px] text-outline mt-1">
                        {showStatusDetails ? t('stt.hideStatusDetails') : t('stt.showStatusDetails')}
                      </div>
                    </div>
                    <div className="shrink-0 inline-flex h-9 w-9 items-center justify-center rounded-full border border-white/10 bg-white/[0.03] text-outline">
                      {showStatusDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </div>
                  </button>

                  {showStatusDetails && (
                    <div className="px-4 pb-4 space-y-4 border-t border-white/5">
                      {pipelineMode && (
                        <div className="mt-4 text-[11px] text-outline bg-surface-container-lowest/70 border border-white/5 rounded-lg p-3">
                          <span className="text-secondary font-bold mr-2">{t('stt.pipelineMode')}:</span>
                          {pipelineMode}
                        </div>
                      )}
                      {(formattedElapsedTime || localizedWordAlignmentState) && (
                        <div className="text-[11px] text-outline bg-surface-container-lowest/70 border border-white/5 rounded-lg p-3 space-y-2">
                          {formattedElapsedTime && (
                            <div>
                              <span className="text-secondary font-bold mr-2">{t('stt.elapsedTime')}:</span>
                              {formattedElapsedTime}
                            </div>
                          )}
                          {localizedWordAlignmentState && (
                            <div>
                              <span className="text-secondary font-bold mr-2">{t('stt.wordAlignment')}:</span>
                              {localizedWordAlignmentState}
                            </div>
                          )}
                        </div>
                      )}
                      {providerDebug && (
                        <div className="text-[11px] text-outline bg-surface-container-lowest/70 border border-white/5 rounded-lg p-3 space-y-2">
                          <div>
                            <span className="text-secondary font-bold mr-2">{t('stt.providerProfileId')}:</span>
                            {providerProfileId}
                          </div>
                          <div>
                            <span className="text-secondary font-bold mr-2">{t('stt.providerProfileFamily')}:</span>
                            {providerProfileFamily}
                          </div>
                          <div>
                            <span className="text-secondary font-bold mr-2">{t('stt.localModelProfileId')}:</span>
                            {localModelProfileId}
                          </div>
                          <div>
                            <span className="text-secondary font-bold mr-2">{t('stt.localBaselineConfidence')}:</span>
                            {localBaselineConfidence}
                          </div>
                          <div>
                            <span className="text-secondary font-bold mr-2">{t('stt.localBaselineTaskFamily')}:</span>
                            {localBaselineTaskFamily}
                          </div>
                          <div>
                            <span className="text-secondary font-bold mr-2">{t('stt.localFallbackBaseline')}:</span>
                            {localFallbackBaseline}
                          </div>
                          <div>
                            <span className="text-secondary font-bold mr-2">{t('stt.alignmentSource')}:</span>
                            {alignmentSourceLabel}
                          </div>
                          {providerForcedAlignment?.modelId && (
                            <div>
                              <span className="text-secondary font-bold mr-2">{t('stt.providerForcedAlignmentModel')}:</span>
                              {providerForcedAlignment.modelId}
                            </div>
                          )}
                          {providerHelperChunking?.strategy && (
                            <div>
                              <span className="text-secondary font-bold mr-2">{t('stt.providerHelperChunking')}:</span>
                              {`${formatTechnicalValue(providerHelperChunking.strategy)} / ${providerHelperChunking.maxChunkSec ?? '-'}s / ${providerHelperChunking.chunkCount ?? '-'} chunks`}
                            </div>
                          )}
                          {providerForcedAlignmentError && (
                            <div>
                              <span className="text-secondary font-bold mr-2">{t('stt.providerForcedAlignmentError')}:</span>
                              {providerForcedAlignmentError}
                            </div>
                          )}
                        </div>
                      )}
                      {alignmentDiagnostics && (
                        <div className="text-[11px] text-outline bg-surface-container-lowest/70 border border-white/5 rounded-lg p-3 space-y-2">
                          <div className="text-secondary font-bold">{t('stt.forcedAlignment')}:</div>
                          {localizedForcedAlignmentState && (
                            <div>
                              <span className="text-secondary font-bold mr-2">{t('stt.statusMonitor')}:</span>
                              {localizedForcedAlignmentState}
                            </div>
                          )}
                          <div>
                            <span className="text-secondary font-bold mr-2">{t('stt.alignmentModel')}:</span>
                            {alignmentDiagnostics.modelId}
                          </div>
                          {alignmentDiagnostics.language && (
                            <div>
                              <span className="text-secondary font-bold mr-2">{t('stt.alignmentLanguage')}:</span>
                              {alignmentDiagnostics.language}
                            </div>
                          )}
                          <div>
                            <span className="text-secondary font-bold mr-2">{t('stt.alignedSegments')}:</span>
                            {alignmentDiagnostics.alignedSegmentCount}/{alignmentDiagnostics.attemptedSegmentCount}
                          </div>
                          <div>
                            <span className="text-secondary font-bold mr-2">{t('stt.alignedWords')}:</span>
                            {alignmentDiagnostics.alignedWordCount}
                          </div>
                          {alignmentDiagnostics.avgConfidence != null && (
                            <div>
                              <span className="text-secondary font-bold mr-2">{t('stt.alignmentConfidence')}:</span>
                              {(alignmentDiagnostics.avgConfidence * 100).toFixed(1)}%
                            </div>
                          )}
                          {formattedAlignmentElapsedTime && (
                            <div>
                              <span className="text-secondary font-bold mr-2">{t('stt.alignmentElapsedTime')}:</span>
                              {formattedAlignmentElapsedTime}
                            </div>
                          )}
                        </div>
                      )}
                      {cjkWordDiagnostics && (
                        <div className="text-[11px] text-outline bg-surface-container-lowest/70 border border-white/5 rounded-lg p-3 space-y-2">
                          <div className="text-secondary font-bold">{t('stt.cjkAlignment')}:</div>
                          <div>
                            <span className="text-secondary font-bold mr-2">{t('stt.cjkMergeApplied')}:</span>
                            {cjkWordDiagnostics.mergeApplied ? t('stt.valueYes') : t('stt.valueNo')}
                          </div>
                          <div>
                            <span className="text-secondary font-bold mr-2">{t('stt.cjkChunkSource')}:</span>
                            {cjkWordDiagnostics.chunkSource}
                          </div>
                          <div>
                            <span className="text-secondary font-bold mr-2">{t('stt.cjkRawWords')}:</span>
                            {cjkWordDiagnostics.rawWordCount}
                          </div>
                          <div>
                            <span className="text-secondary font-bold mr-2">{t('stt.cjkMergedWords')}:</span>
                            {cjkWordDiagnostics.mergedWordCount}
                          </div>
                          {typeof cjkWordDiagnostics.lexicalWordCount === 'number' && (
                            <div>
                              <span className="text-secondary font-bold mr-2">{t('stt.cjkLexicalWords')}:</span>
                              {cjkWordDiagnostics.lexicalWordCount}
                            </div>
                          )}
                          <div>
                            <span className="text-secondary font-bold mr-2">{t('stt.cjkRawSingleChars')}:</span>
                            {cjkWordDiagnostics.rawSingleCharCount}
                          </div>
                          <div>
                            <span className="text-secondary font-bold mr-2">{t('stt.cjkMergedSingleChars')}:</span>
                            {cjkWordDiagnostics.mergedSingleCharCount}
                          </div>
                          {typeof cjkWordDiagnostics.lexicalSingleCharCount === 'number' && (
                            <div>
                              <span className="text-secondary font-bold mr-2">{t('stt.cjkLexicalSingleChars')}:</span>
                              {cjkWordDiagnostics.lexicalSingleCharCount}
                            </div>
                          )}
                          <div>
                            <span className="text-secondary font-bold mr-2">{t('stt.cjkReplacementChars')}:</span>
                            {cjkWordDiagnostics.replacementCharCount}
                          </div>
                          {typeof cjkWordDiagnostics.splitSegmentCount === 'number' && (
                            <div>
                              <span className="text-secondary font-bold mr-2">{t('stt.cjkSplitSegments')}:</span>
                              {cjkWordDiagnostics.splitSegmentCount}
                            </div>
                          )}
                          {typeof cjkWordDiagnostics.usedIntlSegmenter === 'boolean' && (
                            <div>
                              <span className="text-secondary font-bold mr-2">{t('stt.cjkSegmenter')}:</span>
                              {cjkWordDiagnostics.usedIntlSegmenter
                                ? `Intl.Segmenter (${cjkWordDiagnostics.segmenterLocale || 'n/a'})`
                                : t('stt.valueNo')}
                            </div>
                          )}
                        </div>
                      )}
                      {pipelineWarnings.length > 0 && (
                        <div className="text-[11px] text-outline bg-surface-container-lowest/70 border border-white/5 rounded-lg p-3 space-y-2">
                          <div className="text-secondary font-bold">{t('stt.pipelineWarnings')}:</div>
                          {pipelineWarnings.map((warning, idx) => (
                            <div key={`${warning}-${idx}`}>- {warning}</div>
                          ))}
                        </div>
                      )}
                      {diarizationDiagnostics && (
                        <div className="text-[11px] text-outline bg-surface-container-lowest/70 border border-white/5 rounded-lg p-3 space-y-2">
                          <div className="text-secondary font-bold">{diarizationCopy.diagnostics}:</div>
                          <div>{diarizationCopy.provider}: {localizedDiarizationProvider}</div>
                          <div>{diarizationCopy.source}: {localizedDiarizationSource}</div>
                          <div>{diarizationCopy.speechSegments}: {diarizationDiagnostics.speechSegmentCount}</div>
                          <div>{diarizationCopy.windows}: {diarizationDiagnostics.vadWindowCount}</div>
                          {diarizationDiagnostics.selectedPass && (
                            <>
                              <div>{diarizationCopy.speakers}: {diarizationDiagnostics.selectedPass.uniqueSpeakerCount}</div>
                              <div>{diarizationCopy.regions}: {diarizationDiagnostics.selectedPass.regionCount}</div>
                              <div>{diarizationCopy.threshold}: {Number(diarizationDiagnostics.selectedPass.threshold || 0).toFixed(2)}</div>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </section>

          <section className="bg-surface-container p-6 rounded-[28px] flex flex-col h-[600px] min-h-0 shadow-xl">
            <audio
              ref={previewAudioRef}
              src={previewAudioUrl || undefined}
              preload="auto"
              onTimeUpdate={handlePreviewAudioTimeUpdate}
              onEnded={handlePreviewAudioEnded}
              className="hidden"
            />
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between mb-6">
              <h3 className="text-sm font-bold text-secondary flex items-center gap-2">
                {isTranscribing ? <Loader2 className="w-4 h-4 animate-spin text-tertiary" /> : <CheckCircle2 className="w-4 h-4 text-tertiary" />}
                {t('stt.transcriptionOutput')}
              </h3>
              {transcription.length > 0 && !isTranscribing && (
                <div className="flex flex-wrap gap-2 animate-in fade-in zoom-in duration-300">
                  <button
                    type="button"
                    onClick={isEditingTranscription ? handleSaveEditedTranscription : handleStartEditTranscription}
                    disabled={isSavingTranscription || (isEditingTranscription && !canSaveEditedTranscription)}
                    className="flex items-center gap-2 px-3 py-1.5 bg-white/5 hover:bg-white/10 text-outline hover:text-secondary border border-white/5 rounded-lg text-[10px] font-bold transition-all uppercase tracking-widest disabled:opacity-60"
                  >
                    {isEditingTranscription
                      ? (isSavingTranscription ? t('settings.autoSaving') : t('dashboard.saveProject'))
                      : t('settings.edit')}
                  </button>
                  <button 
                    onClick={() => handleDownloadTranscript('txt')}
                    disabled={isEditingTranscription}
                    className="flex items-center gap-2 px-3 py-1.5 bg-white/5 hover:bg-white/10 text-outline hover:text-secondary border border-white/5 rounded-lg text-[10px] font-bold transition-all uppercase tracking-widest"
                  >
                    <Download className="w-3.5 h-3.5" />
                    TXT
                  </button>
                  {hasTimecodes && (
                    <>
                      <button 
                        onClick={() => handleDownloadTranscript('srt')}
                        disabled={isEditingTranscription}
                        className="flex items-center gap-2 px-3 py-1.5 bg-white/5 hover:bg-white/10 text-outline hover:text-secondary border border-white/5 rounded-lg text-[10px] font-bold transition-all uppercase tracking-widest"
                      >
                        <Download className="w-3.5 h-3.5" />
                        SRT
                      </button>
                      <button 
                        onClick={() => handleDownloadTranscript('vtt')}
                        disabled={isEditingTranscription}
                        className="flex items-center gap-2 px-3 py-1.5 bg-white/5 hover:bg-white/10 text-outline hover:text-secondary border border-white/5 rounded-lg text-[10px] font-bold transition-all uppercase tracking-widest"
                      >
                        <Download className="w-3.5 h-3.5" />
                        VTT
                      </button>
                    </>
                  )}
                </div>
              )}
            </div>
            
            <div
              ref={transcriptionScrollRef}
              className="flex-1 min-h-0 bg-surface-container-lowest rounded-2xl p-5 overflow-y-auto border border-white/5 text-sm leading-relaxed text-outline/80 custom-scrollbar"
            >
              {isEditingTranscription ? (
                <SubtitleRowsEditor
                  rows={editingTranscriptionRows}
                  issues={transcriptionEditIssues}
                  onChangeRows={setEditingTranscriptionRows}
                  copy={subtitleEditorCopy}
                />
              ) : (
              <div className="space-y-5">
                {parsedTranscription.map((item) => {
                  const canSeekPlay = item.startSeconds != null && Boolean(previewAudioUrl);
                  return (
                    <div
                      key={item.index}
                      ref={(el) => { transcriptLineRefs.current[item.index] = el; }}
                      className={`group relative rounded-lg px-2 py-1 -mx-2 transition-colors ${
                        activePreviewLine === item.index ? 'bg-primary/10 ring-1 ring-primary/30' : ''
                      }`}
                    >
                      <button
                        type="button"
                        onClick={() => handlePreviewLineClick(item.line, item.index)}
                        disabled={!canSeekPlay}
                        className={`w-full text-left whitespace-pre-wrap pr-12 transition-colors ${
                          canSeekPlay ? 'hover:text-secondary cursor-pointer' : 'cursor-default'
                        }`}
                      >
                        {item.timecode ? (
                          <>
                            <span className="text-primary font-mono text-[10px] mr-3">[{item.timecode}]</span>
                            {item.speaker ? (
                              <>
                                <span className="text-tertiary font-bold mr-2">[{item.speaker}]</span>
                                {item.text}
                              </>
                            ) : item.text}
                          </>
                        ) : (
                          item.text
                        )}
                      </button>
                      {activePreviewLine === item.index && (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            stopPreviewPlayback();
                          }}
                          className="absolute right-2 top-1/2 -translate-y-1/2 inline-flex items-center gap-1 rounded-md border border-white/10 bg-surface-container-high px-2 py-1 text-[10px] font-bold text-secondary hover:bg-white/10 opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto focus:opacity-100 focus:pointer-events-auto transition-opacity"
                          title={t('stt.stopPlayback')}
                        >
                          <Square className="w-3 h-3" />
                          {t('stt.stopPlayback')}
                        </button>
                      )}
                    </div>
                  );
                })}
                {isTranscribing && transcription.length === 0 && (
                  <div className="rounded-[28px] border border-primary/15 bg-primary/[0.04] px-5 py-7 text-center">
                    <div className="mx-auto flex max-w-sm flex-col items-center gap-5">
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
                      <div className="w-full space-y-3 rounded-[24px] border border-white/8 bg-surface-container-lowest/70 px-4 py-4 text-left">
                        {[74, 91, 58, 81, 66].map((width, index) => (
                          <div
                            key={`${width}-${index}`}
                            className="flex items-center gap-3"
                            style={{ animationDelay: `${index * 160}ms` }}
                          >
                            <span className="min-w-[66px] shrink-0 rounded-full border border-primary/12 bg-primary/12 px-2.5 py-1 text-[10px] font-mono text-primary/90 text-center animate-pulse">
                              [{`00:00:${String(index * 3).padStart(2, '0')}`}]
                            </span>
                            <div className="h-3 rounded-full bg-white/6 overflow-hidden flex-1">
                              <div
                                className="h-full rounded-full bg-gradient-to-r from-primary/10 via-primary/35 to-primary/10 animate-pulse"
                                style={{ width: `${width}%`, animationDelay: `${index * 220}ms` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
                {!isTranscribing && transcription.length === 0 && (
                  <p className="text-outline/30 italic text-center mt-20">{t('stt.clickToStart')}</p>
                )}
              </div>
              )}
            </div>
            <button 
              onClick={handleNextStep}
              disabled={!canProceedToNextStep}
              className={`mt-6 w-full py-4 font-bold rounded-2xl border transition-all flex items-center justify-center gap-3 group disabled:opacity-50 ${
                canProceedToNextStep
                  ? 'border-tertiary/30 bg-gradient-to-r from-tertiary/85 via-primary/85 to-primary text-white shadow-[0_18px_45px_rgba(95,224,183,0.22)] hover:brightness-110'
                  : 'border-white/10 bg-white/5 text-secondary hover:bg-white/10'
              }`}
            >
              {nextStepLabel}
              <ArrowRight className={`w-5 h-5 transition-transform ${canProceedToNextStep ? 'group-hover:translate-x-1' : ''}`} />
            </button>
          </section>
        </div>
      </div>
      </div>
      {/* Project asset selection modal */}
      {showAssetModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200">
          <div className="bg-[#13151A] border border-white/10 rounded-[30px] w-full max-w-[1080px] shadow-2xl overflow-hidden flex flex-col max-h-[90vh] animate-in zoom-in-95 duration-200">
            <div className="flex items-start justify-between border-b border-white/5 bg-white/5 px-8 py-6">
              <div className="space-y-3">
                <div className="space-y-1.5">
                  <h2 className="text-[1.75rem] font-bold tracking-tight text-white">{t('stt.modalTitle')}</h2>
                  <p className="text-sm text-outline">{t('stt.modalSubtitle')}</p>
                  <p className="text-sm font-semibold text-primary">{project?.name}</p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full border border-white/8 bg-white/5 px-3 py-1.5 text-[11px] font-semibold text-outline/85">
                    {t('stt.totalFiles').replace('{count}', assets.length.toString())}
                  </span>
                  {project?.notes && (
                    <span className="rounded-full border border-primary/15 bg-primary/10 px-3 py-1.5 text-[11px] font-semibold text-primary/85">
                      {t('dashboard.projectNotes')}
                    </span>
                  )}
                </div>
              </div>
              <button onClick={() => setShowAssetModal(false)} className="rounded-xl p-2 text-outline transition-colors hover:bg-white/8 hover:text-white">
                <X className="h-6 w-6" />
              </button>
            </div>

            <div className="px-8 py-7 overflow-y-auto custom-scrollbar flex-1">
              <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_320px]">
                <div className="space-y-5">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <h3 className="text-sm font-semibold text-white/90">{t('stt.existingAssets')}</h3>
                      <p className="mt-1 text-xs leading-5 text-outline/72">
                        {assets.length === 0 ? t('dashboard.noMaterials') : t('dashboard.preview')}
                      </p>
                    </div>
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="inline-flex shrink-0 items-center gap-2 rounded-xl border border-primary/20 bg-primary/10 px-4 py-2 text-xs font-bold text-primary transition-all hover:bg-primary/15"
                    >
                      <Upload className="h-4 w-4" />
                      {t('stt.uploadNew')}
                    </button>
                  </div>

                  <div className="rounded-[24px] border border-white/6 bg-white/[0.02] min-h-[320px] relative overflow-hidden">
                    {isLoadingAssets && (
                      <div className="absolute inset-0 bg-black/20 backdrop-blur-[1px] flex items-center justify-center z-10">
                        <Loader2 className="w-8 h-8 animate-spin text-primary" />
                      </div>
                    )}
                    <div className="p-4 space-y-3">
                      {assets.length === 0 && !isLoadingAssets && (
                        <div className="px-8 py-14 text-center">
                          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                            <Upload className="h-6 w-6" />
                          </div>
                          <p className="mx-auto max-w-md text-sm leading-6 text-outline/65">{t('dashboard.noMaterials')}</p>
                        </div>
                      )}
                      {assets.map((asset) => (
                        <label
                          key={asset.name}
                          className={`flex items-center justify-between gap-4 rounded-2xl border px-4 py-4 transition-all cursor-pointer ${
                            selectedAssetName === asset.name
                              ? 'border-primary/30 bg-primary/8 shadow-[0_12px_30px_rgba(79,70,229,0.12)]'
                              : 'border-white/5 bg-surface-container-lowest hover:bg-white/5'
                          }`}
                        >
                          <div className="flex min-w-0 items-center gap-4 overflow-hidden">
                            <div className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl ${selectedAssetName === asset.name ? 'bg-primary/15 text-primary' : 'bg-tertiary/10 text-tertiary'}`}>
                              <Music className="w-5 h-5" />
                            </div>
                            <div className="min-w-0">
                              <div className="flex flex-wrap items-center gap-2">
                                <span className="truncate text-sm font-bold text-secondary" title={asset.name}>{asset.name}</span>
                                <span className="rounded-full border border-white/8 bg-white/[0.04] px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest text-outline/75">
                                  {t('dashboard.categoryAudio')}
                                </span>
                              </div>
                              <div className="mt-1 flex flex-wrap gap-2 text-[10px] font-bold uppercase tracking-widest text-outline">
                                <span className="rounded bg-white/5 px-1.5 py-0.5">{asset.size}</span>
                                <span>{asset.date}</span>
                              </div>
                            </div>
                          </div>
                          <div className="flex shrink-0 items-center gap-3">
                            <div className={`w-6 h-6 rounded-full border-2 flex items-center justify-center transition-colors ${selectedAssetName === asset.name ? 'border-primary bg-primary' : 'border-outline/50'}`}>
                              {selectedAssetName === asset.name && <div className="w-3 h-3 bg-white rounded-full" />}
                            </div>
                          </div>
                          <input
                            type="radio"
                            name="assetSelection"
                            className="hidden"
                            checked={selectedAssetName === asset.name}
                            onChange={() => setSelectedAssetName(asset.name)}
                          />
                        </label>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="space-y-5 lg:sticky lg:top-0">
                  <div className="space-y-4 rounded-[24px] border border-white/6 bg-white/[0.02] p-5">
                    <div>
                      <h4 className="text-sm font-semibold text-white/90">{t('stt.uploadNew')}</h4>
                      <p className="mt-1 text-xs leading-5 text-outline/72">{t('stt.onlyAudio')}</p>
                    </div>
                    <input
                      type="file"
                      ref={fileInputRef}
                      className="hidden"
                      accept=".mp3,.wav,.aac,.m4a,.flac"
                      onChange={handleFileUpload}
                    />
                    <div
                      onClick={() => fileInputRef.current?.click()}
                      onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add('border-primary/50', 'bg-primary/5'); }}
                      onDragLeave={(e) => { e.preventDefault(); e.currentTarget.classList.remove('border-primary/50', 'bg-primary/5'); }}
                      onDrop={(e) => { e.currentTarget.classList.remove('border-primary/50', 'bg-primary/5'); handleDrop(e); }}
                      className={`border-2 border-dashed border-white/10 rounded-[24px] px-6 py-10 flex flex-col items-center justify-center text-center hover:border-primary/50 hover:bg-primary/5 transition-all cursor-pointer group relative ${isUploading ? 'pointer-events-none opacity-50' : ''}`}
                    >
                      {isUploading ? (
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
                            {t('dashboard.dragDrop')} <span className="text-primary">{t('dashboard.clickUpload')}</span>
                          </p>
                          <p className="mt-3 text-[10px] font-bold uppercase tracking-widest text-outline">
                            {t('stt.onlyAudio')}
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
                onClick={() => setShowAssetModal(false)}
                className="px-6 py-3 text-sm font-bold text-outline transition-colors hover:bg-white/8 hover:text-white rounded-lg"
              >
                {t('stt.cancel')}
              </button>
              <button 
                onClick={() => setShowAssetModal(false)}
                disabled={!selectedAssetName}
                className="px-8 py-3 bg-primary hover:bg-primary/90 disabled:opacity-50 text-white text-sm font-bold rounded-lg transition-colors shadow-lg shadow-primary/20"
              >
                {t('stt.confirm')}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function StatusItem({
  label,
  progress,
  status,
  tone = 'normal',
}: {
  label: string,
  progress: number,
  status?: string,
  tone?: 'success' | 'error' | 'normal',
}) {
  const textClass = tone === 'success' ? 'text-tertiary' : (tone === 'error' ? 'text-error' : 'text-primary');
  const barClass = tone === 'success' ? 'bg-tertiary' : (tone === 'error' ? 'bg-error' : 'bg-primary');

  return (
    <div className="space-y-3">
      <div className="flex justify-between items-start gap-4 text-[11px] font-bold tracking-wider">
        <span className="text-outline uppercase shrink-0">{label}</span>
        <span className={`${textClass} text-right leading-relaxed max-w-[70%] break-words`}>
          {status || `${Math.round(progress)}%`}
        </span>
      </div>
      <div className="h-1.5 bg-surface-container-lowest rounded-full overflow-hidden">
        <div className={`h-full transition-all duration-500 ${barClass}`} style={{ width: `${progress}%` }} />
      </div>
    </div>
  );
}
