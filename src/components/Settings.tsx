import React from 'react';
import { Save, Plus, Globe, Key, Trash2, Info, Edit2, Check, ChevronDown, Zap, Loader2, AlertCircle, CheckCircle2, X } from 'lucide-react';
import { ApiConfig, ApiModelRequestOptions, Project } from '../types';
import { useLanguage } from '../i18n/LanguageContext';
import { Language } from '../i18n/translations';
import { sanitizeInput, isValidUrl, isValidApiKey, maskApiKey, isMaskedApiKey } from '../utils/security';
import { getJson, HttpRequestError, postJson } from '../utils/http_client';

type LocalModelType = 'asr' | 'translate';

interface LocalModelEntry {
  id: string;
  type: LocalModelType;
  name: string;
  repoId: string;
  sourceFormat?: 'openvino-ir' | 'onnx' | 'tensorflow' | 'tensorflow-lite' | 'paddle' | 'pytorch' | 'jax-flax' | 'keras' | 'gguf' | 'unknown';
  conversionMethod?:
    | 'direct-download'
    | 'openvino-convert-model'
    | 'optimum-export-openvino'
    | 'openvino-ctc-asr-export'
    | 'openvino-qwen3-asr-export'
    | 'unsupported';
  runtimeLayout?:
    | 'asr-whisper'
    | 'asr-ctc'
    | 'asr-qwen3-official'
    | 'translate-llm'
    | 'translate-seq2seq'
    | 'translate-vlm';
  runtime?:
    | 'openvino-whisper-node'
    | 'openvino-ctc-asr'
    | 'openvino-qwen3-asr'
    | 'openvino-seq2seq-translate'
    | 'openvino-llm-node';
  selected: boolean;
  installError?: string | null;
}

interface OpenvinoStatus {
  node: {
    available: boolean;
    version?: string;
    error?: string;
  };
  genai: {
    available: boolean;
    version?: string;
    whisperPipelineAvailable: boolean;
    llmPipelineAvailable: boolean;
    vlmPipelineAvailable: boolean;
    error?: string;
  };
  asr: {
    ready: boolean;
    whisperPipelineAvailable: boolean;
    qwenExplicitKvAvailable: boolean;
    qwenOfficialAvailable: boolean;
    error?: string;
  };
  helper: {
    path: string;
    exists: boolean;
    healthy: boolean;
    error?: string;
  };
}

interface PyannoteStatus {
  tokenConfigured: boolean;
  ready: boolean;
  state: 'ready' | 'partial' | 'missing';
  installing: boolean;
  lastError: string | null;
}

function getLocalModelCopy(language: Language) {
  const maps = {
    'zh-tw': {
      installTitle: '從 Hugging Face 安裝 OpenVINO 模型',
      installType: '模型類型',
      installRepo: 'HF 模型名稱',
      installPlaceholder: '例如：OpenVINO/whisper-large-v3-int8-ov',
      installHint: '所有模型都會從 Hugging Face 下載，非 OV 來源會先轉成 OpenVINO INT8，再驗證可用 runtime。',
      installHintDetail: '只有轉換後能符合 ArcSub runtime 目錄結構的模型才會被安裝。',
      openvinoGenai: 'OpenVINO GenAI',
      asrRuntime: 'ASR Runtime',
      version: '版本',
      whisperSupport: 'Whisper Runtime',
      llmSupport: 'LLM 支援',
      vlmSupport: 'VLM 支援',
      sourceFormat: '來源格式',
      conversionMethod: '轉換方法',
      runtimeLayout: 'Runtime 佈局',
      runtimeEngine: 'Runtime',
      noInstalledModels: '尚未安裝任何本地模型。',
      selected: '目前預設',
      setDefault: '設為預設',
      repoId: 'HF Repo',
      installTypeAsr: 'ASR 模型',
      installTypeTranslate: '翻譯模型',
      installInputRequired: '請輸入 Hugging Face 模型名稱。',
    },
    'zh-cn': {
      installTitle: '从 Hugging Face 安装 OpenVINO 模型',
      installType: '模型类型',
      installRepo: 'HF 模型名称',
      installPlaceholder: '例如：OpenVINO/whisper-large-v3-int8-ov',
      installHint: '所有模型都会从 Hugging Face 下载，非 OV 来源会先转换成 OpenVINO INT8，再验证可用 runtime。',
      installHintDetail: '只有转换后能符合 ArcSub runtime 目录结构的模型才会被安装。',
      openvinoGenai: 'OpenVINO GenAI',
      asrRuntime: 'ASR Runtime',
      version: '版本',
      whisperSupport: 'Whisper Runtime',
      llmSupport: 'LLM 支持',
      vlmSupport: 'VLM 支持',
      sourceFormat: '来源格式',
      conversionMethod: '转换方法',
      runtimeLayout: 'Runtime 布局',
      runtimeEngine: 'Runtime',
      noInstalledModels: '尚未安装任何本地模型。',
      selected: '当前默认',
      setDefault: '设为默认',
      repoId: 'HF Repo',
      installTypeAsr: 'ASR 模型',
      installTypeTranslate: '翻译模型',
      installInputRequired: '请输入 Hugging Face 模型名称。',
    },
    en: {
      installTitle: 'Install OpenVINO Models From Hugging Face',
      installType: 'Model Type',
      installRepo: 'HF Model ID',
      installPlaceholder: 'e.g. OpenVINO/whisper-large-v3-int8-ov',
      installHint: 'All models are fetched from Hugging Face. Non-OV sources are converted into OpenVINO INT8 first, then validated against ArcSub runtime layouts.',
      installHintDetail: 'A model is installed only if the converted output matches a supported ArcSub runtime layout.',
      openvinoGenai: 'OpenVINO GenAI',
      asrRuntime: 'ASR Runtime',
      version: 'Version',
      whisperSupport: 'Whisper Runtime',
      llmSupport: 'LLM Support',
      vlmSupport: 'VLM Support',
      sourceFormat: 'Source Format',
      conversionMethod: 'Conversion',
      runtimeLayout: 'Runtime Layout',
      runtimeEngine: 'Runtime',
      noInstalledModels: 'No local models installed yet.',
      selected: 'Default',
      setDefault: 'Set Default',
      repoId: 'HF Repo',
      installTypeAsr: 'ASR Model',
      installTypeTranslate: 'Translation Model',
      installInputRequired: 'Enter a Hugging Face model id.',
    },
    jp: {
      installTitle: 'Hugging Face から OpenVINO モデルをインストール',
      installType: 'モデル種別',
      installRepo: 'HF モデル名',
      installPlaceholder: '例：OpenVINO/whisper-large-v3-int8-ov',
      installHint: 'すべてのモデルは Hugging Face から取得され、OV 以外はまず OpenVINO INT8 に変換してから runtime 適合性を検証します。',
      installHintDetail: '変換後の出力が ArcSub の runtime レイアウトに一致した場合のみインストールされます。',
      openvinoGenai: 'OpenVINO GenAI',
      asrRuntime: 'ASR Runtime',
      version: 'バージョン',
      whisperSupport: 'Whisper Runtime',
      llmSupport: 'LLM 対応',
      vlmSupport: 'VLM 対応',
      sourceFormat: '入力形式',
      conversionMethod: '変換方法',
      runtimeLayout: 'Runtime レイアウト',
      runtimeEngine: 'Runtime',
      noInstalledModels: 'まだローカルモデルがありません。',
      selected: '既定値',
      setDefault: '既定に設定',
      repoId: 'HF Repo',
      installTypeAsr: 'ASR モデル',
      installTypeTranslate: '翻訳モデル',
      installInputRequired: 'Hugging Face のモデル名を入力してください。',
    },
    de: {
      installTitle: 'OpenVINO-Modelle von Hugging Face installieren',
      installType: 'Modelltyp',
      installRepo: 'HF-Modellname',
      installPlaceholder: 'z. B. OpenVINO/whisper-large-v3-int8-ov',
      installHint: 'Alle Modelle werden von Hugging Face geladen. Nicht-OV-Quellen werden zuerst nach OpenVINO INT8 konvertiert und dann gegen ArcSub-Runtime-Layouts validiert.',
      installHintDetail: 'Installiert wird nur, wenn die konvertierten Dateien zu einem unterstutzten ArcSub-Runtime-Layout passen.',
      openvinoGenai: 'OpenVINO GenAI',
      asrRuntime: 'ASR Runtime',
      version: 'Version',
      whisperSupport: 'Whisper-Runtime',
      llmSupport: 'LLM-Unterstutzung',
      vlmSupport: 'VLM-Unterstutzung',
      sourceFormat: 'Quellformat',
      conversionMethod: 'Konvertierung',
      runtimeLayout: 'Runtime-Layout',
      runtimeEngine: 'Runtime',
      noInstalledModels: 'Noch keine lokalen Modelle installiert.',
      selected: 'Standard',
      setDefault: 'Als Standard setzen',
      repoId: 'HF Repo',
      installTypeAsr: 'ASR-Modell',
      installTypeTranslate: 'Ubersetzungsmodell',
      installInputRequired: 'Geben Sie eine Hugging-Face-Modellkennung ein.',
    },
  } as const;

  return maps[language];
}

function getPyannoteSettingsCopy(language: Language) {
  const maps = {
    'zh-tw': {
      title: 'Pyannote 語者分離',
      subtitle: '管理 Hugging Face 金鑰與 pyannote 語者分離資產。',
      tokenLabel: 'HF_TOKEN',
      tokenPlaceholder: '貼上 Hugging Face Access Token',
      tokenHint: '先在 Hugging Face 接受 pyannote 模型授權，再輸入 HF_TOKEN。',
      saveToken: '儲存金鑰',
      install: '安裝 Pyannote',
      installed: '已安裝',
      installing: '安裝中...',
      ready: '已就緒',
      partial: '部分完成',
      missing: '尚未安裝',
      tokenReady: '已設定金鑰',
      tokenMissing: '尚未設定金鑰',
      refresh: '重新整理',
      saved: 'HF_TOKEN 已儲存。',
      saveFailed: '儲存 HF_TOKEN 失敗。',
      installFailed: 'Pyannote 安裝失敗。',
    },
    'zh-cn': {
      title: 'Pyannote 语者分离',
      subtitle: '管理 Hugging Face 金钥与 pyannote 语者分离资源。',
      tokenLabel: 'HF_TOKEN',
      tokenPlaceholder: '粘贴 Hugging Face Access Token',
      tokenHint: '请先在 Hugging Face 接受 pyannote 模型授权，再输入 HF_TOKEN。',
      saveToken: '保存金钥',
      install: '安装 Pyannote',
      installed: '已安装',
      installing: '安装中...',
      ready: '已就绪',
      partial: '部分完成',
      missing: '尚未安装',
      tokenReady: '已配置金钥',
      tokenMissing: '尚未配置金钥',
      refresh: '刷新',
      saved: 'HF_TOKEN 已保存。',
      saveFailed: '保存 HF_TOKEN 失败。',
      installFailed: 'Pyannote 安装失败。',
    },
    en: {
      title: 'Pyannote Diarization',
      subtitle: 'Manage the Hugging Face token and pyannote diarization assets.',
      tokenLabel: 'HF_TOKEN',
      tokenPlaceholder: 'Paste your Hugging Face access token',
      tokenHint: 'Accept the pyannote model access terms on Hugging Face before entering HF_TOKEN.',
      saveToken: 'Save Token',
      install: 'Install Pyannote',
      installed: 'Installed',
      installing: 'Installing...',
      ready: 'Ready',
      partial: 'Partial',
      missing: 'Missing',
      tokenReady: 'Token configured',
      tokenMissing: 'Token missing',
      refresh: 'Refresh',
      saved: 'HF_TOKEN saved.',
      saveFailed: 'Failed to save HF_TOKEN.',
      installFailed: 'Pyannote installation failed.',
    },
    jp: {
      title: 'Pyannote 話者分離',
      subtitle: 'Hugging Face トークンと pyannote 話者分離アセットを管理します。',
      tokenLabel: 'HF_TOKEN',
      tokenPlaceholder: 'Hugging Face Access Token を貼り付け',
      tokenHint: '先に Hugging Face 上で pyannote モデルの利用承認を済ませてください。',
      saveToken: 'トークン保存',
      install: 'Pyannote をインストール',
      installed: 'インストール済み',
      installing: 'インストール中...',
      ready: '準備完了',
      partial: '一部完了',
      missing: '未インストール',
      tokenReady: 'トークン設定済み',
      tokenMissing: 'トークン未設定',
      refresh: '再読み込み',
      saved: 'HF_TOKEN を保存しました。',
      saveFailed: 'HF_TOKEN の保存に失敗しました。',
      installFailed: 'Pyannote のインストールに失敗しました。',
    },
    de: {
      title: 'Pyannote-Diarisierung',
      subtitle: 'Verwalten Sie das Hugging-Face-Token und die pyannote-Diarisierungsassets.',
      tokenLabel: 'HF_TOKEN',
      tokenPlaceholder: 'Hugging Face Access Token einfuegen',
      tokenHint: 'Akzeptieren Sie zuerst die pyannote-Modellbedingungen auf Hugging Face.',
      saveToken: 'Token speichern',
      install: 'Pyannote installieren',
      installed: 'Installiert',
      installing: 'Installation laeuft...',
      ready: 'Bereit',
      partial: 'Teilweise',
      missing: 'Fehlt',
      tokenReady: 'Token konfiguriert',
      tokenMissing: 'Token fehlt',
      refresh: 'Aktualisieren',
      saved: 'HF_TOKEN gespeichert.',
      saveFailed: 'HF_TOKEN konnte nicht gespeichert werden.',
      installFailed: 'Pyannote-Installation fehlgeschlagen.',
    },
  } as const;

  return maps[language];
}

function isHiddenSettingsLocalModel(_model: Pick<LocalModelEntry, 'repoId' | 'name'>) {
  return false;
}

function formatLocalModelSourceFormat(language: Language, value?: LocalModelEntry['sourceFormat']) {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'openvino-ir') return 'OpenVINO IR';
  if (normalized === 'onnx') return 'ONNX';
  if (normalized === 'tensorflow') return language === 'jp' ? 'TensorFlow' : 'TensorFlow';
  if (normalized === 'tensorflow-lite') return 'TensorFlow Lite';
  if (normalized === 'paddle') return 'PaddlePaddle';
  if (normalized === 'pytorch') return 'PyTorch';
  if (normalized === 'jax-flax') return 'JAX / Flax';
  if (normalized === 'keras') return 'Keras';
  if (normalized === 'gguf') return 'GGUF';
  return normalized ? normalized : 'Unknown';
}

function formatLocalModelConversionMethod(language: Language, value?: LocalModelEntry['conversionMethod']) {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'direct-download') {
    return language === 'zh-tw' ? 'OV 直裝' :
      language === 'zh-cn' ? 'OV 直装' :
      language === 'jp' ? 'OV 直接導入' :
      language === 'de' ? 'OV direkt' :
      'OV Direct';
  }
  if (normalized === 'openvino-convert-model') {
    return language === 'zh-tw' ? 'ov.convert_model' :
      language === 'zh-cn' ? 'ov.convert_model' :
      language === 'jp' ? 'ov.convert_model' :
      language === 'de' ? 'ov.convert_model' :
      'ov.convert_model';
  }
  if (normalized === 'optimum-export-openvino') {
    return language === 'zh-tw' ? 'Optimum Export' :
      language === 'zh-cn' ? 'Optimum Export' :
      language === 'jp' ? 'Optimum Export' :
      language === 'de' ? 'Optimum Export' :
      'Optimum Export';
  }
  if (normalized === 'openvino-ctc-asr-export') {
    return 'CTC ASR Export';
  }
  if (normalized === 'openvino-qwen3-asr-export') {
    return 'Qwen3-ASR Export';
  }
  return normalized ? normalized : 'Unknown';
}

function formatLocalModelRuntimeLayout(language: Language, value?: LocalModelEntry['runtimeLayout']) {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'asr-whisper') return 'ASR / Whisper';
  if (normalized === 'asr-ctc') return 'ASR / CTC';
  if (normalized === 'asr-qwen3-official') return 'ASR / Qwen3 Official';
  if (normalized === 'translate-llm') return 'Translate / LLM';
  if (normalized === 'translate-seq2seq') return 'Translate / Seq2Seq';
  if (normalized === 'translate-vlm') return 'Translate / VLM';
  return normalized ? normalized : 'Unknown';
}

function formatLocalModelRuntimeEngine(value?: LocalModelEntry['runtime']) {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'openvino-whisper-node') return 'OpenVINO Whisper';
  if (normalized === 'openvino-ctc-asr') return 'OpenVINO CTC-ASR';
  if (normalized === 'openvino-qwen3-asr') return 'OpenVINO Qwen3-ASR';
  if (normalized === 'openvino-seq2seq-translate') return 'OpenVINO Seq2Seq';
  if (normalized === 'openvino-llm-node') return 'OpenVINO LLM';
  return normalized ? normalized : 'Unknown';
}

function SummaryCard({
  icon,
  label,
  value,
  hint,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div className="rounded-2xl border border-white/5 bg-surface-container p-5 shadow-[0_18px_50px_rgba(7,10,24,0.18)]">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="text-[10px] font-bold uppercase tracking-widest text-outline">{label}</div>
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary-container/10 text-primary">
          {icon}
        </div>
      </div>
      <div className="text-xl font-bold text-secondary">{value}</div>
      <div className="mt-2 text-xs leading-relaxed text-outline">{hint}</div>
    </div>
  );
}

export default function Settings({ project }: { project: Project | null }) {
  const { language, setLanguage, t } = useLanguage();
  const localCopy = React.useMemo(() => getLocalModelCopy(language), [language]);
  const pyannoteCopy = React.useMemo(() => getPyannoteSettingsCopy(language), [language]);

  const [asrModels, setAsrModels] = React.useState<ApiConfig[]>([]);
  const [isAddingAsr, setIsAddingAsr] = React.useState(false);

  const [translateModels, setTranslateModels] = React.useState<ApiConfig[]>([]);
  const [isAddingTranslate, setIsAddingTranslate] = React.useState(false);

  const [saveStatus, setSaveStatus] = React.useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [settingsLoading, setSettingsLoading] = React.useState(false);
  const [settingsHydrated, setSettingsHydrated] = React.useState(false);
  const [localModels, setLocalModels] = React.useState<LocalModelEntry[]>([]);
  const [localSelection, setLocalSelection] = React.useState({ asrSelectedId: '', translateSelectedId: '' });
  const [openvinoStatus, setOpenvinoStatus] = React.useState<OpenvinoStatus | null>(null);
  const [pyannoteStatus, setPyannoteStatus] = React.useState<PyannoteStatus | null>(null);
  const [pyannoteTokenInput, setPyannoteTokenInput] = React.useState('');
  const [pyannoteBusy, setPyannoteBusy] = React.useState(false);
  const [pyannoteMessage, setPyannoteMessage] = React.useState<string | null>(null);
  const [localStatusLoading, setLocalStatusLoading] = React.useState(false);
  const [localCatalogLoading, setLocalCatalogLoading] = React.useState(false);
  const [localInstallType, setLocalInstallType] = React.useState<LocalModelType>('asr');
  const [localRepoId, setLocalRepoId] = React.useState('');
  const [localInstallBusy, setLocalInstallBusy] = React.useState(false);
  const [localRemovingModelId, setLocalRemovingModelId] = React.useState<string | null>(null);
  const [localError, setLocalError] = React.useState<string | null>(null);
  const [localPanelActive, setLocalPanelActive] = React.useState(false);
  const saveStatusTimeoutRef = React.useRef<number | null>(null);
  const initialBootstrapRef = React.useRef(false);
  const localPanelBootstrapRef = React.useRef(false);
  const settingsRequestRef = React.useRef<Promise<void> | null>(null);
  const localModelsRequestRef = React.useRef<Promise<void> | null>(null);
  const settingsHydratedRef = React.useRef(false);
  const settingsLoadSeqRef = React.useRef(0);
  const localModelsLoadSeqRef = React.useRef(0);
  const mountedRef = React.useRef(true);
  const settingsAbortRef = React.useRef<AbortController | null>(null);
  const localStatusAbortRef = React.useRef<AbortController | null>(null);
  const localCatalogAbortRef = React.useRef<AbortController | null>(null);

  const clearSaveStatusTimer = React.useCallback(() => {
    if (saveStatusTimeoutRef.current !== null) {
      window.clearTimeout(saveStatusTimeoutRef.current);
      saveStatusTimeoutRef.current = null;
    }
  }, []);

  const scheduleSaveStatusReset = React.useCallback((delay = 2500) => {
    clearSaveStatusTimer();
    saveStatusTimeoutRef.current = window.setTimeout(() => {
      setSaveStatus('idle');
      saveStatusTimeoutRef.current = null;
    }, delay);
  }, [clearSaveStatusTimer]);

  const loadSettings = React.useCallback(async () => {
    if (settingsRequestRef.current) {
      return settingsRequestRef.current;
    }

    const loadSeq = settingsLoadSeqRef.current + 1;
    settingsLoadSeqRef.current = loadSeq;
    let requestTask: Promise<void>;

    requestTask = (async () => {
      settingsAbortRef.current?.abort();
      const controller = new AbortController();
      settingsAbortRef.current = controller;
      setSettingsLoading(true);
      try {
        const data = await getJson<{ asrModels?: ApiConfig[]; translateModels?: ApiConfig[] }>('/api/settings', {
          signal: controller.signal,
          cache: 'no-store',
          timeoutMs: 8000,
          retries: 1,
          dedupe: false,
          cancelPreviousKey: 'settings:full',
        });
        if (!mountedRef.current || controller.signal.aborted || settingsLoadSeqRef.current !== loadSeq) return;
        setAsrModels(Array.isArray(data.asrModels) ? data.asrModels : []);
        setTranslateModels(Array.isArray(data.translateModels) ? data.translateModels : []);
        setSettingsHydrated(true);
      } catch (err) {
        if (controller.signal.aborted || settingsLoadSeqRef.current !== loadSeq) return;
        console.error('Failed to load settings', err);
      } finally {
        if (mountedRef.current && settingsLoadSeqRef.current === loadSeq) {
          setSettingsLoading(false);
        }
        if (settingsAbortRef.current === controller) {
          settingsAbortRef.current = null;
        }
        if (settingsRequestRef.current === requestTask) {
          settingsRequestRef.current = null;
        }
      }
    })();

    settingsRequestRef.current = requestTask;
    return requestTask;
  }, []);

  React.useEffect(() => {
    settingsHydratedRef.current = settingsHydrated;
  }, [settingsHydrated]);

  const loadLocalModels = React.useCallback(async () => {
    if (localModelsRequestRef.current) {
      return localModelsRequestRef.current;
    }

    const loadSeq = localModelsLoadSeqRef.current + 1;
    localModelsLoadSeqRef.current = loadSeq;
    let requestTask: Promise<void>;

    requestTask = (async () => {
      localStatusAbortRef.current?.abort();
      localCatalogAbortRef.current?.abort();
      const statusController = new AbortController();
      const catalogController = new AbortController();
      localStatusAbortRef.current = statusController;
      localCatalogAbortRef.current = catalogController;
      setLocalStatusLoading(true);
      setLocalCatalogLoading(true);
      const loadingSafeguard = window.setTimeout(() => {
        if (!mountedRef.current) return;
        setLocalStatusLoading(false);
        setLocalCatalogLoading(false);
        if (localModelsRequestRef.current === requestTask) {
          localModelsRequestRef.current = null;
        }
      }, 20_000);

      try {
        const statusTask = (async () => {
          try {
            const [statusData, pyannoteData] = await Promise.all([
              getJson<OpenvinoStatus>('/api/openvino/status', {
                signal: statusController.signal,
                cache: 'no-store',
                timeoutMs: 10_000,
                retries: 1,
                dedupe: false,
                cancelPreviousKey: 'settings:openvino-status',
              }),
              getJson<PyannoteStatus>('/api/runtime/pyannote/status', {
                signal: statusController.signal,
                cache: 'no-store',
                timeoutMs: 10_000,
                retries: 1,
                dedupe: false,
                cancelPreviousKey: 'settings:pyannote-status',
              }),
            ]);
            if (!mountedRef.current || statusController.signal.aborted || localModelsLoadSeqRef.current !== loadSeq) return;
            setOpenvinoStatus(statusData);
            setPyannoteStatus(pyannoteData);
          } catch (error) {
            if (statusController.signal.aborted || localModelsLoadSeqRef.current !== loadSeq) return;
            console.error('Failed to load OpenVINO status', error);
          } finally {
            if (mountedRef.current) {
              setLocalStatusLoading(false);
            }
            if (localStatusAbortRef.current === statusController) {
              localStatusAbortRef.current = null;
            }
          }
        })();

        const modelsTask = (async () => {
          try {
            const data = await getJson<{ catalog?: LocalModelEntry[]; selection?: { asrSelectedId?: string; translateSelectedId?: string } }>(
              '/api/local-models',
              {
                signal: catalogController.signal,
                cache: 'no-store',
                timeoutMs: 10_000,
                retries: 1,
                dedupe: false,
                cancelPreviousKey: 'settings:local-models',
              }
            );
            if (!mountedRef.current || catalogController.signal.aborted || localModelsLoadSeqRef.current !== loadSeq) return;
            setLocalModels(Array.isArray(data.catalog) ? data.catalog : []);
            setLocalSelection(data.selection || { asrSelectedId: '', translateSelectedId: '' });
          } catch (error) {
            if (catalogController.signal.aborted || localModelsLoadSeqRef.current !== loadSeq) return;
            console.error('Failed to load local models', error);
          } finally {
            if (mountedRef.current) {
              setLocalCatalogLoading(false);
            }
            if (localCatalogAbortRef.current === catalogController) {
              localCatalogAbortRef.current = null;
            }
          }
        })();

        await Promise.allSettled([statusTask, modelsTask]);
      } finally {
        window.clearTimeout(loadingSafeguard);
        if (localModelsRequestRef.current === requestTask) {
          localModelsRequestRef.current = null;
        }
      }
    })();

    localModelsRequestRef.current = requestTask;
    return requestTask;
  }, []);

  React.useEffect(() => {
    if (initialBootstrapRef.current) return;
    initialBootstrapRef.current = true;
    void loadSettings();
  }, [loadSettings]);

  React.useEffect(() => {
    if (localPanelBootstrapRef.current || !localPanelActive) return;
    localPanelBootstrapRef.current = true;
    void loadLocalModels();
  }, [localPanelActive, loadLocalModels]);

  React.useEffect(() => {
    if (typeof window === 'undefined') {
      setLocalPanelActive(true);
      return;
    }

    const idle = (window as any).requestIdleCallback as ((callback: () => void, options?: { timeout: number }) => number) | undefined;
    const cancelIdle = (window as any).cancelIdleCallback as ((id: number) => void) | undefined;
    if (idle && cancelIdle) {
      const idleId = idle(() => setLocalPanelActive(true), { timeout: 350 });
      return () => cancelIdle(idleId);
    }

    const timeoutId = window.setTimeout(() => setLocalPanelActive(true), 200);
    return () => window.clearTimeout(timeoutId);
  }, []);

  React.useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      initialBootstrapRef.current = false;
      localPanelBootstrapRef.current = false;
      clearSaveStatusTimer();
      settingsAbortRef.current?.abort();
      localStatusAbortRef.current?.abort();
      localCatalogAbortRef.current?.abort();
      settingsAbortRef.current = null;
      localStatusAbortRef.current = null;
      localCatalogAbortRef.current = null;
      settingsRequestRef.current = null;
      localModelsRequestRef.current = null;
    };
  }, [clearSaveStatusTimer]);

  const persistSettings = React.useCallback(async (next: Partial<{
    asrModels: ApiConfig[];
    translateModels: ApiConfig[];
    interfaceLanguage: Language;
  }>) => {
    if (!settingsHydrated) {
      try {
        await loadSettings();
      } catch {
        return;
      }
      if (!settingsHydratedRef.current) {
        return;
      }
    }
    setSaveStatus('saving');
    try {
      const saved = await postJson<{
        asrModels?: ApiConfig[];
        translateModels?: ApiConfig[];
        interfaceLanguage?: Language;
      }>('/api/settings', next, {
        timeoutMs: 10_000,
      });
      if (Object.prototype.hasOwnProperty.call(next, 'asrModels')) {
        setAsrModels(Array.isArray(saved.asrModels) ? saved.asrModels : (next.asrModels || []));
      }
      if (Object.prototype.hasOwnProperty.call(next, 'translateModels')) {
        setTranslateModels(Array.isArray(saved.translateModels) ? saved.translateModels : (next.translateModels || []));
      }
      setSaveStatus('success');
      scheduleSaveStatusReset();
    } catch (error) {
      console.error('Failed to save settings', error);
      setSaveStatus('error');
      scheduleSaveStatusReset(4000);
    }
  }, [loadSettings, scheduleSaveStatusReset, settingsHydrated]);

  const handleGlobalSave = async () => {
    await persistSettings({
      asrModels,
      translateModels,
      interfaceLanguage: language,
    });
  };

  const handleSaveAsr = (updatedModel: ApiConfig) => {
    const nextAsrModels = asrModels.map(m => m.id === updatedModel.id ? updatedModel : m);
    setAsrModels(nextAsrModels);
    void persistSettings({
      asrModels: nextAsrModels,
    });
  };
  const handleDeleteAsr = (id: string) => {
    const nextAsrModels = asrModels.filter(m => m.id !== id);
    setAsrModels(nextAsrModels);
    void persistSettings({
      asrModels: nextAsrModels,
    });
  };
  const handleAddAsr = (newModel: ApiConfig) => {
    const nextAsrModels = [...asrModels, { ...newModel, id: Date.now().toString() }];
    setAsrModels(nextAsrModels);
    setIsAddingAsr(false);
    void persistSettings({
      asrModels: nextAsrModels,
    });
  };

  const handleSaveTranslate = (updatedModel: ApiConfig) => {
    const nextTranslateModels = translateModels.map(m => m.id === updatedModel.id ? updatedModel : m);
    setTranslateModels(nextTranslateModels);
    void persistSettings({
      translateModels: nextTranslateModels,
    });
  };
  const handleDeleteTranslate = (id: string) => {
    const nextTranslateModels = translateModels.filter(m => m.id !== id);
    setTranslateModels(nextTranslateModels);
    void persistSettings({
      translateModels: nextTranslateModels,
    });
  };
  const handleAddTranslate = (newModel: ApiConfig) => {
    const nextTranslateModels = [...translateModels, { ...newModel, id: Date.now().toString() }];
    setTranslateModels(nextTranslateModels);
    setIsAddingTranslate(false);
    void persistSettings({
      translateModels: nextTranslateModels,
    });
  };

  const visibleLocalModels = localModels.filter((model) => !isHiddenSettingsLocalModel(model));
  const localAsrModels = visibleLocalModels.filter((model) => model.type === 'asr');
  const localTranslateModels = visibleLocalModels.filter((model) => model.type === 'translate');
  const selectedLocalInstallError = localError || visibleLocalModels.find((model) => model.installError)?.installError || null;
  const isLocalSectionLoading = localStatusLoading || localCatalogLoading;
  const localLoadingHint = localStatusLoading && localCatalogLoading
    ? t('settings.localLoadingAll')
    : localStatusLoading
      ? t('settings.localLoadingStatus')
      : localCatalogLoading
        ? t('settings.localLoadingCatalog')
        : '';
  const localRuntimeReady = Boolean(openvinoStatus?.node?.available && (openvinoStatus?.genai?.available || openvinoStatus?.asr?.ready));
  const localAsrInstallReady = Boolean(openvinoStatus?.asr?.ready);
  const localTranslateInstallReady = Boolean(
    openvinoStatus?.node?.available &&
    openvinoStatus?.genai?.available &&
    (openvinoStatus?.genai?.llmPipelineAvailable || openvinoStatus?.genai?.vlmPipelineAvailable)
  );
  const localInstallRuntimeReady = localInstallType === 'asr' ? localAsrInstallReady : localTranslateInstallReady;
  const localInstallRuntimeBlockedHint = localInstallType === 'asr'
    ? `${localCopy.asrRuntime}: ${t('settings.localStatusUnavailable')}`
    : `${localCopy.openvinoGenai}: ${t('settings.localStatusUnavailable')}`;
  const currentLanguageLabel =
    language === 'zh-tw' ? t('lang.zh-tw') :
    language === 'zh-cn' ? t('lang.zh-cn') :
    language === 'jp' ? t('lang.jp') :
    language === 'de' ? t('lang.de') :
    t('lang.en');
  const saveStatusLabel =
    saveStatus === 'saving'
      ? t('settings.autoSaving')
      : saveStatus === 'success'
        ? t('settings.autoSaved')
        : saveStatus === 'error'
          ? t('settings.saveRetry')
          : t('settings.autoSaveEnabled');

  const handleSelectLocalModel = async (type: LocalModelType, modelId: string) => {
    setLocalError(null);
    try {
      const data = await postJson<{ catalog?: LocalModelEntry[]; selection?: { asrSelectedId?: string; translateSelectedId?: string } }>(
        '/api/local-models/select',
        { type, modelId },
        { timeoutMs: 20_000 }
      );
      setLocalModels(Array.isArray(data.catalog) ? data.catalog : []);
      setLocalSelection(data.selection || { asrSelectedId: '', translateSelectedId: '' });
    } catch (error: any) {
      setLocalError(String(error?.message || t('settings.localSelectFailed')));
    }
  };

  const handleInstallLocalModel = async () => {
    const repoId = localRepoId.trim();
    if (!repoId) {
      setLocalError(localCopy.installInputRequired);
      return;
    }
    if (!localInstallRuntimeReady) {
      setLocalError(localInstallRuntimeBlockedHint);
      return;
    }
    setLocalInstallBusy(true);
    setLocalError(null);
    try {
      const data = await postJson<{ catalog?: LocalModelEntry[]; selection?: { asrSelectedId?: string; translateSelectedId?: string } }>(
        '/api/local-models/install',
        { type: localInstallType, repoId },
        { timeoutMs: 300_000, retries: 0 }
      );
      setLocalModels(Array.isArray(data.catalog) ? data.catalog : []);
      setLocalSelection(data.selection || { asrSelectedId: '', translateSelectedId: '' });
      setLocalRepoId('');
    } catch (error: any) {
      setLocalError(String(error?.message || t('settings.localInstallFailed')));
    } finally {
      setLocalInstallBusy(false);
    }
  };

  const handleRemoveLocalModel = async (modelId: string) => {
    setLocalRemovingModelId(modelId);
    setLocalError(null);
    try {
      const data = await postJson<{ catalog?: LocalModelEntry[]; selection?: { asrSelectedId?: string; translateSelectedId?: string } }>(
        '/api/local-models/remove',
        { modelId },
        { timeoutMs: 60_000 }
      );
      setLocalModels(Array.isArray(data.catalog) ? data.catalog : []);
      setLocalSelection(data.selection || { asrSelectedId: '', translateSelectedId: '' });
    } catch (error: any) {
      setLocalError(String(error?.message || t('settings.localRemoveFailed')));
    } finally {
      setLocalRemovingModelId(null);
    }
  };

  const handleSavePyannoteToken = async () => {
    const token = pyannoteTokenInput.trim();
    if (!token) {
      setPyannoteMessage(pyannoteCopy.saveFailed);
      return;
    }
    setPyannoteBusy(true);
    setPyannoteMessage(null);
    try {
      const data = await postJson<{ success?: boolean; status?: PyannoteStatus }>(
        '/api/runtime/pyannote/token',
        { token },
        { timeoutMs: 30_000, retries: 0 }
      );
      setPyannoteStatus(data.status || null);
      setPyannoteTokenInput('');
      setPyannoteMessage(pyannoteCopy.saved);
      await loadLocalModels();
    } catch (error: any) {
      setPyannoteMessage(String(error?.message || pyannoteCopy.saveFailed));
    } finally {
      setPyannoteBusy(false);
    }
  };

  const handleInstallPyannote = async () => {
    setPyannoteBusy(true);
    setPyannoteMessage(null);
    try {
      const data = await postJson<{ success?: boolean; status?: PyannoteStatus }>(
        '/api/runtime/pyannote/install',
        {},
        { timeoutMs: 900_000, retries: 0 }
      );
      setPyannoteStatus(data.status || null);
      await loadLocalModels();
    } catch (error: any) {
      setPyannoteMessage(String(error?.message || pyannoteCopy.installFailed));
    } finally {
      setPyannoteBusy(false);
    }
  };

  const localizedPyannoteState =
    pyannoteStatus?.state === 'ready'
      ? pyannoteCopy.ready
      : pyannoteStatus?.state === 'partial'
        ? pyannoteCopy.partial
        : pyannoteCopy.missing;

  return (
    <div className="mx-auto max-w-6xl space-y-8 animate-in fade-in duration-500">
      <div className="rounded-3xl border border-white/5 bg-surface-container px-7 py-6 shadow-[0_24px_80px_rgba(6,10,22,0.24)]">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div className="space-y-2">
          <div className="flex items-center gap-4">
            <h2 className="text-3xl font-bold text-secondary tracking-tight">{t('settings.title')}</h2>
            {project && (
              <div className="flex items-center gap-2 rounded-full border border-primary-container/20 bg-primary-container/10 px-3 py-1.5 animate-in fade-in zoom-in duration-500">
                <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse shadow-[0_0_8px_rgba(var(--primary),0.8)]" />
                <span className="text-[11px] font-bold text-primary">{project.name}</span>
              </div>
            )}
          </div>
          <p className="text-outline">{t('settings.subtitle')}</p>
        </div>
        <button 
          onClick={handleGlobalSave}
          disabled={saveStatus !== 'idle' || settingsLoading || !settingsHydrated}
          className={`flex items-center gap-2 px-8 py-4 rounded-xl font-bold transition-all active:scale-95 shadow-lg shadow-primary-container/20 disabled:opacity-50 ${
            saveStatus === 'success'
              ? 'bg-tertiary text-white'
              : saveStatus === 'error'
                ? 'bg-error/15 text-error border border-error/20 hover:bg-error/20'
                : 'bg-primary-container text-white hover:bg-primary-container/90'
          }`}
        >
          {saveStatus === 'saving' || settingsLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : 
           saveStatus === 'success' ? <CheckCircle2 className="w-5 h-5" /> : 
           saveStatus === 'error' ? <AlertCircle className="w-5 h-5" /> :
           <Save className="w-5 h-5" />}
          {saveStatusLabel}
        </button>
        </div>
        <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
          <SummaryCard
            icon={<Plus className="h-4 w-4" />}
            label={t('settings.asrModels')}
            value={String(asrModels.length)}
            hint={t('settings.addAsr')}
          />
          <SummaryCard
            icon={<Globe className="h-4 w-4" />}
            label={t('settings.transModels')}
            value={String(translateModels.length)}
            hint={t('settings.addTrans')}
          />
          <SummaryCard
            icon={<Zap className="h-4 w-4" />}
            label={t('settings.localModelsTitle')}
            value={localRuntimeReady ? t('settings.localStatusReady') : t('settings.localStatusUnavailable')}
            hint={t('settings.localModelsSubtitle')}
          />
          <SummaryCard
            icon={<Globe className="h-4 w-4" />}
            label={t('settings.interfaceLang')}
            value={currentLanguageLabel}
            hint={t('settings.langNote')}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-8">
        <div className="grid grid-cols-1 gap-8 xl:grid-cols-2">
        {/* ASR Section */}
        <section className="rounded-3xl border border-white/5 bg-surface-container p-7 shadow-[0_18px_50px_rgba(7,10,24,0.18)]">
          <div className="mb-6 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary-container/10">
              <Plus className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-secondary">{t('settings.asrModels')}</h3>
              <p className="mt-1 text-xs text-outline">{t('settings.addAsr')}</p>
            </div>
          </div>
          
          <div className="space-y-4">
            {asrModels.map(model => (
              <ModelItem 
                key={model.id} 
                model={model} 
                modelType="asr"
                onSave={handleSaveAsr}
                onDelete={() => handleDeleteAsr(model.id)}
              />
            ))}
            
            {isAddingAsr ? (
              <ModelItem 
                model={{ id: '', name: '', url: '', key: '' }} 
                modelType="asr"
                isNew={true}
                onSave={handleAddAsr}
                onCancelNew={() => setIsAddingAsr(false)}
                onDelete={() => {}}
              />
            ) : (
              <button 
                onClick={() => setIsAddingAsr(true)}
                className="w-full py-5 border-2 border-dashed border-white/10 rounded-2xl text-outline hover:border-primary/30 hover:text-primary hover:bg-primary/5 transition-all flex items-center justify-center gap-2 group"
              >
                <Plus className="w-5 h-5 group-hover:scale-110 transition-transform" />
                <span className="text-sm font-bold uppercase tracking-widest">{t('settings.addAsr')}</span>
              </button>
            )}
          </div>
        </section>

        {/* Translation Section */}
        <section className="rounded-3xl border border-white/5 bg-surface-container p-7 shadow-[0_18px_50px_rgba(7,10,24,0.18)]">
          <div className="mb-6 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary-container/10">
              <Globe className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-secondary">{t('settings.transModels')}</h3>
              <p className="mt-1 text-xs text-outline">{t('settings.addTrans')}</p>
            </div>
          </div>
          
          <div className="space-y-4">
            {translateModels.map(model => (
              <ModelItem 
                key={model.id} 
                model={model} 
                modelType="translate"
                onSave={handleSaveTranslate}
                onDelete={() => handleDeleteTranslate(model.id)}
              />
            ))}

            {isAddingTranslate ? (
              <ModelItem 
                model={{ id: '', name: '', url: '', key: '' }} 
                modelType="translate"
                isNew={true}
                onSave={handleAddTranslate}
                onCancelNew={() => setIsAddingTranslate(false)}
                onDelete={() => {}}
              />
            ) : (
              <button 
                onClick={() => setIsAddingTranslate(true)}
                className="w-full py-5 border-2 border-dashed border-white/10 rounded-2xl text-outline hover:border-primary/30 hover:text-primary hover:bg-primary/5 transition-all flex items-center justify-center gap-2 group"
              >
                <Plus className="w-5 h-5 group-hover:scale-110 transition-transform" />
                <span className="text-sm font-bold uppercase tracking-widest">{t('settings.addTrans')}</span>
              </button>
            )}
          </div>
        </section>
        </div>

        {/* Local OpenVINO Models */}
        {localPanelActive ? (
        <section className="rounded-3xl border border-white/5 bg-surface-container p-7 shadow-[0_18px_50px_rgba(7,10,24,0.18)] space-y-6">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div>
              <h3 className="text-xl font-bold text-secondary">{t('settings.localModelsTitle')}</h3>
              <p className="text-xs text-outline mt-1">{t('settings.localModelsSubtitle')}</p>
            </div>
            <button
              type="button"
              onClick={() => void loadLocalModels()}
              disabled={isLocalSectionLoading}
              className="px-4 py-2 rounded-lg border border-white/10 text-xs font-bold text-outline hover:text-secondary hover:bg-white/5 transition-all disabled:opacity-60 disabled:cursor-not-allowed inline-flex items-center gap-2"
            >
              {isLocalSectionLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : null}
              {t('settings.localRefresh')}
            </button>
          </div>

          {isLocalSectionLoading && (
            <div className="inline-flex items-center gap-2 rounded-lg border border-white/5 bg-surface-container-lowest px-3 py-2 text-xs text-outline">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-primary" />
              <span>{localLoadingHint}</span>
            </div>
          )}

          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className="rounded-2xl border border-white/5 bg-surface-container-lowest p-5">
              <div className="text-[10px] font-bold text-outline uppercase tracking-widest mb-1">{t('settings.openvinoNode')}</div>
              <div className={`text-sm font-bold ${
                localStatusLoading ? 'text-outline' : openvinoStatus?.node?.available ? 'text-tertiary' : 'text-error'
              }`}>
                {localStatusLoading
                  ? t('settings.localStatusChecking')
                  : openvinoStatus?.node?.available
                    ? t('settings.localStatusReady')
                    : t('settings.localStatusUnavailable')}
              </div>
              {!localStatusLoading && !openvinoStatus?.node?.available && openvinoStatus?.node?.error && (
                <div className="text-[11px] text-error/80 mt-2 break-all">{openvinoStatus.node.error}</div>
              )}
              {!localStatusLoading && openvinoStatus?.node?.version && (
                <div className="text-[11px] text-outline mt-2">
                  {localCopy.version}: {openvinoStatus.node.version}
                </div>
              )}
            </div>
            <div className="rounded-2xl border border-white/5 bg-surface-container-lowest p-5">
              <div className="text-[10px] font-bold text-outline uppercase tracking-widest mb-1">{localCopy.openvinoGenai}</div>
              <div className={`text-sm font-bold ${
                localStatusLoading ? 'text-outline' : openvinoStatus?.genai?.available ? 'text-tertiary' : 'text-error'
              }`}>
                {localStatusLoading
                  ? t('settings.localStatusChecking')
                  : openvinoStatus?.genai?.available
                    ? t('settings.localStatusReady')
                    : t('settings.localStatusUnavailable')}
              </div>
              {!localStatusLoading && openvinoStatus?.genai?.version && (
                <div className="text-[11px] text-outline mt-2">
                  {localCopy.version}: {openvinoStatus.genai.version}
                </div>
              )}
              {!localStatusLoading && (
                <div className="text-[11px] text-outline mt-2 space-y-1">
                  <div>{localCopy.whisperSupport}: {openvinoStatus?.genai?.whisperPipelineAvailable ? t('settings.localStatusReady') : t('settings.localStatusUnavailable')}</div>
                  <div>{localCopy.llmSupport}: {openvinoStatus?.genai?.llmPipelineAvailable ? t('settings.localStatusReady') : t('settings.localStatusUnavailable')}</div>
                  <div>{localCopy.vlmSupport}: {openvinoStatus?.genai?.vlmPipelineAvailable ? t('settings.localStatusReady') : t('settings.localStatusUnavailable')}</div>
                </div>
              )}
              {!localStatusLoading && openvinoStatus?.genai?.error && (
                <div className="text-[11px] text-error/80 mt-2 break-all">{openvinoStatus.genai.error}</div>
              )}
            </div>
            <div className="rounded-2xl border border-white/5 bg-surface-container-lowest p-5">
              <div className="text-[10px] font-bold text-outline uppercase tracking-widest mb-1">{localCopy.asrRuntime}</div>
              <div className={`text-sm font-bold ${
                localStatusLoading
                  ? 'text-outline'
                  : openvinoStatus?.asr?.ready
                    ? 'text-tertiary'
                    : 'text-error'
              }`}>
                {localStatusLoading
                  ? t('settings.localStatusChecking')
                  : openvinoStatus?.asr?.ready
                    ? t('settings.localStatusReady')
                    : t('settings.localStatusUnavailable')}
              </div>
              {!localStatusLoading && (
                <div className="text-[11px] text-outline mt-2 space-y-1">
                  <div>{localCopy.whisperSupport}: {openvinoStatus?.asr?.whisperPipelineAvailable ? t('settings.localStatusReady') : t('settings.localStatusUnavailable')}</div>
                </div>
              )}
              {!localStatusLoading && openvinoStatus?.asr?.error && (
                <div className="text-[11px] text-error/80 mt-2 break-all">{openvinoStatus.asr.error}</div>
              )}
            </div>
          </div>

          <div className="rounded-2xl border border-white/5 bg-surface-container-lowest p-6 space-y-4">
            <div className="flex items-start justify-between gap-4 flex-wrap">
              <div>
                <div className="text-sm font-bold text-secondary">{pyannoteCopy.title}</div>
                <div className="mt-1 text-[11px] text-outline">{pyannoteCopy.subtitle}</div>
              </div>
              <button
                type="button"
                onClick={() => void loadLocalModels()}
                disabled={pyannoteBusy || isLocalSectionLoading}
                className="px-4 py-2 rounded-lg border border-white/10 text-xs font-bold text-outline hover:text-secondary hover:bg-white/5 transition-all disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {pyannoteCopy.refresh}
              </button>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <div className="rounded-2xl border border-white/5 bg-surface-container-high p-4">
                <div className="text-[10px] font-bold text-outline uppercase tracking-widest mb-1">{pyannoteCopy.title}</div>
                <div className={`text-sm font-bold ${pyannoteStatus?.ready ? 'text-tertiary' : pyannoteStatus?.state === 'partial' ? 'text-yellow-300' : 'text-error'}`}>
                  {localizedPyannoteState}
                </div>
                {pyannoteStatus?.lastError && !pyannoteStatus.ready && (
                  <div className="text-[11px] text-error/80 mt-2 break-all">{pyannoteStatus.lastError}</div>
                )}
              </div>
              <div className="rounded-2xl border border-white/5 bg-surface-container-high p-4">
                <div className="text-[10px] font-bold text-outline uppercase tracking-widest mb-1">{pyannoteCopy.tokenLabel}</div>
                <div className={`text-sm font-bold ${pyannoteStatus?.tokenConfigured ? 'text-tertiary' : 'text-error'}`}>
                  {pyannoteStatus?.tokenConfigured ? pyannoteCopy.tokenReady : pyannoteCopy.tokenMissing}
                </div>
                <div className="text-[11px] text-outline mt-2">{pyannoteCopy.tokenHint}</div>
              </div>
              <div className="rounded-2xl border border-white/5 bg-surface-container-high p-4">
                <div className="text-[10px] font-bold text-outline uppercase tracking-widest mb-1">{localCopy.openvinoGenai}</div>
                <div className={`text-sm font-bold ${pyannoteStatus?.ready ? 'text-tertiary' : 'text-outline'}`}>
                  {pyannoteStatus?.ready
                    ? pyannoteCopy.installed
                    : pyannoteStatus?.installing || pyannoteBusy
                      ? pyannoteCopy.installing
                      : pyannoteCopy.install}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_180px_180px] gap-3">
              <div className="space-y-2">
                <label className="block text-[10px] font-bold text-outline uppercase tracking-widest">{pyannoteCopy.tokenLabel}</label>
                <input
                  type="password"
                  value={pyannoteTokenInput}
                  onChange={(e) => setPyannoteTokenInput(e.target.value)}
                  disabled={pyannoteBusy}
                  placeholder={pyannoteCopy.tokenPlaceholder}
                  className="w-full bg-surface-container-high border border-white/10 rounded-xl py-3 px-4 text-white focus:ring-2 focus:ring-primary-container outline-none placeholder:text-outline/40"
                />
              </div>
              <div className="space-y-2">
                <label className="block text-[10px] font-bold text-outline uppercase tracking-widest opacity-0">
                  {pyannoteCopy.saveToken}
                </label>
                <button
                  type="button"
                  onClick={() => void handleSavePyannoteToken()}
                  disabled={pyannoteBusy || !pyannoteTokenInput.trim()}
                  className="w-full px-4 py-3 rounded-xl border border-white/10 text-sm font-bold text-secondary hover:bg-white/5 transition-all disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  {pyannoteCopy.saveToken}
                </button>
              </div>
              <div className="space-y-2">
                <label className="block text-[10px] font-bold text-outline uppercase tracking-widest opacity-0">
                  {pyannoteCopy.install}
                </label>
                <button
                  type="button"
                  onClick={() => void handleInstallPyannote()}
                  disabled={pyannoteBusy || pyannoteStatus?.ready || !pyannoteStatus?.tokenConfigured}
                  className="w-full px-4 py-3 rounded-xl border border-white/10 text-sm font-bold text-secondary hover:bg-white/5 transition-all disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {pyannoteBusy && !pyannoteStatus?.ready ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
                  {pyannoteStatus?.ready
                    ? pyannoteCopy.installed
                    : pyannoteBusy
                      ? pyannoteCopy.installing
                      : pyannoteCopy.install}
                </button>
              </div>
            </div>

            {pyannoteMessage && (
              <div className={`text-[11px] ${pyannoteMessage === pyannoteCopy.saved ? 'text-tertiary' : 'text-error/80'}`}>
                {pyannoteMessage}
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-white/5 bg-surface-container-lowest p-6 space-y-4">
            <div>
              <div className="text-sm font-bold text-secondary">{localCopy.installTitle}</div>
              <div className="mt-1 text-[11px] text-outline">{localCopy.installHint}</div>
              <div className="mt-1 text-[11px] text-outline/80">{localCopy.installHintDetail}</div>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-[180px_minmax(0,1fr)_160px] gap-3">
              <div className="space-y-2">
                <label className="block text-[10px] font-bold text-outline uppercase tracking-widest">{localCopy.installType}</label>
                <select
                  value={localInstallType}
                  onChange={(e) => setLocalInstallType(e.target.value as LocalModelType)}
                  disabled={localInstallBusy || isLocalSectionLoading}
                  className="w-full bg-surface-container-high border border-white/10 rounded-xl py-3 px-4 text-white appearance-none focus:ring-2 focus:ring-primary-container outline-none"
                >
                  <option value="asr">{localCopy.installTypeAsr}</option>
                  <option value="translate">{localCopy.installTypeTranslate}</option>
                </select>
              </div>
              <div className="space-y-2">
                <label className="block text-[10px] font-bold text-outline uppercase tracking-widest">{localCopy.installRepo}</label>
                <input
                  type="text"
                  value={localRepoId}
                  onChange={(e) => setLocalRepoId(e.target.value)}
                  disabled={localInstallBusy || isLocalSectionLoading || !localInstallRuntimeReady}
                  placeholder={localCopy.installPlaceholder}
                  className="w-full bg-surface-container-high border border-white/10 rounded-xl py-3 px-4 text-white focus:ring-2 focus:ring-primary-container outline-none placeholder:text-outline/40"
                />
              </div>
              <div className="space-y-2">
                <label className="block text-[10px] font-bold text-outline uppercase tracking-widest opacity-0">
                  {t('settings.localInstall')}
                </label>
                <button
                  type="button"
                  onClick={() => void handleInstallLocalModel()}
                  disabled={localInstallBusy || isLocalSectionLoading || !localInstallRuntimeReady}
                  className="w-full px-4 py-3 rounded-xl border border-white/10 text-sm font-bold text-secondary hover:bg-white/5 transition-all disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {localInstallBusy ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
                  {localInstallBusy ? t('settings.localInstalling') : t('settings.localInstall')}
                </button>
              </div>
            </div>
            {!isLocalSectionLoading && !localInstallRuntimeReady && (
              <div className="text-[11px] text-error/80">{localInstallRuntimeBlockedHint}</div>
            )}
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div className="rounded-2xl border border-white/5 bg-surface-container-lowest p-6 space-y-4">
              <div>
                <label className="block text-[10px] font-bold text-outline uppercase tracking-widest">{t('settings.asrLocalModel')}</label>
              </div>
              <div className="space-y-3">
                {localAsrModels.length === 0 && (
                  <div className="text-[11px] text-outline">{localCopy.noInstalledModels}</div>
                )}
                {localAsrModels.map((model) => (
                  <div key={model.id} className="rounded-xl border border-white/5 bg-surface-container-high p-4 flex items-start justify-between gap-3">
                    <button
                      type="button"
                      onClick={() => void handleSelectLocalModel('asr', model.id)}
                      disabled={localCatalogLoading}
                      className="flex-1 rounded-xl -m-2 p-2 text-left transition-colors hover:bg-white/5 hover:text-white disabled:opacity-60 disabled:hover:bg-transparent"
                    >
                      <div className="flex items-center gap-2 flex-wrap">
                        <div className="text-sm font-bold text-secondary">{model.name}</div>
                        {model.selected && (
                          <span className="px-2 py-0.5 rounded-full bg-tertiary/10 text-tertiary text-[10px] font-bold uppercase tracking-widest">
                            {localCopy.selected}
                          </span>
                        )}
                      </div>
                      <div className="text-[11px] text-outline mt-2 break-all">
                        {localCopy.repoId}: {model.repoId}
                      </div>
                      <div className="mt-2 grid grid-cols-1 gap-1 text-[11px] text-outline sm:grid-cols-2">
                        <div>{localCopy.sourceFormat}: {formatLocalModelSourceFormat(language, model.sourceFormat)}</div>
                        <div>{localCopy.conversionMethod}: {formatLocalModelConversionMethod(language, model.conversionMethod)}</div>
                        <div>{localCopy.runtimeLayout}: {formatLocalModelRuntimeLayout(language, model.runtimeLayout)}</div>
                        <div>{localCopy.runtimeEngine}: {formatLocalModelRuntimeEngine(model.runtime)}</div>
                      </div>
                      {!model.selected && (
                        <div className="text-[11px] text-primary mt-2">{localCopy.setDefault}</div>
                      )}
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleRemoveLocalModel(model.id)}
                      disabled={localRemovingModelId === model.id}
                      className="px-4 py-2 rounded-lg border border-white/10 text-sm font-bold text-secondary hover:bg-white/5 transition-all disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                      {localRemovingModelId === model.id ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
                      {localRemovingModelId === model.id ? t('settings.localRemoving') : t('settings.localRemove')}
                    </button>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-2xl border border-white/5 bg-surface-container-lowest p-6 space-y-4">
              <div>
                <label className="block text-[10px] font-bold text-outline uppercase tracking-widest">{t('settings.translateLocalModel')}</label>
              </div>
              <div className="space-y-3">
                {localTranslateModels.length === 0 && (
                  <div className="text-[11px] text-outline">{localCopy.noInstalledModels}</div>
                )}
                {localTranslateModels.map((model) => (
                  <div key={model.id} className="rounded-xl border border-white/5 bg-surface-container-high p-4 flex items-start justify-between gap-3">
                    <button
                      type="button"
                      onClick={() => void handleSelectLocalModel('translate', model.id)}
                      disabled={localCatalogLoading}
                      className="flex-1 rounded-xl -m-2 p-2 text-left transition-colors hover:bg-white/5 hover:text-white disabled:opacity-60 disabled:hover:bg-transparent"
                    >
                      <div className="flex items-center gap-2 flex-wrap">
                        <div className="text-sm font-bold text-secondary">{model.name}</div>
                        {model.selected && (
                          <span className="px-2 py-0.5 rounded-full bg-tertiary/10 text-tertiary text-[10px] font-bold uppercase tracking-widest">
                            {localCopy.selected}
                          </span>
                        )}
                      </div>
                      <div className="text-[11px] text-outline mt-2 break-all">
                        {localCopy.repoId}: {model.repoId}
                      </div>
                      <div className="mt-2 grid grid-cols-1 gap-1 text-[11px] text-outline sm:grid-cols-2">
                        <div>{localCopy.sourceFormat}: {formatLocalModelSourceFormat(language, model.sourceFormat)}</div>
                        <div>{localCopy.conversionMethod}: {formatLocalModelConversionMethod(language, model.conversionMethod)}</div>
                        <div>{localCopy.runtimeLayout}: {formatLocalModelRuntimeLayout(language, model.runtimeLayout)}</div>
                        <div>{localCopy.runtimeEngine}: {formatLocalModelRuntimeEngine(model.runtime)}</div>
                      </div>
                      {!model.selected && (
                        <div className="text-[11px] text-primary mt-2">{localCopy.setDefault}</div>
                      )}
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleRemoveLocalModel(model.id)}
                      disabled={localRemovingModelId === model.id}
                      className="px-4 py-2 rounded-lg border border-white/10 text-sm font-bold text-secondary hover:bg-white/5 transition-all disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                      {localRemovingModelId === model.id ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
                      {localRemovingModelId === model.id ? t('settings.localRemoving') : t('settings.localRemove')}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {selectedLocalInstallError && (
            <div className="text-xs text-error bg-error/10 border border-error/20 rounded-lg p-3">
              {selectedLocalInstallError}
            </div>
          )}
        </section>
        ) : (
          <section className="rounded-3xl border border-white/5 bg-surface-container p-7 shadow-[0_18px_50px_rgba(7,10,24,0.18)] space-y-4">
            <div className="h-5 w-56 rounded bg-white/8 animate-pulse" />
            <div className="h-4 w-80 rounded bg-white/6 animate-pulse" />
            <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
              <div className="h-24 rounded-2xl bg-surface-container-lowest animate-pulse" />
              <div className="h-24 rounded-2xl bg-surface-container-lowest animate-pulse" />
              <div className="h-24 rounded-2xl bg-surface-container-lowest animate-pulse" />
            </div>
          </section>
        )}

        {/* Interface Settings */}
        <div className="grid grid-cols-1 gap-8 xl:grid-cols-[minmax(0,1.2fr)_minmax(280px,0.8fr)]">
          <section className="rounded-3xl border border-white/5 bg-surface-container p-7 shadow-[0_18px_50px_rgba(7,10,24,0.18)]">
            <div className="mb-6 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary-container/10">
                <Globe className="w-5 h-5 text-primary" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-secondary">{t('settings.interfaceLang')}</h3>
                <p className="mt-1 text-xs text-outline">{t('settings.langNote')}</p>
              </div>
            </div>
            <div className="relative max-w-sm">
              <select 
                value={language}
                onChange={(e) => {
                  const nextLanguage = e.target.value as Language;
                  setLanguage(nextLanguage);
                  void persistSettings({
                    interfaceLanguage: nextLanguage,
                  });
                }}
                disabled={!settingsHydrated || settingsLoading}
                className="w-full bg-surface-container-high border border-white/10 rounded-xl py-4 px-5 text-white appearance-none focus:ring-2 focus:ring-primary-container outline-none transition-all cursor-pointer [&>option]:bg-surface-container-high [&>option]:text-white"
              >
                <option value="zh-tw" className="bg-surface-container-high text-white">{t('lang.zh-tw')}</option>
                <option value="zh-cn" className="bg-surface-container-high text-white">{t('lang.zh-cn')}</option>
                <option value="en" className="bg-surface-container-high text-white">{t('lang.en')}</option>
                <option value="de" className="bg-surface-container-high text-white">{t('lang.de')}</option>
                <option value="jp" className="bg-surface-container-high text-white">{t('lang.jp')}</option>
              </select>
              <div className="absolute right-5 top-1/2 -translate-y-1/2 pointer-events-none text-outline">
                <ChevronDown className="w-5 h-5" />
              </div>
            </div>
          </section>

          <div className="flex flex-col justify-center rounded-3xl border border-primary-container/20 bg-primary-container/5 p-7 shadow-[0_18px_50px_rgba(7,10,24,0.18)]">
            <div className="mb-6 flex h-12 w-12 items-center justify-center rounded-2xl bg-primary-container">
              <Info className="w-6 h-6 text-white" />
            </div>
            <h4 className="text-secondary font-bold mb-3">{t('settings.aboutModels')}</h4>
            <p className="text-xs text-outline leading-relaxed">
              {t('settings.aboutModelsDesc')}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function serializeModelOptions(options: ApiModelRequestOptions | undefined): string {
  if (!options || !isPlainObject(options) || Object.keys(options).length === 0) return '';
  try {
    return JSON.stringify(options, null, 2);
  } catch {
    return '';
  }
}

function parseModelOptions(raw: string): {
  value?: ApiModelRequestOptions;
  errorKey?: 'settings.advancedOptionsErrorObject' | 'settings.advancedOptionsErrorInvalidJson';
} {
  const trimmed = String(raw || '').trim();
  if (!trimmed) return { value: undefined };

  try {
    const parsed = JSON.parse(trimmed);
    if (!isPlainObject(parsed)) {
      return { errorKey: 'settings.advancedOptionsErrorObject' };
    }
    return { value: parsed as ApiModelRequestOptions };
  } catch {
    return { errorKey: 'settings.advancedOptionsErrorInvalidJson' };
  }
}

const ModelItem: React.FC<{ 
  model: ApiConfig, 
  modelType: 'asr' | 'translate',
  onSave: (model: ApiConfig) => void, 
  onDelete: () => void,
  isNew?: boolean,
  onCancelNew?: () => void
}> = ({ 
  model, 
  modelType,
  onSave, 
  onDelete, 
  isNew = false,
  onCancelNew
}) => {
  const { t } = useLanguage();
  const [isEditing, setIsEditing] = React.useState(isNew);
  const [editedModel, setEditedModel] = React.useState(model);
  const [modelOptionsText, setModelOptionsText] = React.useState(() => serializeModelOptions(model.options));
  const [testStatus, setTestStatus] = React.useState<'idle' | 'testing' | 'success' | 'failed'>('idle');
  const [testError, setTestError] = React.useState<string | null>(null);
  const [errors, setErrors] = React.useState<{name?: string, url?: string, key?: string, options?: string}>({});

  React.useEffect(() => {
    if (isEditing) return;
    setEditedModel(model);
    setModelOptionsText(serializeModelOptions(model.options));
  }, [isEditing, model]);

  const validate = (data: ApiConfig, optionsError?: string | null) => {
    const newErrors: {name?: string, url?: string, key?: string, options?: string} = {};
    if (!data.name.trim()) newErrors.name = t('settings.errorNameRequired');
    if (!isValidUrl(data.url)) newErrors.url = t('settings.errorInvalidUrl');
    if (data.key && !isValidApiKey(data.key) && !isMaskedApiKey(data.key)) {
      newErrors.key = t('settings.errorInvalidKey');
    }
    if (modelType === 'translate' && optionsError) {
      newErrors.options = optionsError;
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleTestConnection = async () => {
    const parsedOptions = isEditing && modelType === 'translate' ? parseModelOptions(modelOptionsText) : {};
    if (parsedOptions.errorKey) {
      const localized = t(parsedOptions.errorKey);
      setErrors((prev) => ({ ...prev, options: localized }));
      setTestStatus('failed');
      setTestError(localized);
      setTimeout(() => {
        setTestStatus('idle');
      }, 5000);
      return;
    }

    const target = isEditing
      ? {
          ...editedModel,
          options: modelType === 'translate' ? parsedOptions.value : editedModel.options,
        }
      : model;
    if (!target.url) return;
    const configuredTimeoutMs = Number(target.options?.timeoutMs);
    const connectionTimeoutMs =
      Number.isFinite(configuredTimeoutMs) && configuredTimeoutMs > 0
        ? Math.round(configuredTimeoutMs)
        : modelType === 'translate'
          ? 120_000
          : 20_000;

    const parseHttpErrorBody = (bodyText: string) => {
      const raw = String(bodyText || '').trim();
      if (!raw) return '';
      try {
        const parsed = JSON.parse(raw);
        const nested = parsed?.error;
        const nestedMessage =
          typeof nested === 'string'
            ? nested
            : nested?.message || parsed?.message || parsed?.detail || parsed?.error_description;
        if (typeof nestedMessage === 'string' && nestedMessage.trim()) {
          return nestedMessage.trim();
        }
      } catch {
        // Non-JSON body; use raw text.
      }
      return raw;
    };
    
    setTestStatus('testing');
    setTestError(null);
    try {
      const data = await postJson<{ success?: boolean; message?: string; error?: string }>(
        '/api/settings/test-connection',
        {
          url: target.url,
          key: target.key,
          type: modelType,
          modelId: target.id,
          model: target.model || '',
          name: target.name || '',
          options: target.options,
        },
        { timeoutMs: connectionTimeoutMs, retries: 0 }
      );
      
      if (data.success) {
        setTestStatus('success');
      } else {
        setTestStatus('failed');
        setTestError(data.error || t('settings.testFailed'));
      }
    } catch (error: any) {
      setTestStatus('failed');
      const rawMessage = String(error?.message || '').trim();
      const isAbortLike = error?.name === 'AbortError' || /abort|aborted|timeout/i.test(rawMessage);
      if (isAbortLike) {
        const timeoutMessage = t('settings.connectionTimeout').replace(
          '{seconds}',
          String(Math.round(connectionTimeoutMs / 1000))
        );
        setTestError(timeoutMessage);
      } else if (error instanceof HttpRequestError) {
        setTestError(parseHttpErrorBody(error.bodyText) || rawMessage || t('settings.testFailed'));
      } else {
        setTestError(rawMessage || t('settings.testFailed'));
      }
    } finally {
      // Always reset to idle after 5 seconds to let user try again
      setTimeout(() => {
        setTestStatus('idle');
      }, 5000);
    }
  };

  const handleSave = () => {
    const parsedOptions = modelType === 'translate' ? parseModelOptions(modelOptionsText) : {};
    const sanitized = {
      ...editedModel,
      name: sanitizeInput(editedModel.name),
      url: editedModel.url.trim(),
      key: editedModel.key.trim(),
      options: modelType === 'translate' ? parsedOptions.value : editedModel.options,
    };
    const optionsError = parsedOptions.errorKey ? t(parsedOptions.errorKey) : null;
    if (!validate(sanitized, optionsError)) return;
    onSave(sanitized);
    setIsEditing(false);
  };

  const handleCancel = () => {
    if (isNew && onCancelNew) {
      onCancelNew();
    } else {
      setEditedModel(model);
      setModelOptionsText(serializeModelOptions(model.options));
      setIsEditing(false);
    }
  };

  if (!isEditing) {
    return (
      <div className="rounded-2xl border border-white/5 bg-surface-container-lowest p-5 transition-all hover:border-white/10 hover:bg-surface-container-lowest/90">
        <div className="space-y-4">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div className="min-w-0">
              <div className="text-[10px] font-bold uppercase tracking-widest text-outline">{t('settings.modelName')}</div>
              <div className="mt-1 truncate text-base font-bold text-secondary">{model.name}</div>
            </div>
            <div className="inline-flex w-fit items-center rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[11px] font-bold text-outline">
              {model.model || (modelType === 'asr' ? 'whisper-1' : 'gpt-4o-mini')}
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="rounded-xl border border-white/5 bg-surface-container-high px-4 py-3 sm:col-span-2">
              <div className="text-[10px] font-bold uppercase tracking-widest text-outline">{t('settings.apiUrl')}</div>
              <div className="mt-1 truncate text-sm text-secondary">{model.url}</div>
            </div>
            <div className="rounded-xl border border-white/5 bg-surface-container-high px-4 py-3">
              <div className="text-[10px] font-bold uppercase tracking-widest text-outline">{t('settings.apiKey')}</div>
              <div className="mt-1 flex items-center gap-2 text-sm text-secondary">
                <Key className="h-3.5 w-3.5 text-outline" />
                {model.key ? maskApiKey(model.key) : t('settings.notSet')}
              </div>
            </div>
            <div className="rounded-xl border border-white/5 bg-surface-container-high px-4 py-3">
              <div className="text-[10px] font-bold uppercase tracking-widest text-outline">{t('settings.modelId')}</div>
              <div className="mt-1 truncate text-sm text-secondary">
                {model.model || (modelType === 'asr' ? 'whisper-1' : 'gpt-4o-mini')}
              </div>
            </div>
            {modelType === 'translate' && model.options && (
              <div className="rounded-xl border border-white/5 bg-surface-container-high px-4 py-3 sm:col-span-2">
                <div className="text-[10px] font-bold uppercase tracking-widest text-outline">
                  {t('settings.advancedOptionsTitle')}
                </div>
                <div className="mt-1 truncate text-xs text-secondary">
                  {Object.keys(model.options).length > 0
                    ? t('settings.advancedOptionsConfigured')
                    : t('settings.notSet')}
                </div>
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
          <button 
            onClick={handleTestConnection}
            disabled={testStatus === 'testing'}
            className={`flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-bold transition-all ${
              testStatus === 'success' ? 'border border-tertiary/20 bg-tertiary/10 text-tertiary' : 
              testStatus === 'failed' ? 'border border-error/20 bg-error/10 text-error' : 
              'border border-white/10 text-outline hover:bg-white/8 hover:text-white'
            }`}
          >
            {testStatus === 'testing' ? <Loader2 className="h-4 w-4 animate-spin" /> : 
             testStatus === 'success' ? <CheckCircle2 className="h-4 w-4" /> :
             testStatus === 'failed' ? <AlertCircle className="h-4 w-4" /> :
             <Zap className="h-4 w-4" />}
            <span>
              {testStatus === 'testing' ? t('settings.testing') : 
               testStatus === 'success' ? t('settings.testSuccess') :
               testStatus === 'failed' ? t('settings.testFailed') :
               t('settings.testConnection')}
            </span>
          </button>
          <button 
            onClick={() => setIsEditing(true)} 
            className="flex items-center justify-center gap-2 rounded-xl border border-white/10 px-4 py-2.5 text-sm font-bold text-outline transition-colors hover:bg-primary/10 hover:text-primary"
          >
            <Edit2 className="h-4 w-4" />
            <span>{t('settings.edit')}</span>
          </button>
          <button 
            onClick={onDelete} 
            className="flex items-center justify-center gap-2 rounded-xl border border-white/10 px-4 py-2.5 text-sm font-bold text-outline transition-all hover:border-red-400/30 hover:bg-red-500/10 hover:text-red-300"
          >
            <Trash2 className="h-4 w-4" />
            <span>{t('settings.delete')}</span>
          </button>
        </div>
        </div>
        {testError && (
          <div className="mt-4 flex items-start gap-3 rounded-xl border border-error/20 bg-error/10 p-4 animate-in slide-in-from-top-2 duration-200">
            <AlertCircle className="w-4 h-4 text-error mt-0.5" />
            <div className="flex-1">
              <div className="text-xs font-bold text-error mb-1">{t('settings.testFailed')}</div>
              <div className="text-[11px] text-error/80 leading-relaxed">{testError}</div>
            </div>
            <button onClick={() => setTestError(null)} className="text-error/50 hover:text-error">
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="animate-in fade-in zoom-in-95 duration-200 rounded-2xl border border-primary/30 bg-surface-container-high p-6 shadow-lg shadow-primary/5 transition-all">
      <div className="mb-6 flex items-center gap-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
          {isNew ? <Plus className="w-4 h-4 text-primary" /> : <Edit2 className="w-4 h-4 text-primary" />}
        </div>
        <h4 className="text-sm font-bold text-secondary">{isNew ? t('settings.addModel') : t('settings.editModel')}</h4>
      </div>
      
      <div className="mb-8 grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-4">
        <div className="space-y-2">
          <label className="text-[10px] font-bold text-primary uppercase tracking-widest">{t('settings.modelName')} <span className="text-error">*</span></label>
          <input 
            type="text" 
            value={editedModel.name} 
            onChange={e => {
              setEditedModel({...editedModel, name: e.target.value});
              if (errors.name) setErrors({...errors, name: undefined});
            }}
            placeholder={t('settings.namePlaceholder')}
            className={`w-full bg-surface-container-lowest border rounded-xl text-sm py-3.5 px-4 text-secondary focus:ring-2 focus:ring-primary-container outline-none transition-all placeholder:text-outline/30 ${
              errors.name ? 'border-error/50' : 'border-white/10'
            }`}
          />
          {errors.name && <p className="text-[10px] text-error font-bold mt-1">{errors.name}</p>}
        </div>
        <div className="space-y-2">
          <label className="text-[10px] font-bold text-primary uppercase tracking-widest">{t('settings.apiUrl')} <span className="text-error">*</span></label>
          <input 
            type="text" 
            value={editedModel.url} 
            onChange={e => {
              setEditedModel({...editedModel, url: e.target.value});
              if (errors.url) setErrors({...errors, url: undefined});
            }}
            placeholder={modelType === 'asr' ? 'https://api.openai.com/v1/audio/transcriptions' : 'https://api.openai.com/v1/chat/completions'}
            className={`w-full bg-surface-container-lowest border rounded-xl text-sm py-3.5 px-4 text-secondary focus:ring-2 focus:ring-primary-container outline-none transition-all placeholder:text-outline/30 ${
              errors.url ? 'border-error/50' : 'border-white/10'
            }`}
          />
          {errors.url && <p className="text-[10px] text-error font-bold mt-1">{errors.url}</p>}
        </div>
        <div className="space-y-2">
          <label className="text-[10px] font-bold text-primary uppercase tracking-widest">{t('settings.apiKey')}</label>
          <input 
            type="password" 
            value={editedModel.key} 
            onChange={e => {
              setEditedModel({...editedModel, key: e.target.value});
              if (errors.key) setErrors({...errors, key: undefined});
            }}
            placeholder="sk-..."
            className={`w-full bg-surface-container-lowest border rounded-xl text-sm py-3.5 px-4 text-secondary focus:ring-2 focus:ring-primary-container outline-none transition-all placeholder:text-outline/30 ${
              errors.key ? 'border-error/50' : 'border-white/10'
            }`}
          />
          {errors.key && <p className="text-[10px] text-error font-bold mt-1">{errors.key}</p>}
        </div>
        <div className="space-y-2">
          <label className="text-[10px] font-bold text-primary uppercase tracking-widest">{t('settings.modelId')}</label>
          <input 
            type="text" 
            value={editedModel.model || ''} 
            onChange={e => setEditedModel({...editedModel, model: e.target.value})}
            placeholder={modelType === 'asr' ? 'whisper-1' : 'gpt-4o-mini'}
            className="w-full bg-surface-container-lowest border border-white/10 rounded-xl text-sm py-3.5 px-4 text-secondary focus:ring-2 focus:ring-primary-container outline-none transition-all placeholder:text-outline/30"
          />
          <p className="text-[10px] text-outline/50 mt-1">{t('settings.modelIdHint')}</p>
        </div>
        {modelType === 'translate' && (
          <div className="space-y-2 md:col-span-2 xl:col-span-4">
            <label className="text-[10px] font-bold text-primary uppercase tracking-widest">
              {t('settings.advancedOptionsJsonTitle')}
            </label>
            <textarea
              value={modelOptionsText}
              onChange={(e) => {
                setModelOptionsText(e.target.value);
                if (errors.options) setErrors({ ...errors, options: undefined });
              }}
              placeholder={`{\n  \"sampling\": {\n    \"temperature\": 1,\n    \"topP\": 0.95,\n    \"maxOutputTokens\": 16384\n  },\n  \"body\": {\n    \"stream\": true,\n    \"chat_template_kwargs\": {\n      \"enable_thinking\": true,\n      \"clear_thinking\": false\n    }\n  },\n  \"headers\": {\n    \"Accept\": \"text/event-stream\"\n  },\n  \"timeoutMs\": 180000\n}`}
              rows={12}
              className={`w-full bg-surface-container-lowest border rounded-xl text-sm py-3.5 px-4 text-secondary font-mono focus:ring-2 focus:ring-primary-container outline-none transition-all placeholder:text-outline/30 ${
                errors.options ? 'border-error/50' : 'border-white/10'
              }`}
            />
            {errors.options ? (
              <p className="text-[10px] text-error font-bold mt-1">{errors.options}</p>
            ) : (
              <p className="text-[10px] text-outline/50 mt-1">
                {t('settings.advancedOptionsHint')}
              </p>
            )}
          </div>
        )}
      </div>
      
      <div className="flex flex-col gap-3 border-t border-white/5 pt-6 sm:flex-row sm:justify-end">
        <button 
          onClick={handleTestConnection}
          disabled={testStatus !== 'idle' || !editedModel.url}
          className={`flex items-center justify-center gap-2 rounded-xl border px-6 py-2.5 text-sm font-bold transition-all ${
            testStatus === 'success' ? 'text-tertiary border-tertiary/30 bg-tertiary/5' : 
            testStatus === 'failed' ? 'text-error border-error/30 bg-error/5' : 
            'text-outline border-white/10 hover:text-white hover:bg-white/8'
          }`}
        >
          {testStatus === 'testing' ? <Loader2 className="w-4 h-4 animate-spin" /> : 
           testStatus === 'success' ? <CheckCircle2 className="w-4 h-4" /> :
           testStatus === 'failed' ? <AlertCircle className="w-4 h-4" /> :
           <Zap className="w-4 h-4" />}
          {testStatus === 'testing' ? t('settings.testing') : 
           testStatus === 'success' ? t('settings.testSuccess') :
           testStatus === 'failed' ? t('settings.testFailed') :
           t('settings.testConnection')}
        </button>
        <button 
          onClick={handleCancel} 
          className="px-6 py-2.5 text-sm font-bold text-outline transition-colors hover:text-white"
        >
          {t('modal.cancel')}
        </button>
        <button 
          onClick={handleSave} 
          disabled={!editedModel.name || !editedModel.url}
          className="flex items-center justify-center gap-2 rounded-xl bg-primary px-8 py-2.5 text-sm font-bold text-white shadow-lg shadow-primary/20 transition-all hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50 active:scale-95"
        >
          <Check className="w-4 h-4" />
          {isNew ? t('settings.confirmAdd') : t('settings.saveChanges')}
        </button>
      </div>
    </div>
  );
}


