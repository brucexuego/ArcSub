import type { ForcedAlignmentSelectionContext } from './shared/types.js';

type AlignmentConversionMethod =
  | 'openvino-convert-model'
  | 'openvino-ctc-asr-export';

type AlignmentSourceFormat = 'onnx' | 'pytorch';

export type AlignmentCtcModelProfile = {
  id: string;
  kind: 'ctc';
  modelId: string;
  sourceFormat?: AlignmentSourceFormat;
  conversionMethod?: AlignmentConversionMethod;
  runtimeLayout?: 'asr-ctc';
  envOverrides?: Record<string, string>;
};

export type AlignmentNativeModelProfile = {
  id: string;
  kind: 'native';
  modelId: string;
  nativeBackend: string;
};

export type AlignmentModelProfile = AlignmentCtcModelProfile | AlignmentNativeModelProfile;

type NativeAlignmentOverride = {
  profileId: string;
  matches(context: ForcedAlignmentSelectionContext): boolean;
};

const alignmentProfiles: AlignmentModelProfile[] = [
  {
    id: 'mms-300m-forced-aligner-v1',
    kind: 'ctc',
    modelId: 'onnx-community/mms-300m-1130-forced-aligner-ONNX',
    sourceFormat: 'onnx',
    conversionMethod: 'openvino-convert-model',
    runtimeLayout: 'asr-ctc',
    envOverrides: {
      OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION: '1',
    },
  },
  {
    id: 'ja-ntqai-surface-v1',
    kind: 'ctc',
    modelId: 'NTQAI/wav2vec2-large-japanese',
    sourceFormat: 'pytorch',
    conversionMethod: 'openvino-ctc-asr-export',
    runtimeLayout: 'asr-ctc',
    envOverrides: {
      OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION: '1',
    },
  },
  {
    id: 'ko-kresnik-xlsr-korean-v1',
    kind: 'ctc',
    modelId: 'kresnik/wav2vec2-large-xlsr-korean',
    sourceFormat: 'pytorch',
    conversionMethod: 'openvino-ctc-asr-export',
    runtimeLayout: 'asr-ctc',
    envOverrides: {
      OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION: '1',
    },
  },
  {
    id: 'en-jonatas-xlsr53-english-v1',
    kind: 'ctc',
    modelId: 'jonatasgrosman/wav2vec2-large-xlsr-53-english',
    sourceFormat: 'pytorch',
    conversionMethod: 'openvino-ctc-asr-export',
    runtimeLayout: 'asr-ctc',
    envOverrides: {
      OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION: '1',
    },
  },
  {
    id: 'ru-jonatas-xlsr53-russian-v1',
    kind: 'ctc',
    modelId: 'jonatasgrosman/wav2vec2-large-xlsr-53-russian',
    sourceFormat: 'pytorch',
    conversionMethod: 'openvino-ctc-asr-export',
    runtimeLayout: 'asr-ctc',
    envOverrides: {
      OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION: '1',
    },
  },
  {
    id: 'pt-jonatas-xlsr53-portuguese-v1',
    kind: 'ctc',
    modelId: 'jonatasgrosman/wav2vec2-large-xlsr-53-portuguese',
    sourceFormat: 'pytorch',
    conversionMethod: 'openvino-ctc-asr-export',
    runtimeLayout: 'asr-ctc',
    envOverrides: {
      OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION: '1',
    },
  },
  {
    id: 'zh-cn-jonatas-xlsr53-chinese-v1',
    kind: 'ctc',
    modelId: 'jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn',
    sourceFormat: 'pytorch',
    conversionMethod: 'openvino-ctc-asr-export',
    runtimeLayout: 'asr-ctc',
    envOverrides: {
      OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION: '1',
    },
  },
  {
    id: 'pl-jonatas-xlsr53-polish-v1',
    kind: 'ctc',
    modelId: 'jonatasgrosman/wav2vec2-large-xlsr-53-polish',
    sourceFormat: 'pytorch',
    conversionMethod: 'openvino-ctc-asr-export',
    runtimeLayout: 'asr-ctc',
    envOverrides: {
      OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION: '1',
    },
  },
  {
    id: 'ar-jonatas-xlsr53-arabic-v1',
    kind: 'ctc',
    modelId: 'jonatasgrosman/wav2vec2-large-xlsr-53-arabic',
    sourceFormat: 'pytorch',
    conversionMethod: 'openvino-ctc-asr-export',
    runtimeLayout: 'asr-ctc',
    envOverrides: {
      OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION: '1',
    },
  },
  {
    id: 'nl-jonatas-xlsr53-dutch-v1',
    kind: 'ctc',
    modelId: 'jonatasgrosman/wav2vec2-large-xlsr-53-dutch',
    sourceFormat: 'pytorch',
    conversionMethod: 'openvino-ctc-asr-export',
    runtimeLayout: 'asr-ctc',
    envOverrides: {
      OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION: '1',
    },
  },
  {
    id: 'el-jonatas-xlsr53-greek-v1',
    kind: 'ctc',
    modelId: 'jonatasgrosman/wav2vec2-large-xlsr-53-greek',
    sourceFormat: 'pytorch',
    conversionMethod: 'openvino-ctc-asr-export',
    runtimeLayout: 'asr-ctc',
    envOverrides: {
      OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION: '1',
    },
  },
  {
    id: 'fa-jonatas-xlsr53-persian-v1',
    kind: 'ctc',
    modelId: 'jonatasgrosman/wav2vec2-large-xlsr-53-persian',
    sourceFormat: 'pytorch',
    conversionMethod: 'openvino-ctc-asr-export',
    runtimeLayout: 'asr-ctc',
    envOverrides: {
      OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION: '1',
    },
  },
  {
    id: 'hu-jonatas-xlsr53-hungarian-v1',
    kind: 'ctc',
    modelId: 'jonatasgrosman/wav2vec2-large-xlsr-53-hungarian',
    sourceFormat: 'pytorch',
    conversionMethod: 'openvino-ctc-asr-export',
    runtimeLayout: 'asr-ctc',
    envOverrides: {
      OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION: '1',
    },
  },
  {
    id: 'fi-jonatas-xlsr53-finnish-v1',
    kind: 'ctc',
    modelId: 'jonatasgrosman/wav2vec2-large-xlsr-53-finnish',
    sourceFormat: 'pytorch',
    conversionMethod: 'openvino-ctc-asr-export',
    runtimeLayout: 'asr-ctc',
    envOverrides: {
      OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION: '1',
    },
  },
  {
    id: 'es-jonatas-xlsr53-spanish-v1',
    kind: 'ctc',
    modelId: 'jonatasgrosman/wav2vec2-large-xlsr-53-spanish',
    sourceFormat: 'pytorch',
    conversionMethod: 'openvino-ctc-asr-export',
    runtimeLayout: 'asr-ctc',
    envOverrides: {
      OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION: '1',
    },
  },
  {
    id: 'fr-jonatas-xlsr53-french-v1',
    kind: 'ctc',
    modelId: 'jonatasgrosman/wav2vec2-large-xlsr-53-french',
    sourceFormat: 'pytorch',
    conversionMethod: 'openvino-ctc-asr-export',
    runtimeLayout: 'asr-ctc',
    envOverrides: {
      OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION: '1',
    },
  },
  {
    id: 'de-jonatas-xlsr53-german-v1',
    kind: 'ctc',
    modelId: 'jonatasgrosman/wav2vec2-large-xlsr-53-german',
    sourceFormat: 'pytorch',
    conversionMethod: 'openvino-ctc-asr-export',
    runtimeLayout: 'asr-ctc',
    envOverrides: {
      OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION: '1',
    },
  },
  {
    id: 'it-jonatas-xlsr53-italian-v1',
    kind: 'ctc',
    modelId: 'jonatasgrosman/wav2vec2-large-xlsr-53-italian',
    sourceFormat: 'pytorch',
    conversionMethod: 'openvino-ctc-asr-export',
    runtimeLayout: 'asr-ctc',
    envOverrides: {
      OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION: '1',
    },
  },
  {
    id: 'en-xenova-legacy-v1',
    kind: 'ctc',
    modelId: 'Xenova/wav2vec2-base-960h',
  },
  {
    id: 'qwen3-official-forced-aligner-v1',
    kind: 'native',
    modelId: 'Qwen/Qwen3-ForcedAligner-0.6B',
    nativeBackend: 'qwen3-forced-aligner',
  },
];

const nativeOverrides: NativeAlignmentOverride[] = [
  {
    profileId: 'qwen3-official-forced-aligner-v1',
    matches(context) {
      return Boolean(
        context.providerMeta?.forcedAlignment &&
          context.providerMeta.forcedAlignment.backend === 'qwen3-forced-aligner' &&
          context.providerMeta.forcedAlignment.applied
      );
    },
  },
];

const profileMap = new Map(alignmentProfiles.map((profile) => [profile.id, profile] as const));

export class AlignmentModelProfileRegistry {
  static list() {
    return [...alignmentProfiles];
  }

  static get(profileId: string) {
    return profileMap.get(String(profileId || '').trim()) || null;
  }

  static require(profileId: string) {
    const profile = this.get(profileId);
    if (!profile) {
      throw new Error(`Unknown alignment profile: ${profileId}`);
    }
    return profile;
  }

  static resolveNativeOverride(context: ForcedAlignmentSelectionContext) {
    const override = nativeOverrides.find((entry) => entry.matches(context));
    return override ? this.require(override.profileId) : null;
  }
}

