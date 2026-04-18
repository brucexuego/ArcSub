import type { LocalTranslateModelStrategy, LocalTranslatePromptStyle } from './types.js';
import type { TranslationQualityMode } from '../llm/orchestrators/translation_quality_policy.js';

interface LocalPromptLabels {
  additionalInstructionsLabel?: string;
  inputLabel?: string;
  inputJsonLabel?: string;
}

interface BuildLocalTranslationPromptInput {
  text: string;
  systemPrompt: string;
  sourceAwarePrompt?: string;
  customPrompt?: string;
  translationQualityMode?: TranslationQualityMode;
  modelStrategy: LocalTranslateModelStrategy;
  promptStyle?: LocalTranslatePromptStyle;
  labels?: LocalPromptLabels | null;
  deepseekPromptBuilder?: ((userText: string) => string) | null;
  deepseekPlainPromptBuilder?: ((userText: string) => string) | null;
}

interface BuildLocalJsonRepairPromptInput {
  payload: string;
  basePrompt: string;
  sourceAwarePrompt?: string;
  translationQualityMode?: TranslationQualityMode;
  modelStrategy: LocalTranslateModelStrategy;
  promptStyle?: LocalTranslatePromptStyle;
  labels?: LocalPromptLabels | null;
  deepseekPromptBuilder?: ((userText: string) => string) | null;
  deepseekPlainPromptBuilder?: ((userText: string) => string) | null;
}

export function buildQwenChatMlPrompt(input: { systemPrompt: string; userText: string }) {
  const systemPrompt = String(input.systemPrompt || '').trim();
  const userText = String(input.userText || '').trim();
  return [
    '<|im_start|>system',
    systemPrompt,
    '<|im_end|>',
    '<|im_start|>user',
    userText,
    '<|im_end|>',
    '<|im_start|>assistant',
    '',
  ].join('\n');
}

export function buildQwen3NonThinkingChatPrompt(input: { systemPrompt: string; userText: string }) {
  const systemPrompt = String(input.systemPrompt || '').trim();
  const userText = String(input.userText || '').trim();
  return [
    '<|im_start|>system',
    systemPrompt,
    '<|im_end|>',
    '<|im_start|>user',
    userText,
    '<|im_end|>',
    '<|im_start|>assistant',
    '<think>',
    '',
    '</think>',
    '',
  ].join('\n');
}

export function buildPhi4ChatPrompt(input: { systemPrompt: string; userText: string }) {
  const systemPrompt = String(input.systemPrompt || '').trim();
  const userText = String(input.userText || '').trim();
  return `<|system|>${systemPrompt}<|end|><|user|>${userText}<|end|><|assistant|>`;
}

export function buildGemmaChatPrompt(input: { systemPrompt: string; userText: string }) {
  const systemPrompt = String(input.systemPrompt || '').trim();
  const userText = String(input.userText || '').trim();
  const mergedUserText = [systemPrompt, userText].filter(Boolean).join('\n\n');
  return [
    '<bos><start_of_turn>user',
    mergedUserText,
    '<end_of_turn>',
    '<start_of_turn>model',
    '',
  ].join('\n');
}

function buildResolvedSystemPrompt(input: {
  basePrompt: string;
  sourceAwarePrompt?: string;
  customPrompt?: string;
  labels?: LocalPromptLabels | null;
}) {
  const additionalInstructionsLabel = input.labels?.additionalInstructionsLabel || 'Additional instructions:';
  return [
    String(input.basePrompt || '').trim(),
    ...(String(input.sourceAwarePrompt || '').trim()
      ? ['', additionalInstructionsLabel, String(input.sourceAwarePrompt || '').trim()]
      : []),
    ...(String(input.customPrompt || '').trim()
      ? ['', additionalInstructionsLabel, String(input.customPrompt || '').trim()]
      : []),
  ]
    .filter(Boolean)
    .join('\n');
}

export function buildLocalTranslationPrompt(input: BuildLocalTranslationPromptInput) {
  const promptStyle = input.promptStyle || input.modelStrategy.promptStyle;
  const resolvedSystemPrompt = buildResolvedSystemPrompt({
    basePrompt: input.systemPrompt,
    sourceAwarePrompt: input.sourceAwarePrompt,
    customPrompt: input.customPrompt,
    labels: input.labels,
  });

  if (promptStyle === 'qwen3_non_thinking') {
    return buildQwen3NonThinkingChatPrompt({
      systemPrompt: resolvedSystemPrompt,
      userText: input.text,
    });
  }

  if (promptStyle === 'qwen_chatml') {
    return buildQwenChatMlPrompt({
      systemPrompt: resolvedSystemPrompt,
      userText: input.text,
    });
  }

  if (promptStyle === 'phi4_chat') {
    return buildPhi4ChatPrompt({
      systemPrompt: resolvedSystemPrompt,
      userText: input.text,
    });
  }

  if (promptStyle === 'gemma_plain') {
    return buildGemmaChatPrompt({
      systemPrompt: resolvedSystemPrompt,
      userText: input.text,
    });
  }

  if (promptStyle === 'deepseek_r1_distill_qwen' && input.deepseekPromptBuilder) {
    const mergedUserText = [
      resolvedSystemPrompt,
      '',
      input.labels?.inputLabel || 'Input:',
      '',
      input.text,
    ]
      .filter(Boolean)
      .join('\n');
    return input.deepseekPromptBuilder(mergedUserText);
  }

  if (promptStyle === 'deepseek_r1_plain' && input.deepseekPlainPromptBuilder) {
    const mergedUserText = [
      resolvedSystemPrompt,
      '',
      input.labels?.inputLabel || 'Input:',
      '',
      input.text,
    ]
      .filter(Boolean)
      .join('\n');
    return input.deepseekPlainPromptBuilder(mergedUserText);
  }

  return [
    resolvedSystemPrompt,
    input.labels?.inputLabel || 'Input:',
    '',
    input.text,
  ].join('\n');
}

export function buildLocalJsonRepairPrompt(input: BuildLocalJsonRepairPromptInput) {
  const promptStyle = input.promptStyle || input.modelStrategy.promptStyle;
  const resolvedBase = buildResolvedSystemPrompt({
    basePrompt: input.basePrompt,
    sourceAwarePrompt: input.sourceAwarePrompt,
    labels: input.labels,
  });

  if (promptStyle === 'qwen3_non_thinking') {
    return buildQwen3NonThinkingChatPrompt({
      systemPrompt: resolvedBase,
      userText: input.payload,
    });
  }

  if (promptStyle === 'qwen_chatml') {
    return buildQwenChatMlPrompt({
      systemPrompt: resolvedBase,
      userText: input.payload,
    });
  }

  if (promptStyle === 'phi4_chat') {
    return buildPhi4ChatPrompt({
      systemPrompt: resolvedBase,
      userText: input.payload,
    });
  }

  if (promptStyle === 'gemma_plain') {
    return buildGemmaChatPrompt({
      systemPrompt: resolvedBase,
      userText: input.payload,
    });
  }

  if (promptStyle === 'deepseek_r1_distill_qwen' && input.deepseekPromptBuilder) {
    const mergedUserText = [
      resolvedBase,
      '',
      input.labels?.inputJsonLabel || 'Input JSON:',
      '',
      input.payload,
    ]
      .filter(Boolean)
      .join('\n');
    return input.deepseekPromptBuilder(mergedUserText);
  }

  if (promptStyle === 'deepseek_r1_plain' && input.deepseekPlainPromptBuilder) {
    const mergedUserText = [
      resolvedBase,
      '',
      input.labels?.inputJsonLabel || 'Input JSON:',
      '',
      input.payload,
    ]
      .filter(Boolean)
      .join('\n');
    return input.deepseekPlainPromptBuilder(mergedUserText);
  }

  return [
    resolvedBase,
    '',
    input.labels?.inputJsonLabel || 'Input JSON:',
    input.payload,
  ].join('\n');
}
