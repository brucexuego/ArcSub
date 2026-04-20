import fs from 'node:fs';
import path from 'node:path';
import { getLocalModelInstallDir, type LocalModelDefinition } from '../../local_model_catalog.js';
import type { LocalTranslateModelStrategy, LocalTranslatePromptSignature } from './types.js';

export type { LocalTranslateModelStrategy, LocalTranslatePromptSignature, LocalTranslatePromptStyle } from './types.js';

const localTranslatePromptSignatureCache = new Map<string, LocalTranslatePromptSignature>();
const localTranslateModelStrategyCache = new Map<string, LocalTranslateModelStrategy>();
const DEEPSEEK_USER_TOKEN = '<\uff5cUser\uff5c>';
const DEEPSEEK_ASSISTANT_THINK_TOKEN = '<\uff5cAssistant\uff5c><think>';

function isQwenFamilyLocalModel(localModel: LocalModelDefinition) {
  const combined = `${localModel.id} ${localModel.repoId} ${localModel.displayName}`.toLowerCase();
  return combined.includes('qwen');
}

export function readLocalTranslatePromptSignature(localModel: LocalModelDefinition): LocalTranslatePromptSignature {
  const cacheKey = `${localModel.id}:${localModel.repoId}:${localModel.localSubdir}`;
  const cached = localTranslatePromptSignatureCache.get(cacheKey);
  if (cached) return cached;

  const modelDir = getLocalModelInstallDir(localModel);
  const tokenizerConfigPath = path.join(modelDir, 'tokenizer_config.json');
  const chatTemplatePath = path.join(modelDir, 'chat_template.jinja');
  const signatureParts = [localModel.id, localModel.repoId, localModel.displayName, localModel.localSubdir];

  if (fs.existsSync(tokenizerConfigPath)) {
    try {
      const parsed = JSON.parse(fs.readFileSync(tokenizerConfigPath, 'utf8'));
      signatureParts.push(String(parsed?.chat_template || ''), String(parsed?.bos_token || ''), String(parsed?.eos_token || ''));
    } catch {
      // Fall back to name-based detection if tokenizer metadata is unreadable.
    }
  }

  if (fs.existsSync(chatTemplatePath)) {
    try {
      signatureParts.push(fs.readFileSync(chatTemplatePath, 'utf8'));
    } catch {
      // Fall back to name-based detection if the Jinja template is unreadable.
    }
  }

  const signature = {
    fingerprint: signatureParts.join('\n').toLowerCase(),
    tokenizerConfigPath: fs.existsSync(tokenizerConfigPath) ? tokenizerConfigPath : null,
    chatTemplatePath: fs.existsSync(chatTemplatePath) ? chatTemplatePath : null,
  } satisfies LocalTranslatePromptSignature;
  localTranslatePromptSignatureCache.set(cacheKey, signature);
  return signature;
}

export function inferLocalTranslateModelStrategy(localModel: LocalModelDefinition): LocalTranslateModelStrategy {
  const cacheKey = `${localModel.id}:${localModel.repoId}:${localModel.localSubdir}`;
  const cached = localTranslateModelStrategyCache.get(cacheKey);
  if (cached) return cached;

  const combined = `${localModel.id} ${localModel.repoId} ${localModel.displayName} ${localModel.localSubdir}`.toLowerCase();
  const signature = readLocalTranslatePromptSignature(localModel).fingerprint;

  let strategy: LocalTranslateModelStrategy;

  if (
    localModel.runtime === 'openvino-seq2seq-translate' ||
    localModel.runtimeLayout === 'translate-seq2seq' ||
    /(?:^|[-_])(flan-|mt5|umt5|t5|bart|mbart|marian|pegasus|m2m100|m2m_100|fsmt|prophetnet)(?:$|[-_])/.test(combined)
  ) {
    strategy = {
      family: 'seq2seq',
      promptStyle: 'generic',
      generationStyle: 'generic',
      qwenOptimized: false,
      qwen3Optimized: false,
    };
  } else if (
    combined.includes('deepseek-r1') ||
    signature.includes(DEEPSEEK_USER_TOKEN.toLowerCase()) ||
    signature.includes(DEEPSEEK_ASSISTANT_THINK_TOKEN.toLowerCase())
  ) {
    strategy = {
      family: 'deepseek_r1_distill_qwen',
      promptStyle: 'deepseek_r1_distill_qwen',
      generationStyle: 'deepseek_r1',
      qwenOptimized: false,
      qwen3Optimized: false,
    };
  } else if (combined.includes('qwen3') || signature.includes('enable_thinking') || signature.includes('/no_think')) {
    strategy = {
      family: 'qwen3',
      promptStyle: 'qwen3_non_thinking',
      generationStyle: 'qwen3',
      qwenOptimized: true,
      qwen3Optimized: true,
    };
  } else if (
    combined.includes('qwen2.5') ||
    (combined.includes('qwen') && combined.includes('instruct')) ||
    (signature.includes('<|im_start|>system') && signature.includes('you are qwen'))
  ) {
    strategy = {
      family: 'qwen2_5',
      promptStyle: 'qwen_chatml',
      generationStyle: 'qwen',
      qwenOptimized: true,
      qwen3Optimized: false,
    };
  } else if (
    combined.includes('phi-4') ||
    combined.includes('phi_4') ||
    combined.includes('phi4') ||
    (signature.includes('<|system|>') &&
      signature.includes('<|user|>') &&
      signature.includes('<|assistant|>') &&
      signature.includes('<|end|>'))
  ) {
    strategy = {
      family: 'phi4',
      promptStyle: 'phi4_chat',
      generationStyle: 'generic',
      qwenOptimized: false,
      qwen3Optimized: false,
    };
  } else if (
    combined.includes('translategemma') ||
    (signature.includes('source_lang_code') && signature.includes('target_lang_code'))
  ) {
    strategy = {
      family: 'translategemma',
      promptStyle: 'translategemma_google',
      generationStyle: 'generic',
      qwenOptimized: false,
      qwen3Optimized: false,
    };
  } else if (
    combined.includes('gemma-3') ||
    combined.includes('gemma 3') ||
    signature.includes('<start_of_turn>user') ||
    signature.includes('start_of_turn')
  ) {
    strategy = {
      family: 'gemma3',
      promptStyle: 'gemma_plain',
      generationStyle: 'generic',
      qwenOptimized: false,
      qwen3Optimized: false,
    };
  } else {
    strategy = {
      family: 'generic',
      promptStyle: 'generic',
      generationStyle: isQwenFamilyLocalModel(localModel) ? 'qwen' : 'generic',
      qwenOptimized: isQwenFamilyLocalModel(localModel),
      qwen3Optimized: false,
    };
  }

  localTranslateModelStrategyCache.set(cacheKey, strategy);
  return strategy;
}
