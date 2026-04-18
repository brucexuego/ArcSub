import type { CanonicalLlmRequest, CanonicalMessage, CanonicalMessagePart } from '../canonical/llm_types.js';

function textFromParts(parts: CanonicalMessagePart[]) {
  return parts
    .map((part) => (part.type === 'text' ? part.text : ''))
    .filter(Boolean)
    .join('');
}

function findMessageText(messages: CanonicalMessage[], role: CanonicalMessage['role']) {
  return messages
    .filter((message) => message.role === role)
    .map((message) => textFromParts(message.parts))
    .filter(Boolean)
    .join('\n\n');
}

export function getCanonicalInstructions(request: CanonicalLlmRequest) {
  return String(request.instructions || '').trim();
}

export function getCanonicalUserText(request: CanonicalLlmRequest) {
  const text = findMessageText(request.messages, 'user');
  return String(text || '').trim();
}

export function buildOpenAiChatMessages(request: CanonicalLlmRequest) {
  const messages: Array<{ role: 'system' | 'user'; content: string }> = [];
  const instructions = getCanonicalInstructions(request);
  const userText = getCanonicalUserText(request);

  if (instructions) {
    messages.push({ role: 'system', content: instructions });
  }
  if (userText) {
    messages.push({ role: 'user', content: userText });
  }
  return messages;
}

export function buildAnthropicMessages(request: CanonicalLlmRequest) {
  return [{ role: 'user' as const, content: getCanonicalUserText(request) }];
}

export function buildGeminiContents(request: CanonicalLlmRequest) {
  return [
    {
      role: 'user',
      parts: [{ text: getCanonicalUserText(request) }],
    },
  ];
}

export function buildGeminiSystemInstruction(request: CanonicalLlmRequest) {
  const instructions = getCanonicalInstructions(request);
  if (!instructions) return undefined;
  return {
    parts: [{ text: instructions }],
  };
}

export function buildOllamaPrompt(request: CanonicalLlmRequest) {
  const instructions = getCanonicalInstructions(request);
  const userText = getCanonicalUserText(request);
  return instructions ? `${instructions}\n\n${userText}` : userText;
}

export function wantsJsonObject(request: CanonicalLlmRequest) {
  return request.structuredOutput?.mode === 'json_object';
}

export function wantsJsonSchema(request: CanonicalLlmRequest) {
  return request.structuredOutput?.mode === 'json_schema' ? request.structuredOutput : null;
}
