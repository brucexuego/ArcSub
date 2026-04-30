import assert from 'node:assert/strict';
import fs from 'fs-extra';

async function main() {
  const asrService = await fs.readFile('server/services/asr_service.ts', 'utf8');
  assert.equal(asrService.includes('\uFFFD'), false, 'asr_service.ts contains replacement characters.');
  assert.ok(asrService.includes('\\u3002') && asrService.includes('\\uFF01') && asrService.includes('\\uFF1F'), 'CJK sentence punctuation guard is missing.');
  assert.ok(asrService.includes('\\u300C') && asrService.includes('\\u300D') && asrService.includes('\\u300E') && asrService.includes('\\u300F'), 'CJK bracket punctuation guard is missing.');
  assert.ok(asrService.includes('\\u2026'), 'Ellipsis punctuation guard is missing.');
  assert.ok(asrService.includes("(['’])"), 'Display text apostrophe normalization must preserve straight and curly apostrophes.');
  assert.ok(asrService.includes('([\\(\\[\\{¿¡])'), 'Display text opening punctuation normalization must preserve inverted punctuation.');
  assert.ok(asrService.includes("['’-]"), 'Lexical token regex must preserve apostrophes and hyphen without treating question marks as word joiners.');

  const files = [
    'server/services/asr_service.ts',
    'server/services/cloud_asr/runtime.ts',
    'server/services/cloud_asr/resolver.ts',
    'server/services/cloud_asr/connection_probe.ts',
  ];
  for (const file of files) {
    const text = await fs.readFile(file, 'utf8');
    assert.equal(text.includes('legacy_runtime'), false, `${file} still references legacy_runtime.`);
    assert.equal(text.includes('cloud_asr_adapter'), false, `${file} still references cloud_asr_adapter.`);
    assert.equal(text.includes('cloud_asr_provider'), false, `${file} still references cloud_asr_provider.`);
  }

  const cloudRuntime = await fs.readFile('server/services/cloud_asr/runtime.ts', 'utf8');
  assert.ok(
    cloudRuntime.includes("if (resolvedProvider.provider === 'google-gemini')") &&
      cloudRuntime.includes('enforceGeminiFreeTierRateLimit'),
    'Gemini free-tier limiter must be gated to the google-gemini provider.'
  );
  const geminiProvider = await fs.readFile('server/services/cloud_asr/providers/google_gemini_audio.ts', 'utf8');
  const envExample = await fs.readFile('.env.example', 'utf8');
  for (const envName of [
    'ASR_GEMINI_FREE_TIER_LIMITER_ENABLED',
    'ASR_GEMINI_FREE_TIER_RPM',
    'ASR_GEMINI_FREE_TIER_TPM',
    'ASR_GEMINI_FREE_TIER_RPD',
  ]) {
    assert.ok(geminiProvider.includes(envName), `Gemini provider does not read ${envName}.`);
    assert.ok(envExample.includes(envName), `.env.example does not document ${envName}.`);
  }

  const diarizationService = await fs.readFile('server/diarization_service.ts', 'utf8');
  const speechToText = await fs.readFile('src/components/SpeechToText.tsx', 'utf8');
  const textTranslation = await fs.readFile('src/components/TextTranslation.tsx', 'utf8');
  const transcribeRoute = await fs.readFile('server/http/routes/transcribe_route.ts', 'utf8');
  for (const [label, text] of [
    ['diarization service', diarizationService],
    ['SpeechToText', speechToText],
    ['TextTranslation', textTranslation],
    ['transcribe route', transcribeRoute],
  ] as const) {
    assert.equal(text.includes('GoogleGenAI'), false, `${label} still imports Gemini semantic diarization code.`);
    assert.equal(text.includes('performSemanticDiarization'), false, `${label} still exposes semantic diarization.`);
    assert.equal(text.includes('semanticFallbackEnabled'), false, `${label} still references semantic diarization fallback.`);
    assert.equal(text.includes('providerSemantic'), false, `${label} still renders semantic diarization provider labels.`);
    assert.equal(text.includes('sourceSemantic'), false, `${label} still renders semantic diarization source labels.`);
  }

  console.log('ASR source guard checks passed.');
}

void main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
