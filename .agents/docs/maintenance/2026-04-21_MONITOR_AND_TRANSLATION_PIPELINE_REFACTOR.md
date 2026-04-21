# 2026-04-21 Monitor And Translation Pipeline Refactor

Local maintenance record for the monitor/debug/i18n refactor completed on `2026-04-21`.

This note is intentionally AI-facing and implementation-oriented. It records:
- what changed
- what contract assumptions now exist
- how the change set was validated without exhausting local VRAM

## Scope

Primary goals of this refactor:
- unify STT and Translation monitor behavior around the shared `RunMonitor`
- make debug/progress output more consistent and easier for end users to understand
- stop leaking raw warning codes into the main UI
- improve long-string handling in monitor surfaces
- align newer translation-model pipeline messages, especially `TranslateGemma`, with the same localization and monitor conventions

Main code paths touched:
- [src/components/RunMonitor.tsx](../../../src/components/RunMonitor.tsx)
- [shared/run_monitor.ts](../../../shared/run_monitor.ts)
- [src/components/SpeechToText.tsx](../../../src/components/SpeechToText.tsx)
- [src/components/TextTranslation.tsx](../../../src/components/TextTranslation.tsx)
- [src/components/VideoDownloader.tsx](../../../src/components/VideoDownloader.tsx)
- [src/components/VideoPlayer.tsx](../../../src/components/VideoPlayer.tsx)
- [src/i18n/translations.ts](../../../src/i18n/translations.ts)
- [server/http/routes/run_progress_events.ts](../../../server/http/routes/run_progress_events.ts)
- [server/http/routes/transcribe_route.ts](../../../server/http/routes/transcribe_route.ts)
- [server/http/routes/translation_routes.ts](../../../server/http/routes/translation_routes.ts)
- [server/services/local_asr/debug.ts](../../../server/services/local_asr/debug.ts)
- [server/services/translation_service.ts](../../../server/services/translation_service.ts)
- [server/services/local_llm/orchestrators/local_translation_orchestrator.ts](../../../server/services/local_llm/orchestrators/local_translation_orchestrator.ts)

## Resulting monitor rules

### 1. Shared monitor surface

STT and Translation now rely on the same monitor component and the same high-level data model.

Important expectations:
- `RunMonitor` is the canonical monitor shell for run-progress UI
- the user-facing surface should prefer localized descriptions over raw warning codes
- technical detail still belongs in expandable diagnostics, not in primary status copy

### 2. Progress contract

Backend progress events now support a code-first shape through:
- [server/http/routes/run_progress_events.ts](../../../server/http/routes/run_progress_events.ts)

Legacy raw `message` strings still exist for compatibility, but UI logic should prefer:
- `event.code`
- `event.stage`
- `event.progressHint`

Do not add new UI behavior that depends only on parsing free-form backend English strings if a stable event code can be used instead.

### 3. Warning and issue handling

Warning presentation is now split into:
- localized user-facing text in the page layer
- structured `warningIssues` / `errorIssues` for monitor sections and future aggregation

Newer translation-pipeline warnings that must remain mapped:
- `translategemma_batch_translation_applied`
- `translategemma_recursive_chunk_split_applied`
- `translategemma_single_line_retry_applied`
- `residual_line_retry_triggered`
- `residual_line_retry_applied`
- `cloud_context_parse_failed`
- `cloud_context_chunk_split`
- `cloud_context_split_depth_exhausted`
- `cloud_context_single_line_fallback`
- `quality_issue_*`

Do not regress to showing these raw codes directly in the normal UI.

## Translation pipeline alignment

This refactor also verified that the newer translation-model path was brought into the same monitor/debug system.

Confirmed behaviors:
- `TranslateGemma` requests can carry `sourceLang` from STT when appropriate
- Translation debug now includes `quality`, `timing`, `diagnostics`, `warningIssues`, `errorIssues`, and `artifacts`
- the local orchestrator and translation service both classify newer warning codes into structured `RunIssue` entries
- UI warning text for `TranslateGemma` and cloud-context retries is localized for:
  - `zh-tw`
  - `zh-cn`
  - `en`
  - `jp`
  - `de`

The most important cleanup here was not only adding keys, but rewriting some messages to be less engineering-heavy for end users.

Examples of wording that were intentionally softened:
- "token-aware batch translation" -> describe as splitting subtitles into smaller batches to avoid sending too much content at once
- "cloud context window" -> describe as translating with surrounding subtitle context

## i18n notes

This change set added and refined monitor/pipeline wording in:
- [src/i18n/translations.ts](../../../src/i18n/translations.ts)

Special care points:
- newer translation-pipeline overrides are applied late so older overrides do not silently replace them
- long status messages must remain readable in monitor pills and details panels
- technical identifiers such as provider names, profile ids, model ids, and backend names should remain technical unless there is a clear user-facing alias

## UI layout rules reinforced here

### 1. Long-string behavior

`RunMonitor` was adjusted so narrow layouts and long messages do not destroy the card layout.

Current expectations:
- header content may stack
- status pills may wrap
- details summary may wrap instead of hard-truncating everything
- details fields should break or stack cleanly on small widths
- badge overflow should compress rather than exploding the layout

### 2. Consistency over per-page invention

If future STT / Translation / Downloader changes need another monitor tweak, bias toward changing:
- [src/components/RunMonitor.tsx](../../../src/components/RunMonitor.tsx)

Avoid rebuilding separate monitor patterns per page unless there is a strong reason.

## Validation strategy used

This refactor was explicitly validated with a low-VRAM strategy.

Reason:
- the local machine only has about `8 GB` VRAM available
- simultaneous model-matrix testing is more likely to create false failures, warm-cache pollution, and repeated reruns than to improve confidence

### Rule followed

Do not test by loading multiple heavy ASR or translation models at the same time.

Prefer:
- one real ASR model at a time
- release runtime
- one real translation model at a time

Avoid:
- dual-ASR comparisons in the same run
- dual-translation comparisons in the same run
- combined ASR + translation heavy runtime loads kept resident together unless strictly necessary

## Validation completed

### Static validation

Completed:
- `npm run -s typecheck`
- `npm run -s build`
- `npm run -s build:server`

### i18n / contract checks

Completed:
- direct script verification that new translation-pipeline monitor keys exist for:
  - `zh-tw`
  - `zh-cn`
  - `en`
  - `jp`
  - `de`

### Browser smoke

Completed with a low-load mocked-flow strategy:
- created a fresh project
- uploaded a `1` second test video through the real upload flow
- verified Downloader monitor behavior
- verified STT monitor behavior with a mocked ASR event stream
- verified Translation monitor behavior with a mocked TranslateGemma event stream
- verified Player page rendering
- verified Settings page rendering

Observed result:
- no browser page errors
- no console errors
- no raw warning code leakage in the normal STT / Translation UI

### Serial real-model smoke

Completed with real local models, but strictly serial:

1. Generated a short real speech WAV file
2. Uploaded it into a fresh project
3. Ran one real ASR model only:
   - `local_asr_openvino_whisper_medium_int8_ov`
4. Released ASR runtime
5. Ran one real translation model only:
   - `local_translate_google_translategemma_4b_it`
6. Released translation runtime

Observed output:
- ASR produced non-empty English text
- TranslateGemma produced non-empty `zh-TW` text
- both returned structured debug data with timing and issue fields

## Important findings from testing

### 1. The testing assumption was wrong, not the code

During smoke validation, the first translation assertion assumed:
- `strictJsonLineRepair=1`

Actual UI behavior in:
- [src/components/TextTranslation.tsx](../../../src/components/TextTranslation.tsx)

Current default:
- `strictJsonLineRepairEnabled === false`
- request sends `strictJsonLineRepair=0`

This was a test-harness assumption issue, not a product bug.

### 2. Real-model ASR quality is separate from monitor correctness

In real-model smoke, the ASR output rendered:
- `ArcSub` as `Oxup`

That is a model recognition error on short synthesized speech, not evidence of a monitor/debug contract failure.

## What to keep in mind for future work

- If monitor wording changes, update `translations.ts` in the same change set.
- If progress-stage behavior changes, prefer updating event-code mapping rather than adding more raw-string parsing.
- If new translation-specialized warnings are introduced, update both:
  - localization mapping in the UI
  - structured `warningIssues` classification in the backend
- If a future refactor changes debug shape, record it here or in the matching contract/architecture doc immediately.

## Follow-up candidates

Reasonable future follow-up work:
- add a small dedicated contract doc for shared run-monitor fields if the current shape grows further
- add a lightweight automated smoke script under `.private/` for repeatable serial real-model validation
- consider a focused `check:i18n` script if translation-key completeness keeps expanding
