# AI Docs Index

This directory is the canonical home for AI-facing technical documentation in `ArcSub`.

Root rule:
- Keep only [AGENTS.md](../../AGENTS.md) at the repository root as the agent entry point.
- Put future agent-oriented contracts, architecture notes, checklists, and runtime integration guides under `.agents/docs`.

Current structure:
- [playbooks/AGENTS_CHECKLIST.md](./playbooks/AGENTS_CHECKLIST.md)
- [playbooks/TRANSLATION_MODEL_INTEGRATION_PLAYBOOK.md](./playbooks/TRANSLATION_MODEL_INTEGRATION_PLAYBOOK.md)
- [architecture/ASR_TRANSLATION_INTEGRATION.md](./architecture/ASR_TRANSLATION_INTEGRATION.md)
- [architecture/ALIGNMENT_ARCHITECTURE.md](./architecture/ALIGNMENT_ARCHITECTURE.md)
- [architecture/DIARIZATION_GUIDE.md](./architecture/DIARIZATION_GUIDE.md)
- [architecture/DEPLOYMENT_RELEASE_ARCHITECTURE.md](./architecture/DEPLOYMENT_RELEASE_ARCHITECTURE.md)
- [architecture/OPENVINO_LOCAL_MODELS.md](./architecture/OPENVINO_LOCAL_MODELS.md)
- [contracts/TRANSLATION_CONTRACT.md](./contracts/TRANSLATION_CONTRACT.md)
- [contracts/DIARIZATION_CONTRACT.md](./contracts/DIARIZATION_CONTRACT.md)
- [maintenance/DOC_SYNC_CHECKLIST.md](./maintenance/DOC_SYNC_CHECKLIST.md)
- [maintenance/2026-04-21_MONITOR_AND_TRANSLATION_PIPELINE_REFACTOR.md](./maintenance/2026-04-21_MONITOR_AND_TRANSLATION_PIPELINE_REFACTOR.md)

Placement rule:
- Agent-facing docs go here.
- User-facing product docs stay under `docs/` or the repository root `README*` files.

Recommended reading order for major STT / translation work:
1. [architecture/ASR_TRANSLATION_INTEGRATION.md](./architecture/ASR_TRANSLATION_INTEGRATION.md)
2. [playbooks/TRANSLATION_MODEL_INTEGRATION_PLAYBOOK.md](./playbooks/TRANSLATION_MODEL_INTEGRATION_PLAYBOOK.md)
3. [architecture/ALIGNMENT_ARCHITECTURE.md](./architecture/ALIGNMENT_ARCHITECTURE.md)
4. [architecture/DIARIZATION_GUIDE.md](./architecture/DIARIZATION_GUIDE.md)
5. [architecture/DEPLOYMENT_RELEASE_ARCHITECTURE.md](./architecture/DEPLOYMENT_RELEASE_ARCHITECTURE.md)
6. [contracts/TRANSLATION_CONTRACT.md](./contracts/TRANSLATION_CONTRACT.md)
7. [architecture/OPENVINO_LOCAL_MODELS.md](./architecture/OPENVINO_LOCAL_MODELS.md)
8. [maintenance/DOC_SYNC_CHECKLIST.md](./maintenance/DOC_SYNC_CHECKLIST.md)

Category intent:
- `playbooks/`: agent operating checklists and execution habits
- `architecture/`: cross-module system maps, runtime architecture, and implementation structure
- `contracts/`: non-negotiable behavior that code must preserve
- `maintenance/`: rules for documentation sync and long-term upkeep
