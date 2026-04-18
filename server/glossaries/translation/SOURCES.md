# Translation Glossary Sources

This directory contains two kinds of glossary content:

- source-managed base and reviewed template layers maintained through ArcSub glossary tooling
- locale-specific override files such as `zh-tw/*` and `zh-cn/*` for product or region preferences

## Integrated Sources

### NAER Computing Terminology

- Source: National Academy for Educational Research open data
- Dataset: `國家教育研究院-兩岸對照名詞-計算機學術名詞`
- URL: <https://data.gov.tw/dataset/15275>
- Direct CSV used by the glossary maintenance workflow:
  <https://opendata.naer.edu.tw/學術名詞/國家教育研究院-兩岸對照名詞-計算機學術名詞.csv>
- Primary use:
  - `zh-hant` reviewed baseline terms
  - `zh-hans` reviewed baseline terms
- Notes:
  - The script uses NAER as the authoritative source-presence check for shared computing terms.
  - ArcSub may intentionally keep a product-preferred wording when the official term is correct but less natural for subtitle or AI contexts.

### Glosario

- Source: The Carpentries Glosario project
- Repository: <https://github.com/carpentries/glosario>
- Canonical glossary file used by the glossary maintenance workflow:
  <https://raw.githubusercontent.com/carpentries/glosario/main/glossary.yml>
- Primary use:
  - English canonical terminology validation
  - A small set of Git/software terms with stable community wording

### JMdict / EDRDG

- Source: EDRDG JMdict project
- Project page: <https://www.edrdg.org/wiki/index.php/JMdict-EDICT_Dictionary_Project>
- FTP archive used by the glossary maintenance workflow:
  `ftp://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz`
- Primary use:
  - Japanese term candidate validation for reviewed mappings
- Notes:
  - JMdict is a general dictionary, not a software-only glossary.
  - ArcSub only syncs a reviewed subset and does not blindly import every candidate.

## Local Review Policy

Some terms intentionally differ from the upstream source wording because ArcSub optimizes for:

- subtitle readability
- AI and ASR technical contexts
- Taiwan-preferred Traditional Chinese software wording
- modern Japanese software vocabulary

Examples:

- `speech recognition` keeps `語音辨識` instead of mechanically forcing a less common official variant
- `token` is normalized to `詞元` / `词元` in the technical subtitle layer because ArcSub is AI-heavy and the auth-token sense is not the dominant target here
- `deployment`, `latency`, `context window`, and `speaker diarization` remain reviewed product terms until a better directly licensable upstream glossary is added
- `subtitle_asr_recovery`, `subtitle_concise_spoken`, and `subtitle_formal_precise` include reviewed template vocabulary even when there is no single upstream source because those terms need to stay aligned with ArcSub's diagnostics, editor wording, and subtitle tone targets
- prompt-safety vocabulary such as `prompt injection`, `jailbreak`, `guardrail`, and ASR diagnostic labels such as `misrecognition`, `hesitation`, and `speaker change` are also reviewed locally because ArcSub needs stable wording across runtime logs, prompt templates, and subtitle repair flows

When adding more sources later, keep source-managed shared layers conservative and place product-specific deviations in locale override files where possible.
