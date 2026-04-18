# Security Policy

Public guides:

- English: [docs/en/getting-started.md](./docs/en/getting-started.md)
- 繁體中文: [docs/zh-TW/getting-started.md](./docs/zh-TW/getting-started.md)
- 日本語: [docs/ja/getting-started.md](./docs/ja/getting-started.md)

Please avoid posting full exploit details in a public issue.

## Preferred reporting path

1. Use a private GitHub Security Advisory if available.
2. If private reporting is not available, open a minimal public issue and request a private channel.

## Examples of security-sensitive areas

- request filtering and access checks
- settings storage and secret handling
- upload and file-processing paths
- code that validates untrusted input

## General guidance

- do not commit secrets, tokens, or private local data
- do not attach raw `.env`, full `runtime/` snapshots, or personal media/project files to public reports
- keep reproduction steps as small as possible in public reports
- when in doubt, prefer private reporting first
