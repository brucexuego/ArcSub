# Contributing to ArcSub

Public guides:

- English: [docs/en/getting-started.md](./docs/en/getting-started.md)
- 繁體中文: [docs/zh-TW/getting-started.md](./docs/zh-TW/getting-started.md)
- 日本語: [docs/ja/getting-started.md](./docs/ja/getting-started.md)

## Before opening a pull request

For source-repo work, use the normal dev startup helpers:

```powershell
npm install
.\start.ps1
```

```bash
npm install
./start.sh
```

Before opening a pull request, please run:

```powershell
npm run -s check
```

If your change affects startup, runtime preparation, or deployment behavior, it is also helpful to check:

```powershell
npm run clean:runtime:dry
```

## Contribution expectations

- keep public documentation clear and up to date
- avoid committing secrets or local-only data
- keep user-facing wording polite, clear, and consistent across languages
- keep release-facing docs and source-repo docs aligned
- verify changes before submitting

## Security issues

If your change involves a security concern, please also read [SECURITY.md](./SECURITY.md).

Please also follow [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) in issues, pull requests, and discussions.
