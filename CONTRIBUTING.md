# Contributing to ArcSub

Public guides:

- English: [docs/en/getting-started.md](./docs/en/getting-started.md)
- 繁體中文: [docs/zh-TW/getting-started.md](./docs/zh-TW/getting-started.md)
- 日本語: [docs/ja/getting-started.md](./docs/ja/getting-started.md)

## Before opening a pull request

Please open pull requests from a feature branch, not from `main`.

Good branch names are short and describe the change:

- `docs/add-german-readme`
- `docs/improve-installation-guide`
- `ui/clarify-empty-states`

Keep each pull request focused on one purpose. Smaller pull requests are easier
to review, test, and merge.

For source-repo work, use the normal dev startup helpers:

```powershell
npm install
.\start.ps1
```

```bash
npm install
./start.sh
```

For code changes, please run:

```powershell
npm run -s check
```

For documentation-only changes, a manual review of the changed links and text is
usually enough.

If your change affects startup, runtime preparation, or deployment behavior, it
is also helpful to check:

```powershell
npm run clean:runtime:dry
```

## Contribution expectations

- keep public documentation clear and up to date
- avoid committing secrets or local-only data
- keep user-facing wording polite, clear, and consistent across languages
- keep release-facing docs and source-repo docs aligned
- verify changes before submitting

## Documentation and translation contributions

Documentation and translation contributions are welcome. Please keep them
user-facing and avoid internal implementation details that normal users do not
need.

When adding a new translated README:

- add the new `README.<locale>.md` file
- add the new language link to the existing root README files
- keep the same public structure as the existing README files where practical
- link download instructions to the latest GitHub Release instead of a fixed
  version number
- if localized docs such as `docs/<locale>/` do not exist yet, link to the
  existing English docs for now
- do not include private paths, local runtime snapshots, API keys, tokens,
  personal media, or `.env` content

## Security issues

If your change involves a security concern, please also read [SECURITY.md](./SECURITY.md).

Please also follow [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) in issues, pull requests, and discussions.
