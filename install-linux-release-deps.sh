#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
portable_node_bin="$script_dir/.arcsub-bootstrap/node/linux-x64/bin/node"
portable_npm_cli="$script_dir/.arcsub-bootstrap/node/linux-x64/lib/node_modules/npm/bin/npm-cli.js"

log() {
  printf '[ArcSub] %s\n' "$1"
}

if [[ ! -f "$script_dir/package.json" ]]; then
  printf '[ArcSub] Missing package.json in %s\n' "$script_dir" >&2
  exit 1
fi

cd "$script_dir"

if [[ -f "$script_dir/install-linux-system-deps.sh" ]]; then
  log "Tip: run ./install-linux-system-deps.sh first if this machine still lacks system packages."
fi

log "Installing production dependencies..."
if [[ -x "$portable_node_bin" && -f "$portable_npm_cli" ]]; then
  PATH="$(dirname "$portable_node_bin"):$PATH" \
  npm_node_execpath="$portable_node_bin" \
  npm_execpath="$portable_npm_cli" \
  "$portable_node_bin" "$portable_npm_cli" install --omit=dev --no-fund --no-audit --ignore-scripts
  PATH="$(dirname "$portable_node_bin"):$PATH" "$portable_node_bin" scripts/finalize-runtime-install.mjs
elif command -v npm >/dev/null 2>&1; then
  npm install --omit=dev --no-fund --no-audit --ignore-scripts
  node scripts/finalize-runtime-install.mjs
else
  printf '[ArcSub] npm is required. Run ./deploy.sh or ./install-linux-system-deps.sh first.\n' >&2
  exit 1
fi

if [[ -f "$script_dir/scripts/preflight-linux-runtime.sh" ]]; then
  log "Running Linux preflight..."
  bash "$script_dir/scripts/preflight-linux-runtime.sh"
fi

if [[ ! -f "$script_dir/.env" && -f "$script_dir/.env.example" ]]; then
  log "No .env found. Copy .env.example to .env and set ENCRYPTION_KEY before production use."
fi

log "Production dependency install completed."
