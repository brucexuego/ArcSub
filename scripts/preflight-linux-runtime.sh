#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
portable_node="$script_dir/.arcsub-bootstrap/node/linux-x64/bin/node"

log() {
  printf '[ArcSub] %s\n' "$1"
}

fail() {
  printf '[ArcSub] ERROR: %s\n' "$1" >&2
  exit 1
}

warn() {
  printf '[ArcSub] WARNING: %s\n' "$1" >&2
}

require_file() {
  local relative_path="$1"
  [[ -f "$script_dir/$relative_path" ]] || fail "Missing required file: $relative_path"
}

require_dir() {
  local relative_path="$1"
  [[ -d "$script_dir/$relative_path" ]] || fail "Missing required directory: $relative_path"
}

resolve_node() {
  if [[ -x "$portable_node" ]]; then
    printf '%s\n' "$portable_node"
    return 0
  fi
  command -v node 2>/dev/null || true
}

log "Running Linux runtime preflight..."

require_file "package.json"
require_file "build/server/index.js"
require_file "dist/index.html"
require_file "scripts/deploy-manifest.json"
require_file "scripts/install-deployment-assets.mjs"
require_file "scripts/finalize-runtime-install.mjs"
require_file "deploy.sh"
require_file "start.production.sh"
require_file "collect-diagnostics.sh"
require_dir "node_modules"

node_bin="$(resolve_node)"
[[ -n "$node_bin" ]] || fail "Node.js 22+ is required."
node_major="$("$node_bin" -p 'process.versions.node.split(`.`)[0]' 2>/dev/null || echo 0)"
[[ "$node_major" =~ ^[0-9]+$ && "$node_major" -ge 22 ]] || fail "Node.js 22+ is required; found major version $node_major."

for command_name in python3 ffmpeg espeak-ng; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    fail "Missing required command: $command_name"
  fi
done

if [[ ! -x "$script_dir/runtime/tools/yt-dlp" ]] && ! command -v yt-dlp >/dev/null 2>&1; then
  fail "Missing yt-dlp. Run ./install-linux-system-deps.sh or ./deploy.sh."
fi

for package_dir in \
  "express" \
  "fs-extra" \
  "lowdb" \
  "openvino-node" \
  "openvino-genai-node" \
  "onnxruntime-node" \
  "sherpa-onnx"; do
  require_dir "node_modules/$package_dir"
done

if [[ ! -f "$script_dir/.env" ]]; then
  warn ".env is not present yet. Run ./deploy.sh to create it and generate ENCRYPTION_KEY."
fi

log "Linux runtime preflight completed."
