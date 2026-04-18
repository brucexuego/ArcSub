#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_entry="$script_dir/build/server/index.js"
portable_node="$script_dir/.arcsub-bootstrap/node/linux-x64/bin/node"
listen_host="${HOST:-127.0.0.1}"
listen_port="${PORT:-3000}"
no_browser=0

prepend_runtime_path() {
  local value="$1"
  local current="$2"
  if [[ -z "$value" ]]; then
    printf '%s' "$current"
    return 0
  fi
  if [[ -z "$current" ]]; then
    printf '%s' "$value"
    return 0
  fi
  printf '%s:%s' "$value" "$current"
}

refresh_runtime_env() {
  local openvino_bin="$script_dir/node_modules/openvino-node/bin"
  local openvino_genai_bin="$script_dir/node_modules/openvino-genai-node/bin"

  if [[ -d "$openvino_bin" ]]; then
    export LD_LIBRARY_PATH
    LD_LIBRARY_PATH="$(prepend_runtime_path "$openvino_bin" "${LD_LIBRARY_PATH:-}")"
  fi
  if [[ -d "$openvino_genai_bin" ]]; then
    export LD_LIBRARY_PATH
    LD_LIBRARY_PATH="$(prepend_runtime_path "$openvino_genai_bin" "${LD_LIBRARY_PATH:-}")"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      listen_host="${2:-$listen_host}"
      shift 2
      ;;
    --port)
      listen_port="${2:-$listen_port}"
      shift 2
      ;;
    --no-browser)
      no_browser=1
      shift
      ;;
    *)
      printf '[ArcSub] Unknown argument: %s\n' "$1" >&2
      exit 1
      ;;
  esac
done

resolve_node_bin() {
  if [[ -x "$portable_node" ]]; then
    printf '%s\n' "$portable_node"
    return 0
  fi

  if command -v node >/dev/null 2>&1; then
    local system_node
    system_node="$(command -v node)"
    if [[ "$("$system_node" -p 'process.versions.node.split(`.`)[0]' 2>/dev/null || echo 0)" -ge 22 ]]; then
      printf '%s\n' "$system_node"
      return 0
    fi
  fi

  printf '[ArcSub] Node.js 22+ not found. Run ./deploy.sh first.\n' >&2
  exit 1
}

print_release_summary() {
  local manifest_path="$script_dir/release-manifest.json"
  if [[ ! -f "$manifest_path" ]]; then
    return
  fi

  local release_line
  release_line="$("$node_bin" -e 'const fs=require("fs");const path=process.argv[1];try{const m=JSON.parse(fs.readFileSync(path,"utf8"));const parts=[];if(m.version)parts.push(`version=${m.version}`);if(m.target)parts.push(`target=${m.target}`);if(m.buildId)parts.push(`buildId=${m.buildId}`);if(m.gitHead)parts.push(`git=${m.gitHead}`);if(parts.length)console.log(parts.join(" "));}catch{}' "$manifest_path" 2>/dev/null || true)"
  if [[ -n "$release_line" ]]; then
    printf '[ArcSub] release=%s\n' "$release_line"
  fi
}

open_browser_after_ready() {
  local url="$1"
  local readiness_url="$url/api/runtime/readiness"
  if [[ "$no_browser" -eq 1 ]]; then
    return
  fi

  (
    for _ in $(seq 1 60); do
      if curl -fsS --max-time 5 "$readiness_url" >/dev/null 2>&1; then
        if command -v xdg-open >/dev/null 2>&1; then
          nohup xdg-open "$url" >/dev/null 2>&1 &
        elif command -v gio >/dev/null 2>&1; then
          nohup gio open "$url" >/dev/null 2>&1 &
        fi
        exit 0
      fi
      sleep 2
    done
    exit 0
  ) &
}

if [[ ! -f "$build_entry" ]]; then
  printf '[ArcSub] Missing production server entry: %s\n' "$build_entry" >&2
  printf '[ArcSub] Run ./deploy.sh first.\n' >&2
  exit 1
fi

export NODE_ENV="${NODE_ENV:-production}"
export HOST="$listen_host"
export PORT="$listen_port"
export OPENVINO_TRANSLATE_HELPER_LOAD_TIMEOUT_MS="${OPENVINO_TRANSLATE_HELPER_LOAD_TIMEOUT_MS:-900000}"
export OPENVINO_TRANSLATE_HELPER_HEAVY_LOAD_TIMEOUT_MS="${OPENVINO_TRANSLATE_HELPER_HEAVY_LOAD_TIMEOUT_MS:-900000}"
cd "$script_dir"
refresh_runtime_env
node_bin="$(resolve_node_bin)"
printf '[ArcSub] Starting production server...\n'
printf '[ArcSub] node=%s\n' "$node_bin"
printf '[ArcSub] url=http://%s:%s\n' "$listen_host" "$listen_port"
print_release_summary
open_browser_after_ready "http://$listen_host:$listen_port"
exec "$node_bin" "$build_entry"
