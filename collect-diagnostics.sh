#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime_root="$script_dir/runtime"
projects_root="$runtime_root/projects"
logs_root="$runtime_root/logs"
deploy_root="$runtime_root/deploy"
env_path="$script_dir/.env"

project_id=""
output_dir=""
include_env_snapshot=0

log() {
  printf '[ArcSub] %s\n' "$1"
}

usage() {
  cat <<'EOF'
Usage: bash ./collect-diagnostics.sh [--project-id ID] [--output-dir PATH] [--include-env-snapshot]

Collects packaged-runtime diagnostics into runtime/diagnostics and produces a .tar.gz archive.
EOF
}

resolve_abs_path() {
  local value="$1"
  if [[ -z "$value" ]]; then
    return 0
  fi
  python3 - "$script_dir" "$value" <<'PY'
import os
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
value = os.path.expandvars(sys.argv[2])
path = pathlib.Path(value)
if not path.is_absolute():
    path = base / path
print(path.resolve())
PY
}

copy_if_exists() {
  local source_path="$1"
  local target_path="$2"
  [[ -e "$source_path" ]] || return 1
  mkdir -p "$(dirname "$target_path")"
  cp -f "$source_path" "$target_path"
}

mask_secret() {
  local value="$1"
  if [[ -z "$value" ]]; then
    printf ''
    return 0
  fi
  local length="${#value}"
  if (( length > 12 )); then
    length=12
  fi
  printf '%*s' "$length" '' | tr ' ' '*'
}

latest_project_id() {
  [[ -d "$projects_root" ]] || return 0
  python3 - "$projects_root" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
candidates = []
for child in root.iterdir():
    if not child.is_dir():
        continue
    transcription = child / "assets" / "transcription.json"
    if transcription.exists():
        candidates.append((transcription.stat().st_mtime, child.name))
candidates.sort(reverse=True)
if candidates:
    print(candidates[0][1])
PY
}

node_version() {
  local portable_node="$script_dir/.arcsub-bootstrap/node/linux-x64/bin/node"
  if [[ -x "$portable_node" ]]; then
    "$portable_node" --version 2>/dev/null || true
    return 0
  fi
  if command -v node >/dev/null 2>&1; then
    node --version 2>/dev/null || true
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-id)
      project_id="${2:-}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:-}"
      shift 2
      ;;
    --include-env-snapshot)
      include_env_snapshot=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$output_dir" ]]; then
  output_dir="$runtime_root/diagnostics/arcsub-diagnostics-$(date +%Y%m%d-%H%M%S)"
else
  output_dir="$(resolve_abs_path "$output_dir")"
fi
mkdir -p "$output_dir"

if [[ -z "$project_id" ]]; then
  project_id="$(latest_project_id)"
fi

log "Collecting diagnostics..."
log "Output: $output_dir"
if [[ -n "$project_id" ]]; then
  log "Project: $project_id"
else
  log "Project: not found"
fi

copied_entries=()

if [[ -n "$project_id" ]]; then
  transcription_path="$projects_root/$project_id/assets/transcription.json"
  if copy_if_exists "$transcription_path" "$output_dir/project/transcription.json"; then
    copied_entries+=("project/transcription.json")
  fi
fi

for file_name in asset-readiness.json; do
  if copy_if_exists "$deploy_root/$file_name" "$output_dir/deploy/$file_name"; then
    copied_entries+=("deploy/$file_name")
  fi
done

if copy_if_exists "$logs_root/asr.log" "$output_dir/logs/asr.log"; then
  copied_entries+=("logs/asr.log")
fi

if copy_if_exists "$script_dir/console.txt" "$output_dir/logs/console.txt"; then
  copied_entries+=("logs/console.txt")
fi

python3 - "$output_dir/paths.json" "$script_dir" "$runtime_root" "$projects_root" "$logs_root" "$deploy_root" "$env_path" <<'PY'
import json
import sys

payload = {
    "root": sys.argv[2],
    "runtime": sys.argv[3],
    "projects": sys.argv[4],
    "logs": sys.argv[5],
    "deploy": sys.argv[6],
    "env": sys.argv[7],
}
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(payload, fh, ensure_ascii=False, indent=2)
PY
copied_entries+=("paths.json")

python3 - "$env_path" "$output_dir/env-snapshot.json" <<'PY'
import json
import pathlib
import sys

interesting_keys = [
    "HOST",
    "PORT",
    "OPENVINO_BASELINE_DEVICE",
    "OPENVINO_LOCAL_ASR_DEVICE",
    "OPENVINO_HELPER_PYTHON",
    "OPENVINO_WHISPER_AUTO_FALLBACK_LANGUAGES",
    "INTEL_OPENVINO_DIR",
    "OPENVINO_DIR",
    "OPENVINO_INSTALL_DIR",
    "HF_TOKEN",
    "ENCRYPTION_KEY",
]

result = {key: None for key in interesting_keys}
env_path = pathlib.Path(sys.argv[1])
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in result:
            continue
        if key in {"HF_TOKEN", "ENCRYPTION_KEY"}:
            result[key] = "*" * min(len(value), 12) if value else ""
        else:
            result[key] = value

with open(sys.argv[2], "w", encoding="utf-8") as fh:
    json.dump(result, fh, ensure_ascii=False, indent=2)
PY

if [[ "$include_env_snapshot" -eq 1 ]]; then
  copied_entries+=("env-snapshot.json")
fi

python3 - "$output_dir/summary.json" "$project_id" "$(node_version)" "$(uname -a)" "${copied_entries[@]}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

summary_path = sys.argv[1]
project_id = sys.argv[2] or None
node_version = sys.argv[3] or None
os_version = sys.argv[4] or None
copied = sys.argv[5:]

payload = {
    "collectedAt": datetime.now(timezone.utc).isoformat(),
    "machineName": os.environ.get("HOSTNAME") or os.uname().nodename,
    "userName": os.environ.get("USER"),
    "projectId": project_id,
    "nodeVersion": node_version,
    "osVersion": os_version,
    "copied": copied,
}
with open(summary_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, ensure_ascii=False, indent=2)
PY

archive_path="${output_dir}.tar.gz"
rm -f "$archive_path"
tar -czf "$archive_path" -C "$(dirname "$output_dir")" "$(basename "$output_dir")"

log "Diagnostics collected."
log "Archive: $archive_path"
