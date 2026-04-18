#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$script_dir"
target_dir="$(pwd)"
clear_asr_cache=0
dry_run=0

log() {
  printf '[ArcSub] %s\n' "$1"
}

usage() {
  cat <<'EOF'
Usage: bash ./update-release.sh [--source-dir PATH] [--target-dir PATH] [--clear-asr-cache] [--dry-run]

Copies updated release files into an existing ArcSub installation while preserving:
  - .arcsub-bootstrap/
  - node_modules/
  - runtime/
  - .env
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir)
      source_dir="$2"
      shift 2
      ;;
    --target-dir)
      target_dir="$2"
      shift 2
      ;;
    --clear-asr-cache)
      clear_asr_cache=1
      shift
      ;;
    --dry-run)
      dry_run=1
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

source_dir="$(cd "$source_dir" && pwd)"
target_dir="$(cd "$target_dir" && pwd)"

if [[ ! -f "$source_dir/release-manifest.json" ]]; then
  printf 'release-manifest.json is missing in source directory: %s\n' "$source_dir" >&2
  exit 1
fi

if [[ "$source_dir" == "$target_dir" ]]; then
  printf 'Source and target directories must be different for incremental update.\n' >&2
  exit 1
fi

copy_entries=(
  build
  dist
  public
  server/glossaries
  tools_src/openvino_asr_env.py
  tools_src/openvino_genai_translate_helper.mjs
  tools_src/openvino_translate_helper.py
  tools_src/openvino_whisper_helper.py
  tools_src/convert_hf_model_to_openvino.py
  tools_src/convert_official_qwen3_asr.py
  tools_src/qwen3_asr_official_support.py
  tools_src/qwen_asr_runtime.py
  tools_src/prepare_pyannote_vbx.py
  tools_src/export_pyannote.py
  .env.example
  README.md
  collect-diagnostics.sh
  deploy.ps1
  deploy.sh
  start.production.ps1
  start.production.sh
  update-release.ps1
  update-release.sh
  install-linux-system-deps.sh
  install-linux-release-deps.sh
  scripts/preflight-linux-runtime.sh
  scripts/install-deployment-assets.mjs
  scripts/install-pyannote-assets.mjs
  scripts/finalize-runtime-install.mjs
  scripts/runtime-smoke.mjs
  scripts/deploy-manifest.json
  package.json
  release-manifest.json
)

copy_entry() {
  local relative_path="$1"
  local source_path="$source_dir/$relative_path"
  local target_path="$target_dir/$relative_path"

  [[ -e "$source_path" ]] || return 0

  if [[ "$dry_run" -eq 1 ]]; then
    log "[dry-run] copy $relative_path"
    return 0
  fi

  mkdir -p "$(dirname "$target_path")"
  rm -rf "$target_path"
  cp -a "$source_path" "$target_path"
}

log "Incremental update source: $source_dir"
log "Incremental update target: $target_dir"
log "Preserving .arcsub-bootstrap, node_modules, runtime, and .env"

package_changed=0
if [[ -f "$source_dir/package.json" && -f "$target_dir/package.json" ]]; then
  source_hash="$(sha256sum "$source_dir/package.json" | awk '{print $1}')"
  target_hash="$(sha256sum "$target_dir/package.json" | awk '{print $1}')"
  [[ "$source_hash" == "$target_hash" ]] || package_changed=1
fi

for entry in "${copy_entries[@]}"; do
  copy_entry "$entry"
done

if [[ "$clear_asr_cache" -eq 1 ]]; then
  asr_cache_dir="$target_dir/runtime/models/openvino-cache/asr"
  if [[ -d "$asr_cache_dir" ]]; then
    if [[ "$dry_run" -eq 1 ]]; then
      log "[dry-run] remove $asr_cache_dir"
    else
      rm -rf "$asr_cache_dir"
      log "Cleared ASR cache: $asr_cache_dir"
    fi
  fi
fi

if [[ "$package_changed" -eq 1 ]]; then
  printf '[ArcSub] WARNING: package.json changed. Existing node_modules were preserved. Run bash ./deploy.sh in the target directory if runtime dependencies need refreshing.\n' >&2
fi

if [[ "$dry_run" -eq 1 ]]; then
  log "Dry run complete. No files were modified."
else
  log "Incremental update complete. Restart ArcSub to load the new build."
fi
