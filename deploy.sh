#!/usr/bin/env bash
set -euo pipefail

required_node_major="${ARCSUB_NODE_MAJOR:-22}"
required_python_major=3
required_python_minor=12
skip_build=0
skip_pyannote=0
preinstall_local_model_python=0
skip_local_model_python=0

log() {
  printf '[ArcSub] %s\n' "$1"
}

fail() {
  printf '[ArcSub] ERROR: %s\n' "$1" >&2
  exit 1
}

dotenv_get() {
  local file_path="$1"
  local key="$2"
  [[ -f "$file_path" ]] || return 0
  grep -E "^${key}=" "$file_path" | head -n 1 | sed -E "s/^${key}=//"
}

dotenv_set() {
  local file_path="$1"
  local key="$2"
  local value="$3"
  mkdir -p "$(dirname "$file_path")"
  if [[ -f "$file_path" ]] && grep -qE "^${key}=" "$file_path"; then
    python3 - "$file_path" "$key" "$value" <<'PY'
import pathlib
import sys

file_path = pathlib.Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
lines = file_path.read_text(encoding="utf-8").splitlines()
updated = False
for index, line in enumerate(lines):
    if line.startswith(f"{key}="):
        lines[index] = f"{key}={value}"
        updated = True
        break
if not updated:
    if lines and lines[-1] != "":
        lines.append("")
    lines.append(f"{key}={value}")
file_path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
PY
    return 0
  fi

  if [[ -f "$file_path" && -s "$file_path" ]]; then
    printf '\n' >>"$file_path"
  fi
  printf '%s=%s\n' "$key" "$value" >>"$file_path"
}

pyannote_ready() {
  [[ -f "$script_dir/runtime/models/pyannote/segmentation/model.xml" ]] &&
  [[ -f "$script_dir/runtime/models/pyannote/embedding/model.xml" ]] &&
  [[ -f "$script_dir/runtime/models/pyannote/plda/vbx.json" ]]
}

ensure_pyannote() {
  [[ "$skip_pyannote" -eq 0 ]] || return 0
  pyannote_ready && {
    log "Pyannote assets already installed."
    return 0
  }

  local env_path="$script_dir/.env"
  local token="${HF_TOKEN:-}"
  if [[ -z "$token" ]]; then
    token="$(dotenv_get "$env_path" "HF_TOKEN")"
  fi

  while ! pyannote_ready; do
    if [[ -n "$token" ]]; then
      log "Installing pyannote assets..."
      HF_TOKEN="$token" run_cmd "$script_dir" "$node_bin" scripts/install-pyannote-assets.mjs --readiness-out runtime/deploy/asset-readiness.json && return 0
      printf '[ArcSub] WARNING: pyannote installation failed with the current HF token.\n' >&2
    fi

    if [[ ! -t 0 ]]; then
      log "Skipping pyannote installation because no interactive terminal is available."
      return 0
    fi

    printf '[ArcSub] HF_TOKEN is optional and is used for Pyannote assets plus Hugging Face models that require approval, login, or private access.\n'
    printf '[ArcSub] Accept the pyannote model access on Hugging Face before installing Pyannote assets. The same token is saved for later local-model downloads.\n'
    read -r -s -p 'HF_TOKEN (leave blank to skip Pyannote asset install for now): ' token_input
    printf '\n'
    token_input="${token_input#"${token_input%%[![:space:]]*}"}"
    token_input="${token_input%"${token_input##*[![:space:]]}"}"
    if [[ -z "$token_input" ]]; then
      log "Skipping pyannote installation."
      return 0
    fi

    token="$token_input"
    dotenv_set "$env_path" "HF_TOKEN" "$token"
    export HF_TOKEN="$token"
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build)
      skip_build=1
      ;;
    --skip-pyannote)
      skip_pyannote=1
      ;;
    --preinstall-local-model-python)
      preinstall_local_model_python=1
      ;;
    --skip-local-model-python)
      skip_local_model_python=1
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
  shift
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
portable_node_dir="$script_dir/.arcsub-bootstrap/node/linux-x64"
portable_node_bin="$portable_node_dir/bin/node"
portable_npm_cli="$portable_node_dir/lib/node_modules/npm/bin/npm-cli.js"
portable_python_dir="$script_dir/.arcsub-bootstrap/python/linux-x64"
portable_python_bin="$portable_python_dir/bin/python3"
resolved_python_bin=""
resolved_node_bin=""
resolved_env_path=""

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

python_version_of() {
  local python_bin="$1"
  "$python_bin" - <<'PY'
import sys
print(f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}")
PY
}

python_meets_minimum() {
  local python_bin="$1"
  local version
  version="$(python_version_of "$python_bin" 2>/dev/null || true)"
  [[ -n "$version" ]] || return 1
  local major="${version%%.*}"
  local rest="${version#*.}"
  local minor="${rest%%.*}"
  [[ "$major" -gt "$required_python_major" ]] && return 0
  [[ "$major" -eq "$required_python_major" && "$minor" -ge "$required_python_minor" ]] && return 0
  return 1
}

node_major_of() {
  local node_bin="$1"
  "$node_bin" -p 'process.versions.node.split(`.`)[0]' 2>/dev/null || echo 0
}

ensure_python() {
  if [[ -x "$portable_python_bin" ]] && python_meets_minimum "$portable_python_bin"; then
    resolved_python_bin="$portable_python_bin"
    return 0
  fi

  local configured_python="${OPENVINO_HELPER_PYTHON:-}"
  local base_python=""
  if [[ -n "$configured_python" ]] && python_meets_minimum "$configured_python"; then
    base_python="$configured_python"
  fi

  if [[ -z "$base_python" ]] && command -v python3 >/dev/null 2>&1; then
    local system_python
    system_python="$(command -v python3)"
    if python_meets_minimum "$system_python"; then
      base_python="$system_python"
    fi
  fi

  if [[ -z "$base_python" && -x "$script_dir/install-linux-system-deps.sh" ]]; then
    printf '[ArcSub] Installing Linux system packages...\n' >&2
    bash "$script_dir/install-linux-system-deps.sh" >&2
  fi

  if [[ -z "$base_python" ]]; then
    command -v python3 >/dev/null 2>&1 || fail "python3 is required. Run ./install-linux-system-deps.sh first."
    local refreshed_python
    refreshed_python="$(command -v python3)"
    python_meets_minimum "$refreshed_python" || fail "python3 3.12+ is required."
    base_python="$refreshed_python"
  fi

  printf '[ArcSub] Creating app-local Python virtual environment...\n' >&2
  rm -rf "$portable_python_dir"
  mkdir -p "$(dirname "$portable_python_dir")"
  run_cmd "$script_dir" "$base_python" -m venv "$portable_python_dir" >&2
  [[ -x "$portable_python_bin" ]] || fail "Python virtual environment bootstrap failed."
  run_cmd "$script_dir" "$portable_python_bin" -m pip install --upgrade pip setuptools wheel >&2
  resolved_python_bin="$portable_python_bin"
}

ensure_asr_helper_python_dependencies() {
  log "Ensuring ASR helper Python dependencies..."
  local probe_output
  probe_output="$("$python_bin" - <<'PY'
import importlib.util

def missing(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except ModuleNotFoundError:
        return True

modules = [
    "openvino",
    "openvino_genai",
    "transformers",
    "huggingface_hub",
    "librosa",
    "accelerate",
    "optimum.intel",
]
missing_names = [name for name in modules if missing(name)]
print("\n".join(missing_names))
PY
)"

  mapfile -t missing_modules < <(printf '%s\n' "$probe_output" | sed '/^\s*$/d')
  if [[ "${#missing_modules[@]}" -eq 0 ]]; then
    log "ASR helper Python dependencies already available."
    return 0
  fi

  local packages=()
  local add_package
  add_package() {
    local pkg="$1"
    for existing in "${packages[@]:-}"; do
      [[ "$existing" == "$pkg" ]] && return 0
    done
    packages+=("$pkg")
  }

  local name
  for name in "${missing_modules[@]}"; do
    case "$name" in
      openvino) add_package "openvino" ;;
      openvino_genai) add_package "openvino-genai" ;;
      transformers) add_package "transformers" ;;
      huggingface_hub) add_package "huggingface_hub" ;;
      librosa) add_package "librosa" ;;
      accelerate) add_package "accelerate" ;;
      optimum.intel) add_package "optimum-intel[openvino]" ;;
    esac
  done

  [[ "${#packages[@]}" -gt 0 ]] || return 0
  run_cmd "$script_dir" "$python_bin" -m pip install --upgrade "${packages[@]}"
}

ensure_whisper_asr_helper_python_dependencies() {
  log "Ensuring Whisper ASR helper Python dependencies..."
  local probe_output
  probe_output="$("$python_bin" - <<'PY'
import importlib.util

def missing(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except ModuleNotFoundError:
        return True

modules = [
    "openvino",
    "openvino_genai",
    "transformers",
    "librosa",
    "numpy",
]
missing_names = [name for name in modules if missing(name)]
print("\n".join(missing_names))
PY
)"

  mapfile -t missing_modules < <(printf '%s\n' "$probe_output" | sed '/^\s*$/d')
  if [[ "${#missing_modules[@]}" -eq 0 ]]; then
    log "Whisper ASR helper Python dependencies already available."
    return 0
  fi

  local packages=()
  local add_package
  add_package() {
    local pkg="$1"
    for existing in "${packages[@]:-}"; do
      [[ "$existing" == "$pkg" ]] && return 0
    done
    packages+=("$pkg")
  }

  local name
  for name in "${missing_modules[@]}"; do
    case "$name" in
      openvino) add_package "openvino" ;;
      openvino_genai) add_package "openvino-genai" ;;
      transformers) add_package "transformers" ;;
      librosa) add_package "librosa" ;;
      numpy) add_package "numpy" ;;
    esac
  done

  [[ "${#packages[@]}" -gt 0 ]] || return 0
  run_cmd "$script_dir" "$python_bin" -m pip install --upgrade "${packages[@]}"
}

ensure_alignment_python_dependencies() {
  log "Ensuring alignment conversion Python dependencies..."
  local probe_output
  probe_output="$("$python_bin" - <<'PY'
import importlib.util

def missing(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except ModuleNotFoundError:
        return True

modules = [
    "openvino",
    "nncf",
    "huggingface_hub",
    "openvino_tokenizers",
    "transformers",
    "optimum.intel",
    "torch",
    "sentencepiece",
]
missing_names = [name for name in modules if missing(name)]
print("\n".join(missing_names))
PY
)"

  mapfile -t missing_modules < <(printf '%s\n' "$probe_output" | sed '/^\s*$/d')
  if [[ "${#missing_modules[@]}" -eq 0 ]]; then
    log "Alignment conversion Python dependencies already available."
    return 0
  fi

  local packages=()
  local add_package
  add_package() {
    local pkg="$1"
    for existing in "${packages[@]:-}"; do
      [[ "$existing" == "$pkg" ]] && return 0
    done
    packages+=("$pkg")
  }

  local name
  for name in "${missing_modules[@]}"; do
    case "$name" in
      openvino) add_package "openvino" ;;
      nncf) add_package "nncf" ;;
      huggingface_hub) add_package "huggingface_hub" ;;
      openvino_tokenizers) add_package "openvino-tokenizers" ;;
      transformers) add_package "transformers" ;;
      optimum.intel) add_package "optimum-intel[openvino]" ;;
      torch) add_package "torch" ;;
      sentencepiece) add_package "sentencepiece" ;;
    esac
  done

  [[ "${#packages[@]}" -gt 0 ]] || return 0
  run_cmd "$script_dir" "$python_bin" -m pip install --upgrade "${packages[@]}"
}

ensure_portable_node() {
  if [[ -x "$portable_node_bin" && "$(node_major_of "$portable_node_bin")" -ge "$required_node_major" ]]; then
    resolved_node_bin="$portable_node_bin"
    return 0
  fi

  command -v curl >/dev/null 2>&1 || fail "curl is required to bootstrap portable Node.js."
  command -v python3 >/dev/null 2>&1 || fail "python3 is required to bootstrap portable Node.js."
  command -v tar >/dev/null 2>&1 || fail "tar is required to bootstrap portable Node.js."

  printf '[ArcSub] Installing Linux system packages...\n' >&2
  if [[ -x "$script_dir/install-linux-system-deps.sh" ]]; then
    bash "$script_dir/install-linux-system-deps.sh" --skip-node --skip-system-packages >&2
  fi

  local version
  version="$(python3 - <<'PY'
import json
import urllib.request

with urllib.request.urlopen("https://nodejs.org/dist/index.json", timeout=60) as response:
    data = json.load(response)

for item in data:
    version = str(item.get("version", ""))
    if version.startswith("v22."):
        print(version)
        break
else:
    raise SystemExit("Unable to resolve latest Node.js v22 release.")
PY
)"

  local archive_name="node-${version}-linux-x64.tar.xz"
  local download_url="https://nodejs.org/dist/${version}/${archive_name}"
  local bootstrap_dir="$script_dir/.arcsub-bootstrap"
  local download_dir="$bootstrap_dir/downloads"
  local extract_dir="$bootstrap_dir/extract"
  mkdir -p "$download_dir"
  rm -rf "$extract_dir" "$portable_node_dir"
  mkdir -p "$(dirname "$portable_node_dir")"

  curl -fsSL "$download_url" -o "$download_dir/$archive_name"
  mkdir -p "$extract_dir"
  tar -xJf "$download_dir/$archive_name" -C "$extract_dir"
  local extracted_node_dir="$extract_dir/node-${version}-linux-x64"
  # On some WSL + drvfs mounts, renaming extracted directories can fail with EPERM.
  # Fall back to copy+remove so release packaging can still complete.
  if ! mv "$extracted_node_dir" "$portable_node_dir"; then
    cp -a "$extracted_node_dir" "$portable_node_dir"
    rm -rf "$extracted_node_dir"
  fi
  rm -rf "$extract_dir"

  [[ -x "$portable_node_bin" ]] || fail "Portable Node.js bootstrap failed."
  resolved_node_bin="$portable_node_bin"
}

ensure_linux_system_deps() {
  local installer="$script_dir/install-linux-system-deps.sh"
  [[ -x "$installer" ]] || return 0

  local need_system_packages=0
  local need_bundled_ytdlp=0
  local bundled_ytdlp_path="$script_dir/runtime/tools/yt-dlp"

  for tool in python3 curl ffmpeg espeak-ng; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      need_system_packages=1
      break
    fi
  done

  if [[ ! -x "$bundled_ytdlp_path" ]]; then
    need_bundled_ytdlp=1
  fi

  if [[ "$need_system_packages" -eq 0 && "$need_bundled_ytdlp" -eq 0 ]]; then
    return 0
  fi

  local args=()
  if [[ "$need_system_packages" -eq 0 ]]; then
    args+=(--skip-system-packages)
  fi
  args+=(--skip-node)

  printf '[ArcSub] Ensuring Linux system dependencies...\n' >&2
  bash "$installer" "${args[@]}" >&2
}

run_cmd() {
  local workdir="$1"
  shift
  (
    cd "$workdir"
    "$@"
  )
}

run_npm_cmd() {
  local workdir="$1"
  shift
  if [[ "$node_bin" == "$portable_node_bin" ]]; then
    [[ -f "$portable_npm_cli" ]] || fail "Portable npm-cli.js not found."
    local node_dir
    node_dir="$(dirname "$portable_node_bin")"
    (
      cd "$workdir"
      PATH="$node_dir:$PATH" \
      npm_node_execpath="$portable_node_bin" \
      npm_execpath="$portable_npm_cli" \
      "$node_bin" "$portable_npm_cli" "$@"
    )
    return 0
  fi

  command -v npm >/dev/null 2>&1 || fail "npm not found for Node.js runtime."
  run_cmd "$workdir" "$(command -v npm)" "$@"
}

verify_node_native_runtimes() {
  [[ -d "$script_dir/node_modules" ]] || return 1
  (
    cd "$script_dir"
    "$node_bin" --input-type=module - <<'NODE'
await import('onnxruntime-node');
await import('@huggingface/transformers');
NODE
  ) >/dev/null 2>&1
}

install_release_runtime_dependencies() {
  log "Installing release runtime dependencies..."
  run_npm_cmd "$script_dir" install --omit=dev --no-fund --no-audit --ignore-scripts
  run_cmd "$script_dir" "$node_bin" scripts/finalize-runtime-install.mjs
}

ensure_dotenv_file() {
  local env_path="$script_dir/.env"
  if [[ ! -f "$env_path" ]]; then
    if [[ -f "$script_dir/.env.example" ]]; then
      cp "$script_dir/.env.example" "$env_path"
    else
      : > "$env_path"
    fi
  fi

  local current_key
  current_key="$(dotenv_get "$env_path" "ENCRYPTION_KEY")"
  if [[ -z "$current_key" || "$current_key" == "replace_with_random_64_hex" ]]; then
    local generated_key
    generated_key="$("${python_bin:-python3}" - <<'PY'
import secrets

print(secrets.token_hex(32))
PY
)"
    dotenv_set "$env_path" "ENCRYPTION_KEY" "$generated_key"
    log "Generated a fresh ENCRYPTION_KEY in .env"
  elif [[ ! "$current_key" =~ ^[0-9a-fA-F]{64}$ ]]; then
    log "Keeping existing custom ENCRYPTION_KEY in .env"
  fi

  resolved_env_path="$env_path"
}

log "Preparing production deployment..."
log "Workspace: $script_dir"

ensure_linux_system_deps
ensure_python
python_bin="$resolved_python_bin"
export OPENVINO_HELPER_PYTHON="$python_bin"
ensure_dotenv_file
env_path="$resolved_env_path"
dotenv_set "$env_path" "OPENVINO_HELPER_PYTHON" "$python_bin"
if [[ "$skip_local_model_python" -eq 0 ]]; then
  ensure_whisper_asr_helper_python_dependencies
else
  log "Skipping local ASR helper Python dependency preinstall."
fi
if [[ "$preinstall_local_model_python" -eq 1 && "$skip_local_model_python" -eq 0 ]]; then
  ensure_asr_helper_python_dependencies
  ensure_alignment_python_dependencies
fi

ensure_portable_node
node_bin="$resolved_node_bin"
is_source_workspace=0
[[ -d "$script_dir/src" && -f "$script_dir/server/index.ts" ]] && is_source_workspace=1
has_workspace_tooling=0
[[ -x "$script_dir/node_modules/.bin/tsc" && -x "$script_dir/node_modules/.bin/vite" ]] && has_workspace_tooling=1

log "node=$node_bin"
log "python=$python_bin"
if [[ "$node_bin" == "$portable_node_bin" ]]; then
  log "npm=$portable_npm_cli"
else
  log "npm=$(command -v npm)"
fi

if [[ "$is_source_workspace" -eq 1 ]]; then
  if [[ ! -d "$script_dir/node_modules" || "$has_workspace_tooling" -ne 1 ]]; then
    log "Installing workspace dependencies..."
    run_npm_cmd "$script_dir" install
  fi
  if [[ "$skip_build" -ne 1 ]]; then
    log "Building production assets..."
    run_npm_cmd "$script_dir" run -s build:prod
  fi
else
  if [[ ! -d "$script_dir/node_modules" ]]; then
    install_release_runtime_dependencies
  elif ! verify_node_native_runtimes; then
    log "Runtime dependencies failed native preload; reinstalling release runtime dependencies..."
    rm -rf "$script_dir/node_modules" "$script_dir/package-lock.json"
    install_release_runtime_dependencies
  fi
fi

refresh_runtime_env

asset_args=(scripts/install-deployment-assets.mjs --readiness-out runtime/deploy/asset-readiness.json --skip-pyannote)
log "Installing deployment assets..."
run_cmd "$script_dir" "$node_bin" "${asset_args[@]}"

ensure_pyannote

log "Deployment completed."
log "Readiness: runtime/deploy/asset-readiness.json"
