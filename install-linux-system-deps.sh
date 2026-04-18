#!/usr/bin/env bash
set -euo pipefail

NODE_MAJOR="${NODE_MAJOR:-22}"
NODE_INSTALL_ROOT="${NODE_INSTALL_ROOT:-/usr/local/lib/nodejs}"
PACKAGE_MANAGER="${PACKAGE_MANAGER:-}"
dry_run=0
skip_node=0
skip_system_packages=0
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bundled_tools_dir="${YT_DLP_INSTALL_DIR:-$script_dir/runtime/tools}"
bundled_ytdlp_path="${YT_DLP_INSTALL_PATH:-$bundled_tools_dir/yt-dlp}"

resolve_ytdlp_download_url() {
  if [[ -n "${YT_DLP_DOWNLOAD_URL:-}" ]]; then
    printf '%s\n' "$YT_DLP_DOWNLOAD_URL"
    return 0
  fi

  local arch
  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64)
      printf '%s\n' "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux"
      ;;
    aarch64|arm64)
      printf '%s\n' "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux_aarch64"
      ;;
    *)
      printf '%s\n' "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp"
      ;;
  esac
}

log() {
  printf '[ArcSub] %s\n' "$1"
}

warn() {
  printf '[ArcSub] WARNING: %s\n' "$1" >&2
}

fail() {
  printf '[ArcSub] ERROR: %s\n' "$1" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=1
      ;;
    --skip-node)
      skip_node=1
      ;;
    --skip-system-packages)
      skip_system_packages=1
      ;;
    --package-manager)
      shift
      [[ $# -gt 0 ]] || fail "--package-manager requires a value"
      PACKAGE_MANAGER="$1"
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
  shift
done

if [[ ! -f /etc/os-release ]]; then
  fail "This script only supports Linux environments with /etc/os-release."
fi

# shellcheck disable=SC1091
source /etc/os-release

if [[ -z "$PACKAGE_MANAGER" ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    PACKAGE_MANAGER="apt-get"
  elif command -v dnf >/dev/null 2>&1; then
    PACKAGE_MANAGER="dnf"
  elif command -v yum >/dev/null 2>&1; then
    PACKAGE_MANAGER="yum"
  else
    fail "Unsupported Linux distribution. Supported package managers: apt-get, dnf, yum."
  fi
fi

SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    fail "sudo is required when not running as root."
  fi
fi

run_cmd() {
  if [[ "$dry_run" -eq 1 ]]; then
    printf '[dry-run] %s\n' "$*"
    return 0
  fi
  "$@"
}

run_with_sudo() {
  if [[ -n "$SUDO" ]]; then
    run_cmd "$SUDO" "$@"
  else
    run_cmd "$@"
  fi
}

install_linux_node() {
  require_command curl
  require_command python3
  require_command tar

  local native_node_path
  native_node_path="$(command -v node 2>/dev/null || true)"
  if [[ -n "$native_node_path" && "$native_node_path" != /mnt/* && "$native_node_path" != *.exe && "$native_node_path" != *.cmd ]]; then
    local node_major
    node_major="$(node -p 'process.versions.node.split(`.`)[0]' 2>/dev/null || echo 0)"
    if [[ "$node_major" =~ ^[0-9]+$ && "$node_major" -ge "$NODE_MAJOR" ]]; then
      log "Native Linux Node.js already available at $native_node_path"
      return
    fi
  fi

  log "Installing native Linux Node.js v$NODE_MAJOR from nodejs.org..."
  local node_version
  node_version="$(python3 - "$NODE_MAJOR" <<'PY'
import json
import sys
import urllib.request

major = sys.argv[1]
with urllib.request.urlopen("https://nodejs.org/dist/index.json", timeout=30) as response:
    data = json.load(response)

for item in data:
    version = str(item.get("version", ""))
    if version.startswith(f"v{major}."):
        print(version)
        break
else:
    raise SystemExit(f"Unable to find latest Node.js v{major} release.")
PY
)"
  local archive_name="node-${node_version}-linux-x64.tar.xz"
  local download_url="https://nodejs.org/dist/${node_version}/${archive_name}"
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' EXIT

  run_cmd curl -fsSL "$download_url" -o "$tmp_dir/$archive_name"
  run_with_sudo mkdir -p "$NODE_INSTALL_ROOT"
  run_with_sudo tar -xJf "$tmp_dir/$archive_name" -C "$NODE_INSTALL_ROOT"

  local install_dir="$NODE_INSTALL_ROOT/node-${node_version}-linux-x64"
  run_with_sudo ln -sfn "$install_dir/bin/node" /usr/local/bin/node
  run_with_sudo ln -sfn "$install_dir/bin/npm" /usr/local/bin/npm
  run_with_sudo ln -sfn "$install_dir/bin/npx" /usr/local/bin/npx

  rm -rf "$tmp_dir"
  trap - EXIT
}

install_official_ytdlp_binary() {
  require_command curl
  local download_url
  download_url="$(resolve_ytdlp_download_url)"
  log "Installing official yt-dlp release binary..."
  run_cmd mkdir -p "$bundled_tools_dir"
  run_cmd curl -fsSL "$download_url" -o "$bundled_ytdlp_path"
  run_cmd chmod 755 "$bundled_ytdlp_path"
}

install_system_packages() {
  case "$PACKAGE_MANAGER" in
    apt-get)
      log "Installing Linux system packages with apt-get..."
      run_with_sudo apt-get update
      run_with_sudo apt-get install -y \
        ca-certificates \
        curl \
        xz-utils \
        ffmpeg \
        espeak-ng \
        python3 \
        python3-venv
      ;;
    dnf)
      log "Installing Linux system packages with dnf..."
      run_with_sudo dnf install -y \
        ca-certificates \
        curl \
        xz \
        ffmpeg \
        espeak-ng \
        python3
      ;;
    yum)
      log "Installing Linux system packages with yum..."
      run_with_sudo yum install -y \
        ca-certificates \
        curl \
        xz \
        ffmpeg \
        espeak-ng \
        python3
      ;;
    *)
      fail "Unsupported package manager: $PACKAGE_MANAGER"
      ;;
  esac
}

if [[ "$skip_system_packages" -ne 1 ]]; then
  install_system_packages
fi

if [[ "$skip_node" -ne 1 ]]; then
  install_linux_node
fi

install_official_ytdlp_binary

for tool in node npm python3 curl ffmpeg espeak-ng; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf '%s=%s\n' "$tool" "$(command -v "$tool")"
  else
    warn "$tool is still missing after install attempt."
  fi
done

if [[ -x "$bundled_ytdlp_path" ]]; then
  printf 'yt-dlp=%s\n' "$bundled_ytdlp_path"
elif command -v yt-dlp >/dev/null 2>&1; then
  printf 'yt-dlp=%s\n' "$(command -v yt-dlp)"
else
  warn "yt-dlp is still missing after install attempt."
fi

if [[ "$PACKAGE_MANAGER" == "dnf" || "$PACKAGE_MANAGER" == "yum" ]]; then
  warn "On Red Hat family distributions, ffmpeg may require extra repositories such as EPEL or RPM Fusion."
fi

log "Linux system dependency setup completed."
