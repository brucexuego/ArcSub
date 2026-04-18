#!/usr/bin/env bash
set -euo pipefail

no_browser=0
dry_run=0
ports=(3000 24678)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-browser)
      no_browser=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --ports)
      shift
      ports=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        ports+=("$1")
        shift
      done
      ;;
    *)
      echo "[ArcSub] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

write_step() {
  printf '[ArcSub] %s\n' "$1"
}

write_info() {
  printf '[ArcSub] %s\n' "$1"
}

get_dotenv_value() {
  local key="$1"
  local default_value="${2:-}"
  local env_path="$script_dir/.env"
  if [[ ! -f "$env_path" ]]; then
    printf '%s' "$default_value"
    return
  fi

  python3 - "$env_path" "$key" "$default_value" <<'PY'
import re
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
key = sys.argv[2]
default_value = sys.argv[3]
pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*(.*)$")
for line in env_path.read_text(encoding="utf-8").splitlines():
    match = pattern.match(line)
    if match:
        print(match.group(1).strip(), end="")
        break
else:
    print(default_value, end="")
PY
}

resolve_logs_dir() {
  local explicit_logs_dir="${APP_LOGS_DIR:-}"
  if [[ -z "$explicit_logs_dir" ]]; then
    explicit_logs_dir="$(get_dotenv_value "APP_LOGS_DIR")"
  fi
  if [[ -n "$explicit_logs_dir" ]]; then
    if [[ "$explicit_logs_dir" = /* ]]; then
      printf '%s\n' "$explicit_logs_dir"
    else
      printf '%s\n' "$script_dir/$explicit_logs_dir"
    fi
    return
  fi

  local runtime_dir="${APP_RUNTIME_DIR:-}"
  if [[ -z "$runtime_dir" ]]; then
    runtime_dir="$(get_dotenv_value "APP_RUNTIME_DIR" "runtime")"
  fi
  if [[ -z "$runtime_dir" ]]; then
    runtime_dir="runtime"
  fi
  if [[ "$runtime_dir" = /* ]]; then
    printf '%s\n' "$runtime_dir/logs"
  else
    printf '%s\n' "$script_dir/$runtime_dir/logs"
  fi
}

append_unique_pid() {
  local pid="$1"
  [[ -z "$pid" ]] && return
  [[ ! "$pid" =~ ^[0-9]+$ ]] && return
  if [[ -n "${seen_pids[$pid]:-}" ]]; then
    return
  fi
  seen_pids[$pid]=1
  kill_pids+=("$pid")
}

collect_port_pids() {
  local port
  for port in "${ports[@]}"; do
    if command -v lsof >/dev/null 2>&1; then
      while IFS= read -r pid; do
        append_unique_pid "$pid"
      done < <(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
      continue
    fi

    if command -v ss >/dev/null 2>&1; then
      while IFS= read -r pid; do
        append_unique_pid "$pid"
      done < <(
        ss -ltnp 2>/dev/null |
          awk -v port="$port" '
            index($0, ":" port) && match($0, /pid=[0-9]+/) {
              print substr($0, RSTART + 4, RLENGTH - 4)
            }
          ' |
          sort -u || true
      )
    fi
  done
}

collect_workspace_node_pids() {
  if ! command -v pgrep >/dev/null 2>&1; then
    return
  fi

  while IFS= read -r line; do
    local pid="${line%% *}"
    local cmd="${line#* }"
    if [[ "$cmd" == "$pid" ]]; then
      cmd=""
    fi
    if [[ "$cmd" == *"$script_dir"* ]] || [[ "$cmd" =~ tsx[[:space:]]+watch[[:space:]]+server(/|\\)index\.ts ]] || [[ "$cmd" =~ tsx[[:space:]]+watch[[:space:]]+server\.ts ]]; then
      append_unique_pid "$pid"
    fi
  done < <(pgrep -af 'node|npm|tsx' || true)
}

kill_process_tree() {
  local pid="$1"
  if [[ "$dry_run" -eq 1 ]]; then
    write_info "DryRun: would kill PID $pid"
    return
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    return
  fi

  if command -v pgrep >/dev/null 2>&1; then
    while IFS= read -r child; do
      [[ -n "$child" ]] && kill_process_tree "$child"
    done < <(pgrep -P "$pid" || true)
  fi

  kill -TERM "$pid" 2>/dev/null || true
  sleep 0.3
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

open_browser() {
  local url="$1"
  if [[ "$no_browser" -eq 1 ]]; then
    return
  fi
  if command -v xdg-open >/dev/null 2>&1; then
    nohup xdg-open "$url" >/dev/null 2>&1 &
  fi
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
declare -A seen_pids=()
kill_pids=()

write_step "Preparing development startup..."
write_info "Workspace: $script_dir"

collect_port_pids
collect_workspace_node_pids

if [[ ${#kill_pids[@]} -gt 0 ]]; then
  write_step "Stopping stale processes: ${kill_pids[*]}"
  for pid in "${kill_pids[@]}"; do
    kill_process_tree "$pid"
  done
  if [[ "$dry_run" -eq 0 ]]; then
    sleep 0.8
  fi
else
  write_step "No stale process found."
fi

if [[ "$dry_run" -eq 1 ]]; then
  write_step "DryRun finished. Dev server not started."
  exit 0
fi

open_browser "http://127.0.0.1:3000"

logs_dir="$(resolve_logs_dir)"
mkdir -p "$logs_dir"
log_path="$logs_dir/dev-$(date +%Y%m%d-%H%M%S).log"

write_step "Starting dev server (npm run dev)..."
write_info "Dev log: $log_path"

cd "$script_dir"
npm run dev 2>&1 | tee "$log_path"
