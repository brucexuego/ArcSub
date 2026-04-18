param(
  [switch]$NoBrowser,
  [switch]$DryRun,
  [int[]]$Ports = @(3000, 24678)
)

$ErrorActionPreference = "Stop"

function Set-ConsoleUtf8() {
  $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
  [Console]::InputEncoding = $utf8NoBom
  [Console]::OutputEncoding = $utf8NoBom
  $global:OutputEncoding = $utf8NoBom
  $env:PYTHONIOENCODING = "utf-8"
  cmd /c chcp 65001 > $null
}

function Write-Step($message) {
  Write-Host "[ArcSub] $message" -ForegroundColor Cyan
}

function Write-Info($message) {
  Write-Host "[ArcSub] $message" -ForegroundColor Gray
}

function Get-DotEnvValue([string]$Key, [string]$Default = "") {
  $envPath = Join-Path $scriptPath ".env"
  if (-not (Test-Path $envPath)) { return $Default }
  $line = Get-Content $envPath -ErrorAction SilentlyContinue |
    Where-Object { $_ -match "^\s*$Key\s*=" } |
    Select-Object -First 1
  if (-not $line) { return $Default }
  $value = ($line -replace "^\s*$Key\s*=\s*", "").Trim()
  return $value
}

function Resolve-LogsDir([string]$WorkspaceRoot) {
  $explicitLogsDir = $env:APP_LOGS_DIR
  if ([string]::IsNullOrWhiteSpace($explicitLogsDir)) {
    $explicitLogsDir = Get-DotEnvValue "APP_LOGS_DIR"
  }
  if (-not [string]::IsNullOrWhiteSpace($explicitLogsDir)) {
    if ([System.IO.Path]::IsPathRooted($explicitLogsDir)) {
      return $explicitLogsDir
    }
    return Join-Path $WorkspaceRoot $explicitLogsDir
  }

  $runtimeDir = $env:APP_RUNTIME_DIR
  if ([string]::IsNullOrWhiteSpace($runtimeDir)) {
    $runtimeDir = Get-DotEnvValue "APP_RUNTIME_DIR" "runtime"
  }
  if ([string]::IsNullOrWhiteSpace($runtimeDir)) { $runtimeDir = "runtime" }
  if ([System.IO.Path]::IsPathRooted($runtimeDir)) {
    return Join-Path $runtimeDir "logs"
  }
  return Join-Path (Join-Path $WorkspaceRoot $runtimeDir) "logs"
}

function Kill-ProcessTree([int]$ProcessId) {
  if ($ProcessId -le 0) { return }
  if ($DryRun) {
    Write-Info "DryRun: would kill PID $ProcessId"
    return
  }
  $existing = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
  if (-not $existing) { return }

  cmd /c "taskkill /F /T /PID $ProcessId >nul 2>&1"
  if ($LASTEXITCODE -ne 0) {
    $stillExists = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($stillExists) {
      throw "Failed to terminate stale process PID $ProcessId (exit code: $LASTEXITCODE)."
    }
  }
}

function Get-PidsByPorts([int[]]$TargetPorts) {
  $pids = New-Object System.Collections.Generic.HashSet[int]
  foreach ($port in $TargetPorts) {
    try {
      $conns = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
      foreach ($conn in $conns) {
        if ($conn.OwningProcess -gt 0) {
          $null = $pids.Add([int]$conn.OwningProcess)
        }
      }
    } catch {
      # Ignore missing port bindings.
    }
  }
  return $pids
}

function Get-WorkspaceNodePids([string]$WorkspaceRoot) {
  $pids = New-Object System.Collections.Generic.HashSet[int]
  $workspacePattern = [regex]::Escape($WorkspaceRoot)
  $processes = Get-CimInstance Win32_Process -Filter "Name='node.exe' OR Name='npm.exe' OR Name='npm.cmd'" -ErrorAction SilentlyContinue

  foreach ($proc in $processes) {
    $cmd = [string]$proc.CommandLine
    if ([string]::IsNullOrWhiteSpace($cmd)) { continue }
    if ($cmd -match $workspacePattern -or $cmd -match "tsx\s+watch\s+server(?:/|\\)index\.ts" -or $cmd -match "tsx\s+watch\s+server\.ts") {
      $null = $pids.Add([int]$proc.ProcessId)
    }
  }

  return $pids
}

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptPath

try {
  Set-ConsoleUtf8
  Write-Step "Preparing development startup..."
  Write-Info "Workspace: $scriptPath"

  $killPids = New-Object System.Collections.Generic.HashSet[int]
  $portPids = Get-PidsByPorts -TargetPorts $Ports
  foreach ($procId in $portPids) { $null = $killPids.Add($procId) }

  $workspacePids = Get-WorkspaceNodePids -WorkspaceRoot $scriptPath
  foreach ($procId in $workspacePids) { $null = $killPids.Add($procId) }

  if ($killPids.Count -gt 0) {
    Write-Step "Stopping stale processes: $($killPids -join ', ')"
    foreach ($procId in $killPids) {
      Kill-ProcessTree -ProcessId $procId
    }
    if (-not $DryRun) {
      Start-Sleep -Milliseconds 800
    }
  } else {
    Write-Step "No stale process found."
  }

  if ($DryRun) {
    Write-Step "DryRun finished. Dev server not started."
    exit 0
  }

  if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:3000" | Out-Null
  }

  $logsDir = Resolve-LogsDir -WorkspaceRoot $scriptPath
  if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
  }

  $logFileName = "dev-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss")
  $logPath = Join-Path $logsDir $logFileName

  Write-Step "Starting dev server (npm run dev)..."
  Write-Info "Dev log: $logPath"
  npm run dev 2>&1 | Tee-Object -FilePath $logPath
  if ($LASTEXITCODE -ne 0) {
    throw "Dev server exited with code $LASTEXITCODE. Check log: $logPath"
  }
}
finally {
  Pop-Location
}
