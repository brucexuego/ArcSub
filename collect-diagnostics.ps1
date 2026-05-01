param(
  [string]$ProjectId = "",
  [string]$OutputDir = "",
  [switch]$IncludeEnvSnapshot
)

$ErrorActionPreference = "Stop"

function Write-ArcSub([string]$Message) {
  Write-Host "[ArcSub] $Message" -ForegroundColor Cyan
}

function Resolve-AbsolutePath([string]$PathValue) {
  if ([string]::IsNullOrWhiteSpace($PathValue)) {
    return $null
  }

  $expanded = [Environment]::ExpandEnvironmentVariables($PathValue)
  if ([System.IO.Path]::IsPathRooted($expanded)) {
    return [System.IO.Path]::GetFullPath($expanded)
  }

  return [System.IO.Path]::GetFullPath((Join-Path (Get-Location).Path $expanded))
}

function Ensure-Directory([string]$PathValue) {
  if (-not (Test-Path -LiteralPath $PathValue)) {
    New-Item -ItemType Directory -Path $PathValue -Force | Out-Null
  }
}

function Copy-IfExists([string]$SourcePath, [string]$TargetPath) {
  if (-not (Test-Path -LiteralPath $SourcePath)) {
    return $false
  }

  $parent = Split-Path -Parent $TargetPath
  if (-not [string]::IsNullOrWhiteSpace($parent)) {
    Ensure-Directory $parent
  }

  Copy-Item -LiteralPath $SourcePath -Destination $TargetPath -Force
  return $true
}

function Get-NodeVersion([string]$ScriptRoot) {
  $portableNode = Join-Path $ScriptRoot ".arcsub-bootstrap\node\windows-x64\node.exe"
  if (Test-Path -LiteralPath $portableNode) {
    try {
      return (& $portableNode --version 2>$null)
    } catch {
      return $null
    }
  }

  $systemNode = Get-Command node -ErrorAction SilentlyContinue
  if ($systemNode) {
    try {
      return (& $systemNode.Source --version 2>$null)
    } catch {
      return $null
    }
  }

  return $null
}

function Get-LatestProjectId([string]$ProjectsRoot) {
  if (-not (Test-Path -LiteralPath $ProjectsRoot)) {
    return $null
  }

  $candidate = Get-ChildItem -LiteralPath $ProjectsRoot -Directory -ErrorAction SilentlyContinue |
    ForEach-Object {
      $transcriptionPath = Join-Path $_.FullName "assets\transcription.json"
      if (Test-Path -LiteralPath $transcriptionPath) {
        [pscustomobject]@{
          ProjectId = $_.Name
          TranscriptionPath = $transcriptionPath
          LastWriteTime = (Get-Item -LiteralPath $transcriptionPath).LastWriteTime
        }
      }
    } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

  return $candidate?.ProjectId
}

function Get-ProjectDiagnostics([string]$ProjectsRoot, [string]$ResolvedProjectId) {
  if ([string]::IsNullOrWhiteSpace($ResolvedProjectId)) {
    return $null
  }

  $projectRoot = Join-Path $ProjectsRoot $ResolvedProjectId
  $transcriptionPath = Join-Path $projectRoot "assets\transcription.json"
  if (-not (Test-Path -LiteralPath $transcriptionPath)) {
    return $null
  }

  return [pscustomobject]@{
    ProjectId = $ResolvedProjectId
    ProjectRoot = $projectRoot
    TranscriptionPath = $transcriptionPath
  }
}

function Read-EnvSnapshot([string]$EnvPath) {
  $interestingKeys = @(
    'HOST',
    'PORT',
    'OPENVINO_BASELINE_DEVICE',
    'OPENVINO_LOCAL_ASR_DEVICE',
    'OPENVINO_LOCAL_TRANSLATE_DEVICE',
    'OPENVINO_LOCAL_MODEL_PRELOAD_ENABLED',
    'OPENVINO_HELPER_PYTHON',
    'OPENVINO_WHISPER_AUTO_FALLBACK_LANGUAGES',
    'PYANNOTE_DEVICE',
    'TEN_VAD_PROVIDER',
    'OPENVINO_QWEN3_ASR_FORCED_ALIGNER_DEVICE',
    'INTEL_OPENVINO_DIR',
    'OPENVINO_DIR',
    'OPENVINO_INSTALL_DIR',
    'HF_TOKEN',
    'ENCRYPTION_KEY'
  )

  $result = [ordered]@{}
  foreach ($key in $interestingKeys) {
    $result[$key] = $null
  }

  if (-not (Test-Path -LiteralPath $EnvPath)) {
    return $result
  }

  foreach ($line in Get-Content -LiteralPath $EnvPath -ErrorAction SilentlyContinue) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    if ($line.TrimStart().StartsWith('#')) { continue }
    $parts = $line -split '=', 2
    if ($parts.Count -ne 2) { continue }
    $key = $parts[0].Trim()
    if (-not $result.Contains($key)) { continue }
    $value = $parts[1]
    if ($key -in @('HF_TOKEN', 'ENCRYPTION_KEY')) {
      if ([string]::IsNullOrWhiteSpace($value)) {
        $result[$key] = ''
      } else {
        $result[$key] = ('*' * [Math]::Min($value.Length, 12))
      }
    } else {
      $result[$key] = $value
    }
  }

  return $result
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$runtimeRoot = Join-Path $scriptRoot "runtime"
$projectsRoot = Join-Path $runtimeRoot "projects"
$logsRoot = Join-Path $runtimeRoot "logs"
$deployRoot = Join-Path $runtimeRoot "deploy"
$envPath = Join-Path $scriptRoot ".env"
$resolvedOutputDir = if ($OutputDir) { Resolve-AbsolutePath $OutputDir } else { Join-Path $runtimeRoot ("diagnostics\arcsub-diagnostics-" + (Get-Date -Format "yyyyMMdd-HHmmss")) }
Ensure-Directory $resolvedOutputDir

$resolvedProjectId = if ($ProjectId) { $ProjectId.Trim() } else { Get-LatestProjectId $projectsRoot }
$projectDiagnostics = Get-ProjectDiagnostics $projectsRoot $resolvedProjectId

Write-ArcSub "Collecting diagnostics..."
Write-ArcSub "Output: $resolvedOutputDir"
if ($projectDiagnostics) {
  Write-ArcSub "Project: $($projectDiagnostics.ProjectId)"
} else {
  Write-ArcSub "Project: not found"
}

$copied = @()

if ($projectDiagnostics) {
  if (Copy-IfExists $projectDiagnostics.TranscriptionPath (Join-Path $resolvedOutputDir "project\transcription.json")) {
    $copied += 'project/transcription.json'
  }
}

$deployFiles = @(
  'asset-readiness.json'
)
foreach ($fileName in $deployFiles) {
  $source = Join-Path $deployRoot $fileName
  $target = Join-Path $resolvedOutputDir ("deploy\" + $fileName)
  if (Copy-IfExists $source $target) {
    $copied += ("deploy/" + $fileName)
  }
}

$asrLogPath = Join-Path $logsRoot "asr.log"
if (Copy-IfExists $asrLogPath (Join-Path $resolvedOutputDir "logs\asr.log")) {
  $copied += 'logs/asr.log'
}

$consoleLogPath = Join-Path $scriptRoot "console.txt"
if (Copy-IfExists $consoleLogPath (Join-Path $resolvedOutputDir "logs\console.txt")) {
  $copied += 'logs/console.txt'
}

$pathSummary = [ordered]@{
  root = $scriptRoot
  runtime = $runtimeRoot
  projects = $projectsRoot
  logs = $logsRoot
  deploy = $deployRoot
  env = $envPath
}
$pathSummary | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $resolvedOutputDir "paths.json") -Encoding UTF8
$copied += 'paths.json'

$summary = [ordered]@{
  collectedAt = (Get-Date).ToString("o")
  machineName = $env:COMPUTERNAME
  userName = $env:USERNAME
  projectId = $projectDiagnostics?.ProjectId
  nodeVersion = Get-NodeVersion $scriptRoot
  powershellVersion = $PSVersionTable.PSVersion.ToString()
  osVersion = [Environment]::OSVersion.VersionString
  copied = $copied
}
$summary | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $resolvedOutputDir "summary.json") -Encoding UTF8

$envSnapshot = Read-EnvSnapshot $envPath
$envSnapshot | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $resolvedOutputDir "env-snapshot.json") -Encoding UTF8
if ($IncludeEnvSnapshot) {
  $copied += 'env-snapshot.json'
}

$zipPath = "$resolvedOutputDir.zip"
if (Test-Path -LiteralPath $zipPath) {
  Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -Path (Join-Path $resolvedOutputDir '*') -DestinationPath $zipPath -Force

Write-ArcSub "Diagnostics collected."
Write-ArcSub "Zip: $zipPath"
