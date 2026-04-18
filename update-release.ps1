param(
  [string]$SourceDir,
  [string]$TargetDir,
  [switch]$ClearAsrCache,
  [switch]$DryRun
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

function Copy-ReleaseEntry([string]$ResolvedSourceRoot, [string]$ResolvedTargetRoot, [string]$RelativePath, [switch]$PreviewOnly) {
  $sourcePath = Join-Path $ResolvedSourceRoot $RelativePath
  if (-not (Test-Path -LiteralPath $sourcePath)) {
    return
  }

  $targetPath = Join-Path $ResolvedTargetRoot $RelativePath
  $targetParent = Split-Path -Parent $targetPath
  if (-not [string]::IsNullOrWhiteSpace($targetParent) -and -not (Test-Path -LiteralPath $targetParent)) {
    if ($PreviewOnly) {
      Write-ArcSub "[dry-run] mkdir $targetParent"
    } else {
      New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
    }
  }

  if ($PreviewOnly) {
    Write-ArcSub "[dry-run] copy $RelativePath"
    return
  }

  Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Recurse -Force
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$resolvedSourceDir = Resolve-AbsolutePath $(if ($SourceDir) { $SourceDir } else { $scriptRoot })
$resolvedTargetDir = Resolve-AbsolutePath $(if ($TargetDir) { $TargetDir } else { (Get-Location).Path })

if (-not (Test-Path -LiteralPath $resolvedSourceDir)) {
  throw "Source release directory does not exist: $resolvedSourceDir"
}

if (-not (Test-Path -LiteralPath $resolvedTargetDir)) {
  throw "Target installation directory does not exist: $resolvedTargetDir"
}

$manifestPath = Join-Path $resolvedSourceDir "release-manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath)) {
  throw "release-manifest.json is missing in source directory: $resolvedSourceDir"
}

if ($resolvedSourceDir.TrimEnd('\') -eq $resolvedTargetDir.TrimEnd('\')) {
  throw "Source and target directories must be different for incremental update."
}

$copyEntries = @(
  "build",
  "dist",
  "public",
  "server/glossaries",
  "tools_src/openvino_asr_env.py",
  "tools_src/openvino_genai_translate_helper.mjs",
  "tools_src/openvino_translate_helper.py",
  "tools_src/openvino_whisper_helper.py",
  "tools_src/convert_hf_model_to_openvino.py",
  "tools_src/convert_official_qwen3_asr.py",
  "tools_src/qwen3_asr_official_support.py",
  "tools_src/qwen_asr_runtime.py",
  "tools_src/prepare_pyannote_vbx.py",
  "tools_src/export_pyannote.py",
  ".env.example",
  "README.md",
  "collect-diagnostics.ps1",
  "deploy.ps1",
  "deploy.sh",
  "start.production.ps1",
  "start.production.sh",
  "update-release.ps1",
  "update-release.sh",
  "install-linux-system-deps.sh",
  "install-linux-release-deps.sh",
  "scripts/preflight-linux-runtime.sh",
  "scripts/install-deployment-assets.mjs",
  "scripts/install-pyannote-assets.mjs",
  "scripts/finalize-runtime-install.mjs",
  "scripts/runtime-smoke.mjs",
  "scripts/deploy-manifest.json",
  "package.json",
  "release-manifest.json"
)

Write-ArcSub "Incremental update source: $resolvedSourceDir"
Write-ArcSub "Incremental update target: $resolvedTargetDir"
Write-ArcSub "Preserving .arcsub-bootstrap, node_modules, runtime, and .env"

$sourcePackageJson = Join-Path $resolvedSourceDir "package.json"
$targetPackageJson = Join-Path $resolvedTargetDir "package.json"
$packageChanged = $false
if ((Test-Path -LiteralPath $sourcePackageJson) -and (Test-Path -LiteralPath $targetPackageJson)) {
  $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourcePackageJson).Hash
  $targetHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $targetPackageJson).Hash
  $packageChanged = $sourceHash -ne $targetHash
}

foreach ($entry in $copyEntries) {
  Copy-ReleaseEntry -ResolvedSourceRoot $resolvedSourceDir -ResolvedTargetRoot $resolvedTargetDir -RelativePath $entry -PreviewOnly:$DryRun
}

if ($ClearAsrCache) {
  $asrCacheDir = Join-Path $resolvedTargetDir "runtime\models\openvino-cache\asr"
  if (Test-Path -LiteralPath $asrCacheDir) {
    if ($DryRun) {
      Write-ArcSub "[dry-run] remove $asrCacheDir"
    } else {
      Remove-Item -LiteralPath $asrCacheDir -Recurse -Force
      Write-ArcSub "Cleared ASR cache: $asrCacheDir"
    }
  }
}

if ($packageChanged) {
  Write-Warning "package.json changed. Existing node_modules were preserved. Run .\deploy.ps1 in the target directory if runtime dependencies need refreshing."
}

if ($DryRun) {
  Write-ArcSub "Dry run complete. No files were modified."
} else {
  Write-ArcSub "Incremental update complete. Restart ArcSub to load the new build."
}
