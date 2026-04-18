param(
  [switch]$DryRun,
  [string]$ListenHost = "",
  [int]$Port = 0,
  [switch]$NoBrowser
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

function Write-Step([string]$Message) {
  Write-Host "[ArcSub] $Message" -ForegroundColor Cyan
}

function Get-ListenHost([string]$ConfiguredHost) {
  if (-not [string]::IsNullOrWhiteSpace($ConfiguredHost)) {
    return $ConfiguredHost
  }
  if (-not [string]::IsNullOrWhiteSpace($env:HOST)) {
    return $env:HOST
  }
  return "127.0.0.1"
}

function Get-ListenPort([int]$ConfiguredPort) {
  if ($ConfiguredPort -gt 0) {
    return $ConfiguredPort
  }
  $envPort = 0
  if ([int]::TryParse([string]$env:PORT, [ref]$envPort) -and $envPort -gt 0) {
    return $envPort
  }
  return 3000
}

function Start-BrowserLauncher([string]$Url, [switch]$Disabled) {
  if ($Disabled) { return }

  $launcherScript = @'
param([string]$Url)
$ErrorActionPreference = "SilentlyContinue"
$deadline = (Get-Date).AddMinutes(2)
while ((Get-Date) -lt $deadline) {
  try {
    $response = Invoke-WebRequest -UseBasicParsing -Uri ($Url.TrimEnd('/') + "/api/runtime/readiness") -TimeoutSec 5
    if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) {
      Start-Process $Url | Out-Null
      exit 0
    }
  } catch {
  }
  Start-Sleep -Seconds 2
}
exit 0
'@

  Start-Job -ScriptBlock {
    param($scriptText, $targetUrl)
    $tempPath = Join-Path ([System.IO.Path]::GetTempPath()) ("arcsub-open-browser-" + [Guid]::NewGuid().ToString("N") + ".ps1")
    [System.IO.File]::WriteAllText($tempPath, $scriptText, [System.Text.UTF8Encoding]::new($false))
    try {
      & powershell -ExecutionPolicy Bypass -File $tempPath -Url $targetUrl | Out-Null
    } finally {
      Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
    }
  } -ArgumentList $launcherScript, $Url | Out-Null
}

function Get-PortableNodeExe([string]$RootDir) {
  $candidate = Join-Path $RootDir ".arcsub-bootstrap\node\windows-x64\node.exe"
  if (Test-Path $candidate) { return $candidate }
  return $null
}

function Get-NodeMajor([string]$NodeExe) {
  try {
    $version = & $NodeExe -p "process.versions.node.split('.')[0]"
    return [int]$version
  } catch {
    return 0
  }
}

function Resolve-NodeExe([string]$RootDir) {
  $portable = Get-PortableNodeExe -RootDir $RootDir
  if ($portable -and (Get-NodeMajor $portable) -ge 22) {
    return $portable
  }

  $systemNodeCommand = Get-Command node -ErrorAction SilentlyContinue
  $systemNode = if ($systemNodeCommand) { $systemNodeCommand.Source } else { $null }
  if ($systemNode -and (Get-NodeMajor $systemNode) -ge 22) {
    return $systemNode
  }

  throw "Node.js 22+ not found. Run .\deploy.ps1 first."
}

function Get-ReleaseSummary([string]$RootDir) {
  $manifestPath = Join-Path $RootDir "release-manifest.json"
  if (-not (Test-Path $manifestPath)) {
    return $null
  }

  try {
    $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
    $parts = @()
    if ($manifest.version) { $parts += "version=$($manifest.version)" }
    if ($manifest.target) { $parts += "target=$($manifest.target)" }
    if ($manifest.buildId) { $parts += "buildId=$($manifest.buildId)" }
    if ($manifest.gitHead) { $parts += "git=$($manifest.gitHead)" }
    if ($parts.Count -eq 0) {
      return $null
    }
    return ($parts -join " ")
  }
  catch {
    return $null
  }
}

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildEntry = Join-Path $scriptPath "build\server\index.js"

if (-not (Test-Path $buildEntry)) {
  throw "Missing production server entry: $buildEntry. Run .\deploy.ps1 first."
}

Set-ConsoleUtf8
$nodeExe = Resolve-NodeExe -RootDir $scriptPath
$resolvedHost = Get-ListenHost -ConfiguredHost $ListenHost
$resolvedPort = Get-ListenPort -ConfiguredPort $Port
$browserUrl = "http://${resolvedHost}:${resolvedPort}"

Write-Step "Starting production server..."
Write-Host "[ArcSub] node=$nodeExe" -ForegroundColor Gray
Write-Host "[ArcSub] url=$browserUrl" -ForegroundColor Gray
$releaseSummary = Get-ReleaseSummary -RootDir $scriptPath
if ($releaseSummary) {
  Write-Host "[ArcSub] release=$releaseSummary" -ForegroundColor Gray
}

if ($DryRun) {
  Write-Step "Dry run finished."
  exit 0
}

$env:HOST = $resolvedHost
$env:PORT = [string]$resolvedPort
$env:NODE_ENV = "production"
Start-BrowserLauncher -Url $browserUrl -Disabled:$NoBrowser

Push-Location $scriptPath
try {
  & $nodeExe $buildEntry
  if ($LASTEXITCODE -ne 0) {
    throw "Production server exited with code $LASTEXITCODE."
  }
}
finally {
  Pop-Location
}
