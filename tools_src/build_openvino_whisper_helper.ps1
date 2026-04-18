param(
  [switch]$Clean
)

$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $PSScriptRoot
$openvinoNodeBin = Join-Path $workspace "node_modules\openvino-node\bin"
$openvinoGenaiBin = Join-Path $workspace "node_modules\openvino-genai-node\bin"

if (!(Test-Path $openvinoNodeBin) -or !(Test-Path $openvinoGenaiBin)) {
  throw "OpenVINO node runtime bins not found. Please run npm install first."
}

$env:OPENVINO_LIB_PATHS = "$openvinoNodeBin;$openvinoGenaiBin"
$env:PATH = "$openvinoNodeBin;$openvinoGenaiBin;$env:PATH"

$workPath = Join-Path $workspace "build\pyinstaller"
$distPath = Join-Path $workspace "runtime\tools"
$specPath = $workPath
$entryScript = Join-Path $PSScriptRoot "openvino_whisper_helper.py"

if ($Clean) {
  Remove-Item -LiteralPath $workPath -Recurse -Force -ErrorAction SilentlyContinue
}

python -m PyInstaller `
  --noconfirm `
  --onefile `
  --name openvino_whisper_helper `
  --distpath $distPath `
  --workpath $workPath `
  --specpath $specPath `
  $entryScript

Write-Host "Built: $distPath\openvino_whisper_helper.exe" -ForegroundColor Green

