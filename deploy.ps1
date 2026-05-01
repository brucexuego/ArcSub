param(
  [switch]$SkipBuild,
  [switch]$SkipPyannote,
  [switch]$PreinstallLocalModelPython
)

$ErrorActionPreference = "Stop"
$requiredNodeMajor = 22
$requiredPythonVersion = [Version]"3.12.0"

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

function Write-Info([string]$Message) {
  Write-Host "[ArcSub] $Message" -ForegroundColor Gray
}

function Get-DeployManifest([string]$RootDir) {
  $manifestPath = Join-Path $RootDir "scripts\deploy-manifest.json"
  if (-not (Test-Path $manifestPath)) {
    throw "Missing deploy manifest: $manifestPath"
  }
  return Get-Content $manifestPath -Raw | ConvertFrom-Json
}

function Get-NodeMajor([string]$NodeExe) {
  try {
    return [int](& $NodeExe -p "process.versions.node.split('.')[0]")
  } catch {
    return 0
  }
}

function Get-DotEnvValue([string]$Path, [string]$Key, [string]$Default = "") {
  if (-not (Test-Path $Path)) {
    return $Default
  }

  $line = Get-Content $Path -ErrorAction SilentlyContinue |
    Where-Object { $_ -match "^\s*$([regex]::Escape($Key))\s*=" } |
    Select-Object -First 1
  if (-not $line) {
    return $Default
  }

  return ($line -replace "^\s*$([regex]::Escape($Key))\s*=\s*", "").Trim()
}

function Set-DotEnvValue([string]$Path, [string]$Key, [string]$Value) {
  $encoding = [System.Text.UTF8Encoding]::new($false)
  $lines = [System.Collections.Generic.List[string]]::new()
  if (Test-Path $Path) {
    foreach ($line in Get-Content $Path) {
      $lines.Add($line)
    }
  }

  $pattern = "^\s*$([regex]::Escape($Key))\s*="
  $updated = $false
  for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match $pattern) {
      $lines[$i] = "$Key=$Value"
      $updated = $true
      break
    }
  }

  if (-not $updated) {
    if ($lines.Count -gt 0 -and $lines[$lines.Count - 1] -ne "") {
      $lines.Add("")
    }
    $lines.Add("$Key=$Value")
  }

  [System.IO.File]::WriteAllLines($Path, $lines, $encoding)
}

function Read-PlainSecret([string]$Prompt) {
  $secure = Read-Host -Prompt $Prompt -AsSecureString
  $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
  try {
    return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
  } finally {
    if ($pointer -ne [IntPtr]::Zero) {
      [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
  }
}

function Test-PyannoteReady([string]$RootDir) {
  $paths = @(
    (Join-Path $RootDir "runtime\models\pyannote\segmentation\model.xml"),
    (Join-Path $RootDir "runtime\models\pyannote\embedding\model.xml"),
    (Join-Path $RootDir "runtime\models\pyannote\plda\vbx.json")
  )
  foreach ($candidate in $paths) {
    if (-not (Test-Path $candidate)) {
      return $false
    }
  }
  return $true
}

function Install-PyannoteAssets([string]$RootDir, [string]$NodeExe) {
  $null = Invoke-Process -FilePath $NodeExe -ArgumentList @("scripts/install-pyannote-assets.mjs", "--readiness-out", "runtime/deploy/asset-readiness.json") -WorkingDirectory $RootDir
}

function Ensure-PyannoteDeployment([string]$RootDir, [string]$NodeExe, [string]$EnvPath) {
  if (Test-PyannoteReady -RootDir $RootDir) {
    Write-Info "Pyannote assets already installed."
    return
  }

  $token = Get-DotEnvValue -Path $EnvPath -Key "HF_TOKEN"

  while (-not (Test-PyannoteReady -RootDir $RootDir)) {
    if (-not [string]::IsNullOrWhiteSpace($token)) {
      try {
        Write-Step "Installing pyannote assets..."
        $env:HF_TOKEN = $token
        Install-PyannoteAssets -RootDir $RootDir -NodeExe $NodeExe
        return
      } catch {
        Write-Host "[ArcSub] Pyannote installation failed: $($_.Exception.Message)" -ForegroundColor Yellow
      }
    }

    Write-Host "[ArcSub] HF_TOKEN is optional and is used for Pyannote assets plus Hugging Face models that require approval, login, or private access." -ForegroundColor Yellow
    Write-Host "[ArcSub] Accept the pyannote model access on Hugging Face before installing Pyannote assets. The same token is saved for later local-model downloads." -ForegroundColor DarkYellow
    $tokenInput = (Read-PlainSecret -Prompt "HF_TOKEN (leave blank to skip Pyannote asset install for now)").Trim()
    if ([string]::IsNullOrWhiteSpace($tokenInput)) {
      Write-Info "Skipping pyannote installation."
      Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
      return
    }

    $token = $tokenInput
    Set-DotEnvValue -Path $EnvPath -Key "HF_TOKEN" -Value $token
    $env:HF_TOKEN = $token
  }
}

function New-RandomHex([int]$ByteCount) {
  $bytes = New-Object byte[] $ByteCount
  $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
  try {
    $rng.GetBytes($bytes)
  } finally {
    if ($rng) {
      $rng.Dispose()
    }
  }
  return ([System.BitConverter]::ToString($bytes) -replace "-", "").ToLowerInvariant()
}

function Ensure-DotEnvFile([string]$WorkspaceRoot) {
  $envPath = Join-Path $WorkspaceRoot ".env"
  $examplePath = Join-Path $WorkspaceRoot ".env.example"
  if (-not (Test-Path $envPath)) {
    Write-Step "Creating .env from .env.example"
    Copy-Item -LiteralPath $examplePath -Destination $envPath
  }

  $currentKey = Get-DotEnvValue -Path $envPath -Key "ENCRYPTION_KEY"
  if ([string]::IsNullOrWhiteSpace($currentKey) -or $currentKey -eq "replace_with_random_64_hex") {
    $newKey = New-RandomHex 32
    Set-DotEnvValue -Path $envPath -Key "ENCRYPTION_KEY" -Value $newKey
    Write-Info "Generated a fresh ENCRYPTION_KEY in .env"
  } elseif ($currentKey -notmatch "^[0-9a-fA-F]{64}$") {
    Write-Info "Keeping existing custom ENCRYPTION_KEY in .env"
  }

  return $envPath
}

function Get-PortableNodeExe([string]$RootDir) {
  $candidate = Join-Path $RootDir ".arcsub-bootstrap\node\windows-x64\node.exe"
  if (Test-Path $candidate) { return $candidate }
  return $null
}

function Ensure-PortableNode([string]$RootDir) {
  $portable = Get-PortableNodeExe -RootDir $RootDir
  if ($portable -and (Get-NodeMajor $portable) -ge $requiredNodeMajor) {
    return $portable
  }

  Write-Step "Bootstrapping portable Node.js..."
  $index = Invoke-RestMethod -Uri "https://nodejs.org/dist/index.json" -TimeoutSec 60
  $release = $index | Where-Object { $_.version -match '^v22\.' } | Select-Object -First 1
  if (-not $release) {
    throw "Unable to resolve latest Node.js v22 release."
  }

  $version = [string]$release.version
  $archiveName = "node-$version-win-x64.zip"
  $downloadUrl = "https://nodejs.org/dist/$version/$archiveName"
  $bootstrapDir = Join-Path $RootDir ".arcsub-bootstrap"
  $downloadDir = Join-Path $bootstrapDir "downloads"
  $targetDir = Join-Path $bootstrapDir "node\windows-x64"
  $tempExtractDir = Join-Path $bootstrapDir "node\_extract"
  $zipPath = Join-Path $downloadDir $archiveName

  New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
  if (Test-Path $tempExtractDir) {
    Remove-Item -Recurse -Force $tempExtractDir
  }
  if (Test-Path $targetDir) {
    Remove-Item -Recurse -Force $targetDir
  }

  Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath -TimeoutSec 600
  Expand-Archive -LiteralPath $zipPath -DestinationPath $tempExtractDir -Force

  $extractedRoot = Join-Path $tempExtractDir "node-$version-win-x64"
  if (-not (Test-Path $extractedRoot)) {
    throw "Portable Node.js archive did not extract as expected."
  }

  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $targetDir) | Out-Null
  Move-Item -LiteralPath $extractedRoot -Destination $targetDir
  Remove-Item -Recurse -Force $tempExtractDir

  $portableExe = Join-Path $targetDir "node.exe"
  if (-not (Test-Path $portableExe)) {
    throw "Portable Node.js bootstrap failed: node.exe missing."
  }
  return $portableExe
}

function Get-NpmCmd([string]$NodeExe) {
  $nodeDir = Split-Path -Parent $NodeExe
  $npmCmd = Join-Path $nodeDir "npm.cmd"
  if (Test-Path $npmCmd) {
    return $npmCmd
  }

  $systemNpmCommand = Get-Command npm -ErrorAction SilentlyContinue
  $systemNpm = if ($systemNpmCommand) { $systemNpmCommand.Source } else { $null }
  if ($systemNpm) {
    return $systemNpm
  }

  throw "npm not found next to Node.js runtime."
}

function Get-PythonVersion([string]$PythonCommand) {
  try {
    return (& $PythonCommand -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')" 2>$null).Trim()
  } catch {
    return ""
  }
}

function Test-IsWindowsAppsAlias([string]$CommandPath) {
  if ([string]::IsNullOrWhiteSpace($CommandPath)) {
    return $false
  }
  $normalized = $CommandPath.Replace('/', '\')
  return $normalized -like "*\AppData\Local\Microsoft\WindowsApps\*"
}

function Test-PythonVersionAtLeast([string]$VersionText, [Version]$MinimumVersion) {
  if ([string]::IsNullOrWhiteSpace($VersionText)) {
    return $false
  }
  try {
    return ([Version]$VersionText) -ge $MinimumVersion
  } catch {
    return $false
  }
}

function Test-PipAvailable([string]$PythonExe) {
  try {
    & $PythonExe -m pip --version > $null
    return $LASTEXITCODE -eq 0
  } catch {
    return $false
  }
}

function Ensure-PipAvailable([string]$PythonExe, [string]$RootDir, $Manifest) {
  if (Test-PipAvailable -PythonExe $PythonExe) {
    return
  }

  $getPipUrl = [string]$Manifest.python.getPipUrl
  $getPipUrl = $getPipUrl.Trim()
  if ([string]::IsNullOrWhiteSpace($getPipUrl)) {
    throw "Python pip bootstrap metadata is missing from scripts/deploy-manifest.json"
  }

  $downloadDir = Join-Path $RootDir ".arcsub-bootstrap\downloads"
  $getPipPath = Join-Path $downloadDir "get-pip.py"
  New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
  if (-not (Test-Path $getPipPath)) {
    Invoke-WebRequest -Uri $getPipUrl -OutFile $getPipPath -TimeoutSec 600
  }

  $pythonDir = Split-Path -Parent $PythonExe
  $envPatch = @{
    PATH = "$pythonDir;$env:PATH"
  }
  $null = Invoke-Process -FilePath $PythonExe -ArgumentList @($getPipPath, "--no-warn-script-location") -WorkingDirectory $RootDir -ExtraEnv $envPatch
}

function Ensure-AsrHelperPythonDependencies([string]$PythonExe, [string]$RootDir) {
  Write-Step "Ensuring ASR helper Python dependencies..."
  $probeScript = @'
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
'@

  $probeScriptPath = Join-Path $RootDir ".arcsub-bootstrap\downloads\probe_asr_helper_deps.py"
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $probeScriptPath) | Out-Null
  [System.IO.File]::WriteAllText($probeScriptPath, $probeScript, [System.Text.UTF8Encoding]::new($false))

  $missingOutput = & $PythonExe $probeScriptPath 2>$null
  $missing = @(
    ($missingOutput -split "`r?`n" | ForEach-Object { [string]$_ } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  )
  if ($missing.Count -eq 0) {
    Write-Info "ASR helper Python dependencies already available."
    return
  }

  $packageMap = @{
    "openvino" = @("openvino")
    "openvino_genai" = @("openvino-genai")
    "transformers" = @("transformers")
    "huggingface_hub" = @("huggingface_hub")
    "librosa" = @("librosa")
    "accelerate" = @("accelerate")
    "optimum.intel" = @("optimum-intel[openvino]")
  }

  $packages = New-Object System.Collections.Generic.List[string]
  foreach ($name in $missing) {
    foreach ($package in ($packageMap[$name] | Where-Object { $_ })) {
      if (-not $packages.Contains($package)) {
        $packages.Add($package)
      }
    }
  }
  if ($packages.Count -eq 0) {
    return
  }

  $envPatch = @{
    PATH = "$(Split-Path -Parent $PythonExe);$env:PATH"
  }
  $args = @("-m", "pip", "install", "--upgrade") + @($packages.ToArray())
  $null = Invoke-Process -FilePath $PythonExe -ArgumentList $args -WorkingDirectory $RootDir -ExtraEnv $envPatch
}

function Ensure-AlignmentPythonDependencies([string]$PythonExe, [string]$RootDir) {
  Write-Step "Ensuring alignment conversion Python dependencies..."
  $probeScript = @'
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
'@

  $probeScriptPath = Join-Path $RootDir ".arcsub-bootstrap\downloads\probe_alignment_deps.py"
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $probeScriptPath) | Out-Null
  [System.IO.File]::WriteAllText($probeScriptPath, $probeScript, [System.Text.UTF8Encoding]::new($false))

  $missingOutput = & $PythonExe $probeScriptPath 2>$null
  $missing = @(
    ($missingOutput -split "`r?`n" | ForEach-Object { [string]$_ } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  )
  if ($missing.Count -eq 0) {
    Write-Info "Alignment conversion Python dependencies already available."
    return
  }

  $packageMap = @{
    "openvino" = @("openvino")
    "nncf" = @("nncf")
    "huggingface_hub" = @("huggingface_hub")
    "openvino_tokenizers" = @("openvino-tokenizers")
    "transformers" = @("transformers")
    "optimum.intel" = @("optimum-intel[openvino]")
    "torch" = @("torch")
    "sentencepiece" = @("sentencepiece")
  }

  $packages = New-Object System.Collections.Generic.List[string]
  foreach ($name in $missing) {
    foreach ($package in ($packageMap[$name] | Where-Object { $_ })) {
      if (-not $packages.Contains($package)) {
        $packages.Add($package)
      }
    }
  }
  if ($packages.Count -eq 0) {
    return
  }

  $envPatch = @{
    PATH = "$(Split-Path -Parent $PythonExe);$env:PATH"
  }
  $args = @("-m", "pip", "install", "--upgrade") + @($packages.ToArray())
  $null = Invoke-Process -FilePath $PythonExe -ArgumentList $args -WorkingDirectory $RootDir -ExtraEnv $envPatch
}

function Get-PortablePythonExe([string]$RootDir) {
  $candidate = Join-Path $RootDir ".arcsub-bootstrap\python\windows-x64\python.exe"
  if (Test-Path $candidate) { return $candidate }
  return $null
}

function Ensure-PortablePython([string]$RootDir, $Manifest) {
  $portable = Get-PortablePythonExe -RootDir $RootDir
  if ($portable) {
    $portableVersion = Get-PythonVersion $portable
    if (Test-PythonVersionAtLeast -VersionText $portableVersion -MinimumVersion $requiredPythonVersion) {
      return $portable
    }
  }

  $configuredPython = [string]$env:OPENVINO_HELPER_PYTHON
  $configuredPython = $configuredPython.Trim()
  if (-not [string]::IsNullOrWhiteSpace($configuredPython)) {
    $configuredVersion = Get-PythonVersion $configuredPython
    if (Test-PythonVersionAtLeast -VersionText $configuredVersion -MinimumVersion $requiredPythonVersion) {
      return $configuredPython
    }
  }

  $systemPythonCommand = Get-Command python -ErrorAction SilentlyContinue
  $systemPython = if ($systemPythonCommand) { $systemPythonCommand.Source } else { $null }
  if ($systemPython -and -not (Test-IsWindowsAppsAlias -CommandPath $systemPython)) {
    $systemVersion = Get-PythonVersion $systemPython
    if (Test-PythonVersionAtLeast -VersionText $systemVersion -MinimumVersion $requiredPythonVersion) {
      return $systemPython
    }
  }

  $launcher = Get-Command py -ErrorAction SilentlyContinue
  if ($launcher) {
    try {
      $launcherPath = (& $launcher.Source -3.12 -c "import sys; print(sys.executable)" 2>$null).Trim()
      $launcherVersion = (& $launcher.Source -3.12 -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')" 2>$null).Trim()
      if (Test-PythonVersionAtLeast -VersionText $launcherVersion -MinimumVersion $requiredPythonVersion) {
        return $launcherPath
      }
    } catch {
    }
  }

  $pythonVersion = [string]$Manifest.python.windowsVersion
  $pythonVersion = $pythonVersion.Trim()
  $embedUrl = [string]$Manifest.python.windowsEmbedUrl
  $embedUrl = $embedUrl.Trim()
  $getPipUrl = [string]$Manifest.python.getPipUrl
  $getPipUrl = $getPipUrl.Trim()
  if ([string]::IsNullOrWhiteSpace($pythonVersion) -or [string]::IsNullOrWhiteSpace($embedUrl) -or [string]::IsNullOrWhiteSpace($getPipUrl)) {
    throw "Python bootstrap metadata is missing from scripts/deploy-manifest.json"
  }

  Write-Step "Bootstrapping portable Python $pythonVersion..."
  $bootstrapDir = Join-Path $RootDir ".arcsub-bootstrap"
  $downloadDir = Join-Path $bootstrapDir "downloads"
  $targetDir = Join-Path $bootstrapDir "python\windows-x64"
  $extractDir = Join-Path $bootstrapDir "python\_extract"
  $archiveName = "python-$pythonVersion-embed-amd64.zip"
  $archivePath = Join-Path $downloadDir $archiveName
  $getPipPath = Join-Path $downloadDir "get-pip.py"

  New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
  if (Test-Path $extractDir) {
    Remove-Item -Recurse -Force $extractDir
  }
  if (Test-Path $targetDir) {
    Remove-Item -Recurse -Force $targetDir
  }

  Invoke-WebRequest -Uri $embedUrl -OutFile $archivePath -TimeoutSec 600
  Invoke-WebRequest -Uri $getPipUrl -OutFile $getPipPath -TimeoutSec 600
  Expand-Archive -LiteralPath $archivePath -DestinationPath $extractDir -Force

  New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
  $extractedItems = Get-ChildItem -Path $extractDir -Force
  if (-not $extractedItems) {
    throw "Portable Python archive extracted no files."
  }
  foreach ($item in $extractedItems) {
    Move-Item -LiteralPath $item.FullName -Destination $targetDir
  }
  Remove-Item -Recurse -Force $extractDir

  $pythonExe = Join-Path $targetDir "python.exe"
  if (-not (Test-Path $pythonExe)) {
    throw "Portable Python bootstrap failed: python.exe missing."
  }

  $pthPath = Get-ChildItem -Path $targetDir -Filter "python*._pth" -File | Select-Object -First 1
  if ($pthPath) {
    $content = Get-Content $pthPath.FullName
    $updated = $content | ForEach-Object {
      if ($_ -match '^\s*#\s*import site\s*$') { 'import site' } else { $_ }
    }
    [System.IO.File]::WriteAllLines($pthPath.FullName, $updated, [System.Text.UTF8Encoding]::new($false))
  }

  $envPatch = @{
    PATH = "$targetDir;$env:PATH"
  }
  $null = Invoke-Process -FilePath $pythonExe -ArgumentList @($getPipPath, "--no-warn-script-location") -WorkingDirectory $targetDir -ExtraEnv $envPatch
  return $pythonExe
}

function Get-NpmCliPath([string]$NodeExe) {
  $nodeDir = Split-Path -Parent $NodeExe
  $candidate = Join-Path $nodeDir "node_modules\npm\bin\npm-cli.js"
  if (Test-Path $candidate) {
    return $candidate
  }
  return $null
}

function Invoke-Process([string]$FilePath, [string[]]$ArgumentList, [string]$WorkingDirectory, [hashtable]$ExtraEnv = @{}) {
  $originalEnv = @{}
  $stdoutPath = [System.IO.Path]::GetTempFileName()
  $stderrPath = [System.IO.Path]::GetTempFileName()
  try {
    foreach ($entry in $ExtraEnv.GetEnumerator()) {
      $originalEnv[$entry.Key] = [Environment]::GetEnvironmentVariable($entry.Key, "Process")
      [Environment]::SetEnvironmentVariable($entry.Key, [string]$entry.Value, "Process")
    }

    $process = Start-Process `
      -FilePath $FilePath `
      -ArgumentList $ArgumentList `
      -WorkingDirectory $WorkingDirectory `
      -RedirectStandardOutput $stdoutPath `
      -RedirectStandardError $stderrPath `
      -NoNewWindow `
      -Wait `
      -PassThru

    if (Test-Path $stdoutPath) {
      foreach ($line in Get-Content $stdoutPath -ErrorAction SilentlyContinue) {
        if ($null -ne $line) {
          Write-Host ([string]$line) -ForegroundColor DarkGray
        }
      }
    }
    if (Test-Path $stderrPath) {
      foreach ($line in Get-Content $stderrPath -ErrorAction SilentlyContinue) {
        if ($null -ne $line) {
          Write-Host ([string]$line) -ForegroundColor DarkGray
        }
      }
    }
    if ($process.ExitCode -ne 0) {
      throw "Command failed: $FilePath $($ArgumentList -join ' ')"
    }
  } finally {
    foreach ($key in $originalEnv.Keys) {
      [Environment]::SetEnvironmentVariable($key, $originalEnv[$key], "Process")
    }
    Remove-Item $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
  }
}

function Invoke-Npm([string]$NodeExe, [string]$NpmCmd, [string]$WorkingDirectory, [string[]]$Arguments) {
  $nodeDir = Split-Path -Parent $NodeExe
  $envPatch = @{
    PATH = "$nodeDir;$env:PATH"
  }

  $npmCli = Get-NpmCliPath -NodeExe $NodeExe
  if ($npmCli) {
    $npmCliArgs = @($npmCli) + $Arguments
    $null = Invoke-Process -FilePath $NodeExe -ArgumentList $npmCliArgs -WorkingDirectory $WorkingDirectory -ExtraEnv $envPatch
    return
  }

  $null = Invoke-Process -FilePath $NpmCmd -ArgumentList $Arguments -WorkingDirectory $WorkingDirectory -ExtraEnv $envPatch
}

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifest = Get-DeployManifest -RootDir $scriptPath
Set-ConsoleUtf8
Write-Step "Preparing production deployment..."
Write-Info "Workspace: $scriptPath"

$nodeExe = Ensure-PortableNode -RootDir $scriptPath
$pythonExe = Ensure-PortablePython -RootDir $scriptPath -Manifest $manifest
Ensure-PipAvailable -PythonExe $pythonExe -RootDir $scriptPath -Manifest $manifest
if ($PreinstallLocalModelPython) {
  Ensure-AsrHelperPythonDependencies -PythonExe $pythonExe -RootDir $scriptPath
  Ensure-AlignmentPythonDependencies -PythonExe $pythonExe -RootDir $scriptPath
} else {
  Write-Info "Deferring local-model Python dependency install until helper/conversion use."
}
$npmCmd = Get-NpmCmd -NodeExe $nodeExe
$isSourceWorkspace = (Test-Path (Join-Path $scriptPath "src")) -and (Test-Path (Join-Path $scriptPath "server\index.ts"))
$hasWorkspaceTooling =
  (Test-Path (Join-Path $scriptPath "node_modules\.bin\tsc.cmd")) -and
  (Test-Path (Join-Path $scriptPath "node_modules\.bin\vite.cmd"))

Write-Info "node=$nodeExe"
Write-Info "npm=$npmCmd"
Write-Info "python=$pythonExe"
Write-Info "mode=$(if ($isSourceWorkspace) { 'source' } else { 'release' })"

$envPath = Ensure-DotEnvFile -WorkspaceRoot $scriptPath
Set-DotEnvValue -Path $envPath -Key "OPENVINO_HELPER_PYTHON" -Value $pythonExe
$env:OPENVINO_HELPER_PYTHON = $pythonExe
$env:PATH = "$(Split-Path -Parent $pythonExe);$env:PATH"
Write-Info ".env=$envPath"

if ($isSourceWorkspace) {
  if (-not (Test-Path (Join-Path $scriptPath "node_modules")) -or -not $hasWorkspaceTooling) {
    Write-Step "Installing workspace dependencies..."
    Invoke-Npm -NodeExe $nodeExe -NpmCmd $npmCmd -WorkingDirectory $scriptPath -Arguments @("install")
  }
  if (-not $SkipBuild) {
    Write-Step "Building production assets..."
    Invoke-Npm -NodeExe $nodeExe -NpmCmd $npmCmd -WorkingDirectory $scriptPath -Arguments @("run", "-s", "build:prod")
  }
} else {
  if (-not (Test-Path (Join-Path $scriptPath "node_modules"))) {
    Write-Step "Installing release runtime dependencies..."
    Invoke-Npm -NodeExe $nodeExe -NpmCmd $npmCmd -WorkingDirectory $scriptPath -Arguments @("install", "--omit=dev", "--no-fund", "--no-audit")
  }
}

$assetArgs = @("scripts/install-deployment-assets.mjs", "--readiness-out", "runtime/deploy/asset-readiness.json", "--skip-pyannote")
Write-Step "Installing deployment assets..."
$null = Invoke-Process -FilePath $nodeExe -ArgumentList $assetArgs -WorkingDirectory $scriptPath

if (-not $SkipPyannote) {
  Ensure-PyannoteDeployment -RootDir $scriptPath -NodeExe $nodeExe -EnvPath $envPath
}

Write-Step "Deployment completed."
Write-Info "Readiness: runtime/deploy/asset-readiness.json"
