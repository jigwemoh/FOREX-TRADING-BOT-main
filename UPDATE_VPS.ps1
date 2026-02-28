# UPDATE_VPS.ps1
# PowerShell script to update the FOREX-TRADING-BOT pipeline on your VPS

param(
    [string]$RepoUrl = "https://github.com/jigwemoh/FOREX-TRADING-BOT-main.git",
    [string]$RepoPath = "C:\FOREX-TRADING-BOT-main",
    [string]$Branch = "main"
)

Write-Host "[INFO] Stopping running bot processes (if any)..."
Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "*$RepoPath*" } |
    Stop-Process -Force -ErrorAction SilentlyContinue

if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Git is not installed or not in PATH. Install Git for Windows: https://git-scm.com/download/win"
    exit 1
}

if (!(Test-Path $RepoPath)) {
    Write-Host "[INFO] Repository folder not found. Cloning into $RepoPath ..."
    git clone $RepoUrl $RepoPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] git clone failed"
        exit 1
    }
}

Set-Location $RepoPath

function Invoke-SafeGitPull {
    param(
        [string]$TargetBranch
    )

    git pull origin $TargetBranch
    if ($LASTEXITCODE -eq 0) {
        return $true
    }

    Write-Host "[WARN] git pull failed (likely untracked-file conflicts). Creating backup and retrying..."

    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $backupDir = "C:\FOREX-BOT-BACKUPS\PRE_PULL_$timestamp"
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

    if (Test-Path "ALL_MODELS") {
        Copy-Item "ALL_MODELS" -Destination (Join-Path $backupDir "ALL_MODELS") -Recurse -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path "config.json") {
        Copy-Item "config.json" -Destination (Join-Path $backupDir "config.json") -Force -ErrorAction SilentlyContinue
    }

    Write-Host "[INFO] Backup saved to $backupDir"

    git reset --hard
    git clean -fd -- ALL_MODELS .vscode
    if (Test-Path ".DS_Store") { Remove-Item ".DS_Store" -Force -ErrorAction SilentlyContinue }
    if (Test-Path "ALL_MODELS\.DS_Store") { Remove-Item "ALL_MODELS\.DS_Store" -Force -ErrorAction SilentlyContinue }

    git pull origin $TargetBranch
    return ($LASTEXITCODE -eq 0)
}

if (!(Test-Path ".git")) {
    Write-Host "[ERROR] $RepoPath is not a git repository."
    Write-Host "[INFO] Remove this folder and re-run this script so it can clone fresh."
    exit 1
}

Write-Host "[INFO] Pulling latest code from GitHub..."
git fetch origin
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] git fetch failed"
    exit 1
}

git checkout $Branch
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] git checkout $Branch failed"
    exit 1
}

if (-not (Invoke-SafeGitPull -TargetBranch $Branch)) {
    Write-Host "[ERROR] git pull failed"
    exit 1
}

Write-Host "[INFO] Updating Python dependencies..."
if (Test-Path ".venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
} else {
    Write-Host "[WARN] .venv not found at $RepoPath. Install dependencies in your active environment."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
}

Write-Host "[INFO] Update complete in $RepoPath"
