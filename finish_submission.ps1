# SafeFall AI - finish the GitHub submission.
#
# Run this AFTER creating an empty repository at https://github.com/new
# (name: safefall-ai, Public, and do NOT tick "Add a README/.gitignore/license").
#
#   .\finish_submission.ps1 -User YOUR-GITHUB-USERNAME
#
# It wires up the remote, pushes, and verifies that everything Streamlit Cloud
# needs actually made it to GitHub.

param(
    [Parameter(Mandatory = $true)][string]$User,
    [string]$Repo = "safefall-ai"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$url = "https://github.com/$User/$Repo.git"
Write-Host "Repository: $url" -ForegroundColor Cyan

# --- remote ---------------------------------------------------------------- #
$existing = git remote 2>$null
if ($existing -contains "origin") {
    git remote set-url origin $url
    Write-Host "  updated existing 'origin'"
} else {
    git remote add origin $url
    Write-Host "  added 'origin'"
}

# --- push ------------------------------------------------------------------ #
Write-Host ""
Write-Host "Pushing (about 36 MB). A browser sign-in may appear the first time." -ForegroundColor Cyan
git push -u origin main
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Push failed." -ForegroundColor Red
    Write-Host "  * 'updates were rejected' means the repo was created WITH a README."
    Write-Host "    Delete it and recreate it empty, or run:"
    Write-Host "      git pull --rebase origin main; git push -u origin main"
    Write-Host "  * 'repository not found' means the name or username is wrong,"
    Write-Host "    or the repo has not been created yet."
    exit 1
}

# --- verify what actually landed ------------------------------------------- #
Write-Host ""
Write-Host "Verifying the pushed tree..." -ForegroundColor Cyan
$needed = @(
    "app.py", "requirements.txt", "packages.txt", ".streamlit/config.toml",
    "models/feature_scaler.json", "models/model_metadata.json",
    "results/metrics_summary.json"
)
# Read the pushed tree once, then check membership. (An earlier version tested
# $LASTEXITCODE after a Select-String pipeline, which reports the exit code of
# Select-String rather than of git, so it could both miss real failures and
# double-count the same file as missing.)
$tree = git ls-tree -r origin/main --name-only
$missing = @()
foreach ($f in $needed) {
    if ($tree -contains $f) { Write-Host "  OK    $f" } else { $missing += $f }
}
$weights = ($tree | Where-Object { $_ -like "models/ensemble_member_*_weights.npz" }).Count
Write-Host "  OK    $weights ensemble weight files"

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "MISSING from the push:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  $_" }
    Write-Host "Fix with:  git add -f $($missing -join ' '); git commit -m 'Add missing files'; git push"
    exit 1
}

Write-Host ""
Write-Host "PUSHED SUCCESSFULLY" -ForegroundColor Green
Write-Host "  Repository : https://github.com/$User/$Repo"
Write-Host ""
Write-Host "Next: deploy at https://share.streamlit.io" -ForegroundColor Cyan
Write-Host "  Repository      : $User/$Repo"
Write-Host "  Branch          : main"
Write-Host "  Main file path  : app.py"
Write-Host "  Advanced -> Python version : 3.11 or 3.12  (NOT 3.13 - mediapipe has no wheels)"
