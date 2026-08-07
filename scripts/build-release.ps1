# Build a self-contained release binary for this machine into dist/.
# HTML, JS, and PNG assets are compile-time embedded (see src/web.rs) — no
# sidecar files are required next to the executable.
#
# Producing Windows + macOS + Linux from one PC is not reliable (especially
# macOS from Windows). For all three platforms, run the Release workflow:
#   gh workflow run release.yml
# or push a version tag: git tag v0.3.1 && git push origin v0.3.1

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$triple = (rustc -vV | Select-String "^host:").ToString().Split(":")[1].Trim()
$version = (Select-String -Path Cargo.toml -Pattern '^version\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value

$os = if ($triple -match "windows") { "windows" }
elseif ($triple -match "apple-darwin") { "macos" }
elseif ($triple -match "linux") { "linux" }
else { "unknown" }

$arch = if ($triple -match "aarch64|arm64") { "aarch64" }
elseif ($triple -match "x86_64|amd64") { "x86_64" }
else { "unknown" }

$ext = if ($os -eq "windows") { ".exe" } else { "" }
$artifact = "tinyinference-$os-$arch$ext"

Write-Host "Building self-contained tinyinference $version for $triple → dist/$artifact"

cargo build --release --locked
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

New-Item -ItemType Directory -Force -Path dist | Out-Null
$src = Join-Path "target/release" ("tinyinference" + $ext)
Copy-Item -Force $src (Join-Path "dist" $artifact)

Write-Host ""
Write-Host "Done: dist/$artifact"
Write-Host "This single file includes the chat UI, admin UI, orb.js, and icons."
Write-Host ""
Write-Host "Other OS binaries: push a v* tag or run  gh workflow run release.yml"
