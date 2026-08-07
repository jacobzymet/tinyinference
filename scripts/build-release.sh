#!/usr/bin/env bash
# Build a self-contained release binary for this machine into dist/.
# HTML, JS, and PNG assets are compile-time embedded (see src/web.rs).
#
# For Windows + macOS + Linux artifacts, use GitHub Actions:
#   gh workflow run release.yml
# or: git tag v0.3.1 && git push origin v0.3.1

set -euo pipefail
cd "$(dirname "$0")/.."

triple="$(rustc -vV | awk '/^host:/{print $2}')"
version="$(awk -F\" '/^version/{print $2; exit}' Cargo.toml)"

os=unknown
arch=unknown
case "$triple" in
  *windows*) os=windows ;;
  *apple-darwin*) os=macos ;;
  *linux*) os=linux ;;
esac
case "$triple" in
  aarch64*|arm64*) arch=aarch64 ;;
  x86_64*|amd64*) arch=x86_64 ;;
esac

ext=
[[ "$os" == windows ]] && ext=.exe
artifact="tinyinference-${os}-${arch}${ext}"

echo "Building self-contained tinyinference ${version} for ${triple} → dist/${artifact}"

cargo build --release --locked
mkdir -p dist
cp -f "target/release/tinyinference${ext}" "dist/${artifact}"
chmod +x "dist/${artifact}" 2>/dev/null || true

echo
echo "Done: dist/${artifact}"
echo "This single file includes the chat UI, admin UI, orb.js, and icons."
echo
echo "Other OS binaries: push a v* tag or run  gh workflow run release.yml"
