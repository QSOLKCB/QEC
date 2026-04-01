#!/bin/sh
# QEC Rust TUI Auto-Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh | sh
set -eu

# --- Configuration ---
REPO="QSOLKCB/QEC"
ASSET_NAME="qec-tui-linux-x86_64.tar.gz"
INSTALL_DIR="/usr/local/bin"
BINARY_NAME="qec-tui"

# --- Resolve latest release tag ---
API_URL="https://api.github.com/repos/${REPO}/releases/latest"
printf "Fetching latest release from %s...\n" "${REPO}"

RELEASE_JSON=$(curl -fsSL "${API_URL}")
TAG=$(printf '%s' "${RELEASE_JSON}" | grep '"tag_name"' | head -1 | sed 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')

if [ -z "${TAG}" ]; then
    printf "Error: could not resolve latest release tag\n" >&2
    exit 1
fi

printf "Latest release: %s\n" "${TAG}"

# --- Download release asset ---
DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET_NAME}"
TMPDIR=$(mktemp -d)
trap 'rm -rf "${TMPDIR}"' EXIT

printf "Downloading %s...\n" "${ASSET_NAME}"
curl -fsSL -o "${TMPDIR}/${ASSET_NAME}" "${DOWNLOAD_URL}"

# --- Extract binary ---
printf "Extracting...\n"
tar -xzf "${TMPDIR}/${ASSET_NAME}" -C "${TMPDIR}"

if [ ! -f "${TMPDIR}/${BINARY_NAME}" ]; then
    printf "Error: binary '%s' not found in archive\n" "${BINARY_NAME}" >&2
    exit 1
fi

# --- Install ---
chmod +x "${TMPDIR}/${BINARY_NAME}"

if [ -w "${INSTALL_DIR}" ]; then
    cp "${TMPDIR}/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}"
else
    printf "Installing to %s (requires sudo)...\n" "${INSTALL_DIR}"
    sudo cp "${TMPDIR}/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}"
fi

# --- Verify and report ---
INSTALLED_VERSION=$("${INSTALL_DIR}/${BINARY_NAME}" --version 2>/dev/null || printf '%s' "${TAG}")
printf "QEC TUI %s installed successfully\nRun with: qec-tui\n" "${INSTALLED_VERSION}"
