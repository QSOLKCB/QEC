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

RELEASE_JSON=$(curl -fsSL "${API_URL}") || {
    printf "Error: failed to fetch latest release metadata from GitHub\n" >&2
    exit 1
}

if command -v jq >/dev/null 2>&1; then
    TAG=$(printf '%s' "${RELEASE_JSON}" | jq -r 'select(.tag_name and .assets) | .tag_name // empty')
else
    printf "Warning: jq not found, falling back to grep/sed JSON parsing\n" >&2

    if ! printf '%s' "${RELEASE_JSON}" | grep -q '"tag_name"' || \
       ! printf '%s' "${RELEASE_JSON}" | grep -q '"assets"' ; then
        printf "Error: unexpected GitHub API response (missing tag_name/assets)\n" >&2
        exit 1
    fi

    TAG=$(printf '%s' "${RELEASE_JSON}" \
        | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
        | head -1)
fi

if [ -z "${TAG}" ]; then
    printf "Error: could not resolve latest release tag from GitHub API response\n" >&2
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

if [ ! -d "${INSTALL_DIR}" ]; then
    if command -v sudo >/dev/null 2>&1; then
        sudo mkdir -p "${INSTALL_DIR}"
    else
        printf "Error: %s does not exist and 'sudo' is not available to create it.\n" "${INSTALL_DIR}" >&2
        exit 1
    fi
fi

if [ -w "${INSTALL_DIR}" ]; then
    cp "${TMPDIR}/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}"
elif command -v sudo >/dev/null 2>&1; then
    printf "Installing to %s (requires sudo)...\n" "${INSTALL_DIR}"
    if ! sudo cp "${TMPDIR}/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}"; then
        printf "Error: failed to install to %s via sudo. Please install manually.\n" "${INSTALL_DIR}" >&2
        exit 1
    fi
else
    printf "Error: cannot write to %s and 'sudo' is not available.\n" "${INSTALL_DIR}" >&2
    printf "Please rerun this script as root or manually copy %s to %s.\n" "${TMPDIR}/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}" >&2
    exit 1
fi

# --- Verify and report ---
INSTALLED_VERSION=$("${INSTALL_DIR}/${BINARY_NAME}" --version 2>/dev/null || printf '%s' "${TAG}")
printf "QEC TUI %s installed successfully\nRun with: qec-tui\n" "${INSTALLED_VERSION}"
