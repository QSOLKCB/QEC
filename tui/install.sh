#!/bin/sh
# QEC Rust TUI Auto-Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh | sh
set -eu

REPO="QSOLKCB/QEC"
ASSET_NAME="qec-tui-linux-x86_64.tar.gz"
INSTALL_DIR="/usr/local/bin"
BINARY_NAME="qec-tui"
INSTALL_DIR=${QEC_INSTALL_DIR:-${INSTALL_DIR}}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

for tool in curl tar python3 mktemp install; do
    command -v "${tool}" >/dev/null 2>&1 || fail "required command '${tool}' is missing"
done

PLATFORM=$(uname -s)
ARCH=$(uname -m)
case "${PLATFORM}/${ARCH}" in
    Linux/x86_64) ;;
    Linux/aarch64|Linux/arm64|Darwin/x86_64|Darwin/arm64)
        ASSET_NAME="" ;; # No prebuilt package is currently defined for these hosts.
    *) fail "unsupported platform: ${PLATFORM}/${ARCH}" ;;
esac

API_URL="https://api.github.com/repos/${REPO}/releases/latest"
printf 'Fetching latest release from %s...\n' "${REPO}"
RELEASE_JSON=$(curl -fsSL "${API_URL}") || fail "failed to fetch latest release metadata from GitHub"

# Parse the actual asset list; a release tag alone does not promise a binary.
RELEASE_INFO=$(printf '%s' "${RELEASE_JSON}" | python3 -c '
import json, re, sys
try:
    release = json.load(sys.stdin)
    tag = release["tag_name"]
    assets = release["assets"]
    if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise ValueError("invalid release tag")
    if not isinstance(assets, list):
        raise ValueError("invalid asset list")
    urls = [a["browser_download_url"] for a in assets if a["name"] == sys.argv[1]]
    if len(urls) > 1:
        raise ValueError("duplicate binary assets")
    url = urls[0] if urls else ""
    expected = "https://github.com/" + sys.argv[2] + "/releases/download/" + tag + "/" + sys.argv[1]
    if urls and (not isinstance(url, str) or url != expected):
        raise ValueError("unexpected binary asset URL")
    print(tag)
    print(url)
except (KeyError, TypeError, ValueError) as error:
    sys.exit("Invalid GitHub release metadata: " + str(error))
' "${ASSET_NAME}" "${REPO}") || fail "could not resolve GitHub release metadata"
TAG=$(printf '%s\n' "${RELEASE_INFO}" | sed -n '1p')
ASSET_URL=$(printf '%s\n' "${RELEASE_INFO}" | sed -n '2p')
printf 'Latest release: %s\n' "${TAG}"

work_dir=$(mktemp -d)
trap 'rm -rf "${work_dir}"' 0
trap 'exit 1' HUP INT TERM

if [ -n "${ASSET_URL}" ]; then
    printf 'Downloading %s...\n' "${ASSET_NAME}"
    curl -fsSL -o "${work_dir}/binary.tar.gz" "${ASSET_URL}" || fail "failed to download ${ASSET_NAME}"
    # Extract only the expected executable, not arbitrary archive contents.
    tar -xzf "${work_dir}/binary.tar.gz" -C "${work_dir}" "${BINARY_NAME}" || fail "binary not found in release archive"
    candidate="${work_dir}/${BINARY_NAME}"
else
    printf 'No prebuilt TUI asset for %s/%s in %s; building from release source.\n' "${PLATFORM}" "${ARCH}" "${TAG}"
    command -v cargo >/dev/null 2>&1 || fail "source build requires Rust/Cargo (on Ubuntu: sudo apt install cargo build-essential)"
    command -v cc >/dev/null 2>&1 || fail "source build requires a C compiler/linker (on Ubuntu: sudo apt install build-essential)"
    DOWNLOAD_URL="https://github.com/${REPO}/archive/refs/tags/${TAG}.tar.gz"
    curl -fsSL -o "${work_dir}/source.tar.gz" "${DOWNLOAD_URL}" || fail "failed to download ${TAG} source"
    mkdir "${work_dir}/source"
    tar -xzf "${work_dir}/source.tar.gz" -C "${work_dir}/source" --strip-components=1 || fail "failed to extract release source"
    printf 'Building qec-tui with the release Cargo.lock...\n'
    CARGO_TARGET_DIR="${work_dir}/target" cargo build --locked --release --manifest-path "${work_dir}/source/tui/Cargo.toml" || fail "Rust TUI build failed"
    candidate="${work_dir}/target/release/${BINARY_NAME}"
fi

[ -f "${candidate}" ] && [ ! -L "${candidate}" ] || fail "binary '${BINARY_NAME}' is missing or is a symlink"
chmod +x "${candidate}"
# Older releases (including v173.0) ignore --version and open the TUI.
# Installation must not launch an interactive program from curl | sh.

if [ ! -d "${INSTALL_DIR}" ]; then
    if mkdir -p "${INSTALL_DIR}" 2>/dev/null; then
        :
    elif command -v sudo >/dev/null 2>&1; then
        sudo mkdir -p "${INSTALL_DIR}" || fail "cannot create ${INSTALL_DIR}"
    else
        fail "cannot create ${INSTALL_DIR}; set QEC_INSTALL_DIR to a writable directory"
    fi
fi
if [ -w "${INSTALL_DIR}" ]; then
    install -m 755 "${candidate}" "${INSTALL_DIR}/${BINARY_NAME}"
elif command -v sudo >/dev/null 2>&1; then
    printf 'Installing to %s (requires sudo)...\n' "${INSTALL_DIR}"
    sudo install -m 755 "${candidate}" "${INSTALL_DIR}/${BINARY_NAME}"
else
    fail "cannot write to ${INSTALL_DIR}; set QEC_INSTALL_DIR to a writable directory"
fi

printf '%s installed successfully from QEC %s\n' "${BINARY_NAME}" "${TAG}"
printf 'Run with: %s/%s\n' "${INSTALL_DIR}" "${BINARY_NAME}"
case ":${PATH}:" in
    *":${INSTALL_DIR}:"*) ;;
    *) printf 'Add %s to PATH to run qec-tui by name.\n' "${INSTALL_DIR}" ;;
esac
printf 'For Python engine commands, activate your QEC virtual environment (see INSTALL.md).\n'
