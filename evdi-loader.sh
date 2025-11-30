#!/bin/bash
#  ╦╔═╗╔═╗╦╔═╦╔═╔╗╔╦╔═╗╔═╗  ╔═╗╦
#  ║╠═╣║  ╠╩╗╠╩╗║║║║╠╣ ║╣   ╠═╣║
# ╚╝╩ ╩╚═╝╩ ╩╩ ╩╝╚╝╩╚  ╚═╝  ╩ ╩╩    JACKKNIFE AI™
#
# EVDI Loader for Immutable Fedora (Silverblue/Aurora/Bazzite)
#
# Problem: DKMS builds evdi but can't install to read-only /lib/modules.
# Solution: Load evdi directly from the DKMS build path.
#
# This script:
#   - Skips if evdi is already loaded
#   - Auto-builds for new kernels via dkms autoinstall
#   - Decompresses .ko.xz if needed
#   - Loads from the correct kernel-versioned path
#
# Install to: /usr/local/bin/evdi-loader.sh
# Author: Alexander Casavant / JACKKNIFE AI™
# License: MIT

set -euo pipefail

KERNEL=$(uname -r)
EVDI_VERSION="${EVDI_VERSION:-1.14.11}"
DKMS_BASE="/var/lib/dkms/evdi/${EVDI_VERSION}"
MODULE_PATH="${DKMS_BASE}/${KERNEL}/x86_64/module/evdi.ko"
MODULE_PATH_XZ="${MODULE_PATH}.xz"

log() {
    echo "[evdi-loader] $*"
}

# Check if module already loaded
if lsmod | grep -q "^evdi"; then
    log "evdi already loaded"
    exit 0
fi

# Check if DKMS source exists
if [ ! -d "${DKMS_BASE}" ]; then
    log "ERROR: evdi DKMS source not found at ${DKMS_BASE}"
    log "Install displaylink package first: rpm-ostree install displaylink"
    exit 1
fi

# Check if module exists for current kernel, build if not
if [ ! -f "$MODULE_PATH" ] && [ ! -f "$MODULE_PATH_XZ" ]; then
    log "Building evdi for kernel ${KERNEL}..."
    if ! dkms autoinstall; then
        log "ERROR: dkms autoinstall failed"
        exit 1
    fi
fi

# Decompress if needed
if [ -f "$MODULE_PATH_XZ" ] && [ ! -f "$MODULE_PATH" ]; then
    log "Decompressing evdi module..."
    xz -dk "$MODULE_PATH_XZ"
fi

# Load the module
if [ -f "$MODULE_PATH" ]; then
    log "Loading evdi from ${MODULE_PATH}..."
    if insmod "$MODULE_PATH"; then
        log "evdi loaded successfully"
    else
        log "ERROR: insmod failed"
        exit 1
    fi
else
    log "ERROR: Could not find evdi module at ${MODULE_PATH}"
    exit 1
fi
