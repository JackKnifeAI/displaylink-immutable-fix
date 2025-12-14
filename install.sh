#!/bin/bash
#  ╦╔═╗╔═╗╦╔═╦╔═╔╗╔╦╔═╗╔═╗  ╔═╗╦
#  ║╠═╣║  ╠╩╗╠╩╗║║║║╠╣ ║╣   ╠═╣║
# ╚╝╩ ╩╚═╝╩ ╩╩ ╩╝╚╝╩╚  ╚═╝  ╩ ╩╩    JACKKNIFE AI™
#
# Install DisplayLink fix for immutable Fedora (Silverblue/Aurora/Bazzite)
#
# Usage: sudo ./install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "     ██╗ █████╗  ██████╗██╗  ██╗██╗  ██╗███╗   ██╗██╗███████╗███████╗"
echo "     ██║██╔══██╗██╔════╝██║ ██╔╝██║ ██╔╝████╗  ██║██║██╔════╝██╔════╝"
echo "     ██║███████║██║     █████╔╝ █████╔╝ ██╔██╗ ██║██║█████╗  █████╗  "
echo "██   ██║██╔══██║██║     ██╔═██╗ ██╔═██╗ ██║╚██╗██║██║██╔══╝  ██╔══╝  "
echo "╚█████╔╝██║  ██║╚██████╗██║  ██╗██║  ██╗██║ ╚████║██║██║     ███████╗"
echo " ╚════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝     ╚══════╝"
echo "                            JACKKNIFE AI™"
echo ""
echo "Installing DisplayLink fix for immutable Fedora..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo ./install.sh"
    exit 1
fi

# Check if displaylink package is installed
if ! rpm -q displaylink &>/dev/null; then
    echo "ERROR: displaylink package not installed"
    echo "Install it first with: rpm-ostree install <displaylink-rpm-url>"
    exit 1
fi

# Install the loader script
echo "Installing evdi-loader.sh..."
install -m 755 "${SCRIPT_DIR}/evdi-loader.sh" /usr/local/bin/evdi-loader.sh

# Install the systemd override
echo "Installing systemd override..."
mkdir -p /etc/systemd/system/displaylink-driver.service.d
install -m 644 "${SCRIPT_DIR}/displaylink-driver.service.d/override.conf" \
    /etc/systemd/system/displaylink-driver.service.d/override.conf

# Add [Install] section to enable the service
echo "Configuring service to start on boot..."
cat > /etc/systemd/system/displaylink-driver.service.d/enable.conf << 'EOF'
[Install]
WantedBy=graphical.target
EOF

# Reload systemd
echo "Reloading systemd..."
systemctl daemon-reload

# Enable the service for autostart
echo "Enabling displaylink-driver service..."
systemctl enable displaylink-driver

# Build evdi for current kernel if needed
KERNEL=$(uname -r)
if [ ! -f "/var/lib/dkms/evdi/1.14.11/${KERNEL}/x86_64/module/evdi.ko" ] && \
   [ ! -f "/var/lib/dkms/evdi/1.14.11/${KERNEL}/x86_64/module/evdi.ko.xz" ]; then
    echo "Building evdi for kernel ${KERNEL}..."
    dkms autoinstall
fi

# Start the service
echo "Starting displaylink-driver service..."
systemctl restart displaylink-driver

# Check status
if systemctl is-active --quiet displaylink-driver; then
    echo ""
    echo "SUCCESS! DisplayLink is now running."
    echo "Check 'ls /sys/class/drm/' for your display."
else
    echo ""
    echo "WARNING: Service may not have started correctly."
    echo "Check: systemctl status displaylink-driver"
fi
