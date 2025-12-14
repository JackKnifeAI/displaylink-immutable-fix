```
     ██╗ █████╗  ██████╗██╗  ██╗██╗  ██╗███╗   ██╗██╗███████╗███████╗
     ██║██╔══██╗██╔════╝██║ ██╔╝██║ ██╔╝████╗  ██║██║██╔════╝██╔════╝
     ██║███████║██║     █████╔╝ █████╔╝ ██╔██╗ ██║██║█████╗  █████╗
██   ██║██╔══██║██║     ██╔═██╗ ██╔═██╗ ██║╚██╗██║██║██╔══╝  ██╔══╝
╚█████╔╝██║  ██║╚██████╗██║  ██╗██║  ██╗██║ ╚████║██║██║     ███████╗
 ╚════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝     ╚══════╝
                             █████╗ ██╗
                            ██╔══██╗██║
                            ███████║██║
                            ██╔══██║██║
                            ██║  ██║██║
                            ╚═╝  ╚═╝╚═╝
                                     JACKKNIFE AI™
```

# DisplayLink Fix for Immutable Fedora

Fixes DisplayLink USB displays (like Mobile Pixels Duex Pro) on immutable Fedora variants:
- Fedora Silverblue
- Fedora Kinoite
- Universal Blue Aurora
- Universal Blue Bazzite

## The Problem

On immutable Fedora, DKMS builds the `evdi` kernel module but **cannot install it** to the read-only `/lib/modules` directory. This causes the `displaylink-driver.service` to fail because `modprobe evdi` can't find the module.

```
modprobe: FATAL: Module evdi not found in directory /lib/modules/6.x.x-xxx.fc4x.x86_64
```

## The Solution

This fix:
1. **Loads evdi directly from the DKMS build path** (`/var/lib/dkms/evdi/...`)
2. **Auto-rebuilds on kernel updates** via `dkms autoinstall`
3. **Integrates with systemd** so DisplayLink starts automatically on boot
4. **Enables the service** with proper systemd configuration for persistence across reboots

## Installation

### Prerequisites

1. Install the displaylink package:
```bash
# Download the latest from https://github.com/displaylink-rpm/displaylink-rpm/releases
rpm-ostree install ./fedora-43-displaylink-*.rpm
# Reboot
systemctl reboot
```

2. After reboot, build evdi:
```bash
sudo dkms autoinstall
```

### Install the Fix

```bash
git clone https://github.com/JackKnifeAI/displaylink-immutable-fix.git
cd displaylink-immutable-fix
sudo ./install.sh
```

### Manual Installation

```bash
# Copy the loader script
sudo cp evdi-loader.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/evdi-loader.sh

# Create systemd overrides
sudo mkdir -p /etc/systemd/system/displaylink-driver.service.d
sudo cp displaylink-driver.service.d/override.conf \
    /etc/systemd/system/displaylink-driver.service.d/

# Enable autostart on boot
sudo bash -c 'cat > /etc/systemd/system/displaylink-driver.service.d/enable.conf << EOF
[Install]
WantedBy=graphical.target
EOF'

# Reload, enable, and start
sudo systemctl daemon-reload
sudo systemctl enable displaylink-driver
sudo systemctl restart displaylink-driver
```

## Verify It Works

```bash
# Check service status
systemctl status displaylink-driver

# Check for new display
ls /sys/class/drm/
# Should show card0-DVI-I-1 or similar for your USB display
```

## After Kernel Updates

The fix handles kernel updates automatically. On first boot after a kernel update:
1. `evdi-loader.sh` detects no module for the new kernel
2. Runs `dkms autoinstall` to build it
3. Loads the freshly built module
4. DisplayLink starts normally

## Troubleshooting

### Service doesn't start on boot

Check if the service is enabled:
```bash
systemctl is-enabled displaylink-driver
```

If it shows "disabled" or "static", enable it:
```bash
sudo systemctl enable displaylink-driver
```

Verify the enable.conf file exists:
```bash
cat /etc/systemd/system/displaylink-driver.service.d/enable.conf
```

Should contain:
```
[Install]
WantedBy=graphical.target
```

### evdi module not loading

Check DKMS status:
```bash
dkms status | grep evdi
```

If not built for current kernel, rebuild:
```bash
sudo dkms autoinstall
```

Manually test the loader:
```bash
sudo /usr/local/bin/evdi-loader.sh
```

### After reboot, still not working

1. Check service status:
```bash
systemctl status displaylink-driver
```

2. Check module:
```bash
lsmod | grep evdi
```

3. Check logs:
```bash
journalctl -u displaylink-driver -b
```

## How It Works

### evdi-loader.sh
- Checks if evdi is already loaded (skips if so)
- Checks if module exists for current kernel
- If not, runs `dkms autoinstall` to build it
- Decompresses `.ko.xz` if needed
- Loads module via `insmod` from DKMS path

### systemd override
- Clears the default `ExecStartPre=/sbin/modprobe evdi` (which fails)
- Replaces with `ExecStartPre=/usr/local/bin/evdi-loader.sh`

## Tested On

- [x] Aurora (Universal Blue) - Fedora 43
- [x] Mobile Pixels Duex Pro (DisplayLink vendor 17e9)

## Contributing

PRs welcome! This fix should ideally be integrated into:
- Universal Blue's `ujust enable-displaylink` command
- The displaylink-rpm package itself

## License

MIT

## Credits

- **JACKKNIFE AI™** - Alexander Casavant
- Fix developed with assistance from Claude (Anthropic)
- DisplayLink RPM maintainers: https://github.com/displaylink-rpm/displaylink-rpm
- Universal Blue team: https://universal-blue.org

---
```
 ╔═╗╦ ╦╔╦╗  ╔╦╗╦ ╦╔═╗  ╔╗ ╦ ╦╦  ╦  ╔═╗╦ ╦╦╔╦╗
 ║  ║ ║ ║    ║ ╠═╣║╣   ╠╩╗║ ║║  ║  ╚═╗╠═╣║ ║
 ╚═╝╚═╝ ╩    ╩ ╩ ╩╚═╝  ╚═╝╚═╝╩═╝╩═╝╚═╝╩ ╩╩ ╩
```
