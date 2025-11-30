# Bug Report: flatpak-preinstall fails on Aurora-DX latest - Bazaar package missing

## Summary
`flatpak-preinstall.service` continuously fails and respawns on Aurora-DX latest because it's trying to install `io.github.kolunmi.Bazaar` which no longer exists in the fedora flatpak remote.

## Environment
- **Image**: `ghcr.io/ublue-os/aurora-dx:latest`
- **Version**: `latest-43.20251129.1`
- **Date**: 2025-11-29

## Steps to Reproduce
1. Rebase to `ghcr.io/ublue-os/aurora-dx:latest`
2. Reboot
3. Check service status: `systemctl status flatpak-preinstall.service`

## Expected Behavior
Flatpak preinstall should complete successfully or gracefully skip unavailable packages.

## Actual Behavior
Service enters restart loop, failing every ~30 seconds:

```
● flatpak-preinstall.service - Preinstall Flatpaks
     Active: activating (auto-restart) (Result: exit-code)
    Process: ExecStart=/usr/bin/flatpak preinstall -y (code=exited, status=1/FAILURE)
```

Journal shows:
```
error: No such ref 'app/io.github.kolunmi.Bazaar/x86_64/stable' in remote fedora
```

Service has restarted 300+ times, spamming logs.

## Root Cause
`/usr/share/flatpak/preinstall.d/bazaar.preinstall` references a package that no longer exists:

```
[Flatpak Preinstall io.github.kolunmi.Bazaar]
Branch=stable
IsRuntime=false
```

Verification:
```bash
flatpak remote-ls fedora --app | grep -i bazaar
# Returns nothing - package doesn't exist
```

## Impact
- Constant service restart spam in logs
- May prevent other flatpak operations
- Users may think system is broken (it kind of is)

## Workaround
```bash
sudo systemctl mask flatpak-preinstall.service
```

## Suggested Fix
1. Remove `/usr/share/flatpak/preinstall.d/bazaar.preinstall` from the image
2. Or update to use flathub remote if package moved there
3. Consider adding error handling to skip unavailable packages gracefully

## Additional Context
This affects fresh rebases to Aurora-DX latest. Users on mobile data or with slow connections will have degraded experience due to constant retry spam.

---
Reported by: **JACKKNIFE AI™**
