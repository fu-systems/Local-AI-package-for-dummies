#!/bin/sh
# Toolshed installer for Linux. Per-user, no root, no package manager.
set -eu

APP_DIR="${HOME}/.local/share/toolshed/app"
BIN_DIR="${HOME}/.local/bin"
DESKTOP_DIR="${HOME}/.local/share/applications"
ICON_DIR="${HOME}/.local/share/icons/hicolor/256x256/apps"
SRC="$(cd "$(dirname "$0")" && pwd)"

ASSUME_YES=0
[ "${1:-}" = "--yes" ] && ASSUME_YES=1

say() { printf '%s\n' "$*"; }

# Probe by LIBRARY, not by package: this has to work on Arch, Fedora and
# openSUSE too, so only the remediation text names Debian package names.
missing=""
for lib in libGL.so.1 libEGL.so.1 libglib-2.0.so.0 libxkbcommon.so.0; do
    if ! ldconfig -p 2>/dev/null | grep -q "$lib"; then
        missing="${missing} ${lib}"
    fi
done
if [ -n "$missing" ]; then
    say "Some system libraries Toolshed needs are missing:${missing}"
    say ""
    say "On Ubuntu or Debian:"
    say "    sudo apt install libgl1 libegl1 libglib2.0-0 libxkbcommon0 libxcb-cursor0"
    say "On Fedora:  sudo dnf install mesa-libGL mesa-libEGL glib2 libxkbcommon xcb-util-cursor"
    say "On Arch:    sudo pacman -S libglvnd glib2 libxkbcommon xcb-util-cursor"
    say ""
    if [ "$ASSUME_YES" -eq 0 ]; then
        printf 'Carry on anyway? [y/N] '
        read -r reply
        case "$reply" in [Yy]*) ;; *) exit 1 ;; esac
    fi
fi

say "Installing Toolshed to ${APP_DIR}"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR" "$BIN_DIR" "$DESKTOP_DIR" "$ICON_DIR"
cp -a "$SRC/." "$APP_DIR/"
# Keep uninstall.sh. Removing it left no uninstaller anywhere on the machine
# once the tarball was gone, which made "uninstall" mean "delete some folders
# and hope", and is how half-removed installs got reinstalled on top of.
rm -f "$APP_DIR/install.sh"
chmod +x "$APP_DIR/toolshed" "$APP_DIR/uninstall.sh"

# A WRAPPER, not a symlink, and this matters.
#
# Qt gives QT_QPA_PLATFORM_PLUGIN_PATH precedence over the QT_PLUGIN_PATH that
# PyInstaller's runtime hook sets, and the dynamic loader searches
# LD_LIBRARY_PATH before DT_RUNPATH. A KDE, conda, Steam or ROCm environment
# that exports either will silently load a foreign Qt against our libraries and
# produce the notorious "Could not load the Qt platform plugin xcb ... even
# though it was found". Clearing them is the fix, and a symlink cannot do it.
cat > "$BIN_DIR/toolshed" <<'LAUNCHER'
#!/bin/sh
unset QT_QPA_PLATFORM_PLUGIN_PATH QT_PLUGIN_PATH QML2_IMPORT_PATH
unset LD_LIBRARY_PATH LD_PRELOAD
exec "$HOME/.local/share/toolshed/app/toolshed" "$@"
LAUNCHER
chmod +x "$BIN_DIR/toolshed"

# So removing it is a command you can find, not a file you have to still have.
cat > "$BIN_DIR/toolshed-uninstall" <<'UNINSTALLER'
#!/bin/sh
exec "$HOME/.local/share/toolshed/app/uninstall.sh" "$@"
UNINSTALLER
chmod +x "$BIN_DIR/toolshed-uninstall"

cp "$APP_DIR/toolshed.png" "$ICON_DIR/toolshed.png" 2>/dev/null || true
cp "$APP_DIR/toolshed.desktop" "$DESKTOP_DIR/toolshed.desktop" 2>/dev/null || true
update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true

say ""
say "Done. Launch it from your applications menu, or run: toolshed"
say "To remove it later:  toolshed-uninstall"
case ":${PATH}:" in
    *":${BIN_DIR}:"*) ;;
    *) say ""
       say "Note: ${BIN_DIR} is not on your PATH, so the 'toolshed' command"
       say "will not be found until you add it. The menu entry works regardless." ;;
esac
