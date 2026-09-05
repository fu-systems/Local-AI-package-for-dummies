#!/bin/sh
# Removes Toolshed itself. Never touches models, outputs, or anything the user made.
set -eu

APP_DIR="${HOME}/.local/share/toolshed/app"
BIN_DIR="${HOME}/.local/bin"
DESKTOP_DIR="${HOME}/.local/share/applications"
ICON_DIR="${HOME}/.local/share/icons/hicolor/256x256/apps"

rm -rf "$APP_DIR"
rm -f "$BIN_DIR/toolshed" "$DESKTOP_DIR/toolshed.desktop" "$ICON_DIR/toolshed.png"
rmdir "${HOME}/.local/share/toolshed" 2>/dev/null || true
update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true

printf '%s\n' "Toolshed has been removed."
printf '%s\n' ""
printf '%s\n' "Your models and the things you made were NOT deleted. They are still in"
printf '%s\n' "the folder you chose during setup. Delete it yourself if you want the space."
