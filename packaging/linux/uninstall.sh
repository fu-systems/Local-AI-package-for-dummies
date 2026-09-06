#!/bin/sh
# Remove Toolshed completely, keeping only the models.
#
# What survives: <root>/models -- the folders and the model files inside them,
# untouched, because they are tens of gigabytes and re-downloading them is the
# slowest part of any reinstall.
#
# What goes: everything else, everywhere. The private Python workspace, the
# ComfyUI engine, its user directory and database, injected workflows, the
# manifest, logs, part-finished downloads, engine options, outputs, the app
# itself, the launcher, the desktop entry and the icon.
#
# This used to remove only the app and leave the whole data root behind, so a
# reinstall landed on top of a half-finished one and inherited its problems --
# a venv built by an interrupted run, a manifest describing files that were no
# longer there, an engine tree from a different version.
set -eu

# Run from a copy in a temporary file.
#
# This script lives inside the app directory it is about to delete, and a POSIX
# shell reads a script incrementally as it executes it -- so removing the file
# out from under itself can leave the remaining lines unread, half-done. Which
# is the exact class of problem this whole script exists to clean up.
if [ -z "${TOOLSHED_UNINSTALL_RELOCATED:-}" ]; then
  copy=$(mktemp "${TMPDIR:-/tmp}/toolshed-uninstall.XXXXXX") || exit 1
  cat "$0" > "${copy}"
  TOOLSHED_UNINSTALL_RELOCATED=1
  # Hand down our own pid. The shell that launched this script was started as
  # ".../toolshed/app/uninstall.sh", so it matches the app directory pattern
  # below and the copy would otherwise terminate the very shell waiting on it
  # -- the work finishes, and the caller still sees it killed by a signal.
  TOOLSHED_UNINSTALL_PARENT=$$
  export TOOLSHED_UNINSTALL_RELOCATED TOOLSHED_UNINSTALL_PARENT
  sh "${copy}" "$@"
  status=$?
  rm -f "${copy}"
  exit ${status}
fi

ROOT="${TOOLSHED_ROOT:-${HOME}/Toolshed}"
# A trailing slash would turn the "is this / or $HOME" checks below into
# string comparisons that never match, and "$ROOT/" is exactly what a shell's
# tab completion produces.
ROOT="${ROOT%/}"
APP_DIR="${HOME}/.local/share/toolshed/app"
BIN_DIR="${HOME}/.local/bin"
DESKTOP_DIR="${HOME}/.local/share/applications"
ICON_DIR="${HOME}/.local/share/icons/hicolor/256x256/apps"

KEEP="models"

say() { printf '%s\n' "$*"; }

# --- refuse to run somewhere catastrophic -----------------------------------
#
# This script deletes directory trees. If HOME were empty or ROOT were / or the
# home directory itself, the commands below would be a disaster rather than an
# uninstall. Checked explicitly rather than trusted.
case "${ROOT}" in
  ""|"/"|"/*"|".")
    say "Refusing to uninstall: '${ROOT}' is not a sensible place for Toolshed's data."
    exit 1
    ;;
esac
if [ -z "${HOME:-}" ] || [ "${ROOT}" = "${HOME}" ] || [ "${ROOT}" = "${HOME}/" ]; then
  say "Refusing to uninstall: the data folder must not be your home directory."
  exit 1
fi

# --- stop anything still running out of these folders ------------------------
#
# Deleting the workspace from under a running engine leaves exactly the
# half-state this script exists to prevent. Processes are matched by the
# absolute path they were launched from, never by name, so this can only ever
# match our own.
ours() {
  [ -d /proc ] || return 0
  for pid_dir in /proc/[0-9]*; do
    pid=${pid_dir#/proc/}
    [ "${pid}" = "$$" ] && continue
    [ "${pid}" = "${TOOLSHED_UNINSTALL_PARENT:-}" ] && continue
    cmd=$(tr '\0' ' ' < "${pid_dir}/cmdline" 2>/dev/null) || continue
    case "${cmd}" in
      *"${ROOT}/engine/"*|*"${ROOT}/runtime/"*|*"${APP_DIR}/"*) printf '%s\n' "${pid}" ;;
    esac
  done
}

stop_ours() {
  running=$(ours)
  [ -n "${running}" ] || return 0        # nothing of ours is up; do not wait
  for pid in ${running}; do
    say "Stopping something still running: pid ${pid}"
    kill "${pid}" 2>/dev/null || true
  done
  # Give them a moment to go quietly, then insist.
  sleep 2
  for pid in $(ours); do
    kill -9 "${pid}" 2>/dev/null || true
  done
}
stop_ours

# --- the data root ----------------------------------------------------------
if [ -d "${ROOT}" ]; then
  if [ -d "${ROOT}/${KEEP}" ]; then
    # find, not a shell glob: a glob skips dotfiles, and this has to leave
    # nothing behind. -mindepth 1 keeps the root itself; -maxdepth 1 means a
    # symlink is unlinked rather than followed into.
    find "${ROOT}" -mindepth 1 -maxdepth 1 ! -name "${KEEP}" -exec rm -rf {} +
    say "Kept ${ROOT}/${KEEP} and everything in it."
  else
    rm -rf "${ROOT}"
    say "Removed ${ROOT} entirely; there were no models in it."
  fi
fi

# --- the app itself ---------------------------------------------------------
rm -rf "${APP_DIR}"
rm -f "${BIN_DIR}/toolshed" \
      "${BIN_DIR}/toolshed-uninstall" \
      "${DESKTOP_DIR}/toolshed.desktop" \
      "${ICON_DIR}/toolshed.png"
rm -rf "${HOME}/.local/share/toolshed"
update-desktop-database "${DESKTOP_DIR}" 2>/dev/null || true

say ""
say "Toolshed has been removed."
if [ -d "${ROOT}/${KEEP}" ]; then
  say "Your models are still in ${ROOT}/${KEEP}. Nothing else was left anywhere."
  say "Delete that folder too if you want the disk space back."
else
  say "Nothing was left anywhere."
fi
