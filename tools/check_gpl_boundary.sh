#!/bin/sh
# ComfyUI is GPL-3.0-or-later. Toolshed downloads it and drives it as a separate
# process over HTTP, and must never import it, bundle it, or place our code
# under custom_nodes/. See CONTRIBUTING.md.
#
# This lives in a script rather than inline in ci.yml so that it can be tested.
# It is a safety check, and a safety check nobody can exercise is a wish.
#
# Two things are looked for:
#
#   1. an import of ComfyUI anywhere in toolshed/
#   2. ComfyUI named in anything under packaging/, outside a comment
#
# The comment exemption is deliberate and narrow. Bundling ComfyUI would
# redistribute GPL code, which is what this exists to prevent; *writing down*
# that we download it, drive it, or delete it is documentation, and a check
# that forbids describing the boundary makes the boundary harder to explain.
# Comments are recognised per file type -- '#' is a preprocessor directive in
# Inno Setup, not a comment, so a `#define` naming ComfyUI there is still
# caught.
set -eu

fail() {
  echo "::error::$1"
  exit 1
}

# Guard the guard. Without this, a missing toolshed/ makes grep exit 2, an `if`
# reads that as false, and the check passes having examined nothing -- which is
# exactly what it did once.
[ -d toolshed ] || fail "toolshed/ package is missing"

if grep -rnE '^\s*(from|import)\s+(comfy|nodes|folder_paths)\b' \
     --include='*.py' toolshed/; then
  fail "Toolshed must not import ComfyUI. See CONTRIBUTING.md."
fi

if [ -d packaging ]; then
  offenders=""
  for file in $(find packaging -type f); do
    case "${file}" in
      *.iss) comment='(;|//)' ;;   # '#' starts a preprocessor directive here
      *)     comment='#' ;;
    esac
    hits=$(grep -niE 'comfyui' "${file}" \
             | grep -vE "^[0-9]+:[[:space:]]*${comment}" || true)
    if [ -n "${hits}" ]; then
      offenders="${offenders}${file}
${hits}
"
    fi
  done
  if [ -n "${offenders}" ]; then
    printf '%s\n' "${offenders}"
    fail "ComfyUI referenced in packaging outside a comment; it must never be bundled."
  fi
fi

echo "GPL boundary intact"
