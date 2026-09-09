#!/usr/bin/env bash
# Swap /home/ebg/l34n/.env between the Ollama profile and the OpenRouter
# profile. Written for APE-708.
#
# Usage: scripts/switch-provider.sh {ollama|openrouter} [--dry-run]
#
# The active .env is backed up to .env.bak before overwrite.

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
target="${1:-}"
dry="${2:-}"

case "$target" in
  ollama|openrouter) ;;
  *)
    echo "usage: $0 {ollama|openrouter} [--dry-run]" >&2
    exit 2
    ;;
esac

src="$HERE/.env.$target"
dst="$HERE/.env"

if [[ ! -f "$src" ]]; then
  echo "profile file missing: $src" >&2
  exit 1
fi

if [[ "$dry" == "--dry-run" ]]; then
  echo "would copy $src -> $dst"
  exit 0
fi

if [[ -f "$dst" ]]; then
  cp "$dst" "$dst.bak"
fi
cp "$src" "$dst"
echo "activated $target profile ($dst)"
echo "reminder: restart 'ai-researcher serve' (or the systemd unit) to reload."
