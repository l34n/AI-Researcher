#!/usr/bin/env bash
# Install the dashboard and its hourly ingest as systemd --user units.
# User services, not system ones: everything lives in your home directory and
# needs no root.
set -euo pipefail

# Headless shells (ssh without a login session, agent runners, cron) reach the
# user bus only when these are set. Default them to the conventional runtime
# dir so `systemctl --user` works everywhere a user bus actually exists.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"
if [[ ! -S "$XDG_RUNTIME_DIR/bus" ]]; then
  echo "error: no user bus at $XDG_RUNTIME_DIR/bus." >&2
  echo "       log in once (or: sudo loginctl enable-linger $USER) and retry." >&2
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if [[ ! -x "$PROJECT_DIR/.venv/bin/ai-researcher" ]]; then
  echo "error: $PROJECT_DIR/.venv/bin/ai-researcher not found." >&2
  echo "       run:  uv venv && uv pip install -e ." >&2
  exit 1
fi

# %h expands to the *service* home, which is what we want, but the units also
# hardcode the project as ~/AI-Researcher. Rewrite if it lives elsewhere.
mkdir -p "$UNIT_DIR"
for unit in ai-researcher.service ai-researcher-ingest.service ai-researcher-ingest.timer; do
  sed "s|%h/AI-Researcher|$PROJECT_DIR|g" "$PROJECT_DIR/systemd/$unit" > "$UNIT_DIR/$unit"
  echo "  installed $UNIT_DIR/$unit"
done

systemctl --user daemon-reload
systemctl --user enable --now ai-researcher.service
systemctl --user enable --now ai-researcher-ingest.timer

# Without lingering, user services stop at logout and never start at boot.
if ! loginctl show-user "$USER" --property=Linger 2>/dev/null | grep -q "Linger=yes"; then
  echo
  echo "  NOTE: enabling linger so the dashboard survives logout and starts at boot:"
  echo "        sudo loginctl enable-linger $USER"
fi

PORT="$(grep -E '^AIR_PORT=' "$PROJECT_DIR/.env" 2>/dev/null | cut -d= -f2)"
PORT="${PORT:-8899}"
IP="$(hostname -I 2>/dev/null | awk '{print $1}')"

echo
echo "  dashboard : http://localhost:$PORT"
[[ -n "$IP" ]] && echo "  on LAN    : http://$IP:$PORT"
echo
echo "  status    : systemctl --user status ai-researcher"
echo "  logs      : journalctl --user -u ai-researcher -f"
echo "  next run  : systemctl --user list-timers ai-researcher-ingest.timer"
echo "  run now   : systemctl --user start ai-researcher-ingest.service"
