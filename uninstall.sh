#!/usr/bin/env bash
# ----------------------------------------------------------------------
# uninstall.sh  -  remove everything install.sh created.
#
# POSIX counterpart to uninstall.bat. Removes:
#   1. Firewall rules for TCP 8000 / 50052 / 50053 / 8090 (best-effort
#      via ufw or firewall-cmd).
#   2. .venv/                  (Python virtualenv + every backend dep)
#   3. frontend/node_modules/  (every npm dep)
#   4. frontend/dist/          (production build output)
#
# Optionally also removes:
#   5. data/                   (chats, identity.json, memories, uploads,
#                              screenshots, etc.) — prompted before
#                              removal. Keeping it (default) lets you
#                              re-install later without losing chats
#                              or P2P device identity.
#
# Does NOT touch:
#   - The source tree.
#   - Anything outside this folder.
#
# Re-runnable: every step tolerates "already gone".
# ----------------------------------------------------------------------
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo
echo " ===================================================="
echo "  Gigachat - uninstall"
echo " ===================================================="
echo

# -- 1: firewall rules -----------------------------------------------
echo " [1/4] Removing firewall rules ..."
if command -v ufw >/dev/null 2>&1; then
  for port in 8000 50052 50053 8090; do
    sudo -n ufw delete allow "$port/tcp" >/dev/null 2>&1 || \
      sudo ufw delete allow "$port/tcp" || true
  done
elif command -v firewall-cmd >/dev/null 2>&1; then
  for port in 8000 50052 50053 8090; do
    sudo -n firewall-cmd --permanent --remove-port="$port/tcp" >/dev/null 2>&1 || \
      sudo firewall-cmd --permanent --remove-port="$port/tcp" || true
  done
  sudo -n firewall-cmd --reload >/dev/null 2>&1 || sudo firewall-cmd --reload || true
else
  echo "     (no supported firewall manager — skipping)"
fi

# -- 2: .venv --------------------------------------------------------
echo " [2/4] Removing .venv/ ..."
[ -d "$ROOT/.venv" ] && rm -rf "$ROOT/.venv"

# -- 3: frontend/node_modules ----------------------------------------
echo " [3/4] Removing frontend/node_modules/ ..."
[ -d "$ROOT/frontend/node_modules" ] && rm -rf "$ROOT/frontend/node_modules"

# -- 4: frontend/dist ------------------------------------------------
echo " [4/4] Removing frontend/dist/ ..."
[ -d "$ROOT/frontend/dist" ] && rm -rf "$ROOT/frontend/dist"

# -- Optional: data/ wipe (interactive) ------------------------------
echo
if [ -d "$ROOT/data" ]; then
  echo " Optional: also delete user data?"
  echo "   ./data/  contains chat history, paired-device identity,"
  echo "            memories, uploads, screenshots, audit log."
  echo "   Keeping it (default) lets you re-install later without"
  echo "   losing conversations or the P2P device identity peers"
  echo "   paired with."
  echo
  read -r -p "  Delete data/ too? [y/N] " WIPEDATA
  if [ "$WIPEDATA" = "y" ] || [ "$WIPEDATA" = "Y" ]; then
    rm -rf "$ROOT/data"
    echo " [+] data/ removed."
  else
    echo " [+] data/ kept (re-install later picks up where you left off)."
  fi
fi

echo
echo " ===================================================="
echo "  Uninstall complete."
echo
echo "  The source tree is intact. To re-install at any"
echo "  time, run:"
echo "    ./install.sh"
echo " ===================================================="
echo
