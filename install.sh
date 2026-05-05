#!/usr/bin/env bash
# ----------------------------------------------------------------------
# install.sh  -  one-shot Gigachat setup for Linux + macOS.
#
# POSIX counterpart to install.bat. Does end-to-end:
#   1. Creates a Python virtualenv at .venv/ and installs backend deps.
#   2. npm-installs the frontend and builds the production bundle.
#   3. Fetches the patched llama.cpp build IF the host is a supported
#      platform (Windows x64 only today — no-op on Linux / macOS until
#      we publish builds for those; the source patch in
#      vendor/llama.cpp-patches/ is buildable on any platform).
#   4. Best-effort firewall rules for TCP 8000 / 50052 / 50053 / 8090
#      (P2P backend + rpc-server SYCL/CUDA + rpc-server CPU + peer-
#      orchestrated llama-server). Tries ufw first, then firewall-cmd,
#      then prints a manual fallback.
#
# Re-runnable: every step is idempotent. Run again after `git pull` to
# refresh deps + rebuild the frontend bundle.
#
# Uninstall: ./uninstall.sh
#
# Requires: Python 3.12+, Node 20+. Ollama is auto-started by the
# backend when installed; install from https://ollama.com/download.
# ----------------------------------------------------------------------
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo
echo " ===================================================="
echo "  Gigachat - one-shot install"
echo " ===================================================="
echo

# Pick the python executable. Prefer python3.12 explicitly when present
# (Ubuntu 22.04 still ships 3.10 as `python3` by default). Falls
# through to whatever `python3` resolves to on PATH otherwise.
if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN=python3.12
elif command -v python3.13 >/dev/null 2>&1; then
  PYTHON_BIN=python3.13
elif command -v python3.14 >/dev/null 2>&1; then
  PYTHON_BIN=python3.14
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  echo " [!] python3 not found on PATH. Install Python 3.12+ first." >&2
  exit 1
fi
echo " Using $PYTHON_BIN ($($PYTHON_BIN --version))"

# -- 1. Python venv ---------------------------------------------------
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo " [1/5] Creating Python virtualenv at .venv/ ..."
  "$PYTHON_BIN" -m venv .venv
else
  echo " [1/5] Reusing existing .venv/ ..."
fi

# -- 2. Backend deps --------------------------------------------------
echo " [2/5] Installing backend dependencies into .venv/ ..."
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/python" -m pip install -r backend/requirements.txt

# -- 3. Frontend deps + production build ------------------------------
echo " [3/5] Installing + building frontend ..."
if ! command -v npm >/dev/null 2>&1; then
  echo " [!] npm not found. Install Node 20+ first." >&2
  exit 1
fi
( cd "$ROOT/frontend" && npm install && npm run build )

# -- 3.5. Patched llama.cpp build (one-shot fetch from GitHub Release) -
# Same auto-fetch as the .bat installer. Today only Windows x64
# binaries are published; the function returns platform_supported=False
# on Linux / macOS and the install continues — single-device chat
# still works, only split-mode resilience is degraded.
echo " [3.5/5] Fetching patched llama.cpp build (Windows x64 only today; no-op elsewhere) ..."
"$ROOT/.venv/bin/python" -c "from backend.p2p_llama_server import fetch_patched_llama_cpp; r = fetch_patched_llama_cpp(); print('  result:', r)" || true

# -- 4. Firewall rules ------------------------------------------------
# Best-effort: open every port the compute pool needs. Falls through to
# a plain instructional print on systems without ufw / firewall-cmd —
# Linux distros vary too much to enumerate every option.
echo " [4/5] Adding firewall rules (TCP 8000 / 50052 / 50053 / 8090) ..."
if command -v ufw >/dev/null 2>&1; then
  for port in 8000 50052 50053 8090; do
    sudo -n ufw allow "$port/tcp" >/dev/null 2>&1 || \
      sudo ufw allow "$port/tcp" || true
  done
  echo " [+] ufw rules added."
elif command -v firewall-cmd >/dev/null 2>&1; then
  for port in 8000 50052 50053 8090; do
    sudo -n firewall-cmd --permanent --add-port="$port/tcp" >/dev/null 2>&1 || \
      sudo firewall-cmd --permanent --add-port="$port/tcp" || true
  done
  sudo -n firewall-cmd --reload >/dev/null 2>&1 || sudo firewall-cmd --reload || true
  echo " [+] firewalld rules added + reloaded."
else
  echo " [-] No supported firewall manager detected (ufw / firewall-cmd)."
  echo "     If your distro uses a different firewall, manually allow inbound TCP"
  echo "     8000 / 50052 / 50053 / 8090 from the LAN profile only. Loopback-only"
  echo "     usage (single-device chat from this machine) needs no firewall changes."
fi

# -- Done -------------------------------------------------------------
echo
echo " ===================================================="
echo "  Setup complete."
echo
echo "  Now run ONE of:"
echo "    ./dev.sh     development (Vite hot-reload, port 5173)"
echo "    ./start.sh   production single-port (http://localhost:8000)"
echo
echo "  To remove everything (firewall rules, .venv, node_modules,"
echo "  frontend build), run:"
echo "    ./uninstall.sh"
echo " ===================================================="
echo
