#!/usr/bin/env bash
# ----------------------------------------------------------------------
# scripts/test.sh — local smoke-test runner.
#
# Same command CI runs (`pytest -m smoke ... -x`), so a green local run
# is a strong signal a push won't go red. Designed to double as the
# body of a `pre-push` git hook (see .githooks/pre-push) so a forgotten
# `pytest` doesn't ship a regression.
#
# Exit code propagates from pytest: 0 = pass, non-zero = fail. The
# pre-push hook short-circuits the push when this script exits non-zero.
#
# Usage:
#   scripts/test.sh           # run smoke tier on the current Python
#   scripts/test.sh -v        # extra args forwarded to pytest verbatim
# ----------------------------------------------------------------------

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec python -m pytest -m smoke --no-header -q --tb=short -x "$@"
