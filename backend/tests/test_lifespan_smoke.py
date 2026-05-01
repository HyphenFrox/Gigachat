"""Regression: backend lifespan startup must not crash with NameError /
AttributeError on missing module-level state.

This exists because the rest of the test suite uses FastAPI's
TestClient via direct request handlers — TestClient does NOT execute
the app's lifespan context manager. So a forgotten `import os` or a
missing `log = logging.getLogger(...)` at the top of `app.py` slips
past every other test and only blows up when a real uvicorn boots.
We learned that the hard way; this test pins it.

We don't assert that the lifespan completes successfully — that
depends on Ollama being installed, the rendezvous being reachable,
ports being free, etc. We ONLY assert that the FAILURE MODE isn't
NameError / AttributeError, which would mean a missing import or a
typo'd identifier. Any other exception (network refused, port
collision, etc.) is accepted as a real-environment failure outside
this test's scope.
"""
from __future__ import annotations

import asyncio

import pytest

from backend.app import app, lifespan


pytestmark = pytest.mark.smoke


def test_lifespan_has_no_module_level_name_errors():
    """Drive the lifespan once. The only failure modes we reject are
    NameError and AttributeError — both indicate a forgotten import
    or a typo that would crash a real uvicorn boot."""
    async def _drive():
        try:
            async with lifespan(app):
                pass
        except (NameError, AttributeError) as e:
            pytest.fail(
                f"lifespan startup raised {type(e).__name__}: {e}. "
                "This means a module-level identifier is missing or "
                "misspelled — a forgotten `import` at the top of "
                "backend/app.py is the most common cause."
            )
        except Exception:
            # Any non-NameError exception is acceptable here. CI may
            # not have Ollama running; the rendezvous may be down;
            # the test box may not have a free port — none of those
            # are within scope of this smoke test. Real boots will
            # surface those errors via their own logging path.
            pass

    asyncio.run(_drive())
