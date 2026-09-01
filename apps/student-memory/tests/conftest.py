"""Pytest global configuration ensuring unit and integration test suites run safely."""

from __future__ import annotations

import os

# Default MEMORY_AUTH_ENABLED to false for non-auth tests unless explicitly specified
if "MEMORY_AUTH_ENABLED" not in os.environ:
    os.environ["MEMORY_AUTH_ENABLED"] = "false"
