"""
catan_api.py - backwards-compatible entry point.

The application lives in ``main.py``. This module keeps ``uvicorn catan_api:app``
working for existing setups.
"""

from main import app, router

__all__ = ["app", "router"]
