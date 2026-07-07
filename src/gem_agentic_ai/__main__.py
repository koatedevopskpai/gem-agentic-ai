"""
__main__.py — Enables `python -m gem_agentic_ai`.

Python looks for this file when a package is run with the -m flag.
It simply delegates to the CLI entry point in main.py.

Usage:
    python -m gem_agentic_ai demo
    python -m gem_agentic_ai chat
    python -m gem_agentic_ai simple --verbose
"""

from gem_agentic_ai.main import main

main()
