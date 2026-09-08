"""Multi-bar chart pattern detection (Head & Shoulders, triangles, flags,
cup & handle, ...) — feeds Strategy #13 (``pattern_recognition``).

See docs/pattern_recognition_plan.md for the design rationale and status.
"""

from __future__ import annotations

from firm.patterns.match import PatternMatch
from firm.patterns.scanner import scan_symbol

__all__ = ["PatternMatch", "scan_symbol"]
