"""Rotating structured logging: console stays human-readable, file is JSON
and bounded in size regardless of how long a process runs."""

from __future__ import annotations

import json
import logging

import pytest


@pytest.fixture()
def fresh_firm_logger():
    """setup_logging() no-ops if the root logger already has handlers
    (module-level idempotency guard) — reset it so each test gets a clean
    run, and restore real state afterward so other tests aren't affected.

    pytest's own logging plugin re-attaches a LogCaptureHandler to the root
    logger between fixture setup and the test body running (it wraps the
    test *call*, not just collection), so clearing here alone isn't enough —
    each test must also clear immediately before invoking setup_logging().
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    root.handlers.clear()
    yield root
    root.handlers.clear()
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)


class TestSetupLogging:
    def test_console_handler_is_human_readable(self, fresh_firm_logger):
        from firm.logging_setup import setup_logging

        fresh_firm_logger.handlers.clear()  # pytest re-adds its own between setup and here
        setup_logging()
        console = fresh_firm_logger.handlers[0]
        record = logging.LogRecord(
            "firm.test", logging.INFO, __file__, 1, "hello world", None, None
        )
        formatted = console.format(record)
        assert "hello world" in formatted
        # file:function:line — needed to jump straight to the source of a
        # log line instead of grepping the message text for a landmark.
        assert f"{__file__.rsplit('/', 1)[-1]}:" in formatted
        # Not JSON — a plain line, not a '{...}' structure.
        with pytest.raises(json.JSONDecodeError):
            json.loads(formatted)

    def test_file_handler_emits_valid_json(self, fresh_firm_logger, tmp_path):
        from firm.logging_setup import setup_logging

        log_path = tmp_path / "test.log"
        fresh_firm_logger.handlers.clear()  # pytest re-adds its own between setup and here
        setup_logging(log_file=log_path)
        log = logging.getLogger("firm.test.module")
        log.info("a structured message")

        for h in fresh_firm_logger.handlers:
            h.flush()

        line = log_path.read_text().strip().splitlines()[-1]
        payload = json.loads(line)
        assert payload["msg"] == "a structured message"
        assert payload["logger"] == "firm.test.module"
        assert payload["level"] == "INFO"
        assert "ts" in payload
        # file:function:line — same traceability the console formatter gets.
        assert payload["file"] == "test_logging_setup.py"
        assert payload["function"] == "test_file_handler_emits_valid_json"
        assert isinstance(payload["line"], int)

    def test_file_rotation_bounds_disk_usage(self, fresh_firm_logger, tmp_path):
        from firm.logging_setup import setup_logging

        log_path = tmp_path / "rotating.log"
        fresh_firm_logger.handlers.clear()  # pytest re-adds its own between setup and here
        setup_logging(log_file=log_path, max_bytes=2000, backup_count=2)
        log = logging.getLogger("firm.test.rotation")
        for _ in range(500):
            log.info("padding message to force rotation " + "x" * 50)

        rotated = sorted(tmp_path.glob("rotating.log*"))
        # Base file + at most backup_count rotated files — never unbounded.
        assert 1 <= len(rotated) <= 3
        total_bytes = sum(p.stat().st_size for p in rotated)
        # Comfortably under (backup_count + 1) * max_bytes with slack for
        # the in-flight file exceeding max_bytes slightly before rotating.
        assert total_bytes < 3 * 2000 * 2
