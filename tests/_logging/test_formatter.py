"""Unit tests for the JSON logging formatter.

``JSONFormatter`` is a pure ``logging.Formatter`` subclass: it turns a
``LogRecord`` into a JSON string. These tests build records by hand and assert
the parsed JSON, so they need no logging configuration or external services.
"""

from __future__ import annotations

import json
import logging
import sys

from autointent._logging.formatter import JSONFormatter


def _raise_value_error() -> None:
    error_message = "boom"
    raise ValueError(error_message)


def _record(msg: str = "hello world") -> logging.LogRecord:
    return logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=42,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_format_emits_message_and_timestamp() -> None:
    payload = json.loads(JSONFormatter().format(_record()))

    assert payload["message"] == "hello world"
    # timestamp is rendered as a UTC ISO-8601 string
    assert payload["timestamp"].endswith("+00:00")


def test_format_interpolates_message_args() -> None:
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="value is %s",
        args=("forty-two",),
        exc_info=None,
    )

    payload = json.loads(JSONFormatter().format(record))

    assert payload["message"] == "value is forty-two"


def test_fmt_keys_remap_and_consume_always_fields() -> None:
    formatter = JSONFormatter(fmt_keys={"level": "levelname", "logger": "name", "msg": "message"})

    payload = json.loads(formatter.format(_record()))

    # "level"/"logger" are pulled from record attributes
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    # "msg" maps to the always-field "message", which is then consumed
    assert payload["msg"] == "hello world"
    assert "message" not in payload
    # unmapped always-fields still appear
    assert "timestamp" in payload


def test_format_includes_exception_info() -> None:
    try:
        _raise_value_error()
    except ValueError:
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="failed",
            args=(),
            exc_info=sys.exc_info(),
        )

    payload = json.loads(JSONFormatter().format(record))

    assert "ValueError: boom" in payload["exc_info"]


def test_format_includes_stack_info() -> None:
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="with stack",
        args=(),
        exc_info=None,
        sinfo="Stack (most recent call last):\n  fake frame",
    )

    payload = json.loads(JSONFormatter().format(record))

    assert "Stack (most recent call last):" in payload["stack_info"]


def test_format_includes_extra_record_attributes() -> None:
    record = _record()
    record.custom_field = "extra-value"

    payload = json.loads(JSONFormatter().format(record))

    assert payload["custom_field"] == "extra-value"
