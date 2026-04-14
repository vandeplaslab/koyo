"""Tests for koyo.audit."""

from contextlib import redirect_stdout
from io import StringIO

from koyo.audit import open_file_audit_hook


def test_open_file_audit_hook_ignores_other_events():
    stream = StringIO()
    with redirect_stdout(stream):
        open_file_audit_hook("import", "module")
    assert stream.getvalue() == ""


def test_open_file_audit_hook_reports_open_call_stack():
    def nested_call():
        open_file_audit_hook("open", "example.txt", "r")

    stream = StringIO()
    with redirect_stdout(stream):
        nested_call()

    output = stream.getvalue()
    assert "open example.txt r was called:" in output
    assert "Function nested_call" in output
    assert "test_open_file_audit_hook_reports_open_call_stack" in output
