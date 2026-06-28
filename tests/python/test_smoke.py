"""Smoke tests for the Python binding.

Confirms that the compiled ``vectorvault`` extension imports and reports a
well-formed version string.
"""

import re

import vectorvault


def test_extension_imports_and_reports_version():
    version = vectorvault.version()
    assert isinstance(version, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)


def test_module_version_attribute_matches():
    assert vectorvault.__version__ == vectorvault.version()
