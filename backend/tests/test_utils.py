"""
Basic tests for utility functions
"""
import unittest
import sys
import os
import json
import tempfile
import pickle
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.datetime_utils import get_utc_now, format_utc_datetime, get_utc_timestamp
from utils.file_handler import process_pickle_file


class TestDateTimeUtils(unittest.TestCase):
    """Test datetime utility functions"""
    
    def test_get_utc_now(self):
        """Test get_utc_now returns datetime"""
        result = get_utc_now()
        self.assertIsInstance(result, datetime)
    
    def test_get_utc_timestamp(self):
        """Test get_utc_timestamp returns float"""
        result = get_utc_timestamp()
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)
    
    def test_format_utc_datetime_default(self):
        """Test format_utc_datetime with default (current time)"""
        result = format_utc_datetime()
        self.assertIsInstance(result, str)
        # ISO format should contain 'T'
        self.assertIn('T', result)
    
    def test_format_utc_datetime_with_value(self):
        """Test format_utc_datetime with specific datetime"""
        dt = datetime(2023, 12, 25, 15, 30, 45)
        result = format_utc_datetime(dt)
        self.assertIsInstance(result, str)
        self.assertIn('2023', result)
        self.assertIn('12', result)
        self.assertIn('25', result)


class TestStringUtils(unittest.TestCase):
    """Test basic string utilities"""
    
    def test_string_truncation_logic(self):
        """Test basic truncation logic"""
        text = "This is a very long text that should be truncated"
        max_length = 20
        
        if len(text) > max_length:
            truncated = text[:max_length] + '...'
        else:
            truncated = text
        
        self.assertEqual(len(truncated), 23)  # 20 + "..."
        self.assertTrue(truncated.endswith('...'))
    
    def test_file_size_calculation(self):
        """Test file size calculation logic"""
        # Test bytes
        size = 500
        self.assertLess(size, 1024)
        
        # Test KB
        size_kb = 1500
        kb_value = size_kb / 1024.0
        self.assertGreater(kb_value, 1.0)
        self.assertLess(kb_value, 1024.0)


class TestProcessPickleFileSecurity(unittest.TestCase):
    """
    Security regression tests for CWE-502 (Deserialization of Untrusted Data).

    The original process_pickle_file() used pickle.load(), which executes
    arbitrary Python code embedded in a serialized payload.  The remediation
    replaces that sink with json.load(), which cannot execute code.

    These tests verify:
      1. Legitimate JSON data files are parsed correctly (functionality preserved).
      2. A raw pickle payload is rejected — pickle bytes are not valid JSON.
      3. A malicious pickle payload that would run code via __reduce__ is rejected.
      4. Corrupt / empty files return a safe error dict, not an exception.
    """

    def setUp(self):
        """Create a temporary directory for test files."""
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove temporary files after each test."""
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_file(self, name, content, mode='w'):
        path = os.path.join(self.tmp_dir, name)
        with open(path, mode) as fh:
            fh.write(content)
        return path

    def _write_binary_file(self, name, data):
        path = os.path.join(self.tmp_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        return path

    # ------------------------------------------------------------------
    # Positive / functionality tests
    # ------------------------------------------------------------------

    def test_valid_json_dict_is_parsed(self):
        """process_pickle_file should parse a valid JSON object file."""
        payload = {"key": "value", "count": 42, "flag": True}
        path = self._write_file('data.pkl', json.dumps(payload))
        result = process_pickle_file(path)
        self.assertEqual(result, payload)

    def test_valid_json_list_is_parsed(self):
        """process_pickle_file should parse a valid JSON array file."""
        payload = [1, "two", 3.0, None]
        path = self._write_file('data.pkl', json.dumps(payload))
        result = process_pickle_file(path)
        self.assertEqual(result, payload)

    def test_valid_json_nested_structure(self):
        """process_pickle_file should handle nested JSON structures."""
        payload = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
        path = self._write_file('data.pkl', json.dumps(payload))
        result = process_pickle_file(path)
        self.assertEqual(result['users'][0]['name'], 'Alice')
        self.assertEqual(result['users'][1]['id'], 2)

    # ------------------------------------------------------------------
    # Security / regression tests
    # ------------------------------------------------------------------

    def test_raw_pickle_bytes_are_rejected(self):
        """
        A file containing raw pickle bytes must NOT be loaded successfully.

        pickle.dumps() produces bytes that are not valid JSON.  json.load()
        should raise a decoding error, which process_pickle_file catches and
        returns as {'error': ...}.  This confirms the pickle sink is gone.
        """
        # Serialize a harmless dict with pickle to simulate an attacker payload
        pickle_bytes = pickle.dumps({"should": "not be loaded"})
        path = self._write_binary_file('evil.pkl', pickle_bytes)
        result = process_pickle_file(path)
        # Must return an error dict, never the deserialized object
        self.assertIn('error', result)
        # The deserialized value must not appear in the result
        self.assertNotIn('should', result)

    def test_malicious_pickle_with_reduce_is_rejected(self):
        """
        A pickle payload using __reduce__ (classic RCE vector) must be rejected.

        Under the old pickle.load() implementation this payload would execute
        os.system('id') at deserialization time.  With json.load() the bytes
        are not valid JSON and parsing fails harmlessly.
        """
        class MaliciousPayload:
            """Simulates a pickle RCE payload via __reduce__."""
            def __reduce__(self):
                # In the original vulnerable code this would execute at load time.
                # We use os.getpid() (harmless) to verify execution would occur.
                import os
                return (os.getpid, ())

        malicious_bytes = pickle.dumps(MaliciousPayload())
        path = self._write_binary_file('malicious.pkl', malicious_bytes)
        result = process_pickle_file(path)
        # Must return an error dict
        self.assertIn('error', result)
        # The return value must not be an integer (os.getpid() return type),
        # which would indicate the payload executed.
        self.assertNotIsInstance(result, int)

    def test_empty_file_returns_error_dict(self):
        """An empty file must return {'error': ...}, not raise an exception."""
        path = self._write_file('empty.pkl', '')
        result = process_pickle_file(path)
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)

    def test_corrupt_file_returns_error_dict(self):
        """A file with random binary garbage must return {'error': ...}."""
        # NUL bytes and high-byte sequences that are never valid JSON
        garbage = b"\x80\x04\x95\x00\x00\x00\x00\x00\x00\x00\x00."
        path = self._write_binary_file('corrupt.pkl', garbage)
        result = process_pickle_file(path)
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)

    def test_nonexistent_file_returns_error_dict(self):
        """A missing file must return {'error': ...}, not raise an exception."""
        path = os.path.join(self.tmp_dir, 'does_not_exist.pkl')
        result = process_pickle_file(path)
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)

    def test_pickle_module_not_used_as_deserializer(self):
        """
        Confirm that process_pickle_file does not import or call pickle.load.

        This test inspects the source of the function to ensure the dangerous
        sink has been removed, providing a static-analysis-style regression guard.
        """
        import inspect
        import utils.file_handler as fh_module

        # pickle must not be imported at module level
        self.assertFalse(
            hasattr(fh_module, 'pickle'),
            "The 'pickle' module must not be imported in file_handler.py"
        )

        # The function body must not reference pickle.load
        source = inspect.getsource(process_pickle_file)
        self.assertNotIn(
            'pickle.load',
            source,
            "process_pickle_file must not call pickle.load()"
        )
        self.assertNotIn(
            'pickle.loads',
            source,
            "process_pickle_file must not call pickle.loads()"
        )

    def test_json_module_used_as_deserializer(self):
        """
        Confirm that process_pickle_file uses json.load as the safe deserializer.
        """
        import inspect
        source = inspect.getsource(process_pickle_file)
        self.assertIn(
            'json.load',
            source,
            "process_pickle_file must use json.load() as the safe deserializer"
        )


class TestRoleBadgeXSSPrevention(unittest.TestCase):
    """
    Security regression tests for CWE-79 (Stored XSS) in the role_badge Jinja2 filter.

    The original role_badge() embedded the role value directly into an HTML
    string without escaping, then the template rendered it with |safe, allowing
    stored XSS via a malicious role value in the database.

    The remediation uses html.escape() so that special HTML characters in the
    role value are neutralised before they reach the browser.

    These tests verify:
      1. Normal role values produce correct badge markup (functionality preserved).
      2. XSS payloads are escaped and cannot execute as HTML/JS.
      3. The html.escape() call is present in the source (static regression guard).
      4. The badge CSS class is derived from the allowlist, not the tainted input.
    """

    def setUp(self):
        """Import role_badge with a fake Jinja2 context (contextfilter needs it)."""
        # Provide a minimal mapping that satisfies @contextfilter expectations.
        # contextfilter passes the context as the first argument; a plain dict works.
        self.ctx = {}

    def _call_role_badge(self, role):
        """Helper: call the filter the same way Jinja2 does (ctx, value)."""
        from utils.jinja_filters import role_badge
        return role_badge(self.ctx, role)

    # ------------------------------------------------------------------
    # Positive / functionality tests
    # ------------------------------------------------------------------

    def test_admin_role_produces_danger_badge(self):
        """role_badge('admin') should produce a badge with the danger class."""
        result = self._call_role_badge('admin')
        self.assertIn('badge-danger', result)
        self.assertIn('admin', result)

    def test_project_manager_role_produces_primary_badge(self):
        """role_badge('project_manager') should produce a badge with the primary class."""
        result = self._call_role_badge('project_manager')
        self.assertIn('badge-primary', result)
        self.assertIn('project_manager', result)

    def test_team_member_role_produces_secondary_badge(self):
        """role_badge('team_member') should produce a badge with the secondary class."""
        result = self._call_role_badge('team_member')
        self.assertIn('badge-secondary', result)
        self.assertIn('team_member', result)

    def test_unknown_role_produces_secondary_badge(self):
        """An unknown (but benign) role falls back to the secondary class."""
        result = self._call_role_badge('viewer')
        self.assertIn('badge-secondary', result)
        self.assertIn('viewer', result)

    def test_output_is_a_span_element(self):
        """role_badge should always produce a <span> element."""
        result = self._call_role_badge('admin')
        self.assertTrue(result.startswith('<span'))
        self.assertTrue(result.endswith('</span>'))

    # ------------------------------------------------------------------
    # Security / XSS regression tests
    # ------------------------------------------------------------------

    def test_script_tag_xss_payload_is_escaped(self):
        """
        A role value containing a <script> tag must be HTML-escaped so the
        browser renders it as text, not as executable JavaScript.

        The unescaped payload '<script>alert(1)</script>' would trigger XSS.
        After escaping it becomes '&lt;script&gt;alert(1)&lt;/script&gt;'.
        """
        xss_payload = '<script>alert(1)</script>'
        result = self._call_role_badge(xss_payload)
        # The raw payload must NOT appear in the output
        self.assertNotIn('<script>', result)
        self.assertNotIn('</script>', result)
        # The escaped form must be present instead
        self.assertIn('&lt;script&gt;', result)
        self.assertIn('&lt;/script&gt;', result)

    def test_event_handler_xss_payload_is_escaped(self):
        """
        A role value designed to break out of an attribute context and inject
        an event handler (e.g. '"><img src=x onerror=alert(1)>') must be escaped.
        """
        xss_payload = '"><img src=x onerror=alert(1)>'
        result = self._call_role_badge(xss_payload)
        self.assertNotIn('<img', result)
        self.assertNotIn('onerror', result)
        # Double-quote in the payload must be escaped
        self.assertIn('&quot;', result)

    def test_angle_brackets_are_escaped(self):
        """Angle brackets in a role value must be converted to HTML entities."""
        result = self._call_role_badge('<malicious>')
        self.assertNotIn('<malicious>', result)
        self.assertIn('&lt;malicious&gt;', result)

    def test_ampersand_is_escaped(self):
        """Ampersands in a role value must be escaped to prevent entity injection."""
        result = self._call_role_badge('a & b')
        self.assertNotIn(' & ', result)
        self.assertIn('&amp;', result)

    def test_double_quote_is_escaped(self):
        """Double-quotes in a role value are escaped (quote=True used in html.escape)."""
        result = self._call_role_badge('role"name')
        self.assertNotIn('"name', result)
        self.assertIn('&quot;', result)

    def test_xss_payload_does_not_affect_badge_class(self):
        """
        The CSS class on the <span> element is derived from the known allowlist
        (admin/project_manager/team_member), never from the tainted role string.
        An unrecognised XSS payload must fall back to 'secondary', not inject
        attacker-controlled content into the class attribute.
        """
        xss_payload = 'admin"><script>alert(1)</script>'
        result = self._call_role_badge(xss_payload)
        # Class must be one of the safe allowlist values (falls back to secondary)
        self.assertIn('badge-secondary', result)
        # The script tag in the payload must be escaped
        self.assertNotIn('<script>', result)

    def test_none_role_does_not_raise(self):
        """role_badge(None) must not raise an exception and must return safe HTML."""
        result = self._call_role_badge(None)
        self.assertIsInstance(result, str)
        self.assertIn('<span', result)
        self.assertNotIn('<script>', result)

    def test_html_escape_is_used_in_source(self):
        """
        Confirm that role_badge uses html.escape() as the sanitizer.

        This static-analysis-style test ensures the SAST-recognised API is
        present in the source code so that the scanner can verify the fix.
        """
        import inspect
        from utils import jinja_filters
        source = inspect.getsource(jinja_filters.role_badge)
        self.assertIn(
            'html.escape',
            source,
            "role_badge must call html.escape() to sanitize the role value"
        )

    def test_html_module_imported(self):
        """Confirm that the html standard-library module is imported in jinja_filters."""
        import utils.jinja_filters as jf_module
        self.assertTrue(
            hasattr(jf_module, 'html'),
            "The 'html' standard-library module must be imported in jinja_filters.py"
        )


if __name__ == '__main__':
    unittest.main()

