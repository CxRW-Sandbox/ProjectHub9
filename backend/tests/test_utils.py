"""
Basic tests for utility functions
"""
import unittest
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.datetime_utils import get_utc_now, format_utc_datetime, get_utc_timestamp
from utils.jinja_filters import role_badge


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


class TestRoleBadgeXSSPrevention(unittest.TestCase):
    """Tests for role_badge filter - verifies Stored XSS fix (CWE-79).

    The role_badge filter produces HTML that the template marks |safe.
    Any user-controlled role value stored in the DB must be HTML-encoded
    before being embedded in the returned HTML string.
    """

    # role_badge is a @contextfilter; pass None as the context (not used by the filter)
    CTX = None

    def _call(self, role):
        return role_badge(self.CTX, role)

    # --- Normal / happy-path tests ---

    def test_admin_role_returns_danger_badge(self):
        """Admin role produces a badge with the 'danger' colour class."""
        result = self._call('admin')
        self.assertIn('badge-danger', result)
        self.assertIn('admin', result)

    def test_project_manager_role_returns_primary_badge(self):
        """project_manager role produces a badge with the 'primary' colour class."""
        result = self._call('project_manager')
        self.assertIn('badge-primary', result)
        self.assertIn('project_manager', result)

    def test_team_member_role_returns_secondary_badge(self):
        """team_member role produces a badge with the 'secondary' colour class."""
        result = self._call('team_member')
        self.assertIn('badge-secondary', result)
        self.assertIn('team_member', result)

    def test_unknown_role_returns_secondary_badge(self):
        """Unknown role falls back to the 'secondary' colour class."""
        result = self._call('unknown_role')
        self.assertIn('badge-secondary', result)

    def test_output_is_span_element(self):
        """Output must be a <span> element."""
        result = self._call('admin')
        self.assertTrue(result.strip().startswith('<span'))
        self.assertIn('</span>', result)

    # --- XSS / Stored-XSS regression tests ---

    def test_script_tag_in_role_is_escaped(self):
        """A <script> tag stored as the role value must not appear raw in output.

        This is the core Stored XSS regression test.  The role value is read from
        the DB (untrusted) and the template uses |safe, so role_badge MUST escape
        the value before embedding it.
        """
        xss_payload = '<script>alert(1)</script>'
        result = self._call(xss_payload)
        # The literal payload must NOT appear unescaped
        self.assertNotIn('<script>', result)
        self.assertNotIn('</script>', result)
        # HTML-encoded form should be present instead
        self.assertIn('&lt;script&gt;', result)

    def test_event_handler_in_role_is_escaped(self):
        """An inline event-handler payload stored as role must be escaped."""
        xss_payload = '"><img src=x onerror=alert(1)>'
        result = self._call(xss_payload)
        self.assertNotIn('<img', result)
        self.assertNotIn('onerror', result)
        self.assertIn('&lt;', result)

    def test_javascript_uri_in_role_is_escaped(self):
        """A javascript: URI stored as role must be escaped."""
        xss_payload = "javascript:alert('xss')"
        result = self._call(xss_payload)
        # The apostrophe and colon are not inherently dangerous here, but
        # the whole value must not break out of the element as raw HTML
        self.assertNotIn('<', result.replace('&lt;', ''))
        self.assertNotIn('>', result.replace('&gt;', ''))

    def test_html_entities_in_role_are_encoded(self):
        """HTML special characters in role value must be entity-encoded."""
        role_with_html_chars = 'manager & <owner>'
        result = self._call(role_with_html_chars)
        self.assertNotIn('<owner>', result)
        self.assertIn('&amp;', result)
        self.assertIn('&lt;owner&gt;', result)

    def test_double_quote_in_role_is_escaped(self):
        """A double-quote stored as role must be escaped to prevent attribute injection."""
        xss_payload = '"onmouseover="alert(1)'
        result = self._call(xss_payload)
        # html.escape by default escapes " to &quot;
        self.assertNotIn('"onmouseover=', result)

    def test_none_role_does_not_raise(self):
        """Passing None as role must not raise an exception."""
        try:
            result = self._call(None)
            self.assertIn('<span', result)
        except Exception as e:
            self.fail(f"role_badge raised an exception for None input: {e}")

    def test_empty_string_role_does_not_raise(self):
        """Empty string role must not raise an exception."""
        result = self._call('')
        self.assertIn('<span', result)
        self.assertIn('badge-secondary', result)

    def test_xss_payload_does_not_appear_as_raw_html(self):
        """Comprehensive check: any angle bracket from a stored payload must be encoded."""
        payloads = [
            '<b>bold</b>',
            '<svg/onload=alert(1)>',
            '&#60;script&#62;alert(1)&#60;/script&#62;',
            '<script>alert(1)</script>',
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                result = self._call(payload)
                # After escaping, no raw < or > should exist outside the wrapper span tags
                inner = result[result.index('>') + 1: result.rindex('<')]
                self.assertNotIn('<', inner, msg=f"Raw '<' found in output for payload: {payload}")
                self.assertNotIn('>', inner, msg=f"Raw '>' found in output for payload: {payload}")


if __name__ == '__main__':
    unittest.main()

