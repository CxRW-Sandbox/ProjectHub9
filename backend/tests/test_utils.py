"""
Basic tests for utility functions
"""
import unittest
import sys
import os
from datetime import datetime
import html as html_module

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.datetime_utils import get_utc_now, format_utc_datetime, get_utc_timestamp


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
    """
    Tests for the role_badge Jinja2 filter to verify Stored XSS remediation.

    The taint flow: attacker stores a malicious role value in the database →
    User.query.all() reads it → render_template passes users to admin.html →
    the template renders {{ user.role|role_badge|safe }}.

    The |safe marker tells Jinja2 to trust the HTML returned by role_badge, so
    XSS prevention MUST happen inside role_badge itself via html.escape().
    """

    def _make_context(self):
        """Return a minimal Jinja2-style context object (dict works for contextfilter)."""
        return {}

    def _call_role_badge(self, role):
        """Invoke role_badge the same way Jinja2's @contextfilter does."""
        from utils.jinja_filters import role_badge
        return role_badge(self._make_context(), role)

    # ------------------------------------------------------------------
    # Positive cases: known roles produce correct, safe HTML
    # ------------------------------------------------------------------

    def test_admin_role_produces_danger_badge(self):
        """Admin role maps to badge-danger CSS class."""
        result = self._call_role_badge('admin')
        self.assertIn('badge-danger', result)
        self.assertIn('admin', result)

    def test_project_manager_role_produces_primary_badge(self):
        """Project manager role maps to badge-primary CSS class."""
        result = self._call_role_badge('project_manager')
        self.assertIn('badge-primary', result)

    def test_team_member_role_produces_secondary_badge(self):
        """Team member role maps to badge-secondary CSS class."""
        result = self._call_role_badge('team_member')
        self.assertIn('badge-secondary', result)

    def test_unknown_role_falls_back_to_secondary(self):
        """Unknown roles fall back to badge-secondary (no XSS via CSS class)."""
        result = self._call_role_badge('unknown_role')
        self.assertIn('badge-secondary', result)

    def test_output_is_span_element(self):
        """Output is a <span> element regardless of role."""
        result = self._call_role_badge('admin')
        self.assertTrue(result.startswith('<span'))
        self.assertTrue(result.endswith('</span>'))

    # ------------------------------------------------------------------
    # Negative / attack cases: malicious role values must be escaped
    # ------------------------------------------------------------------

    def test_script_tag_in_role_is_escaped(self):
        """A <script> tag stored as a role value must not appear literally in output."""
        malicious_role = '<script>alert("xss")</script>'
        result = self._call_role_badge(malicious_role)
        # The raw <script> tag must NOT appear in the output
        self.assertNotIn('<script>', result)
        self.assertNotIn('</script>', result)
        # The angle brackets must be HTML-encoded
        self.assertIn('&lt;script&gt;', result)

    def test_img_onerror_in_role_is_escaped(self):
        """An img onerror payload stored as a role must not execute."""
        malicious_role = '<img src=x onerror=alert(1)>'
        result = self._call_role_badge(malicious_role)
        self.assertNotIn('<img', result)
        self.assertIn('&lt;img', result)

    def test_event_handler_in_role_is_escaped(self):
        """An onmouseover event handler embedded in role must be HTML-encoded."""
        malicious_role = 'admin" onmouseover="alert(document.cookie)'
        result = self._call_role_badge(malicious_role)
        self.assertNotIn('onmouseover', result)
        # The double-quote must be encoded
        self.assertIn('&quot;', result)

    def test_javascript_protocol_in_role_is_escaped(self):
        """A javascript: URI stored as role must be HTML-encoded."""
        malicious_role = 'javascript:alert(1)'
        result = self._call_role_badge(malicious_role)
        # The colon is not special in HTML, but < and > would be; the important
        # check is that angle brackets in any wrapping attempt are encoded.
        # Also verify the value itself is inside the HTML as escaped text only.
        self.assertNotIn('<script', result.lower())

    def test_html_entities_are_encoded(self):
        """Angle brackets and double-quotes in role are replaced by HTML entities."""
        malicious_role = '<b onload="evil()">'
        result = self._call_role_badge(malicious_role)
        self.assertNotIn('<b', result)
        # html.escape encodes <, >, &, " by default
        self.assertIn('&lt;', result)
        self.assertIn('&gt;', result)

    def test_ampersand_in_role_is_escaped(self):
        """Ampersands in role values are HTML-encoded to prevent entity injection."""
        malicious_role = 'admin&extra=<script>'
        result = self._call_role_badge(malicious_role)
        self.assertNotIn('<script>', result)
        self.assertIn('&amp;', result)

    def test_escaped_output_matches_html_escape(self):
        """The role text in the badge exactly equals html.escape(role)."""
        malicious_role = '<b>bold</b>'
        result = self._call_role_badge(malicious_role)
        expected_safe_role = html_module.escape(malicious_role)
        self.assertIn(expected_safe_role, result)

    def test_null_bytes_in_role_are_handled(self):
        """Role values containing null bytes (\\x00) do not cause crashes."""
        # Use escape sequence, not a literal control byte
        malicious_role = 'admin\x00<script>alert(1)</script>'
        result = self._call_role_badge(malicious_role)
        self.assertNotIn('<script>', result)

    def test_unicode_angle_brackets_in_role_are_escaped(self):
        """Unicode representations of angle brackets are HTML-encoded."""
        # < = <, > = >
        malicious_role = '<script>alert(1)</script>'
        result = self._call_role_badge(malicious_role)
        # After html.escape, the < and > should be encoded
        self.assertNotIn('<script>', result)
        self.assertIn('&lt;script&gt;', result)

    def test_css_class_never_contains_user_input(self):
        """
        The CSS class name (badge-*) is always drawn from the fixed allow-list,
        never from unescaped user input, even for unknown roles.
        """
        malicious_role = 'danger" style="background:url(javascript:alert(1))'
        result = self._call_role_badge(malicious_role)
        # The CSS class must be the fallback 'secondary', not the injected value
        self.assertIn('badge-secondary', result)
        # The injected style attribute must NOT appear
        self.assertNotIn('style=', result)


if __name__ == '__main__':
    unittest.main()

