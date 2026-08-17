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


if __name__ == '__main__':
    unittest.main()

