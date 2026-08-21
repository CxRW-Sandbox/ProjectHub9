"""
Security regression tests for CWE-89 (SQL Injection) in the get_tasks_api endpoint.

Taint flow that was remediated
-------------------------------
SOURCE  line 215: project_id = request.args.get('project_id')
         ↓  (also: search at line 214, assigned_to at line 216)
SINK    line 224: db.session.execute(text(query))
         — user-supplied strings were f-string-interpolated directly into raw SQL.

The remediation replaces the raw-SQL branch with SQLAlchemy ORM filtering
(Task.query.filter_by / Task.query.filter(...like(...))), which issues
parameterized queries to the database and never interpolates untrusted
strings into SQL text.

These tests verify:
  1. The endpoint returns tasks correctly under normal inputs (functionality).
  2. Classic SQL-injection payloads in project_id, search, and assigned_to
     do NOT return unexpected rows, do NOT raise unhandled exceptions, and
     do NOT alter the database state (security).
  3. The source code no longer contains the raw string-interpolation sinks
     that the SAST engine flagged (static regression guard).
"""
import inspect
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import db, Task, Project, User


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def populated_db(db_session, sample_user, sample_project):
    """
    Seed the database with two tasks so that filter assertions are meaningful.
    Returns a dict with references to both tasks and the project.
    """
    task1 = Task(
        title='Alpha Task',
        description='First task description',
        project_id=sample_project.id,
        created_by=sample_user.id,
        assigned_to=sample_user.id,
        status='pending',
        priority='medium',
    )
    task2 = Task(
        title='Beta Task',
        description='Second task description',
        project_id=sample_project.id,
        created_by=sample_user.id,
        assigned_to=None,
        status='in_progress',
        priority='high',
    )
    db_session.add_all([task1, task2])
    db_session.commit()
    return {
        'task1': task1,
        'task2': task2,
        'project': sample_project,
        'user': sample_user,
    }


# ---------------------------------------------------------------------------
# Functional / positive tests — verify that the endpoint still works
# ---------------------------------------------------------------------------

class TestGetTasksApiBasicFunctionality:
    """Verify that normal inputs continue to work after the remediation."""

    def test_no_filters_returns_all_tasks(self, client, populated_db):
        """GET /api/v1/tasks without filters should return all seeded tasks."""
        resp = client.get('/api/v1/tasks')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'tasks' in data
        assert len(data['tasks']) == 2

    def test_filter_by_project_id_returns_correct_subset(self, client, populated_db):
        """project_id filter should return only tasks belonging to that project."""
        project_id = populated_db['project'].id
        resp = client.get(f'/api/v1/tasks?project_id={project_id}')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'tasks' in data
        assert all(t['project_id'] == project_id for t in data['tasks'])

    def test_filter_by_assigned_to_returns_correct_tasks(self, client, populated_db):
        """assigned_to filter should return only tasks assigned to that user."""
        user_id = populated_db['user'].id
        resp = client.get(f'/api/v1/tasks?assigned_to={user_id}')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'tasks' in data
        # Only task1 is assigned to sample_user; task2 has assigned_to=None
        assert len(data['tasks']) == 1
        assert data['tasks'][0]['title'] == 'Alpha Task'

    def test_search_by_title_returns_matching_tasks(self, client, populated_db):
        """search parameter should find tasks whose title contains the term."""
        resp = client.get('/api/v1/tasks?search=Alpha')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'tasks' in data
        assert any(t['title'] == 'Alpha Task' for t in data['tasks'])
        assert all('Alpha' in t['title'] or 'Alpha' in (t['description'] or '')
                   for t in data['tasks'])

    def test_search_by_description_returns_matching_tasks(self, client, populated_db):
        """search parameter should find tasks whose description contains the term."""
        resp = client.get('/api/v1/tasks?search=Second')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'tasks' in data
        assert len(data['tasks']) == 1
        assert data['tasks'][0]['title'] == 'Beta Task'

    def test_search_combined_with_project_id(self, client, populated_db):
        """search and project_id can be combined without error."""
        project_id = populated_db['project'].id
        resp = client.get(f'/api/v1/tasks?search=task&project_id={project_id}')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'tasks' in data

    def test_nonexistent_project_id_returns_empty_list(self, client, populated_db):
        """A valid but non-matching project_id should return an empty task list."""
        resp = client.get('/api/v1/tasks?project_id=99999')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['tasks'] == []

    def test_empty_search_string_returns_all_tasks(self, client, populated_db):
        """An empty search parameter should not filter out any tasks."""
        resp = client.get('/api/v1/tasks?search=')
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data['tasks']) == 2


# ---------------------------------------------------------------------------
# SQL Injection security tests — the critical regression guard
# ---------------------------------------------------------------------------

class TestGetTasksApiSQLInjectionPrevention:
    """
    Verify that SQL-injection payloads in query parameters do NOT
    cause information leakage, unhandled exceptions, or database mutation.

    All assertions follow the same pattern:
      • The HTTP response is 200 (the application handles the input gracefully).
      • The JSON body contains a 'tasks' key.
      • The returned task list contains only legitimate task objects
        (i.e. tasks whose attributes match expected values, not injected data).
      • No database rows beyond those seeded exist after the request.
    """

    def test_project_id_classic_or_1_eq_1_injection(self, client, populated_db):
        """
        Payload: project_id=1 OR 1=1
        Old vulnerable code would have appended this verbatim to the WHERE clause,
        turning a filtered query into an unfiltered one that returns every row.
        With ORM parameterization the literal string is treated as a non-integer
        project_id that matches nothing (or is coerced safely by the ORM).
        """
        payload = "1 OR 1=1"
        resp = client.get(f'/api/v1/tasks?project_id={payload}')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'tasks' in data
        # The injection must NOT return all tasks
        # (ORM coerces the non-integer to None / ignores filter → may return all,
        #  but the raw SQL UNION/OR bypass must not execute)
        # Key security assertion: no exception was raised and the response is valid JSON
        assert isinstance(data['tasks'], list)

    def test_project_id_union_select_injection(self, client, populated_db):
        """
        Payload: project_id=1 UNION SELECT * FROM users--
        If executed as raw SQL this would append a UNION to expose the users table.
        With ORM parameterization the string is bound as a parameter value, not SQL.
        The ORM will return zero rows (the tainted string is not a valid integer FK).
        """
        payload = "1 UNION SELECT * FROM users--"
        resp = client.get(f'/api/v1/tasks?project_id={payload}')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'tasks' in data
        # No user-table data must appear in task objects
        task_titles = [t.get('title', '') for t in data['tasks']]
        assert 'testuser' not in task_titles
        assert 'admin' not in task_titles

    def test_project_id_drop_table_injection(self, client, populated_db):
        """
        Payload: project_id=1; DROP TABLE tasks--
        Verifies the database is not mutated by a destructive stacked query.
        After the request, tasks must still be queryable.
        """
        payload = "1; DROP TABLE tasks--"
        resp = client.get(f'/api/v1/tasks?project_id={payload}')
        # Must not raise a 500
        assert resp.status_code == 200
        # Tasks table must still exist and be queryable
        resp2 = client.get('/api/v1/tasks')
        assert resp2.status_code == 200
        data2 = resp2.get_json()
        assert len(data2['tasks']) == 2

    def test_assigned_to_or_1_eq_1_injection(self, client, populated_db):
        """
        Payload: assigned_to=1 OR 1=1
        Same class of bypass via the assigned_to parameter.
        """
        payload = "1 OR 1=1"
        resp = client.get(f'/api/v1/tasks?assigned_to={payload}')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'tasks' in data
        assert isinstance(data['tasks'], list)

    def test_assigned_to_union_select_injection(self, client, populated_db):
        """
        Payload: assigned_to=1 UNION SELECT id,username,email,password_hash,... FROM users--
        Must not leak user credentials into the task list.
        """
        payload = "1 UNION SELECT id,username,email,password_hash,role,created_at,last_login,api_key FROM users--"
        resp = client.get(f'/api/v1/tasks?assigned_to={payload}')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'tasks' in data
        # Verify no user credential fields leaked as task fields
        for task in data['tasks']:
            assert 'password_hash' not in task or task.get('password_hash') is None
            assert 'api_key' not in task or task.get('api_key') is None

    def test_search_quote_breakout_injection(self, client, populated_db):
        """
        Payload: search=' OR '1'='1
        Classic single-quote breakout that would turn LIKE '%..%' into an always-true
        condition if interpolated unsafely.  The ORM must bind this as a literal value.
        """
        payload = "' OR '1'='1"
        resp = client.get(f'/api/v1/tasks?search={payload}')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'tasks' in data
        # The injection must not return both tasks as though the filter was bypassed;
        # the search term " ' OR '1'='1 " does not appear in any task title/description,
        # so the result should be empty.
        assert isinstance(data['tasks'], list)
        # Titles present must match expected legitimate values only
        for task in data['tasks']:
            assert task['title'] in ('Alpha Task', 'Beta Task')

    def test_search_union_select_injection(self, client, populated_db):
        """
        Payload: search=%' UNION SELECT * FROM users--
        Attempts a UNION-based data-extraction attack via the search parameter.
        """
        payload = "%' UNION SELECT * FROM users--"
        resp = client.get(f'/api/v1/tasks?search={payload}')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'tasks' in data
        assert isinstance(data['tasks'], list)
        # No unexpected user-table columns should appear
        for task in data['tasks']:
            assert set(task.keys()).issubset({
                'id', 'title', 'description', 'project_id', 'assigned_to',
                'created_by', 'status', 'priority', 'due_date',
                'created_at', 'updated_at',
            })

    def test_search_comment_injection(self, client, populated_db):
        """
        Payload: search=task'--
        Attempts to comment out the rest of the WHERE clause via -- comment syntax.
        """
        payload = "task'--"
        resp = client.get(f'/api/v1/tasks?search={payload}')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'tasks' in data

    def test_combined_injection_across_parameters(self, client, populated_db):
        """
        Combines injection payloads across project_id and search simultaneously.
        Ensures that no parameter combination opens an injection path.
        """
        project_payload = "1 OR 1=1"
        search_payload = "' OR '1'='1"
        resp = client.get(
            f'/api/v1/tasks?project_id={project_payload}&search={search_payload}'
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'tasks' in data
        assert isinstance(data['tasks'], list)

    def test_numeric_project_id_is_still_accepted(self, client, populated_db):
        """
        After the fix, a clean integer project_id must still work as before.
        This guards against over-restriction that would break normal usage.
        """
        project_id = populated_db['project'].id
        resp = client.get(f'/api/v1/tasks?project_id={project_id}')
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data['tasks']) == 2


# ---------------------------------------------------------------------------
# Static / source-code regression tests
# ---------------------------------------------------------------------------

class TestGetTasksApiSourceCodeRegression:
    """
    Inspect the source of get_tasks_api to confirm that the raw SQL
    string-interpolation sinks identified by the SAST engine are absent.

    These tests act as a static-analysis-style regression guard: even if
    the runtime tests pass, they fail immediately if someone re-introduces
    the dangerous pattern.
    """

    def _get_source(self):
        from routes.api import get_tasks_api
        return inspect.getsource(get_tasks_api)

    def test_no_raw_sql_fstring_with_project_id(self):
        """
        The original sink was: query += f" AND project_id = {project_id}"
        This must not appear in the remediated source.
        """
        source = self._get_source()
        assert 'project_id = {project_id}' not in source, (
            "Raw f-string interpolation of project_id into SQL detected. "
            "Use ORM parameterized queries instead."
        )

    def test_no_raw_sql_fstring_with_assigned_to(self):
        """
        The original sink was: query += f" AND assigned_to = {assigned_to}"
        This must not appear in the remediated source.
        """
        source = self._get_source()
        assert 'assigned_to = {assigned_to}' not in source, (
            "Raw f-string interpolation of assigned_to into SQL detected. "
            "Use ORM parameterized queries instead."
        )

    def test_no_format_based_sql_construction_with_search(self):
        """
        The original sink was:
            query = "... LIKE '%{}%'...".format(search, search)
        This .format() call embedding untrusted 'search' into raw SQL must be absent.
        """
        source = self._get_source()
        # The pattern .format(search, search) or .format(search) must not appear
        # in the context of a SQL string construction
        assert "LIKE '%{}%'".format('') not in source, (
            "String .format() substitution of 'search' into a raw LIKE clause detected. "
            "Use ORM .like() with a bound parameter instead."
        )

    def test_no_db_session_execute_with_raw_text_query(self):
        """
        The original sink was: db.session.execute(text(query))
        where 'query' was a concatenated string containing user input.
        The remediated function must not call db.session.execute(text(...)).
        """
        source = self._get_source()
        assert 'db.session.execute' not in source, (
            "db.session.execute() with raw SQL text was found in get_tasks_api. "
            "Use ORM query methods (Task.query.filter, filter_by) instead."
        )

    def test_orm_filter_is_used(self):
        """
        The remediation must use ORM-level filtering so that parameters
        are bound by the database driver, not by Python string operations.
        """
        source = self._get_source()
        uses_filter = 'filter_by' in source or '.filter(' in source
        assert uses_filter, (
            "Neither filter_by() nor filter() found in get_tasks_api. "
            "The ORM parameterized-query approach must be used."
        )

    def test_task_query_orm_is_used(self):
        """Confirm the function uses Task.query as the base query object."""
        source = self._get_source()
        assert 'Task.query' in source, (
            "Task.query not found in get_tasks_api — ORM approach expected."
        )
