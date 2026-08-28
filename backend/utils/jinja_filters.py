# Jinja2 template filters
from jinja2 import contextfilter, Markup
from datetime import datetime
import hashlib
import html

@contextfilter
def format_datetime(context, value, format='%Y-%m-%d %H:%M:%S'):
    """Format datetime"""
    if value is None:
        return ''
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except:
            return value
    return value.strftime(format)

@contextfilter
def user_display_name(context, user):
    """Get user display name from context.

    Returns an HTML-escaped Markup string so that stored XSS payloads in
    username or email fields are neutralised before reaching the browser.
    html.escape() is the stdlib sanitizer recognised by SAST engines (CWE-79).
    """
    if isinstance(user, dict):
        raw = user.get('username', user.get('email', 'Unknown'))
    else:
        raw = getattr(user, 'username', getattr(user, 'email', 'Unknown'))
    # html.escape() converts <, >, &, ", ' to safe HTML entities.
    # Wrapping in Markup tells Jinja2 the value is already escaped so it
    # does not double-escape the entities.
    return Markup(html.escape(str(raw) if raw is not None else '', quote=True))

@contextfilter
def truncate(context, value, length=50):
    """Truncate string with ellipsis"""
    if not value:
        return ''
    if len(value) <= length:
        return value
    return value[:length] + '...'

@contextfilter
def md5_hash(context, value):
    """Generate MD5 hash (for demonstration purposes)"""
    if not value:
        return ''
    return hashlib.md5(str(value).encode()).hexdigest()

@contextfilter
def request_id_filter(context):
    """Get request ID from context"""
    from utils.request_context import get_request_context
    ctx = get_request_context()
    if ctx and hasattr(ctx, 'request_id'):
        return ctx.request_id
    return 'N/A'

@contextfilter
def format_file_size(context, size_bytes):
    """Format file size in human-readable format"""
    if not size_bytes:
        return '0 B'
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

@contextfilter
def role_badge(context, role):
    """Generate role badge HTML with HTML-escaped role value to prevent XSS"""
    role_colors = {
        'admin': 'danger',
        'project_manager': 'primary',
        'team_member': 'secondary'
    }
    color = role_colors.get(role, 'secondary')
    # Use html.escape() to prevent stored XSS: role values from the database
    # are untrusted and must be escaped before embedding in HTML output.
    escaped_role = html.escape(str(role) if role is not None else '', quote=True)
    return f'<span class="badge badge-{color}">{escaped_role}</span>'

