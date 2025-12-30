"""
Utility functions for the Multi-Course RAG Student Assistant.
"""
import logging
from datetime import datetime
from typing import Optional
import re

import config

logger = logging.getLogger(__name__)


def sanitize_course_id(course_id: str) -> str:
    """
    Sanitize and normalize a course ID.
    
    Args:
        course_id: Raw course ID input
        
    Returns:
        Sanitized course ID (uppercase, no special chars except dash/underscore)
    """
    # Remove leading/trailing whitespace
    course_id = course_id.strip()
    
    # Replace spaces with underscores
    course_id = course_id.replace(" ", "_")
    
    # Remove special characters except dash and underscore
    course_id = re.sub(r'[^A-Za-z0-9_-]', '', course_id)
    
    # Convert to uppercase
    course_id = course_id.upper()
    
    return course_id or config.MISC_COURSE_ID


def format_timestamp(timestamp_str: str) -> str:
    """
    Format an ISO timestamp string for display.
    
    Args:
        timestamp_str: ISO format timestamp
        
    Returns:
        Human-readable date string
    """
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%b %d, %Y %I:%M %p")
    except (ValueError, TypeError):
        return timestamp_str or "Unknown"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def get_file_icon(file_type: str) -> str:
    """
    Get an emoji icon for a file type.
    
    Args:
        file_type: File extension or type
        
    Returns:
        Emoji icon
    """
    icons = {
        "pdf": "📄",
        "txt": "📝",
        "text": "📝",
        "lecture": "📚",
        "assignment": "📋",
        "syllabus": "📜",
        "exam": "📝",
        "schedule": "📅",
        "misc": "📎"
    }
    return icons.get(file_type.lower(), "📄")


def get_doc_type_icon(doc_type: str) -> str:
    """
    Get an emoji icon for a document type.
    
    Args:
        doc_type: Document type
        
    Returns:
        Emoji icon
    """
    icons = {
        "lecture": "📚",
        "assignment": "📋",
        "syllabus": "📜",
        "exam": "✏️",
        "schedule": "📅",
        "misc": "📎"
    }
    return icons.get(doc_type.lower(), "📄")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def validate_course_id(course_id: str) -> tuple[bool, Optional[str]]:
    """
    Validate a course ID.
    
    Args:
        course_id: Course ID to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not course_id or not course_id.strip():
        return False, "Course ID cannot be empty"
    
    sanitized = sanitize_course_id(course_id)
    
    if len(sanitized) < 2:
        return False, "Course ID must be at least 2 characters"
    
    if len(sanitized) > 50:
        return False, "Course ID cannot exceed 50 characters"
    
    return True, None

