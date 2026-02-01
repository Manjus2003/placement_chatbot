"""
🔒 INPUT VALIDATOR MODULE
=========================
Comprehensive input validation and sanitization for user queries.

Security Features:
- Length validation (min/max)
- HTML/XSS prevention
- SQL injection detection
- Control character removal
- Unicode normalization
- Gibberish detection
- Language detection
- Topic relevance checking
- Rate limiting

Usage:
    from input_validator import validate_input, RateLimiter
    
    is_valid, cleaned_query, info = validate_input(user_query)
    if not is_valid:
        print(info['errors'])
    else:
        # Process cleaned_query
"""

import re
import html
import unicodedata
from typing import Tuple, Dict, List, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time


# ============================================================
# CONFIGURATION
# ============================================================

# Length constraints
MIN_QUERY_LENGTH = 10  # characters
OPTIMAL_QUERY_LENGTH = (150, 500)  # (min_optimal, max_optimal)
MAX_QUERY_LENGTH = 1500  # characters
HARD_LIMIT = 2000  # absolute maximum

# Placement-related keywords for relevance checking
PLACEMENT_KEYWORDS = {
    # Core terms
    'company', 'companies', 'placement', 'placements', 'job', 'jobs',
    'interview', 'interviews', 'ctc', 'package', 'salary', 'offer',
    'role', 'roles', 'position', 'positions',
    
    # Process terms
    'eligibility', 'eligible', 'cgpa', 'branch', 'skill', 'skills',
    'preparation', 'prepare', 'round', 'rounds', 'process', 'apply',
    'application', 'career', 'hire', 'hiring', 'recruit', 'recruitment',
    
    # Company names (sample - will be extended)
    'amazon', 'google', 'microsoft', 'intel', 'nvidia', 'apple',
    'meta', 'facebook', 'ibm', 'oracle', 'sap', 'cisco', 'dell',
    'hp', 'infosys', 'wipro', 'tcs', 'cognizant', 'accenture',
    
    # Technical terms
    'software', 'engineer', 'developer', 'programmer', 'analyst',
    'data', 'scientist', 'machine learning', 'ml', 'ai', 'artificial intelligence',
    'fullstack', 'backend', 'frontend', 'devops', 'cloud',
    
    # Educational terms
    'mtech', 'btech', 'mba', 'student', 'graduate', 'degree',
    'university', 'college', 'campus', 'qualification'
}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ValidationResult:
    """Result of input validation"""
    is_valid: bool
    cleaned_query: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks_passed: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'is_valid': self.is_valid,
            'cleaned_query': self.cleaned_query,
            'errors': self.errors,
            'warnings': self.warnings,
            'checks_passed': self.checks_passed,
            'metadata': self.metadata
        }


# ============================================================
# CORE VALIDATION FUNCTIONS
# ============================================================

def validate_type(query: Any) -> Tuple[bool, str]:
    """
    Validate input type - must be a string.
    
    Returns:
        (is_valid, error_message)
    """
    if query is None:
        return False, "Query cannot be None"
    
    if not isinstance(query, str):
        return False, f"Query must be a string, got {type(query).__name__}"
    
    return True, ""


def validate_empty(query: str) -> Tuple[bool, str]:
    """
    Check if query is empty or only whitespace.
    
    Returns:
        (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Query cannot be empty"
    
    return True, ""


def validate_min_length(query: str, min_length: int = MIN_QUERY_LENGTH) -> Tuple[bool, str]:
    """
    Ensure query meets minimum length requirement.
    
    Returns:
        (is_valid, error_message)
    """
    cleaned = query.strip()
    
    if len(cleaned) < min_length:
        return False, f"Query too short. Minimum {min_length} characters required (got {len(cleaned)})"
    
    return True, ""


def validate_max_length(query: str, max_length: int = MAX_QUERY_LENGTH) -> Tuple[bool, str, str]:
    """
    Ensure query doesn't exceed maximum length.
    Truncates if too long.
    
    Returns:
        (is_valid, cleaned_query, warning_message)
    """
    if len(query) <= max_length:
        return True, query, ""
    
    # Truncate at word boundary
    truncated = query[:max_length]
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.8:  # If space is in last 20%
        truncated = truncated[:last_space]
    
    warning = f"Query truncated from {len(query)} to {len(truncated)} characters"
    
    return True, truncated, warning


def sanitize_html(query: str) -> str:
    """
    Remove HTML tags and escape special characters to prevent XSS attacks.
    
    Args:
        query: Input string that may contain HTML
        
    Returns:
        Sanitized string
    """
    # Remove HTML/XML tags
    query = re.sub(r'<[^>]+>', '', query)
    
    # Escape HTML entities
    query = html.escape(query, quote=True)
    
    # Remove script-like patterns (case insensitive)
    query = re.sub(r'javascript\s*:', '', query, flags=re.IGNORECASE)
    query = re.sub(r'on\w+\s*=', '', query, flags=re.IGNORECASE)
    
    # Remove data URIs
    query = re.sub(r'data:[\w/]+;base64,', '', query, flags=re.IGNORECASE)
    
    return query


def detect_sql_injection(query: str) -> Tuple[bool, str]:
    """
    Detect potential SQL injection patterns.
    
    Returns:
        (has_sql_injection, error_message)
    """
    # Dangerous SQL patterns
    sql_patterns = [
        r'\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE)\s+TABLE\b',
        r'\b(UNION|SELECT)\s+.*\bFROM\b',
        r'--\s*$',  # SQL comment at end
        r'/\*.*?\*/',  # Multi-line comment
        r'\'\s*OR\s*\'?\d+\'?\s*=\s*\'?\d+',  # OR 1=1 pattern
        r';\s*(DROP|DELETE|INSERT|UPDATE)',
        r'exec\s*\(',
        r'xp_\w+',  # Extended stored procedures
        r'sp_\w+',  # Stored procedures
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True, "Invalid query pattern detected. Please use plain text."
    
    return False, ""


def sanitize_special_chars(query: str) -> str:
    """
    Remove or replace potentially problematic characters while preserving
    meaningful punctuation.
    
    Args:
        query: Input string
        
    Returns:
        Sanitized string
    """
    # Remove control characters (except newline, tab, carriage return)
    query = ''.join(
        char for char in query 
        if char.isprintable() or char in '\n\t\r'
    )
    
    # Normalize unicode characters (NFKC = compatibility normalization)
    query = unicodedata.normalize('NFKC', query)
    
    # Remove zero-width characters and other invisible unicode
    invisible_chars = [
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\ufeff',  # Zero-width no-break space (BOM)
        '\u00ad',  # Soft hyphen
    ]
    for char in invisible_chars:
        query = query.replace(char, '')
    
    # Normalize whitespace
    query = re.sub(r'[\r\n]+', ' ', query)  # Replace newlines with space
    query = re.sub(r'\t+', ' ', query)       # Replace tabs with space
    query = re.sub(r' +', ' ', query)        # Collapse multiple spaces
    
    # Strip leading/trailing whitespace
    query = query.strip()
    
    # Limit repeated punctuation (e.g., ??? → ??)
    query = re.sub(r'([!?.]){3,}', r'\1\1', query)
    
    return query


def is_gibberish(query: str) -> bool:
    """
    Detect if query is random characters or gibberish.
    
    Uses heuristics:
    - Vowel ratio (English text has ~40% vowels)
    - Excessive repeated characters
    - Word-like patterns
    
    Returns:
        True if gibberish, False if legitimate
    """
    # Remove whitespace and punctuation for analysis
    alpha_only = ''.join(c for c in query.lower() if c.isalpha())
    
    if len(alpha_only) == 0:
        return True
    
    # Check vowel ratio (English: ~35-45%)
    vowels = sum(1 for c in alpha_only if c in 'aeiou')
    vowel_ratio = vowels / len(alpha_only)
    
    if vowel_ratio < 0.15 or vowel_ratio > 0.70:
        return True  # Likely gibberish
    
    # Check for excessive repeated characters (e.g., "aaaaaaa")
    if re.search(r'(.)\1{6,}', alpha_only):
        return True
    
    # Check for alternating consonant/vowel patterns (e.g., "kakaka")
    if re.search(r'([^aeiou][aeiou]){6,}', alpha_only, re.IGNORECASE):
        return True
    
    # Check if words exist (split by spaces)
    words = query.split()
    if len(words) == 0:
        return True
    
    # Check average word length (gibberish often has very short or very long words)
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < 2 or avg_word_len > 20:
        return True
    
    return False


def is_english(query: str) -> bool:
    """
    Simple English language detection.
    
    Checks if majority of characters are ASCII (English alphabet).
    
    Returns:
        True if likely English, False otherwise
    """
    if len(query) == 0:
        return False
    
    # Count ASCII characters (0-127)
    ascii_chars = sum(1 for c in query if ord(c) < 128)
    ascii_ratio = ascii_chars / len(query)
    
    # At least 80% ASCII for English
    return ascii_ratio >= 0.80


def is_placement_related(query: str) -> Tuple[bool, float]:
    """
    Check if query is related to placements/jobs/companies.
    
    Returns:
        (is_relevant, confidence_score)
    """
    query_lower = query.lower()
    
    # Count keyword matches
    matches = sum(1 for keyword in PLACEMENT_KEYWORDS if keyword in query_lower)
    
    # Calculate confidence score (0.0 to 1.0)
    # 3+ matches = high confidence (1.0)
    confidence = min(matches / 3.0, 1.0)
    
    # Consider relevant if at least 1 keyword
    is_relevant = confidence >= 0.33  # At least 1 keyword match
    
    return is_relevant, confidence


# ============================================================
# MAIN VALIDATION FUNCTION
# ============================================================

def validate_input(
    query: Any,
    min_length: int = MIN_QUERY_LENGTH,
    max_length: int = MAX_QUERY_LENGTH,
    check_relevance: bool = True
) -> ValidationResult:
    """
    Comprehensive input validation pipeline.
    
    Performs all validation checks and sanitization in order:
    1. Type validation
    2. Empty check
    3. Length validation
    4. Security sanitization (HTML, SQL)
    5. Character sanitization
    6. Semantic validation (gibberish, language, relevance)
    
    Args:
        query: User input to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        check_relevance: Whether to check topic relevance
        
    Returns:
        ValidationResult object with status, cleaned query, and metadata
    """
    result = ValidationResult(
        is_valid=True,
        cleaned_query="",
        metadata={}
    )
    
    # ============================================================
    # LAYER 1: Type & Empty Validation
    # ============================================================
    
    is_valid, error = validate_type(query)
    if not is_valid:
        result.is_valid = False
        result.errors.append(error)
        return result
    result.checks_passed.append("type_check")
    
    is_valid, error = validate_empty(query)
    if not is_valid:
        result.is_valid = False
        result.errors.append(error)
        return result
    result.checks_passed.append("empty_check")
    
    # ============================================================
    # LAYER 2: Length Validation
    # ============================================================
    
    is_valid, error = validate_min_length(query, min_length)
    if not is_valid:
        result.is_valid = False
        result.errors.append(error)
        result.cleaned_query = query.strip()
        return result
    result.checks_passed.append("min_length")
    
    is_valid, query, warning = validate_max_length(query, max_length)
    if warning:
        result.warnings.append(warning)
    result.checks_passed.append("max_length")
    
    # ============================================================
    # LAYER 3: Security Sanitization
    # ============================================================
    
    # HTML/XSS prevention
    original_query = query
    query = sanitize_html(query)
    if query != original_query:
        result.warnings.append("HTML tags were removed from your query")
    result.checks_passed.append("html_sanitized")
    
    # SQL injection detection
    has_sql, error = detect_sql_injection(query)
    if has_sql:
        result.is_valid = False
        result.errors.append(error)
        result.cleaned_query = query
        return result
    result.checks_passed.append("sql_injection_check")
    
    # Special character sanitization
    query = sanitize_special_chars(query)
    result.checks_passed.append("special_chars_sanitized")
    
    # ============================================================
    # LAYER 4: Semantic Validation
    # ============================================================
    
    # Gibberish detection
    if is_gibberish(query):
        result.is_valid = False
        result.errors.append("Query appears to be gibberish. Please rephrase with meaningful words.")
        result.cleaned_query = query
        return result
    result.checks_passed.append("gibberish_check")
    
    # Language detection
    if not is_english(query):
        result.warnings.append("Non-English query detected. Results may be less accurate.")
    result.checks_passed.append("language_check")
    
    # Relevance check
    if check_relevance:
        is_relevant, confidence = is_placement_related(query)
        result.metadata['relevance_confidence'] = confidence
        
        if not is_relevant:
            result.warnings.append(
                "Query may not be placement-related. Results might be less accurate."
            )
        result.checks_passed.append("relevance_check")
    
    # ============================================================
    # SUCCESS
    # ============================================================
    
    result.cleaned_query = query
    result.metadata['original_length'] = len(str(original_query))
    result.metadata['cleaned_length'] = len(query)
    
    return result


# ============================================================
# RATE LIMITING
# ============================================================

class RateLimiter:
    """
    Rate limiter to prevent abuse through excessive requests.
    
    Usage:
        limiter = RateLimiter(max_requests=10, time_window=60)
        is_allowed, message = limiter.is_allowed(session_id)
    """
    
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_log: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, session_id: str) -> Tuple[bool, str]:
        """
        Check if request is allowed for this session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            (is_allowed, message)
        """
        current_time = time.time()
        
        # Clean old requests outside time window
        self.request_log[session_id] = [
            req_time for req_time in self.request_log[session_id]
            if current_time - req_time < self.time_window
        ]
        
        # Check rate limit
        request_count = len(self.request_log[session_id])
        
        if request_count >= self.max_requests:
            # Calculate wait time
            oldest_request = self.request_log[session_id][0]
            wait_time = int(self.time_window - (current_time - oldest_request) + 1)
            
            return False, (
                f"⚠️ Rate limit exceeded. "
                f"Please wait {wait_time} seconds before trying again. "
                f"(Limit: {self.max_requests} requests per {self.time_window} seconds)"
            )
        
        # Log this request
        self.request_log[session_id].append(current_time)
        
        return True, ""
    
    def reset(self, session_id: str):
        """Reset rate limit for a specific session."""
        if session_id in self.request_log:
            del self.request_log[session_id]
    
    def get_remaining_requests(self, session_id: str) -> int:
        """Get remaining requests for a session."""
        current_time = time.time()
        
        # Clean old requests
        self.request_log[session_id] = [
            req_time for req_time in self.request_log[session_id]
            if current_time - req_time < self.time_window
        ]
        
        request_count = len(self.request_log[session_id])
        return max(0, self.max_requests - request_count)


# ============================================================
# ERROR MESSAGES
# ============================================================

ERROR_MESSAGES = {
    'empty': "⚠️ Please enter a query to get started.",
    'too_short': "⚠️ Query is too short. Please be more specific (minimum {min_length} characters).",
    'too_long': "⚠️ Query is too long and has been truncated to {max_length} characters.",
    'gibberish': "⚠️ I couldn't understand your query. Please rephrase it with meaningful words.",
    'sql_injection': "⚠️ Invalid characters detected. Please use plain text.",
    'xss_attempt': "⚠️ HTML/script tags are not allowed.",
    'rate_limit': "⚠️ Too many requests. Please wait {wait_time} seconds before trying again.",
    'not_relevant': "💡 Your query doesn't seem placement-related. Results may be less accurate.",
    'non_english': "💡 Non-English queries may not work as expected.",
    'processing_error': "❌ Sorry, I couldn't process your query. Please try again or rephrase.",
    'invalid_type': "⚠️ Invalid input type. Please provide a text query.",
}


def get_friendly_error(error_type: str, **kwargs) -> str:
    """
    Get user-friendly error message.
    
    Args:
        error_type: Type of error
        **kwargs: Format arguments for the message
        
    Returns:
        Formatted error message
    """
    template = ERROR_MESSAGES.get(error_type, ERROR_MESSAGES['processing_error'])
    return template.format(**kwargs)


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def quick_validate(query: str) -> Tuple[bool, str]:
    """
    Quick validation for simple use cases.
    
    Returns:
        (is_valid, cleaned_query_or_error)
    """
    result = validate_input(query)
    
    if result.is_valid:
        return True, result.cleaned_query
    else:
        error_msg = "; ".join(result.errors)
        return False, error_msg


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("INPUT VALIDATOR - TEST SUITE")
    print("=" * 60)
    
    test_cases = [
        ("Compare Amazon vs Google", True, "Valid query"),
        ("ok", False, "Too short"),
        ("<script>alert('XSS')</script>Tell me about Amazon", True, "XSS attempt"),
        ("Amazon'; DROP TABLE companies;--", False, "SQL injection"),
        ("asdfghjkl qwertyuiop", False, "Gibberish"),
        ("What's the weather today?", True, "Not placement-related"),
        ("A" * 2000, True, "Very long query"),
    ]
    
    print("\n🧪 Running test cases...\n")
    
    for i, (query, expected_valid, description) in enumerate(test_cases, 1):
        print(f"Test {i}: {description}")
        print(f"Input: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        result = validate_input(query)
        
        status = "✅ PASS" if result.is_valid == expected_valid else "❌ FAIL"
        print(f"Result: {status}")
        print(f"Valid: {result.is_valid}")
        
        if result.errors:
            print(f"Errors: {result.errors}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        
        print(f"Checks Passed: {len(result.checks_passed)}")
        print("-" * 60)
    
    print("\n✅ Test suite completed!")
