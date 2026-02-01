"""
🧪 INPUT VALIDATOR TESTS
========================
Comprehensive security and validation tests for input_validator.py

Test Categories:
1. Length validation tests
2. Security tests (XSS, SQL injection)
3. Sanitization tests
4. Semantic validation tests (gibberish, language, relevance)
5. Rate limiting tests
6. Edge cases

Run with: python test_input_validator.py
"""

import sys
import time
from input_validator import (
    validate_input,
    RateLimiter,
    sanitize_html,
    detect_sql_injection,
    is_gibberish,
    is_english,
    is_placement_related,
    quick_validate
)


# ============================================================
# TEST UTILITIES
# ============================================================

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.total = 0
    
    def add_pass(self):
        self.passed += 1
        self.total += 1
    
    def add_fail(self):
        self.failed += 1
        self.total += 1
    
    def print_summary(self):
        print("\n" + "=" * 60)
        print(f"TEST SUMMARY: {self.passed}/{self.total} tests passed")
        if self.failed > 0:
            print(f"❌ {self.failed} tests FAILED")
        else:
            print("✅ All tests PASSED!")
        print("=" * 60)


def run_test(test_name: str, condition: bool, result: TestResult):
    """Run a single test and update results."""
    if condition:
        print(f"✅ {test_name}")
        result.add_pass()
    else:
        print(f"❌ {test_name}")
        result.add_fail()


# ============================================================
# TEST SUITE 1: LENGTH VALIDATION
# ============================================================

def test_length_validation():
    print("\n" + "=" * 60)
    print("TEST SUITE 1: LENGTH VALIDATION")
    print("=" * 60)
    
    result = TestResult()
    
    # Test 1.1: Valid length
    validation = validate_input("Compare Amazon vs Google")
    run_test("Valid query length", validation.is_valid, result)
    
    # Test 1.2: Too short (< 10 chars)
    validation = validate_input("ok")
    run_test("Reject too short query", not validation.is_valid, result)
    
    # Test 1.3: Minimum boundary (exactly 10 chars)
    validation = validate_input("0123456789")
    run_test("Accept minimum length", validation.is_valid, result)
    
    # Test 1.4: Very long query (> 1500 chars)
    long_query = "A" * 2000
    validation = validate_input(long_query)
    run_test("Truncate long query", len(validation.cleaned_query) <= 1500, result)
    run_test("Warn about truncation", len(validation.warnings) > 0, result)
    
    # Test 1.5: Empty query
    validation = validate_input("")
    run_test("Reject empty query", not validation.is_valid, result)
    
    # Test 1.6: Whitespace only
    validation = validate_input("   \n\t   ")
    run_test("Reject whitespace-only query", not validation.is_valid, result)
    
    return result


# ============================================================
# TEST SUITE 2: SECURITY (XSS & SQL INJECTION)
# ============================================================

def test_security():
    print("\n" + "=" * 60)
    print("TEST SUITE 2: SECURITY TESTS")
    print("=" * 60)
    
    result = TestResult()
    
    # Test 2.1: Basic XSS attempt
    xss_query = "<script>alert('XSS')</script>Tell me about Amazon"
    cleaned = sanitize_html(xss_query)
    run_test("Remove <script> tags", "<script>" not in cleaned, result)
    run_test("Preserve content after XSS", "Amazon" in cleaned, result)
    
    # Test 2.2: JavaScript event handler
    xss_query2 = '<img src=x onerror="alert(1)">'
    cleaned2 = sanitize_html(xss_query2)
    run_test("Remove event handlers", "onerror" not in cleaned2.lower(), result)
    
    # Test 2.3: Data URI
    xss_query3 = "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg=="
    cleaned3 = sanitize_html(xss_query3)
    run_test("Remove data URIs", "data:" not in cleaned3, result)
    
    # Test 2.4: SQL injection - DROP TABLE
    sql_query = "Amazon'; DROP TABLE companies;--"
    has_sql, _ = detect_sql_injection(sql_query)
    run_test("Detect DROP TABLE", has_sql, result)
    
    # Test 2.5: SQL injection - UNION SELECT
    sql_query2 = "Amazon' UNION SELECT * FROM users--"
    has_sql2, _ = detect_sql_injection(sql_query2)
    run_test("Detect UNION SELECT", has_sql2, result)
    
    # Test 2.6: SQL injection - OR 1=1
    sql_query3 = "Amazon' OR '1'='1"
    has_sql3, _ = detect_sql_injection(sql_query3)
    run_test("Detect OR 1=1 pattern", has_sql3, result)
    
    # Test 2.7: Legitimate query should pass
    legit_query = "Compare Amazon and Google CTC for ML roles"
    has_sql4, _ = detect_sql_injection(legit_query)
    run_test("Allow legitimate query", not has_sql4, result)
    
    # Test 2.8: Full validation rejects SQL injection
    validation = validate_input("Amazon'; DROP TABLE companies;--")
    run_test("Validation rejects SQL injection", not validation.is_valid, result)
    
    return result


# ============================================================
# TEST SUITE 3: SANITIZATION
# ============================================================

def test_sanitization():
    print("\n" + "=" * 60)
    print("TEST SUITE 3: SANITIZATION")
    print("=" * 60)
    
    result = TestResult()
    
    # Test 3.1: Remove control characters
    query_with_controls = "Compare\x00Amazon\x1F"
    validation = validate_input(query_with_controls)
    run_test("Remove control characters", 
             "\x00" not in validation.cleaned_query and "\x1F" not in validation.cleaned_query, 
             result)
    
    # Test 3.2: Normalize excessive whitespace
    query_spaces = "Compare   Amazon    vs     Google"
    validation = validate_input(query_spaces)
    run_test("Normalize whitespace", 
             "   " not in validation.cleaned_query, 
             result)
    
    # Test 3.3: Limit repeated punctuation
    query_punct = "What is Amazon's CTC????"
    validation = validate_input(query_punct)
    run_test("Limit repeated punctuation", 
             "????" not in validation.cleaned_query, 
             result)
    
    # Test 3.4: Remove zero-width characters
    query_zw = "Compare\u200bAmazon\u200cvs\u200dGoogle"
    validation = validate_input(query_zw)
    run_test("Remove zero-width chars", 
             "\u200b" not in validation.cleaned_query, 
             result)
    
    # Test 3.5: Replace newlines with spaces
    query_newlines = "Compare\nAmazon\nvs\nGoogle"
    validation = validate_input(query_newlines)
    run_test("Replace newlines", 
             "\n" not in validation.cleaned_query and " " in validation.cleaned_query, 
             result)
    
    return result


# ============================================================
# TEST SUITE 4: SEMANTIC VALIDATION
# ============================================================

def test_semantic_validation():
    print("\n" + "=" * 60)
    print("TEST SUITE 4: SEMANTIC VALIDATION")
    print("=" * 60)
    
    result = TestResult()
    
    # Test 4.1: Detect gibberish
    run_test("Detect gibberish (asdfghjkl)", 
             is_gibberish("asdfghjkl qwertyuiop"), 
             result)
    
    # Test 4.2: Accept valid query
    run_test("Accept valid English", 
             not is_gibberish("Compare Amazon vs Google"), 
             result)
    
    # Test 4.3: Detect repeated characters
    run_test("Detect repeated chars", 
             is_gibberish("aaaaaaaaaa"), 
             result)
    
    # Test 4.4: Detect English
    run_test("Detect English query", 
             is_english("Compare Amazon vs Google"), 
             result)
    
    # Test 4.5: Detect non-English
    run_test("Detect non-English", 
             not is_english("比较亚马逊和谷歌"), 
             result)
    
    # Test 4.6: Placement relevance - high confidence
    is_rel, conf = is_placement_related("Amazon CTC for ML roles with 8.5 CGPA")
    run_test("Detect placement-related query", 
             is_rel and conf > 0.5, 
             result)
    
    # Test 4.7: Placement relevance - low confidence
    is_rel2, conf2 = is_placement_related("What's the weather today?")
    run_test("Detect non-placement query", 
             not is_rel2 and conf2 < 0.33, 
             result)
    
    # Test 4.8: Full validation warns about non-placement query
    validation = validate_input("What's the weather in Bangalore?")
    has_relevance_warning = any("placement-related" in w.lower() for w in validation.warnings)
    run_test("Warn about non-placement query", 
             has_relevance_warning, 
             result)
    
    return result


# ============================================================
# TEST SUITE 5: RATE LIMITING
# ============================================================

def test_rate_limiting():
    print("\n" + "=" * 60)
    print("TEST SUITE 5: RATE LIMITING")
    print("=" * 60)
    
    result = TestResult()
    
    # Test 5.1: Create rate limiter
    limiter = RateLimiter(max_requests=3, time_window=10)
    run_test("Create rate limiter", limiter is not None, result)
    
    # Test 5.2: First request allowed
    is_allowed, _ = limiter.is_allowed("session1")
    run_test("First request allowed", is_allowed, result)
    
    # Test 5.3: Second request allowed
    is_allowed2, _ = limiter.is_allowed("session1")
    run_test("Second request allowed", is_allowed2, result)
    
    # Test 5.4: Third request allowed
    is_allowed3, _ = limiter.is_allowed("session1")
    run_test("Third request allowed", is_allowed3, result)
    
    # Test 5.5: Fourth request blocked (exceeds limit)
    is_allowed4, message = limiter.is_allowed("session1")
    run_test("Fourth request blocked", not is_allowed4, result)
    run_test("Block message provided", len(message) > 0, result)
    
    # Test 5.6: Different session not affected
    is_allowed5, _ = limiter.is_allowed("session2")
    run_test("Different session allowed", is_allowed5, result)
    
    # Test 5.7: Get remaining requests
    remaining = limiter.get_remaining_requests("session1")
    run_test("Remaining requests = 0", remaining == 0, result)
    
    # Test 5.8: Reset session
    limiter.reset("session1")
    is_allowed6, _ = limiter.is_allowed("session1")
    run_test("Reset allows new requests", is_allowed6, result)
    
    return result


# ============================================================
# TEST SUITE 6: EDGE CASES
# ============================================================

def test_edge_cases():
    print("\n" + "=" * 60)
    print("TEST SUITE 6: EDGE CASES")
    print("=" * 60)
    
    result = TestResult()
    
    # Test 6.1: None input
    validation = validate_input(None)
    run_test("Reject None input", not validation.is_valid, result)
    
    # Test 6.2: Integer input
    validation = validate_input(12345)
    run_test("Reject integer input", not validation.is_valid, result)
    
    # Test 6.3: List input
    validation = validate_input(["Compare", "Amazon"])
    run_test("Reject list input", not validation.is_valid, result)
    
    # Test 6.4: Query with only numbers
    validation = validate_input("123456789012345")
    run_test("Reject numeric gibberish", not validation.is_valid, result)
    
    # Test 6.5: Query with special characters only
    validation = validate_input("!@#$%^&*()")
    run_test("Reject special char gibberish", not validation.is_valid, result)
    
    # Test 6.6: Unicode query (emojis)
    validation = validate_input("Compare Amazon 🚀 vs Google 🔥")
    run_test("Handle unicode emojis", validation.is_valid, result)
    
    # Test 6.7: Mixed case preservation
    validation = validate_input("Compare Amazon Vs GOOGLE")
    run_test("Preserve case", "Amazon" in validation.cleaned_query, result)
    
    # Test 6.8: Quick validate convenience function
    is_valid, result_or_error = quick_validate("Compare Amazon vs Google")
    run_test("Quick validate success", is_valid, result)
    
    # Test 6.9: Quick validate failure
    is_valid2, result_or_error2 = quick_validate("ok")
    run_test("Quick validate failure", not is_valid2, result)
    
    return result


# ============================================================
# TEST SUITE 7: INTEGRATION TESTS
# ============================================================

def test_integration():
    print("\n" + "=" * 60)
    print("TEST SUITE 7: INTEGRATION TESTS")
    print("=" * 60)
    
    result = TestResult()
    
    # Test 7.1: Valid placement query - full pipeline
    query = "Compare Amazon vs Google for ML roles with 8.5 CGPA"
    validation = validate_input(query)
    
    run_test("Valid query passes all checks", validation.is_valid, result)
    run_test("No errors for valid query", len(validation.errors) == 0, result)
    run_test("Cleaned query similar to original", 
             len(validation.cleaned_query) >= len(query) * 0.9, 
             result)
    run_test("Multiple checks passed", 
             len(validation.checks_passed) >= 6, 
             result)
    
    # Test 7.2: XSS attempt - blocked but content preserved
    query2 = "<script>alert(1)</script>Tell me about Amazon placement"
    validation2 = validate_input(query2)
    
    run_test("XSS query still valid after cleaning", validation2.is_valid, result)
    run_test("Script tag removed", "<script>" not in validation2.cleaned_query, result)
    run_test("Content preserved", "Amazon" in validation2.cleaned_query, result)
    run_test("Warning issued", len(validation2.warnings) > 0, result)
    
    # Test 7.3: Complex validation metadata
    query3 = "What is Intel's CTC package?"
    validation3 = validate_input(query3)
    
    run_test("Metadata includes original length", 
             'original_length' in validation3.metadata, 
             result)
    run_test("Metadata includes cleaned length", 
             'cleaned_length' in validation3.metadata, 
             result)
    run_test("Relevance confidence calculated", 
             'relevance_confidence' in validation3.metadata, 
             result)
    
    return result


# ============================================================
# MAIN TEST RUNNER
# ============================================================

def run_all_tests():
    """Run all test suites and print summary."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "INPUT VALIDATOR - TEST SUITE" + " " * 20 + "║")
    print("╚" + "=" * 58 + "╝")
    
    all_results = []
    
    # Run all test suites
    all_results.append(test_length_validation())
    all_results.append(test_security())
    all_results.append(test_sanitization())
    all_results.append(test_semantic_validation())
    all_results.append(test_rate_limiting())
    all_results.append(test_edge_cases())
    all_results.append(test_integration())
    
    # Aggregate results
    total_result = TestResult()
    for res in all_results:
        total_result.passed += res.passed
        total_result.failed += res.failed
        total_result.total += res.total
    
    # Print final summary
    total_result.print_summary()
    
    # Return exit code
    return 0 if total_result.failed == 0 else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
