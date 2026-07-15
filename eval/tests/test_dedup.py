"""normalize_code/code_hash must treat formatting/case as irrelevant and content as load-bearing."""

from __future__ import annotations

from eval.dedup import code_hash, normalize_code

SNIPPET = """public int add(int a, int b) {
    // adds two numbers
    return a + b;
}"""

REFORMATTED = """public int   add(int a, int b) {
    /* adds
       two numbers */
    return a + b;
}"""

DIFFERENT = """public int subtract(int a, int b) {
    return a - b;
}"""


def test_normalize_code_strips_comments_and_collapses_whitespace() -> None:
    normalized = normalize_code(SNIPPET)
    assert "adds two numbers" not in normalized
    assert "  " not in normalized


def test_normalize_code_is_case_insensitive() -> None:
    assert normalize_code("Return X;") == normalize_code("return x;")


def test_code_hash_matches_across_comment_and_whitespace_reformatting() -> None:
    assert code_hash(SNIPPET) == code_hash(REFORMATTED)


def test_code_hash_differs_for_different_code() -> None:
    assert code_hash(SNIPPET) != code_hash(DIFFERENT)


def test_code_hash_is_deterministic() -> None:
    assert code_hash(SNIPPET) == code_hash(SNIPPET)


def test_normalize_code_preserves_slashes_inside_string_literals() -> None:
    code = 'String url = "http://example.com"; // fetch it'
    normalized = normalize_code(code)
    assert "http://example.com" in normalized
    assert "fetch it" not in normalized


def test_code_hash_distinguishes_strings_that_look_like_comments() -> None:
    first = 'String url = "http://example.com";'
    second = 'String url = "http://other.org";'
    assert code_hash(first) != code_hash(second)


def test_normalize_code_handles_block_comment_marker_inside_string() -> None:
    code = 'String s = "a /* not a comment */ b"; return s;'
    normalized = normalize_code(code)
    assert "not a comment" in normalized


def test_normalize_code_handles_empty_input() -> None:
    assert normalize_code("") == ""
