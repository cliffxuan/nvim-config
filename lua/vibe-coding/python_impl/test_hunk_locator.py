# test_hunk_locator.py
# Adjust this import to where you placed the functions under test.
# e.g. from vibe_patch_locate import find_hunk_location, parse_hunk
import pytest
from hunk_locator import find_hunk_location, parse_hunk


@pytest.fixture
def original_text():
    return (
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "def sub(a, b):\n"
        "    return a - b\n"
        "\n"
        "def mul(a, b):\n"
        "    return a * b\n"
        "\n"
        "def div(a, b):\n"
        "    if b == 0:\n"
        "        raise ZeroDivisionError\n"
        "    return a / b\n"
    )


def test_parse_hunk_header_numbers():
    hunk = "@@ -10,3 +20,4 @@\n a\n-b\n+c\n d\n"
    parsed = parse_hunk(hunk)
    hdr = parsed["header"]
    assert hdr["old_start"] == 10
    assert hdr["old_count"] == 3
    assert hdr["new_start"] == 20
    assert hdr["new_count"] == 4
    # old_segment is context+deletions only
    assert parsed["old_segment"] == ["a", "b", "d"]


def test_find_location_exact_header_match(original_text):
    # This hunk modifies 'add' return line; includes the blank line as context
    hunk = (
        "@@ -1,4 +1,4 @@\n"
        " def add(a, b):\n"
        "-    return a + b\n"
        "+    return a + b  # sum\n"
        " \n"  # blank context line
        " def sub(a, b):\n"
    )
    res = find_hunk_location(hunk, original_text)
    assert res is not None
    start, end, score, meta = res
    # In the original, the relevant window is lines [0..4)
    assert (start, end) == (0, 4)
    assert score > 0.9
    assert meta["parsed_header"]["old_start"] == 1
    assert meta["old_segment_len"] == 4


def test_find_location_with_wrong_header_numbers(original_text):
    # Deliberately bogus header; body still points to the first block
    hunk = (
        "@@ -100,4 +200,4 @@\n"
        " def add(a, b):\n"
        "-    return a + b\n"
        "+    return a + b  # sum\n"
        " \n"
        " def sub(a, b):\n"
    )
    start, end, score, meta = find_hunk_location(hunk, original_text)
    assert (start, end) == (0, 4)
    assert score > 0.85  # should still be a strong match despite wrong header
    assert meta["search_scope"] in ("hint-band", "global")


def test_find_location_with_no_header_numbers(original_text):
    # Deliberately bogus header; body still points to the first block
    hunk = (
        "@@ ... @@\n"
        " def add(a, b):\n"
        "-    return a + b\n"
        "+    return a + b  # sum\n"
        " \n"
        " def sub(a, b):\n"
    )
    start, end, score, meta = find_hunk_location(hunk, original_text)
    assert (start, end) == (0, 4)
    assert score > 0.85  # should still be a strong match despite wrong header
    assert meta["search_scope"] in ("hint-band", "global")


def test_find_location_missing_header_entirely(original_text):
    # No header line at all—global scan fallback
    hunk = (
        " def add(a, b):\n"
        "-    return a + b\n"
        "+    return a + b  # sum\n"
        " \n"
        " def sub(a, b):\n"
    )
    start, end, score, meta = find_hunk_location(
        hunk, original_text, trust_header_hint=True
    )
    assert (start, end) == (0, 4)
    assert score > 0.85
    assert meta["parsed_header"]["old_start"] is None
    assert meta["search_scope"] == "global"


def test_whitespace_and_tabs_are_tolerated(original_text):
    # Same region as above, but deletions use tabs; also no space after the diff marker
    hunk = (
        "@@ -1,4 +1,4 @@\n"
        " def add(a, b):\n"
        "-\treturn a + b\n"  # uses a tab; no extra space after '-'
        "+\treturn a + b  # sum\n"
        " \n"
        " def sub(a, b):\n"
    )
    start, end, score, meta = find_hunk_location(hunk, original_text)
    assert (start, end) == (0, 4)
    assert score > 0.80  # whitespace-tolerant matching should still be good


def test_malformed_markers_treated_as_context(original_text):
    # Lines that start with an unexpected marker are treated as context
    hunk = (
        "@@ -4,3 +4,4 @@\n"
        "~def sub(a, b):\n"  # malformed; should be treated as context text
        "-    return a - b\n"
        "+    return a - b  # minus\n"
        " \n"
    )
    start, end, score, meta = find_hunk_location(hunk, original_text)
    # 'sub' block spans lines 3..6 in the original (including trailing blank line)
    assert (start, end) == (3, 6)
    assert score > 0.75


def test_end_index_is_exclusive_and_zero_based(original_text):
    hunk = (
        "@@ -7,3 +7,4 @@\n"
        " def mul(a, b):\n"
        "-    return a * b\n"
        "+    return a * b  # multiply\n"
        " \n"
    )
    start, end, score, meta = find_hunk_location(hunk, original_text)
    # In original_text, 'mul' block lines are:
    # 6: def mul(a, b):
    # 7:     return a * b
    # 8: <blank>
    assert start == 6
    assert end == 9  # end-exclusive; should cover exactly those three lines
    # double-check using the actual slice
    lines = original_text.splitlines()
    assert lines[start:end] == ["def mul(a, b):", "    return a * b", ""]


def test_multiple_similar_regions_respect_header_hint_band_when_present():
    original = (
        "def echo(x):\n"
        "    return x\n"
        "\n"
        "def echo(x):\n"
        "    return x\n"
        "\n"
        "def echo(x):\n"
        "    return x\n"
        "\n"
    )
    # Hunk targets the middle occurrence, but header numbers are slightly off.
    # With a small search_radius, we want it to stay around the hinted middle.
    hunk = "@@ -5,2 +5,2 @@\n def echo(x):\n-    return x\n+    return x  # middle\n"
    # If the function obeys the hint band, it should select the middle block (lines 3..5)
    start, end, score, meta = find_hunk_location(
        hunk, original, trust_header_hint=True, search_radius=3
    )
    assert (start, end) == (3, 5)  # def line + return line
    assert score > 0.85
