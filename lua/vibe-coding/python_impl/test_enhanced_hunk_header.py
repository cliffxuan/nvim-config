#!/usr/bin/env python3
"""
Test the enhanced _calculate_hunk_header_from_anchor functionality.
"""

import sys
import traceback
from pathlib import Path

# Add the parent directory to the path to import fix_diff
sys.path.insert(0, str(Path(__file__).parent))

from fix_diff import DiffContext, DiffFixer


def test_calculate_hunk_header_from_anchor_basic():
    """Test basic functionality of enhanced _calculate_hunk_header_from_anchor."""
    fixer = DiffFixer()

    original_lines = [
        "def function():",
        "    return True",
        "",
        "def other():",
        "    pass",
    ]

    diff_lines = [
        " def function():",
        "-    return True",
        "+    return False",
        " ",
        " def other():",
    ]

    context = DiffContext(
        lines=["@@"] + diff_lines,  # Add dummy header
        original_lines=original_lines,
        line_index=0,
        preserve_filenames=False,
    )

    result = fixer._calculate_hunk_header_from_anchor(context, 1)
    expected = "@@ -1,4 +1,4 @@"

    print(f"Result: {result}")
    print(f"Expected: {expected}")
    assert result == expected, f"Expected {expected}, got {result}"


def test_reconcile_lines_from_anchor_simple():
    """Test line-by-line reconciliation with simple diff."""
    fixer = DiffFixer()

    original_lines = ["line1", "line2", "line3"]

    diff_lines = [" line1", "-line2", "+new_line2", " line3"]

    result = fixer._reconcile_lines_from_anchor(1, diff_lines, original_lines)

    print(f"Reconciliation result: {result}")

    assert result["start_line"] == 1
    assert result["orig_count"] == 3  # 2 context + 1 deletion
    assert result["new_count"] == 3  # 2 context + 1 addition


def test_get_line_splits_for_reconciliation():
    """Test line splitting for reconciliation."""
    fixer = DiffFixer()

    original_lines = ["first_part", "second_part", "third_part"]

    # Test line that could be split
    line = "first_partsecond_part"

    splits = fixer._get_line_splits_for_reconciliation(line, original_lines, 0)

    print(f"Splits found: {splits}")

    # Should find at least one valid split
    assert len(splits) > 0

    # Best split should have high confidence
    if splits:
        best_split = splits[0]
        assert best_split["confidence"] > 0.8
        assert len(best_split["parts"]) == 2
        assert best_split["parts"][0] == "first_part"
        assert best_split["parts"][1] == "second_part"


def test_validate_split_against_original():
    """Test split validation against original lines."""
    fixer = DiffFixer()

    original_lines = ["part1", "part2", "part3"]

    # Valid split
    valid_split = {"parts": ["part1", "part2"], "confidence": 0.9}

    assert fixer._validate_split_against_original(valid_split, original_lines, 0)

    # Invalid split
    invalid_split = {"parts": ["wrong1", "wrong2"], "confidence": 0.9}

    assert not fixer._validate_split_against_original(invalid_split, original_lines, 0)


def test_joined_line_detection_integration():
    """Integration test for joined line detection in hunk header calculation."""
    fixer = DiffFixer()

    original_content = """def func1():
    return 1

def func2():
    return 2"""

    # Diff with joined lines
    diff_content = """--- a/test.py
+++ b/test.py
@@ ... @@
 def func1():    return 1
 
 def func2():
+    print("added")
     return 2"""

    result = fixer.fix_diff(diff_content, original_content, "test.py")

    print(f"Fixed diff result:\n{result}")

    # Should have proper hunk header and handle joined lines
    lines = result.strip().split("\n")
    hunk_header = None
    for line in lines:
        if line.startswith("@@"):
            hunk_header = line
            break

    print(f"Found hunk header: {hunk_header}")

    assert hunk_header is not None
    # Should have reasonable line counts (not @@ ... @@)
    assert "..." not in hunk_header


def run_tests():
    """Run all tests."""
    tests = [
        test_calculate_hunk_header_from_anchor_basic,
        test_reconcile_lines_from_anchor_simple,
        test_get_line_splits_for_reconciliation,
        test_validate_split_against_original,
        test_joined_line_detection_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\n=== Running {test.__name__} ===")
            test()
            print(f"✓ {test.__name__} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n=== RESULTS ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
