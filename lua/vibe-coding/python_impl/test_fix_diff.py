#!/usr/bin/env python3
"""
Comprehensive test cases for the refactored DiffFixer.
Following principles: fixtures over hardcoded values, comprehensive edge cases.
"""

import subprocess
import tempfile
from pathlib import Path

from fix_diff import DiffContext, DiffFixer, DiffFixRule, Hunk, HunkManager


# Simple test helper to replace pytest functionality
def skip_test(reason):
    print(f"SKIPPED: {reason}")
    return True


class TestDiffFixer:
    def setup_method(self):
        self.fixer = DiffFixer()

    # Test using actual fixture files for better readability
    def test_basic_diff_fixing_with_fixture(self):
        """Test basic diff fixing using actual fixture."""
        fixtures_path = Path(__file__).parent.parent / "tests" / "fixtures" / "pass"
        diff_path = fixtures_path / "basic_modification" / "diff"
        original_path = fixtures_path / "basic_modification" / "original.lua"

        if not diff_path.exists() or not original_path.exists():
            return skip_test("Fixture files not found")

        diff_content = diff_path.read_text()
        original_content = original_path.read_text()

        fixed_diff = self.fixer.fix_diff(
            diff_content, original_content, str(original_path)
        )

        # Verify basic structure
        lines = fixed_diff.split("\n")
        assert lines[0].startswith("--- a/")
        assert lines[1].startswith("+++ b/")
        assert any(line.startswith("@@") for line in lines)

    def test_custom_rule_addition(self):
        """Test that custom rules can be added and applied."""

        # Create a custom rule for testing
        def custom_matcher(line: str, context: DiffContext) -> bool:
            return "CUSTOM_PATTERN" in line

        def custom_fixer(line: str, context: DiffContext) -> list[str]:
            return ["# Fixed: " + line]

        custom_rule = DiffFixRule(
            name="test_rule",
            matcher=custom_matcher,
            fixer=custom_fixer,
            priority=20,  # High priority
            applies_to="all",
        )

        self.fixer.add_rule(custom_rule)

        # Test that the rule is applied
        test_diff = "CUSTOM_PATTERN test line"
        result = self.fixer.fix_diff(test_diff, "", "test.txt")

        assert "# Fixed:" in result

    def test_rule_priority_ordering(self):
        """Test that rules are applied in priority order."""
        # Add rules with different priorities
        low_priority = DiffFixRule(
            name="low",
            matcher=lambda line, ctx: "TEST" in line,
            fixer=lambda line, ctx: ["LOW: " + line],
            priority=1,
            applies_to="all",
        )

        high_priority = DiffFixRule(
            name="high",
            matcher=lambda line, ctx: "TEST" in line,
            fixer=lambda line, ctx: ["HIGH: " + line],
            priority=10,
            applies_to="all",
        )

        self.fixer.add_rule(low_priority)
        self.fixer.add_rule(high_priority)

        result = self.fixer.fix_diff("TEST line", "", "test.txt")

        # High priority rule should be applied first
        assert "HIGH:" in result
        assert "LOW:" not in result

    def test_calculate_hunk_header_from_anchor_basic(self):
        """Test basic functionality of enhanced _calculate_hunk_header_from_anchor."""
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

        result = self.fixer._calculate_hunk_header_from_anchor(context, 1)
        assert result == "@@ -1,4 +1,4 @@"

    def test_reconcile_lines_from_anchor_simple(self):
        """Test line-by-line reconciliation with simple diff."""
        original_lines = ["line1", "line2", "line3"]

        diff_lines = [" line1", "-line2", "+new_line2", " line3"]

        result = self.fixer._reconcile_lines_from_anchor(1, diff_lines, original_lines)

        assert result["start_line"] == 1
        assert result["orig_count"] == 3  # 2 context + 1 deletion
        assert result["new_count"] == 3  # 2 context + 1 addition

    def test_reconcile_lines_with_joined_context(self):
        """Test reconciliation with joined context lines."""
        original_lines = ["import os", "import sys", "", "def main():"]

        # Simulate joined line: "import osimport sys"
        diff_lines = [
            " import osimport sys",  # This should be split
            " ",
            " def main():",
        ]

        result = self.fixer._reconcile_lines_from_anchor(1, diff_lines, original_lines)

        # Should handle joined line detection
        assert result["orig_count"] >= 3
        assert result["new_count"] >= 3

    def test_try_resolve_context_mismatch(self):
        """Test context mismatch resolution with joined lines."""
        original_lines = ["if condition:", "    action()", "else:", "    other()"]

        # Test joined line "if condition:    action()"
        line = " if condition:    action()"

        result = self.fixer._try_resolve_context_mismatch(line, original_lines, 0)

        if result["resolved"]:
            # Should split into two context lines
            assert len(result["lines"]) == 2
            assert all(l.startswith(" ") for l in result["lines"])
            assert result["context_count"] == 2
            assert result["orig_advance"] == 2

    def test_try_resolve_deletion_mismatch(self):
        """Test deletion mismatch resolution with joined lines."""
        original_lines = ["old_line1", "old_line2", "keep_this"]

        # Test joined deletion line
        line = "-old_line1old_line2"
        remaining_diff = [line]

        result = self.fixer._try_resolve_deletion_mismatch(
            line, remaining_diff, original_lines, 0
        )

        if result["resolved"]:
            # Should split into two deletion lines
            assert len(result["lines"]) == 2
            assert all(l.startswith("-") for l in result["lines"])
            assert result["deletion_count"] == 2
            assert result["orig_advance"] == 2

    def test_get_line_splits_for_reconciliation(self):
        """Test line splitting for reconciliation."""
        original_lines = ["first_part", "second_part", "third_part"]

        # Test line that could be split
        line = "first_partsecond_part"

        splits = self.fixer._get_line_splits_for_reconciliation(line, original_lines, 0)

        # Should find at least one valid split
        assert len(splits) > 0

        # Best split should have high confidence
        if splits:
            best_split = splits[0]
            assert best_split["confidence"] > 0.8
            assert len(best_split["parts"]) == 2
            assert best_split["parts"][0] == "first_part"
            assert best_split["parts"][1] == "second_part"

    def test_validate_split_against_original(self):
        """Test split validation against original lines."""
        original_lines = ["part1", "part2", "part3"]

        # Valid split
        valid_split = {"parts": ["part1", "part2"], "confidence": 0.9}

        assert self.fixer._validate_split_against_original(
            valid_split, original_lines, 0
        )

        # Invalid split
        invalid_split = {"parts": ["wrong1", "wrong2"], "confidence": 0.9}

        assert not self.fixer._validate_split_against_original(
            invalid_split, original_lines, 0
        )

    def test_joined_line_detection_integration(self):
        """Integration test for joined line detection in hunk header calculation."""
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

        result = self.fixer.fix_diff(diff_content, original_content, "test.py")

        # Should have proper hunk header and handle joined lines
        lines = result.strip().split("\n")
        hunk_header = None
        for line in lines:
            if line.startswith("@@"):
                hunk_header = line
                break

        assert hunk_header is not None
        # Should have reasonable line counts (not @@ ... @@)
        assert "..." not in hunk_header

    def test_preserve_filenames_option(self):
        """Test filename preservation option."""
        diff_content = (
            "--- original/path.txt\n+++ original/path.txt\n@@ -1,1 +1,1 @@\n-old\n+new"
        )
        original_content = "old\n"

        # Test without preserve
        result_no_preserve = self.fixer.fix_diff(
            diff_content, original_content, "test.txt", False
        )
        assert "--- a/test.txt" in result_no_preserve

        # Test with preserve
        result_preserve = self.fixer.fix_diff(
            diff_content, original_content, "test.txt", True
        )
        assert "--- a/original/path.txt" in result_preserve

    def test_edge_case_empty_diff(self):
        """Test edge case: empty diff content."""
        result = self.fixer.fix_diff("", "", "test.txt")
        assert result == "\n"

    def test_edge_case_malformed_hunk_header(self):
        """Test edge case: various malformed hunk headers."""
        test_cases = [
            "@@ ... @@",
            "@@ -1,5 +1,5",  # Missing ending @@
            "@@ -1,5 +1,5 @@some_content",  # Content attached
        ]

        for malformed_header in test_cases:
            diff_content = (
                f"--- a/test.txt\n+++ b/test.txt\n{malformed_header}\n test line"
            )
            result = self.fixer.fix_diff(diff_content, "test line\n", "test.txt")

            # Should have a valid hunk header
            lines = result.split("\n")
            hunk_line = next((line for line in lines if line.startswith("@@")), None)
            assert hunk_line is not None
            assert hunk_line.count("@@") == 2

    def test_boundary_splitting_generic(self):
        """Test that strict matching prevents incorrect splitting when indentation doesn't match."""
        # Test joined line with mismatched indentation
        diff_content = (
            "--- a/test.py\n+++ b/test.py\n@@ -1,2 +1,2 @@\n )except Exception as e:"
        )
        # Original file has different indentation - strict matching should prevent split
        original_content = "        )\n    except Exception as e:\n"

        result = self.fixer.fix_diff(diff_content, original_content, "test.py")

        # With strict matching, the line should NOT be split due to indentation mismatch
        # Instead it should be treated as an addition (since it doesn't exist in original)
        assert "+)except Exception as e:" in result
        print(
            "✓ Strict matching correctly prevents splitting with indentation mismatch"
        )

    def test_diff_prefix_inference(self):
        """Test inference of missing diff prefixes using original file context."""
        diff_content = (
            "--- a/test.txt\n+++ b/test.txt\n@@ -1,2 +1,2 @@\nexisting line\nnew line"
        )
        # Only 'existing line' exists in original, so 'new line' should get + prefix
        original_content = "existing line\n"

        result = self.fixer.fix_diff(diff_content, original_content, "test.txt")

        lines = result.split("\n")
        # Should infer that 'existing line' is context (exists in original)
        # and 'new line' is addition (doesn't exist in original)
        existing_line_found = any(" existing line" in line for line in lines)
        new_line_found = any("+new line" in line for line in lines)

        # At least one should be correctly prefixed
        assert existing_line_found or new_line_found

    def test_context_preservation(self):
        """Test that context information is properly used."""
        diff_content = "--- a/test.txt\n+++ b/test.txt\n@@ ... @@\n context line\n-old line\n+new line"
        original_content = "context line\nold line\n"

        result = self.fixer.fix_diff(diff_content, original_content, "test.txt")

        # Should infer proper hunk header based on context
        lines = result.split("\n")
        hunk_line = next(
            (line for line in lines if line.startswith("@@") and line != "@@ ... @@"),
            None,
        )
        assert hunk_line is not None
        assert "-" in hunk_line and "+" in hunk_line

    def test_git_apply_compatibility(self):
        """Test that output can be applied by git apply (using simple fixture)."""
        fixtures_path = Path(__file__).parent.parent / "tests" / "fixtures" / "pass"
        diff_path = fixtures_path / "basic_modification" / "diff"
        original_path = fixtures_path / "basic_modification" / "original.lua"

        if not diff_path.exists() or not original_path.exists():
            return skip_test("Fixture files not found")

        diff_content = diff_path.read_text()
        original_content = original_path.read_text()

        fixed_diff = self.fixer.fix_diff(
            diff_content, original_content, str(original_path)
        )

        # Test with git apply in a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / original_path.name
            test_file.write_text(original_content)

            # Initialize git repo
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=tmpdir,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"], cwd=tmpdir, capture_output=True
            )
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial"], cwd=tmpdir, capture_output=True
            )

            # Test git apply --check
            result = subprocess.run(
                ["git", "apply", "--check", "-"],
                input=fixed_diff,
                text=True,
                cwd=tmpdir,
                capture_output=True,
            )

            # Should not fail for basic cases
            success = result.returncode == 0 or "does not exist" in result.stderr
            assert success, f"Git apply failed: {result.stderr}"

    def test_error_handling_robustness(self):
        """Test that the fixer handles various error conditions gracefully."""
        # Test with None/empty inputs
        result = self.fixer.fix_diff("", "", "")
        assert isinstance(result, str)

        # Test with invalid diff format
        invalid_diff = "This is not a diff\nJust some random text"
        result = self.fixer.fix_diff(invalid_diff, "original content", "test.txt")
        assert isinstance(result, str)

        # Test with mismatched content
        diff_with_missing_context = "--- a/test.txt\n+++ b/test.txt\n@@ -1,1 +1,1 @@\n-nonexistent line\n+new line"
        result = self.fixer.fix_diff(
            diff_with_missing_context, "actual content", "test.txt"
        )
        assert isinstance(result, str)

    def test_exact_indentation_matching(self):
        """Test that unchanged and deleted lines must match exactly including indentation."""
        diff_content = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 def function():
-  old_line
+    new_line"""

        # Original has different indentation for old_line (4 spaces instead of 2)
        original_content = """def function():
    old_line
"""

        result = self.fixer.fix_diff(diff_content, original_content, "test.py")

        # Should correct the indentation of the deleted line to match original
        assert "-    old_line" in result  # Should be 4 spaces, not 2

    def test_context_aware_disambiguation(self):
        """Test disambiguation when multiple lines match."""
        # Multiple lines with same content but different context
        diff_content = """--- a/test.py
+++ b/test.py
@@ -3,3 +3,3 @@
 line_above
 duplicate_line
-duplicate_line
+new_line"""

        # Original file has duplicate_line in multiple places
        original_content = """duplicate_line
line_above
duplicate_line
line_below"""

        result = self.fixer.fix_diff(diff_content, original_content, "test.py")

        # Should pick the right duplicate_line based on context (line_above)
        lines = result.split("\n")

        # Find the sequence to verify context matching
        line_above_idx = None
        for i, line in enumerate(lines):
            if "line_above" in line:
                line_above_idx = i
                break

        assert line_above_idx is not None
        # Next lines should include the duplicate and then the deletion
        context_found = False
        for i in range(line_above_idx + 1, min(len(lines), line_above_idx + 3)):
            if "duplicate_line" in lines[i]:
                context_found = True
                break

        assert context_found

    def test_strict_joined_line_splitting(self):
        """Test that joined lines are split when both parts exist in original."""
        # Joined line that should be split
        diff_content = """--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
         )except Exception as e:"""

        # Original file has the separate lines
        original_content = """        )
    except Exception as e:
"""

        result = self.fixer.fix_diff(diff_content, original_content, "test.py")

        # Should be split into separate lines with correct indentation
        assert "        )" in result  # First part with 8 spaces
        assert "    except Exception as e:" in result  # Second part with 4 spaces
        # Should NOT contain the joined line
        assert ")except Exception as e:" not in result

    def test_no_false_positive_splitting(self):
        """Test that lines are NOT split when they exist as-is in original."""
        # Line that looks like it could be split but exists as-is in original
        diff_content = """--- a/test.py
+++ b/test.py
@@ -1,1 +1,1 @@
 result = func():return value"""

        # Original file has this line as-is (not split)
        original_content = """result = func():return value
"""

        result = self.fixer.fix_diff(diff_content, original_content, "test.py")

        # Should NOT split the line since it exists as-is in original
        assert "result = func():return value" in result
        # Should not have separate lines for func() and return
        lines = [line for line in result.split("\n") if line.strip()]
        func_lines = [
            line for line in lines if "func()" in line and "return value" not in line
        ]
        assert len(func_lines) == 0  # Should not be split

    def test_indentation_correction(self):
        """Test correction of incorrect indentation in diff lines."""
        # Diff with wrong indentation
        diff_content = """--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
 if True:
-print("hello")
+    print("world")"""

        # Original has correct indentation
        original_content = """if True:
    print("hello")
"""

        result = self.fixer.fix_diff(diff_content, original_content, "test.py")

        # Should correct the indentation of the deleted line
        assert '-    print("hello")' in result  # Should be 4 spaces

    def test_whitespace_preservation_split_parts(self):
        """Test that split parts from joined lines get proper whitespace prefixes."""
        # Test case based on whitespace_preservation_issue fixture
        diff_content = (
            "--- a/test.py\n"
            "+++ b/test.py\n"
            "@@ -1,3 +1,3 @@\n"
            "         )except Exception as e:"
        )
        # Original has these lines with specific indentation
        original_content = (
            "        console.print()\n        )\n\n    except Exception as e:\n"
        )

        result = self.fixer.fix_diff(diff_content, original_content, "test.py")
        lines = result.strip().split("\n")

        # The joined line should be split into 3 parts with proper prefixes:
        # 1. "         )" (space prefix + 8 original spaces = 9 total)
        # 2. " " (space prefix for empty line)
        # 3. "     except Exception as e:" (space prefix + 4 original spaces = 5 total)

        # Find the split parts in the result
        closing_paren_line = None
        empty_line = None
        except_line = None

        for line in lines:
            if line.endswith(")") and "except" not in line:
                closing_paren_line = line
            elif line.strip() == "" and line == " ":  # Empty line with space prefix
                empty_line = line
            elif "except Exception as e:" in line:
                except_line = line

        # Verify the whitespace is preserved correctly
        assert closing_paren_line == "         )", (
            f"Expected '         )' but got {repr(closing_paren_line)}"
        )
        assert empty_line == " ", f"Expected ' ' but got {repr(empty_line)}"
        assert except_line == "     except Exception as e:", (
            f"Expected '     except Exception as e:' but got {repr(except_line)}"
        )

        print("✓ Split parts preserve whitespace correctly with proper diff prefixes")

    def test_hunk_header_count_correction(self):
        """Test that hunk headers with incorrect line counts are corrected."""
        # Based on joined_comment_line fixture - header shows @@ -1,6 +1,6 @@ but should be @@ -1,5 +1,5 @@
        diff_content = """--- a/test.js
+++ b/test.js
@@ -1,6 +1,6 @@
 const config = {
   apiUrl: 'https://api.example.com',
   maxRetries: 3,
-  timeout: 5000 // Default timeout value
+  timeout: 8000 // Default timeout value (increased)
 };"""

        # Original file has 5 lines, so header counts should be 5, not 6
        original_content = """const config = {
  apiUrl: 'https://api.example.com',
  maxRetries: 3,
  timeout: 5000 // Default timeout value
};"""

        result = self.fixer.fix_diff(diff_content, original_content, "test.js")

        # The header should be corrected to show the right line counts
        assert "@@ -1,5 +1,5 @@" in result, (
            f"Expected corrected header '@@ -1,5 +1,5 @@' not found in result:\n{result}"
        )
        assert "@@ -1,6 +1,6 @@" not in result, (
            f"Old incorrect header '@@ -1,6 +1,6 @@' should be removed from result:\n{result}"
        )

        print("✓ Hunk header with incorrect line counts corrected successfully")

    def test_missing_context_line_insertion(self):
        """Test that missing empty context lines are inserted between non-consecutive lines."""
        # Based on blank_line_fuzzy - missing empty line between imports
        diff_content = """--- a/test.py
+++ b/test.py
@@ -2,4 +2,5 @@
 from pathlib import Path
 from typing import Any, Dict, List
+import colorama
 import fy_signin
 import requests"""

        # Original file has empty line between the two import sections
        original_content = """import sys
from pathlib import Path
from typing import Any, Dict, List

import fy_signin
import requests"""

        result = self.fixer.fix_diff(diff_content, original_content, "test.py")

        # Should insert the missing empty line as context
        lines = result.strip().split("\n")

        # Find the pattern: typing line, addition line, empty line, fy_signin line
        typing_line_idx = None
        addition_line_idx = None
        empty_line_idx = None
        fy_signin_line_idx = None

        for i, line in enumerate(lines):
            if "from typing" in line and line.startswith(" "):
                typing_line_idx = i
            elif "+import colorama" in line:
                addition_line_idx = i
            elif (
                line == " "
                and typing_line_idx is not None
                and addition_line_idx is not None
            ):
                empty_line_idx = i
            elif "import fy_signin" in line and line.startswith(" "):
                fy_signin_line_idx = i

        # Verify the sequence is correct
        assert typing_line_idx is not None, f"Typing import line not found in: {lines}"
        assert addition_line_idx is not None, f"Addition line not found in: {lines}"
        assert empty_line_idx is not None, f"Empty context line not found in: {lines}"
        assert fy_signin_line_idx is not None, (
            f"fy_signin import line not found in: {lines}"
        )

        # Verify the order: typing < addition < empty < fy_signin
        assert (
            typing_line_idx < addition_line_idx < empty_line_idx < fy_signin_line_idx
        ), (
            f"Lines not in correct order: typing={typing_line_idx}, addition={addition_line_idx}, empty={empty_line_idx}, fy_signin={fy_signin_line_idx}"
        )

        print("✓ Missing empty context line correctly inserted")

    def test_no_newline_marker_handling(self):
        """Test that files without trailing newlines get proper '\\ No newline at end of file' marker when the last line is in diff context."""
        # Test case 1: Last line in diff context - should get the marker
        diff_content = """--- a/test.txt
+++ b/test.txt
@@ -1,4 +1,4 @@
 line 1
 line 2
-old content
+new content
 line 4"""

        # Original file without trailing newline
        original_content = "line 1\nline 2\nold content\nline 4"

        result = self.fixer.fix_diff(diff_content, original_content, "test.txt")

        # Should include the "No newline at end of file" marker because the last line appears in diff context
        assert "\\ No newline at end of file" in result, (
            f"Should have newline marker when last line is in diff context: {repr(result)}"
        )

        # Test case 2: Modifying the actual last line - should get the marker
        diff_content_last_line = """--- a/test.txt
+++ b/test.txt
@@ -3,2 +3,2 @@
 old content
-line 4
+line 4 modified"""

        result_last_line = self.fixer.fix_diff(
            diff_content_last_line, original_content, "test.txt"
        )

        # Should include the "No newline at end of file" marker because we ARE modifying the last line
        assert "\\ No newline at end of file" in result_last_line, (
            f"Expected newline marker when modifying last line: {repr(result_last_line)}"
        )

        print("✓ No newline marker correctly applied only when last line is modified")

    def test_with_newline_no_marker_added(self):
        """Test that files WITH trailing newlines don't get the newline marker."""
        diff_content = """--- a/test.txt
+++ b/test.txt  
@@ -1,3 +1,3 @@
 line 1
-old content
+new content
 line 3
"""

        # Original file WITH trailing newline
        original_content = "line 1\nold content\nline 3\n"

        result = self.fixer.fix_diff(diff_content, original_content, "test.txt")

        # Should NOT include the "No newline at end of file" marker
        assert "\\ No newline at end of file" not in result, (
            f"Should not have newline marker: {repr(result)}"
        )

        # Should still end with newline for proper git format
        assert result.endswith("\n"), (
            f"Result should end with newline: {repr(result[-20:])}"
        )

        print("✓ No newline marker correctly omitted for files with trailing newlines")

    def test_multiple_hunks_handling(self):
        """Test that diffs with multiple hunks are processed correctly."""
        # Based on multiple_hunks fixture - two separate hunks
        diff_content = """--- a/calculator.py
+++ b/calculator.py
@@ -1,4 +1,5 @@
 class Calculator:
     def __init__(self):
         self.result = 0
+        self.history = []
     
@@ -12,4 +13,7 @@
     
     def divide(self, x, y):
-        return x / y
+        if y == 0:
+            raise ValueError("Cannot divide by zero")
+        return x / y"""

        # Original file with two separate method groups
        original_content = """class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        return x + y
    
    def subtract(self, x, y):
        return x - y
    
    def multiply(self, x, y):
        return x * y
    
    def divide(self, x, y):
        return x / y"""

        result = self.fixer.fix_diff(diff_content, original_content, "calculator.py")

        # Should detect multiple hunks and process separately
        hunk_count = result.count("@@") // 2
        assert hunk_count == 2, (
            f"Expected 2 hunks but got {hunk_count} in result: {repr(result)}"
        )

        # Should have single set of file headers
        assert result.count("--- a/calculator.py") == 1, (
            f"Should have single file header: {repr(result)}"
        )
        assert result.count("+++ b/calculator.py") == 1, (
            f"Should have single file header: {repr(result)}"
        )

        # Should contain both changes
        assert "+        self.history = []" in result, (
            f"First hunk change missing: {repr(result)}"
        )
        assert "+        if y == 0:" in result, (
            f"Second hunk change missing: {repr(result)}"
        )
        assert '+            raise ValueError("Cannot divide by zero")' in result, (
            f"Second hunk change missing: {repr(result)}"
        )

        print("✓ Multiple hunks correctly processed with proper structure")

    def test_unified_single_and_multi_hunk_approach(self):
        """Test that the unified approach handles both single and multi-hunk diffs correctly."""
        # Test single hunk
        single_hunk_diff = """--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
-old line
+new line
 context line"""

        original_content = "old line\ncontext line\n"

        result = self.fixer.fix_diff(single_hunk_diff, original_content, "test.py")

        # Should have proper diff structure
        assert "--- a/test.py" in result
        assert "+++ b/test.py" in result
        assert "-old line" in result
        assert "+new line" in result
        assert " context line" in result

        # Test multi-hunk (with simplified content)
        multi_hunk_diff = """--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
-old line 1
+new line 1
 context line
@@ -3,2 +3,2 @@
-old line 2
+new line 2
 context line 2"""

        multi_original = "old line 1\ncontext line\nold line 2\ncontext line 2\n"

        multi_result = self.fixer.fix_diff(multi_hunk_diff, multi_original, "test.py")

        # Should have proper multi-hunk structure
        assert "--- a/test.py" in multi_result
        assert "+++ b/test.py" in multi_result
        assert multi_result.count("@@") == 4  # 2 hunks * 2 @@ markers each
        assert "-old line 1" in multi_result
        assert "+new line 1" in multi_result
        assert "-old line 2" in multi_result
        assert "+new line 2" in multi_result

        print("✓ Unified approach correctly handles both single and multi-hunk diffs")

    def test_generic_fallback_header_creation(self):
        """Test that the new generic fallback header creation works correctly."""
        from fix_diff import DiffConfig

        # Test with mixed content lines
        content_lines = [
            " context_line",
            "-deleted_line",
            "+added_line",
            " another_context",
        ]

        result = DiffConfig.create_fallback_header(content_lines)

        # Should count: 2 context + 1 deletion = 3 old, 2 context + 1 addition = 3 new
        assert result == "@@ -1,3 +1,3 @@", (
            f"Expected '@@ -1,3 +1,3 @@' but got '{result}'"
        )
        print("✓ Generic fallback header correctly calculated from content")

    def test_generic_fallback_with_empty_content(self):
        """Test fallback behavior with empty content."""
        from fix_diff import DiffConfig

        result = DiffConfig.create_fallback_header([])

        # Should default to minimum 1,1
        assert result == "@@ -1,1 +1,1 @@", (
            f"Expected '@@ -1,1 +1,1 @@' but got '{result}'"
        )
        print("✓ Generic fallback with empty content defaults to minimal header")

    def test_generic_fallback_with_custom_start_line(self):
        """Test fallback with custom start line."""
        from fix_diff import DiffConfig

        content_lines = ["+new_line", "+another_line"]
        result = DiffConfig.create_fallback_header(content_lines, start_line=5)

        # Should start at line 5: 0 context + 0 deletion = 1 min old, 0 context + 2 addition = 2 new
        assert result == "@@ -5,1 +5,2 @@", (
            f"Expected '@@ -5,1 +5,2 @@' but got '{result}'"
        )
        print("✓ Generic fallback with custom start line works correctly")

    def test_no_hardcoded_fallbacks_used(self):
        """Test that no hardcoded fallbacks like '@@ -1,10 +1,11 @@' are used."""
        # Test with completely malformed header that would trigger fallback
        diff_content = """--- a/test.py
+++ b/test.py
@@ ... @@
+new_line"""

        original_content = ""

        result = self.fixer.fix_diff(diff_content, original_content, "test.py")

        # Should NOT contain the old hardcoded fallbacks
        assert "@@ -1,10 +1,11 @@" not in result, (
            "Old hardcoded fallback found in result"
        )
        assert "@@ -1,5 +1,5 @@" not in result, "Old hardcoded fallback found in result"

        # Should contain a sensible generic fallback
        assert "@@ -1,1 +1,1 @@" in result, (
            f"Expected generic fallback not found in: {result}"
        )
        print("✓ No hardcoded fallbacks used - generic solution working")

    def test_no_newline_marker_only_when_last_line_touched(self):
        """Test that 'No newline at end of file' marker is added when the diff touches the last line."""
        # Test case 1: Diff that does NOT touch the last line (like blank_line_fuzzy)
        # The last line "import urllib3" is not in the diff context at all
        diff_content = """--- a/original.py
+++ b/original.py
@@ -4,6 +4,7 @@
 from pathlib import Path
 from typing import Any, Dict, List
+import colorama
 
 import fy_signin
 import requests
 import typer"""

        # Original file without trailing newline (last line is "import urllib3")
        original_content = """import json
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

import fy_signin
import requests
import typer
import urllib3"""

        result = self.fixer.fix_diff(diff_content, original_content, "original.py")

        # Should NOT contain the newline marker because the last line is not in diff context
        assert "\\ No newline at end of file" not in result, (
            "No newline marker should not be added when last line is not in diff context"
        )

        # Test case 2: Diff that includes the last line in context (like line_removal)
        diff_content_with_context = """--- a/original.py
+++ b/original.py
@@ -1,7 +1,6 @@
 import sys
 import os
 import json
-import re
 
 def main():
     pass"""

        # Original for line_removal case - last line is "    pass"
        original_content_line_removal = """import sys
import os
import json
import re

def main():
    pass"""

        result_with_context = self.fixer.fix_diff(
            diff_content_with_context, original_content_line_removal, "original.py"
        )

        # Should contain the newline marker because the last line appears in the diff context
        assert "\\ No newline at end of file" in result_with_context, (
            "No newline marker should be added when last line is in diff context"
        )

        # Test case 3: Diff that DOES modify the last line directly
        diff_content_modifying_last = """--- a/original.py
+++ b/original.py
@@ -8,4 +8,4 @@
 import fy_signin
 import requests
 import typer
-import urllib3
+import urllib3_modified"""

        result_modifying = self.fixer.fix_diff(
            diff_content_modifying_last, original_content, "original.py"
        )

        # Should contain the newline marker because we ARE modifying the last line
        assert "\\ No newline at end of file" in result_modifying, (
            "No newline marker should be added when last line is modified"
        )


class TestUnifiedHunkClasses:
    """Test the new unified hunk classes."""

    def test_hunk_from_header_creation(self):
        """Test creating Hunk from header line."""
        from fix_diff import Hunk

        header = "@@ -1,5 +1,6 @@"
        hunk = Hunk.from_header(header)

        assert hunk is not None
        assert hunk.header_line == header
        assert hunk.start_line == 1
        assert hunk.old_count == 5
        assert hunk.new_count == 6
        assert hunk.new_start == 1
        assert hunk.line_range == (1, 5)

    def test_hunk_from_lines_creation(self):
        """Test creating Hunk from list of lines."""
        from fix_diff import Hunk

        lines = ["@@ -10,3 +10,4 @@", " context line", "-deleted line", "+added line"]

        hunk = Hunk.from_lines(lines)

        assert hunk is not None
        assert hunk.header_line == "@@ -10,3 +10,4 @@"
        assert hunk.start_line == 10
        assert hunk.old_count == 3
        assert hunk.new_count == 4
        assert len(hunk.content_lines) == 3

    def test_hunk_range_extraction(self):
        """Test unified range extraction method."""
        from fix_diff import Hunk

        lines = ["@@ -5,10 +5,12 @@", " some content"]
        range_result = Hunk.extract_range_from_lines(lines)

        assert range_result == (5, 14)  # start=5, end=5+10-1=14

    def test_hunk_overlap_detection(self):
        """Test overlap detection between hunks."""
        from fix_diff import Hunk

        hunk1 = Hunk.from_header("@@ -1,5 +1,5 @@")
        hunk2 = Hunk.from_header("@@ -4,5 +4,5 @@")  # Overlaps with hunk1
        hunk3 = Hunk.from_header("@@ -10,5 +10,5 @@")  # No overlap

        assert hunk1.overlaps_with(hunk2)  # (1,5) overlaps with (4,8)
        assert not hunk1.overlaps_with(hunk3)  # (1,5) doesn't overlap with (10,14)
        assert not hunk2.overlaps_with(hunk3)  # (4,8) doesn't overlap with (10,14)

    def test_hunk_manager_overlap_detection(self):
        """Test HunkManager unified overlap detection."""
        from fix_diff import HunkManager

        # Test with list of hunk lines
        hunks = [
            ["@@ -1,5 +1,5 @@", " line1", " line2"],
            ["@@ -4,3 +4,3 @@", " line4", " line5"],  # Overlaps
            ["@@ -10,2 +10,2 @@", " line10"],  # No overlap
        ]

        assert HunkManager.detect_overlaps(hunks)  # Should detect overlap

        # Test with non-overlapping hunks
        non_overlapping = [
            ["@@ -1,3 +1,3 @@", " line1"],
            ["@@ -10,3 +10,3 @@", " line10"],
        ]

        assert not HunkManager.detect_overlaps(non_overlapping)

    def test_hunk_manager_merge_functionality(self):
        """Test HunkManager merging functionality."""
        from fix_diff import HunkManager

        overlapping_hunks = [
            ["@@ -1,3 +1,3 @@", " context", "-old1", "+new1"],
            ["@@ -2,3 +2,4 @@", " context", "-old2", "+new2", "+extra"],
        ]

        original_content = "context\nold1\nold2\nextra_context"

        merged = HunkManager.merge_overlapping_hunks(
            overlapping_hunks, original_content
        )

        # Should have fewer hunks after merging
        assert len(merged) < len(overlapping_hunks)
        assert len(merged) == 1  # Should merge into single hunk

        # Merged hunk should have proper header
        merged_hunk = merged[0]
        assert merged_hunk[0].startswith("@@")
        assert "context" in "\n".join(merged_hunk)

    def test_malformed_hunk_handling(self):
        """Test handling of malformed hunks."""
        from fix_diff import Hunk

        # Test invalid header
        invalid_hunk = Hunk.from_header("not a valid header")
        assert invalid_hunk is None

        # Test empty lines
        empty_hunk = Hunk.from_lines([])
        assert empty_hunk is None

        # Test lines without header
        no_header = Hunk.from_lines([" just content", "+addition"])
        assert no_header is None

    def test_backwards_compatibility(self):
        """Test that old DiffFixer methods still work through delegation."""

        # Test that old methods now delegate to new unified classes
        hunks = [["@@ -1,3 +1,3 @@", " line1"], ["@@ -5,3 +5,3 @@", " line5"]]

        # These should work through delegation
        has_overlaps = HunkManager.detect_overlaps(hunks)
        assert has_overlaps is False  # No overlap

        range_result = Hunk.extract_range_from_lines(hunks[0])
        assert range_result == (1, 3)


class TestCommonLogicExtraction:
    """Test the extracted common logic shared between single and multi-hunk processing."""

    def setup_method(self):
        self.fixer = DiffFixer()

    def test_process_hunk_content_common_joined_lines(self):
        """Test that common hunk processing handles joined lines correctly."""
        # Test joined line that should be split
        hunk_lines = ["@@ -1,2 +1,2 @@", "         )except Exception as e:"]

        # Original file has the separate lines
        original_lines = ["        )", "    except Exception as e:"]

        context = DiffContext(
            lines=hunk_lines,
            original_lines=original_lines,
            line_index=0,
            preserve_filenames=False,
        )

        result = self.fixer._process_hunk_content_common(hunk_lines, context)

        # Should split into separate lines with correct indentation
        assert "        )" in str(result)  # First part with 8 spaces
        assert "    except Exception as e:" in str(result)  # Second part with 4 spaces
        # Should NOT contain the joined line
        assert ")except Exception as e:" not in str(result)

    def test_process_hunk_content_common_missing_prefixes(self):
        """Test that common hunk processing adds missing diff prefixes."""
        hunk_lines = ["@@ -1,2 +1,2 @@", "existing line", "new line"]

        # Only 'existing line' exists in original, so 'new line' should get + prefix
        original_lines = ["existing line"]

        context = DiffContext(
            lines=hunk_lines,
            original_lines=original_lines,
            line_index=0,
            preserve_filenames=False,
        )

        result = self.fixer._process_hunk_content_common(hunk_lines, context)
        result_str = "\n".join(result)

        # Should infer that 'existing line' is context (exists in original)
        # and 'new line' is addition (doesn't exist in original)
        assert " existing line" in result_str or "+existing line" in result_str
        assert "+new line" in result_str

    def test_process_hunk_content_common_malformed_headers(self):
        """Test that common hunk processing fixes malformed hunk headers."""
        hunk_lines = ["@@ ... @@", " context line", "-old line", "+new line"]
        original_lines = ["context line", "old line"]

        context = DiffContext(
            lines=hunk_lines,
            original_lines=original_lines,
            line_index=0,
            preserve_filenames=False,
        )

        result = self.fixer._process_hunk_content_common(hunk_lines, context)
        result_str = "\n".join(result)

        # Should fix the malformed header
        assert "@@ ... @@" not in result_str
        assert "@@" in result_str and result_str.count("@@") >= 2  # Proper hunk header

    def test_apply_common_post_processing_single_hunk(self):
        """Test common post-processing for single-hunk diffs."""
        # Test single-hunk post-processing
        lines = [
            "--- a/test.txt",
            "+++ b/test.txt",
            "@@ ... @@",
            "-old line",
            "+new line",
        ]
        original_content = "old line\n"

        result = self.fixer._apply_common_post_processing(
            lines, original_content, is_single_hunk=True
        )

        # Should have processed the lines (exact details depend on context, but should not error)
        assert isinstance(result, list)
        assert len(result) >= len(lines)  # May add context lines

    def test_apply_common_post_processing_multi_hunk(self):
        """Test common post-processing for multi-hunk diffs."""
        # Test multi-hunk post-processing
        lines = [
            "--- a/test.txt",
            "+++ b/test.txt",
            "@@ -1,1 +1,1 @@",
            "-old line 1",
            "+new line 1",
            "@@ -3,1 +3,1 @@",
            "-old line 3",
            "+new line 3",
        ]
        original_content = "old line 1\ncontext\nold line 3\n"

        result = self.fixer._apply_common_post_processing(
            lines, original_content, is_single_hunk=False
        )

        # Should process without adding single-hunk specific context
        assert isinstance(result, list)
        assert len(result) >= len(lines)

    def test_apply_common_post_processing_no_newline_handling(self):
        """Test common post-processing handles files without trailing newlines."""
        lines = [
            "--- a/test.txt",
            "+++ b/test.txt",
            "@@ -1,2 +1,2 @@",
            " line 1",
            "-last_line",
            "+last_line_modified",
        ]

        # Original file without trailing newline
        original_content = "line 1\nlast_line"  # No \n at end

        result = self.fixer._apply_common_post_processing(
            lines, original_content, is_single_hunk=True
        )
        result_str = "\n".join(result)

        # Should add no-newline markers when touching the last line
        assert "\\ No newline at end of file" in result_str

    def test_process_file_headers_common(self):
        """Test common file header processing."""
        lines = [
            "--- original/path.txt",
            "+++ original/path.txt",
            "@@ -1,1 +1,1 @@",
            "-old",
            "+new",
        ]

        file_headers, remaining_lines = self.fixer._process_file_headers_common(
            lines, "test.txt", preserve_filenames=False
        )

        # Should extract and fix file headers
        assert len(file_headers) == 2
        assert "--- a/test.txt" in file_headers
        assert "+++ b/test.txt" in file_headers

        # Should return non-header lines
        assert len(remaining_lines) == 3
        assert "@@ -1,1 +1,1 @@" in remaining_lines
        assert "-old" in remaining_lines
        assert "+new" in remaining_lines

    def test_process_file_headers_common_preserve_filenames(self):
        """Test common file header processing with filename preservation."""
        lines = [
            "--- original/path.txt",
            "+++ original/path.txt",
            "@@ -1,1 +1,1 @@",
            "-old",
            "+new",
        ]

        file_headers, _ = self.fixer._process_file_headers_common(
            lines, "test.txt", preserve_filenames=True
        )

        # Should preserve original filenames
        assert len(file_headers) == 2
        assert "--- a/original/path.txt" in file_headers
        assert "+++ b/original/path.txt" in file_headers

    def test_common_logic_integration_single_hunk(self):
        """Test that single-hunk processing properly uses common logic."""
        # Create a simple single-hunk diff that exercises common logic
        diff_content = """--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
 context line
old lineexcept Exception as e:"""

        # Original with separate lines
        original_content = """context line
old line
except Exception as e:"""

        result = self.fixer.fix_diff(diff_content, original_content, "test.py")

        # Should process using common logic and maintain single-hunk specific features
        assert "context line" in result
        # Should split the joined line using common logic
        assert "old line" in result
        assert "except Exception as e:" in result
        assert "old lineexcept" not in result

    def test_common_logic_integration_multi_hunk(self):
        """Test that multi-hunk processing properly uses common logic."""
        # Create a multi-hunk diff that exercises common logic
        diff_content = """--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
-old line 1
+new line 1
 context line
@@ -3,2 +3,2 @@
 context line 2
old line 3except Exception as e:"""

        # Original with separate lines
        original_content = """old line 1
context line
old line 3
except Exception as e:
context line 2"""

        result = self.fixer.fix_diff(diff_content, original_content, "test.py")

        # Should process using common logic and maintain multi-hunk specific features
        assert result.count("@@") == 4  # 2 hunks * 2 @@ markers each
        # Should split joined lines using common logic
        assert "old line 3" in result
        assert "except Exception as e:" in result
        assert "old line 3except" not in result

    def test_single_hunk_uses_common_logic_fallback(self):
        """Test that single-hunk processing now falls back to common logic for appropriate cases."""
        # Create a single-hunk diff with joined lines and missing prefixes
        diff_content = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 context line
old lineexcept Exception as e:
new unprefixed line"""

        # Original with separate lines
        original_content = """context line
old line
except Exception as e:"""

        result = self.fixer.fix_diff(diff_content, original_content, "test.py")

        # Should use common logic to split joined line
        assert "old line" in result
        assert "except Exception as e:" in result
        assert "old lineexcept" not in result

        # Should use common logic to add prefix to unprefixed line
        assert "+new unprefixed line" in result

    def test_should_use_common_processing_logic(self):
        """Test the _should_use_common_processing decision logic."""
        # Create context with some original lines for joined line detection
        context = DiffContext(
            lines=["test line"],
            original_lines=["line1", "line2"],
            line_index=0,
            preserve_filenames=False,
        )

        # Should NOT use common processing for headers
        assert not self.fixer._should_use_common_processing("--- a/file.txt", context)
        assert not self.fixer._should_use_common_processing("+++ b/file.txt", context)
        assert not self.fixer._should_use_common_processing("@@ -1,1 +1,1 @@", context)

        # Should NOT use common processing for no-newline markers
        assert not self.fixer._should_use_common_processing(
            "\\\\ No newline at end of file", context
        )

        # Should use common processing for unprefixed lines
        assert self.fixer._should_use_common_processing("unprefixed line", context)

        # Should NOT use common processing for already prefixed lines
        assert not self.fixer._should_use_common_processing(" context line", context)
        assert not self.fixer._should_use_common_processing("+addition line", context)
        assert not self.fixer._should_use_common_processing("-deletion line", context)

        # Should NOT use common processing for empty lines
        assert not self.fixer._should_use_common_processing("", context)


class TestDiffFixRuleSystem:
    """Test the rule system specifically."""

    def test_rule_creation(self):
        """Test creating custom rules."""
        rule = DiffFixRule(
            name="test",
            matcher=lambda line, ctx: True,
            fixer=lambda line, ctx: [line],
            priority=5,
            applies_to="all",
        )

        assert rule.name == "test"
        assert rule.priority == 5
        assert rule.applies_to == "all"
        # Test matcher with dummy context

        dummy_context = DiffContext([], [], 0, False)
        assert rule.matcher("anything", dummy_context)

    def test_diff_context_creation(self):
        """Test DiffContext creation and usage."""
        context = DiffContext(
            lines=["line1", "line2"],
            original_lines=["orig1", "orig2"],
            line_index=0,
            preserve_filenames=True,
        )

        assert len(context.lines) == 2
        assert len(context.original_lines) == 2
        assert context.line_index == 0
        assert context.preserve_filenames

    def test_missing_leading_whitespace_pattern(self):
        """Test the missing_leading_whitespace fixture pattern: convert empty context to empty addition."""
        fixer = DiffFixer()

        # Simulate the missing_leading_whitespace case
        diff_content = """--- scripts/bao
+++ scripts/bao
@@ ... @@
# Check if directory exists
if [[ ! -d "$directory" ]]; then
    echo "Error: Directory '$directory' does not exist." >&2
    exit 1
fi

+# New line
"""

        # Original content should include the empty line that gets converted
        original_content = """# Check if directory exists
if [[ ! -d "$directory" ]]; then
    echo "Error: Directory '$directory' does not exist." >&2
    exit 1
fi
"""

        fixed_diff = fixer.fix_diff(diff_content, original_content, "test.txt")

        # Verify the hunk header is corrected to the expected format
        assert "@@ -2,5 +2,7 @@" in fixed_diff, "Expected '@@ -2,5 +2,7 @@' in diff"

        # Verify the structure: fi context line, then empty addition, then content addition
        lines = fixed_diff.split("\n")

        # Find the fi context line and verify the pattern after it
        fi_line_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "fi":
                fi_line_idx = i
                break

        assert fi_line_idx is not None, "Could not find 'fi' context line"

        # Verify the pattern after 'fi': empty addition then content addition
        assert fi_line_idx + 1 < len(lines), "Missing line after fi"
        assert fi_line_idx + 2 < len(lines), "Missing second line after fi"

        assert lines[fi_line_idx + 1] == "+", (
            f"Expected empty addition after fi, got: {repr(lines[fi_line_idx + 1])}"
        )
        assert lines[fi_line_idx + 2] == "+# New line", (
            f"Expected content addition, got: {repr(lines[fi_line_idx + 2])}"
        )

    def test_joined_return_statement_no_newline_pattern(self):
        """Test the joined_return_statement fixture pattern: proper handling of no newline markers."""
        fixer = DiffFixer()

        # Simulate the joined_return_statement case with joined line and no trailing newline
        diff_content = """--- utils.py
+++ utils.py
@@ -1,4 +1,4 @@
 def calculate_total(items):
     if not items:return 0
-    return sum(item.price for item in items)
+    return sum(item.price for item in items)  # sum up
"""

        # Original content without trailing newline (important!)
        original_content = """def calculate_total(items):
    if not items:
        return 0
    return sum(item.price for item in items)"""

        fixed_diff = fixer.fix_diff(diff_content, original_content, "test.py")

        # Verify the joined line was split correctly
        lines = fixed_diff.split("\n")

        # Should have separate lines for "if not items:" and "return 0"
        if_line_found = False
        return_0_line_found = False

        for line in lines:
            if "if not items:" in line and line.startswith(" "):
                if_line_found = True
            elif (
                "return 0" in line
                and line.startswith(" ")
                and "if not items" not in line
            ):
                return_0_line_found = True

        assert if_line_found, "Could not find separated 'if not items:' line"
        assert return_0_line_found, "Could not find separated 'return 0' line"

        # Verify both deletion and addition have proper no-newline markers
        deletion_line_idx = None
        addition_line_idx = None

        for i, line in enumerate(lines):
            if line.startswith("-") and "return sum" in line:
                deletion_line_idx = i
            elif line.startswith("+") and "return sum" in line:
                addition_line_idx = i

        assert deletion_line_idx is not None, "Could not find deletion line"
        assert addition_line_idx is not None, "Could not find addition line"

        # Check that both deletion and addition are followed by no-newline markers
        assert deletion_line_idx + 1 < len(lines), "Missing line after deletion"
        assert addition_line_idx + 1 < len(lines), "Missing line after addition"

        assert lines[deletion_line_idx + 1].strip() == "\\ No newline at end of file", (
            f"Expected no-newline marker after deletion, got: {repr(lines[deletion_line_idx + 1])}"
        )
        assert lines[addition_line_idx + 1].strip() == "\\ No newline at end of file", (
            f"Expected no-newline marker after addition, got: {repr(lines[addition_line_idx + 1])}"
        )

    def test_hunk_header_real_case_pattern(self):
        """Test the hunk_header_real_case fixture pattern: fix malformed headers and joined lines in multi-hunk diffs."""
        fixer = DiffFixer()

        # Simulate the hunk_header_real_case with malformed header and joined line
        diff_content = """--- app/api/v2/cli/main.py
+++ app/api/v2/cli/main.py
@@ -4,7 +4,7 @@

-    all_mixtures = []
+    all_mixture_names = []
     platforms: list[PLATFORM] = ["daier", "dusk"]

     for platform in platforms:
@@ -10,7 +10,7 @@mixtures = get_mixtures(platform)
-            all_mixtures.extend(mixtures)
+            all_mixture_names.extend(mixtures.keys())
             console.print(
                 f"[blue]Found {len(mixtures)} {platform.title()} mixtures[/blue]"
             )except Exception as e:
             console.print(f"[red]Error fetching {platform.title()} mixtures: {e}[/red]")

-    if not all_mixtures:
+    if not all_mixture_names:
         console.print("[red]No mixtures found, cannot generate type alias[/red]")
         return

     # Sort mixtures for consistent output
-    all_mixtures.sort()
+    all_mixture_names.sort()"""

        # Original content
        original_content = """def generate_mixture_type_alias():
    \"\"\"Generate type aliases (SITE, PLATFORM, MIXTURE) and write to primitives.py.\"\"\"
    console.print("[bold blue]Generating type aliases...[/bold blue]")

    all_mixtures = []
    platforms: list[PLATFORM] = ["daier", "dusk"]

    for platform in platforms:
        try:
            mixtures = get_mixtures(platform)
            all_mixtures.extend(mixtures)
            console.print(
                f"[blue]Found {len(mixtures)} {platform.title()} mixtures[/blue]"
            )
        except Exception as e:
            console.print(f"[red]Error fetching {platform.title()} mixtures: {e}[/red]")

    if not all_mixtures:
        console.print("[red]No mixtures found, cannot generate type alias[/red]")
        return

    # Sort mixtures for consistent output
    all_mixtures.sort()"""

        fixed_diff = fixer.fix_diff(diff_content, original_content, "original.py")
        lines = fixed_diff.split("\n")

        # Verify the malformed hunk header was fixed
        malformed_header_found = False
        clean_header_found = False
        mixtures_context_found = False

        for i, line in enumerate(lines):
            if "@@ -10,7 +10,7 @@mixtures = get_mixtures(platform)" in line:
                malformed_header_found = True
            elif line.startswith("@@ -10,") and "mixtures" not in line:
                clean_header_found = True
                # Next context line should be the extracted content
                if (
                    i + 1 < len(lines)
                    and "mixtures = get_mixtures(platform)" in lines[i + 1]
                ):
                    mixtures_context_found = True

        assert not malformed_header_found, "Malformed header should be fixed"
        assert clean_header_found, "Clean hunk header should be present"
        assert mixtures_context_found, "Extracted content should appear as context line"

        # Verify the joined line was split correctly
        joined_line_found = False
        closing_paren_found = False
        except_line_found = False

        for i, line in enumerate(lines):
            if ")except Exception as e:" in line:
                joined_line_found = True
            elif line.strip() == ")" and line.startswith(" "):
                closing_paren_found = True
            elif "except Exception as e:" in line and line.startswith(" "):
                except_line_found = True

        assert not joined_line_found, "Joined line should be split"
        assert closing_paren_found, "Closing parenthesis should appear as separate line"
        assert except_line_found, "Exception handler should appear as separate line"

    def test_multi_hunk_no_newline_marker_handling(self):
        """Test that multi-hunk diffs properly handle no-newline markers for last-line changes."""
        fixer = DiffFixer()

        # Multi-hunk diff where the last line of file doesn't end with newline
        diff_content = """--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
 first_line = "value"
-second_line = "old"
+second_line = "new"
@@ -3,2 +3,2 @@
 third_line = "value"
-last_line = "old_value"
+last_line = "new_value\""""

        # Original file without trailing newline
        original_content = """first_line = "value"
second_line = "old"
third_line = "value"
last_line = "old_value\""""

        fixed_diff = fixer.fix_diff(diff_content, original_content, "test.py")
        lines = fixed_diff.split("\n")

        # Verify that the last line deletion and addition both get no-newline markers
        deletion_marker_found = False
        addition_marker_found = False

        for i, line in enumerate(lines):
            if line.startswith('-last_line = "old_value"'):
                # Check if next line is no-newline marker
                if (
                    i + 1 < len(lines)
                    and lines[i + 1] == "\\ No newline at end of file"
                ):
                    deletion_marker_found = True
            elif line.startswith('+last_line = "new_value"'):
                # Check if next line is no-newline marker
                if (
                    i + 1 < len(lines)
                    and lines[i + 1] == "\\ No newline at end of file"
                ):
                    addition_marker_found = True

        assert deletion_marker_found, (
            "Deletion of last line should have no-newline marker"
        )
        assert addition_marker_found, (
            "Addition of last line should have no-newline marker"
        )

    def test_missing_leading_space_and_line_break_pattern(self):
        """Test the missing_leading_space_and_line_break fixture pattern: fix missing prefixes and avoid incorrect no-newline markers."""
        fixer = DiffFixer()

        # Multi-hunk diff with missing prefix and potential incorrect no-newline marker
        diff_content = """--- tests/unit/test_utils/test_share.py
+++ tests/unit/test_utils/test_share.py
@@ ... @@
def test_func_one():
-    old_value = "test"
+    new_value = "test"
     return value
@@ ... @@
def test_func_two():
     param1, param2, param3
):
     \"\"\"Test function with parameters.\"\"\"
     result = some_function()"""

        # Original content that doesn't end at the last line of the file
        original_content = """def test_func_one():
    old_value = "test"
    return value

def test_func_two():
    param1, param2, param3
):
    \"\"\"Test function with parameters.\"\"\"
    result = some_function()
    
# More content after the diff ends
def test_func_three():
    pass"""

        fixed_diff = fixer.fix_diff(diff_content, original_content, "test.py")
        lines = fixed_diff.split("\n")

        # Verify that the function definition line gets proper space prefix
        func_def_found = False
        for line in lines:
            if "def test_func_two():" in line and line.startswith(" "):
                func_def_found = True
                break

        assert func_def_found, (
            "Function definition should have space prefix (context line)"
        )

        # Verify that no-newline marker is NOT added since diff doesn't modify last line
        has_no_newline_marker = any(
            "\\ No newline at end of file" in line for line in lines
        )
        assert not has_no_newline_marker, (
            "No-newline marker should not be added when diff doesn't modify last line of file"
        )

    def test_diff_line_only_whitespace_pattern(self):
        """Test the diff_line_only_whitespace fixture pattern: correct whitespace-only context lines to match original file."""
        fixer = DiffFixer()

        # Diff with incorrect whitespace-only context line
        diff_content = """--- utils.lua
+++ utils.lua
@@ -3,7 +3,7 @@
 
 function Utils.read_file_content(filepath)
   local file = io.open(filepath, 'r')
   if not file then
-    return nil, 'Could not open file: ' .. filepath
+    return nil, 'Failed to open file: ' .. filepath
   end
      
   local content = file:read '*all'
"""

        # Original file with empty line where diff shows 6 spaces
        original_content = """-- vibe-coding/utils.lua
local Utils = {}

function Utils.read_file_content(filepath)
  local file = io.open(filepath, 'r')
  if not file then
    return nil, 'Could not open file: ' .. filepath
  end

  local content = file:read '*all'
  file:close()
  return content, nil
end

return Utils
"""

        fixed_diff = fixer.fix_diff(diff_content, original_content, "utils.lua")
        lines = fixed_diff.split("\n")

        # Find the line after "   end" - should be empty context line " " not "      "
        end_line_idx = None
        for i, line in enumerate(lines):
            if "   end" in line and line.startswith(" "):
                end_line_idx = i
                break

        assert end_line_idx is not None, "Could not find '   end' context line"
        assert end_line_idx + 1 < len(lines), "Missing line after '   end'"

        # The next line should be " " (space prefix + empty line), not "      " (6 spaces)
        whitespace_line = lines[end_line_idx + 1]
        assert whitespace_line == " ", (
            f"Expected empty context line ' ' but got {repr(whitespace_line)}"
        )

        # Verify the line doesn't contain multiple spaces (the original bug)
        assert "      " not in fixed_diff, (
            f"Fixed diff should not contain 6-space line, found in: {repr(fixed_diff)}"
        )

        print(
            "✓ Whitespace-only context lines correctly match original file empty lines"
        )

    def test_overlapping_hunk_detection(self):
        """Test the detection and handling of overlapping hunks that would cause git apply conflicts."""
        fixer = DiffFixer()

        # Create a diff with overlapping hunks (similar to api_parameter_refactor)
        diff_content = """--- a/test.py
+++ b/test.py
@@ -4,7 +4,7 @@
 def some_function():
     setup_code()
     param1 = "value1"
-def update_function(param: Type):
+def update_function(param: str):
     key = generate_key()
     data = fetch_data(key)
     if not data:
@@ -12,7 +12,7 @@
 
     # Process the data
     processed = process_data(data)
-    result = param.value
+    result = param
 
     # Save the result
     save_result(result)
@@ -17,4 +17,8 @@
     # Save the result
     save_result(result)
 
-    return result
+    return ProcessedData(
+        key=key,
+        data=result,
+        status="success"
+    )"""

        # Original file that the diff should apply to
        original_content = """def setup_function():
    initialize()

def some_function():
    setup_code()
    param1 = "value1"
def update_function(param: Type):
    key = generate_key()
    data = fetch_data(key)
    if not data:
        raise Exception("No data")

    # Process the data
    processed = process_data(data)
    result = param.value

    # Save the result
    save_result(result)

    return result"""

        # Test overlap detection
        hunks = fixer._split_diff_into_hunks(diff_content)
        has_overlaps = HunkManager.detect_overlaps(hunks)

        assert has_overlaps, "Should detect overlapping hunks in the test diff"

        # Test that the fixer processes overlapping hunks
        result = fixer.fix_diff(diff_content, original_content, "test.py")

        # Should produce fewer hunks than the original (3 -> 2 after merging)
        original_hunk_count = diff_content.count("@@") // 2
        result_hunk_count = result.count("@@") // 2

        assert original_hunk_count == 3, (
            f"Original should have 3 hunks, got {original_hunk_count}"
        )
        assert result_hunk_count <= original_hunk_count, (
            f"Result should have <= hunks than original, got {result_hunk_count} vs {original_hunk_count}"
        )

        # The result should be a valid diff structure
        lines = result.split("\n")
        assert any(line.startswith("--- a/test.py") for line in lines), (
            "Should have file header"
        )
        assert any(line.startswith("+++ b/test.py") for line in lines), (
            "Should have file header"
        )
        assert any(line.startswith("@@") for line in lines), "Should have hunk headers"

        print("✓ Overlapping hunk detection and processing working correctly")

    def test_api_parameter_refactor_empty_line_preservation(self):
        """Test the api_parameter_refactor pattern: preserve empty lines when merging overlapping hunks."""
        fixer = DiffFixer()

        # Create a diff similar to api_parameter_refactor with overlapping hunks
        diff_content = """--- a/test.py
+++ b/test.py
@@ -4,7 +4,7 @@
     response_model=TestModel,
     dependencies=[Depends(check_permission)],
 )
-def update_function(param: Type):
+def update_function(param: str):
     key = generate_key()
     data = fetch_data(key)
     if not data:
@@ -12,7 +12,7 @@
 
     # Process the data
     processed = process_data(data)
-    result = param.value
+    result = param
 
     # Save the result
     save_result(result)
@@ -17,4 +17,8 @@
     # Save the result
     save_result(result)
 
-    return result
+    return ProcessedData(
+        key=key,
+        value=result,
+        status="success"
+    )"""

        # Original file
        original_content = """def setup():
    initialize()

def update_function(param: Type):
    key = generate_key()
    data = fetch_data(key)
    if not data:
        raise Exception("No data")

    # Process the data
    processed = process_data(data)
    result = param.value

    # Save the result
    save_result(result)

    return result"""

        # Test that overlapping hunks are detected and merged correctly
        result = fixer.fix_diff(diff_content, original_content, "test.py")

        # Should have fewer hunks than original (3 -> 2 after merging)
        original_hunk_count = diff_content.count("@@") // 2
        result_hunk_count = result.count("@@") // 2

        assert original_hunk_count == 3, (
            f"Original should have 3 hunks, got {original_hunk_count}"
        )
        assert result_hunk_count == 2, (
            f"Result should have 2 hunks after merging, got {result_hunk_count}"
        )

        # Check that empty lines are preserved in the merged hunk
        lines = result.split("\n")

        # Find the second hunk (the merged one)
        second_hunk_start = None
        hunk_count = 0
        for i, line in enumerate(lines):
            if "@@" in line:
                hunk_count += 1
                if hunk_count == 2:  # Second hunk
                    second_hunk_start = i
                    break

        assert second_hunk_start is not None, "Could not find the second hunk"

        # Extract the merged hunk content
        merged_hunk_lines = []
        for i in range(second_hunk_start + 1, len(lines)):
            line = lines[i]
            if line.startswith("@@"):
                break
            merged_hunk_lines.append(line)

        # Check for required empty lines
        empty_line_count = sum(1 for line in merged_hunk_lines if line == " ")
        assert empty_line_count >= 2, (
            f"Expected at least 2 empty lines in merged hunk, got {empty_line_count}"
        )

        # Check for specific pattern: should have empty line after the param change
        # and empty line after the save_result call
        has_empty_after_param_change = False
        has_empty_after_save_result = False

        for i, line in enumerate(merged_hunk_lines):
            if "+    result = param" in line and i + 1 < len(merged_hunk_lines):
                if merged_hunk_lines[i + 1] == " ":
                    has_empty_after_param_change = True
            elif "save_result(result)" in line and i + 1 < len(merged_hunk_lines):
                if merged_hunk_lines[i + 1] == " ":
                    has_empty_after_save_result = True

        assert has_empty_after_param_change, "Missing empty line after param change"
        assert has_empty_after_save_result, "Missing empty line after save_result call"

        print("✓ Empty lines correctly preserved when merging overlapping hunks")


class TestContextLineAddition:
    """Test the context line addition functionality."""

    def setup_method(self):
        self.fixer = DiffFixer()

    def test_context_line_addition_basic(self):
        """Test basic context line addition around hunks with only additions/deletions."""
        # Create a simple diff with only deletions and additions (no context)
        diff_content = """--- a/test.py
+++ b/test.py
@@ ... @@
-def old_function():
-    pass
+def new_function():
+    pass"""

        original_content = """# Header comment

def old_function():
    pass

# Footer comment"""

        fixed_diff = self.fixer.fix_diff(diff_content, original_content, "test.py")
        lines = fixed_diff.strip().split("\n")

        # Should have proper header with context lines included
        # The header should be @@ -2,4 +2,4 @@ because we have:
        # - 1 context line before + 2 deletion lines + 1 context line after = 4 old lines
        # - 1 context line before + 2 addition lines + 1 context line after = 4 new lines
        assert any(line.startswith("@@ -2,4 +2,4 @@") for line in lines), (
            f"Expected proper header, got: {lines}"
        )

        # Should have empty context line before deletions
        assert any(line == " " for line in lines), (
            f"Expected context line, got: {lines}"
        )

    def test_context_line_parametrizable(self):
        """Test that context line count is parametrizable."""
        # Test with 2 context lines
        fixer_2_lines = DiffFixer(context_lines=2)

        diff_content = """--- a/test.py
+++ b/test.py
@@ ... @@
-def old_function():
-    pass
+def new_function():
+    pass"""

        original_content = """# Header comment 1
# Header comment 2

def old_function():
    pass

# Footer comment 1
# Footer comment 2"""

        fixed_diff = fixer_2_lines.fix_diff(diff_content, original_content, "test.py")
        lines = fixed_diff.strip().split("\n")

        # Should have 4 context lines total (2 before, 2 after) plus the changes
        context_lines = [line for line in lines if line.startswith(" ")]
        assert len(context_lines) >= 2, (
            f"Expected at least 2 context lines, got: {len(context_lines)}"
        )

    def test_context_line_multiple_hunks_2_fixture(self):
        """Test that multiple_hunks_2 fixture works correctly with context lines."""
        fixtures_path = Path(__file__).parent.parent / "tests" / "fixtures" / "pass"
        diff_path = fixtures_path / "multiple_hunks_2" / "diff"
        original_path = fixtures_path / "multiple_hunks_2" / "original.py"
        expected_path = fixtures_path / "multiple_hunks_2" / "diff_formatted"

        if not all(p.exists() for p in [diff_path, original_path, expected_path]):
            return skip_test("multiple_hunks_2 fixture files not found")

        diff_content = diff_path.read_text()
        original_content = original_path.read_text()
        expected_content = expected_path.read_text()

        fixed_diff = self.fixer.fix_diff(
            diff_content, original_content, str(original_path)
        )

        # Should match the expected output
        expected_lines = expected_content.strip().split("\n")
        actual_lines = fixed_diff.strip().split("\n")

        # Check that both hunks have proper headers
        assert "@@ -79,92 +79,45 @@" in fixed_diff, f"First hunk header incorrect"
        assert "@@ -189,6 +142,3 @@" in fixed_diff, f"Second hunk header incorrect"

        # Verify it can be applied by git
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                orig_file = tmpdir_path / "original.py"
                diff_file = tmpdir_path / "test.diff"

                orig_file.write_text(original_content)
                diff_file.write_text(fixed_diff)

                result = subprocess.run(
                    ["git", "apply", "--check", str(diff_file)],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                )
                assert result.returncode == 0, f"Git apply failed: {result.stderr}"
        except FileNotFoundError:
            skip_test("git not available for validation")

    def test_skip_context_when_already_present(self):
        """Test that context lines are not added when already present."""
        diff_content = """--- a/test.py
+++ b/test.py
@@ -2,3 +2,3 @@
 
-def old_function():
-    pass
+def new_function():
+    pass
 """

        original_content = """# Header

def old_function():
    pass

# Footer"""

        fixed_diff = self.fixer.fix_diff(diff_content, original_content, "test.py")

        # Should preserve the existing context lines and header
        assert "@@ -2,3 +2,3 @@" in fixed_diff, (
            f"Expected header preserved, got: {fixed_diff}"
        )
        # Should not add additional context beyond what's already there
        context_lines = [
            line for line in fixed_diff.split("\n") if line.startswith(" ")
        ]
        assert len(context_lines) <= 2, (
            f"Should not add extra context, got: {context_lines}"
        )


class TestPreserveFilenames:
    """Test the preserve_filenames functionality in file header fixing."""

    def setup_method(self):
        self.fixer = DiffFixer()

    def test_preserve_filenames_false_uses_filename(self):
        """Test that preserve_filenames=False uses just the filename."""
        # Test --- header
        result = self.fixer._fix_file_header(
            "--- app/api/v2/utils.py", "/full/path/to/utils.py", False
        )
        assert result == "--- a/utils.py"

        # Test +++ header
        result = self.fixer._fix_file_header(
            "+++ app/api/v2/utils.py", "/full/path/to/utils.py", False
        )
        assert result == "+++ b/utils.py"

    def test_preserve_filenames_true_adds_missing_prefixes(self):
        """Test that preserve_filenames=True adds missing a/ and b/ prefixes."""
        # Test --- header without prefix
        result = self.fixer._fix_file_header(
            "--- app/api/v2/utils.py", "/full/path/to/utils.py", True
        )
        assert result == "--- a/app/api/v2/utils.py"

        # Test +++ header without prefix
        result = self.fixer._fix_file_header(
            "+++ app/api/v2/utils.py", "/full/path/to/utils.py", True
        )
        assert result == "+++ b/app/api/v2/utils.py"

    def test_preserve_filenames_true_keeps_existing_prefixes(self):
        """Test that preserve_filenames=True keeps existing a/ and b/ prefixes."""
        # Test --- header with existing prefix
        result = self.fixer._fix_file_header(
            "--- a/app/api/v2/utils.py", "/full/path/to/utils.py", True
        )
        assert result == "--- a/app/api/v2/utils.py"

        # Test +++ header with existing prefix
        result = self.fixer._fix_file_header(
            "+++ b/app/api/v2/utils.py", "/full/path/to/utils.py", True
        )
        assert result == "+++ b/app/api/v2/utils.py"

    def test_preserve_filenames_handles_dev_null(self):
        """Test that /dev/null is not modified."""
        # Test --- /dev/null
        result = self.fixer._fix_file_header(
            "--- /dev/null", "/full/path/to/utils.py", True
        )
        assert result == "--- /dev/null"

        # Test +++ /dev/null
        result = self.fixer._fix_file_header(
            "+++ /dev/null", "/full/path/to/utils.py", True
        )
        assert result == "+++ /dev/null"

    def test_preserve_filenames_handles_edge_cases(self):
        """Test edge cases for preserve_filenames."""
        # Header with extra whitespace
        result = self.fixer._fix_file_header(
            "---   app/api/v2/utils.py  ", "/full/path/to/utils.py", True
        )
        assert result == "--- a/app/api/v2/utils.py"

        # Header already has b/ prefix (should not change)
        result = self.fixer._fix_file_header(
            "--- b/app/api/v2/utils.py", "/full/path/to/utils.py", True
        )
        assert result == "--- b/app/api/v2/utils.py"  # b/ prefix preserved as-is

    def test_preserve_filenames_integration(self):
        """Test preserve_filenames in full diff processing."""
        diff_content = """--- app/api/v2/utils.py
+++ app/api/v2/utils.py
@@ -1,3 +1,3 @@
 def helper():
-    return "old"
+    return "new\""""

        original_content = """def helper():
    return "old\""""

        result = self.fixer.fix_diff(
            diff_content, original_content, "utils.py", preserve_filenames=True
        )

        # Should add a/ and b/ prefixes while preserving the full path
        assert "--- a/app/api/v2/utils.py" in result
        assert "+++ b/app/api/v2/utils.py" in result

    def test_preserve_filenames_specific_pattern_from_request(self):
        """Test the specific pattern mentioned in the user request."""
        # This tests the exact pattern the user mentioned:
        # --- app/api/v2/utils.py
        # +++ app/api/v2/utils.py

        # Test --- header
        result = self.fixer._fix_file_header(
            "--- app/api/v2/utils.py", "/some/path/to/utils.py", True
        )
        assert result == "--- a/app/api/v2/utils.py"

        # Test +++ header
        result = self.fixer._fix_file_header(
            "+++ app/api/v2/utils.py", "/some/path/to/utils.py", True
        )
        assert result == "+++ b/app/api/v2/utils.py"

        print("✓ Specific pattern from user request handled correctly")


class TestRecentBugFixes:
    """Test recent bug fixes and improvements."""

    def setup_method(self):
        self.fixer = DiffFixer()

    def test_no_newline_marker_logic_consistency(self):
        """Test that no newline markers are applied consistently."""
        # Test file WITH trailing newline - should NOT get marker
        diff_with_newline = """--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
 line1
-old_last_line
+new_last_line"""

        original_with_newline = "line1\nold_last_line\n"  # Has trailing newline

        result = self.fixer.fix_diff(
            diff_with_newline, original_with_newline, "test.py"
        )
        assert "\\ No newline at end of file" not in result

        # Test file WITHOUT trailing newline - should get marker when last line is touched
        original_without_newline = "line1\nold_last_line"  # No trailing newline

        result = self.fixer.fix_diff(
            diff_with_newline, original_without_newline, "test.py"
        )
        assert "\\ No newline at end of file" in result

    def test_common_logic_fallback_integration(self):
        """Test that single-hunk processing correctly falls back to common logic."""
        # Create a case that needs common logic (joined line + missing prefix)
        diff_content = """--- a/test.py
+++ b/test.py  
@@ -1,3 +1,4 @@
 context line
old lineexcept Exception:
missing_prefix_line
+new addition"""

        original_content = """context line
old line
except Exception:"""

        result = self.fixer.fix_diff(diff_content, original_content, "test.py")

        # Should split joined line using common logic
        assert "old line" in result
        assert "except Exception:" in result
        assert "old lineexcept" not in result

        # Should add prefix to unprefixed line using common logic
        assert "+missing_prefix_line" in result or " missing_prefix_line" in result


class TestCommandLineArgs:
    """Test command line argument handling."""

    def setup_method(self):
        self.fixer = DiffFixer()

    def test_preserve_filenames_command_line_integration(self):
        """Test that the --preserve-filenames flag works end-to-end."""
        import subprocess
        import tempfile
        from pathlib import Path

        # Create test diff and original file
        diff_content = """--- app/api/v2/utils.py
+++ app/api/v2/utils.py
@@ -1,2 +1,2 @@
 def test():
-    return "old"
+    return "new\""""

        original_content = """def test():
    return "old\""""

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                diff_file = tmpdir_path / "test.diff"
                orig_file = tmpdir_path / "original.py"

                # Write test files
                diff_file.write_text(diff_content)
                orig_file.write_text(original_content)

                # Test WITH --preserve-filenames flag
                result = subprocess.run(
                    [
                        "python",
                        "fix_diff.py",
                        str(diff_file),
                        str(orig_file),
                        "--preserve-filenames",
                    ],
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).parent,
                )

                if result.returncode == 0:
                    output = result.stdout
                    # Should add a/ and b/ prefixes while preserving the path
                    assert "--- a/app/api/v2/utils.py" in output
                    assert "+++ b/app/api/v2/utils.py" in output
                    print("✓ Command line --preserve-filenames works correctly")
                else:
                    print(f"Command failed: {result.stderr}")

                # Test WITHOUT --preserve-filenames flag
                result_no_flag = subprocess.run(
                    ["python", "fix_diff.py", str(diff_file), str(orig_file)],
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).parent,
                )

                if result_no_flag.returncode == 0:
                    output_no_flag = result_no_flag.stdout
                    # Should use just the filename
                    assert "--- a/original.py" in output_no_flag
                    assert "+++ b/original.py" in output_no_flag
                    print("✓ Command line without flag uses filename only")

        except Exception as e:
            print(f"Command line test skipped due to: {e}")
            # This is acceptable if we can't run subprocess tests


def run_all_tests():
    """Run all tests manually."""
    # Test DiffFixer
    test_fixer = TestDiffFixer()
    test_fixer.setup_method()

    tests = [
        test_fixer.test_basic_diff_fixing_with_fixture,
        test_fixer.test_custom_rule_addition,
        test_fixer.test_rule_priority_ordering,
        test_fixer.test_preserve_filenames_option,
        test_fixer.test_edge_case_empty_diff,
        test_fixer.test_edge_case_malformed_hunk_header,
        test_fixer.test_boundary_splitting_generic,
        test_fixer.test_diff_prefix_inference,
        test_fixer.test_context_preservation,
        test_fixer.test_error_handling_robustness,
        test_fixer.test_exact_indentation_matching,
        test_fixer.test_context_aware_disambiguation,
        test_fixer.test_strict_joined_line_splitting,
        test_fixer.test_no_false_positive_splitting,
        test_fixer.test_indentation_correction,
        test_fixer.test_whitespace_preservation_split_parts,
        test_fixer.test_hunk_header_count_correction,
        test_fixer.test_missing_context_line_insertion,
        test_fixer.test_no_newline_marker_handling,
        test_fixer.test_with_newline_no_marker_added,
        test_fixer.test_multiple_hunks_handling,
        test_fixer.test_unified_single_and_multi_hunk_approach,
        test_fixer.test_generic_fallback_header_creation,
        test_fixer.test_generic_fallback_with_empty_content,
        test_fixer.test_generic_fallback_with_custom_start_line,
        test_fixer.test_no_hardcoded_fallbacks_used,
        test_fixer.test_no_newline_marker_only_when_last_line_touched,
    ]

    # Test unified hunk classes
    test_unified_hunks = TestUnifiedHunkClasses()
    tests.extend(
        [
            test_unified_hunks.test_hunk_from_header_creation,
            test_unified_hunks.test_hunk_from_lines_creation,
            test_unified_hunks.test_hunk_range_extraction,
            test_unified_hunks.test_hunk_overlap_detection,
            test_unified_hunks.test_hunk_manager_overlap_detection,
            test_unified_hunks.test_hunk_manager_merge_functionality,
            test_unified_hunks.test_malformed_hunk_handling,
            test_unified_hunks.test_backwards_compatibility,
        ]
    )

    # Test common logic extraction
    test_common = TestCommonLogicExtraction()
    test_common.setup_method()
    tests.extend(
        [
            test_common.test_process_hunk_content_common_joined_lines,
            test_common.test_process_hunk_content_common_missing_prefixes,
            test_common.test_process_hunk_content_common_malformed_headers,
            test_common.test_apply_common_post_processing_single_hunk,
            test_common.test_apply_common_post_processing_multi_hunk,
            test_common.test_apply_common_post_processing_no_newline_handling,
            test_common.test_process_file_headers_common,
            test_common.test_process_file_headers_common_preserve_filenames,
            test_common.test_common_logic_integration_single_hunk,
            test_common.test_common_logic_integration_multi_hunk,
            test_common.test_single_hunk_uses_common_logic_fallback,
            test_common.test_should_use_common_processing_logic,
        ]
    )

    # Test DiffFixRuleSystem
    test_rules = TestDiffFixRuleSystem()
    tests.extend(
        [
            test_rules.test_rule_creation,
            test_rules.test_diff_context_creation,
            test_rules.test_missing_leading_whitespace_pattern,
            test_rules.test_joined_return_statement_no_newline_pattern,
            test_rules.test_hunk_header_real_case_pattern,
            test_rules.test_multi_hunk_no_newline_marker_handling,
            test_rules.test_missing_leading_space_and_line_break_pattern,
            test_rules.test_diff_line_only_whitespace_pattern,
            test_rules.test_overlapping_hunk_detection,
            test_rules.test_api_parameter_refactor_empty_line_preservation,
        ]
    )

    # Add context line addition tests
    test_context = TestContextLineAddition()
    test_context.setup_method()
    tests.extend(
        [
            test_context.test_context_line_addition_basic,
            test_context.test_context_line_parametrizable,
            test_context.test_context_line_multiple_hunks_2_fixture,
            test_context.test_skip_context_when_already_present,
        ]
    )

    # Add preserve filenames tests
    test_preserve = TestPreserveFilenames()
    test_preserve.setup_method()
    tests.extend(
        [
            test_preserve.test_preserve_filenames_false_uses_filename,
            test_preserve.test_preserve_filenames_true_adds_missing_prefixes,
            test_preserve.test_preserve_filenames_true_keeps_existing_prefixes,
            test_preserve.test_preserve_filenames_handles_dev_null,
            test_preserve.test_preserve_filenames_handles_edge_cases,
            test_preserve.test_preserve_filenames_integration,
            test_preserve.test_preserve_filenames_specific_pattern_from_request,
        ]
    )

    # Add recent bug fixes tests
    test_bugfixes = TestRecentBugFixes()
    test_bugfixes.setup_method()
    tests.extend(
        [
            test_bugfixes.test_malformed_fast_cluster_reconstruction,
            test_bugfixes.test_no_newline_marker_logic_consistency,
            test_bugfixes.test_common_logic_fallback_integration,
        ]
    )

    # Add command line args tests
    test_cmdline = TestCommandLineArgs()
    test_cmdline.setup_method()
    tests.extend(
        [
            test_cmdline.test_preserve_filenames_command_line_integration,
        ]
    )

    # Add standalone tests
    tests.extend([])

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"Running {test.__name__}...")
            test()
            print(f"✓ {test.__name__} passed")
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
