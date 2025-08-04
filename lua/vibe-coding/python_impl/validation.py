"""
Generic diff validation module - Python implementation
Implements the same generic validation logic as the Lua version with enhanced whitespace preservation.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Issue:
    """Represents a validation issue found in diff content."""

    line: int
    type: str
    message: str
    severity: str = "info"
    original_text: Optional[str] = None
    split_lines: Optional[List[str]] = None
    corrected_line: Optional[str] = None


class Validation:
    """Generic diff validation with minimal pattern matching."""

    @staticmethod
    def try_fix_missing_context_prefix(
        line: str, line_num: int
    ) -> Tuple[Optional[str], Optional[Issue]]:
        """
        Generic approach: if line is not empty and doesn't start with +/-, treat as context.

        Args:
            line: The line to potentially fix
            line_num: The line number for error reporting

        Returns:
            Tuple of (fixed_line, issue) or (None, None) if can't fix
        """
        # Generic approach: if line is not empty and doesn't start with +/-, treat as context
        if line and not re.match(r"^[+-]", line):
            fixed_line = " " + line
            message = f"Added missing space prefix for context line: {line[:50]}"
            if len(line) > 50:
                message += "..."

            issue = Issue(
                line=line_num, type="context_fix", message=message, severity="info"
            )
            return fixed_line, issue

        return None, None

    @staticmethod
    def fix_hunk_content_line(line: str, line_num: int) -> Tuple[str, Optional[Issue]]:
        """
        Generic hunk content line fixing.

        Args:
            line: The line to fix
            line_num: The line number

        Returns:
            Tuple of (fixed_line, issue)
        """
        # Handle empty lines - they're valid as-is
        if not line:
            return line, None

        # Handle valid diff operators
        if line.startswith((" ", "-", "+")):
            return line, None

        # Skip header-like lines that should be ignored (not treated as context)
        if line.startswith(("---", "+++")):
            return line, None

        # Try to fix context lines missing the leading space
        fixed_line, issue = Validation.try_fix_missing_context_prefix(line, line_num)
        if fixed_line:
            return fixed_line, issue

        # If we can't fix it, report as invalid
        return line, Issue(
            line=line_num,
            type="invalid_line",
            message=f"Invalid line in hunk: {line}",
            severity="warning",
        )

    @staticmethod
    def validate_and_fix_diff(diff_content: str) -> Tuple[str, List[Issue]]:
        """
        Enhanced diff validation and fixing with whitespace preservation.

        This implements the same validation pipeline as the Lua version:
        1. fix_file_paths
        2. smart_validate_against_original
        3. fix_context_whitespace
        4. validate_and_fix_diff (basic fixes)
        5. format_diff

        Args:
            diff_content: The raw diff content to validate

        Returns:
            Tuple of (cleaned_diff_content, issues_found)
        """
        # Run the enhanced validation pipeline
        current_content = diff_content
        all_issues = []

        # Step 1: Basic diff validation and fixing
        current_content, step1_issues = Validation._basic_validate_and_fix(
            current_content
        )
        all_issues.extend(step1_issues)

        # Step 2: Smart validation against original file
        current_content, step2_issues = Validation.smart_validate_against_original(
            current_content
        )
        all_issues.extend(step2_issues)

        # Step 3: Context whitespace fixing
        current_content, step3_issues = Validation.fix_context_whitespace(
            current_content
        )
        all_issues.extend(step3_issues)

        return current_content, all_issues

    @staticmethod
    def _basic_validate_and_fix(diff_content: str) -> Tuple[str, List[Issue]]:
        """
        Basic diff validation and fixing (original logic).

        Args:
            diff_content: The raw diff content to validate

        Returns:
            Tuple of (cleaned_diff_content, issues_found)
        """
        lines = diff_content.split("\n")
        issues = []
        fixed_lines = []

        # Track state for validation
        has_header = False
        in_hunk = False

        for i, line in enumerate(lines):
            fixed_line = line
            line_num = i + 1  # 1-based line numbers

            # Check for diff headers
            if line.startswith("---") or line.startswith("+++"):
                has_header = True
                fixed_line, issue = Validation._fix_header_line(line, line_num)
                if issue:
                    issues.append(issue)
            # Check for hunk headers
            elif line.startswith("@@"):
                in_hunk = True
                fixed_line, issue = Validation._fix_hunk_header(line, line_num)
                if issue:
                    issues.append(issue)
            # Check context and change lines
            elif in_hunk:
                fixed_line, issue = Validation.fix_hunk_content_line(line, line_num)
                if issue:
                    issues.append(issue)

            fixed_lines.append(fixed_line)

        # Final validation
        if not has_header:
            issues.append(
                Issue(
                    line=1,
                    type="missing_header",
                    message="Diff missing file headers",
                    severity="error",
                )
            )

        return "\n".join(fixed_lines), issues

    @staticmethod
    def smart_validate_against_original(diff_content: str) -> Tuple[str, List[Issue]]:
        """
        Smart validation that compares diff context with original file content.

        Args:
            diff_content: The diff content to validate

        Returns:
            Tuple of (fixed_diff_content, issues_found)
        """
        lines = diff_content.split("\n")
        issues = []
        fixed_lines = []

        current_file = None
        original_lines = None
        current_hunk_start = None

        for i, line in enumerate(lines):
            line_num = i + 1

            # Track hunk headers to get hint information
            if line.startswith("@@"):
                match = re.search(r"@@ -(\d+)", line)
                current_hunk_start = int(match.group(1)) if match else None

            # Track current file being processed
            if line.startswith("---"):
                file_path_match = re.match(r"^---\s+(.+)$", line)
                if file_path_match:
                    file_path = file_path_match.group(1)
                    if file_path != "/dev/null" and Validation._looks_like_file(
                        file_path
                    ):
                        current_file = Validation._resolve_file_path(file_path)
                        if current_file and os.path.exists(current_file):
                            try:
                                with open(
                                    current_file, "r", encoding="utf-8", errors="ignore"
                                ) as f:
                                    content = f.read()
                                    original_lines = content.splitlines()
                            except:
                                original_lines = None
                        else:
                            original_lines = None

            # Process context lines in hunks
            if line.startswith(" ") and original_lines:
                context_result = Validation._process_context_line(
                    line, line_num, original_lines, issues, True, current_hunk_start
                )
                fixed_lines.extend(context_result)
            # Also check lines that should be context but are missing the space prefix
            elif (
                original_lines
                and line
                and not line.startswith(("+", "@", "-"))
                and not line.startswith("---")
                and not line.startswith("+++")
            ):
                context_result = Validation._process_context_line(
                    line, line_num, original_lines, issues, False, current_hunk_start
                )
                fixed_lines.extend(context_result)
            else:
                fixed_lines.append(line)

        return "\n".join(fixed_lines), issues

    @staticmethod
    def fix_context_whitespace(diff_content: str) -> Tuple[str, List[Issue]]:
        """
        Fixes context lines in diff that have incorrect whitespace by matching original file.

        Args:
            diff_content: The diff content to fix

        Returns:
            Tuple of (fixed_diff_content, issues_found)
        """
        lines = diff_content.split("\n")
        issues = []
        fixed_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check if this line starts a hunk
            if line.startswith("@@"):
                hunk_result = Validation._fix_hunk_whitespace(lines, i, issues)
                fixed_lines.extend(hunk_result["fixed_lines"])
                i = hunk_result["next_line_index"]
            else:
                fixed_lines.append(line)
                i += 1

        return "\n".join(fixed_lines), issues

    @staticmethod
    def _process_context_line(
        line: str,
        line_num: int,
        original_lines: List[str],
        issues: List[Issue],
        has_space_prefix: bool,
        hint_line_num: Optional[int],
    ) -> List[str]:
        """
        Processes a context line for smart validation.

        Args:
            line: The line to process
            line_num: The line number
            original_lines: Original file lines
            issues: Issues list to append to
            has_space_prefix: Whether the line already has a space prefix
            hint_line_num: Hint for expected line number in original file

        Returns:
            List of fixed lines
        """
        result = []
        context_text = line[1:] if has_space_prefix else line
        issue = Validation._validate_context_against_original(
            context_text, original_lines, line_num, hint_line_num
        )

        if not issue:
            result.append(line)
            return result

        issues.append(issue)

        # Handle simple whitespace mismatch - use the corrected line
        if issue.type == "whitespace_mismatch" and issue.corrected_line:
            issue.message += " (auto-corrected whitespace)"
            issue.severity = "info"
            result.append(" " + issue.corrected_line)
            return result

        # Handle joined lines formatting issue
        if issue.type == "formatting_issue" and issue.split_lines:
            issue.message += (
                f" (auto-corrected: split into {len(issue.split_lines)} lines)"
            )
            issue.severity = "info"
            # Add space prefix for diff format
            for split_line in issue.split_lines:
                result.append(" " + split_line)
            return result

        # Final fallback
        if has_space_prefix:
            result.append(line)
        else:
            result.append(" " + line)
            issue.message += " (added space prefix)"
            issue.severity = "warning"

        return result

    @staticmethod
    def _fix_header_line(line: str, line_num: int) -> Tuple[str, Optional[Issue]]:
        """Fix header line issues."""
        if re.match(r"^---\s*$", line):
            return "--- /dev/null", Issue(
                line=line_num, type="header", message="Fixed empty old file header"
            )
        elif re.match(r"^\+\+\+\s*$", line):
            return "+++ /dev/null", Issue(
                line=line_num, type="header", message="Fixed empty new file header"
            )
        return line, None

    @staticmethod
    def _fix_hunk_header(line: str, line_num: int) -> Tuple[str, Optional[Issue]]:
        """Fix hunk header issues."""
        # Try to parse hunk header
        match = re.match(r"^@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@", line)

        if not match:
            # Try simpler format without counts
            simple_match = re.match(r"^@@ -(\d+) \+(\d+) @@", line)
            if simple_match:
                old_start, new_start = simple_match.groups()
                return f"@@ -{old_start},1 +{new_start},1 @@", Issue(
                    line=line_num,
                    type="hunk_header",
                    message="Added missing line counts",
                )
            else:
                # Check for malformed headers
                if line.startswith("@@ "):
                    return "@@ -1,1 +1,1 @@", Issue(
                        line=line_num,
                        type="hunk_header",
                        message="Normalized malformed hunk header (line numbers ignored for search-and-replace)",
                        severity="info",
                    )
                else:
                    return line, Issue(
                        line=line_num,
                        type="hunk_header",
                        message="Invalid hunk header format",
                        severity="error",
                    )

        return line, None

    @staticmethod
    def _looks_like_file(path: str) -> bool:
        """Check if a path looks like a file path."""
        if not path or path in ["/dev/null", "/dev/stdin", "/dev/stdout"]:
            return False
        # Simple heuristic: has an extension or is a reasonable path
        return "." in path or "/" in path

    @staticmethod
    def _resolve_file_path(path: str) -> Optional[str]:
        """Resolve file path by searching filesystem."""
        # Clean path
        path = path.strip()

        # Remove common VCS prefixes
        if path.startswith(("a/", "b/")):
            path = path[2:]

        # If it's already absolute and exists, return it
        if os.path.isabs(path) and os.path.exists(path):
            return path

        # Try relative to current directory
        if os.path.exists(path):
            return os.path.abspath(path)

        # Try searching in common locations
        search_dirs = [
            ".",
            "tests",
            "tests/fixtures",
            "../tests",  # When running from python_impl subdirectory
            "../tests/fixtures",  # When running from python_impl subdirectory
            "lua/vibe-coding/tests",
            "lua/vibe-coding/tests/fixtures",
            "config/nvim/lua/vibe-coding/tests",
            "config/nvim/lua/vibe-coding/tests/fixtures",
        ]

        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    if path.endswith("/"):
                        continue
                    basename = os.path.basename(path)
                    if basename in files:
                        candidate = os.path.join(root, basename)
                        if candidate.endswith(path) or path in candidate:
                            return os.path.abspath(candidate)

        return None

    @staticmethod
    def _validate_context_against_original(
        context_text: str,
        original_lines: List[str],
        line_num: int,
        hint_line_num: Optional[int] = None,
    ) -> Optional[Issue]:
        """
        Validates a context line against the original file content.

        Args:
            context_text: The text from the context line (without leading space)
            original_lines: Array of lines from the original file
            line_num: Line number for error reporting
            hint_line_num: Optional hint for the expected line number in original file

        Returns:
            Issue object if validation fails, None if validation passes
        """
        # Skip empty lines
        if not context_text:
            return None

        # Check if this exact line exists in the original file
        for orig_line in original_lines:
            if orig_line == context_text:
                return None  # Found exact match, all good

        # No exact match found, check for simple whitespace differences
        context_trimmed = context_text.strip()
        if context_trimmed:
            matches = []
            for i, orig_line in enumerate(original_lines):
                orig_trimmed = orig_line.strip()
                if orig_trimmed == context_trimmed:
                    matches.append({"line": orig_line, "line_num": i + 1})

            if matches:
                # If we have multiple matches, prefer the one closest to the hint
                best_match = matches[0]
                if hint_line_num and len(matches) > 1:
                    best_distance = abs(matches[0]["line_num"] - hint_line_num)
                    for match in matches:
                        distance = abs(match["line_num"] - hint_line_num)
                        if distance < best_distance:
                            best_distance = distance
                            best_match = match

                # Found content match with different whitespace
                message = f"Context line has incorrect whitespace: {context_text[:50]}"
                if len(context_text) > 50:
                    message += "..."

                return Issue(
                    line=line_num,
                    type="whitespace_mismatch",
                    message=message,
                    severity="info",
                    original_text=context_text,
                    corrected_line=best_match["line"],
                )

        # Check for joined lines
        prefix_match = Validation._find_prefix_match(
            context_text, original_lines, hint_line_num
        )
        if prefix_match:
            message = f"Joined line detected (prefix match): {context_text[:60]}"
            if len(context_text) > 60:
                message += "..."

            return Issue(
                line=line_num,
                type="formatting_issue",
                message=message,
                severity="warning",
                original_text=context_text,
                split_lines=prefix_match,
            )

        # Generic case: line not found in original
        message = f"Context line not found in original file: {context_text[:50]}"
        if len(context_text) > 50:
            message += "..."

        return Issue(
            line=line_num,
            type="context_mismatch",
            message=message,
            severity="warning",
            original_text=context_text,
        )

    @staticmethod
    def _find_prefix_match(
        context_text: str,
        original_lines: List[str],
        hint_line_num: Optional[int] = None,
    ) -> Optional[List[str]]:
        """
        Finds if context_text is a prefix of lines in the original file and extracts the split.

        Args:
            context_text: The potentially joined line
            original_lines: Array of lines from the original file
            hint_line_num: Optional hint for preferred line number

        Returns:
            Array of split lines that reconstruct the context_text, or None if no match
        """
        if not original_lines or not context_text:
            return None

        # Remove leading/trailing whitespace for comparison
        trimmed_context = context_text.strip()

        # Collect all possible matches first
        all_matches = []

        # Try to find a sequence of original lines that when concatenated would create context_text
        for start_idx in range(len(original_lines)):
            matched_lines = []

            # Try building up the context_text from consecutive original lines
            for end_idx in range(start_idx, min(start_idx + 5, len(original_lines))):
                orig_line = original_lines[end_idx]
                trimmed_orig = orig_line.strip()

                # Include all lines (even empty ones) in the match sequence
                matched_lines.append(orig_line)

                # Only process non-empty lines for matching
                if trimmed_orig:
                    # Try direct concatenation of trimmed non-empty lines
                    joined = ""
                    for line in matched_lines:
                        line_trimmed = line.strip()
                        if line_trimmed:
                            if not joined:
                                joined = line_trimmed
                            else:
                                joined = joined + line_trimmed

                    # Check if this matches our context text
                    if joined == trimmed_context:
                        all_matches.append(
                            {
                                "lines": matched_lines.copy(),
                                "start_line": start_idx + 1,
                                "end_line": end_idx + 1,
                            }
                        )

        if not all_matches:
            return None

        # If we have multiple matches, prefer the one closest to the hint
        if hint_line_num and len(all_matches) > 1:
            best_match = all_matches[0]
            best_distance = abs(all_matches[0]["start_line"] - hint_line_num)

            for match in all_matches:
                distance = abs(match["start_line"] - hint_line_num)
                if distance < best_distance:
                    best_distance = distance
                    best_match = match

            return best_match["lines"]

        # Return the first match if no hint provided
        return all_matches[0]["lines"]

    @staticmethod
    def _fix_hunk_whitespace(
        lines: List[str], hunk_start_index: int, issues: List[Issue]
    ) -> Dict:
        """
        Fixes whitespace for an entire hunk by matching the complete context sequence.

        Args:
            lines: All diff lines
            hunk_start_index: Index where the hunk starts (@@)
            issues: Issues array to append to

        Returns:
            Dict with 'fixed_lines' and 'next_line_index'
        """
        hunk_header = lines[hunk_start_index]
        fixed_lines = [hunk_header]

        # Parse the file path from preceding lines
        current_file = None
        for i in range(hunk_start_index - 1, -1, -1):
            if lines[i].startswith("+++"):
                match = re.match(r"^\+\+\+\s+(.+)$", lines[i])
                if match:
                    current_file = match.group(1)
                    break

        # Read original file
        original_lines = None
        if current_file:
            resolved_path = Validation._resolve_file_path(current_file)
            if resolved_path and os.path.exists(resolved_path):
                try:
                    with open(
                        resolved_path, "r", encoding="utf-8", errors="ignore"
                    ) as f:
                        content = f.read()
                        original_lines = content.splitlines()
                except:
                    original_lines = None

        if not original_lines:
            # Can't fix without original file, copy hunk as-is
            i = hunk_start_index + 1
            while (
                i < len(lines)
                and not lines[i].startswith("@@")
                and not lines[i].startswith("---")
                and not lines[i].startswith("+++")
            ):
                fixed_lines.append(lines[i])
                i += 1
            return {"fixed_lines": fixed_lines, "next_line_index": i}

        # Parse hunk header for line number
        match = re.search(r"@@ -(\d+)", hunk_header)
        hunk_start_line = int(match.group(1)) if match else None

        # Process each line in the hunk
        i = hunk_start_index + 1
        hunk_end_index = i

        # Find end of hunk
        while (
            hunk_end_index < len(lines)
            and not lines[hunk_end_index].startswith("@@")
            and not lines[hunk_end_index].startswith("---")
            and not lines[hunk_end_index].startswith("+++")
        ):
            hunk_end_index += 1

        # Process each line in the hunk
        while i < hunk_end_index:
            line = lines[i]

            if line.startswith(" "):
                # This is a context line - apply intelligent whitespace correction
                context_text = line[1:]  # Remove leading space
                corrected_line = Validation._fix_single_context_line_whitespace(
                    context_text, original_lines, hunk_start_line
                )

                if corrected_line and corrected_line != context_text:
                    fixed_line = " " + corrected_line
                    fixed_lines.append(fixed_line)

                    if fixed_line != line:
                        issues.append(
                            Issue(
                                line=i + 1,
                                severity="info",
                                type="whitespace_fix",
                                message="Fixed context line whitespace to match original file",
                            )
                        )
                else:
                    # No correction needed or possible
                    fixed_lines.append(line)
            else:
                # Not a context line (+ or -), keep as-is
                fixed_lines.append(line)

            i += 1

        return {"fixed_lines": fixed_lines, "next_line_index": hunk_end_index}

    @staticmethod
    def _fix_single_context_line_whitespace(
        context_text: str, original_lines: List[str], hint_start_line: Optional[int]
    ) -> Optional[str]:
        """
        Fixes a single context line's whitespace to match the original file.

        Args:
            context_text: The context line text (without leading space)
            original_lines: The original file lines
            hint_start_line: Hint for where to start searching

        Returns:
            The corrected line text, or None if no correction needed/possible
        """
        if not original_lines:
            return None

        # First, check for exact matches (this handles most cases)
        for orig_line in original_lines:
            if orig_line == context_text:
                return None  # Already correct, no change needed

        # Check for matches where only whitespace differs
        context_trimmed = context_text.strip()
        if not context_trimmed:
            # Handle empty lines - find the closest empty line in original
            best_empty_line = None
            best_distance = float("inf")

            for line_num, orig_line in enumerate(original_lines):
                if not orig_line.strip():
                    distance = (
                        abs(line_num + 1 - hint_start_line) if hint_start_line else 0
                    )
                    if distance < best_distance:
                        best_distance = distance
                        best_empty_line = orig_line

            return best_empty_line or ""  # Use closest empty line or fallback

        # Look for lines with matching content but different whitespace, prioritizing those near hint
        matches = []
        for line_num, orig_line in enumerate(original_lines):
            orig_trimmed = orig_line.strip()
            if orig_trimmed == context_trimmed:
                matches.append({"line": orig_line, "line_num": line_num + 1})

        if matches:
            # If we have a hint, prefer the match closest to the hint
            if hint_start_line and len(matches) > 1:
                matches.sort(key=lambda m: abs(m["line_num"] - hint_start_line))
            return matches[0]["line"]

        # Try fuzzy matching for cases where content might be slightly different
        # but represents the same logical line
        best_match = None
        best_score = 0
        best_distance = float("inf")

        for line_num, orig_line in enumerate(original_lines):
            score = Validation._calculate_line_similarity(context_text, orig_line)
            if score > 0.8:
                distance = abs(line_num + 1 - hint_start_line) if hint_start_line else 0
                # Prefer closer matches when scores are similar
                if score > best_score or (
                    score == best_score and distance < best_distance
                ):
                    best_match = orig_line
                    best_score = score
                    best_distance = distance

        if best_match:
            return best_match

        # No correction possible
        return None

    @staticmethod
    def _calculate_line_similarity(line1: str, line2: str) -> float:
        """
        Calculates similarity between two lines (0.0 to 1.0).

        Args:
            line1: First line
            line2: Second line

        Returns:
            Similarity score
        """
        # Simple similarity based on trimmed content
        trimmed1 = line1.strip()
        trimmed2 = line2.strip()

        if trimmed1 == trimmed2:
            return 1.0

        if not trimmed1 or not trimmed2:
            return 0.0

        # Check if one is a substring of the other
        if trimmed2 in trimmed1 or trimmed1 in trimmed2:
            return 0.9

        # Simple word-based comparison
        words1 = set(trimmed1.split())
        words2 = set(trimmed2.split())

        if not words1 and not words2:
            return 0.0

        common_words = len(words1 & words2)
        total_words = len(words1 | words2)

        if total_words == 0:
            return 0.0

        return common_words / total_words
