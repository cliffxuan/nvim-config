#!/usr/bin/env python3
"""
Fix malformatted diff files to make them compatible with git apply.
Refactored to follow generic, configurable principles instead of hard-coded patterns.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from hunk_locator import find_hunk_location


# Configuration and Constants
class DiffConfig:
    """Central configuration for diff processing."""

    # Patterns
    HUNK_HEADER_PATTERN = re.compile(r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@")
    MALFORMED_HEADER_SIMPLE = "@@ ... @@"

    # Prefixes and markers
    DIFF_PREFIXES = (" ", "+", "-")
    FILE_PREFIXES = ("---", "+++")
    HUNK_PREFIX = "@@"
    NO_NEWLINE_MARKER = "\\ No newline at end of file"

    # Processing constants
    CONTEXT_WINDOW_SIZE = 3
    DEFAULT_FALLBACK_START_LINE = 1  # Generic fallback to file start
    MIN_HUNK_SIZE = 1  # Minimum lines in a hunk section
    DEFAULT_CONTEXT_LINES = 1  # Default number of context lines to add around hunks

    MIN_HUNK_LOCATION_CONFIDENCE = 0.825

    @staticmethod
    def create_fallback_header(
        content_lines: list[str], start_line: int | None = None
    ) -> str:
        """Create a generic fallback hunk header based on actual content."""
        if start_line is None:
            start_line = DiffConfig.DEFAULT_FALLBACK_START_LINE

        # Count actual diff content to make a reasonable header
        context_count = sum(
            1 for line in content_lines if line.startswith(" ") or not line.strip()
        )
        deletion_count = sum(1 for line in content_lines if line.startswith("-"))
        addition_count = sum(1 for line in content_lines if line.startswith("+"))

        # Calculate counts, ensuring minimum of 1
        old_count = max(DiffConfig.MIN_HUNK_SIZE, context_count + deletion_count)
        new_count = max(DiffConfig.MIN_HUNK_SIZE, context_count + addition_count)

        return f"@@ -{start_line},{old_count} +{start_line},{new_count} @@"

    @classmethod
    def extract_clean_header(cls, line: str) -> str:
        """Extract clean hunk header from malformed line."""
        # Try to find a valid hunk header pattern
        match = re.search(r"(@@ -\d+,\d+ \+\d+,\d+ @@)", line)
        if match:
            return match.group(1)

        # Check if this looks like a header with content joined (missing space after @@)
        match = re.search(r"(@@ -\d+,\d+ \+\d+,\d+ @@)(.+)", line)
        if match:
            # Return just the header part, ignore the joined content
            return match.group(1)

        # If no valid pattern found, create a minimal generic header
        # This should rarely be used since most malformed headers contain some valid pattern
        return cls.create_fallback_header([])

    @staticmethod
    def is_deletion_line(line: str) -> bool:
        """Check if a line is a deletion line."""
        return line.startswith("-") and not line.startswith("---")

    @staticmethod
    def is_addition_line(line: str) -> bool:
        """Check if a line is a deletion line."""
        return line.startswith("+") and not line.startswith("+++")

    @staticmethod
    def is_file_header(line: str) -> bool:
        """Check if a line is a file header."""
        return line.startswith(DiffConfig.FILE_PREFIXES)


@dataclass
class DiffContext:
    """Context information for diff processing."""

    lines: list[str]
    original_lines: list[str]
    line_index: int
    preserve_filenames: bool

    # Enhanced context tracking
    processed_indices: set[int] = field(default_factory=set)
    last_original_index: int = -1

    def has_incorrect_line_counts(self, header_line: str) -> bool:
        """Check if the line counts in the header match the actual diff content."""
        # Parse the current header counts
        match = re.match(r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@", header_line)
        if not match:
            return True  # Can't parse, treat as malformed

        _, count_old, _, count_new = map(int, match.groups())

        # Calculate what the counts should be based on the actual content
        context_lines = 0
        deletion_lines = 0
        addition_lines = 0

        # Count lines in the hunk (after the header)
        for i in range(self.line_index + 1, len(self.lines)):
            line = self.lines[i]
            if line.startswith(" "):
                context_lines += 1
            elif line.startswith("-"):
                deletion_lines += 1
            elif line.startswith("+"):
                addition_lines += 1
            elif not line.strip():  # Empty line - count as context
                context_lines += 1
            else:
                # No prefix - treat as context line
                context_lines += 1

        # Calculate expected counts
        expected_old_count = context_lines + deletion_lines
        expected_new_count = context_lines + addition_lines

        # Return True if counts don't match
        return count_old != expected_old_count or count_new != expected_new_count

    def get_potential_splits(self, line: str) -> list[list[str]]:
        """Generate potential ways to split a line using only original file structure as context."""
        splits = []

        # Strip diff prefix if present
        clean_line = line
        if line.startswith((" ", "+", "-")):
            clean_line = line[1:]

        if not clean_line.strip():
            return splits

        # Generic approach: Only attempt splits at natural boundaries that would
        # result in complete lines that exist in the original file.
        # This is more conservative than pattern-based but still generic.

        # Try splitting where we can find exact line matches in original for both parts
        line_length = len(clean_line)

        # Only try splits that would create meaningful, complete lines
        for i in range(1, line_length):
            left_part = clean_line[:i].rstrip()  # Remove trailing whitespace
            right_part = clean_line[i:].lstrip()  # Remove leading whitespace

            # Skip if either part would be empty
            if not left_part.strip() or not right_part.strip():
                continue

            # Find all possible left part matches
            left_matches = []
            for i, orig_line in enumerate(self.original_lines):
                if orig_line == left_part:
                    left_matches.append(i)

            if not left_matches:
                continue

            # Find all possible right part matches with correct indentation
            right_stripped = right_part.strip()
            right_matches = []
            for i, orig_line in enumerate(self.original_lines):
                if orig_line.strip() == right_stripped:
                    right_matches.append((i, orig_line))

            if not right_matches:
                continue

            # Find the best pairing based on proximity
            best_split = None
            best_distance = float("inf")

            for left_idx in left_matches:
                for right_idx, right_line in right_matches:
                    distance = abs(right_idx - left_idx)
                    if distance < best_distance:
                        # Check if there should be an empty line between the parts
                        if right_idx == left_idx + 2:
                            # Parts are separated by one line - check if it's empty
                            if (
                                left_idx + 1 < len(self.original_lines)
                                and not self.original_lines[left_idx + 1].strip()
                            ):
                                # Insert empty line between parts
                                best_split = [left_part, "", right_line]
                            else:
                                best_split = [left_part, right_line]
                        else:
                            best_split = [left_part, right_line]
                        best_distance = distance

            if best_split:
                splits.append(best_split)

        return splits

    def find_similar_match(self, content: str) -> str | None:
        """Find match based on stripped content (ignoring indentation)."""
        stripped_content = content.strip()
        if stripped_content:
            for orig_line in self.original_lines:
                if orig_line.strip() == stripped_content:
                    return orig_line
        else:
            for orig_line in self.original_lines:
                if not orig_line.strip():
                    return orig_line
        return None

    def find_exact_line_match(self, line_content: str) -> tuple[int, str] | None:
        """Find exact line match in original file, considering context for disambiguation."""
        # Fallback to original implementation for backward compatibility
        clean_line = line_content

        matches = []
        for i, orig_line in enumerate(self.original_lines):
            if orig_line == clean_line:  # Exact match including indentation
                matches.append((i, orig_line))

        if not matches:
            return None

        if len(matches) == 1:
            return matches[0]

        # Multiple matches - use context to disambiguate
        context_window = 3

        for match_line_idx, _ in matches:
            score = 0

            # Check lines before and after the potential match
            for offset in range(-context_window, context_window + 1):
                if offset == 0:  # Skip the target line itself
                    continue

                diff_idx = self.line_index + offset
                orig_idx = match_line_idx + offset

                if 0 <= diff_idx < len(self.lines) and 0 <= orig_idx < len(
                    self.original_lines
                ):
                    diff_line = self.lines[diff_idx]
                    orig_line = self.original_lines[orig_idx]

                    # Clean both lines for comparison
                    clean_diff_line = diff_line
                    if diff_line.startswith((" ", "+", "-")):
                        clean_diff_line = diff_line[1:]

                    if clean_diff_line == orig_line:
                        score += 1

            # Return the first match with any contextual support
            if score > 0:
                return (match_line_idx, matches[0][1])

        return matches[0]

    def format_split_part_with_context(self, part: str) -> str:
        """Format a split part with appropriate diff prefix based on strict original context."""
        # Skip the early return check - split parts need proper diff prefixes added
        # The original check was incorrectly treating content with leading spaces as already prefixed

        # Handle empty lines specially - they should be context lines if they exist in original
        if not part.strip():
            # Check if there are empty lines in the original file
            for orig_line in self.original_lines:
                if not orig_line.strip():
                    return " " + part  # Context line - empty line exists in original
            return "+" + part  # Addition - no empty lines in original

        # Check if this part exists exactly in original (including indentation)
        match = self.find_exact_line_match(part)
        if match:
            # Part already has correct indentation from original file, just add diff prefix
            return " " + part  # Context line - exact match found
        else:
            return "+" + part  # Addition - no exact match

    def add_appropriate_prefix(self, line: str) -> str:
        """Determine and add appropriate diff prefix based on strict original file matching."""
        if line.startswith((" ", "+", "-")):
            return line

        if not line.strip():
            return line

        # Check if this exact line (with indentation) exists in the original file
        match = self.find_exact_line_match(line)
        if match:
            return " " + line  # Context line - exact match found
        else:
            return "+" + line  # Addition - no exact match

    def find_properly_indented_line(self, content: str) -> str:
        """Find the line with proper indentation from the original file."""
        content_stripped = content.strip()
        if not content_stripped:
            return ""

        # Look for the line in the original file and return it with proper indentation
        for orig_line in self.original_lines:
            if orig_line.strip() == content_stripped:
                return orig_line

        # If not found, return the content as-is
        return content

    def split_joined_line_with_context(self, line: str) -> list[str]:
        """Split joined line using strict original file context matching."""
        potential_splits = self.get_potential_splits(line)

        # Find the best split based on exact matches in original file
        best_split = None
        best_match_score = 0

        for split_parts in potential_splits:
            score = 0
            for part in split_parts:
                # Count exact matches (including indentation)
                match = self.find_exact_line_match(part)
                if match:
                    score += 1

            if score > best_match_score:
                best_match_score = score
                best_split = split_parts

        # Only split if we found exact matches for the parts
        if best_split and len(best_split) > 1 and best_match_score > 1:
            # Apply proper diff prefixes based on strict original context
            result = []
            for part in best_split:
                formatted_part = self.format_split_part_with_context(part)
                result.append(formatted_part)
            return result

        # Fallback: treat as regular line
        return [self.add_appropriate_prefix(line)]

    def infer_hunk_header(self) -> str:
        """Infer proper hunk header from surrounding context."""
        # Look for the first deletion or context line to find the actual start position
        # This is more reliable for multi-hunk diffs than searching for general context

        result = find_hunk_location(
            "\n".join(self.lines), "\n".join(self.original_lines)
        )
        if (
            result and result[2] > DiffConfig.MIN_HUNK_LOCATION_CONFIDENCE
        ):  # TODO increase this as high as possible
            return self.calculate_hunk_header_from_anchor(result[0] + 1)
        # TODO avoid doing stuff below
        for i in range(self.line_index + 1, len(self.lines)):
            line = self.lines[i]

            # Skip empty lines and additions (focus on deletions and context)
            if not line.strip() or line.startswith("+"):
                continue

            # Look for deletion or context lines that can anchor our position
            search_text = None
            if line.startswith("-"):
                search_text = line[1:]  # Remove deletion prefix
            elif line.startswith(" "):
                search_text = line[1:]  # Remove context prefix
            else:
                # Raw line without prefix - could be missing prefix
                search_text = line

            # TODO: search_text is only one line
            if not search_text or not search_text.strip():
                continue

            # Find this exact line in the original file
            for j, orig_line in enumerate(self.original_lines):
                if orig_line == search_text:
                    # Found exact match - this is our anchor point
                    start_line = j + 1  # Convert to 1-based line number

                    # Calculate line counts based on actual diff content
                    return self.calculate_hunk_header_from_anchor(start_line)

        # Fallback: try to find any recognizable content for positioning
        candidates = []

        for i in range(self.line_index + 1, min(self.line_index + 10, len(self.lines))):
            if i >= len(self.lines):
                break

            line = self.lines[i]
            if not line.strip():
                continue

            # Extract searchable content
            search_text = line
            if line.startswith((" ", "+", "-")):
                search_text = line[1:]

            if not search_text.strip():
                continue

            # Find the most unique/specific lines for better positioning
            for j, orig_line in enumerate(self.original_lines):
                if orig_line == search_text:
                    # Store candidate with specificity score (longer lines are more unique)
                    specificity = len(search_text.strip())
                    candidates.append((specificity, j + 1))  # Store 1-based line number
                    break

        if candidates:
            # Choose the most specific (longest) match for better accuracy
            candidates.sort(key=lambda x: x[0], reverse=True)
            matched_line_num = candidates[0][1]  # 1-based

            return self.calculate_hunk_header_from_anchor(matched_line_num)

        # Ultimate fallback: create header based on available context lines
        remaining_lines = (
            self.lines[self.line_index + 1 :]
            if self.line_index + 1 < len(self.lines)
            else []
        )
        return DiffConfig.create_fallback_header(remaining_lines)

    def calculate_hunk_header_from_anchor(self, anchor_line: int) -> str:
        """Calculate hunk header using line-by-line reconciliation from anchor point.

        Enhanced to start from anchor_line and reconcile context.original_lines
        with the diff lines for more effective joined line detection.
        """
        # Extract diff content lines (everything after header until next hunk or end)
        diff_lines = []
        for i in range(self.line_index + 1, len(self.lines)):
            line = self.lines[i]
            # Stop if we hit another hunk header or file header
            if line.startswith(("@@", "---", "+++")):
                break
            diff_lines.append(line)

        if not diff_lines:
            return f"@@ -{anchor_line},1 +{anchor_line},1 @@"

        # Perform line-by-line reconciliation starting from anchor_line
        reconciliation_result = DiffFixer._reconcile_lines_from_anchor(
            anchor_line, diff_lines, self.original_lines
        )

        # Extract counts from reconciliation
        start_line = reconciliation_result["start_line"]
        orig_count = reconciliation_result["orig_count"]
        new_count = reconciliation_result["new_count"]

        # Ensure minimum counts of 1
        orig_count = max(1, orig_count)
        new_count = max(1, new_count)

        return f"@@ -{start_line},{orig_count} +{start_line},{new_count} @@"

    def fix_hunk_header(self, line: str) -> list[str]:
        """Fix malformed hunk headers."""
        if line.strip() == "@@ ... @@":
            # Infer line numbers from context
            return [self.infer_hunk_header()]

        # Handle content joined with hunk header or malformed headers
        if line.count("@@") != 2 or not re.match(
            r"^@@ -\d+,\d+ \+\d+,\d+ @@\s*$", line.strip()
        ):
            clean_header = DiffConfig.extract_clean_header(line)

            # Check if there was content joined to the header
            match = re.search(r"(@@ -\d+,\d+ \+\d+,\d+ @@)(.+)", line)
            if match and match.group(2).strip():
                # Return both the header and the joined content as separate lines
                joined_content_raw = match.group(2).strip()  # Remove extra spaces

                # Find the proper indentation from the original file
                properly_indented_content = self.find_properly_indented_line(
                    joined_content_raw
                )

                # Add appropriate diff prefix to the joined content
                if properly_indented_content:
                    content_line = " " + properly_indented_content
                else:
                    content_line = "+" + joined_content_raw

                # Verify the header line numbers are correct by checking if the joined content
                # actually exists at the claimed line number, and recalculate line counts
                header_match = DiffConfig.HUNK_HEADER_PATTERN.match(clean_header)
                if header_match and properly_indented_content:
                    claimed_start_line = int(header_match.group(1))
                    correct_start_line = claimed_start_line  # Default to claimed

                    # Check if the joined content exists at the claimed line
                    claimed_idx = claimed_start_line - 1  # Convert to 0-based
                    if (
                        claimed_idx >= len(self.original_lines)
                        or self.original_lines[claimed_idx] != properly_indented_content
                    ):
                        # Find the correct line number for the joined content
                        for i, orig_line in enumerate(self.original_lines):
                            if orig_line == properly_indented_content:
                                correct_start_line = i + 1  # Convert to 1-based
                                break

                    # Always recalculate line counts for headers with joined content
                    # since they're often wrong regardless of start line correctness

                    # Find the last line that has content in this hunk
                    hunk_content_lines = []
                    last_line_with_content = (
                        correct_start_line - 1
                    )  # Start from the start line

                    # Collect all hunk content lines
                    for j in range(self.line_index + 1, len(self.lines)):
                        next_line = self.lines[j]
                        # Stop if we hit another hunk header or file header
                        if next_line.startswith(("@@", "---", "+++")):
                            break
                        hunk_content_lines.append(next_line)

                    # Process the hunk content to find actual line references
                    deletion_count = 0
                    addition_count = 0

                    # Count actual line types and track the span
                    for line in hunk_content_lines:
                        if DiffConfig.is_deletion_line(line):
                            deletion_count += 1
                            # Find this deleted line in the original to track span
                            deleted_content = line[1:]
                            for k, orig_line in enumerate(self.original_lines):
                                if (
                                    orig_line == deleted_content
                                    and k >= correct_start_line - 1
                                ):
                                    last_line_with_content = max(
                                        last_line_with_content, k
                                    )
                                    break
                        elif DiffConfig.is_addition_line(line):
                            addition_count += 1
                        elif line.strip():  # Any other content line
                            # Try to find this line in the original to track span
                            line_content = line[1:] if line.startswith(" ") else line
                            for k, orig_line in enumerate(self.original_lines):
                                if (
                                    orig_line == line_content
                                    and k >= correct_start_line - 1
                                ):
                                    last_line_with_content = max(
                                        last_line_with_content, k
                                    )
                                    break

                    # Calculate the actual span from correct start to last content line
                    actual_span = last_line_with_content - (correct_start_line - 1) + 1

                    # The line counts should reflect the actual span
                    # For malformed headers, we assume the same old/new count since we're correcting position
                    old_count = actual_span
                    new_count = actual_span

                    clean_header = f"@@ -{correct_start_line},{old_count} +{correct_start_line},{new_count} @@"

                return [clean_header, content_line]

            return [clean_header]

        # Check if the header has incorrect line counts
        if self.has_incorrect_line_counts(line):
            # Infer correct header from content
            return [self.infer_hunk_header()]

        return [line]

    def would_split_improve_match(self, line: str) -> bool:
        """Check if splitting a line would better match the original file structure using strict matching."""
        # First check if the exact line (including indentation) exists in the original
        clean_line = line
        if line.startswith((" ", "+", "-")):
            clean_line = line[1:]

        exact_match = self.find_exact_line_match(clean_line)
        if exact_match:
            return False  # Don't split if exact match exists

        # Try different split strategies and see if parts match exactly
        potential_splits = self.get_potential_splits(line)

        for split_parts in potential_splits:
            # Check if split parts individually exist exactly in original
            exact_matches = 0
            for part in split_parts:
                exact_match = self.find_exact_line_match(part)
                if exact_match:
                    exact_matches += 1

            # Only split if we have exact matches for multiple parts
            if exact_matches > 1:
                return True

        return False

    def is_joined_line_in_context(self, line: str) -> bool:
        """Check if line appears to be joined by using only original file structure as context."""
        if not line.strip() or line.startswith(("@@",)):
            return False

        # Strip diff prefixes
        line_content = line
        if line.startswith((" ", "+", "-")):
            line_content = line[1:]

        # Skip empty content after prefix removal
        if not line_content.strip():
            return False

        # Generic approach: A line is likely joined if:
        # 1. It doesn't exist as-is in the original file, AND
        # 2. It can be split into parts that DO exist in the original file

        # First check if the line exists as-is in original
        exact_match = self.find_exact_line_match(line_content)
        if exact_match:
            return False  # Line exists as-is, no need to split

        # Check if we can split this line into parts that exist in original
        return self.would_split_improve_match(line)

    def validate_unchanged_line(self, line: str) -> str:
        """Validate that an unchanged line exists in the original file with exact indentation."""
        if not line.startswith(" "):
            return line

        content = line[1:]  # Remove the space prefix
        match = self.find_exact_line_match(content)

        if match:
            # Line exists in original with exact indentation - keep as unchanged
            return line
        else:
            # Try to find the closest match and correct indentation
            corrected_line = self.find_similar_match(content)
            if corrected_line is not None:
                # corrected_line already has proper indentation from original, just add diff prefix
                return " " + corrected_line
            else:
                # No match found - might be an addition instead
                return "+" + content

    def is_malformed_hunk_header(self, line: str) -> bool:
        """Check if a hunk header is malformed or has incorrect counts."""
        if not line.startswith("@@"):
            return False

        # Check basic format issues
        if (
            line.strip() == "@@ ... @@"
            or line.count("@@") != 2
            or not re.match(r"^@@ -\d+,\d+ \+\d+,\d+ @@\s*$", line)
        ):
            return True

        # Check if the line counts in the header match the actual diff content
        return self.has_incorrect_line_counts(line)

    def process_deletion_line(self, line: str) -> dict:
        """Process a deletion line and return the processed line with metadata.

        Returns:
            dict: Contains 'line' (processed line) and 'line_index' (original file index or None)
        """
        deletion_content = line[1:]  # Remove - prefix

        # Find exact match in original file
        exact_match = self.find_exact_line_match(deletion_content)

        if exact_match:
            # Valid deletion - line exists in original with correct indentation
            return {
                "line": line,  # Keep as-is
                "line_index": exact_match[0],  # Return the matched index
            }
        else:
            # Try to find similar line with corrected indentation
            corrected_line = self.find_similar_match(deletion_content)
            if corrected_line:
                return {
                    "line": "-" + corrected_line,
                    "line_index": None,  # Don't update index for corrected lines
                }
            else:
                # No match found - keep original but don't update index
                return {"line": line, "line_index": None}

    def process_context_line(self, line: str) -> dict:
        """Process a context line (space prefix + content).

        Returns:
            dict: Contains 'should_continue', 'lines', 'line_index'
        """
        content_after_prefix = line[1:]
        exact_match = self.find_exact_line_match(content_after_prefix)

        if exact_match:
            # Only check for missing context if we haven't just processed a deletion
            # and we're dealing with import-like patterns
            current_original_index = exact_match[0]
            lines_to_add = []

            # This is a properly formatted unchanged line - keep as-is
            lines_to_add.append(line)

            return {
                "should_continue": True,
                "lines": lines_to_add,
                "line_index": current_original_index,
            }
        elif self.is_joined_line_in_context(line):
            # Process as potential joined line instead - continue to rule processing
            return {
                "should_continue": False,
                "lines": [],
                "line_index": None,
            }
        else:
            # Line starts with space but doesn't match - this might be a line
            # missing a prefix where the content happens to start with spaces.
            # Check if the line as-is (without stripping) exists in original
            whole_line_match = self.find_exact_line_match(line)
            if whole_line_match:
                # The whole line (including leading spaces) exists in original
                # This is content missing its diff prefix - add space prefix
                return {
                    "should_continue": True,
                    "lines": [" " + line],
                    "line_index": None,
                }
            else:
                # Neither interpretation works - this is likely a malformed context line
                # that should be an addition instead
                return {
                    "should_continue": True,
                    "lines": ["+" + content_after_prefix],
                    "line_index": None,
                }

    def process_hunk_content(self) -> list[str]:
        """Common hunk content processing logic shared between single and multi-hunk processing."""
        processed_lines = []

        for i, line in enumerate(self.lines):
            self.line_index = i
            if DiffConfig.is_file_header(line):
                continue

            # Handle deletion lines
            if DiffConfig.is_deletion_line(line):
                processed_result = self.process_deletion_line(line)
                processed_lines.append(processed_result["line"])
                continue

            # Handle whitespace-only self lines
            if line.startswith(" ") and not line.strip():
                processed_result = self.validate_unchanged_line(line)
                processed_lines.append(processed_result)
                continue

            # Handle properly formatted self lines
            if line.startswith(" ") and line.strip():
                processed_result = self.process_context_line(line)
                if processed_result["should_continue"]:
                    processed_lines.extend(processed_result["lines"])
                    continue
                # If should_continue is False, continue to rule processing

            # Handle hunk headers with malformed content or incorrect counts
            if line.startswith("@@"):
                if self.is_malformed_hunk_header(line):
                    fix_result = self.fix_hunk_header(line)
                    processed_lines.extend(fix_result)
                else:
                    processed_lines.append(line)
                continue

            # Handle joined line detection and splitting
            if self.is_joined_line_in_context(line):
                split_result = self.split_joined_line_with_context(line)
                processed_lines.extend(split_result)
                continue

            # Handle missing diff prefixes
            if line.strip() and not line.startswith(
                ("@@", "---", "+++", " ", "+", "-")
            ):
                prefixed_line = self.add_appropriate_prefix(line)
                processed_lines.append(prefixed_line)
                continue

            # Keep other lines as-is
            processed_lines.append(line)

        return processed_lines


class HunkManager:
    """Unified manager for hunk operations including overlap detection and merging."""

    @staticmethod
    def extract_range_from_lines(hunk_lines: list[str]) -> tuple[int, int] | None:
        """Extract line range from hunk lines - unified method replacing duplicates."""
        if not hunk_lines:
            return None

        for line in hunk_lines:
            if line.startswith("@@"):
                match = DiffConfig.HUNK_HEADER_PATTERN.match(line)
                if match:
                    start_line, line_count = int(match.group(1)), int(match.group(2))
                    return (start_line, start_line + line_count - 1)
        return None

    @staticmethod
    def detect_overlaps(hunks: list[list[str]]) -> bool:
        """Unified overlap detection replacing duplicate methods."""
        if len(hunks) <= 1:
            return False

        # Extract ranges using unified method
        ranges = []
        for hunk_lines in hunks:
            hunk_range = HunkManager.extract_range_from_lines(hunk_lines)
            if hunk_range:
                ranges.append(hunk_range)

        # Check for overlaps using unified logic
        return HunkManager._has_range_overlaps(ranges)

    @staticmethod
    def _has_range_overlaps(ranges: list[tuple[int, int]]) -> bool:
        """Check if any ranges overlap - unified logic."""
        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                range1, range2 = ranges[i], ranges[j]
                if range1[0] <= range2[1] and range2[0] <= range1[1]:
                    return True
        return False

    @staticmethod
    def merge_overlapping_hunks(hunks: list[list[str]]) -> list[list[str]]:
        """Merge overlapping hunks with unified logic."""
        if not HunkManager.detect_overlaps(hunks):
            return hunks
        merged = []
        i = 0

        while i < len(hunks):
            current_hunk = hunks[i]
            current_range = HunkManager.extract_range_from_lines(current_hunk)

            if not current_range:
                merged.append(current_hunk)
                i += 1
                continue

            # Find overlapping hunks
            overlapping_group = [current_hunk]
            j = i + 1

            while j < len(hunks):
                next_hunk = hunks[j]
                next_range = HunkManager.extract_range_from_lines(next_hunk)

                if next_range and HunkManager._ranges_overlap(
                    current_range, next_range
                ):
                    overlapping_group.append(next_hunk)
                    current_range = (
                        min(current_range[0], next_range[0]),
                        max(current_range[1], next_range[1]),
                    )
                    j += 1
                else:
                    break

            if len(overlapping_group) > 1:
                merged_hunk = HunkManager._merge_hunk_group(overlapping_group)
                merged.append(merged_hunk)
                i = j
            else:
                merged.append(current_hunk)
                i += 1

        return merged

    @staticmethod
    def _ranges_overlap(range1: tuple[int, int], range2: tuple[int, int]) -> bool:
        """Check if two ranges overlap - unified method."""
        return range1[0] <= range2[1] and range2[0] <= range1[1]

    @staticmethod
    def _merge_hunk_group(hunk_group: list[list[str]]) -> list[str]:
        """Merge group of hunks preserving empty lines."""
        if len(hunk_group) == 1:
            return hunk_group[0]

        # Collect content while preserving empty lines
        all_content = []
        seen_content = set()

        for hunk in hunk_group:
            for line in hunk:
                # Skip headers
                if line.startswith(("---", "+++", "@@")):
                    continue

                # Always preserve empty context lines
                if line == " ":
                    all_content.append(line)
                elif line not in seen_content:
                    all_content.append(line)
                    seen_content.add(line)

        # Calculate counts
        context_count = sum(
            1 for line in all_content if line.startswith(" ") or not line.strip()
        )
        deletion_count = sum(1 for line in all_content if line.startswith("-"))
        addition_count = sum(1 for line in all_content if line.startswith("+"))

        # Use first hunk's start line
        start_line = DiffConfig.DEFAULT_FALLBACK_START_LINE
        first_range = HunkManager.extract_range_from_lines(hunk_group[0])
        if first_range:
            start_line = first_range[0]

        old_count = context_count + deletion_count
        new_count = context_count + addition_count

        new_header = f"@@ -{start_line},{old_count} +{start_line},{new_count} @@"
        return [new_header] + all_content

    @staticmethod
    def split_diff_into_hunks(diff_content: str) -> list[list[str]]:
        """Split a diff into individual hunks, each containing header + file headers."""
        lines = diff_content.strip().split("\n")
        hunks = []
        current_hunk = []
        file_headers = []

        # Collect file headers first
        i = 0
        while i < len(lines) and (
            lines[i].startswith("---") or lines[i].startswith("+++")
        ):
            file_headers.append(lines[i])
            i += 1

        # Process hunks
        while i < len(lines):
            line = lines[i]
            if line.startswith("@@"):
                # Start of new hunk
                if current_hunk:
                    hunks.append(file_headers + current_hunk)
                current_hunk = [line]  # Start with hunk header
            else:
                current_hunk.append(line)
            i += 1

        # Add the last hunk
        if current_hunk:
            hunks.append(file_headers + current_hunk)

        return hunks


class DiffFixer:
    """Main diff fixer orchestrating all components with improved architecture."""

    @classmethod
    def _process_file_headers(
        cls, lines: list[str], original_file_path: str, preserve_filenames: bool
    ) -> tuple[list[str], list[str]]:
        """Extract and process file headers common to both single and multi-hunk processing.

        Returns:
            tuple: (file_headers, remaining_lines)
        """
        file_headers = []
        remaining_lines = []

        for line in lines:
            if DiffConfig.is_file_header(line):
                fixed_header = cls._fix_file_header(
                    line, original_file_path, preserve_filenames
                )
                file_headers.append(fixed_header)
            else:
                remaining_lines.append(line)

        return file_headers, remaining_lines

    @staticmethod
    def _adjust_hunk_header_line_numbers(
        header_line: str, cumulative_offset: int
    ) -> str:
        """Adjust hunk header line numbers based on cumulative offset from previous hunks."""
        match = re.match(r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@", header_line)
        if not match:
            return header_line

        old_start, old_count, new_start, new_count = map(int, match.groups())

        # Adjust the new start line based on cumulative offset
        adjusted_new_start = new_start + cumulative_offset

        return f"@@ -{old_start},{old_count} +{adjusted_new_start},{new_count} @@"

    @staticmethod
    def _calculate_hunk_line_offset(hunk_lines: list[str]) -> int:
        """Calculate the net line offset this hunk contributes (additions - deletions)."""
        additions = 0
        deletions = 0

        for line in hunk_lines:
            if DiffConfig.is_addition_line(line):
                additions += 1
            elif DiffConfig.is_deletion_line(line):
                deletions += 1

        return additions - deletions

    @staticmethod
    def _add_missing_empty_line_additions(lines: list[str]) -> list[str]:
        """Add missing empty line additions before content additions following empty context lines.

        This detects patterns like:
        - Context line with content (e.g., ' fi')
        - Context empty line (e.g., ' ')
        - Addition with content (e.g., '+# New line')

        And converts the empty context to an empty addition, then adds the content addition.
        """
        if not lines:
            return lines

        result = []
        i = 0

        while i < len(lines):
            current_line = lines[i]

            # Look for the pattern: content context -> empty context -> content addition
            if i + 2 < len(lines):
                line1 = current_line
                line2 = lines[i + 1]
                line3 = lines[i + 2]

                # Pattern: context with content -> empty context -> content addition
                pattern_match = (
                    line1.startswith(" ")
                    and line1.strip()  # Context with content
                    and (line2.startswith(" ") or line2 == "")
                    and not line2.strip()  # Empty context (with or without prefix)
                    and line3.startswith("+")
                    and line3.strip()
                )  # Content addition

                if pattern_match:
                    # Add the content context line
                    result.append(line1)

                    # Convert the empty context line to an empty addition
                    result.append("+")

                    # Add the content addition
                    result.append(line3)

                    # Skip all three lines since we processed them
                    i += 3
                    continue

            result.append(current_line)
            i += 1

        return result

    @staticmethod
    def _recalculate_hunk_header(hunk_lines: list[str], original_content: str) -> str:
        """Recalculate hunk header with correct line numbers based on actual content."""

        # Find the current header
        header_line = None
        content_lines = []

        for line in hunk_lines:
            if line.startswith("@@"):
                header_line = line
            else:
                content_lines.append(line)

        if not header_line or not content_lines:
            return header_line or "@@ -1,1 +1,1 @@"

        # Find the first recognizable content line to anchor line number calculation

        result = find_hunk_location("\n".join(hunk_lines), original_content)
        if not result or result[2] <= DiffConfig.MIN_HUNK_LOCATION_CONFIDENCE:
            # Fallback to existing header if we can't find anchor
            return header_line
        anchor_line_num = result[0] + 1

        # Count lines in this hunk
        context_count = 0
        deletion_count = 0
        addition_count = 0

        for line in content_lines:
            if line.startswith(" "):
                context_count += 1
            elif line.startswith("-"):
                deletion_count += 1
            elif line.startswith("+"):
                addition_count += 1
            elif not line.strip():
                context_count += 1  # Empty lines are context

        # Calculate hunk boundaries
        old_count = context_count + deletion_count
        new_count = context_count + addition_count

        # The hunk should start from the anchor line
        old_start = anchor_line_num
        new_start = anchor_line_num  # Will be adjusted by caller for sequencing

        return f"@@ -{old_start},{old_count} +{new_start},{new_count} @@"

    @classmethod
    def run(
        cls,
        diff_content: str,
        original_content: str,
        original_file_path: str,
        preserve_filenames: bool = False,
        min_context_lines: int = 1,
    ) -> str:
        """Fix malformatted diff using configured rules.

        Uses unified processing logic but maintains compatibility by routing
        single-hunk diffs through optimized path.
        """
        if not diff_content.strip():
            return "\n"

        hunks = HunkManager.split_diff_into_hunks(diff_content)
        if not hunks:
            return "\n"

        # Fix file headers using common processing
        fixed_headers, _ = cls._process_file_headers(
            diff_content.strip().split("\n"), original_file_path, preserve_filenames
        )

        # Generic multi-hunk processing: fix each hunk individually, then merge with proper sequencing
        all_fixed_hunks = []

        # Process each hunk individually using common logic
        for hunk_content in hunks:
            if not hunk_content:
                continue

            context = DiffContext(
                lines=hunk_content,
                original_lines=original_content.split("\n"),
                line_index=0,
                preserve_filenames=preserve_filenames,
            )

            hunk_lines = context.process_hunk_content()
            hunk_lines = cls._add_context_lines_around_hunk(
                hunk_lines,
                original_content,
                min_context_lines,
            )
            hunk_lines = cls._add_missing_empty_line_additions(hunk_lines)

            all_fixed_hunks.append(hunk_lines)

        all_fixed_hunks = HunkManager.merge_overlapping_hunks(all_fixed_hunks)

        fixed_hunks = []
        line_offset = 0

        for hunk_lines in all_fixed_hunks:
            if not hunk_lines:
                continue

            # Process this hunk's lines
            for line in hunk_lines:
                if line.startswith("@@"):
                    # Check if header is already properly formatted (not malformed)
                    # First recalculate the header with correct line counts
                    corrected_header = cls._recalculate_hunk_header(
                        hunk_lines, original_content
                    )

                    # Then adjust for proper multi-hunk sequencing
                    adjusted_header = cls._adjust_hunk_header_line_numbers(
                        corrected_header, line_offset
                    )
                    fixed_hunks.append(adjusted_header)

                    # Calculate how this hunk changes the line count for subsequent hunks
                    line_offset += cls._calculate_hunk_line_offset(hunk_lines)
                else:
                    fixed_hunks.append(line)

        # Combine results
        fixed_lines = fixed_headers + fixed_hunks

        result = "\n".join(fixed_lines)

        # Diff patches should always end with a newline for proper git format
        # unless the result is empty
        if result and not result.endswith("\n"):
            result += "\n"
        return result

    @staticmethod
    def _fix_file_header(
        line: str, original_file_path: str, preserve_filenames: bool
    ) -> str:
        """Fix file header line."""
        if preserve_filenames:
            # Even when preserving filenames, fix missing a/ and b/ prefixes
            if line.startswith("---"):
                # Extract the file path from the header
                file_path = line[3:].strip()  # Remove "---" and whitespace
                # Add a/ prefix if not already present and not /dev/null
                if file_path != "/dev/null" and not file_path.startswith(("a/", "b/")):
                    return f"--- a/{file_path}"
                return line
            elif line.startswith("+++"):
                # Extract the file path from the header
                file_path = line[3:].strip()  # Remove "+++" and whitespace
                # Add b/ prefix if not already present and not /dev/null
                if file_path != "/dev/null" and not file_path.startswith(("a/", "b/")):
                    return f"+++ b/{file_path}"
                return line
            else:
                return line

        filename = Path(original_file_path).name
        if line.startswith("---"):
            return f"--- a/{filename}"
        else:
            return f"+++ b/{filename}"

    @classmethod
    def _reconcile_lines_from_anchor(
        cls, anchor_line: int, diff_lines: list[str], original_lines: list[str]
    ) -> dict:
        """Reconcile diff lines with original lines starting from anchor point.

        This method performs line-by-line matching to detect joined lines more effectively
        by comparing diff content with the original file structure.

        Returns:
            dict: Contains 'start_line', 'orig_count', 'new_count', 'processed_lines'
        """
        processed_lines = []
        orig_position = anchor_line - 1  # Convert to 0-based index
        context_lines = 0
        deletion_lines = 0
        addition_lines = 0

        # missing empty lines in the diff
        # TODO: modify processed_lines instead and use the result
        while diff_lines[0] != original_lines[orig_position][1:]:
            if original_lines[orig_position].strip() == "":
                orig_position += 1
            else:
                break
        anchor_line = orig_position + 1

        i = 0
        while i < len(diff_lines):
            line = diff_lines[i]

            # Handle different line types
            if line.startswith(" "):
                # Context line - should match original
                content = line[1:]
                if (
                    orig_position < len(original_lines)
                    and original_lines[orig_position] == content
                ):
                    # Direct match - advance both positions
                    processed_lines.append(line)
                    context_lines += 1
                    orig_position += 1
                else:
                    # No direct match - might be joined line or misaligned
                    join_result = cls._try_resolve_context_mismatch(
                        line, original_lines, orig_position
                    )
                    if join_result["resolved"]:
                        processed_lines.extend(join_result["lines"])
                        context_lines += join_result["context_count"]
                        deletion_lines += join_result["deletion_count"]
                        addition_lines += join_result["addition_count"]
                        orig_position += join_result["orig_advance"]
                        i += (
                            join_result["diff_advance"] - 1
                        )  # -1 because loop will increment
                    else:
                        # Treat as regular context line
                        processed_lines.append(line)
                        context_lines += 1
                        orig_position += 1

            elif line.startswith("-"):
                # Deletion line - should match original at current position
                content = line[1:]
                if (
                    orig_position < len(original_lines)
                    and original_lines[orig_position] == content
                ):
                    # Direct match - advance original position
                    processed_lines.append(line)
                    deletion_lines += 1
                    orig_position += 1
                else:
                    # Check for joined deletion line
                    join_result = cls._try_resolve_deletion_mismatch(
                        line, original_lines, orig_position
                    )
                    if join_result["resolved"]:
                        processed_lines.extend(join_result["lines"])
                        context_lines += join_result["context_count"]
                        deletion_lines += join_result["deletion_count"]
                        addition_lines += join_result["addition_count"]
                        orig_position += join_result["orig_advance"]
                        i += join_result["diff_advance"] - 1
                    else:
                        # Keep as-is, don't advance original position (might be out of sync)
                        processed_lines.append(line)
                        deletion_lines += 1

            elif line.startswith("+"):
                # Addition line - doesn't consume original lines
                processed_lines.append(line)
                addition_lines += 1

            elif not line.strip():
                # Empty line - treat as context
                if (
                    orig_position < len(original_lines)
                    and not original_lines[orig_position].strip()
                ):
                    # Matches empty line in original
                    processed_lines.append(" " + line)
                    context_lines += 1
                    orig_position += 1
                else:
                    # No matching empty line - treat as addition
                    processed_lines.append("+" + line)
                    addition_lines += 1

            else:
                # Line without prefix - determine appropriate prefix
                if (
                    orig_position < len(original_lines)
                    and original_lines[orig_position] == line
                ):
                    # Matches original - context line
                    processed_lines.append(" " + line)
                    context_lines += 1
                    orig_position += 1
                else:
                    # Check for joined line pattern
                    join_result = cls._try_resolve_unprefixed_mismatch(
                        line, original_lines, orig_position
                    )
                    if join_result["resolved"]:
                        processed_lines.extend(join_result["lines"])
                        context_lines += join_result["context_count"]
                        deletion_lines += join_result["deletion_count"]
                        addition_lines += join_result["addition_count"]
                        orig_position += join_result["orig_advance"]
                        i += join_result["diff_advance"] - 1
                    else:
                        # Default to addition
                        processed_lines.append("+" + line)
                        addition_lines += 1

            i += 1

        # Calculate final position span
        final_orig_position = orig_position
        lines_spanned = final_orig_position - (anchor_line - 1)

        return {
            "start_line": anchor_line,
            "orig_count": context_lines + deletion_lines,
            "new_count": context_lines + addition_lines,
            # TODO: processed_lines should be used
            "processed_lines": processed_lines,
            "lines_spanned": lines_spanned,
        }

    @classmethod
    def _try_resolve_context_mismatch(
        cls,
        line: str,
        original_lines: list[str],
        orig_pos: int,
    ) -> dict:
        """Try to resolve context line mismatch by checking for joined lines."""
        content = line[1:]  # Remove space prefix

        # Try to split the line and match parts to consecutive original lines
        potential_splits = cls._get_line_splits_for_reconciliation(
            content, original_lines, orig_pos
        )

        for split_info in potential_splits:
            if cls._validate_split_against_original(
                split_info, original_lines, orig_pos
            ):
                # Found valid split
                split_lines = []
                for part in split_info["parts"]:
                    if part.strip():  # Non-empty parts
                        split_lines.append(" " + part)
                    else:  # Empty parts
                        split_lines.append(" ")

                return {
                    "resolved": True,
                    "lines": split_lines,
                    "context_count": len(split_lines),
                    "deletion_count": 0,
                    "addition_count": 0,
                    "orig_advance": len(split_lines),
                    "diff_advance": 1,
                }

        return {"resolved": False}

    @classmethod
    def _try_resolve_deletion_mismatch(
        cls,
        line: str,
        original_lines: list[str],
        orig_pos: int,
    ) -> dict:
        """Try to resolve deletion line mismatch by checking for joined lines."""
        content = line[1:]  # Remove - prefix

        # Try to split the line and match parts to consecutive original lines
        potential_splits = cls._get_line_splits_for_reconciliation(
            content, original_lines, orig_pos
        )

        for split_info in potential_splits:
            if cls._validate_split_against_original(
                split_info, original_lines, orig_pos
            ):
                # Found valid split - all parts are deletions
                split_lines = []
                for part in split_info["parts"]:
                    split_lines.append("-" + part)

                return {
                    "resolved": True,
                    "lines": split_lines,
                    "context_count": 0,
                    "deletion_count": len(split_lines),
                    "addition_count": 0,
                    "orig_advance": len(split_lines),
                    "diff_advance": 1,
                }

        return {"resolved": False}

    @classmethod
    def _try_resolve_unprefixed_mismatch(
        cls,
        line: str,
        original_lines: list[str],
        orig_pos: int,
    ) -> dict:
        """Try to resolve unprefixed line mismatch by checking for joined lines."""
        # Try to split the line and match parts to consecutive original lines
        potential_splits = cls._get_line_splits_for_reconciliation(
            line, original_lines, orig_pos
        )

        for split_info in potential_splits:
            if cls._validate_split_against_original(
                split_info, original_lines, orig_pos
            ):
                # Found valid split - determine prefix for each part
                split_lines = []
                for i, part in enumerate(split_info["parts"]):
                    expected_orig_pos = orig_pos + i
                    if (
                        expected_orig_pos < len(original_lines)
                        and original_lines[expected_orig_pos] == part
                    ):
                        # Context line
                        split_lines.append(" " + part)
                    else:
                        # Addition line
                        split_lines.append("+" + part)

                context_count = sum(1 for ln in split_lines if ln.startswith(" "))
                addition_count = sum(1 for ln in split_lines if ln.startswith("+"))

                return {
                    "resolved": True,
                    "lines": split_lines,
                    "context_count": context_count,
                    "deletion_count": 0,
                    "addition_count": addition_count,
                    "orig_advance": context_count,  # Only advance for context lines
                    "diff_advance": 1,
                }

        return {"resolved": False}

    @staticmethod
    def _get_line_splits_for_reconciliation(
        line: str, original_lines: list[str], start_pos: int
    ) -> list[dict]:
        """Get potential line splits for reconciliation with original lines.

        Returns list of split info dicts with 'parts' and 'confidence' keys.
        """
        if not line.strip():
            return []

        splits = []
        for split_point in range(1, len(line)):
            left_part = line[:split_point].rstrip()
            right_part = line[split_point:].lstrip()

            if not left_part.strip() or not right_part.strip():
                continue

            # Check if left part matches original at start_pos
            if (
                start_pos + 1 < len(original_lines)
                and original_lines[start_pos] == left_part
                and original_lines[start_pos + 1].strip() == right_part.strip()
            ):
                splits.append(
                    {
                        "parts": [left_part, original_lines[start_pos + 1]],
                        "confidence": 0.9,
                    }
                )

        # Sort by confidence
        splits.sort(key=lambda x: x["confidence"], reverse=True)
        return splits[:3]  # Return top 3 candidates

    @staticmethod
    def _validate_split_against_original(
        split_info: dict, original_lines: list[str], start_pos: int
    ) -> bool:
        """Validate that split parts match consecutive original lines."""
        parts = split_info["parts"]

        for i, part in enumerate(parts):
            expected_pos = start_pos + i
            if (
                expected_pos >= len(original_lines)
                or original_lines[expected_pos] != part
            ):
                return False

        return True

    @classmethod
    def _add_context_lines_around_hunk(
        cls, hunk_lines: list[str], original_content: str, min_context_lines: int = 1
    ) -> list[str]:
        """Add context lines around a hunk if it has only added/deleted lines (no existing context)."""
        if not hunk_lines or not original_content.strip():
            # Don't add context for empty files or new file creation
            return hunk_lines

        # Find the header line
        header_idx = None
        for i, line in enumerate(hunk_lines):
            if line.startswith("@@"):
                header_idx = i
                break

        if header_idx is None:
            return hunk_lines

        header_line = hunk_lines[header_idx]
        content_lines = hunk_lines[header_idx + 1 :]

        # For malformed headers like "@@ ... @@", we need to infer the start line
        # by finding the first deletion line in the original content
        start_line = None
        original_lines = original_content.split("\n")

        if header_line.strip() == "@@ ... @@":
            # Find the first deletion line to determine the start position
            # For multi-hunk diffs, use the first deletions line we can find
            start_line = None

            for line in content_lines:
                if line.startswith("-"):
                    deletion_content = line[1:]  # Remove prefix
                    # Find this exact line in the original
                    for i, orig_line in enumerate(original_lines):
                        if orig_line == deletion_content:
                            start_line = i + 1  # Convert to 1-based
                            break
                    if start_line:
                        break

            if not start_line:
                start_line = 1
        else:
            # Parse header to get start line number
            match = DiffConfig.HUNK_HEADER_PATTERN.match(header_line)
            if match:
                claimed_start_line = int(match.group(1))
                # Verify that the content actually exists at the claimed line
                # If not, search for the correct location
                start_line = cls._verify_and_correct_start_line(
                    claimed_start_line, content_lines, original_lines
                )
            else:
                return hunk_lines

        # Don't add context lines for hunks that start at line 1 (beginning of file)
        # as there's no ambiguity and context may not be helpful, but only for non-malformed headers
        # since malformed headers might have wrong line numbers that need correction
        if start_line <= 1 and header_line.strip() != "@@ ... @@":
            return hunk_lines

        # Add context lines before the hunk content
        context_before = []
        for i in range(1, min_context_lines + 1):
            line_idx = start_line - 1 - i  # Convert to 0-based and go backwards
            if (
                0 <= line_idx < len(original_lines)
                and content_lines[min_context_lines - i].startswith(("-", "+"))
                # and line_idx + 1 < len(original_lines)  # do not if reached file end
            ):
                context_before.insert(0, " " + original_lines[line_idx])

        # Add context lines after the hunk content
        context_after = []
        # Find the last line that would be affected by the hunk
        addition_count = sum(1 for line in content_lines if line.startswith("+"))
        old_count = max(1, len(content_lines) - addition_count)
        last_affected_line = (
            start_line - 1 + old_count - 1
        )  # 0-based index of last affected line

        for i in range(1, min_context_lines + 1):
            line_idx = last_affected_line + i
            if (
                0 <= line_idx < len(original_lines)
                and content_lines[-i].startswith(("-", "+"))
                and line_idx + 1 < len(original_lines)  # do not if reached file end
            ):
                context_after.append(" " + original_lines[line_idx])

        # Combine: header + context_before + content + context_after
        updated_content_lines = context_before + content_lines + context_after

        # Recalculate header with new line counts
        updated_deletion_count = sum(
            1 for line in updated_content_lines if line.startswith("-")
        )
        updated_addition_count = sum(
            1 for line in updated_content_lines if line.startswith("+")
        )

        # Adjust start line to account for context before
        updated_start_line = start_line - len(context_before)
        updated_old_count = len(updated_content_lines) - updated_addition_count
        updated_new_count = len(updated_content_lines) - updated_deletion_count

        updated_header = f"@@ -{updated_start_line},{updated_old_count} +{updated_start_line},{updated_new_count} @@"

        result = hunk_lines[:header_idx] + [updated_header] + updated_content_lines
        # print("\n".join(result))
        return result

    @staticmethod
    def _verify_and_correct_start_line(
        claimed_start_line, content_lines, original_lines
    ) -> int:
        """Verify that the content actually exists at the claimed line.
        If not, search for the correct location and return the corrected start line."""
        # Find the first deletion or context line to use as an anchor
        anchor_content = None
        for line in content_lines:
            if line.startswith(("-", " ")) and line.strip():
                anchor_content = line[1:]  # Remove prefix
                break

        if not anchor_content:
            return claimed_start_line  # No anchor to verify with

        # Check if the claimed line actually contains the expected content
        claimed_idx = claimed_start_line - 1  # Convert to 0-based
        if (
            0 <= claimed_idx < len(original_lines)
            and original_lines[claimed_idx] == anchor_content
        ):
            return claimed_start_line  # Line number is correct

        # Line number is wrong, search for the correct location
        for i, orig_line in enumerate(original_lines):
            if orig_line == anchor_content:
                return i + 1  # Convert to 1-based

        # If we can't find the content, return the claimed line as fallback
        return claimed_start_line


def main():
    parser = argparse.ArgumentParser(description="Fix malformatted diff files")
    parser.add_argument("diff_file", help="Path to the diff file")
    parser.add_argument("original_file", help="Path to the original file")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument(
        "--preserve-filenames",
        action="store_true",
        help="Keep original filenames from diff instead of using actual file path",
    )

    args = parser.parse_args()

    try:
        diff_path = Path(args.diff_file)
        original_path = Path(args.original_file)

        if not diff_path.exists():
            print(f"Error: Diff file '{diff_path}' not found", file=sys.stderr)
            sys.exit(1)

        if not original_path.exists():
            print(f"Error: Original file '{original_path}' not found", file=sys.stderr)
            sys.exit(1)

        diff_content = diff_path.read_text()
        original_content = original_path.read_text()

        fixed_diff = DiffFixer.run(
            diff_content,
            original_content,
            str(original_path),
            args.preserve_filenames,
        )

        if args.output:
            Path(args.output).write_text(fixed_diff)
            print(f"Fixed diff written to {args.output}")
        else:
            print(fixed_diff)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
