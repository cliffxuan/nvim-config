from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

HUNK_HEADER_RE = re.compile(
    r"""^@@\s*-\s*(?P<old_start>\d+)(?:,(?P<old_count>\d+))?
          \s+\+\s*(?P<new_start>\d+)(?:,(?P<new_count>\d+))?
          \s*@@""",
    re.VERBOSE,
)


def _split_lines(text: str) -> list[str]:
    # Preserve file logical lines without trailing newlines.
    return text.splitlines()


def _normalize_line(s: str) -> str:
    # Whitespace-tolerant normalization: collapse all runs of whitespace to single space
    # and strip edges. Keeps content but ignores indentation/line-break weirdness.
    return re.sub(r"\s+", " ", s).strip()


def _strip_marker(line: str) -> str:
    # Remove leading diff marker and one following space if present.
    if not line:
        return line
    if line[0] in " +-":
        return line[2:] if len(line) >= 2 and line[1] == " " else line[1:]
    return line


def parse_hunk(hunk_text: str) -> dict[str, Any]:
    """
    Parse a unified-diff hunk block into components.

    Returns:
        {
          'header': {'old_start': int|None, 'old_count': int|None, 'new_start': int|None, 'new_count': int|None},
          'lines': [str, ...],      # raw lines inside the hunk (excluding the header)
          'old_segment': [str, ...] # context + deletions as plain code lines
        }
    """
    lines = _split_lines(hunk_text)
    header_info = {
        "old_start": None,
        "old_count": None,
        "new_start": None,
        "new_count": None,
    }

    # Extract header if present
    if lines and lines[0].startswith("@@"):
        m = HUNK_HEADER_RE.match(lines[0])
        if m:
            header_info = {
                "old_start": int(m.group("old_start"))
                if m.group("old_start")
                else None,
                "old_count": int(m.group("old_count"))
                if m.group("old_count")
                else None,
                "new_start": int(m.group("new_start"))
                if m.group("new_start")
                else None,
                "new_count": int(m.group("new_count"))
                if m.group("new_count")
                else None,
            }
        body = lines[1:]
    else:
        body = lines

    # Build the "old" side (context + deletions) as it appears in the original file
    old_segment: list[str] = []
    for ln in body:
        if not ln:
            # Keep empty line as-is (no marker); rare in malformed hunks
            old_segment.append(ln)
            continue
        tag = ln[0]
        if tag == " " or tag == "-":
            old_segment.append(_strip_marker(ln))
        elif tag in "+":
            # additions are not part of the original file segment
            continue
        else:
            # Unknown marker; treat as context line content
            old_segment.append(_strip_marker(ln))

    return {
        "header": header_info,
        "lines": body,
        "old_segment": old_segment,
    }


def _window_similarity(a: list[str], b: list[str]) -> float:
    """Whitespace-insensitive similarity between two lists of lines."""
    a_norm = "\n".join(_normalize_line(x) for x in a)
    b_norm = "\n".join(_normalize_line(x) for x in b)
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def find_hunk_location(
    hunk_text: str,
    original_text: str,
    *,
    trust_header_hint: bool = True,
    search_radius: int = 200,  # lines around the header hint to search if present
) -> tuple[int, int, float, dict[str, Any]] | None:
    """
    Find the best (start_line, end_line) in the original file where the hunk should apply.

    Args:
        hunk_text: The unified-diff hunk (may have imperfect header/whitespace).
        original_text: Full contents of the original file.
        trust_header_hint: If True, prioritize scanning around the old_start from header, if present.
        search_radius: Number of lines to search above/below the hinted start.

    Returns:
        (start_idx, end_idx, score, meta) or None if no reasonable location found.
        - start_idx/end_idx are 0-based, end-exclusive line indexes in original_text.
        - score is a similarity ratio (0..1).
        - meta includes parsed header and algorithm details.
    """
    parsed = parse_hunk(hunk_text)
    header = parsed["header"]
    old_segment = parsed["old_segment"]

    orig_lines = _split_lines(original_text)
    n = len(orig_lines)
    k = max(1, len(old_segment))  # window size (at least 1 line)

    # Early exit for trivial hunks
    if k == 0:
        return None

    # If header has old_count, prefer that as window size (but clamp reasonably)
    if header.get("old_count"):
        k = max(1, min(header["old_count"], n))

    candidates: list[tuple[int, int, float, str]] = []

    def _scan_range(lo: int, hi: int, label: str):
        lo = max(0, lo)
        hi = min(n, hi)
        if hi <= lo:
            return
        for start in range(lo, hi - k + 1):
            window = orig_lines[start : start + k]
            score = _window_similarity(window, old_segment)
            candidates.append((start, start + k, score, label))

    # Strategy 1: If header has old_start, search a band around that location (more precise & fast)
    if trust_header_hint and header.get("old_start"):
        # Header old_start is 1-based in unified diff; convert to 0-based.
        hinted_start = max(0, header["old_start"] - 1)
        _scan_range(
            hinted_start - search_radius, hinted_start + search_radius, "hint-band"
        )

    # Strategy 2: Global scan fallback (can be slower for huge files; still linear in practice)
    # Only do it if we didn't find a strong candidate in the hint band or no header
    if not candidates:
        _scan_range(0, n, "global")

    # Pick best candidate
    if not candidates:
        return None

    # Prefer higher score; break ties by choosing the one closer to hinted start if available
    best = max(candidates, key=lambda x: x[2])

    # Optional: slight preference to "hint-band" over "global" when scores are equal
    same_score = [c for c in candidates if abs(c[2] - best[2]) < 1e-9]
    if len(same_score) > 1:
        pref = [c for c in same_score if c[3] == "hint-band"]
        if pref:
            best = pref[0]

    start_idx, end_idx, score, label = best
    meta = {
        "method": "sequence_matcher_whitespace_tolerant",
        "search_scope": label,
        "parsed_header": header,
        "old_segment_len": len(old_segment),
        "window_size_used": end_idx - start_idx,
    }
    return (start_idx, end_idx, score, meta)
