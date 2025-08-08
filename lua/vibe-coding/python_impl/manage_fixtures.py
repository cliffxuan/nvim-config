#!/usr/bin/env python3
"""
Unified command-line tool for managing diff fixtures.

Provides two main subcommands:
1. process - Process all fixtures to create diff_formatted files and optionally generate expected files
2. check - Compare the performance of the DiffFixer against the original

Usage examples:
  python manage_fixtures.py process                    # Process diffs only
  python manage_fixtures.py process --generate-expected # Generate expected files only
  python manage_fixtures.py process --all              # Do both operations
  python manage_fixtures.py process -p                 # Preserve filenames in diffs
  python manage_fixtures.py process --filter "joined"  # Process only fixtures with "joined" in name

  python manage_fixtures.py check                      # Test all fixtures
  python manage_fixtures.py check -f "joined"          # Test fixtures with "joined" in name
  python manage_fixtures.py check -u                   # Use pre-generated diff_formatted files
"""

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# Import from local modules
from fix_diff import DiffFixer

# Define paths - navigate up from python_impl to tests/fixtures/pass
PWD = Path(__file__).absolute().parent.parent / "tests" / "fixtures" / "pass"


def extract_filenames_from_diff(diff_content):
    """Extract the source and target filenames from diff headers."""
    lines = diff_content.strip().split("\n")
    source_file = None
    target_file = None

    for line in lines:
        if line.startswith("--- "):
            # Extract filename from "--- a/filename" or "--- filename"
            match = re.match(r"---\s+(?:a/)?(.+)$", line)
            if match:
                source_file = match.group(1)
        elif line.startswith("+++ "):
            # Extract filename from "+++ b/filename" or "+++ filename"
            match = re.match(r"\+\+\+\s+(?:b/)?(.+)$", line)
            if match:
                target_file = match.group(1)

    return source_file, target_file


def generate_expected_files(filter_name=None):
    """Generate expected files by applying diff_formatted to original files."""
    current_dir = PWD
    generated_count = 0
    failed_count = 0
    skipped_count = 0

    print("Generating expected files by applying diff_formatted to original files...")
    if filter_name:
        print(f"Filtering fixtures by name containing: '{filter_name}'\n")

    # Find all directories that contain diff_formatted and original files
    for item in current_dir.iterdir():
        if not item.is_dir() or item.name in ["venv", "__pycache__", ".git"]:
            continue

        # Apply name filter if specified
        if filter_name and filter_name.lower() not in item.name.lower():
            continue

        diff_formatted = item / "diff_formatted"

        # Find original file (could be .py, .lua, .js, .txt, etc.)
        original_files = list(item.glob("original.*"))

        if not diff_formatted.exists():
            continue

        if not original_files:
            print(f"⚠ Skipped {item.name}: No original file found")
            skipped_count += 1
            continue

        original_file = original_files[0]  # Take first original file found

        try:
            # Read the diff to understand what filename it expects
            with open(diff_formatted, "r") as f:
                diff_content = f.read()

            source_filename, target_filename = extract_filenames_from_diff(diff_content)

            if not source_filename or not target_filename:
                print(f"✗ Failed to extract filenames from {item.name}")
                failed_count += 1
                continue

            # Use a temporary directory for applying the diff
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Initialize git repo in temp directory
                subprocess.run(
                    ["git", "init"], cwd=temp_path, capture_output=True, check=True
                )
                subprocess.run(
                    ["git", "config", "user.email", "test@example.com"],
                    cwd=temp_path,
                    capture_output=True,
                )
                subprocess.run(
                    ["git", "config", "user.name", "Test"],
                    cwd=temp_path,
                    capture_output=True,
                )

                # Copy original file with the expected name
                expected_source = temp_path / source_filename
                expected_source.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original_file, expected_source)

                # Add and commit the original file
                subprocess.run(
                    ["git", "add", "."], cwd=temp_path, capture_output=True, check=True
                )
                subprocess.run(
                    ["git", "commit", "-m", "Initial"],
                    cwd=temp_path,
                    capture_output=True,
                    check=True,
                )

                # Apply the diff
                result = subprocess.run(
                    ["git", "apply", "--verbose", "-"],
                    input=diff_content,
                    text=True,
                    cwd=temp_path,
                    capture_output=True,
                )

                if result.returncode == 0:
                    # Copy the result to expected file in original location
                    result_file = temp_path / target_filename
                    if result_file.exists():
                        # Determine expected filename based on original file extension
                        expected_file = item / f"expected{original_file.suffix}"
                        shutil.copy2(result_file, expected_file)
                        print(f"✓ Generated expected file for {item.name}")
                        generated_count += 1
                    else:
                        print(f"✗ Result file not found for {item.name}")
                        failed_count += 1
                else:
                    print(
                        f"✗ Failed to apply diff for {item.name}: {result.stderr.strip()}"
                    )
                    failed_count += 1

        except Exception as e:
            print(f"✗ Error processing {item.name}: {e}")
            failed_count += 1

    print("\nExpected files generation summary:")
    print(f"  Generated: {generated_count} expected files")
    print(f"  Failed: {failed_count} fixtures")
    print(f"  Skipped: {skipped_count} fixtures")

    return generated_count, failed_count, skipped_count


def process_fixtures(preserve_filenames=False, filter_name=None):
    """Process all fixture directories and create diff_formatted files."""
    current_dir = PWD
    processed_count = 0
    failed_count = 0

    print("Processing fixture diffs to create diff_formatted files...")
    if filter_name:
        print(f"Filtering fixtures by name containing: '{filter_name}'\n")

    # Find all directories that contain diff and original files
    for item in current_dir.iterdir():
        if not item.is_dir() or item.name in ["venv", "__pycache__", ".git"]:
            continue

        # Apply name filter if specified
        if filter_name and filter_name.lower() not in item.name.lower():
            continue

        diff_file = item / "diff"

        # Find original file (could be .py, .lua, .js, .txt, etc.)
        original_files = list(item.glob("original.*"))
        if diff_file.exists() and original_files:
            original_file = original_files[0]  # Take first original file found
            output_file = item / "diff_formatted"
            try:
                # Run the fix_diff.py script (use absolute path)
                fix_diff_path = Path(__file__).parent / "fix_diff.py"
                cmd = [
                    "python",
                    str(fix_diff_path),
                    str(diff_file),
                    str(original_file),
                    "-o",
                    str(output_file),
                ]

                if preserve_filenames:
                    cmd.append("--preserve-filenames")

                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )

                print(f"✓ Processed {item.name}")
                processed_count += 1

            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to process {item.name}: {e.stderr.strip()}")
                failed_count += 1
            except Exception as e:
                print(f"✗ Error processing {item.name}: {e}")
                failed_count += 1

    print("\nDiff processing summary:")
    print(f"  Processed: {processed_count} fixtures")
    print(f"  Failed: {failed_count} fixtures")

    return processed_count, failed_count


def generate_correct_diff(fixture_dir, original_file, expected_file):
    """Generate what the correct diff should look like by comparing original vs expected."""
    if not expected_file or not expected_file.exists():
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Copy original file to temp directory
        temp_file = tmpdir_path / original_file.name
        temp_file.write_text(original_file.read_text())

        # Initialize git
        subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=tmpdir,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmpdir,
            capture_output=True,
        )
        subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial"], cwd=tmpdir, capture_output=True
        )

        # Replace with expected content
        temp_file.write_text(expected_file.read_text())

        # Generate diff
        result = subprocess.run(
            ["git", "diff"], cwd=tmpdir, capture_output=True, text=True
        )

        return result.stdout if result.returncode == 0 else None


def test_fixer_on_fixtures(filter_name=None, use_formatted=False, debug=False):
    """Test the fixer on existing fixtures."""
    fixer = DiffFixer()

    success_count = 0
    total_count = 0
    successful_fixtures = []
    failed_fixtures = []
    content_validated_count = 0
    git_only_count = 0

    print("Testing DiffFixer on fixtures...\n")
    if filter_name:
        print(f"Filtering fixtures by name containing: '{filter_name}'\n")
    if use_formatted:
        print("Using pre-generated formatted diffs (diff_formatted)\n")
    if debug:
        print("Debug mode enabled: will show correct diff for failures\n")

    for fixture_name in sorted(PWD.iterdir()):
        if not fixture_name.is_dir() or fixture_name.name in [
            "venv",
            "__pycache__",
            ".git",
        ]:
            continue

        # Apply name filter if specified
        if filter_name and filter_name.lower() not in fixture_name.name.lower():
            continue

        fixture_dir = Path(PWD / fixture_name)
        if not fixture_dir.exists():
            continue

        # Choose diff file based on use_formatted flag
        if use_formatted:
            diff_file = fixture_dir / "diff_formatted"
        else:
            diff_file = fixture_dir / "diff"

        original_files = list(fixture_dir.glob("original.*"))
        expected_files = list(fixture_dir.glob("expected.*"))

        if not diff_file.exists() or not original_files:
            continue

        original_file = original_files[0]
        expected_file = expected_files[0] if expected_files else None
        total_count += 1

        try:
            # Read files
            diff_content = diff_file.read_text()
            original_content = original_file.read_text()

            # Fix with fixer (only if not using pre-formatted)
            if use_formatted:
                # Use the pre-formatted diff as-is
                fixed_diff = diff_content
            else:
                # Generate fixed diff on the fly
                fixed_diff = fixer.fix_diff(
                    diff_content, original_content, str(original_file)
                )

            # Test with git apply
            with tempfile.TemporaryDirectory() as tmpdir:
                test_file = Path(tmpdir) / original_file.name
                test_file.write_text(original_content)

                # Initialize git
                subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
                subprocess.run(
                    ["git", "config", "user.email", "test@example.com"],
                    cwd=tmpdir,
                    capture_output=True,
                )
                subprocess.run(
                    ["git", "config", "user.name", "Test"],
                    cwd=tmpdir,
                    capture_output=True,
                )
                subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
                subprocess.run(
                    ["git", "commit", "-m", "Initial"], cwd=tmpdir, capture_output=True
                )

                # Test git apply --check first
                check_result = subprocess.run(
                    ["git", "apply", "--check", "-"],
                    input=fixed_diff,
                    text=True,
                    cwd=tmpdir,
                    capture_output=True,
                )

                if check_result.returncode != 0:
                    mode_indicator = " [formatted]" if use_formatted else ""
                    print(
                        f"✗ {fixture_name.name}{mode_indicator}: {check_result.stderr.strip()}"
                    )

                    if debug:
                        print(f"\n--- DEBUG INFO for {fixture_name.name} ---")
                        correct_diff = generate_correct_diff(
                            fixture_dir, original_file, expected_file
                        )
                        if correct_diff:
                            print("## What the correct diff should look like:")
                            print(correct_diff)
                        print("## What the fixer generated:")
                        print(fixed_diff)
                        print("--- END DEBUG INFO ---\n")

                    failed_fixtures.append(fixture_name)
                    continue

                # Apply the diff and check the result
                apply_result = subprocess.run(
                    ["git", "apply", "-"],
                    input=fixed_diff,
                    text=True,
                    cwd=tmpdir,
                    capture_output=True,
                )

                if apply_result.returncode != 0:
                    mode_indicator = " [formatted]" if use_formatted else ""
                    print(
                        f"✗ {fixture_name.name}{mode_indicator}: Apply failed - {apply_result.stderr.strip()}"
                    )

                    if debug:
                        print(f"\n--- DEBUG INFO for {fixture_name.name} ---")
                        correct_diff = generate_correct_diff(
                            fixture_dir, original_file, expected_file
                        )
                        if correct_diff:
                            print("## What the correct diff should look like:")
                            print(correct_diff)
                        print("## What the fixer generated:")
                        print(fixed_diff)
                        print("--- END DEBUG INFO ---\n")

                    failed_fixtures.append(fixture_name)
                    continue

                # Read the result after applying the diff
                actual_content = test_file.read_text()

                # Compare with expected if expected file exists
                if expected_file and expected_file.exists():
                    expected_content = expected_file.read_text()
                    if actual_content == expected_content:
                        mode_indicator = " [formatted]" if use_formatted else ""
                        print(f"✓ {fixture_name.name}{mode_indicator}: Success")
                        success_count += 1
                        content_validated_count += 1
                        successful_fixtures.append(fixture_name)
                    else:
                        mode_indicator = " [formatted]" if use_formatted else ""
                        print(
                            f"✗ {fixture_name.name}{mode_indicator}: Content mismatch"
                        )
                        # Show more informative diff for debugging
                        actual_lines = actual_content.splitlines()
                        expected_lines = expected_content.splitlines()

                        if len(actual_lines) != len(expected_lines):
                            print(
                                f"    Line count differs: expected {len(expected_lines)}, got {len(actual_lines)}"
                            )

                        # Show first few differing lines
                        for i, (actual_line, expected_line) in enumerate(
                            zip(actual_lines, expected_lines)
                        ):
                            if actual_line != expected_line:
                                print(f"    Line {i + 1} differs:")
                                print(f"      Expected: {repr(expected_line)}")
                                print(f"      Actual:   {repr(actual_line)}")
                                break  # Show only first difference for brevity

                        # If content is small, show full content
                        if len(actual_content) < 200 and len(expected_content) < 200:
                            print(f"    Full expected: {repr(expected_content)}")
                            print(f"    Full actual:   {repr(actual_content)}")

                        if debug:
                            print(f"\n--- DEBUG INFO for {fixture_name.name} ---")
                            correct_diff = generate_correct_diff(
                                fixture_dir, original_file, expected_file
                            )
                            if correct_diff:
                                print("## What the correct diff should look like:")
                                print(correct_diff)
                            print("## What the fixer generated:")
                            print(fixed_diff)
                            print("--- END DEBUG INFO ---\n")

                        failed_fixtures.append(fixture_name)
                else:
                    # No expected file - just check git apply worked
                    mode_indicator = " [formatted]" if use_formatted else ""
                    print(
                        f"✓ {fixture_name.name}{mode_indicator}: Success (no expected file)"
                    )
                    success_count += 1
                    git_only_count += 1
                    successful_fixtures.append(fixture_name)

        except Exception as e:
            mode_indicator = " [formatted]" if use_formatted else ""
            print(f"✗ {fixture_name.name}{mode_indicator}: Error - {e}")
            failed_fixtures.append(fixture_name)

    print(f"\n{'=' * 50}")
    print("FIXER RESULTS")
    print(f"{'=' * 50}")
    print(
        f"Success rate: {success_count}/{total_count} ({success_count / total_count * 100:.1f}%)"
    )

    if content_validated_count > 0 or git_only_count > 0:
        print(f"  - Content validated: {content_validated_count}")
        print(f"  - Git apply only: {git_only_count}")

    if successful_fixtures:
        print(f"\n✓ SUCCESSFUL ({len(successful_fixtures)}):")
        for fixture in successful_fixtures:
            print(f"  - {fixture.name}")

    if failed_fixtures:
        print(f"\n✗ FAILED ({len(failed_fixtures)}):")
        for fixture in failed_fixtures:
            print(f"  - {fixture.name}")


def process_command(args):
    """Handle the process subcommand."""
    # Determine what operations to perform
    should_process_diffs = not args.generate_expected or args.all
    should_generate_expected = args.generate_expected or args.all

    # Show what we're going to do
    operations = []
    if should_process_diffs:
        operations.append("process diffs to create diff_formatted files")
    if should_generate_expected:
        operations.append("generate expected files by applying diffs")

    print("This script will:")
    for i, op in enumerate(operations, 1):
        print(f"  {i}. {op}")
    print()

    # Confirmation for potentially destructive operations
    if should_generate_expected and not args.force:
        print(
            "WARNING: Generating expected files will overwrite existing expected.* files."
        )
        response = input("Do you want to continue? (y/N): ")
        if response.lower() not in ["y", "yes"]:
            print("Aborted.")
            return

    total_processed = 0
    total_failed = 0
    total_generated = 0
    total_gen_failed = 0

    # Execute operations
    if should_process_diffs:
        processed, failed = process_fixtures(
            preserve_filenames=args.preserve_filenames, filter_name=args.filter
        )
        total_processed += processed
        total_failed += failed

    if should_generate_expected:
        if should_process_diffs:
            print()  # Add spacing between operations
        generated, gen_failed, _ = generate_expected_files(filter_name=args.filter)
        total_generated += generated
        total_gen_failed += gen_failed

    # Final summary
    print(f"\n{'=' * 50}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 50}")

    if should_process_diffs:
        print(f"Diff processing: {total_processed} processed, {total_failed} failed")

    if should_generate_expected:
        print(f"Expected files: {total_generated} generated, {total_gen_failed} failed")

    if (total_failed == 0 and total_gen_failed == 0) and (
        total_processed > 0 or total_generated > 0
    ):
        print("\n✓ All operations completed successfully!")
    elif total_processed > 0 or total_generated > 0:
        print("\n⚠ Operations completed with some failures.")
    else:
        print("\n✗ No operations were successful.")


def check_command(args):
    """Handle the check subcommand."""
    test_fixer_on_fixtures(
        filter_name=args.filter, use_formatted=args.use_formatted, debug=args.debug
    )


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Manage diff fixtures: process diffs and check fixer performance"
    )

    # Create subparsers for the two main commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process subcommand
    process_parser = subparsers.add_parser(
        "process", help="Process fixture diffs and generate expected files"
    )
    process_parser.add_argument(
        "--preserve-filenames",
        "-p",
        action="store_true",
        help="Preserve original filenames in the diff output",
    )
    process_parser.add_argument(
        "--generate-expected",
        "-g",
        action="store_true",
        help="Generate expected files by applying diff_formatted to original files",
    )
    process_parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Do both operations: process diffs AND generate expected files",
    )
    process_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts for destructive operations",
    )
    process_parser.add_argument(
        "--filter",
        "-f",
        help="Filter fixtures by name (partial match, case-insensitive)",
    )

    # Check subcommand
    check_parser = subparsers.add_parser(
        "check", help="Test DiffFixer performance on fixture files"
    )
    check_parser.add_argument(
        "--filter",
        "-f",
        help="Filter fixtures by name (partial match, case-insensitive)",
    )
    check_parser.add_argument(
        "--use-formatted",
        "-u",
        action="store_true",
        help="Use pre-generated diff_formatted files instead of generating on the fly",
    )
    check_parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Show debug information for failed fixtures (what the correct diff should look like)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Handle subcommands
    if args.command == "process":
        process_command(args)
    elif args.command == "check":
        check_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
