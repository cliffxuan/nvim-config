#!/usr/bin/env python3
"""
Run all Python implementation tests.
This script can be used instead of pytest for running tests.
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def run_pattern_tests(pattern):
    """Run tests matching a pattern (fixtures or test modules)."""
    print("=" * 60)
    print(f"Running Tests Matching Pattern: {pattern}")
    print("=" * 60)

    try:
        from fixture_loader_v2 import FixtureLoader

        loader = FixtureLoader()

        # Find matching fixtures
        matching_fixtures = []
        try:
            all_fixtures = loader.load_all()
            for fixture in all_fixtures:
                fixture_path = fixture["_path"]
                fixture_name = fixture["name"]
                # Match against fixture path, name, or category
                if (
                    pattern.lower() in fixture_path.lower()
                    or pattern.lower() in fixture_name.lower()
                    or fixture_path.startswith(pattern)
                ):
                    matching_fixtures.append(fixture)
        except Exception as e:
            print(f"❌ Error loading fixtures: {e}")
            return 1

        # Find matching test modules
        matching_modules = []
        test_modules = [
            "tests.test_validation",
            "tests.test_patcher",
            "tests.test_python_implementation",
        ]

        for module_name in test_modules:
            if pattern.lower() in module_name.lower():
                matching_modules.append(module_name)

        # Check if we found anything
        if not matching_fixtures and not matching_modules:
            print(f"❌ No fixtures or test modules matching pattern '{pattern}'")
            print("\nAvailable fixtures:")
            try:
                categories = loader.get_categories()
                for category in categories:
                    fixtures = loader.load_category(category)
                    print(f"  {category}:")
                    for f in fixtures:
                        print(f"    {f['_path']}")
            except Exception:
                pass
            print("\nAvailable test modules:")
            for module in test_modules:
                print(f"  {module}")
            return 1

        # Show what we found
        if matching_fixtures:
            print(f"📋 Found {len(matching_fixtures)} matching fixture(s):")
            for fixture in matching_fixtures:
                print(f"  - {fixture['_path']}: {fixture['name']}")

        if matching_modules:
            print(f"📋 Found {len(matching_modules)} matching test module(s):")
            for module in matching_modules:
                print(f"  - {module}")

        print("-" * 40)

        total_tests = 0
        passed_tests = 0

        # Run matching test modules first
        for module_name in matching_modules:
            try:
                print(f"\n📋 Running all tests in {module_name}...")
                module = __import__(module_name, fromlist=[""])

                # Get all test functions
                test_functions = [
                    name for name in dir(module) if name.startswith("test_")
                ]

                for test_func_name in test_functions:
                    # Skip parametrized test functions that don't take fixtures
                    if test_func_name in [
                        "test_fixture_case",
                        "test_patcher_fixture_case",
                        "test_validation_fixture_case",
                    ]:
                        continue

                    total_tests += 1
                    test_func = getattr(module, test_func_name)

                    try:
                        # Special handling for fixture-based tests
                        if "fixture" in test_func_name.lower():
                            all_fixtures = loader.load_all()
                            test_func(all_fixtures)
                        else:
                            test_func()

                        print(f"  ✅ {test_func_name}")
                        passed_tests += 1
                    except Exception as e:
                        print(f"  ❌ {test_func_name}: {str(e)}")

            except Exception as e:
                print(f"❌ Failed to import {module_name}: {e}")

        # Run matching fixtures
        test_module_functions = [
            ("tests.test_validation", "test_validation_single_fixture"),
            ("tests.test_patcher", "test_patcher_single_fixture"),
            ("tests.test_python_implementation", "test_implementation_single_fixture"),
        ]

        for fixture in matching_fixtures:
            print(f"\n📋 Running fixture: {fixture['_path']}")
            print(f"📝 {fixture['name']}: {fixture['description']}")

            for module_name, test_func_name in test_module_functions:
                try:
                    module = __import__(module_name, fromlist=[""])

                    if hasattr(module, test_func_name):
                        total_tests += 1
                        test_func = getattr(module, test_func_name)

                        try:
                            test_func(fixture)
                            print(f"  ✅ {module_name}.{test_func_name}")
                            passed_tests += 1
                        except Exception as e:
                            print(f"  ❌ {module_name}.{test_func_name}: {str(e)}")

                except Exception as e:
                    print(f"❌ Failed to import {module_name}: {e}")

        print("\n" + "=" * 60)
        print(f"📊 Test Results: {passed_tests}/{total_tests} passed")

        if passed_tests == total_tests:
            print("🎉 All tests passed!")
            return 0
        else:
            print(f"💥 {total_tests - passed_tests} tests failed")
            return 1

    except Exception as e:
        print(f"❌ Error running pattern tests: {e}")
        return 1


def run_tests():
    """Run all test modules."""
    print("=" * 60)
    print("Running Python Implementation Tests")
    print("=" * 60)

    # Import and run test modules
    test_modules = [
        "tests.test_validation",
        "tests.test_patcher",
        "tests.test_python_implementation",
    ]

    total_tests = 0
    passed_tests = 0

    for module_name in test_modules:
        try:
            print(f"\n📋 Running {module_name}...")
            module = __import__(module_name, fromlist=[""])

            # Get all test functions
            test_functions = [name for name in dir(module) if name.startswith("test_")]

            for test_func_name in test_functions:
                # Skip parametrized test functions that don't take fixtures
                if test_func_name in [
                    "test_fixture_case",
                    "test_patcher_fixture_case",
                    "test_validation_fixture_case",
                ]:
                    continue

                total_tests += 1
                test_func = getattr(module, test_func_name)

                try:
                    # Special handling for fixture-based tests
                    if "fixture" in test_func_name.lower():
                        # Load fixtures
                        from fixture_loader_v2 import FixtureLoader

                        loader = FixtureLoader()
                        all_fixtures = []
                        try:
                            all_fixtures.extend(loader.load_category("pass"))
                        except Exception:
                            pass
                        try:
                            all_fixtures.extend(loader.load_category("fail"))
                        except Exception:
                            pass

                        test_func(all_fixtures)
                    else:
                        test_func()

                    print(f"  ✅ {test_func_name}")
                    passed_tests += 1

                except Exception as e:
                    print(f"  ❌ {test_func_name}: {str(e)}")

        except Exception as e:
            print(f"❌ Failed to import {module_name}: {e}")

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"💥 {total_tests - passed_tests} tests failed")
        return 1


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Run Python implementation tests")
    parser.add_argument(
        "-t",
        "--test",
        help="Run tests matching a pattern (fixtures, modules, or names)",
    )

    args = parser.parse_args()

    if args.test:
        return run_pattern_tests(args.test)
    else:
        return run_tests()


if __name__ == "__main__":
    sys.exit(main())
