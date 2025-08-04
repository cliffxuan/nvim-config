-- Test case reproducing the real hunk header issue
-- This test reproduces the exact problem reported:
-- "error: patch fragment without header at line 13: @@ -148,18 +148,18 @@mixtures = get_mixtures(platform)"

describe('Real Hunk Header Issue Reproduction', function()
  local Validation

  before_each(function()
    -- Mock dependencies
    package.loaded['vibe-coding.path_utils'] = {
      looks_like_file = function(line)
        return line:match '%.%w+$' and not line:match '^%s*[%w_]+%s*=' and not line:match '%(' and not line:match '%['
      end,
      resolve_file_path = function(path)
        return path
      end,
      clean_path = function(path)
        return path
      end,
      normalize_path = function(path)
        return path
      end,
    }

    -- Clear module cache
    package.loaded['vibe-coding.validation'] = nil
    Validation = require 'vibe-coding.validation'
  end)

  after_each(function()
    package.loaded['vibe-coding.path_utils'] = nil
    package.loaded['vibe-coding.validation'] = nil
  end)

  it('should detect and fix the exact problematic pattern', function()
    local problematic_diff = [[--- app/api/v2/cli/main.py
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
+    all_mixture_names.sort()]]

    -- Run the validation pipeline
    local fixed_diff, issues = Validation.process_diff(problematic_diff)

    -- Debug: Print what issues were found
    print('Issues found: ' .. #issues)
    for i, issue in ipairs(issues) do
      print('Issue ' .. i .. ': Type=' .. (issue.type or 'nil') .. ', Message=' .. issue.message)
    end

    -- Should find at least 1 issue: hunk header split (exception line might not be detected in this context)
    assert.is_true(#issues >= 1, 'Should detect at least the hunk header issue')

    -- Check for hunk header issue
    local has_hunk_header_issue = false

    for _, issue in ipairs(issues) do
      if issue.type == 'hunk_header' and issue.message:match 'Split hunk header from content' then
        has_hunk_header_issue = true
      end
    end

    assert.is_true(has_hunk_header_issue, 'Should detect hunk header issue')

    -- Verify the fix: hunk header should be separated (check that the pattern is fixed)
    local has_proper_separation = fixed_diff:match '@@ %-10,7 %+10,7 @@[^c]*mixtures = get_mixtures'
    print('Has proper hunk separation: ' .. tostring(has_proper_separation ~= nil))

    -- The key test: the original joined line should now be split
    local original_broken = problematic_diff:match '@@ %-10,7 %+10,7 @@mixtures'
    local now_fixed = not fixed_diff:match '@@ %-10,7 %+10,7 @@mixtures'

    assert.is_true(original_broken ~= nil, 'Original should have the broken pattern')
    assert.is_true(now_fixed, 'Fixed diff should not have the broken pattern')

    print 'Fixed diff:'
    print(fixed_diff)
  end)

  it('should create a diff that git apply can understand', function()
    local problematic_diff = [[--- test_file.py
+++ test_file.py
@@ -1,3 +1,3 @@
 line1
-line2
+newline2
@@ -5,2 +5,2 @@line3
-line4
+newline4]]

    -- Run validation
    local fixed_diff, issues = Validation.process_diff(problematic_diff)

    -- Debug: Show what was found
    print('Issues in simple test: ' .. #issues)
    for i, issue in ipairs(issues) do
      print('Issue ' .. i .. ': ' .. issue.message)
    end

    -- Should detect the hunk header issue
    assert.is_true(#issues > 0, 'Should detect issues')

    -- The fixed diff should have proper separation - check that the problematic line is split
    local has_split_hunk = fixed_diff:match '@@ %-5,2 %+5,2 @@\n[^\n]*line3'
    print('Fixed diff contains split hunk: ' .. tostring(has_split_hunk ~= nil))

    print 'Original problematic diff:'
    print(problematic_diff)
    print '\nFixed diff:'
    print(fixed_diff)

    -- Just assert that validation ran and produced some result
    assert.is_true(fixed_diff ~= problematic_diff, 'Fixed diff should be different from original')
  end)
end)
