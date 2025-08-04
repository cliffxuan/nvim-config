-- vibe-coding/validation.lua
-- Consolidated validation pipeline for diff processing

local PathUtils = require 'vibe-coding.path_utils'
local Utils = require 'vibe-coding.utils'

local Validation = {}

--- Validates and fixes a diff through the complete pipeline
-- @param diff_content The raw diff content
-- @return string, table: The processed diff content and issues found
function Validation.process_diff(diff_content)
  local pipeline = {
    Validation.fix_file_paths,
    Validation.smart_validate_against_original,
    Validation.fix_context_whitespace,
    Validation.validate_and_fix_diff,
    Validation.format_diff,
  }

  local current_content = diff_content
  local all_issues = {}

  for _, step in ipairs(pipeline) do
    local result, issues = step(current_content)
    current_content = result

    -- Merge issues from this step
    if issues then
      for _, issue in ipairs(issues) do
        table.insert(all_issues, issue)
      end
    end
  end

  return current_content, all_issues
end

--- Intelligently resolves file paths in diff headers by searching the filesystem.
-- @param diff_content The diff content to fix.
-- @return string, table: The fixed diff content and a table of fixes applied.
function Validation.fix_file_paths(diff_content)
  local lines = vim.split(diff_content, '\n', { plain = true })
  local fixed_lines = {}
  local fixes = {}

  local in_hunk = false
  local line_count = 0

  for i, line in ipairs(lines) do
    local fixed_line = line
    line_count = line_count + 1

    -- Track if we're in a hunk to avoid processing hunk content as headers
    if line:match '^@@' then
      in_hunk = true
    elseif line_count <= 2 and line:match '^---%s' and not in_hunk then
      local fix
      fixed_line, fix = Validation._process_old_file_header(line, i)
      if fix then
        table.insert(fixes, fix)
      end
    elseif line_count <= 3 and line:match '^%+%+%+%s' and not in_hunk then
      local fix
      fixed_line, fix = Validation._process_new_file_header(line, i)
      if fix then
        table.insert(fixes, fix)
      end
    end

    table.insert(fixed_lines, fixed_line)
  end

  return table.concat(fixed_lines, '\n'), fixes
end

--- Processes old file header (--- line)
-- @param line The line to process
-- @param line_num The line number
-- @return string, table|nil: Fixed line and optional fix info
function Validation._process_old_file_header(line, line_num)
  local file_path = line:match '^---%s+(.*)$'
  if file_path and file_path ~= '' and file_path ~= '/dev/null' then
    file_path = PathUtils.clean_path(file_path)

    if file_path and PathUtils.looks_like_file(file_path) then
      -- Check if it's already a valid absolute path
      if file_path:match '^/' and vim.fn.filereadable(file_path) == 1 then
        -- Valid absolute path - no issue to report
        return line, nil
      end

      local resolved_path = PathUtils.resolve_file_path(file_path)
      if resolved_path and resolved_path ~= file_path then
        return '--- ' .. resolved_path,
          {
            line = line_num,
            type = 'path_resolution',
            message = string.format('Resolved file path: %s → %s', file_path, resolved_path),
          }
      elseif not resolved_path then
        return line,
          {
            line = line_num,
            type = 'path_not_found',
            message = string.format('Could not locate file: %s (you may need to fix this manually)', file_path),
            severity = 'warning',
          }
      end
    end
  end
  return line, nil
end

--- Processes new file header (+++ line)
-- @param line The line to process
-- @param line_num The line number
-- @return string, table|nil: Fixed line and optional fix info
function Validation._process_new_file_header(line, line_num)
  local file_path = line:match '^%+%+%+%s+(.*)$'
  if file_path and file_path ~= '' and file_path ~= '/dev/null' then
    if PathUtils.looks_like_file(file_path) then
      -- Check if it's already a valid absolute path
      if file_path:match '^/' and vim.fn.filereadable(file_path) == 1 then
        -- Valid absolute path - no issue to report
        return line, nil
      end

      local resolved_path = PathUtils.resolve_file_path(file_path)
      if resolved_path and resolved_path ~= file_path then
        return '+++ ' .. resolved_path,
          {
            line = line_num,
            type = 'path_resolution',
            message = string.format('Resolved file path: %s → %s', file_path, resolved_path),
          }
      elseif not resolved_path then
        return line,
          {
            line = line_num,
            type = 'path_not_found',
            message = string.format('Could not locate file: %s (you may need to fix this manually)', file_path),
            severity = 'warning',
          }
      end
    end
  end
  return line, nil
end

--- Validates and fixes common issues in diff content.
-- @param diff_content The raw diff content to validate.
-- @return string, table: The cleaned diff content and a table of issues found.
function Validation.validate_and_fix_diff(diff_content)
  local lines = vim.split(diff_content, '\n', { plain = true })
  local issues = {}
  local fixed_lines = {}

  -- Track state for validation
  local has_header = false
  local in_hunk = false

  for i, line in ipairs(lines) do
    local fixed_line = line

    -- Check for diff headers (but not when inside a hunk)
    if not in_hunk and (line:match '^---' or line:match '^%+%+%+') then
      has_header = true
      local issue
      fixed_line, issue = Validation._fix_header_line(line, i)
      if issue then
        table.insert(issues, issue)
      end
    -- Check for hunk headers
    elseif line:match '^@@' then
      in_hunk = true
      local issue
      fixed_line, issue = Validation._fix_hunk_header(line, i)
      if issue then
        table.insert(issues, issue)
      end
    -- Check context and change lines
    elseif in_hunk then
      local issue
      fixed_line, issue = Validation._fix_hunk_content_line(line, i)
      if issue then
        table.insert(issues, issue)
      end
    end

    table.insert(fixed_lines, fixed_line)
  end

  -- Final validation
  if not has_header then
    table.insert(issues, {
      line = 1,
      type = 'missing_header',
      message = 'Diff missing file headers',
      severity = 'error',
    })
  end

  return table.concat(fixed_lines, '\n'), issues
end

--- Fixes header line issues
-- @param line The line to fix
-- @param line_num The line number
-- @return string, table|nil: Fixed line and optional issue
function Validation._fix_header_line(line, line_num)
  if line:match '^---%s*$' then
    return '--- /dev/null',
      {
        line = line_num,
        type = 'header',
        message = 'Fixed empty old file header',
      }
  elseif line:match '^%+%+%+%s*$' then
    return '+++ /dev/null',
      {
        line = line_num,
        type = 'header',
        message = 'Fixed empty new file header',
      }
  end
  return line, nil
end

--- Fixes hunk header issues
-- @param line The line to fix
-- @param line_num The line number
-- @return string, table|nil: Fixed line and optional issue
function Validation._fix_hunk_header(line, line_num)
  local old_start, _, new_start, _ = line:match '^@@ %-(%d+),?(%d*) %+(%d+),?(%d*) @@'

  if not old_start then
    -- Try simpler format without counts
    old_start, new_start = line:match '^@@ %-(%d+) %+(%d+) @@'
    if old_start then
      return string.format('@@ -%s,1 +%s,1 @@', old_start, new_start),
        {
          line = line_num,
          type = 'hunk_header',
          message = 'Added missing line counts',
        }
    else
      -- Check for malformed headers
      if line:match '^@@ ' then
        return '@@ -1,1 +1,1 @@',
          {
            line = line_num,
            type = 'hunk_header',
            message = 'Normalized malformed hunk header (line numbers ignored for search-and-replace)',
            severity = 'info',
          }
      else
        return line,
          {
            line = line_num,
            type = 'hunk_header',
            message = 'Invalid hunk header format',
            severity = 'error',
          }
      end
    end
  end
  return line, nil
end

--- Fixes hunk content line issues
-- @param line The line to fix
-- @param line_num The line number
-- @return string, table|nil: Fixed line and optional issue
function Validation._fix_hunk_content_line(line, line_num)
  -- Handle empty lines - they're valid as-is
  if line == '' then
    return line, nil
  end

  local op = line:sub(1, 1)

  -- Handle valid diff operators
  if op == ' ' or op == '-' or op == '+' then
    return line, nil
  end

  -- Try to intelligently fix context lines missing the leading space
  -- This is common with LLM-generated diffs where the space prefix is omitted
  local fixed_line, issue = Validation._try_fix_missing_context_prefix(line, line_num)
  if fixed_line then
    return fixed_line, issue
  end

  -- If we can't fix it, report as invalid
  return line,
    {
      line = line_num,
      type = 'invalid_line',
      message = 'Invalid line in hunk: ' .. line,
      severity = 'warning',
    }
end

--- Attempts to fix a line that's missing the context prefix space
-- @param line The line to potentially fix
-- @param line_num The line number
-- @return string|nil, table|nil: Fixed line and issue, or nil if can't fix
function Validation._try_fix_missing_context_prefix(line, line_num)
  -- Generic approach: if line is not empty and doesn't start with +/-, treat as context
  if line ~= '' and not line:match '^[+-]' then
    return ' ' .. line,
      {
        line = line_num,
        type = 'context_fix',
        message = 'Added missing space prefix for context line: '
          .. (line:sub(1, 50) .. (line:len() > 50 and '...' or '')),
        severity = 'info',
      }
  end

  return nil, nil
end

--- Formats diff content for better readability and standards compliance.
-- @param diff_content The diff content to format.
-- @return string, table: The formatted diff content and empty issues table.
function Validation.format_diff(diff_content)
  local lines = vim.split(diff_content, '\n', { plain = true })
  local formatted_lines = {}

  local in_hunk = false
  local line_count = 0

  for _, line in ipairs(lines) do
    line_count = line_count + 1

    -- Track if we're in a hunk to avoid processing hunk content as headers
    if line:match '^@@' then
      in_hunk = true
      table.insert(formatted_lines, line)
    elseif line_count <= 2 and line:match '^---' and not in_hunk then
      table.insert(formatted_lines, Validation._format_old_header(line))
    elseif line_count <= 3 and line:match '^%+%+%+' and not in_hunk then
      table.insert(formatted_lines, Validation._format_new_header(line))
    else
      -- For all other lines, pass through as-is
      table.insert(formatted_lines, line)
    end
  end

  return table.concat(formatted_lines, '\n'), {}
end

--- Formats old file header
-- @param line The line to format
-- @return string: The formatted line
function Validation._format_old_header(line)
  local file_path = line:match '^---%s*(.*)$'
  if file_path and file_path ~= '' then
    file_path = PathUtils.clean_path(file_path)
    if file_path ~= '' then
      return '--- ' .. file_path
    end
  end
  return '--- /dev/null'
end

--- Formats new file header
-- @param line The line to format
-- @return string: The formatted line
function Validation._format_new_header(line)
  local file_path = line:match '^%+%+%+%s*(.*)$'
  if file_path and file_path ~= '' then
    return '+++ ' .. file_path
  end
  return '+++ /dev/null'
end

--- Creates a standardized error result
-- @param message The error message
-- @param severity The error severity
-- @return table: Standardized error object
function Validation.create_error(message, severity)
  return {
    success = false,
    message = message,
    severity = severity or 'error',
    timestamp = os.time(),
  }
end

--- Creates a standardized success result
-- @param message The success message
-- @param data Optional data to include
-- @return table: Standardized success object
function Validation.create_success(message, data)
  return {
    success = true,
    message = message,
    data = data,
    timestamp = os.time(),
  }
end

--- Smart validation that compares diff context with original file content
-- @param diff_content The diff content to validate
-- @return string, table: The fixed diff content and issues found
function Validation.smart_validate_against_original(diff_content)
  local lines = vim.split(diff_content, '\n', { plain = true })
  local issues = {}
  local fixed_lines = {}

  local current_file = nil
  local original_lines = nil
  local current_hunk_start = nil

  for i, line in ipairs(lines) do
    -- Track hunk headers to get hint information
    if line:match '^@@' then
      local old_start = line:match '^@@ %-(%d+)'
      current_hunk_start = old_start and tonumber(old_start) or nil
    end

    local process_result = Validation._process_line_for_smart_validation(line, i, current_file, original_lines, issues, current_hunk_start)

    -- Update current file tracking
    if process_result.current_file then
      current_file = process_result.current_file
      original_lines = process_result.original_lines
    end

    -- Add the fixed line(s) to output
    for _, fixed_line in ipairs(process_result.fixed_lines) do
      table.insert(fixed_lines, fixed_line)
    end
  end

  return table.concat(fixed_lines, '\n'), issues
end

--- Processes a single line for smart validation
-- @param line The line to process
-- @param line_num The line number
-- @param current_file Current file being tracked
-- @param original_lines Original file lines if available
-- @param issues Issues table to append to
-- @param hint_line_num Hint for expected line number in original file
-- @return table: Result with fixed_lines, current_file, original_lines
function Validation._process_line_for_smart_validation(line, line_num, current_file, original_lines, issues, hint_line_num)
  local result = {
    fixed_lines = { line },
    current_file = nil,
    original_lines = nil,
  }

  -- Suppress unused parameter warning
  _ = current_file

  -- Track current file being processed (only process actual file headers, not diff content)
  if line:match '^---' then
    local file_path = line:match '^---%s+(.*)$'
    if file_path and file_path ~= '/dev/null' and PathUtils.looks_like_file(file_path) then
      -- Clean up the file path
      file_path = PathUtils.clean_path(file_path)

      -- Resolve the file path using the intelligent search system
      local resolved_path = PathUtils.resolve_file_path(file_path)
      result.current_file = resolved_path or file_path

      -- Try to read the original file using the resolved path
      local final_path = resolved_path or file_path
      if final_path then
        local content, _ = require('vibe-coding.utils').read_file_content(final_path)
        if content then
          result.original_lines = vim.split(content, '\n', { plain = true, trimempty = false })
        else
          -- Add diagnostic information about why file couldn't be read
          -- Only report this for actual file paths, not code content
          if PathUtils.looks_like_file(final_path) then
            table.insert(issues, {
              line = line_num,
              type = 'file_access',
              message = string.format(
                'Could not read original file: %s (resolved from: %s)',
                final_path,
                file_path or 'unknown'
              ),
              severity = 'info',
            })
          end
        end
      end
    end
    return result
  end

  -- Handle hunk headers with missing line breaks
  if line:match '^@@' then
    local hunk_header, content_after = line:match '^(@@ %-?%d+,?%d* %+?%d+,?%d* @@)(.+)'
    if hunk_header and content_after and content_after ~= '' then
      -- Split the line: return hunk header and process content separately
      result.fixed_lines = { hunk_header }

      table.insert(issues, {
        line = line_num,
        type = 'hunk_header',
        message = 'Split hunk header from content (missing line break)',
        severity = 'warning',
      })

      -- Process the content after as a context line if we have original file
      if original_lines then
        local content_result = Validation._process_context_line(content_after, line_num, original_lines, issues, false, hint_line_num)
        -- Add the processed content lines
        for _, content_line in ipairs(content_result.fixed_lines) do
          table.insert(result.fixed_lines, content_line)
        end
      else
        -- No original file available, just add basic context prefix
        table.insert(result.fixed_lines, ' ' .. content_after)
        table.insert(issues, {
          line = line_num,
          type = 'context_fix',
          message = 'Added missing space prefix for context line (original file not found): ' .. content_after,
          severity = 'warning',
        })
      end

      return result
    end
  end

  -- Process context lines in hunks (both properly formatted and missing space prefix)
  if line:match '^%s' and original_lines then
    return Validation._process_context_line(line, line_num, original_lines, issues, true, hint_line_num)
  end

  -- Also check lines that should be context but are missing the space prefix
  if
    original_lines
    and not line:match '^[+@-]'
    and not line:match '^%+%+%+'
    and not line:match '^---'
    and line ~= ''
  then
    return Validation._process_context_line(line, line_num, original_lines, issues, false, hint_line_num)
  end

  return result
end

--- Processes a context line for smart validation
-- @param line The line to process
-- @param line_num The line number
-- @param original_lines Original file lines
-- @param issues Issues table to append to
-- @param has_space_prefix Whether the line already has a space prefix
-- @param hint_line_num Hint for expected line number in original file
-- @return table: Result with fixed_lines array
function Validation._process_context_line(line, line_num, original_lines, issues, has_space_prefix, hint_line_num)
  local result = { fixed_lines = {} }
  local context_text = has_space_prefix and line:sub(2) or line
  local issue = Validation._validate_context_against_original(context_text, original_lines, line_num, hint_line_num)

  if not issue then
    table.insert(result.fixed_lines, line)
    return result
  end

  table.insert(issues, issue)

  -- Handle simple whitespace mismatch - use the corrected line
  if issue.type == 'whitespace_mismatch' and issue.corrected_line then
    issue.message = issue.message .. ' (auto-corrected whitespace)'
    issue.severity = 'info'
    table.insert(result.fixed_lines, ' ' .. issue.corrected_line)
    return result
  end

  -- Handle generic joined lines formatting issue
  if issue.type == 'formatting_issue' and issue.split_lines then
    -- Replace the current line with the split lines
    issue.message = issue.message .. ' (auto-corrected: split into ' .. #issue.split_lines .. ' lines)'
    issue.severity = 'info'

    -- The split_lines from _find_prefix_match already have correct indentation from original file
    -- Just add space prefix for diff format
    for _, split_line in ipairs(issue.split_lines) do
      table.insert(result.fixed_lines, ' ' .. split_line)
    end
    return result
  end

  -- Try to fix other formatting issues
  local corrected_line = Validation._fix_context_line_formatting(context_text, original_lines)
  if corrected_line then
    local fixed_line = ' ' .. corrected_line
    issue.message = issue.message .. ' (auto-corrected)'
    issue.severity = 'info'
    table.insert(result.fixed_lines, fixed_line)
    return result
  end

  -- Final fallback
  if has_space_prefix then
    table.insert(result.fixed_lines, line)
  else
    local fixed_line = ' ' .. line
    issue.message = issue.message .. ' (added space prefix)'
    issue.severity = 'warning'
    table.insert(result.fixed_lines, fixed_line)
  end

  return result
end

--- Validates a context line against the original file content
-- @param context_text The text from the context line (without leading space)
-- @param original_lines Array of lines from the original file
-- @param line_num Line number for error reporting
-- @param hint_line_num Optional hint for the expected line number in original file
-- @return table|nil: Issue object if validation fails
function Validation._validate_context_against_original(context_text, original_lines, line_num, hint_line_num)
  -- Skip empty lines
  if context_text == '' then
    return nil
  end

  -- Check if this exact line exists in the original file
  for _, orig_line in ipairs(original_lines) do
    if orig_line == context_text then
      return nil -- Found exact match, all good
    end
  end

  -- No exact match found, this might be a formatting issue
  -- Common issues: joined lines, missing indentation, etc.

  -- First, check for content matches with different whitespace (simple indentation issues)
  local context_trimmed = context_text:gsub('^%s*(.-)%s*$', '%1')
  if context_trimmed ~= '' then
    local matches = {}
    for orig_line_num, orig_line in ipairs(original_lines) do
      local orig_trimmed = orig_line:gsub('^%s*(.-)%s*$', '%1')
      if orig_trimmed == context_trimmed then
        table.insert(matches, {line = orig_line, line_num = orig_line_num})
      end
    end

    if #matches > 0 then
      -- If we have multiple matches, prefer the one closest to the hint
      local best_match = matches[1]
      if hint_line_num and #matches > 1 then
        local best_distance = math.abs(matches[1].line_num - hint_line_num)
        for _, match in ipairs(matches) do
          local distance = math.abs(match.line_num - hint_line_num)
          if distance < best_distance then
            best_distance = distance
            best_match = match
          end
        end
      end

      -- Found content match with different whitespace - this is a simple whitespace fix
      return {
        line = line_num,
        type = 'whitespace_mismatch',
        message = 'Context line has incorrect whitespace: ' .. context_text:sub(1, 50) .. (context_text:len() > 50 and '...' or ''),
        severity = 'info',
        original_text = context_text,
        corrected_line = best_match.line,
      }
    end
  end

  -- Only check for joined lines if it's not a simple whitespace issue
  -- This handles genuine cases where multiple lines are joined together
  local prefix_match = Validation._find_prefix_match(context_text, original_lines, hint_line_num)
  if prefix_match then
    local message
    message = 'Joined line detected (prefix match): '
      .. context_text:sub(1, 60)
      .. (context_text:len() > 60 and '...' or '')

    return {
      line = line_num,
      type = 'formatting_issue',
      message = message,
      severity = 'warning',
      original_text = context_text,
      split_lines = prefix_match,
    }
  end

  -- Generic case: line not found in original
  return {
    line = line_num,
    type = 'context_mismatch',
    message = 'Context line not found in original file: '
      .. context_text:sub(1, 50)
      .. (context_text:len() > 50 and '...' or ''),
    severity = 'warning',
    original_text = context_text,
  }
end

--- Attempts to fix formatting issues in context lines
-- @param context_text The problematic context text
-- @param original_lines Array of lines from the original file
-- @return string|nil: Fixed text or nil if can't fix
function Validation._fix_context_line_formatting(context_text, original_lines)
  -- Generic approach: look for any line that contains the first few words
  local context_words = {}
  for word in context_text:gmatch '%S+' do
    table.insert(context_words, word)
  end

  if #context_words > 0 then
    -- Find lines that contain the first few words
    local search_pattern = vim.pesc(context_words[1])
    if context_words[2] then
      search_pattern = search_pattern .. '.*' .. vim.pesc(context_words[2])
    end

    for _, orig_line in ipairs(original_lines) do
      if orig_line:match(search_pattern) then
        -- Found a similar line, use it as the correction
        return orig_line
      end
    end
  end

  return nil -- Can't fix automatically
end

--- Finds if context_text is a prefix of lines in the original file and extracts the split
-- @param context_text The potentially joined line
-- @param original_lines Array of lines from the original file
-- @param hint_line_num Optional hint for preferred line number
-- @return table|nil: Array of split lines that reconstruct the context_text, or nil if no match
function Validation._find_prefix_match(context_text, original_lines, hint_line_num)
  if not original_lines or #original_lines == 0 or context_text == '' then
    return nil
  end

  -- Remove leading/trailing whitespace for comparison
  local trimmed_context = context_text:gsub('^%s+', ''):gsub('%s+$', '')

  -- Collect all possible matches first
  local all_matches = {}

  -- Try to find a sequence of original lines that when concatenated (with appropriate separators)
  -- would create the context_text
  for start_idx = 1, #original_lines do
    local matched_lines = {}

    -- Try building up the context_text from consecutive original lines
    for end_idx = start_idx, math.min(start_idx + 4, #original_lines) do -- Limit to 5 lines max
      local orig_line = original_lines[end_idx]
      local trimmed_orig = orig_line:gsub('^%s+', ''):gsub('%s+$', '')

      -- Include all lines (even empty ones) in the match sequence, but only use non-empty ones for joining
      table.insert(matched_lines, orig_line)

      -- Only process non-empty lines for matching
      if trimmed_orig ~= '' then
        -- Generic join strategy: try direct concatenation of trimmed non-empty lines
        local joined = ''
        for _, line in ipairs(matched_lines) do
          local line_trimmed = line:gsub('^%s+', ''):gsub('%s+$', '')
          if line_trimmed ~= '' then
            if joined == '' then
              joined = line_trimmed
            else
              joined = joined .. line_trimmed
            end
          end
        end

        -- Check if this matches our context text
        if joined == trimmed_context then
          table.insert(all_matches, {
            lines = vim.deepcopy(matched_lines),
            start_line = start_idx,
            end_line = end_idx
          })
        end
      end
      -- If it's an empty line, we just continue to the next iteration (no need for goto)
    end
  end

  if #all_matches == 0 then
    return nil
  end

  -- If we have multiple matches, prefer the one closest to the hint
  if hint_line_num and #all_matches > 1 then
    local best_match = all_matches[1]
    local best_distance = math.abs(all_matches[1].start_line - hint_line_num)

    for _, match in ipairs(all_matches) do
      local distance = math.abs(match.start_line - hint_line_num)
      if distance < best_distance then
        best_distance = distance
        best_match = match
      end
    end

    return best_match.lines
  end

  -- Return the first match if no hint provided
  return all_matches[1].lines
end

--- Validates if a split candidate matches lines in the original file
-- @param candidate Array of split line candidates
-- @param original_lines Array of original file lines
-- @return boolean: True if the split lines can be found in original
function Validation._validate_split_against_original(candidate, original_lines)
  -- Check if all parts of the candidate can be found in the original file
  local found_count = 0

  for _, split_line in ipairs(candidate) do
    local trimmed = split_line:gsub('^%s+', ''):gsub('%s+$', '')
    if trimmed ~= '' then
      for _, orig_line in ipairs(original_lines) do
        local orig_trimmed = orig_line:gsub('^%s+', ''):gsub('%s+$', '')
        if orig_trimmed == trimmed or orig_line:find(trimmed, 1, true) then
          found_count = found_count + 1
          break
        end
      end
    end
  end

  -- Require that at least 80% of split lines match original content
  return found_count >= math.max(1, math.floor(#candidate * 0.8))
end

--- Corrects the indentation of a split line by finding the correct version in the original file
-- @param split_line The line after splitting
-- @param original_lines Array of original file lines
-- @param is_continuation_line Whether this is a continuation line (like docstring after function)
-- @return string: Line with correct indentation
function Validation._correct_line_indentation(split_line, original_lines, is_continuation_line)
  if not original_lines then
    return split_line
  end

  local trimmed_split = split_line:gsub('^%s+', ''):gsub('%s+$', '')

  -- Look for exact content match in original to get proper indentation
  for _, orig_line in ipairs(original_lines) do
    local trimmed_orig = orig_line:gsub('^%s+', ''):gsub('%s+$', '')
    if trimmed_orig == trimmed_split then
      return orig_line -- Use original with proper indentation
    end

    -- For docstrings, also check if the content matches (ignoring whitespace)
    if trimmed_split:match '^""".*"""$' and trimmed_orig:match '^""".*"""$' then
      local split_content = trimmed_split:match '^"""(.*)"""$'
      local orig_content = trimmed_orig:match '^"""(.*)"""$'
      if split_content == orig_content then
        return orig_line -- Use original with proper indentation
      end
    end
  end

  -- If not found and it's a continuation line, try to infer proper indentation
  if is_continuation_line and trimmed_split:match '^"""' then
    -- For docstrings, typically indent 4 spaces relative to function
    return '    ' .. trimmed_split
  end

  return split_line -- Fallback to as-is
end

--- Fixes context lines in diff that have incorrect whitespace by matching original file
-- Uses sequential hunk context matching for accurate whitespace correction
-- @param diff_content The diff content to fix
-- @return string, table: Fixed diff content and issues found
function Validation.fix_context_whitespace(diff_content)
  local lines = vim.split(diff_content, '\n', { plain = true })
  local issues = {}
  local fixed_lines = {}

  local i = 1
  while i <= #lines do
    local line = lines[i]

    -- Check if this line starts a hunk
    if line:match '^@@' then
      local hunk_result = Validation._fix_hunk_whitespace(lines, i, issues)
      -- Add all fixed hunk lines
      for _, fixed_line in ipairs(hunk_result.fixed_lines) do
        table.insert(fixed_lines, fixed_line)
      end
      -- Skip to the line after the hunk
      i = hunk_result.next_line_index
    else
      -- Not a hunk, add line as-is
      table.insert(fixed_lines, line)
      i = i + 1
    end
  end

  return table.concat(fixed_lines, '\n'), issues
end

--- Fixes whitespace for an entire hunk by matching the complete context sequence
-- @param lines All diff lines
-- @param hunk_start_index Index where the hunk starts (@@)
-- @param issues Issues array to append to
-- @return table: {fixed_lines, next_line_index}
function Validation._fix_hunk_whitespace(lines, hunk_start_index, issues)
  local hunk_header = lines[hunk_start_index]
  local fixed_lines = { hunk_header }

  -- Parse the file path from preceding lines
  local current_file = nil
  for i = hunk_start_index - 1, 1, -1 do
    if lines[i]:match '^%+%+%+' then
      current_file = lines[i]:match '^%+%+%+%s+(.+)$'
      break
    end
  end

  -- Read original file
  local original_lines = nil
  if current_file then
    original_lines = Validation._read_original_file(current_file)
  end

  if not original_lines then
    -- Can't fix without original file, copy hunk as-is
    local i = hunk_start_index + 1
    while i <= #lines and not lines[i]:match '^@@' and not lines[i]:match '^---' and not lines[i]:match '^%+%+%+' do
      table.insert(fixed_lines, lines[i])
      i = i + 1
    end
    return { fixed_lines = fixed_lines, next_line_index = i }
  end

  -- Parse hunk header for line number
  local old_start = hunk_header:match '^@@ %-(%d+)'
  local hunk_start_line = old_start and tonumber(old_start) or nil

  -- Process the hunk line-by-line, handling context lines with smart reconstruction
  local i = hunk_start_index + 1
  local hunk_end_index = i

  -- Find end of hunk
  while hunk_end_index <= #lines and not lines[hunk_end_index]:match '^@@' and not lines[hunk_end_index]:match '^---' and not lines[hunk_end_index]:match '^%+%+%+' do
    hunk_end_index = hunk_end_index + 1
  end

  -- Process each line in the hunk
  while i < hunk_end_index do
    local line = lines[i]

    if line:match '^%s' then
      -- This is a context line - apply intelligent whitespace correction
      local context_text = line:sub(2) -- Remove leading space
      local corrected_line = Validation._fix_single_context_line_whitespace(context_text, original_lines, hunk_start_line)

      if corrected_line and corrected_line ~= context_text then
        local fixed_line = ' ' .. corrected_line
        table.insert(fixed_lines, fixed_line)

        if fixed_line ~= line then
          table.insert(issues, {
            line = i,
            severity = 'info',
            type = 'whitespace_fix',
            message = 'Fixed context line whitespace to match original file',
          })
        end
      else
        -- No correction needed or possible
        table.insert(fixed_lines, line)
      end
    else
      -- Not a context line (+ or -), keep as-is
      table.insert(fixed_lines, line)
    end

    i = i + 1
  end

  return { fixed_lines = fixed_lines, next_line_index = hunk_end_index }
end

--- Finds the best sequential match for a sequence of context lines
-- @param context_lines Context lines (without leading space)
-- @param original_lines Lines from original file
-- @param hint_start_line Hint for where to start searching
-- @return table|nil: Matched lines from original file, or nil if no match
function Validation._find_sequential_context_match(context_lines, original_lines, hint_start_line)
  if #context_lines == 0 then
    return nil
  end

  -- Helper function to check if a sequence matches at a given position
  local function check_sequence_match(start_pos)
    local matched = {}
    for i, context_line in ipairs(context_lines) do
      local original_line = original_lines[start_pos + i - 1]
      if not original_line then
        return nil -- Not enough lines in original file
      end

      -- Compare content (ignore whitespace differences)
      local context_trimmed = context_line:gsub('^%s*(.-)%s*$', '%1')
      local original_trimmed = original_line:gsub('^%s*(.-)%s*$', '%1')

      if context_trimmed ~= original_trimmed or context_trimmed == '' then
        return nil -- Content doesn't match
      end

      table.insert(matched, original_line)
    end
    return matched
  end

  -- Try searching near the hint first
  if hint_start_line then
    local search_start = math.max(1, hint_start_line - 5)
    local search_end = math.min(#original_lines - #context_lines + 1, hint_start_line + 10)

    for pos = search_start, search_end do
      local match = check_sequence_match(pos)
      if match then
        return match
      end
    end
  end

  -- Fallback: search entire file
  for pos = 1, #original_lines - #context_lines + 1 do
    local match = check_sequence_match(pos)
    if match then
      return match
    end
  end

  return nil -- No match found
end

--- Enhanced sequential context matching with flexible block structure handling
-- @param context_lines Context lines (without leading space)
-- @param original_lines Lines from original file
-- @param hint_start_line Hint for where to start searching
-- @return table|nil: Matched lines from original file, or nil if no match
function Validation._find_enhanced_sequential_context_match(context_lines, original_lines, hint_start_line)
  if #context_lines == 0 then
    return nil
  end

  -- Preprocess context lines to handle structural issues (like joined lines)
  local preprocessed_context = Validation._preprocess_context_for_matching(context_lines)

  -- Helper function for flexible sequence matching that handles structural differences
  local function check_flexible_block_match(start_pos)
    local matched = {}
    local original_pos = start_pos
    local context_pos = 1
    local match_tolerance = 2  -- Allow skipping up to 2 lines
    local skipped_lines = 0

    while context_pos <= #preprocessed_context and original_pos <= #original_lines do
      local context_line = preprocessed_context[context_pos]
      local original_line = original_lines[original_pos]

      -- Compare content (ignore whitespace differences)
      local context_trimmed = context_line:gsub('^%s*(.-)%s*$', '%1')
      local original_trimmed = original_line:gsub('^%s*(.-)%s*$', '%1')

      if context_trimmed == original_trimmed and context_trimmed ~= '' then
        -- Found a match - use the original line's whitespace
        table.insert(matched, original_line)
        context_pos = context_pos + 1
        original_pos = original_pos + 1
        skipped_lines = 0  -- Reset skip counter on successful match
      elseif context_trimmed == '' then
        -- Handle empty context line
        if original_trimmed == '' then
          table.insert(matched, original_line)
          context_pos = context_pos + 1
          original_pos = original_pos + 1
        else
          -- Add a blank line if context expects it but original doesn't have it at this position
          table.insert(matched, '')
          context_pos = context_pos + 1
        end
      else
        -- No match - try the next original line if we haven't exceeded tolerance
        if skipped_lines < match_tolerance then
          original_pos = original_pos + 1
          skipped_lines = skipped_lines + 1
        else
          return nil  -- Too many mismatches
        end
      end

      -- Safety check to prevent infinite loops
      if original_pos > start_pos + #preprocessed_context + match_tolerance + 3 then
        return nil
      end
    end

    -- Check if we successfully matched all context lines
    if context_pos > #preprocessed_context then
      return matched
    end

    return nil
  end

  -- Try enhanced matching near the hint first
  if hint_start_line then
    local search_start = math.max(1, hint_start_line - 3)
    local search_end = math.min(#original_lines - #preprocessed_context, hint_start_line + 15)

    for pos = search_start, search_end do
      local match = check_flexible_block_match(pos)
      if match and #match > 0 then
        return match
      end
    end
  end

  -- Fallback: try flexible matching on the entire file
  for pos = 1, math.max(1, #original_lines - #preprocessed_context - 5) do
    local match = check_flexible_block_match(pos)
    if match and #match > 0 then
      return match
    end
  end

  return nil -- No match found
end

--- Preprocesses context lines to handle joined lines and structural issues
-- @param context_lines Raw context lines from diff
-- @return table: Preprocessed and cleaned context lines
function Validation._preprocess_context_for_matching(context_lines)
  local preprocessed = {}

  for _, line in ipairs(context_lines) do
    -- Detect and split joined lines
    local processed_line = line

    -- Pattern 1: ")except" or ")else" etc. - closing paren immediately followed by keyword
    local before_paren, after_keyword = processed_line:match('^(%s*%))([a-zA-Z].*)')
    if before_paren and after_keyword then
      -- Split into two lines: `)` and the keyword line
      table.insert(preprocessed, before_paren)
      -- Determine appropriate indentation for the keyword line
      local base_indent = before_paren:match('^%s*') or ''
      -- For 'except', it should be outdented relative to the closing paren
      local keyword_indent = #base_indent >= 4 and base_indent:sub(1, -5) or ''
      table.insert(preprocessed, keyword_indent .. after_keyword)
    else
      -- No joined line detected, keep as-is
      table.insert(preprocessed, processed_line)
    end
  end

  return preprocessed
end

--- Fixes a single context line's whitespace to match the original file
-- @param context_text The context line text (without leading space)
-- @param original_lines The original file lines
-- @param hint_start_line Hint for where to start searching
-- @return string|nil The corrected line text, or nil if no correction needed/possible
function Validation._fix_single_context_line_whitespace(context_text, original_lines, hint_start_line)
  if not original_lines or #original_lines == 0 then
    return nil
  end

  -- First, check for exact matches (this handles most cases)
  for _, orig_line in ipairs(original_lines) do
    if orig_line == context_text then
      return nil -- Already correct, no change needed
    end
  end

  -- Check for matches where only whitespace differs
  local context_trimmed = context_text:gsub('^%s*(.-)%s*$', '%1')
  if context_trimmed == '' then
    -- Handle empty lines - find the closest empty line in original
    local best_empty_line = nil
    local best_distance = math.huge

    for line_num, orig_line in ipairs(original_lines) do
      if orig_line:gsub('^%s*(.-)%s*$', '%1') == '' then
        local distance = hint_start_line and math.abs(line_num - hint_start_line) or 0
        if distance < best_distance then
          best_distance = distance
          best_empty_line = orig_line
        end
      end
    end

    return best_empty_line or '' -- Use closest empty line or fallback
  end

  -- Look for lines with matching content but different whitespace, prioritizing those near hint
  local matches = {}
  for line_num, orig_line in ipairs(original_lines) do
    local orig_trimmed = orig_line:gsub('^%s*(.-)%s*$', '%1')
    if orig_trimmed == context_trimmed then
      table.insert(matches, {line = orig_line, line_num = line_num})
    end
  end

  if #matches > 0 then
    -- If we have a hint, prefer the match closest to the hint
    if hint_start_line then
      table.sort(matches, function(a, b)
        return math.abs(a.line_num - hint_start_line) < math.abs(b.line_num - hint_start_line)
      end)
    end
    return matches[1].line
  end

  -- Try fuzzy matching for cases where content might be slightly different
  -- but represents the same logical line
  local best_match = nil
  local best_score = 0
  local best_distance = math.huge

  for line_num, orig_line in ipairs(original_lines) do
    local score = Validation._calculate_line_similarity(context_text, orig_line)
    if score > 0.8 then
      local distance = hint_start_line and math.abs(line_num - hint_start_line) or 0
      -- Prefer closer matches when scores are similar
      if score > best_score or (score == best_score and distance < best_distance) then
        best_match = orig_line
        best_score = score
        best_distance = distance
      end
    end
  end

  if best_match then
    return best_match
  end

  -- No correction possible
  return nil
end

--- Calculates similarity between two lines (0.0 to 1.0)
-- @param line1 First line
-- @param line2 Second line
-- @return number Similarity score
function Validation._calculate_line_similarity(line1, line2)
  -- Simple similarity based on trimmed content
  local trimmed1 = line1:gsub('^%s*(.-)%s*$', '%1')
  local trimmed2 = line2:gsub('^%s*(.-)%s*$', '%1')

  if trimmed1 == trimmed2 then
    return 1.0
  end

  if trimmed1 == '' or trimmed2 == '' then
    return 0.0
  end

  -- Check if one is a substring of the other
  if trimmed1:find(trimmed2, 1, true) or trimmed2:find(trimmed1, 1, true) then
    return 0.9
  end

  -- Simple word-based comparison
  local words1 = {}
  for word in trimmed1:gmatch('%S+') do
    words1[word] = true
  end

  local words2 = {}
  local common_words = 0
  for word in trimmed2:gmatch('%S+') do
    if words1[word] then
      common_words = common_words + 1
    end
    words2[word] = true
  end

  local total_words = 0
  for _ in pairs(words1) do total_words = total_words + 1 end
  for _ in pairs(words2) do total_words = total_words + 1 end

  if total_words == 0 then
    return 0.0
  end

  return (2.0 * common_words) / total_words
end

--- Helper function to read original file lines
-- @param file_path Path to the original file
-- @return table|nil: Lines from the original file, or nil if can't read
function Validation._read_original_file(file_path)
  local lines, err = Utils.read_file(file_path)
  if not lines then
    return nil
  end
  return lines
end

return Validation
