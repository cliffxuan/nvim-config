-- Generate a Conventional Commits message from the staged diff by calling the
-- OpenAI API directly (no plugin), and insert it at the top of the current
-- buffer (intended for the fugitive `:Git commit` buffer).
local utils = require 'utils'

local M = {}

local MODEL = 'gpt-4o'

local function api_key()
  utils.set_open_api_key() -- populates vim.env.OPENAI_API_KEY from ~/.config/openai_api_key
  return vim.env.OPENAI_API_KEY or os.getenv 'OPENAI_API_KEY'
end

local PROMPT = 'You are an expert at following the Conventional Commit specification. '
  .. 'Generate ONLY a commit message (a concise subject line under 72 chars, then a '
  .. 'blank line and a body if useful) for the staged diff below. Do not wrap it in '
  .. 'code fences or add any commentary.\n\n'

function M.generate()
  local diff = vim.fn.system 'git diff --staged --no-color'
  if vim.v.shell_error ~= 0 then
    return vim.notify('git diff failed:\n' .. diff, vim.log.levels.ERROR)
  end
  if vim.trim(diff) == '' then
    return vim.notify('No staged changes to summarize', vim.log.levels.WARN)
  end
  local key = api_key()
  if not key or key == '' then
    return vim.notify('OPENAI_API_KEY not set (and ~/.config/openai_api_key missing)', vim.log.levels.ERROR)
  end

  local body = vim.json.encode {
    model = MODEL,
    temperature = 0.2,
    messages = { { role = 'user', content = PROMPT .. diff } },
  }

  local bufnr = vim.api.nvim_get_current_buf()
  vim.notify('Generating commit message…', vim.log.levels.INFO)

  vim.system({
    'curl',
    '-sS',
    'https://api.openai.com/v1/chat/completions',
    '-H',
    'Content-Type: application/json',
    '-H',
    'Authorization: Bearer ' .. key,
    '--data',
    '@-', -- read JSON body from stdin (avoids ARG_MAX / escaping issues)
  }, { text = true, stdin = body }, function(out)
    vim.schedule(function()
      if out.code ~= 0 then
        return vim.notify('curl failed:\n' .. (out.stderr or ''), vim.log.levels.ERROR)
      end
      local ok, decoded = pcall(vim.json.decode, out.stdout)
      if not ok then
        return vim.notify('Could not parse OpenAI response:\n' .. out.stdout, vim.log.levels.ERROR)
      end
      if decoded.error then
        return vim.notify('OpenAI error: ' .. (decoded.error.message or vim.inspect(decoded.error)), vim.log.levels.ERROR)
      end
      local msg = vim.tbl_get(decoded, 'choices', 1, 'message', 'content')
      if not msg then
        return vim.notify('No message in OpenAI response', vim.log.levels.ERROR)
      end
      if not vim.api.nvim_buf_is_valid(bufnr) then
        return
      end
      vim.api.nvim_buf_set_lines(bufnr, 0, 0, false, vim.split(vim.trim(msg), '\n', { plain = true }))
    end)
  end)
end

vim.api.nvim_create_user_command('GptGitCommitMsg', M.generate, { desc = 'Generate commit message from staged diff' })

return M
