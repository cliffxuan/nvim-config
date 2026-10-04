local utils = require 'utils'

if utils.is_wsl() then
  vim.g.clipboard = {
    name = 'win32yank-wsl',
    copy = {
      ['+'] = 'win32yank.exe -i --crlf',
      ['*'] = 'win32yank.exe -i --crlf',
    },
    paste = {
      ['+'] = 'win32yank.exe -o --lf',
      ['*'] = 'win32yank.exe -o --lf',
    },
    cache_enabled = 0,
  }
else
  -- Terminal / Herdr / SSH / OSC 52 clipboard provider
  local cache = { lines = {}, regtype = 'v' }

  local function copy(lines, regtype)
    cache.lines = lines
    cache.regtype = regtype or 'v'

    local s = table.concat(lines, '\n')
    local b64 = vim.base64.encode(s)
    local seq = string.format('\027]52;c;%s\007', b64)

    -- Wrap in tmux DCS sequence if running in plain tmux without Herdr
    if vim.env.TMUX and not vim.env.HERDR_ENV then
      seq = string.format('\027Ptmux;\027\027]52;c;%s\007\027\\', b64)
    end

    -- Send OSC 52 escape sequence directly to terminal emulator/multiplexer
    local ok = pcall(vim.api.nvim_ui_send, seq)
    if not ok then
      pcall(function()
        io.stdout:write(seq)
        io.stdout:flush()
      end)
    end
  end

  local function paste()
    if #cache.lines > 0 then
      return { cache.lines, cache.regtype }
    end
    local reg = vim.fn.getreg '"'
    return { vim.fn.split(reg, '\n'), vim.fn.getregtype '"' }
  end

  vim.g.clipboard = {
    name = 'OSC 52',
    copy = {
      ['+'] = copy,
      ['*'] = copy,
    },
    paste = {
      ['+'] = paste,
      ['*'] = paste,
    },
    cache_enabled = 0,
  }
end

