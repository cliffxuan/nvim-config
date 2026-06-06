-- Run shell commands in a toggleterm floating terminal.
-- Backs the :FloatRun command used by g:ShellCommandPrefix() (see
-- plugin/shell_command_prefix.vim) and the lf file manager.
local M = {}

-- Run `cmd` in a fresh floating terminal.
--   opts.close_on_exit: close the float when the command finishes
--     (default false, so command output stays visible after it exits).
function M.run(cmd, opts)
  opts = opts or {}
  local close_on_exit = opts.close_on_exit or false
  local Terminal = require('toggleterm.terminal').Terminal
  local term = Terminal:new {
    cmd = cmd,
    direction = 'float',
    close_on_exit = close_on_exit,
    on_open = function()
      vim.cmd 'startinsert'
    end,
  }
  term:toggle()
end

return M
