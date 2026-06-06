-- replaces voldikss/vim-floaterm. Backs:
--   - the <leader>t / <c-/> / <c-_> scratch float (lua/keymaps.lua)
--   - the :FloatRun run backend (plugin/shell_command_prefix.vim via lua/run_term.lua)
--   - the lf file manager (plugin/file_manager.vim)
return {
  'akinsho/toggleterm.nvim',
  version = '*',
  opts = {
    direction = 'float',
    float_opts = { border = 'curved' },
  },
}
