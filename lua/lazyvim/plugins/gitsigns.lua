-- replaces mhinz/vim-signify (git gutter signs + hunk nav/diff/undo)
-- and rhysd/git-messenger.vim (inline blame popup).
-- Hunk keymaps live in lua/keymaps.lua under <leader>g*.
return {
  'lewis6991/gitsigns.nvim',
  event = { 'BufReadPre', 'BufNewFile' },
  opts = {},
}
