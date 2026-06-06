-- Completion (blink.cmp) + snippet engine (LuaSnip), replacing nvim-cmp +
-- UltiSnips. Snippets live as VSCode-format JSON under <config>/snippets and
-- are loaded by LuaSnip; LSP capabilities are advertised in lua/config.lua.
return {
  {
    'L3MON4D3/LuaSnip',
    version = 'v2.*',
    config = function()
      require('luasnip.loaders.from_vscode').lazy_load {
        paths = { vim.fn.stdpath 'config' .. '/snippets' },
      }
    end,
  },
  {
    'saghen/blink.cmp',
    version = '1.*',
    dependencies = { 'L3MON4D3/LuaSnip' },
    opts = {
      snippets = { preset = 'luasnip' },
      sources = {
        default = { 'lsp', 'path', 'snippets', 'buffer' },
      },
      -- Keymaps carried over from the old nvim-cmp setup. Tab/S-Tab also jump
      -- snippet tabstops now (UltiSnips used its own trigger before).
      keymap = {
        preset = 'none',
        ['<C-k>'] = { 'select_prev', 'fallback' },
        ['<C-j>'] = { 'select_next', 'fallback' },
        ['<C-p>'] = { 'select_prev', 'fallback' },
        ['<C-n>'] = { 'select_next', 'fallback' },
        ['<Tab>'] = { 'select_next', 'snippet_forward', 'fallback' },
        ['<S-Tab>'] = { 'select_prev', 'snippet_backward', 'fallback' },
        ['<C-[>'] = { 'scroll_documentation_up', 'fallback' },
        ['<C-f>'] = { 'scroll_documentation_up', 'fallback' },
        ['<C-]>'] = { 'scroll_documentation_down', 'fallback' },
        ['<C-d>'] = { 'scroll_documentation_down', 'fallback' },
        ['<C-space>'] = { 'show', 'fallback' },
        ['<C-e>'] = { 'hide', 'fallback' },
        ['<CR>'] = { 'accept', 'fallback' },
      },
      completion = {
        documentation = { auto_show = true },
        menu = { border = 'rounded' },
      },
    },
    opts_extend = { 'sources.default' },
  },
}
