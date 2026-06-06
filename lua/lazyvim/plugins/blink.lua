-- Completion engine (blink.cmp), replacing nvim-cmp. Snippets are handled by
-- mini.snippets (configured in mini.lua); LSP capabilities are advertised in
-- lua/config.lua.
return {
  {
    'saghen/blink.cmp',
    version = '1.*',
    opts = {
      snippets = { preset = 'mini_snippets' },
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
