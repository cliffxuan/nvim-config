-- Mappings.
-- See `:help vim.diagnostic.*` for documentation on any of the below functions
-- local opts = { noremap = true, silent = true }
-- vim.keymap.set('n', '<space>e', vim.diagnostic.open_float, opts)
-- vim.keymap.set('n', '[d', vim.diagnostic.goto_prev, opts)
-- vim.keymap.set('n', ']d', vim.diagnostic.goto_next, opts)
-- vim.keymap.set('n', '<leader>i', vim.diagnostic.setloclist, opts)

-- LSP Keymaps on LspAttach
vim.api.nvim_create_autocmd('LspAttach', {
  group = vim.api.nvim_create_augroup('UserLspConfig', { clear = true }),
  callback = function(args)
    local bufnr = args.buf
    vim.bo[bufnr].omnifunc = 'v:lua.vim.lsp.omnifunc'

    local bufopts = { noremap = true, silent = true, buffer = bufnr }
    vim.keymap.set('n', 'gD', vim.lsp.buf.declaration, bufopts)
    vim.keymap.set('n', 'gd', vim.lsp.buf.definition, bufopts)
    vim.keymap.set('n', '<C-space>', vim.lsp.buf.hover, bufopts)
    vim.keymap.set('n', '<leader>i', vim.lsp.buf.hover, bufopts)
    vim.keymap.set('n', 'gi', vim.lsp.buf.implementation, bufopts)
    vim.keymap.set('n', 'gr', vim.lsp.buf.references, bufopts)
  end,
})

-- Shared config applied to every server (merged with per-server config below):
-- blink.cmp's completion capabilities.
local star_config = {
  flags = {
    debounce_text_changes = 150,
  },
}
local ok_blink, blink = pcall(require, 'blink.cmp')
if ok_blink then
  star_config.capabilities = blink.get_lsp_capabilities()
end
vim.lsp.config('*', star_config)

-- lua_ls needs extra settings on top of the shared '*' config.
vim.lsp.config('lua_ls', {
  settings = {
    Lua = {
      completion = {
        callSnippet = 'Replace',
      },
      runtime = {
        version = 'LuaJIT',
      },
    },
  },
})

-- Enable language servers
vim.lsp.enable 'pyright'
vim.lsp.enable 'vtsls'
vim.lsp.enable 'lua_ls'
vim.lsp.enable 'gopls'

-- Diagnostic gutter signs (the 🚫/⚡ emojis carried over from ALE).
vim.diagnostic.config {
  signs = {
    text = {
      [vim.diagnostic.severity.ERROR] = '🚫',
      [vim.diagnostic.severity.WARN] = '⚡',
      [vim.diagnostic.severity.INFO] = 'ℹ',
      [vim.diagnostic.severity.HINT] = '💡',
    },
  },
}

-- vim: ts=2 sts=2 sw=2 et
