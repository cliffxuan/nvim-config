-- Formatting, replacing ALE's fixer half (the linter half is nvim-lint).
-- Invoked via <leader>x (see lua/keymaps.lua). Mirrors the old ale_fixers.
return {
  'stevearc/conform.nvim',
  opts = {
    formatters_by_ft = {
      -- first available wins (ALE listed these as alternatives)
      python = { 'ruff_format', 'black', 'autopep8', stop_after_first = true },
      go = { 'goimports', 'gofmt' },
      terraform = { 'terraform_fmt' },
      javascript = { 'prettier' },
      css = { 'prettier' },
      typescript = { 'prettier' },
      typescriptreact = { 'prettier' },
      haskell = { 'ormolu' },
      rust = { 'rustfmt' },
      lua = { 'stylua' },
      sh = { 'shfmt' },
    },
    -- carried over from the ale_* option settings
    formatters = {
      shfmt = { prepend_args = { '-i', '2' } },
      rustfmt = { prepend_args = { '--edition', '2018' } },
      stylua = {
        prepend_args = {
          '--column-width=120',
          '--line-endings=Unix',
          '--indent-type=Spaces',
          '--indent-width=2',
          '--quote-style=AutoPreferSingle',
          '--call-parentheses=None',
        },
      },
    },
  },
}
