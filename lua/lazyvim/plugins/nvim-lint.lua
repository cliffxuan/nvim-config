-- Linting, replacing ALE's linter half (formatting is conform.nvim).
-- Mirrors the old ale_linters. Rust/Haskell diagnostics come from their LSPs
-- (rustaceanvim / haskell-tools), so only the standalone CLI linters live here.
return {
  'mfussenegger/nvim-lint',
  event = { 'BufReadPost', 'BufWritePost', 'InsertLeave' },
  config = function()
    require('lint').linters_by_ft = {
      python = { 'ruff' },
    }
    vim.api.nvim_create_autocmd({ 'BufReadPost', 'BufWritePost', 'InsertLeave' }, {
      group = vim.api.nvim_create_augroup('nvim_lint', { clear = true }),
      callback = function()
        require('lint').try_lint()
      end,
    })
  end,
}
