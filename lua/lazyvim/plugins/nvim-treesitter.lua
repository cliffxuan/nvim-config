-- nvim-treesitter `main` (the rewrite for Neovim 0.12+). It only installs
-- parsers/queries; highlighting, folds and indent are switched on per buffer
-- below. Needs the `tree-sitter` CLI (>= 0.26.1) and a C compiler on PATH.
local ensure_installed = {
  'c',
  'lua',
  'vim',
  'vimdoc',
  'query',
  'javascript',
  'typescript',
  'tsx',
  'html',
  'css',
  'rust',
  'python',
  'go',
  'json',
  'yaml',
  'toml',
  'bash',
  'markdown',
  'markdown_inline',
  'dockerfile',
}

-- experimental and ships no highlight queries; let native htmldjango syntax
-- handle jinja files instead (see vim.filetype.add in init.lua)
local ignore_install = { htmldjango = true }

return {
  {
    'nvim-treesitter/nvim-treesitter',
    branch = 'main',
    lazy = false, -- main does not support lazy-loading
    build = ':TSUpdate',
    config = function()
      local ts = require 'nvim-treesitter'
      ts.setup()
      ts.install(ensure_installed) -- async; no-op when already installed

      local available = {}
      for _, lang in ipairs(ts.get_available()) do
        available[lang] = true
      end

      local function enable(buf, lang)
        if not vim.api.nvim_buf_is_valid(buf) or not pcall(vim.treesitter.start, buf, lang) then
          return
        end
        -- experimental in main, but matches the old `indent = { enable = true }`
        vim.bo[buf].indentexpr = "v:lua.require'nvim-treesitter'.indentexpr()"
      end

      vim.api.nvim_create_autocmd('FileType', {
        group = vim.api.nvim_create_augroup('treesitter_start', { clear = true }),
        callback = function(args)
          local buf = args.buf
          local lang = vim.treesitter.language.get_lang(args.match)
          if not lang or ignore_install[lang] then
            return
          end
          if vim.treesitter.language.add(lang) then
            enable(buf, lang)
          elseif available[lang] then
            -- replaces master's `auto_install = true`
            ts.install({ lang }):await(function()
              vim.schedule(function()
                enable(buf, lang)
              end)
            end)
          end
        end,
      })

      -- Treesitter folds via Neovim's native (C-backed) foldexpr, not
      -- nvim-treesitter's legacy VimScript foldexpr (O(n^2) per redraw).
      vim.opt.foldmethod = 'expr'
      vim.opt.foldexpr = 'v:lua.vim.treesitter.foldexpr()'
      -- Start with folds closed
      vim.opt.foldlevel = 0
    end,
  },
}
