return {
  {
    'nvim-treesitter/nvim-treesitter',
    build = ':TSUpdate',
    config = function()
      -- Compat shim for Neovim 0.12 + the frozen nvim-treesitter `master`.
      -- master's query directive/predicate handlers expect `match[id]` to be a
      -- single TSNode, relying on the `all = false` option to iter_matches.
      -- Neovim 0.12 removed `all = false`, so `match[id]` is now always a list
      -- of nodes; the handlers then pass a table to get_node_text/node:range()
      -- and crash (e.g. set-lang-from-info-string! via render-markdown on
      -- markdown code fences). Re-register master's handlers through a wrapper
      -- that collapses the list back to a single node (its last element, which
      -- matches the old single-node semantics), then restore the originals so
      -- other plugins' 0.12-correct directives are left untouched.
      -- Remove this once nvim-treesitter is migrated to the `main` branch.
      do
        local query = require('vim.treesitter.query')
        local orig_directive = query.add_directive
        local orig_predicate = query.add_predicate
        local function collapse(match)
          local out = {}
          for id, node in pairs(match) do
            -- a TSNode is userdata; a 0.12 capture list is a plain table
            out[id] = type(node) == 'table' and node[#node] or node
          end
          return out
        end
        local function wrap(register)
          return function(name, handler, opts)
            return register(name, function(match, ...)
              return handler(collapse(match), ...)
            end, opts)
          end
        end
        query.add_directive = wrap(orig_directive)
        query.add_predicate = wrap(orig_predicate)
        package.loaded['nvim-treesitter.query_predicates'] = nil
        require('nvim-treesitter.query_predicates') -- re-register via wrappers
        query.add_directive = orig_directive
        query.add_predicate = orig_predicate
      end

      require('nvim-treesitter.install').prefer_git = true
      require('nvim-treesitter.configs').setup {
        ensure_installed = {
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
        },
        auto_install = true,
        sync_install = false,
        highlight = {
          enable = true,
          additional_vim_regex_highlighting = false,
        },
        indent = { enable = true },
        modules = {},
        -- experimental and ships no highlight queries; let native htmldjango
        -- syntax handle jinja files instead (see vim.filetype.add in init.lua)
        ignore_install = { 'htmldjango' },
        fold = {
          enable = true,
        },
      }
      -- Set the foldmethod to use Treesitter. Use Neovim's native (C-backed)
      -- foldexpr, not nvim-treesitter's legacy VimScript `nvim_treesitter#foldexpr()`
      -- which is O(n^2) per redraw and freezes the UI on large/nested files
      -- (e.g. opening a big JSON via Telescope, which forces several redraws).
      vim.opt.foldmethod = 'expr'
      vim.opt.foldexpr = 'v:lua.vim.treesitter.foldexpr()'
      -- Start with folds closed
      vim.opt.foldlevel = 0
    end,
  },
}
