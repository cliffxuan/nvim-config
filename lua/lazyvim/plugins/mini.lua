return {
  'echasnovski/mini.nvim',
  config = function()
    require('mini.align').setup()
    require('mini.basics').setup()
    require('mini.bracketed').setup() -- replaces tpope/vim-unimpaired
    require('mini.cursorword').setup()
    require('mini.files').setup()
    require('mini.indentscope').setup() -- replaces Yggdroot/indentLine
    require('mini.jump').setup()
    require('mini.move').setup()
    require('mini.notify').setup()
    require('mini.pairs').setup()
    -- snippet engine; loads VSCode-format JSON from <config>/snippets/
    -- (<lang>.json or <lang>/*.json) keyed by the treesitter language of the
    -- buffer. `sh` files use the `bash` parser, so map sh -> bash.json too
    -- (covers the rare case where no bash parser is installed).
    require('mini.snippets').setup {
      snippets = {
        require('mini.snippets').gen_loader.from_lang {
          lang_patterns = { sh = { '**/bash.json' } },
        },
      },
    }
    require('mini.splitjoin').setup()
    require('mini.starter').setup()
    require('mini.tabline').setup()

    -- replaces tpope/vim-surround, mapped to keep vim-surround muscle memory
    -- (ys/ds/cs + visual S) and avoid clobbering the flash `s` motion.
    require('mini.surround').setup {
      mappings = {
        add = 'ys',
        delete = 'ds',
        replace = 'cs',
        find = '',
        find_left = '',
        highlight = '',
        update_n_lines = '',
        suffix_last = '',
        suffix_next = '',
      },
      search_method = 'cover_or_next',
    }
    vim.keymap.del('x', 'ys')
    vim.keymap.set('x', 'S', [[:<C-u>lua MiniSurround.add('visual')<CR>]], { silent = true })
    vim.keymap.set('n', 'yss', 'ys_', { remap = true })
  end,
}
