return {
  {
    'folke/flash.nvim',
    lazy = true,
    -- only the explicit `s` jump; leave f/t to mini.jump and `/` search alone.
    opts = {
      modes = {
        search = { enabled = false },
        char = { enabled = false },
      },
    },
  },
  'majutsushi/tagbar',
  'tpope/vim-abolish',
  'tpope/vim-endwise',
  'tpope/vim-fugitive',
  'tpope/vim-repeat',
  'kana/vim-textobj-user',
  {
    'folke/trouble.nvim',
    opts = {},
    cmd = 'Trouble',
  },
  {
    'folke/which-key.nvim',
    event = 'VeryLazy',
    init = function()
      vim.o.timeout = true
      vim.o.timeoutlen = 1000
    end,
    opts = {},
    config = function()
      require('which-key.plugins.presets').operators['y'] = nil
    end,
  },
  {
    'williamboman/mason.nvim',
    config = function()
      require('mason').setup()
    end,
  },
  -- theme
  'morhetz/gruvbox',
  'sickill/vim-monokai',
  'dracula/vim',
  'jnurmine/Zenburn',
  'haishanh/night-owl.vim',
  'ayu-theme/ayu-vim',
  'arcticicestudio/nord-vim',
  'junegunn/seoul256.vim',
  'altercation/vim-colors-solarized',
  'yuttie/hydrangea-vim',
  'NLKNguyen/papercolor-theme',
  {
    'folke/tokyonight.nvim',
    lazy = false,
    priority = 1000,
    opts = {},
    config = function()
      ---@diagnostic disable-next-line: missing-fields
      require('tokyonight').setup {
        style = 'night',
        styles = {
          functions = {},
        },
        on_colors = function(colors)
          ---@diagnostic disable-next-line: inject-field
          colors.terminal_black = '#727169'
        end,
      }
    end,
  },
  'rebelot/kanagawa.nvim',
  'catppuccin/nvim',

  'neovim/nvim-lspconfig',
  'nvim-lua/plenary.nvim',
  -- language specific
  -- haskell-tools manages its own lazy-loading; don't ft-gate or call setup()
  { 'mrcjkb/haskell-tools.nvim', lazy = false },
  { 'hashivim/vim-terraform', ft = 'terraform' },
  { 'OmniSharp/omnisharp-vim', ft = 'cs' },
  { 'Glench/Vim-Jinja2-Syntax', ft = 'jinja' },
  { 'jeetsukumaran/vim-pythonsense', ft = 'python' },
  { 'mrcjkb/rustaceanvim', ft = 'rust' },
  {
    'MeanderingProgrammer/render-markdown.nvim',
    dependencies = { 'nvim-treesitter/nvim-treesitter', 'echasnovski/mini.nvim' },
    ft = 'markdown',
    opts = {},
  },
}
