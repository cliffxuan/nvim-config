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
  {
    'stevearc/aerial.nvim',
    cmd = { 'AerialToggle' },
    dependencies = { 'nvim-treesitter/nvim-treesitter', 'nvim-tree/nvim-web-devicons' },
    opts = {},
  },
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
  'haishanh/night-owl.vim',
  'arcticicestudio/nord-vim',
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
  { 'jeetsukumaran/vim-pythonsense', ft = 'python' },
  { 'mrcjkb/rustaceanvim', ft = 'rust' },
  {
    'MeanderingProgrammer/render-markdown.nvim',
    dependencies = { 'nvim-treesitter/nvim-treesitter', 'nvim-mini/mini.nvim' },
    ft = 'markdown',
    opts = {},
  },
}
