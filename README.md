# nvim-config

cliffxuan's Neovim configuration. This README covers the layout and the
plugin stack so you can find your way around quickly.

## Setup

- **Neovim** with **lazy.nvim** (bootstrapped in `init.lua`). Developed on 0.12;
  the only version-sensitive piece is aerial.nvim, whose default branch needs
  0.12+ — on older Neovim, pin its `nvim-0.11` branch.
- The repo is **symlinked to `~/.config/nvim`**, so `vim.fn.stdpath('config')`
  resolves to this directory.
- Leader = `<Space>`, localleader = `,`.

## Layout

```
init.lua                     -- options, lazy bootstrap, vimscript funcs, requires
lua/
  config.lua                 -- LSP setup (vim.lsp.config/enable), diagnostics, blink caps
  keymaps.lua                -- all keymaps (central), which-key groups
  utils.lua                  -- helpers (project root, visual selection, api key, ...)
  clipboard.lua              -- clipboard integration
  commit_msg.lua             -- AI commit message (OpenAI direct, :GptGitCommitMsg)
  lazyvim/plugins/           -- plugin specs (auto-imported by lazy)
    init.lua                 -- the bulk of the plugin list + simple specs
    blink.lua mini.lua telescope.lua conform.lua nvim-lint.lua
    gitsigns.lua toggleterm.lua codecompanion.lua nvim-treesitter.lua
    lualine.lua lean.lua
snippets/                    -- VSCode-format JSON snippets (loaded by mini.snippets)
after/ftplugin/*/run.vim     -- per-language run/test commands (<leader>r, <localleader>*)
plugin/                      -- small vimscript helpers (run, file manager, etc.)
```

## Plugin stack

| Concern | Plugin |
|---|---|
| Completion | **blink.cmp** (`blink.lua`) |
| Snippets | **mini.snippets** — VSCode JSON in `snippets/`, keyed by treesitter lang (note: `sh` → `bash.json`) |
| LSP | native `vim.lsp` + `nvim-lspconfig` + `mason.nvim`; rust via **rustaceanvim**, haskell via **haskell-tools** |
| Formatting | **conform.nvim** (`<leader>x`) |
| Linting | **nvim-lint** (ruff for python; rust/haskell come from their LSPs) |
| Syntax/indent/folds | **nvim-treesitter** |
| Fuzzy finder | **Telescope** (+ telescope-fzf-native) |
| Git signs / hunks / blame | **gitsigns.nvim** |
| Git commands | **vim-fugitive** (`:Git`, `:Gread`, `:Gwrite`, `:Gvdiff`) |
| Terminal + run backend | **toggleterm.nvim** (backs `<leader>r` via `g:ShellCommandPrefix` → `:FloatRun`) |
| Motions (jump) | **flash.nvim** (`s`); `f`/`t` enhanced by `mini.jump` |
| Symbol outline | **aerial.nvim** (`<leader>uo`) |
| AI | **CodeCompanion** (`<leader>a*`); commit msgs via `commit_msg.lua` |
| Many small utils | **mini.nvim** — surround, bracketed, indentscope, jump, pairs, move, align, files, notify, tabline, starter, cursorword, splitjoin |
| Misc | which-key, trouble, render-markdown, vim-abolish/endwise/repeat, vim-pythonsense |

## Key bindings

Leader groups (discover with which-key — press `<leader>` and wait):

- `<leader>a` — **AI** (CodeCompanion: `aa` chat, `ai` inline, `ap` actions)
- `<leader>g` — **Git** (fugitive + gitsigns + telescope; `gC` = AI commit message)
- `<leader>j` — **Text Search** (telescope grep variants)
- `<leader>n` — **Open file** (telescope find_files in various dirs)
- `<leader>k` — **Change directory** (zoxide / project root pickers)
- `<leader>h` — **History** (oldfiles, search/command history)
- `<leader>e` — **Edit** (open config files, source, new splits)
- `<leader>u` — **Display settings** (colorscheme cycle, number/list toggles, inlay hints)

Other notables: `<leader>f` find files, `<leader>b` buffers, `<leader>t` float terminal,
`<leader>r` run current buffer, `<leader>x` format, `<leader>s` insert snippet,
`s` flash jump, `-` mini.files, `<C-j>`/`<C-k>` next/prev diagnostic.

## Conventions / gotchas

- **Keymaps live in `lua/keymaps.lua`** — add new ones there, not scattered.
- **Snippets**: drop a `<lang>.json` (or `<lang>/*.json`) under `snippets/` in
  VSCode format; mini.snippets keys by *treesitter* language name.
- **Run system**: `<leader>r` and the `<localleader>` test maps in
  `after/ftplugin/*/run.vim` shell out via `g:ShellCommandPrefix()`, which returns
  `:FloatRun` (a toggleterm float). Don't reintroduce floaterm.
- **AI commit message**: `<leader>gC` (or `:GptGitCommitMsg`) calls the OpenAI API
  directly from `lua/commit_msg.lua`; key comes from `~/.config/openai_api_key`.
- Verify config changes headless: `nvim --headless -u init.lua "+lua print('ok')" +qa`.
