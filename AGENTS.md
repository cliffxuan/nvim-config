# AGENTS.md

Operational notes for working on this Neovim config. See `README.md` for the
human-facing plugin map; this file is about *how to make changes safely here*.

## Environment

- **Don't assume a Neovim version** — this config runs on multiple machines. Check
  the actual one with `nvim --version` before relying on version-gated behavior.
- The only version-sensitive plugin is **aerial.nvim**: its default branch needs
  Neovim 0.12+. On older Neovim (< 0.12), pin `branch = 'nvim-0.11'` in the aerial
  spec (`lua/lazyvim/plugins/init.lua`).
- lazy.nvim manager. The repo is **symlinked to `~/.config/nvim`**, so
  `vim.fn.stdpath('config')` == this repo. Running `nvim -u init.lua` from the repo
  still uses the symlinked config dir for `stdpath`.
- `lazy-lock.json` is gitignored.

## Where things go

- **Plugin specs**: `lua/lazyvim/plugins/*.lua`, auto-imported via
  `require('lazy').setup 'lazyvim/plugins'`. Simple one-liners go in
  `lua/lazyvim/plugins/init.lua`; anything needing config gets its own file.
- **Keymaps**: ALL in `lua/keymaps.lua` (don't scatter them). which-key groups are
  declared near the top via `require('which-key').add`.
- **LSP / diagnostics / completion caps**: `lua/config.lua` (uses the 0.11
  `vim.lsp.config` / `vim.lsp.enable` API).
- **Snippets**: `snippets/` as VSCode-format JSON, loaded by **mini.snippets**
  `gen_loader.from_lang()`. Files are keyed by **treesitter language name**, not
  filetype — e.g. shell snippets must be `bash.json` (sh→bash parser), and per-lang
  subdirs work (`snippets/python/*.json`).

## Verify before committing (always)

```sh
# config loads clean:
nvim --headless -u init.lua "+lua print('OK')" +qa
# adding/removing a plugin:
nvim --headless -u init.lua "+Lazy! install" +qa
nvim --headless -u init.lua "+Lazy! clean"   +qa   # prune removed plugins from disk
```
Headless can't exercise UIs (completion menu, snippet expansion, pickers, flash,
formatting) — for those, state clearly that interactive testing is needed.

## Workflow conventions (the user's strong preference)

- **One logical change per commit**; split multi-part work into separate commits.
- Verify headless, commit, **then push only when the user explicitly says "push it"** —
  never auto-push.
- Commit message footer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- When recommending a swap, base it on **actual usage** (grep keymaps/specs first) and
  surface real tradeoffs. Ask (AskUserQuestion) when behavior/risk genuinely diverges.
  Say so plainly when a requested change isn't cleanly possible (e.g. treesitter has no
  jinja highlight queries → native htmldjango instead).

## Gotchas / load-bearing details

- **Run system**: `<leader>r` and the `<localleader>` test maps in
  `after/ftplugin/*/run.vim` shell out via `g:ShellCommandPrefix()`
  (`plugin/shell_command_prefix.vim`), which returns `:FloatRun` — a toggleterm float
  defined there + `lua/run_term.lua`. Don't reintroduce floaterm; keep this contract.
- **AI**: CodeCompanion is the only AI plugin (`<leader>a*`). The commit-message
  feature is intentionally NOT CodeCompanion: `lua/commit_msg.lua` calls the OpenAI API
  directly (`<leader>gC` / `:GptGitCommitMsg`), key from `~/.config/openai_api_key` via
  `utils.set_open_api_key`.
- **Surround** uses `ys`/`ds`/`cs` + visual `S` (mini.surround mapped that way) so `s`
  stays free for **flash.nvim** jumps. Don't remap `s`/`S` carelessly.
- Prefer modern APIs on 0.11: `vim.uv` (not `vim.loop`), `vim.bo[buf]` (not
  `nvim_buf_set_option`), `vim.diagnostic.jump` (not `goto_next/prev`).
- Don't use interactive git flags (`-i`); they aren't supported here.
