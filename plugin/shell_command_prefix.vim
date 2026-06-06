function! g:ShellCommandPrefix()
  if has('nvim')
    return 'FloatRun'
  else
    return '! '
  endif
endfunction

if has('nvim')
  " Run the rest of the line as a shell command in a toggleterm float.
  " luaeval passes <q-args> as _A so quoting/special chars survive intact.
  command! -nargs=+ -complete=shellcmd FloatRun
        \ call luaeval('require("run_term").run(_A)', <q-args>)
endif
