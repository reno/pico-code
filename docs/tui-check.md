# TUI manual check

Run `./bin/pico chat --tui --provider=ollama --model=<model>` (or
`--provider=anthropic`) against a live backend and walk this list. Tick each
box only after actually observing the behavior — this is 7.2's AC, and it is
explicitly manual: nothing here is exercised by `make test`.

- [ ] Scrollback viewport shows prior turns; PgUp/PgDn scroll it.
- [ ] Textarea accepts multi-line input; Enter (not shift+Enter) submits.
- [ ] Spinner appears while waiting on the provider and disappears once the
      reply starts.
- [ ] Assistant text appears incrementally as it streams, not all at once.
- [ ] A tool call shows a running indicator, then flips to ✓ or ✗ once it
      finishes.
- [ ] Markdown in a finished reply (headings, code fences, lists) renders
      formatted, not as raw `#`/`` ``` `` characters.
- [ ] A tool that needs approval (`write_file` with `--allow-write`, or
      `run_command`) shows a modal; `y` approves, `n`/Esc denies, and the
      turn continues either way.
- [ ] Two tools needing approval in the same round show their modals one at
      a time, not stacked.
- [ ] Ctrl+C while a turn is in flight cancels that turn and returns to the
      prompt; the TUI keeps running.
- [ ] Ctrl+D exits the program cleanly, including while a turn is in flight.
- [ ] Resizing the terminal reflows the viewport and textarea without
      corrupting prior output.
