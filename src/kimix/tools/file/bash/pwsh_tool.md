Run a simple PowerShell command. Prefer Python for complex or stateful tasks. Prefer `Glob`/`Grep` tools over `Get-ChildItem`/`Select-String` for file and content search.

PowerShell quick reference:
- Cmdlets use Verb-Noun names: Get-ChildItem (list files), Get-Content (read file), Set-Location (cd), Copy-Item, Move-Item, Remove-Item, New-Item, Select-String (grep), Get-Command, Get-Help.
- Splat cmdlet params with `@{}`: `$p = @{LiteralPath=$f; Destination=$d}; Copy-Item @p`. `$LASTEXITCODE` is for native commands only, not cmdlet success.
- The pipeline `|` passes .NET objects, not plain text; shape with Where-Object, Select-Object, ForEach-Object, Sort-Object, Measure-Object.
- `foreach (...) { }` is a statement, not an expression — cannot be piped directly. Assign first or use `ForEach-Object`.
- Comparison operators: -eq -ne -gt -ge -lt -le, -like (wildcard), -match (regex), -contains (collection membership), -replace (regex replace). Logical operators: -and -or -not (or `!`).
- Chain commands with `;` (always run next) or `&&` / `||` (PowerShell 7+: run next only on success / only on failure).
- Strings: 'single quotes' literal; "double quotes" expand $variables and $(subexpressions). Use `${name}_suffix` for variable boundaries, `$($obj.prop)` for properties. Avoid Bash-style `"\"q\""`; use `'"q"'` or backtick-escaping.
- Here-strings: `@'...'@` (literal) or `@"..."@` (expanded). Opening delimiter last on line; closing delimiter alone at line start. No Bash heredocs (`python - <<'PY'`). Prefer `ConvertTo-Json` over manual JSON escaping.
- Native arguments: `& $exe @argList`. Don't use `$args` (automatic variable). Omitted arg `''` `$null` are distinct. Capture `$LASTEXITCODE` immediately.
- Avoid backtick continuation `` ` ``; trailing space silently breaks. Use `@()` arrays or natural breaks after pipes/commas/operators.
- Avoid `--%` (Stop-Parsing); it disables parsing. Use only for fixed literal native commands.
- Environment variables: Use `$env:NAME` for session-scoped read/write. Use `[Environment]::SetEnvironmentVariable('NAME', 'value', 'Scope')` for persistence. Resolution priority: Process > User > Machine. List with `Get-ChildItem Env:`. Append PATH with `$env:PATH += ';new\path'` — never overwrite, check for duplicates first. Do not use `%NAME%` inside PowerShell. Child process changes do not propagate back to parent.
- $LASTEXITCODE holds the exit code of the last native command; $? is $true if the last command succeeded. Note: $LASTEXITCODE may not be set if native output is piped to a cmdlet; capture it before piping.
- Parameter value expressions must be parenthesized: `-Index (100..120)` not `-Index 100..120`.