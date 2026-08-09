Run a simple PowerShell command. Prefer Python for complex/stateful tasks; prefer `Glob`/`Grep` over `Get-ChildItem`/`Select-String` for search.

Quick reference:
- Cmdlets use Verb-Noun names: Get-ChildItem, Get-Content, Set-Location, Copy-Item, Move-Item, Remove-Item, New-Item, Select-String.
- Splat params with `@{}`: `$p = @{LiteralPath=$f; Destination=$d}; Copy-Item @p`. `$LASTEXITCODE` is native-only.
- Pipeline `|` passes .NET objects, not text; shape with Where-Object, Select-Object, ForEach-Object, Sort-Object, Measure-Object.
- `foreach (...) { }` is a statement, not an expression — assign first or use `ForEach-Object`.
- Operators: -eq -ne -gt -ge -lt -le, -like (wildcard), -match (regex), -contains (membership), -replace (regex); logical: -and -or -not.
- Chain with `;` (always) or `&&`/`||` (PS7+: next only on success/failure).
- Strings: 'single' literal; "double" expand $variables/$(subexpressions). `${name}_suffix` delimits variable names; `$($obj.prop)` = property. Avoid Bash-style `"\"q\""`; use `'"q"'`.
- Here-strings: `@'...'@` literal / `@"..."@` expanded; opener last on line, closer alone at line start. No Bash heredocs; prefer `ConvertTo-Json` over manual escaping.
- Native args: `& $exe @argList`. Don't use `$args`; `''` and `$null` are distinct. Capture `$LASTEXITCODE` immediately.
- Avoid backtick continuation; trailing space silently breaks. Break after pipes/commas/operators, or use `@()` arrays.
- Avoid `--%` (Stop-Parsing) except for fixed literal native commands.
- Env: `$env:NAME` for session scope; `[Environment]::SetEnvironmentVariable('NAME', 'value', 'Scope')` for persistence. Priority: Process > User > Machine. Append PATH with `$env:PATH += ';new\path'` — never overwrite; no `%NAME%`; child changes don't propagate.
- `$LASTEXITCODE` = exit code of last native command; `$?` = success. Unset if piped to a cmdlet — capture before piping.
- Parenthesize parameter value expressions: `-Index (100..120)` not `-Index 100..120`.
