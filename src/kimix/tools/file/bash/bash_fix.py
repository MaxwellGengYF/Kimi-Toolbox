"""Windows Git Bash compatibility fixes for selected native POSIX commands.

Git for Windows ships a substantial POSIX userland, but a few command names
commonly emitted for Linux or macOS are absent even though an equivalent is
already available.  This module rewrites only verified, behaviorally compatible
command words.  It does not install software and deliberately leaves commands
without a faithful equivalent untouched.

Windows-style backslash paths (``D:\\repo\\src``, ``\\\\server\\share``,
``~\\Desktop``, ``.\\build``) — whether used as arguments, redirection targets,
or as the command word itself (``C:\\tools\\rg.exe``) — are rewritten to the
forward-slash spellings Git Bash understands, and the cmd.exe-only
``cd /d <path>`` form loses its flag
(``cd`` accepts a single argument in Bash).  Rewrites are conservative: the
unquoted word must look unambiguously like a Windows path, so quoted data,
tool-level escape sequences, short ambiguous words such as ``a\\nb``, and
single-segment relative paths such as ``foo\\bar`` are preserved byte-for-byte.
Words whose normalized form needs it (spaces, ``&``, ``;``, ...) are emitted
inside double quotes; glob metacharacters stay unquoted so ``D:/x/*.txt`` still
performs pathname expansion.

The scanner is shell-aware: quoted text, comments, heredoc and here-string
bodies, assignments, case patterns, and ordinary arguments are data, not
commands.  Nested command substitutions and process substitutions are scanned
as their own command contexts.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import regex as re

_REV_PERL = (
    "perl '-Mopen=:std,:encoding(UTF-8)' -e '"
    "my $zero = shift @ARGV; my $failed = 0; "
    "sub reverse_fh { my ($fh, $zero) = @_; "
    "local $/ = $zero ? qq(\\0) : qq(\\n); "
    "while (my $record = <$fh>) { "
    "my $ended = $zero ? $record =~ s/\\0\\z// : $record =~ s/\\r?\\n\\z//; "
    "print scalar reverse($record); "
    "print($zero ? qq(\\0) : qq(\\n)) if $ended } } "
    "if (@ARGV) { for my $file (@ARGV) { "
    "if (open my $fh, q(<:encoding(UTF-8)), $file) { reverse_fh($fh, $zero); close $fh } "
    "else { warn qq(rev: $file: $!\\n); $failed = 1 } } } "
    "else { reverse_fh(*STDIN, $zero) } exit $failed'"
)

_NATIVE_DELEGATE = (
    "local __kimix_native=''; __kimix_native=$(type -P {name}) || :; "
    "if [[ -n $__kimix_native ]]; then \"$__kimix_native\" \"$@\"; return; fi; "
)

# Fallbacks whose ``command -v`` hit can be a non-functional placeholder: the
# Microsoft Store App Execution Alias stubs in ``WindowsApps`` print an
# install prompt instead of running the tool.  They get a stub-aware guard
# (define the fallback even when ``command -v`` succeeds) and a delegate
# that refuses stub paths.
_STUB_AWARE_FALLBACKS = frozenset({"pip3", "python3"})

_PGREP_PS_NAME = (
    "$m = Get-Process | Where-Object { $_.Name -match $env:__KIMIX_PAT }; "
    "if ($m) { $m | ForEach-Object { if ($env:__KIMIX_LIST -eq \"1\") { "
    "\"$($_.Id) $($_.Name)\" } else { $_.Id } }; exit 0 } else { exit 1 }"
)
_PGREP_PS_FULL = (
    "$m = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match $env:__KIMIX_PAT }; "
    "if ($m) { $m | ForEach-Object { if ($env:__KIMIX_LIST -eq \"1\") { "
    "\"$($_.ProcessId) $($_.Name)\" } else { $_.ProcessId } }; exit 0 } else { exit 1 }"
)
_PKILL_PS_NAME = (
    "$m = Get-Process | Where-Object { $_.Name -match $env:__KIMIX_PAT }; "
    "if ($m) { $m | Stop-Process -Force; exit 0 } else { exit 1 }"
)
_PKILL_PS_FULL = (
    "$m = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match $env:__KIMIX_PAT }; "
    "if ($m) { $m | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; "
    "exit 0 } else { exit 1 }"
)

_ZIP_PS = (
    "Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem; "
    "$level = [System.IO.Compression.CompressionLevel]$env:__KIMIX_ZIP_LEVEL; "
    "$dest = $env:__KIMIX_ZIP_DEST; "
    "if (Test-Path -LiteralPath $dest) { Remove-Item -LiteralPath $dest -Force }; "
    "$zip = [System.IO.Compression.ZipFile]::Open($dest, [System.IO.Compression.ZipArchiveMode]::Create); "
    "foreach ($p in ($env:__KIMIX_ZIP_PATHS -split \"`n\")) { "
    "$item = Get-Item -LiteralPath $p; $base = $item.Name; "
    "if ($item.PSIsContainer) { $root = $item.FullName; "
    "Get-ChildItem -LiteralPath $root -Recurse -Force | ForEach-Object { "
    "$rel = $_.FullName.Substring($root.Length).TrimStart(\"\\\") -replace \"\\\\\", \"/\"; "
    "if ($_.PSIsContainer) { $zip.CreateEntry($base + \"/\" + $rel + \"/\") | Out-Null } "
    "else { [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $_.FullName, $base + \"/\" + $rel, $level) | Out-Null } } } "
    "else { [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $item.FullName, $base, $level) | Out-Null } }; "
    "$zip.Dispose(); if (Test-Path -LiteralPath $dest) { exit 0 } else { exit 1 }"
)

_TREE_PERL = (
    "perl -e '"
    "my ($maxdepth,$showall,$dirsonly,$noreport,$top)=@ARGV; "
    "print qq($top\\n); "
    "my ($ndirs,$nfiles)=(0,0); "
    "sub walk { my ($path,$prefix,$depth)=@_; "
    "return if $maxdepth && $depth>$maxdepth; "
    "opendir(my $dh,$path) or return; "
    "my @e = grep { ! /^[.][.]?$/ } readdir($dh); closedir($dh); "
    "@e = grep { $showall || ! /^[.]/ } @e; "
    "@e = grep { ! $dirsonly || -d qq($path/$_) } @e; "
    "@e = sort { lc($a) cmp lc($b) } @e; "
    "my $n=@e; my $i=0; "
    "for my $e (@e) { $i++; my $last = $i==$n; my $full = qq($path/$e); "
    "my $isdir = -d $full; "
    "if ($isdir) { $ndirs++ } else { $nfiles++ } "
    "print $prefix, ($last ? qq(`-- ) : qq(|-- )), $e, qq(\\n); "
    "walk($full, $prefix . ($last ? qq(    ) : qq(|   )), $depth+1) "
    "if $isdir && ! -l $full } } "
    "walk($top,q(),1); "
    "my $dw = $ndirs==1 ? q(directory) : q(directories); "
    "my $fw = $nfiles==1 ? q(file) : q(files); "
    "print qq(\\n$ndirs $dw, $nfiles $fw\\n) unless $noreport'"
)

_POWERSHELL_PASTE = (
    "powershell.exe -NoProfile -NonInteractive -Command "
    "'[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
    "[Console]::Out.Write((Get-Clipboard -Raw))'"
)

# Windows cmd-style command fallbacks.  These names are not part of the Git
# Bash POSIX userland and are commonly emitted by agents accustomed to cmd.exe
# or cross-platform documentation.  Each fallback is only installed when Git
# Bash cannot already resolve the command via ``command -v``.

_TASKLIST_PS = (
    "Get-Process | Select-Object Name, Id, CPU, WorkingSet | Format-Table -AutoSize"
)

_TASKKILL_PS = (
    "$force = $env:__KIMIX_FORCE -eq '1'; "
    "if ($env:__KIMIX_PID) { Stop-Process -Id $env:__KIMIX_PID -Force:$force; exit 0 } "
    "$procs = Get-Process | Where-Object { $_.Name -eq $env:__KIMIX_IM }; "
    "if ($procs) { $procs | Stop-Process -Force:$force; exit 0 } else { exit 1 }"
)

_SYSTEMINFO_PS = "Get-ComputerInfo | Format-List"

_KILLALL_PS = (
    "$procs = Get-Process | Where-Object { $_.Name -eq $env:__KIMIX_NAME }; "
    "if ($procs) { $procs | Stop-Process -Force; exit 0 } else { exit 1 }"
)

_PIDOF_PS = (
    "$ids = (Get-Process | Where-Object { $_.Name -eq $env:__KIMIX_NAME }).Id; "
    "if ($ids) { $ids -join \" \"; exit 0 } else { exit 1 }"
)

_COLUMN_PERL = (
    "perl -e '"
    "my $sep = shift @ARGV; $sep = qr/\\s+/ if $sep eq \"DEFAULT\"; "
    "my @rows; my @max; "
    "while (<>) { chomp; my @c = split $sep; push @rows, \\@c; "
    "for my $i (0..$#c) { $max[$i] = length($c[$i]) if !defined $max[$i] || length($c[$i]) > $max[$i]; } } "
    "for my $r (@rows) { print join(\"  \", map { sprintf(\"%-*s\", $max[$_]//0, $r->[$_]) } 0..$#$r), \"\\n\"; }'"
)

_FALLBACK_BODIES = {
    "gtimeout": "timeout \"$@\"",
    "rev": (
        "local __kimix_zero=0; while (( $# )); do case $1 in "
        "-0|--zero) __kimix_zero=1; shift;; "
        "--) shift; break;; "
        "-*) printf '%s\\n' \"rev: unsupported option: $1\" >&2; return 1;; "
        "*) break;; esac; done; "
        + _REV_PERL
        + " -- \"$__kimix_zero\" \"$@\""
    ),
    "xdg-open": "start \"$@\"",
    "open": "start \"$@\"",
    "pbcopy": "clip.exe \"$@\"",
    "pbpaste": _POWERSHELL_PASTE + " \"$@\"",
    "wget": (
        "local __kimix_url='' __kimix_out='' __kimix_stdout=0; "
        "local -a __kimix_args=(); "
        "while (( $# )); do case $1 in "
        "-O|--output-document) __kimix_out=$2; shift 2;; "
        "-O?*) __kimix_out=${1#-O}; shift;; "
        "--output-document=*) __kimix_out=${1#*=}; shift;; "
        "-q|--quiet) __kimix_args+=(-s); shift;; "
        "-c|--continue) __kimix_args+=(-C -); shift;; "
        "--no-check-certificate) __kimix_args+=(-k); shift;; "
        "-T|--timeout) __kimix_args+=(--max-time \"$2\"); shift 2;; "
        "--timeout=*) __kimix_args+=(--max-time \"${1#*=}\"); shift;; "
        "-*) printf '%s\\n' \"wget: unsupported option for curl fallback: $1\" >&2; return 1;; "
        "*) __kimix_url=$1; shift;; esac; done; "
        "if [[ -z $__kimix_url ]]; then "
        "printf '%s\\n' 'wget: missing URL' >&2; return 1; fi; "
        "if [[ $__kimix_out == '-' ]]; then __kimix_stdout=1; fi; "
        "if [[ -z $__kimix_out && $__kimix_stdout -eq 0 ]]; then "
        "__kimix_out=${__kimix_url##*/}; "
        "[[ -n $__kimix_out ]] || __kimix_out=index.html; fi; "
        "if (( __kimix_stdout )); then "
        "curl -fSL \"${__kimix_args[@]}\" -- \"$__kimix_url\"; "
        "else curl -fSL \"${__kimix_args[@]}\" -o \"$__kimix_out\" -- \"$__kimix_url\"; fi"
    ),
    "xclip": (
        "local __kimix_out=0; while (( $# )); do case $1 in "
        "-o|-out) __kimix_out=1; shift;; "
        "-i|-in) shift;; "
        "-selection|-d|-display) shift 2;; "
        "-selection*|-display*) shift;; "
        "-*) printf '%s\\n' \"xclip: unsupported option for clipboard fallback: $1\" >&2; return 1;; "
        "*) shift;; esac; done; "
        "if (( __kimix_out )); then " + _POWERSHELL_PASTE + "; else clip.exe; fi"
    ),
    "xsel": (
        "local __kimix_out=0; while (( $# )); do case $1 in "
        "--output) __kimix_out=1; shift;; "
        "--input|--clipboard|--primary|--secondary) shift;; "
        "--*) printf '%s\\n' \"xsel: unsupported option for clipboard fallback: $1\" >&2; return 1;; "
        "-*) case $1 in *o*) __kimix_out=1;; esac; shift;; "
        "*) shift;; esac; done; "
        "if (( __kimix_out )); then " + _POWERSHELL_PASTE + "; else clip.exe; fi"
    ),
    "wl-copy": (
        "while (( $# )); do case $1 in "
        "-*) printf '%s\\n' \"wl-copy: unsupported option for clipboard fallback: $1\" >&2; return 1;; "
        "*) shift;; esac; done; clip.exe"
    ),
    "wl-paste": (
        "while (( $# )); do case $1 in "
        "-n|--no-newline) shift;; "
        "-*) printf '%s\\n' \"wl-paste: unsupported option for clipboard fallback: $1\" >&2; return 1;; "
        "*) shift;; esac; done; " + _POWERSHELL_PASTE
    ),
    "zip": (
        "local __kimix_archive='' __kimix_level=Optimal __kimix_p='' "
        "__kimix_combo='' __kimix_i=0; "
        "local -a __kimix_paths=() __kimix_wpaths=() __kimix_split=(); "
        "while (( $# )); do "
        "if [[ $1 == -[!-]* && ${#1} -gt 2 ]]; then "
        "__kimix_combo=${1#-}; __kimix_split=(); shift; "
        "for (( __kimix_i=0; __kimix_i<${#__kimix_combo}; __kimix_i++ )); do "
        "__kimix_split+=(-${__kimix_combo:__kimix_i:1}); done; "
        "set -- \"${__kimix_split[@]}\" \"$@\"; continue; fi; "
        "case $1 in "
        "-r|-R|--recurse-paths|-q|--quiet) shift;; "
        "-0) __kimix_level=NoCompression; shift;; "
        "-1) __kimix_level=Fastest; shift;; "
        "-[2-9]) shift;; "
        "-*) printf '%s\\n' \"zip: unsupported option for Compress-Archive fallback: $1\" >&2; return 1;; "
        "*) if [[ -z $__kimix_archive ]]; then __kimix_archive=$1; "
        "else __kimix_paths+=(\"$1\"); fi; shift;; esac; done; "
        "if [[ -z $__kimix_archive || ${#__kimix_paths[@]} -eq 0 ]]; then "
        "printf '%s\\n' 'zip: missing archive name or input paths' >&2; return 1; fi; "
        "for __kimix_p in \"${__kimix_paths[@]}\"; do "
        "__kimix_wpaths+=(\"$(cygpath -w -- \"$__kimix_p\")\"); done; "
        "__kimix_archive=$(cygpath -w -- \"$__kimix_archive\"); "
        "__KIMIX_ZIP_LEVEL=$__kimix_level __KIMIX_ZIP_DEST=$__kimix_archive "
        "__KIMIX_ZIP_PATHS=$(printf '%s\\n' \"${__kimix_wpaths[@]}\") "
        "powershell.exe -NoProfile -NonInteractive -Command '" + _ZIP_PS + "'"
    ),
    "nc": (
        "local __kimix_z=0 __kimix_v=0 __kimix_w='' __kimix_host='' __kimix_port=''; "
        "while (( $# )); do case $1 in "
        "-z) __kimix_z=1; shift;; "
        "-v) __kimix_v=1; shift;; "
        "-zv|-vz) __kimix_z=1; __kimix_v=1; shift;; "
        "-w) __kimix_w=$2; shift 2;; "
        "-w?*) __kimix_w=${1#-w}; shift;; "
        "-*) printf '%s\\n' \"nc: unsupported option for /dev/tcp fallback: $1\" >&2; return 1;; "
        "*) if [[ -z $__kimix_host ]]; then __kimix_host=$1; "
        "elif [[ -z $__kimix_port ]]; then __kimix_port=$1; "
        "else printf '%s\\n' 'nc: too many arguments' >&2; return 1; fi; "
        "shift;; esac; done; "
        "if (( ! __kimix_z )); then "
        "printf '%s\\n' 'nc: only -z (zero-I/O scan) mode is supported by this fallback' >&2; "
        "return 1; fi; "
        "if [[ -z $__kimix_host || -z $__kimix_port ]]; then "
        "printf '%s\\n' 'nc: missing host or port' >&2; return 1; fi; "
        "if [[ -n $__kimix_w ]]; then "
        "timeout \"$__kimix_w\" bash -c 'exec 3<>/dev/tcp/$1/$2' _ "
        "\"$__kimix_host\" \"$__kimix_port\" 2>/dev/null; "
        "else (exec 3<>/dev/tcp/\"$__kimix_host\"/\"$__kimix_port\") 2>/dev/null; fi; "
        "local __kimix_rc=$?; "
        "(( __kimix_rc != 0 )) && __kimix_rc=1; "
        "if (( __kimix_rc == 0 )); then "
        "(( __kimix_v )) && printf '%s\\n' \"Connection to $__kimix_host $__kimix_port port [tcp/*] succeeded!\" >&2; "
        "else "
        "(( __kimix_v )) && printf '%s\\n' \"nc: connect to $__kimix_host port $__kimix_port (tcp) failed\" >&2; fi; "
        "return $__kimix_rc"
    ),
    "pgrep": (
        "local __kimix_list=0 __kimix_full=0 __kimix_pat=''; "
        "while (( $# )); do case $1 in "
        "-l) __kimix_list=1; shift;; "
        "-f) __kimix_full=1; shift;; "
        "-lf|-fl) __kimix_list=1; __kimix_full=1; shift;; "
        "--) shift; break;; "
        "-*) printf '%s\\n' \"pgrep: unsupported option for Get-Process fallback: $1\" >&2; return 1;; "
        "*) __kimix_pat=$1; shift;; esac; done; "
        "if [[ -z $__kimix_pat ]]; then "
        "printf '%s\\n' 'pgrep: missing pattern' >&2; return 1; fi; "
        "if (( __kimix_full )); then "
        "__KIMIX_PAT=$__kimix_pat __KIMIX_LIST=$__kimix_list "
        "powershell.exe -NoProfile -NonInteractive -Command '" + _PGREP_PS_FULL + "'; "
        "else "
        "__KIMIX_PAT=$__kimix_pat __KIMIX_LIST=$__kimix_list "
        "powershell.exe -NoProfile -NonInteractive -Command '" + _PGREP_PS_NAME + "'; fi"
    ),
    "pkill": (
        "local __kimix_full=0 __kimix_pat=''; "
        "while (( $# )); do case $1 in "
        "-f) __kimix_full=1; shift;; "
        "--) shift; break;; "
        "-*) printf '%s\\n' \"pkill: unsupported option for Stop-Process fallback: $1\" >&2; return 1;; "
        "*) __kimix_pat=$1; shift;; esac; done; "
        "if [[ -z $__kimix_pat ]]; then "
        "printf '%s\\n' 'pkill: missing pattern' >&2; return 1; fi; "
        "if (( __kimix_full )); then "
        "__KIMIX_PAT=$__kimix_pat "
        "powershell.exe -NoProfile -NonInteractive -Command '" + _PKILL_PS_FULL + "'; "
        "else "
        "__KIMIX_PAT=$__kimix_pat "
        "powershell.exe -NoProfile -NonInteractive -Command '" + _PKILL_PS_NAME + "'; fi"
    ),
    "traceroute": (
        "local -a __kimix_args=(); "
        "while (( $# )); do case $1 in "
        "-n) __kimix_args+=(-d); shift;; "
        "-m) __kimix_args+=(-h \"$2\"); shift 2;; "
        "-m?*) __kimix_args+=(-h \"${1#-m}\"); shift;; "
        "--max-hop=*) __kimix_args+=(-h \"${1#*=}\"); shift;; "
        "-w) __kimix_args+=(-w \"$(( $2 * 1000 ))\"); shift 2;; "
        "-w?*) __kimix_args+=(-w \"$(( ${1#-w} * 1000 ))\"); shift;; "
        "-*) printf '%s\\n' \"traceroute: unsupported option for tracert fallback: $1\" >&2; return 1;; "
        "*) __kimix_args+=(\"$1\"); shift;; esac; done; "
        "tracert \"${__kimix_args[@]}\""
    ),
    "tree": (
        "local __kimix_depth=0 __kimix_all=0 __kimix_dirs=0 __kimix_noreport=0 "
        "__kimix_dir=''; "
        "while (( $# )); do case $1 in "
        "-L) __kimix_depth=$2; shift 2;; "
        "-L?*) __kimix_depth=${1#-L}; shift;; "
        "-a) __kimix_all=1; shift;; "
        "-d) __kimix_dirs=1; shift;; "
        "--noreport) __kimix_noreport=1; shift;; "
        "--) shift; break;; "
        "-*) printf '%s\\n' \"tree: unsupported option for perl fallback: $1\" >&2; return 1;; "
        "*) __kimix_dir=$1; shift;; esac; done; "
        "[[ -n $__kimix_dir ]] || __kimix_dir=.; "
        + _TREE_PERL
        + " -- \"$__kimix_depth\" \"$__kimix_all\" \"$__kimix_dirs\" \"$__kimix_noreport\" \"$__kimix_dir\""
    ),
    "say": (
        "while (( $# )); do case $1 in "
        "-*) printf '%s\\n' \"say: unsupported option for SAPI fallback: $1\" >&2; return 1;; "
        "*) shift;; esac; done; "
        "__KIMIX_SAY_TEXT=$* "
        "powershell.exe -NoProfile -NonInteractive -Command "
        "'Add-Type -AssemblyName System.Speech; "
        "(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak($env:__KIMIX_SAY_TEXT)'"
    ),
    "python3": 'python "$@"',
    "pip3": 'pip "$@"',
    # Windows cmd-style commands -> POSIX/Git Bash equivalents.
    "copy": (
        "if [[ $# -lt 2 ]]; then "
        "printf '%s\\n' 'copy: missing source or destination' >&2; return 1; fi; "
        "cp -R -- \"$@\""
    ),
    "move": (
        "if [[ $# -lt 2 ]]; then "
        "printf '%s\\n' 'move: missing source or destination' >&2; return 1; fi; "
        "mv -- \"$@\""
    ),
    "del": "rm -- \"$@\"",
    "erase": "rm -- \"$@\"",
    "ren": (
        "if [[ $# -ne 2 ]]; then "
        "printf '%s\\n' 'ren: exactly two arguments required' >&2; return 1; fi; "
        "mv -- \"$1\" \"$2\""
    ),
    "rename": (
        "if [[ $# -ne 2 ]]; then "
        "printf '%s\\n' 'rename: exactly two arguments required' >&2; return 1; fi; "
        "mv -- \"$1\" \"$2\""
    ),
    "rd": "rmdir -- \"$@\"",
    "md": "mkdir -p -- \"$@\"",
    "chdir": "cd -- \"$@\"",
    "cls": "clear",
    "xcopy": "cp -r -- \"$@\"",
    "mklink": (
        "local __kimix_hard=0 __kimix_link='' __kimix_target=''; "
        "while (( $# )); do case $1 in "
        "/D|/d|/J|/j) shift;; "
        "/H|/h) __kimix_hard=1; shift;; "
        "*) if [[ -z $__kimix_link ]]; then __kimix_link=$1; "
        "elif [[ -z $__kimix_target ]]; then __kimix_target=$1; "
        "else printf '%s\\n' 'mklink: too many arguments' >&2; return 1; fi; "
        "shift;; esac; done; "
        "if [[ -z $__kimix_link || -z $__kimix_target ]]; then "
        "printf '%s\\n' 'mklink: missing link name or target' >&2; return 1; fi; "
        "if (( __kimix_hard )); then ln -f -- \"$__kimix_target\" \"$__kimix_link\"; "
        "else ln -s -- \"$__kimix_target\" \"$__kimix_link\"; fi"
    ),
    "findstr": "grep \"$@\"",
    "fc": "diff \"$@\"",
    "where": "which \"$@\"",
    "tasklist": (
        "powershell.exe -NoProfile -NonInteractive -Command '" + _TASKLIST_PS + "'"
    ),
    "taskkill": (
        "local __kimix_force=0 __kimix_pid='' __kimix_im=''; "
        "while (( $# )); do case $1 in "
        "/F|/f) __kimix_force=1; shift;; "
        "/IM|/im) __kimix_im=$2; shift 2;; "
        "/PID|/pid) __kimix_pid=$2; shift 2;; "
        "/*) printf '%s\\n' \"taskkill: unsupported option: $1\" >&2; return 1;; "
        "*) printf '%s\\n' \"taskkill: unsupported argument: $1\" >&2; return 1;; esac; done; "
        "if [[ -n $__kimix_pid ]]; then "
        "__KIMIX_FORCE=$__kimix_force __KIMIX_PID=$__kimix_pid "
        "powershell.exe -NoProfile -NonInteractive -Command '" + _TASKKILL_PS + "'; "
        "elif [[ -n $__kimix_im ]]; then "
        "__KIMIX_FORCE=$__kimix_force __KIMIX_IM=$__kimix_im "
        "powershell.exe -NoProfile -NonInteractive -Command '" + _TASKKILL_PS + "'; "
        "else printf '%s\\n' 'taskkill: missing /PID or /IM' >&2; return 1; fi"
    ),
    "systeminfo": (
        "powershell.exe -NoProfile -NonInteractive -Command '" + _SYSTEMINFO_PS + "'"
    ),
    # POSIX utilities often absent from a bare Git Bash userland.
    "watch": (
        "local __kimix_interval=2; "
        "while (( $# )); do case $1 in "
        "-n) __kimix_interval=$2; shift 2;; "
        "-n?*) __kimix_interval=${1#-n}; shift;; "
        "-t|-d|--no-title|--color) shift;; "
        "--) shift; break;; "
        "-*) printf '%s\\n' \"watch: unsupported option: $1\" >&2; return 1;; "
        "*) break;; esac; done; "
        "if [[ $# -eq 0 ]]; then printf '%s\\n' 'watch: missing command' >&2; return 1; fi; "
        "while true; do clear; \"$@\"; sleep \"$__kimix_interval\"; done"
    ),
    "killall": (
        "if [[ $# -eq 0 ]]; then printf '%s\\n' 'killall: missing process name' >&2; return 1; fi; "
        "__KIMIX_NAME=$1 powershell.exe -NoProfile -NonInteractive -Command '" + _KILLALL_PS + "'"
    ),
    "pidof": (
        "if [[ $# -eq 0 ]]; then printf '%s\\n' 'pidof: missing process name' >&2; return 1; fi; "
        "__KIMIX_NAME=$1 powershell.exe -NoProfile -NonInteractive -Command '" + _PIDOF_PS + "'"
    ),
    "column": (
        "local __kimix_sep='DEFAULT'; "
        "while (( $# )); do case $1 in "
        "-t) shift;; "
        "-s) __kimix_sep=$2; shift 2;; "
        "-s?*) __kimix_sep=${1#-s}; shift;; "
        "-*) printf '%s\\n' \"column: unsupported option for perl fallback: $1\" >&2; return 1;; "
        "*) break;; esac; done; "
        + _COLUMN_PERL
        + " \"$__kimix_sep\" \"$@\""
    ),
}


# GNU ``g``-prefixed command names (the Homebrew coreutils spelling used on
# macOS) map to the very same GNU tools that Git Bash already ships, so the
# mapping is faithful by construction.  ``gtimeout`` is spelled out above.
for _gnu_command in (
    "awk", "cat", "comm", "cp", "cut", "date", "df", "du", "egrep",
    "fgrep", "find", "grep", "head", "join", "ln", "ls", "make", "mkdir",
    "mv", "paste", "readlink", "realpath", "rm", "rmdir", "sed", "seq",
    "shuf", "sort", "split", "stat", "tail", "tar", "tr", "uniq", "wc",
    "xargs",
):
    _FALLBACK_BODIES.setdefault("g" + _gnu_command, f'{_gnu_command} "$@"')


from kimix.native_loader import (
    get_module as _native_get_module,
    use_native as _native_use_native,
)

# Resolved once at import time (stable runtime: result never changes).
_NATIVE_PARSE = _native_get_module("parse")


def _fallback_definition(name: str) -> str:
    body = _FALLBACK_BODIES[name]
    if name in _STUB_AWARE_FALLBACKS:
        # The Microsoft Store App Execution Alias satisfies ``command -v``
        # but is not a working interpreter: define the fallback anyway, and
        # never delegate to the stub path.
        guard = (
            f"if ! command -v {name} >/dev/null 2>&1 "
            f"|| [[ $(type -P {name}) == *WindowsApps* ]]; then "
        )
        delegate = (
            f"local __kimix_native=''; __kimix_native=$(type -P {name}) || :; "
            f"if [[ -n $__kimix_native && $__kimix_native != *WindowsApps* ]]; then "
            f"\"$__kimix_native\" \"$@\"; return; fi; "
        )
    else:
        guard = f"if ! command -v {name} >/dev/null 2>&1; then "
        delegate = _NATIVE_DELEGATE.format(name=name)
    return f"{guard}{name}() {{ {delegate}{body}; }}; fi"


def _single_quote(command: str) -> str:
    """Quote *command* as one literal Bash word."""
    return "'" + command.replace("'", "'\"'\"'") + "'"


def _wrapper_runner(name: str) -> str:
    """Return an executable command for wrappers that cannot invoke functions."""
    script = _fallback_definition(name) + f"; {name} \"$@\""
    return "/usr/bin/bash -c " + _single_quote(script) + " --"


# ``netcat`` is a common synonym for ``nc`` on systems where the binary is
# spelled with the longer name; both are absent from Git Bash, so share the
# same ``/dev/tcp`` zero-I/O fallback.
_FALLBACK_BODIES.setdefault("netcat", _FALLBACK_BODIES["nc"])

_FALLBACKS = {name: _fallback_definition(name) for name in _FALLBACK_BODIES}

_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\+)?=")
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_COMMAND_START_KEYWORDS = frozenset(
    {"!", "{", "if", "then", "elif", "else", "while", "until", "do"}
)
_COMMAND_END_KEYWORDS = frozenset({"fi", "done", "esac"})
_LIST_KEYWORDS = frozenset({"for", "select", "case"})

_COMMAND_WRAPPERS = frozenset(
    {"command", "coproc", "env", "exec", "nohup", "sudo", "time"}
)
_WRAPPER_OPTIONS_WITH_VALUE = {
    "env": frozenset(
        {
            "-u",
            "--unset",
            "-C",
            "--chdir",
            "-S",
            "--split-string",
        }
    ),
    "exec": frozenset({"-a"}),
    "sudo": frozenset(
        {
            "-C",
            "--close-from",
            "-D",
            "--chdir",
            "-g",
            "--group",
            "-h",
            "--host",
            "-p",
            "--prompt",
            "-R",
            "--chroot",
            "-r",
            "--role",
            "-t",
            "--type",
            "-T",
            "--command-timeout",
            "-u",
            "--user",
        }
    ),
    "time": frozenset({"-f", "--format", "-o", "--output"}),
}

# Wrapper options whose value is a filesystem path rather than a name or
# number.  Windows backslash spellings of these values are rewritten for Git
# Bash just like ordinary argument paths; every other option value stays
# opaque.
_WRAPPER_PATH_OPTIONS = {
    "env": frozenset({"-C", "--chdir"}),
    "sudo": frozenset({"-D", "--chdir"}),
    "time": frozenset({"-o", "--output"}),
}
_WRAPPER_PATH_OPTION_LONG = frozenset({"--chdir", "--output"})

_OPERATOR_CHARS = frozenset(";&|()<>\n")
_REDIRECTION_START = frozenset("<>")

# Characters that terminate an unquoted word: operators plus horizontal
# whitespace.  A single frozenset membership test is cheaper than two.
_WORD_END_CHARS = _OPERATOR_CHARS | frozenset(" \t\r")

# Nesting deeper than this is never scanned: ``_find_matching`` and
# ``_scan_range`` recurse once per ``$( ... )`` level, so pathological input
# (e.g. a pasted blob with hundreds of nested substitutions) would otherwise
# cost O(depth * length).  Real commands stay far below this bound; content
# beyond it is simply left byte-for-byte for Bash to handle.
_MAX_NESTING_DEPTH = 1024

# ── Windows path recognition ────────────────────────────────────────────────
# Rewrites apply only to unquoted words that unambiguously look like Windows
# paths; everything else (quotes, expansions, short ambiguous words) is left
# byte-for-byte for Bash to handle.

_PATH_DRIVE_RE = re.compile(r"[A-Za-z]:\\.*")
"""Drive-absolute path such as ``D:\\foo`` or the drive root ``C:\\``."""

_PATH_SEGMENT_RE = re.compile(r"[A-Za-z0-9_.~\\-]+")
"""Decoded value of a plausible multi-segment relative path (no spaces)."""

_PATH_SAFE_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_./:~@%+=-#,[]*?"
)
"""Characters that may appear unquoted in a normalized path word.

Glob metacharacters are included on purpose so ``D:/x/*.txt`` still performs
pathname expansion instead of being quoted into a literal name.
"""

_ESCAPED_LITERAL_CHARS = frozenset(" \t&;|()<>#'\"$`{}!")
"""Chars whose backslash form is a pure Bash escape, not a path separator.

``\\ `` (escaped space) is how a space inside an unquoted word is written;
normalizing it to ``/ `` would invent a directory level that does not exist.
The backslash is dropped and the character kept inside its segment.
"""

@dataclass(frozen=True)
class BashFix:
    """Result of :func:`fix_bash_command`.

    ``replacements`` records each original command name in source order and
    ``path_changes`` each original argument or command word whose
    Windows-style backslashes (or cmd.exe ``/d`` flag) were rewritten for Git
    Bash.  Empty tuples mean the command was returned byte-for-byte unchanged.
    """

    command: str
    replacements: tuple[str, ...] = ()
    path_changes: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        """Return whether any compatibility replacement was made."""
        return bool(self.replacements) or bool(self.path_changes)

    @property
    def warning(self) -> str:
        """Return a concise description of compatibility changes."""
        parts: list[str] = []
        if self.replacements:
            names = ", ".join(f"`{name}`" for name in self.replacements)
            parts.append(
                f"Added Windows Git Bash fallback(s) for native command(s): {names}."
            )
        if self.path_changes:
            words = ", ".join(f"`{word}`" for word in self.path_changes)
            parts.append(
                "Rewrote Windows path(s) for Git Bash (backslashes to forward "
                f"slashes): {words}."
            )
        return " ".join(parts)


@dataclass
class _Wrapper:
    kind: str
    skip_next: bool = False
    opaque: bool = False
    path_value: bool = False


@dataclass
class _HereDoc:
    delimiter: str | None
    strip_tabs: bool
    expands: bool


class _Scanner:
    """Conservative scanner for Bash executable command positions."""

    __slots__ = ("s", "n", "edits", "names", "path_notes", "nest_depth")

    def __init__(self, command: str) -> None:
        self.s = command
        self.n = len(command)
        self.edits: list[tuple[int, int, str]] = []
        self.names: list[str] = []
        self.path_notes: list[str] = []
        self.nest_depth = 0

    def fix(self) -> BashFix:
        try:
            self._scan_range(0, self.n)
        except RecursionError:
            # Malformed or adversarial nesting must never make the Bash tool
            # fail before Bash itself can report the syntax error.
            return BashFix(self.s)
        if not self.names and not self.edits:
            return BashFix(self.s)
        definitions = "\n".join(
            _FALLBACKS[name] for name in dict.fromkeys(self.names)
        )
        if self.edits:
            pieces: list[str] = []
            previous = 0
            for start, end, replacement in sorted(self.edits):
                pieces.extend((self.s[previous:start], replacement))
                previous = end
            pieces.append(self.s[previous:])
            source = "".join(pieces)
        else:
            source = self.s
        prefix = definitions + "\n" if definitions else ""
        return BashFix(prefix + source, tuple(self.names), tuple(self.path_notes))

    @staticmethod
    def _literal_command_name(raw: str) -> str | None:
        """Return the command name produced solely by Bash quote removal.

        Bash permits literal command words such as ``'rev'``, ``\rev`` and
        ``r\"\"ev``.  Only words whose value can be determined without any
        expansion are accepted; parameter/command/arithmetic expansions,
        globbing, and malformed quotes remain untouched for Bash to handle.
        """
        value: list[str] = []
        i = 0
        while i < len(raw):
            ch = raw[i]
            if ch == "\\":
                if i + 1 >= len(raw):
                    return None
                if raw[i + 1] == "\n":
                    i += 2
                    continue
                value.append(raw[i + 1])
                i += 2
                continue
            if ch == "'":
                close = raw.find("'", i + 1)
                if close < 0:
                    return None
                value.append(raw[i + 1 : close])
                i = close + 1
                continue
            if ch == '"':
                i += 1
                while i < len(raw) and raw[i] != '"':
                    inner = raw[i]
                    if inner in "$`":
                        return None
                    if inner == "\\" and i + 1 < len(raw):
                        escaped = raw[i + 1]
                        if escaped in '$`"\\\n':
                            if escaped != "\n":
                                value.append(escaped)
                            i += 2
                            continue
                    value.append(inner)
                    i += 1
                if i >= len(raw):
                    return None
                i += 1
                continue
            if ch in "$`*?[{~":
                return None
            value.append(ch)
            i += 1
        name = "".join(value)
        return name if name in _FALLBACKS else None

    def _scan_range(self, start: int, end: int) -> None:
        """Scan *start..end* as a command context, bounding recursion depth.

        The wrapper exists so pathological nesting (hundreds of nested
        ``$( ... )`` levels) costs O(_MAX_NESTING_DEPTH * length) instead of
        O(depth * length); the innermost levels are left for Bash.
        """
        if self.nest_depth >= _MAX_NESTING_DEPTH:
            return
        self.nest_depth += 1
        try:
            self._scan_range_inner(start, end)
        finally:
            self.nest_depth -= 1

    def _scan_range_inner(self, start: int, end: int) -> None:
        s = self.s
        i = start
        command_expected = True
        redirect_expected = False
        redirect_resume = True
        wrapper: _Wrapper | None = None
        heredoc_operator: str | None = None
        herestring_flag = False
        pending_heredocs: list[_HereDoc] = []
        case_stack: list[str] = []
        function_name_expected = False
        function_body_expected = False

        while i < end:
            ch = s[i]

            if ch in " \t\r":
                i += 1
                continue
            if ch == "\\" and i + 1 < end and s[i + 1] == "\n":
                i += 2
                continue
            if ch == "\n":
                i += 1
                if pending_heredocs:
                    i = self._skip_heredoc_bodies(i, end, pending_heredocs)
                    pending_heredocs.clear()
                command_expected = True
                redirect_expected = False
                heredoc_operator = None
                herestring_flag = False
                wrapper = None
                continue
            if ch == "#" and self._comment_starts(i, start):
                newline = s.find("\n", i + 1, end)
                i = end if newline < 0 else newline
                continue

            process_substitution = ch in _REDIRECTION_START and (
                s.startswith("<(", i) or s.startswith(">(", i)
            )
            if not process_substitution and (
                ch in _REDIRECTION_START
                or (ch == "&" and s.startswith("&>", i))
                or (ch.isdigit() and self._redirection_after_fd(i, end))
            ):
                op_start = i
                if ch.isdigit():
                    while i < end and s[i].isdigit():
                        i += 1
                op, i = self._read_redirection(i, end)
                if op:
                    redirect_resume = command_expected
                    redirect_expected = True
                    herestring_flag = op == "<<<"
                    if op in {"<<", "<<-"}:
                        # The delimiter is captured when the following word is
                        # read; its body starts only after this command line.
                        pass
                    else:
                        op_start = -1
                    if op_start >= 0:
                        heredoc_operator = op
                    continue
                i = op_start

            if redirect_expected:
                if s.startswith("<(" , i) or s.startswith(">(", i):
                    close = self._find_matching(i + 2, end, ")")
                    self._scan_range(i + 2, close if close < end else end)
                    word_end = close + 1 if close < end else end
                else:
                    scan_substitutions = heredoc_operator not in {"<<", "<<-"}
                    word_end = self._read_word(
                        i, end, scan_substitutions=scan_substitutions
                    )
                if word_end <= i:
                    i += 1
                    continue
                if heredoc_operator in {"<<", "<<-"}:
                    heredoc = self._heredoc_delimiter(s[i:word_end])
                    if heredoc is not None:
                        delimiter, expands = heredoc
                        pending_heredocs.append(
                            _HereDoc(delimiter, heredoc_operator == "<<-", expands)
                        )
                elif not herestring_flag:
                    raw_word = s[i:word_end]
                    replacement = self._windows_path_replacement(raw_word)
                    if replacement is not None:
                        self.edits.append((i, word_end, replacement))
                        self.path_notes.append(raw_word)
                i = word_end
                command_expected = redirect_resume
                redirect_expected = False
                heredoc_operator = None
                continue

            if s.startswith("[[", i) and ch == "[":
                function_body_expected = False
                i = self._skip_conditional(i + 2, end)
                command_expected = False
                continue
            if s.startswith("((", i) and ch == "(":
                function_body_expected = False
                i = self._skip_arithmetic(i + 2, end)
                command_expected = False
                continue
            if ch == "$" and s.startswith("$(", i) and not s.startswith("$((", i):
                close = self._find_matching(i + 2, end, ")")
                inner_end = close if close < end else end
                self._scan_range(i + 2, inner_end)
                i = close + 1 if close < end else end
                if case_stack and case_stack[-1] == "word":
                    # A substitution can be the case subject word itself
                    # (``case $(...) in ...``); it ends the header just like
                    # a plain subject word.
                    case_stack[-1] = "await-in"
                command_expected = False
                continue
            if ch == "`":
                close = self._find_backtick_end(i + 1, end)
                self._scan_range(i + 1, close)
                i = close + 1 if close < end else end
                if case_stack and case_stack[-1] == "word":
                    case_stack[-1] = "await-in"
                command_expected = False
                continue
            if ch in _REDIRECTION_START and (
                s.startswith("<(", i) or s.startswith(">(", i)
            ):
                close = self._find_matching(i + 2, end, ")")
                self._scan_range(i + 2, close if close < end else end)
                i = close + 1 if close < end else end
                if case_stack and case_stack[-1] == "word":
                    case_stack[-1] = "await-in"
                command_expected = False
                continue

            op, op_end = self._read_control_operator(i, end)
            if op:
                i = op_end
                if op == "(" and function_body_expected:
                    function_body_expected = False
                    command_expected = True
                elif op == "(":
                    command_expected = True
                elif op == ")":
                    if case_stack and case_stack[-1] == "patterns":
                        case_stack[-1] = "body"
                        command_expected = True
                    else:
                        command_expected = False
                elif op in {";;", ";&", ";;&"}:
                    if case_stack:
                        case_stack[-1] = "patterns"
                        command_expected = False
                    else:
                        command_expected = True
                else:
                    command_expected = True
                redirect_expected = False
                heredoc_operator = None
                wrapper = None
                continue

            word_start = i
            scan_substitutions = heredoc_operator not in {"<<", "<<-"}
            word_end = self._read_word(i, end, scan_substitutions=scan_substitutions)
            if word_end <= i:
                i += 1
                continue
            raw = s[word_start:word_end]
            i = word_end

            if function_name_expected:
                function_name_expected = False
                function_body_expected = True
                command_expected = False
                declaration_end = self._empty_parentheses_end(i, end)
                if declaration_end is not None:
                    i = declaration_end
                continue

            if function_body_expected:
                function_body_expected = False
                if raw == "{":
                    command_expected = True
                    continue

            if case_stack and case_stack[-1] == "word":
                case_stack[-1] = "await-in"
                command_expected = False
                continue
            if case_stack and case_stack[-1] == "await-in" and raw == "in":
                case_stack[-1] = "patterns"
                command_expected = False
                continue
            if case_stack and case_stack[-1] == "patterns":
                if raw == "esac":
                    case_stack.pop()
                command_expected = False
                continue

            if not command_expected:
                if raw in {"then", "do", "else", "elif"}:
                    command_expected = True
                elif raw == "esac" and case_stack:
                    case_stack.pop()
                else:
                    replacement = self._windows_path_replacement(raw)
                    if replacement is not None:
                        self.edits.append((word_start, word_end, replacement))
                        self.path_notes.append(raw)
                    if (
                        _ASSIGNMENT_RE.match(raw)
                        and i < end
                        and s[i] == "("
                    ):
                        # Array literal as a declaration-builtin argument
                        # (``declare -a arr=( ...)``): its elements are data
                        # words, scanned like the command-position form.
                        close = self._find_matching(i + 1, end, ")")
                        self._scan_array_words(
                            i + 1, close if close < end else end
                        )
                        i = close + 1 if close < end else end
                continue

            if raw == "function":
                function_name_expected = True
                command_expected = True
                continue
            declaration_end = self._function_declaration_end(raw, i, end)
            if declaration_end is not None:
                i = declaration_end
                function_body_expected = True
                command_expected = False
                continue
            if raw in _COMMAND_START_KEYWORDS:
                command_expected = True
                continue
            if raw in _COMMAND_END_KEYWORDS:
                if raw == "esac" and case_stack:
                    case_stack.pop()
                command_expected = False
                continue
            if raw in _LIST_KEYWORDS:
                if raw == "case":
                    case_stack.append("word")
                command_expected = False
                continue
            if _ASSIGNMENT_RE.match(raw):
                if i < end and s[i] == "(":
                    close = self._find_matching(i + 1, end, ")")
                    self._scan_array_words(i + 1, close if close < end else end)
                    i = close + 1 if close < end else end
                command_expected = True
                continue

            if raw == "cd":
                self._drop_cmd_cd_flag(i, end)

            executable_wrapper = (
                wrapper is not None and wrapper.kind not in {"coproc", "time"}
            )
            if wrapper is not None and wrapper.kind == "coproc":
                if self._coproc_name_before_compound(raw, i, end):
                    wrapper = None
                    command_expected = True
                    continue
            inline_consumed = False
            if wrapper is not None and wrapper.kind in _WRAPPER_PATH_OPTIONS:
                for option in _WRAPPER_PATH_OPTION_LONG:
                    if raw.startswith(option + "="):
                        value = raw[len(option) + 1 :]
                        replacement = self._windows_path_replacement(value)
                        if replacement is not None:
                            self.edits.append(
                                (word_start, word_end, option + "=" + replacement)
                            )
                            self.path_notes.append(raw)
                        # An inline option also fills a pending value slot
                        # (``env -C --chdir=D:\\x``) and leaves the wrapper
                        # itself active for the command that follows it.
                        wrapper.skip_next = False
                        wrapper.path_value = False
                        command_expected = True
                        inline_consumed = True
                        break
            if inline_consumed:
                continue
            if wrapper is not None:
                path_option_value = wrapper.path_value and wrapper.skip_next
                action = self._consume_wrapper_word(wrapper, raw)
                if action == "skip":
                    if path_option_value:
                        replacement = self._windows_path_replacement(raw)
                        if replacement is not None:
                            self.edits.append((word_start, word_end, replacement))
                            self.path_notes.append(raw)
                    command_expected = True
                    continue
                if action == "inspect":
                    command_expected = False
                    wrapper = None
                    continue

            if raw in _COMMAND_WRAPPERS:
                wrapper = _Wrapper(raw)
                command_expected = True
                continue

            fallback_name = self._literal_command_name(raw)
            if fallback_name is not None:
                self.names.append(fallback_name)
                if executable_wrapper:
                    self.edits.append(
                        (word_start, word_end, _wrapper_runner(fallback_name))
                    )
            else:
                # A command word can itself be a Windows executable path
                # (``C:\tools\rg.exe``); Bash quote removal would eat the
                # backslashes and lose the command, so rewrite it like an
                # argument path.
                replacement = self._windows_path_replacement(raw)
                if replacement is not None:
                    self.edits.append((word_start, word_end, replacement))
                    self.path_notes.append(raw)
            command_expected = False
            wrapper = None

    def _read_word(
        self, start: int, end: int, *, scan_substitutions: bool = True
    ) -> int:
        s = self.s
        i = start
        while i < end:
            ch = s[i]
            if ch in _WORD_END_CHARS:
                break
            if ch == "#" and i == start:
                break
            if ch == "\\":
                i += 2 if i + 1 < end else 1
                continue
            if ch == "'":
                i = self._skip_single_quote(i + 1, end)
                continue
            if ch == '"':
                if scan_substitutions:
                    i = self._skip_double_quote(i + 1, end)
                else:
                    i = self._skip_double_quote_for_matching(i + 1, end)
                continue
            if ch == "`":
                close = self._find_backtick_end(i + 1, end)
                if scan_substitutions:
                    self._scan_range(i + 1, close)
                i = close + 1 if close < end else end
                continue
            if ch == "$":
                if s.startswith("$((", i):
                    i = self._skip_arithmetic(i + 3, end)
                    continue
                if s.startswith("$(", i):
                    close = self._find_matching(i + 2, end, ")")
                    if scan_substitutions:
                        self._scan_range(i + 2, close if close < end else end)
                    i = close + 1 if close < end else end
                    continue
                if s.startswith("${", i):
                    if scan_substitutions:
                        i = self._skip_parameter(i + 2, end)
                    else:
                        i = self._skip_parameter_literal(i + 2, end)
                    continue
                if s.startswith("$'", i):
                    i = self._skip_ansi_quote(i + 2, end)
                    continue
            i += 1
        return i

    def _skip_single_quote(self, i: int, end: int) -> int:
        close = self.s.find("'", i, end)
        return end if close < 0 else close + 1

    def _skip_ansi_quote(self, i: int, end: int) -> int:
        s = self.s
        while i < end:
            if s[i] == "\\":
                i += 2 if i + 1 < end else 1
            elif s[i] == "'":
                return i + 1
            else:
                i += 1
        return end

    def _skip_double_quote(self, i: int, end: int) -> int:
        s = self.s
        while i < end:
            ch = s[i]
            if ch == "\\" and i + 1 < end and s[i + 1] in '$`"\\\n':
                i += 2
            elif ch == '"':
                return i + 1
            elif ch == "`":
                close = self._find_backtick_end(i + 1, end)
                self._scan_range(i + 1, close)
                i = close + 1 if close < end else end
            elif ch == "$" and s.startswith("$(", i) and not s.startswith("$((", i):
                close = self._find_matching(i + 2, end, ")")
                self._scan_range(i + 2, close if close < end else end)
                i = close + 1 if close < end else end
            elif ch == "$" and s.startswith("${", i):
                i = self._skip_parameter(i + 2, end)
            else:
                i += 1
        return end

    def _scan_array_words(self, i: int, end: int) -> None:
        """Scan array literal elements as data words.

        Elements are data, not commands: substitutions inside them are
        scanned as their own command contexts (``_read_word`` handles
        ``$( ... )`` and backquotes), and unquoted words get the same
        Windows path rewrite as ordinary arguments — Bash quote removal
        would otherwise eat their backslashes (``arr=(D:\\x\\y)`` would
        store ``D:xy``).
        """
        s = self.s
        while i < end:
            ch = s[i]
            if ch in " \t\r\n":
                i += 1
                continue
            if ch == "\\" and i + 1 < end and s[i + 1] == "\n":
                i += 2
                continue
            if ch == "#" and self._comment_starts(i, 0):
                newline = s.find("\n", i + 1, end)
                i = end if newline < 0 else newline
                continue
            word_end = self._read_word(i, end)
            if word_end <= i:
                i += 1
                continue
            raw = s[i:word_end]
            replacement = self._windows_path_replacement(raw)
            if replacement is not None:
                self.edits.append((i, word_end, replacement))
                self.path_notes.append(raw)
            i = word_end

    def _scan_expansions(self, i: int, end: int) -> None:
        """Scan executable substitutions in a region whose plain words are data."""
        s = self.s
        while i < end:
            ch = s[i]
            if ch == "\\":
                i += 2 if i + 1 < end else 1
            elif ch == "$" and s.startswith("$'", i):
                i = self._skip_ansi_quote(i + 2, end)
            elif ch == "'":
                i = self._skip_single_quote(i + 1, end)
            elif ch == '"':
                i = self._skip_double_quote(i + 1, end)
            elif ch == "`":
                close = self._find_backtick_end(i + 1, end)
                self._scan_range(i + 1, close)
                i = close + 1 if close < end else end
            elif ch == "$" and s.startswith("$(", i) and not s.startswith("$((", i):
                close = self._find_matching(i + 2, end, ")")
                self._scan_range(i + 2, close if close < end else end)
                i = close + 1 if close < end else end
            elif ch == "$" and s.startswith("$((", i):
                i = self._skip_arithmetic(i + 3, end)
            elif ch == "$" and s.startswith("${", i):
                i = self._skip_parameter(i + 2, end)
            else:
                i += 1

    def _scan_heredoc_expansions(self, i: int, end: int) -> None:
        """Scan substitutions in an expanding heredoc body.

        Quote characters are literal in heredoc bodies; only a backslash can
        suppress the expansion introducers that Bash recognizes there.
        """
        s = self.s
        while i < end:
            ch = s[i]
            if ch == "\\":
                i += 2 if i + 1 < end else 1
            elif ch == "`":
                close = self._find_backtick_end(i + 1, end)
                self._scan_range(i + 1, close)
                i = close + 1 if close < end else end
            elif ch == "$" and s.startswith("$(", i) and not s.startswith("$((", i):
                close = self._find_matching(i + 2, end, ")")
                self._scan_range(i + 2, close if close < end else end)
                i = close + 1 if close < end else end
            elif ch == "$" and s.startswith("$((", i):
                i = self._skip_arithmetic(i + 3, end)
            elif ch == "$" and s.startswith("${", i):
                i = self._skip_parameter(i + 2, end)
            else:
                i += 1

    def _skip_conditional(self, i: int, end: int) -> int:
        """Skip a ``[[ ... ]]`` expression while scanning its substitutions."""
        s = self.s
        while i < end:
            ch = s[i]
            if ch == "]" and s.startswith("]]", i):
                return i + 2
            if ch == "\\":
                i += 2 if i + 1 < end else 1
            elif ch == "$" and s.startswith("$'", i):
                i = self._skip_ansi_quote(i + 2, end)
            elif ch == "'":
                i = self._skip_single_quote(i + 1, end)
            elif ch == '"':
                i = self._skip_double_quote(i + 1, end)
            elif ch == "`":
                close = self._find_backtick_end(i + 1, end)
                self._scan_range(i + 1, close)
                i = close + 1 if close < end else end
            elif ch == "$" and s.startswith("$(", i) and not s.startswith("$((", i):
                close = self._find_matching(i + 2, end, ")")
                self._scan_range(i + 2, close if close < end else end)
                i = close + 1 if close < end else end
            elif ch == "$" and s.startswith("$((", i):
                i = self._skip_arithmetic(i + 3, end)
            else:
                i += 1
        return end

    def _skip_parameter_literal(self, i: int, end: int) -> int:
        s = self.s
        depth = 1
        while i < end:
            if s[i] == "\\":
                i += 2 if i + 1 < end else 1
            elif s[i] == "'":
                i = self._skip_single_quote(i + 1, end)
            elif s[i] == '"':
                i = self._skip_double_quote_for_matching(i + 1, end)
            elif s[i] == "{":
                depth += 1
                i += 1
            elif s[i] == "}":
                depth -= 1
                i += 1
                if depth == 0:
                    return i
            else:
                i += 1
        return end

    def _skip_parameter(self, i: int, end: int) -> int:
        s = self.s
        depth = 1
        while i < end:
            ch = s[i]
            if ch == "\\":
                i += 2 if i + 1 < end else 1
            elif ch == "$" and s.startswith("$(", i) and not s.startswith("$((", i):
                close = self._find_matching(i + 2, end, ")")
                self._scan_range(i + 2, close if close < end else end)
                i = close + 1 if close < end else end
            elif ch == "'":
                i = self._skip_single_quote(i + 1, end)
            elif ch == '"':
                i = self._skip_double_quote(i + 1, end)
            elif ch == "{":
                depth += 1
                i += 1
            elif ch == "}":
                depth -= 1
                i += 1
                if depth == 0:
                    return i
            else:
                i += 1
        return end

    def _skip_arithmetic(self, i: int, end: int) -> int:
        s = self.s
        depth = 1
        while i < end:
            ch = s[i]
            if ch == "$" and s.startswith("$(", i) and not s.startswith("$((", i):
                close = self._find_matching(i + 2, end, ")")
                self._scan_range(i + 2, close if close < end else end)
                i = close + 1 if close < end else end
            elif ch == "(" and s.startswith("((", i):
                depth += 1
                i += 2
            elif ch == ")" and s.startswith("))", i):
                depth -= 1
                i += 2
                if depth == 0:
                    return i
            elif ch == "\\":
                i += 2 if i + 1 < end else 1
            elif ch == "'":
                i = self._skip_single_quote(i + 1, end)
            elif ch == '"':
                i = self._skip_double_quote(i + 1, end)
            else:
                i += 1
        return end

    def _find_backtick_end(self, i: int, end: int) -> int:
        s = self.s
        while i < end:
            if s[i] == "\\":
                i += 2 if i + 1 < end else 1
            elif s[i] == "`":
                return i
            else:
                i += 1
        return end

    def _find_matching(self, i: int, end: int, closing: str) -> int:
        """Find the position of the bracket matching the one at ``i - 2``.

        Recursion is bounded by :data:`_MAX_NESTING_DEPTH`; beyond it the
        region is treated as unmatched so the caller skips it for Bash.
        """
        if self.nest_depth >= _MAX_NESTING_DEPTH:
            return end
        self.nest_depth += 1
        try:
            return self._find_matching_inner(i, end, closing)
        finally:
            self.nest_depth -= 1

    def _find_matching_inner(self, i: int, end: int, closing: str) -> int:
        s = self.s
        depth = 0
        pending_heredocs: list[_HereDoc] = []
        case_stack: list[str] = []
        while i < end:
            ch = s[i]
            if ch == "\\":
                i += 2 if i + 1 < end else 1
            elif ch == "\n":
                i += 1
                if pending_heredocs:
                    i = self._skip_heredoc_bodies(
                        i, end, pending_heredocs, scan_expansions=False
                    )
                    pending_heredocs.clear()
            elif ch == "$" and s.startswith("$((", i):
                i = self._skip_arithmetic(i + 3, end)
                if case_stack and case_stack[-1] == "word":
                    # Arithmetic expansion can be the case subject word
                    # (``case $((...)) in ...``); it ends the header just
                    # like a plain subject word.
                    case_stack[-1] = "await-in"
            elif ch == "<" and s.startswith("<<", i) and not s.startswith("<<<", i):
                strip_tabs = s.startswith("<<-", i)
                delimiter_start = i + (3 if strip_tabs else 2)
                while delimiter_start < end and s[delimiter_start] in " \t\r":
                    delimiter_start += 1
                delimiter_end = self._read_word(
                    delimiter_start, end, scan_substitutions=False
                )
                heredoc = self._heredoc_delimiter(s[delimiter_start:delimiter_end])
                if heredoc is not None:
                    delimiter, expands = heredoc
                    pending_heredocs.append(_HereDoc(delimiter, strip_tabs, expands))
                i = delimiter_end if delimiter_end > delimiter_start else delimiter_start
            elif ch == "'":
                i = self._skip_single_quote(i + 1, end)
            elif ch == '"':
                i = self._skip_double_quote_for_matching(i + 1, end)
            elif ch == "`":
                close = self._find_backtick_end(i + 1, end)
                i = close + 1 if close < end else end
                if case_stack and case_stack[-1] == "word":
                    # A backquote substitution can be the case subject word
                    # (``case `...` in ...``); it ends the header just like
                    # a plain subject word.
                    case_stack[-1] = "await-in"
            elif ch == "#" and self._comment_starts(i, 0):
                newline = s.find("\n", i + 1, end)
                i = end if newline < 0 else newline
            elif ch == ";" and s.startswith(";;&", i):
                if case_stack:
                    case_stack[-1] = "patterns"
                i += 3
            elif ch == ";" and (s.startswith(";;", i) or s.startswith(";&", i)):
                if case_stack:
                    case_stack[-1] = "patterns"
                i += 2
            elif ch not in _WORD_END_CHARS:
                word_end = self._read_word(i, end, scan_substitutions=False)
                if word_end <= i:
                    i += 1
                    continue
                word = s[i:word_end]
                if word == "case":
                    case_stack.append("word")
                elif (
                    case_stack
                    and case_stack[-1] in {"patterns", "body"}
                    and word == "esac"
                ):
                    case_stack.pop()
                elif case_stack and case_stack[-1] == "word":
                    case_stack[-1] = "await-in"
                elif case_stack and case_stack[-1] == "await-in" and word == "in":
                    case_stack[-1] = "patterns"
                i = word_end
            elif ch == "(":
                depth += 1
                i += 1
            elif ch == closing:
                if case_stack and case_stack[-1] == "patterns":
                    case_stack[-1] = "body"
                    i += 1
                elif depth == 0:
                    return i
                else:
                    depth -= 1
                    i += 1
            else:
                i += 1
        return end

    def _skip_double_quote_for_matching(self, i: int, end: int) -> int:
        s = self.s
        while i < end:
            ch = s[i]
            if ch == "\\" and i + 1 < end and s[i + 1] in '$`"\\\n':
                i += 2
            elif ch == '"':
                return i + 1
            elif ch == "`":
                close = self._find_backtick_end(i + 1, end)
                i = close + 1 if close < end else end
            elif ch == "$" and s.startswith("$(", i) and not s.startswith("$((", i):
                close = self._find_matching(i + 2, end, ")")
                i = close + 1 if close < end else end
            else:
                i += 1
        return end

    def _read_control_operator(self, i: int, end: int) -> tuple[str, int]:
        s = self.s
        ch = s[i]
        if ch == ";":
            if s.startswith(";;&", i):
                return ";;&", i + 3
            if s.startswith(";;", i):
                return ";;", i + 2
            if s.startswith(";&", i):
                return ";&", i + 2
            return ";", i + 1
        if ch == "&":
            if s.startswith("&&", i):
                return "&&", i + 2
            return "&", i + 1
        if ch == "|":
            if s.startswith("||", i):
                return "||", i + 2
            if s.startswith("|&", i):
                return "|&", i + 2
            return "|", i + 1
        if ch in "()":
            return ch, i + 1
        return "", i

    def _read_redirection(self, i: int, end: int) -> tuple[str, int]:
        s = self.s
        for op in ("&>>", "&>", "<<<", "<<-", "<<", ">>", "<>", ">|", "<&", ">&", "<", ">"):
            if s.startswith(op, i):
                return op, i + len(op)
        return "", i

    def _redirection_after_fd(self, i: int, end: int) -> bool:
        s = self.s
        while i < end and s[i].isdigit():
            i += 1
        return i < end and s[i] in _REDIRECTION_START

    def _comment_starts(self, i: int, range_start: int) -> bool:
        if i <= range_start:
            return True
        return self.s[i - 1] in " \t\r\n;&|()<>"

    def _empty_parentheses_end(self, i: int, end: int) -> int | None:
        s = self.s
        while i < end and s[i] in " \t\r":
            i += 1
        if i >= end or s[i] != "(":
            return None
        i += 1
        while i < end and s[i] in " \t\r":
            i += 1
        return i + 1 if i < end and s[i] == ")" else None

    def _function_declaration_end(self, raw: str, i: int, end: int) -> int | None:
        if not _NAME_RE.fullmatch(raw):
            return None
        return self._empty_parentheses_end(i, end)

    def _consume_wrapper_word(self, wrapper: _Wrapper, raw: str) -> str:
        if wrapper.skip_next:
            wrapper.skip_next = False
            if wrapper.opaque:
                return "inspect"
            return "skip"
        if wrapper.opaque:
            return "inspect"
        if wrapper.kind == "command" and raw in {"-v", "-V"}:
            return "inspect"
        if wrapper.kind == "command" and (
            raw == "-p"
            or (
                raw.startswith("-")
                and not raw.startswith("--")
                and "p" in raw[1:]
            )
        ):
            wrapper.opaque = True
            return "skip"
        if wrapper.kind == "env" and raw in {"-S", "--split-string"}:
            wrapper.opaque = True
            wrapper.skip_next = True
            return "skip"
        if wrapper.kind == "env" and (
            raw.startswith("--split-string=")
            or (raw.startswith("-S") and raw != "-S")
        ):
            return "inspect"
        if raw == "--":
            return "skip"
        if raw in _WRAPPER_OPTIONS_WITH_VALUE.get(wrapper.kind, ()):
            wrapper.skip_next = True
            if raw in _WRAPPER_PATH_OPTIONS.get(wrapper.kind, ()):
                wrapper.path_value = True
            return "skip"
        if raw.startswith("-"):
            return "skip"
        if wrapper.kind == "env" and _ASSIGNMENT_RE.match(raw):
            return "skip"
        return "command"

    def _coproc_name_before_compound(self, raw: str, i: int, end: int) -> bool:
        if not _NAME_RE.fullmatch(raw):
            return False
        while i < end and self.s[i] in " \t\r":
            i += 1
        if i >= end:
            return False
        if self.s.startswith(("{", "(", "[[", "(("), i):
            return True
        for keyword in ("case", "for", "if", "select", "until", "while"):
            keyword_end = i + len(keyword)
            if self.s.startswith(keyword, i) and (
                keyword_end >= end
                or self.s[keyword_end] in " \t\r\n;&|()<>{}"
            ):
                return True
        return False

    def _heredoc_delimiter(self, raw: str) -> tuple[str | None, bool] | None:
        if not raw:
            return None
        result: list[str] = []
        quoted = False
        matchable = True
        i = 0
        while i < len(raw):
            ch = raw[i]
            if raw.startswith("$'", i):
                quoted = True
                value, i, valid = self._read_ansi_c_delimiter(raw, i + 2)
                result.append(value)
                matchable = matchable and valid
            elif ch == "'":
                quoted = True
                close = raw.find("'", i + 1)
                if close < 0:
                    result.append(raw[i + 1 :])
                    i = len(raw)
                else:
                    result.append(raw[i + 1 : close])
                    i = close + 1
            elif ch == '"':
                quoted = True
                i += 1
                while i < len(raw) and raw[i] != '"':
                    if raw[i] == "\\" and i + 1 < len(raw):
                        escaped = raw[i + 1]
                        if escaped in '$`"\\\n':
                            if escaped != "\n":
                                result.append(escaped)
                            i += 2
                            continue
                    result.append(raw[i])
                    i += 1
                if i < len(raw):
                    i += 1
            elif ch == "\\" and i + 1 < len(raw):
                quoted = True
                result.append(raw[i + 1])
                i += 2
            else:
                result.append(ch)
                i += 1
        delimiter = "".join(result) if matchable else None
        return delimiter, not quoted

    def _read_ansi_c_delimiter(
        self, raw: str, i: int
    ) -> tuple[str, int, bool]:
        result: list[str] = []
        valid = True
        simple = {
            "a": "\a",
            "b": "\b",
            "e": "\x1b",
            "E": "\x1b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
            "v": "\v",
            "\\": "\\",
            "'": "'",
            '"': '"',
            "?": "?",
        }
        while i < len(raw):
            if raw[i] == "'":
                return "".join(result), i + 1, valid
            if raw[i] != "\\" or i + 1 >= len(raw):
                result.append(raw[i])
                i += 1
                continue
            escape = raw[i + 1]
            if escape in simple:
                result.append(simple[escape])
                i += 2
                continue
            if escape in "01234567":
                j = i + 1
                while j < len(raw) and j < i + 4 and raw[j] in "01234567":
                    j += 1
                result.append(chr(int(raw[i + 1 : j], 8)))
                i = j
                continue
            if escape in "xXuU":
                widths = {"x": 2, "X": 2, "u": 4, "U": 8}
                j = i + 2
                limit = min(len(raw), j + widths[escape])
                while j < limit and raw[j] in "0123456789abcdefABCDEF":
                    j += 1
                if j > i + 2:
                    value = int(raw[i + 2 : j], 16)
                    if value <= 0x10FFFF and not 0xD800 <= value <= 0xDFFF:
                        result.append(chr(value))
                    else:
                        # Bash accepts byte sequences outside Python's Unicode
                        # scalar range.  Python text cannot represent the same
                        # delimiter, so mark it unmatchable and conservatively
                        # keep all remaining source inside the heredoc.
                        valid = False
                        result.append(raw[i:j])
                    i = j
                    continue
            result.append("\\" + escape)
            i += 2
        return "".join(result), i, valid

    def _skip_heredoc_bodies(
        self,
        i: int,
        end: int,
        documents: list[_HereDoc],
        *,
        scan_expansions: bool = True,
    ) -> int:
        s = self.s
        for document in documents:
            body_start = i
            logical_line = ""
            logical_start = i
            while i < end:
                newline = s.find("\n", i, end)
                line_end = end if newline < 0 else newline
                line = s[i:line_end]
                compare = line.lstrip("\t") if document.strip_tabs else line
                if not logical_line:
                    logical_start = i
                if document.expands and self._heredoc_line_continues(compare):
                    logical_line += compare[:-1]
                    i = end if newline < 0 else newline + 1
                    continue
                logical_line += compare
                if (
                    document.delimiter is not None
                    and logical_line == document.delimiter
                ):
                    if scan_expansions and document.expands:
                        self._scan_heredoc_expansions(body_start, logical_start)
                    i = end if newline < 0 else newline + 1
                    break
                logical_line = ""
                i = end if newline < 0 else newline + 1
            else:
                if scan_expansions and document.expands:
                    self._scan_heredoc_expansions(body_start, end)
        return i

    @staticmethod
    def _heredoc_line_continues(line: str) -> bool:
        trailing = len(line) - len(line.rstrip("\\"))
        return trailing % 2 == 1

    # ── Windows path normalization for Git Bash ─────────────────────────────

    def _drop_cmd_cd_flag(self, i: int, end: int) -> None:
        """Drop the cmd.exe-only ``cd /d <path>`` flag form.

        Bash ``cd`` accepts a single argument, so ``cd /d D:\\x`` fails with
        "too many arguments".  The flag is deleted only when a path argument
        actually follows it on the same line; bare ``cd /d`` stays untouched.
        """
        s = self.s
        j = i
        while j < end and s[j] in " \t\r":
            j += 1
        if j >= end:
            return
        flag_end = self._read_word(j, end, scan_substitutions=False)
        if flag_end <= j or s[j:flag_end] not in {"/d", "/D"}:
            return
        k = flag_end
        while k < end and s[k] in " \t\r":
            k += 1
        if k >= end or s[k] in _OPERATOR_CHARS or s[k] == "#":
            return
        self.edits.append((j, flag_end, ""))
        self.path_notes.append("cd /d")

    def _windows_path_replacement(self, raw: str) -> str | None:
        """Return the Git Bash spelling of a Windows backslash path word.

        Only unquoted words are considered: quoted text is literal data that
        may carry regexes or tool-level escape sequences.  The word must look
        unambiguously like a Windows path (drive letter, UNC share, root- or
        home-relative, dot-relative, or a multi-segment relative path); short
        ambiguous words such as ``a\nb`` and ``foo\bar`` are left for Bash
        to handle.  Words spanning a backslash-newline line continuation are
        also left untouched: injecting the line break into a rewritten word
        would change the command's line structure.
        """
        if not raw or "\\" not in raw:
            return None
        backslashes = 0
        for ch in raw:
            if ch == "\\":
                backslashes += 1
            elif ch in "'\"`$\n\r":
                return None
        if _PATH_DRIVE_RE.fullmatch(raw):
            pass
        elif raw.startswith("\\\\") and len(raw) > 2:
            pass
        elif raw.startswith("\\") and not raw.startswith("\\\\") and backslashes >= 2:
            # Root-relative paths are not anchored like ``D:\...``: an unquoted
            # word such as ``\a\b`` or ``\033\015`` is far more likely to be a
            # Bash escape sequence than a path, so the segments must look like
            # real directory names before the rewrite happens.
            if not self._plausible_path_segments(raw):
                return None
        elif raw.startswith("~\\"):
            pass
        elif raw.startswith(".\\") or raw.startswith("..\\"):
            pass
        elif backslashes >= 2:
            decoded = self._decode_unquoted_word(raw)
            if (
                len(decoded) < 2
                or not any(ch.isalnum() for ch in decoded)
                or not _PATH_SEGMENT_RE.fullmatch(decoded)
                or not self._plausible_path_segments(raw)
            ):
                return None
        else:
            return None
        return self._quote_path_word(self._normalize_windows_path(raw))

    @staticmethod
    def _plausible_path_segments(raw: str) -> bool:
        """Require at least one segment that looks like a real directory name.

        Bash escape sequences are written with single-letter backslash escapes
        (``\\a``, ``\\n``, ``\\t``, ``\\x``) or pure-digit octal/hex bodies
        (``\\033``, ``\\015``), so words built only from one-character or
        digit-led segments (``\\a\\b``, ``\\033\\015``, ``x\\n\\t``) stay
        ambiguous and are preserved byte-for-byte.  A segment of at least two
        characters starting with a letter (``Users``, ``build``, ``Program``)
        marks the word as a genuine path.
        """
        return any(
            len(segment) >= 2 and segment[0].isalpha()
            for segment in raw.split("\\")
        )

    @staticmethod
    def _decode_unquoted_word(raw: str) -> str:
        """Return the word value after Bash quote removal (unquoted form)."""
        value: list[str] = []
        i = 0
        while i < len(raw):
            ch = raw[i]
            if ch == "\\" and i + 1 < len(raw):
                value.append(raw[i + 1])
                i += 2
            else:
                value.append(ch)
                i += 1
        return "".join(value)

    @staticmethod
    def _normalize_windows_path(raw: str) -> str:
        """Rewrite backslashes as the forward slashes Git Bash understands.

        A leading ``\\\\`` UNC prefix becomes ``//``; a backslash before a
        char from :data:`_ESCAPED_LITERAL_CHARS` is a pure Bash escape (the
        char belongs inside its segment, e.g. ``\\ `` is a space); every other
        backslash separates segments and becomes ``/``.
        """
        out: list[str] = []
        i = 0
        n = len(raw)
        if n >= 2 and raw.startswith("\\\\"):
            out.append("//")
            i = 2
        while i < n:
            ch = raw[i]
            if ch == "\\" and i + 1 < n:
                nxt = raw[i + 1]
                if nxt == "\\":
                    out.append("/")
                elif nxt in _ESCAPED_LITERAL_CHARS:
                    out.append(nxt)
                else:
                    out.append("/")
                    out.append(nxt)
                i += 2
            elif ch == "\\":
                out.append("/")
                i += 1
            else:
                out.append(ch)
                i += 1
        return "".join(out)

    def _quote_path_word(self, normalized: str) -> str:
        """Quote a normalized path only when unquoted emission would break it.

        Safe characters (including glob metacharacters, so ``D:/x/*.txt``
        keeps performing pathname expansion) pass through untouched.  A
        leading ``~`` stays outside the quotes so tilde expansion still
        applies to it.
        """
        if all(ch in _PATH_SAFE_CHARS for ch in normalized):
            return normalized
        if normalized.startswith("~"):
            return "~" + self._quote_path_word(normalized[1:])
        escaped = (
            normalized.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("$", "\\$")
            .replace("`", "\\`")
        )
        return '"' + escaped + '"'


def bash_compatibility_prelude() -> str:
    """Return exported fallback definitions for a persistent Git Bash shell.

    Interactive input can be an incomplete Bash fragment (for example a
    heredoc body or the second half of a quote), so it must never be scanned
    and prefixed independently.  The interactive shell instead executes this
    prelude once and exports the fallback functions across ``exec bash -i``.
    """
    if sys.platform != "win32":
        return ""
    definitions = "\n".join(_FALLBACKS.values())
    exports = "\n".join(
        f"if declare -F {name} >/dev/null; then export -f {name}; fi"
        for name in _FALLBACKS
    )
    return definitions + "\n" + exports


def fix_bash_command(command: str) -> BashFix:
    """Rewrite selected native POSIX commands for Windows Git Bash.

    Non-Windows input is always returned byte-for-byte unchanged.  On Windows,
    only literal command words with verified equivalents are changed; unknown
    or semantically ambiguous commands are left for Bash to handle normally.
    """
    if sys.platform != "win32" or not command:
        return BashFix(command)
    # Native acceleration: kimix_native.parse.fix_bash_command.
    if _native_use_native("PARSE") and _NATIVE_PARSE is not None:
        result = _NATIVE_PARSE.fix_bash_command(command)
        return BashFix(
            command=result.command,
            replacements=tuple(result.replacements),
            path_changes=tuple(result.path_changes),
        )
    # Quoting and escaping can form a literal command name without the source
    # containing it contiguously (for example ``r""ev`` or ``\rev``), so a
    # substring fast path would miss legal executable words.  The scanner is
    # linear and exits without allocating generated shell code when unchanged.
    return _Scanner(command).fix()
