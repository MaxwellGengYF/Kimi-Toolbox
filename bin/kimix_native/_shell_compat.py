"""Vendored pure-Python mirrors of the kimi-agent bash/pwsh scanners.

Copied (with attribution) from the kimi-agent repo's src/kimix/tools/file/bash/
so the shim has a bit-exact fallback when the reference checkout is not
importable. Only the self-contained scanner code is vendored (no tool I/O).

Reference files: bash_fix.py (the _Scanner + fallback data), bash_tool.py
(_process_unquoted + helpers, lines 440-756), pwsh_fix.py, process_pwsh.py.
"""

from __future__ import annotations

# ======================================================================
# bash_fix.py (kimi-agent) - Windows Git Bash compatibility scanner
# ======================================================================
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable

try:
    import regex as re
except ImportError:  # pragma: no cover
    import re  # stdlib: the used patterns are all stdlib-compatible

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
        "while true; do clear; eval \"$*\"; sleep \"$__kimix_interval\"; done"
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
    {"command", "coproc", "env", "exec", "nohup", "sudo", "time",
     "timeout", "stdbuf", "nice", "xargs"}
)

# Fallback names that behave like command wrappers too: the word itself needs
# its fallback definition recorded AND its first non-option operand is a
# command the wrapper executes.  ``gtimeout`` runs ``timeout "$@"`` (an
# executable that execs argv, so its command operand needs the standalone
# runner), while ``watch`` runs its command inside the same shell (its body
# uses ``eval "$*"``, so a same-shell function call suffices).
_FALLBACK_COMMAND_WRAPPERS = {"gtimeout": "timeout", "watch": "watch"}

# Wrappers that require a fixed number of plain operands (options excluded)
# before the command word.  GNU ``timeout`` takes exactly one DURATION operand
# (``timeout 5 rev``, ``timeout -s KILL 5s rev``) before COMMAND, so the
# duration is consumed without being mistaken for the wrapped command.
_WRAPPER_OPERAND_COUNTS = {"timeout": 1}

# Wrapper kinds whose command operand runs inside the current shell rather
# than being exec'd as a new process: the fallback function defined by the
# prefix is directly callable, so the source word is kept (no standalone
# runner rewrite is needed or possible).
_SAME_SHELL_WRAPPERS = frozenset({"coproc", "time", "watch"})

# Shell executables that agents frequently put in front of a command the Bash
# tool is already going to run (``bash cd /c/dev/x && ...``).  Git Bash can
# stand in for these POSIX-family shells, so a redundant leading invocation is
# unwrapped instead of letting bash try to open ``cd`` as a script file.  Only
# shells whose syntax is a subset of bash are listed; ``zsh``/``ksh``/``csh``
# scripts may use incompatible syntax and are deliberately left untouched.
_SHELL_WRAPPERS = frozenset({"bash", "sh", "dash", "ash"})

# Short-option clusters of the shell wrappers that take the next argument as
# an inline command string.  ``bash -c '...'`` is the canonical form;
# ``-lc``/``-cl`` add the (harmless here) login flag.  Clusters such as
# ``-ec``/``-ce``/``-xc`` that change errexit/xtrace semantics are left for
# bash to handle so unwrapping cannot alter the script's meaning.
_SHELL_C_OPTIONS = re.compile(r"^-c$|^-lc$|^-cl$")
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
    "timeout": frozenset(
        {
            "-k",
            "--kill-after",
            "-s",
            "--signal",
        }
    ),
    "stdbuf": frozenset(
        {
            "-o",
            "-e",
            "-i",
            "--output",
            "--error",
            "--input",
        }
    ),
    "nice": frozenset({"-n", "--adjustment"}),
    "xargs": frozenset(
        {
            "-I",
            "-n",
            "-L",
            "-P",
            "-s",
            "-S",
            "-a",
            "-d",
            "-E",
            "--arg-file",
            "--max-args",
            "--max-chars",
            "--max-procs",
            "--max-lines",
            "--replace",
            "--eof",
            "--delimiter",
        }
    ),
    "watch": frozenset({"-n", "--interval"}),
}

# Wrapper options whose value is a filesystem path rather than a name or
# number.  Windows backslash spellings of these values are rewritten for Git
# Bash just like ordinary argument paths; every other option value stays
# opaque.
_WRAPPER_PATH_OPTIONS = {
    "env": frozenset({"-C", "--chdir"}),
    "sudo": frozenset({"-D", "--chdir"}),
    "time": frozenset({"-o", "--output"}),
    "xargs": frozenset({"-a", "--arg-file"}),
}
_WRAPPER_PATH_OPTION_LONG = frozenset({"--chdir", "--output", "--arg-file"})

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


def _windows_temp_dir() -> str:
    """Return the real Windows temp directory as a forward-slash path.

    Git Bash's ``/tmp`` maps to this directory (``cygpath -w /tmp`` reports
    the same value), but Git Bash's ``TMP``/``TEMP`` environment variables are
    MSYS-style ``/tmp`` inside the shell, so they cannot be used to translate
    ``/tmp`` paths for native Windows executables.  The parent (kimi-agent)
    process environment holds the Windows-style spelling, which is what
    ``tempfile.gettempdir()`` returns.  On POSIX hosts this returns ``/tmp``
    (the identity for the rewrite), which keeps the scanner's platform gate
    the single source of truth for when the rewrite applies.
    """
    return tempfile.gettempdir().replace("\\", "/")


@dataclass(frozen=True)
class BashFix:
    """Result of :func:`fix_bash_command`.

    ``replacements`` records each original command name in source order and
    ``path_changes`` each original argument or command word whose
    Windows-style backslashes, Git Bash virtual absolute path (``/tmp/x``,
    ``/c/x``), or cmd.exe ``/d`` flag were rewritten for Git Bash.
    ``shell_wrappers`` records each redundant ``bash``/``sh``
    invocation that was unwrapped (``bash <cmd> ...`` or ``bash -c <script>``)
    so the command runs directly in the Bash tool.  Empty tuples mean the
    command was returned byte-for-byte unchanged.
    """

    command: str
    replacements: tuple[str, ...] = ()
    path_changes: tuple[str, ...] = ()
    shell_wrappers: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        """Return whether any compatibility replacement was made."""
        return (
            bool(self.replacements)
            or bool(self.path_changes)
            or bool(self.shell_wrappers)
        )

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
                f"slashes; Git Bash virtual paths to native spellings): {words}."
            )
        if self.shell_wrappers:
            names = ", ".join(f"`{name}`" for name in self.shell_wrappers)
            parts.append(
                f"Removed redundant shell wrapper(s): {names}."
            )
        return " ".join(parts)


@dataclass
class _BashWrapper:
    kind: str
    skip_next: bool = False
    opaque: bool = False
    path_value: bool = False
    operands: int = 0


@dataclass
class _BashHereDoc:
    delimiter: str | None
    strip_tabs: bool
    expands: bool


class _BashFixScanner:
    """Conservative scanner for Bash executable command positions."""

    __slots__ = ("s", "n", "edits", "names", "path_notes", "shell_notes", "heredoc_events", "nest_depth")

    def __init__(self, command: str) -> None:
        self.s = command
        self.n = len(command)
        self.edits: list[tuple[int, int, str]] = []
        self.names: list[str] = []
        self.path_notes: list[str] = []
        self.shell_notes: list[str] = []
        self.heredoc_events: list[tuple[int, int]] = []
        self.nest_depth = 0

    def fix(self) -> BashFix:
        try:
            self._scan_range(0, self.n)
        except RecursionError:
            # Malformed or adversarial nesting must never make the Bash tool
            # fail before Bash itself can report the syntax error.
            return BashFix(self.s)
        if not self.names and not self.edits and not self.shell_notes:
            return BashFix(self.s)
        unique_names = list(dict.fromkeys(self.names))
        definitions = "\n".join(_FALLBACKS[name] for name in unique_names)
        # Exported fallbacks are inherited by every nested bash (a ``bash -c``
        # operand of a command wrapper, the standalone runner scripts), where
        # the definitions above are not otherwise visible: ``env bash -c
        # 'rev <<< abc'`` keeps ``bash -c`` but the child shell still needs
        # ``rev`` to resolve to the function.
        exports = "\n".join(f"export -f {name}" for name in unique_names)
        source = self._build_source()
        source = _fix_heredoc_trailing_operators(source)
        prefix = definitions + "\n" + exports + "\n" if definitions else ""
        return BashFix(
            prefix + source,
            tuple(self.names),
            tuple(self.path_notes),
            tuple(self.shell_notes),
        )

    def _build_source(self) -> str:
        """Return the source with all recorded edits applied."""
        if not self.edits:
            return self.s
        pieces: list[str] = []
        previous = 0
        for start, end, replacement in sorted(self.edits):
            pieces.extend((self.s[previous:start], replacement))
            previous = end
        pieces.append(self.s[previous:])
        return "".join(pieces)

    @staticmethod
    def _literal_word_value(raw: str) -> str | None:
        """Return the word value produced solely by Bash quote removal.

        Bash permits literal words such as ``'rev'``, ``\rev`` and ``r""ev``.
        Only words whose value can be determined without any expansion are
        accepted; parameter/command/arithmetic expansions, globbing, and
        malformed quotes return ``None`` so the caller leaves them untouched.
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
        return "".join(value)

    @staticmethod
    def _literal_command_name(raw: str) -> str | None:
        """Return the fallback command name produced by Bash quote removal."""
        name = _BashFixScanner._literal_word_value(raw)
        return name if name is not None and name in _FALLBACKS else None

    @staticmethod
    def _shell_wrapper_name(raw: str) -> str | None:
        """Return the shell name when *raw* is a literal ``bash``/``sh`` word."""
        name = _BashFixScanner._literal_word_value(raw)
        return name if name is not None and name in _SHELL_WRAPPERS else None

    @staticmethod
    def _plausible_script_file(raw: str) -> bool:
        """Return True when *raw* looks like a script path, not a command word.

        ``bash cd /c/dev/x`` and ``bash grep ...`` are redundant-wrapper
        mistakes (bash would try to open ``cd``/``grep`` as a script file);
        ``bash scripts/deploy.sh``, ``bash ./tool`` and ``bash app.sh`` are
        legitimate script invocations that must keep the wrapper.  A word with
        a directory separator, a leading ``.`` path segment, or a shell-script
        extension is treated as a script path; a bare command word is not.
        """
        if raw.startswith("./") or raw.startswith("../"):
            return True
        if raw.startswith(".\\") or raw.startswith("..\\"):
            return True
        if "/" in raw or "\\" in raw:
            return True
        lower = raw.lower()
        return any(
            lower.endswith(ext)
            for ext in (".sh", ".bash", ".zsh", ".ksh", ".dash", ".ash", ".bats")
        )

    def _handle_shell_wrapper(
        self,
        shell_name: str,
        word_start: int,
        i: int,
        end: int,
        *,
        assignment_prefix: bool = False,
        wrapped: bool = False,
    ) -> tuple[int, bool] | None:
        """Repair a redundant ``bash``/``sh`` invocation at command position.

        Returns ``(scan index, keep_wrapper)`` when the wrapper was rewritten,
        or ``None`` when the wrapper is left untouched (legitimate
        ``bash script.sh``, semantic options such as ``-e``/``-x``, stdin
        forms, scripts with trailing argv that ``$0``/``$1`` depend on, or an
        assignment prefix such as ``VAR=x bash ...`` whose scoping the rewrite
        could not preserve).  ``keep_wrapper`` asks the caller to leave any
        active command wrapper in place so the wrapped command keeps its
        executable context.

        Two shapes are repaired:
        * ``bash <command> ...`` — the shell word is redundant because the Bash
          tool already runs the whole string via bash, so it is removed
          (``bash cd /c/dev/x && grep ...`` becomes
          ``cd /c/dev/x && grep ...``).
        * ``bash [-l] -c <script>`` — the inline script replaces the wrapper
          (``bash -c 'cd C:\\x && rev'`` becomes ``cd C:/x && rev``) and is
          scanned as its own command context so fallback commands and Windows
          paths inside it are fixed too.  Only the ``-c``/``-lc``/``-cl`` forms
          with the script as the final word of the segment are unwrapped;
          trailing argv, ``-ec``-style clusters and expansions are preserved
          for bash to handle.

        Under an active command wrapper (``env``/``nohup``/``timeout``/``sudo``
        ...) the shell word is an *operand* of that wrapper, so the repair
        must not move text out of its argv (``timeout 5 bash -c 'a && b'``
        unwrapped to ``timeout 5 a && b`` would let ``&&`` split the list):
        the ``-c`` form keeps ``bash -c`` and only fixes the inline script in
        place — the wrapper runs bash natively and the nested bash inherits
        the exported fallback functions — while the prefix form only drops
        the shell word and leaves the wrapper active so the command that
        follows keeps its executable (standalone-runner) context.
        """
        if assignment_prefix:
            # ``VAR=x bash -c 'echo $VAR'``: the assignment is scoped to the
            # shell *process*, so unwrapping would make the outer shell expand
            # ``$VAR`` before the assignment takes effect.  Keep the wrapper.
            return None
        s = self.s
        j = i
        while j < end and s[j] in " \t\r":
            j += 1
        if j >= end or s[j] in _OPERATOR_CHARS or s[j] == "#":
            # Bare ``bash`` (or ``bash && ...`` / ``bash`` at EOF): bash would
            # start a nested shell; that is not a compatibility problem.
            return None
        next_end = self._read_word(j, end, scan_substitutions=False)
        if next_end <= j:
            return None
        next_raw = s[j:next_end]

        if next_raw.startswith("-"):
            # Optional leading login flag, then the ``-c`` family.  Any other
            # option (``-e``, ``-x``, ``-i``, ``--norc``, ...) changes shell
            # behaviour and is left for bash to handle.
            opt_end = next_end
            opt = next_raw
            if opt in ("-l", "-L", "--login"):
                cursor = opt_end
                while cursor < end and s[cursor] in " \t\r":
                    cursor += 1
                if cursor >= end or s[cursor] in _OPERATOR_CHARS or s[cursor] == "#":
                    return None
                opt_end = self._read_word(cursor, end, scan_substitutions=False)
                if opt_end <= cursor:
                    return None
                opt = s[cursor:opt_end]
            if not _SHELL_C_OPTIONS.match(opt):
                return None
            # The word after ``-c`` is the inline script.
            k = opt_end
            while k < end and s[k] in " \t\r":
                k += 1
            if k >= end or s[k] in _OPERATOR_CHARS or s[k] == "#":
                return None  # ``bash -c`` with no script
            script_end = self._read_word(k, end, scan_substitutions=False)
            if script_end <= k:
                return None
            m = script_end
            while m < end and s[m] in " \t\r":
                m += 1
            if m < end and s[m] not in _OPERATOR_CHARS and s[m] != "#":
                return None  # trailing script argv: ``$0``/``$1`` semantics
            script_raw = s[k:script_end]
            script = self._literal_word_value(script_raw)
            if script is None:
                return None  # expansions inside the script: leave for bash
            try:
                inner = _BashFixScanner(script)
                inner._scan_range(0, len(script))
            except RecursionError:
                return None
            fixed = _fix_heredoc_trailing_operators(inner._build_source())
            self.names.extend(n for n in inner.names if n not in self.names)
            self.path_notes.extend(inner.path_notes)
            self.shell_notes.extend(
                n for n in inner.shell_notes if n not in self.shell_notes
            )
            if wrapped:
                # Keep ``<wrapper> bash -c '<script>'`` and fix the script in
                # place: the wrapper runs bash natively and the nested bash
                # inherits the exported fallback functions.
                if inner.edits:
                    self.edits.append((k, script_end, _single_quote(fixed)))
                return script_end, True
            self.edits.append((word_start, script_end, fixed))
            self.shell_notes.append(f"{shell_name} -c")
            return script_end, False

        if self._plausible_script_file(next_raw):
            return None
        # ``bash <command> ...``: drop the redundant shell word itself.  The
        # edit covers the shell word plus the whitespace that separated it from
        # the real command (``bash cd ...`` -> ``cd ...``).
        self.edits.append((word_start, j, ""))
        self.shell_notes.append(shell_name)
        # Under a command wrapper the wrapped command must keep its executable
        # context (``env bash rev`` -> ``env <runner>``), so the caller leaves
        # the active wrapper in place.
        return j, wrapped

    def _watch_command_operand(
        self, word_start: int, word_end: int, raw: str
    ) -> None:
        """Fix the inline script of a quoted ``watch`` command operand.

        procps ``watch`` executes its command line through ``sh -c``, and the
        Git Bash fallback mirrors that with ``eval "$*"`` in the current
        shell.  A quoted operand (``watch -n1 'rev <<< abc'``) is therefore an
        inline script: its literal value is scanned as its own command context
        and re-emitted with the inner fixes applied, so fallback names and
        Windows paths inside it work.  Unquoted operands need no special
        handling here — they flow through the normal command-word rules.
        """
        if not (raw.startswith("'") or raw.startswith('"')):
            return
        script = self._literal_word_value(raw)
        if script is None:
            return  # expansions inside the script: leave for bash
        try:
            inner = _BashFixScanner(script)
            inner._scan_range(0, len(script))
        except RecursionError:
            return
        fixed = _fix_heredoc_trailing_operators(inner._build_source())
        self.names.extend(n for n in inner.names if n not in self.names)
        self.path_notes.extend(inner.path_notes)
        self.shell_notes.extend(
            n for n in inner.shell_notes if n not in self.shell_notes
        )
        if inner.edits:
            self.edits.append((word_start, word_end, _single_quote(fixed)))


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
        wrapper: _BashWrapper | None = None
        heredoc_operator: str | None = None
        herestring_flag = False
        pending_heredocs: list[_BashHereDoc] = []
        case_stack: list[str] = []
        function_name_expected = False
        function_body_expected = False
        # Set when an assignment prefix (``VAR=x``) precedes the command word on
        # this line; a following ``bash`` wrapper must keep its shell-process
        # scoping, so it is not unwrapped (``VAR=x bash -c 'echo $VAR'``).
        assignment_prefix = False

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
                assignment_prefix = False
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
                            _BashHereDoc(delimiter, heredoc_operator == "<<-", expands)
                        )
                elif not herestring_flag:
                    raw_word = s[i:word_end]
                    replacement = self._path_replacement(raw_word)
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
                assignment_prefix = False
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
                    replacement = self._path_replacement(raw)
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
                assignment_prefix = True
                continue

            if raw == "cd":
                self._drop_cmd_cd_flag(i, end)

            executable_wrapper = (
                wrapper is not None and wrapper.kind not in _SAME_SHELL_WRAPPERS
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
                        replacement = self._path_replacement(value)
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
                        replacement = self._path_replacement(raw)
                        if replacement is not None:
                            self.edits.append((word_start, word_end, replacement))
                            self.path_notes.append(raw)
                    command_expected = True
                    continue
                if action == "inspect":
                    command_expected = False
                    wrapper = None
                    continue
                if wrapper is not None and wrapper.kind == "watch":
                    # ``watch`` re-executes its command in the current shell
                    # (``eval "$*"``, matching procps ``watch``'s ``sh -c``),
                    # so a quoted operand is an inline script: scan its
                    # literal value as its own command context so fallback
                    # names and Windows paths inside it are fixed too.  An
                    # unquoted operand falls through to the plain command-word
                    # rules below (``watch`` is a same-shell wrapper, so the
                    # source word resolves against the prefix functions).
                    self._watch_command_operand(word_start, word_end, raw)
                    wrapper = None

            if raw in _COMMAND_WRAPPERS:
                wrapper = _BashWrapper(
                    raw, operands=_WRAPPER_OPERAND_COUNTS.get(raw, 0)
                )
                command_expected = True
                continue
            fallback_wrapper = _FALLBACK_COMMAND_WRAPPERS.get(raw)
            if fallback_wrapper is not None:
                # ``gtimeout 5 rev``/``watch -n1 rev``: the wrapper word is
                # itself a fallback (its definition is recorded) and the
                # command that follows its options/operands is scanned like a
                # wrapped command word.
                self.names.append(raw)
                if executable_wrapper:
                    # The wrapping executable (``xargs gtimeout ...``) cannot
                    # invoke shell functions: swap the word for the standalone
                    # runner.  The operand scan continues through the wrapper
                    # state so the names the wrapped command uses are still
                    # recorded; the prefix exports them, which the runner's
                    # nested bash inherits.
                    self.edits.append(
                        (word_start, word_end, _wrapper_runner(raw))
                    )
                wrapper = _BashWrapper(
                    fallback_wrapper,
                    operands=_WRAPPER_OPERAND_COUNTS.get(fallback_wrapper, 0),
                )
                command_expected = True
                continue

            # Redundant shell invocation (``bash cd ...``, ``bash -c '...'``):
            # only at a plain command position — after ``command``/``env``/
            # ``sudo`` the shell word is the wrapped command itself and is
            # consumed by the wrapper logic above, so it is left untouched.
            shell_name = self._shell_wrapper_name(raw)
            if shell_name is not None:
                handled = self._handle_shell_wrapper(
                    shell_name,
                    word_start,
                    i,
                    end,
                    assignment_prefix=assignment_prefix,
                    wrapped=wrapper is not None,
                )
                if handled is not None:
                    i, keep_wrapper = handled
                    command_expected = True
                    redirect_expected = False
                    heredoc_operator = None
                    if not keep_wrapper:
                        wrapper = None
                    assignment_prefix = False
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
                # (``C:\tools\rg.exe``) or a Git Bash virtual absolute path
                # (``/c/tools/rg.exe``); Bash quote removal would eat the
                # backslashes and lose the command, so rewrite it like an
                # argument path.
                replacement = self._path_replacement(raw)
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
            replacement = self._path_replacement(raw)
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
        pending_heredocs: list[_BashHereDoc] = []
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
                    pending_heredocs.append(_BashHereDoc(delimiter, strip_tabs, expands))
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

    def _consume_wrapper_word(self, wrapper: _BashWrapper, raw: str) -> str:
        if wrapper.skip_next:
            wrapper.skip_next = False
            if wrapper.opaque:
                return "inspect"
            return "skip"
        if wrapper.opaque:
            return "inspect"
        if wrapper.operands > 0 and not raw.startswith("-"):
            # A plain operand before the command word (GNU ``timeout`` takes
            # exactly one DURATION operand) is wrapper data, not the wrapped
            # command; consume it without ending the wrapper.
            wrapper.operands -= 1
            return "skip"
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
        documents: list[_BashHereDoc],
        *,
        scan_expansions: bool = True,
    ) -> int:
        s = self.s
        redir_line_end = -1
        for document in documents:
            body_start = i
            if redir_line_end < 0:
                redir_line_end = body_start - 1
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
        if redir_line_end >= 0:
            self.heredoc_events.append((redir_line_end, i))
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

    def _git_bash_abs_path_replacement(self, raw: str) -> str | None:
        """Return the native Windows spelling of a Git Bash virtual absolute path.

        Git Bash accepts POSIX-style absolute paths that native Windows
        executables cannot resolve.  ``/tmp/x`` maps to the user's Windows temp
        directory (``cygpath -w /tmp``), while a native tool would read it as
        ``<current-drive>:\\tmp\\x``; ``/c/x`` means ``C:/x``, but a native
        tool would read it as ``<current-drive>:\\c\\x``.  Rewriting these
        words to their real Windows spellings makes the same command behave
        identically under Git Bash and native POSIX bash: ``/tmp`` is the
        system temp directory in both, and ``/c/...``/``/d/...`` resolve to
        the same files whether the consumer is an MSYS tool or a native
        executable.  Only unquoted words are considered; quoted text is
        literal data.  A bare single-letter mount (``/c``, ``/d``) is left
        untouched so cmd.exe's ``cd /d <path>`` flag is never mistaken for a
        drive path.
        """
        if not raw:
            return None
        if raw.startswith("/tmp"):
            if raw != "/tmp" and not raw.startswith("/tmp/"):
                return None
            return self._quote_path_word(_windows_temp_dir() + raw[4:])
        if len(raw) >= 3 and raw[0] == "/" and raw[1].isalpha() and raw[2] == "/":
            return self._quote_path_word(raw[1].upper() + ":" + raw[2:])
        return None

    def _path_replacement(self, raw: str) -> str | None:
        """Return the Git Bash spelling for a Windows path word.

        Covers both spellings agents use on Windows: native backslash paths
        (``D:\\repo\\src``) and Git Bash virtual POSIX absolute paths
        (``/tmp/x``, ``/c/x``).  Returns ``None`` when *raw* is not an
        unambiguous path word.
        """
        replacement = self._windows_path_replacement(raw)
        if replacement is not None:
            return replacement
        return self._git_bash_abs_path_replacement(raw)

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


_HEREDOC_TRAILING_OPERATORS = frozenset({"&&", "||", "|", "|&", ";", "&"})


def _read_shell_control_operator(s: str, i: int, n: int) -> tuple[str, int]:
    """Return the control operator at position *i* and the index after it."""
    if i >= n:
        return "", i
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
    return "", i


def _apply_heredoc_operator_move(
    source: str, redir_line_end: int, terminator_end: int
) -> str:
    """Move a control-operator line following a heredoc terminator to the redirection line.

    Bash requires a control operator that continues a heredoc-delimited command
    to appear on the same line as the ``<<`` redirection.  A common model
    mistake is to place the operator on the line after the closing delimiter,
    which produces ``syntax error near unexpected token `&&'``.  This helper
    repairs that pattern while leaving the heredoc body and delimiter intact.
    """
    n = len(source)
    if redir_line_end < 0 or redir_line_end >= n or source[redir_line_end] != "\n":
        return source
    if terminator_end < 0 or terminator_end > n:
        return source

    i = terminator_end
    while i < n:
        ch = source[i]
        if ch in " \t\r":
            i += 1
            continue
        if ch == "\n":
            i += 1
            continue
        if ch == "#":
            nl = source.find("\n", i, n)
            i = n if nl < 0 else nl + 1
            continue
        break
    if i >= n:
        return source

    op, op_end = _read_shell_control_operator(source, i, n)
    if op not in _HEREDOC_TRAILING_OPERATORS:
        return source
    # ``&>`` / ``&>>`` are redirections, not list terminators.
    if op == "&" and op_end < n and source[op_end] == ">":
        return source

    move_start = i
    line_end = source.find("\n", i, n)
    if line_end < 0:
        line_end = n
        move_end = n
    else:
        move_end = line_end + 1

    rest = source[op_end:line_end].lstrip(" \t\r")
    if not rest or rest.startswith("#"):
        k = move_end
        while k < n:
            if source[k] in " \t\r":
                k += 1
                continue
            if source[k] == "\n":
                k += 1
                continue
            if source[k] == "#":
                nl = source.find("\n", k, n)
                k = n if nl < 0 else nl + 1
                continue
            break
        if k >= n:
            return source
        next_line_end = source.find("\n", k, n)
        if next_line_end < 0:
            move_end = n
        else:
            move_end = next_line_end + 1

    lines = source[move_start:move_end].splitlines()
    parts: list[str] = []
    if lines:
        parts.append(lines[0][len(op):].strip())
        parts.extend(line.strip() for line in lines[1:])
    joined = " ".join(part for part in parts if part and not part.startswith("#"))
    moved = f"{op} {joined}\n" if joined else f"{op}\n"

    return (
        source[:redir_line_end]
        + " "
        + moved
        + source[redir_line_end + 1 : move_start]
        + source[move_end:]
    )


def _fix_heredoc_trailing_operators(source: str) -> str:
    """Repair heredoc commands whose trailing control operator is on the wrong line."""
    try:
        scanner = _BashFixScanner(source)
        scanner._scan_range(0, scanner.n)
    except RecursionError:
        return source
    events = scanner.heredoc_events
    if not events:
        return source
    for redir_line_end, terminator_end in reversed(events):
        source = _apply_heredoc_operator_move(source, redir_line_end, terminator_end)
    return source


def fix_bash_command(command: str) -> BashFix:
    """Rewrite selected native POSIX commands for Windows Git Bash.

    The Windows-platform gate lives in the app layer
    (``kimix.tools.file.bash.bash_fix.fix_bash_command``); this scanner is
    byte-identical to the native kernel, which also does not gate on platform.
    Empty input is returned byte-for-byte unchanged.
    """
    if not command:
        return BashFix(command)
    # Quoting and escaping can form a literal command name without the source
    # containing it contiguously (for example ``r""ev`` or ``\rev``), so a
    # substring fast path would miss legal executable words.  The scanner is
    # linear and exits without allocating generated shell code when unchanged.
    result = _BashFixScanner(command).fix()
    fixed = _fix_heredoc_trailing_operators(result.command)
    return BashFix(
        fixed,
        result.replacements,
        result.path_changes,
        result.shell_wrappers,
    )


# ======================================================================
# bash_tool.py (kimi-agent) - _process_unquoted + helpers
# ======================================================================
_BASH_METACHARACTERS = frozenset("()|;&<>$\"`'\"*?[]{}~!#=% \t\n\r")

# In double quotes, \ only escapes these characters.  $ and ` are included
# because \$, \` inside "..." are literal (the $ / ` is escaped, not triggering
# variable expansion or command substitution).
_DQ_ESCAPED = frozenset(('"', '\\', '$', '`'))

# Precompiled regex for finding the next special character in unquoted mode.
# Matches backslash, single quote, double quote, dollar, or backtick.
_UNQUOTED_SPECIAL_RE = re.compile(r'[\\\'"$`]')


def _find_ansi_c_end(cmd: str, start: int) -> int:
    """Return the index AFTER the closing ' of a ``$'...'`` region.

    ``start`` is the position right after the opening ``$'`` (i.e. the first
    character inside the region).  Returns ``-1`` if the region is
    unterminated.  Inside ``$'...'`` every ``\\X`` pair is treated as an
    escape (any character after \\ is skipped over).
    """
    i = start
    length = len(cmd)
    while i < length:
        c = cmd[i]
        if c == "\\" and i + 1 < length:
            i += 2
        elif c == "'":
            return i + 1
        else:
            i += 1
    return -1


def _find_backtick_end(cmd: str, start: int) -> int:
    """Return the index AFTER the closing `` ` `` of a backtick region.

    ``start`` is the position right after the opening `` ` ``.
    Returns ``-1`` if the region is unterminated.  ``\\` `` inside the
    region is an escaped backtick (literal `` ` ``).
    """
    i = start
    length = len(cmd)
    while i < length:
        c = cmd[i]
        if c == "\\" and i + 1 < length:
            i += 2  # skip escaped char (including \`)
        elif c == "`":
            return i + 1
        else:
            i += 1
    return -1


def _find_matching_paren(cmd: str, open_pos: int) -> int:
    """Return the index of the ``)`` matching the ``(`` at ``cmd[open_pos]``.

    Returns ``-1`` if no matching ``)`` is found.  Tracks nested ``$(...)``,
    single-quoted regions, double-quoted regions (including their own
    nested ``$(...)`` and backticks), and backtick regions.
    """
    assert cmd[open_pos] == "("
    depth = 1
    i = open_pos + 1
    length = len(cmd)
    while i < length:
        c = cmd[i]
        if c == "'":
            end = cmd.find("'", i + 1)
            if end == -1:
                return -1
            i = end + 1
        elif c == '"':
            i = _find_dq_end(cmd, i + 1)
            if i == -1:
                return -1
        elif c == "`":
            i = _find_backtick_end(cmd, i + 1)
            if i == -1:
                return -1
        elif c == "$" and i + 1 < length and cmd[i + 1] == "(":
            depth += 1
            i += 2
        elif c == "$" and i + 1 < length and cmd[i + 1] == "'":
            # $'...' ANSI-C quoted region — skip to its closing '
            end = _find_ansi_c_end(cmd, i + 2)
            if end == -1:
                return -1
            i = end
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
            i += 1
        else:
            i += 1
    return -1


def _find_dq_end(cmd: str, start: int) -> int:
    """Return the index AFTER the closing ``"`` of a double-quoted region.

    ``start`` is the position right after the opening ``"``.
    Returns ``-1`` if the region is unterminated.  Recognises ``\\X``
    escapes (``X`` in ``_DQ_ESCAPED``), nested ``$(...)``, ``$'...'``, and
    backtick command substitutions inside the region.
    """
    i = start
    length = len(cmd)
    while i < length:
        c = cmd[i]
        if c == "\\" and i + 1 < length and cmd[i + 1] in _DQ_ESCAPED:
            i += 2  # skip \X (X is escaped: ", \, $, `)
        elif c == '"':
            return i + 1
        elif c == "$" and i + 1 < length and cmd[i + 1] == "(":
            end = _find_matching_paren(cmd, i + 1)
            if end == -1:
                return -1
            i = end + 1
        elif c == "$" and i + 1 < length and cmd[i + 1] == "'":
            end = _find_ansi_c_end(cmd, i + 2)
            if end == -1:
                return -1
            # _find_ansi_c_end returns the index AFTER the closing '
            i = end
        elif c == "`":
            end = _find_backtick_end(cmd, i + 1)
            if end == -1:
                return -1
            # _find_backtick_end returns the index AFTER the closing `
            i = end
        else:
            i += 1
    return -1


def _process_unquoted(cmd: str) -> str:
    """Convert unquoted backslashes to forward slashes in ``cmd``.

    Walks the string in *unquoted mode* (the same rules that apply at the
    top level of a bash command): a bare ``\\`` followed by a non-metachar
    is converted to ``/``, while ``\\`` followed by a bash metacharacter,
    or ``\\`` inside single / double / ANSI-C quotes, is preserved.

    The function also descends into ``$(...)`` and backtick command
    substitutions, processing their *content* in unquoted mode as well
    (because bash runs the content of ``$(...)`` and `` ` ` `` in a
    subshell where it is parsed unquoted — even when the substitution is
    itself nested inside ``"..."``).
    """
    result: list[str] = []
    i = 0
    length = len(cmd)

    while i < length:
        # ---- find the next special character ----
        # Use a single regex search (C-accelerated) to bulk-skip non-special chars.
        m = _UNQUOTED_SPECIAL_RE.search(cmd, i)
        if m:
            nxt = m.start()
            if nxt > i:
                result.append(cmd[i:nxt])
                i = nxt
        else:
            # No more special characters — append the remaining suffix and finish.
            result.append(cmd[i:])
            break

        if i >= length:
            break

        char = cmd[i]

        if char == "'":
            # Single-quoted region — copy literally until closing '
            end = cmd.find("'", i + 1)
            if end == -1:
                result.append(cmd[i:])
                break
            result.append(cmd[i : end + 1])
            i = end + 1

        elif char == '"':
            # Double-quoted region.  First find the end of the region,
            # then walk through it and convert the *content* of any
            # $(...) and `...` sub-regions using unquoted-mode rules
            # (bash runs command substitutions in a subshell where the
            # content is parsed unquoted, so backslashes inside must be
            # converted to '/' just like at the top level).
            dq_end = _find_dq_end(cmd, i + 1)
            if dq_end == -1:
                # Unterminated — copy the rest verbatim
                result.append(cmd[i:])
                break
            j = i + 1
            chunk_start = i
            while j < dq_end:
                # Bulk-skip to the next interesting character inside DQ:
                # backslash, dollar, or backtick.
                m2 = _UNQUOTED_SPECIAL_RE.search(cmd, j, dq_end)
                if m2:
                    nxt2 = m2.start()
                    if nxt2 > j:
                        j = nxt2
                else:
                    # No more special chars inside DQ — rest is verbatim
                    j = dq_end
                    break

                c = cmd[j]
                if c == "\\" and j + 1 < dq_end and cmd[j + 1] in _DQ_ESCAPED:
                    # \X inside DQ: X is escaped.  Skip the pair; it will
                    # be included in the next emitted chunk.
                    j += 2
                elif c == "$" and j + 1 < dq_end and cmd[j + 1] == "(":
                    # $(...) command substitution — process content
                    paren_end = _find_matching_paren(cmd, j + 1)
                    if paren_end == -1 or paren_end >= dq_end:
                        # Unterminated or mismatched — treat rest as verbatim
                        j = dq_end
                        break
                    result.append(cmd[chunk_start:j])
                    result.append("$(")
                    result.append(_process_unquoted(cmd[j + 2 : paren_end]))
                    result.append(")")
                    j = paren_end + 1
                    chunk_start = j
                elif c == "$" and j + 1 < dq_end and cmd[j + 1] == "'":
                    # $'...' ANSI-C region — skip through it (copied
                    # verbatim as part of the next chunk).
                    ac_end = _find_ansi_c_end(cmd, j + 2)
                    if ac_end == -1 or ac_end > dq_end:
                        # Unterminated or extends beyond DQ — treat rest as verbatim
                        j = dq_end
                        break
                    j = ac_end
                elif c == "`":
                    # Backtick command substitution — process content
                    bt_end = _find_backtick_end(cmd, j + 1)
                    if bt_end == -1 or bt_end > dq_end:
                        # Unterminated or extends beyond DQ — treat rest as verbatim
                        j = dq_end
                        break
                    result.append(cmd[chunk_start:j])
                    result.append("`")
                    result.append(_process_unquoted(cmd[j + 1 : bt_end - 1]))
                    result.append("`")
                    j = bt_end
                    chunk_start = j
                else:
                    # Should not reach here — char is not one we handle in DQ
                    j += 1
            # Emit the final chunk (up to and including the closing ")
            result.append(cmd[chunk_start:dq_end])
            i = dq_end

        elif char == "$" and i + 1 < length and cmd[i + 1] == "'":
            # $'...' ANSI-C quoted region at top level — copy literally
            ac_end = _find_ansi_c_end(cmd, i + 2)
            if ac_end == -1:
                result.append(cmd[i:])
                break
            result.append(cmd[i:ac_end])
            i = ac_end

        elif char == "`":
            # Backtick command substitution at top level — process content
            bt_end = _find_backtick_end(cmd, i + 1)
            if bt_end == -1:
                result.append(cmd[i:])
                break
            result.append("`")
            result.append(_process_unquoted(cmd[i + 1 : bt_end - 1]))
            result.append("`")
            i = bt_end

        elif char == "\\":
            if i + 1 < length and cmd[i + 1] in _BASH_METACHARACTERS:
                # Backslash is escaping a bash metacharacter — preserve both.
                # Append atomically so the metacharacter (e.g. ' " $) is not
                # re-processed as a quote-start or ANSI-C region on the next
                # iteration.
                result.append("\\")
                result.append(cmd[i + 1])
                i += 2
            else:
                # Unquoted backslash in a path-like context — convert to /
                result.append("/")
                i += 1

        else:
            # Defensive: nxt should always point to a special char we handle.
            result.append(char)
            i += 1

    return "".join(result)




# pwsh_fix.py (kimi-agent) - PowerShell quoting validator/repair
# ======================================================================


_NORMAL = "normal"
_DQ = "double-quoted"
_SQ = "single-quoted"
_HDQ = "here-double"
_HSQ = "here-single"
_COMMENT = "line-comment"
_BLOCK = "block-comment"

_W_UNCLOSED_DQ = (
    "The command has an unclosed double-quoted string; "
    'appended a closing `"` at the end to make it a legal PowerShell command.'
)
_W_UNCLOSED_SQ = (
    "The command has an unclosed single-quoted string; "
    "appended a closing `'` at the end to make it a legal PowerShell command."
)
_W_UNCLOSED_HDQ = (
    "The command has an unclosed double-quoted here-string; "
    'appended a newline and `"@` at the end to close it.'
)
_W_UNCLOSED_HSQ = (
    "The command has an unclosed single-quoted here-string; "
    "appended a newline and `'@` at the end to close it."
)
_W_UNCLOSED_BLOCK = (
    "The command has an unclosed block comment `<#`; "
    "appended `#>` at the end to close it."
)
_W_TRAILING_COMMENT = (
    "The command ends with a line comment; "
    "appended a newline so the trailing comment does not swallow the "
    "try/catch wrapper used to execute the command."
)
_W_STOP_PARSING = (
    "The command ends with the `--%` stop-parsing marker; "
    "appended a newline so the wrapper is not passed literally to the "
    "native command."
)
_W_COMMENT_ONLY = (
    "The command contains only comments; appended a newline and a no-op "
    "`$null` statement so the try/catch wrapper has a statement to execute."
)
_W_TRAILING_CONTINUATION = (
    "The command ends with a backtick line-continuation; "
    "appended a newline so the continuation does not join with the "
    "try/catch wrapper used to execute the command."
)


@dataclass(frozen=True)
class PwshFix:
    """Result of :func:`fix_pwsh_command`.

    ``command`` is the (possibly repaired) command to execute and ``warning``
    is a human-readable note describing any modification, or ``""`` when the
    command was already valid and left unchanged.
    """

    command: str
    warning: str = ""

    @property
    def changed(self) -> bool:
        """True when the command text differs from the input."""
        return bool(self.warning)


def fix_pwsh_command(cmd: str) -> PwshFix | None:
    """Validate *cmd* with PowerShell quoting rules and repair it if possible.

    Returns a :class:`PwshFix` (``command`` may equal *cmd* when the command
    is already legal), or ``None`` when the command cannot be repaired.
    """
    if not cmd or not cmd.strip():
        return None
    # Fast path: no quote/comment/continuation/here-string/stop-parsing
    # characters at all — the command is valid as-is and cannot affect the
    # try/catch wrapper.  Avoids the O(n) Python scan for plain commands.
    if (
        '"' not in cmd
        and "'" not in cmd
        and "#" not in cmd
        and "`" not in cmd
        and "@" not in cmd
        and "--%" not in cmd
    ):
        return PwshFix(cmd, "")
    return _PwshScanner(cmd).fix()


class _PwshScanner:
    """Single-pass tokenizer implementing PowerShell's quoting rules."""

    __slots__ = ("s", "n")

    def __init__(self, s: str) -> None:
        self.s = s
        self.n = len(s)

    # -- helpers used while skipping $(...) sub-expressions -----------------

    def _skip_sq(self, start: int) -> int:
        """Skip a single-quoted string starting at *start*; index after it."""
        s, n = self.s, self.n
        i = start + 1
        while i < n:
            if s[i] == "'":
                if i + 1 < n and s[i + 1] == "'":
                    i += 2  # '' -> literal single quote
                else:
                    return i + 1  # closing quote
            else:
                i += 1
        return i

    def _skip_dq(self, start: int) -> int:
        """Skip a double-quoted string starting at *start*; index after it."""
        s, n = self.s, self.n
        i = start + 1
        while i < n:
            ch = s[i]
            if ch == "`":
                i += 2 if i + 1 < n else 1
            elif ch == '"':
                if i + 1 < n and s[i + 1] == '"':
                    i += 2  # "" -> literal double quote
                else:
                    return i + 1  # closing quote
            elif ch == "$" and i + 1 < n and s[i + 1] == "(":
                i = self._skip_subexpr(i)
            else:
                i += 1
        return i

    def _skip_block(self, start: int) -> int:
        """Skip a block comment starting at *start*; index after it.

        PowerShell closes block comments at the first ``#>`` — they do not
        nest (verified empirically with pwsh 7.6.2).
        """
        s, n = self.s, self.n
        i = start + 2
        while i < n:
            if s[i] == "#" and i + 1 < n and s[i + 1] == ">":
                return i + 2
            i += 1
        return i

    def _skip_subexpr(self, start: int) -> int:
        """Skip a ``$( ... )`` sub-expression starting at *start*.

        Iterative (no recursion) so deeply nested ``$(...)`` cannot hit the
        interpreter recursion limit.  Nested ``$(...)`` are handled by the
        paren-depth counter; strings and comments are skipped before the
        parens are counted, so a ``)`` inside a string is never mistaken for
        the closing paren.

        Returns the index *after* the matching ``)`` (or ``n`` when the
        sub-expression never closes — the caller then treats the enclosing
        string as unclosed).
        """
        s, n = self.s, self.n
        i = start + 2
        depth = 1
        while i < n and depth:
            ch = s[i]
            if ch == "(":
                depth += 1
                i += 1
            elif ch == ")":
                depth -= 1
                i += 1
            elif ch == "'":
                i = self._skip_sq(i)
            elif ch == '"':
                i = self._skip_dq(i)
            elif ch == "`":
                i += 2 if i + 1 < n else 1
            elif ch == "#":
                if self._at_token_start(i):
                    while i < n and s[i] != "\n":
                        i += 1
                else:
                    i += 1
            elif ch == "<" and i + 1 < n and s[i + 1] == "#":
                i = self._skip_block(i)
            else:
                i += 1
        return i

    # -- token-boundary predicate -------------------------------------------

    def _at_token_start(self, i: int) -> bool:
        """True when *i* starts a fresh token (not glued to a word/identifier)."""
        return i == 0 or not (self.s[i - 1].isalnum() or self.s[i - 1] == "_")

    # -- EOF repair ----------------------------------------------------------

    def _dq_closer(self) -> str:
        """Closing quote for an unclosed double-quoted string.

        A trailing backtick would escape a single appended ``"``, so an odd
        run of trailing backticks needs two quotes (one escaped, one closing).
        """
        k = 0
        for ch in reversed(self.s):
            if ch == "`":
                k += 1
            else:
                break
        return '""' if k % 2 == 1 else '"'

    # -- main scan -----------------------------------------------------------

    def fix(self) -> PwshFix | None:
        s, n = self.s, self.n
        mode = _NORMAL
        here_quote = ""   # opening quote of the current here-string
        line_start = 0    # start of the current line inside a here-string
        saw_code = False  # any real statement code seen (not comments/whitespace)
        last_cont_target = -1  # index after the last backtick-newline continuation
        i = 0
        while i < n:
            ch = s[i]
            if mode == _NORMAL:
                if ch == '"':
                    saw_code = True
                    mode = _DQ
                    i += 1
                elif ch == "'":
                    saw_code = True
                    mode = _SQ
                    i += 1
                elif ch == "`":
                    if i + 1 < n:
                        saw_code = True
                        if s[i + 1] == "\n":
                            last_cont_target = i + 2
                        i += 2  # escaped char (or line continuation)
                    else:
                        # Dangling line-continuation backtick: PowerShell
                        # rejects a backtick with nothing after it.
                        return None
                elif ch == "#" and self._at_token_start(i):
                    mode = _COMMENT
                    i += 1
                elif ch == "<" and i + 1 < n and s[i + 1] == "#":
                    mode = _BLOCK
                    i += 2
                elif (
                    ch == "@"
                    and i + 1 < n
                    and s[i + 1] in ("'", '"')
                    and self._at_token_start(i)
                ):
                    # Here-string opener: @' or @" followed by only
                    # whitespace until end-of-line (or end of input).
                    j = i + 2
                    while j < n and s[j] in " \t\r":
                        j += 1
                    if j == n or s[j] == "\n":
                        saw_code = True
                        here_quote = s[i + 1]
                        mode = _HDQ if here_quote == '"' else _HSQ
                        line_start = n if j == n else j + 1
                        i = line_start
                        continue
                    saw_code = True
                    i += 1  # @' / @" with content on the same line: not a here-string
                elif (
                    ch == "-"
                    and s.startswith("--%", i)
                    and self._at_token_start(i)
                ):
                    # --% stop-parsing: the rest of the line is literal.
                    if not saw_code:
                        # `--%` with no command before it is a PowerShell
                        # parse error ("Missing expression after unary
                        # operator '--'") — nothing to repair.
                        return None
                    nl = s.find("\n", i)
                    if nl == -1:
                        # The literal region reaches EOF and would swallow the
                        # try/catch wrapper — terminate the line explicitly.
                        return PwshFix(s + "\n", _W_STOP_PARSING)
                    i = nl + 1
                elif ch == "$" and i + 1 < n and s[i + 1] == "(":
                    saw_code = True
                    i = self._skip_subexpr(i)
                elif ch.isspace():
                    i += 1
                else:
                    saw_code = True
                    i += 1
            elif mode == _DQ:
                if ch == "`":
                    i += 2 if i + 1 < n else 1
                elif ch == '"':
                    if i + 1 < n and s[i + 1] == '"':
                        i += 2  # "" -> literal double quote
                    else:
                        mode = _NORMAL
                        i += 1
                elif ch == "$" and i + 1 < n and s[i + 1] == "(":
                    i = self._skip_subexpr(i)
                else:
                    i += 1
            elif mode == _SQ:
                if ch == "'":
                    if i + 1 < n and s[i + 1] == "'":
                        i += 2  # '' -> literal single quote
                    else:
                        mode = _NORMAL
                        i += 1
                else:
                    i += 1
            elif mode in (_HDQ, _HSQ):
                if ch == "\n":
                    line_start = i + 1
                    i += 1
                elif (
                    ch == here_quote
                    and i + 1 < n
                    and s[i + 1] == "@"
                    and s[line_start:i].strip() == ""
                ):
                    mode = _NORMAL
                    i += 2
                else:
                    i += 1
            elif mode == _COMMENT:
                if ch == "\n":
                    mode = _NORMAL
                    i += 1
                else:
                    i += 1
            elif mode == _BLOCK:
                if ch == "#" and i + 1 < n and s[i + 1] == ">":
                    mode = _NORMAL
                    i += 2
                else:
                    i += 1

        # -- end of input -----------------------------------------------------
        # A line-continuation backtick whose target line runs to the end of
        # the command would join with the try/catch wrapper added by the tool,
        # silently corrupting the command.  Append a newline so the
        # continuation ends on an empty line instead.
        needs_cont_nl = last_cont_target != -1 and s.rfind("\n") < last_cont_target
        if mode == _NORMAL:
            if saw_code:
                if needs_cont_nl:
                    return PwshFix(s + "\n", _W_TRAILING_CONTINUATION)
                return PwshFix(s, "")
            # Only comments/whitespace: give the wrapper a statement to run.
            return PwshFix(s + "\n$null", _W_COMMENT_ONLY)
        if mode == _DQ:
            fixed = s + self._dq_closer()
            warning = _W_UNCLOSED_DQ
            if needs_cont_nl:
                fixed += "\n"
                warning += "\n" + _W_TRAILING_CONTINUATION
            return PwshFix(fixed, warning)
        if mode == _SQ:
            fixed = s + "'"
            warning = _W_UNCLOSED_SQ
            if needs_cont_nl:
                fixed += "\n"
                warning += "\n" + _W_TRAILING_CONTINUATION
            return PwshFix(fixed, warning)
        if mode == _HDQ:
            fixed = s + '\n"@'
            warning = _W_UNCLOSED_HDQ
            if needs_cont_nl:
                fixed += "\n"
                warning += "\n" + _W_TRAILING_CONTINUATION
            return PwshFix(fixed, warning)
        if mode == _HSQ:
            fixed = s + "\n'@"
            warning = _W_UNCLOSED_HSQ
            if needs_cont_nl:
                fixed += "\n"
                warning += "\n" + _W_TRAILING_CONTINUATION
            return PwshFix(fixed, warning)
        if mode == _COMMENT:
            if saw_code:
                return PwshFix(s + "\n", _W_TRAILING_COMMENT)
            return PwshFix(s + "\n$null", _W_COMMENT_ONLY)
        if mode == _BLOCK:
            if saw_code:
                fixed = s + "#>"
                warning = _W_UNCLOSED_BLOCK
            else:
                fixed = s + "#>\n$null"
                warning = _W_COMMENT_ONLY
            if needs_cont_nl:
                fixed += "\n"
                warning += "\n" + _W_TRAILING_CONTINUATION
            return PwshFix(fixed, warning)
        return None  # pragma: no cover - unreachable


# ======================================================================
# process_pwsh.py (kimi-agent) - PS7 to PS5.1 source transform
# ======================================================================


try:
    import regex as re
except ImportError:  # pragma: no cover
    import re  # stdlib: the used patterns are all stdlib-compatible


# ===========================================================================
# Constants
# ===========================================================================

_PS_KEYWORDS = frozenset({
    "begin", "break", "catch", "class", "continue", "data", "define", "do",
    "dynamicparam", "else", "elseif", "end", "enum", "exit", "filter", "finally",
    "for", "foreach", "from", "function", "hidden", "if", "in", "param",
    "process", "return", "static", "switch", "throw", "trap", "try", "until",
    "using", "var", "while",
})



_EXPR_STOP = "=;|&,"

_DEPTH_OPEN = "([{"
_DEPTH_CLOSE = ")]}"


# ===========================================================================
# Low-level scanners — skip over strings, comments, subexpressions
# ===========================================================================

def _scan_single_quoted(code: str, i: int) -> int:
    """Skip a single-quoted string starting at *i*; return index after it."""
    i += 1
    n = len(code)
    while i < n:
        if code[i] == "'":
            if i + 1 < n and code[i + 1] == "'":
                i += 2          # escaped '' → literal single-quote
            else:
                return i + 1     # closing quote
        else:
            i += 1
    return i


def _scan_double_quoted(code: str, i: int) -> int:
    """Skip a double-quoted string starting at *i*; return index after it."""
    i += 1
    n = len(code)
    while i < n:
        ch = code[i]
        if ch == "`" and i + 1 < n:
            i += 2              # backtick-escaped char
        elif ch == '"':
            return i + 1         # closing quote
        elif ch == "$" and i + 1 < n and code[i + 1] == "(":
            i = _skip_subexpression(code, i)
        else:
            i += 1
    return i


def _scan_block_comment(code: str, i: int) -> int:
    """Skip a block comment ``<# ... #>`` starting at *i*; return index after it."""
    depth = 1
    i += 2
    n = len(code)
    while i < n and depth:
        if code[i] == "<" and i + 1 < n and code[i + 1] == "#":
            depth += 1
            i += 2
        elif code[i] == "#" and i + 1 < n and code[i + 1] == ">":
            depth -= 1
            i += 2
        else:
            i += 1
    return i


def _skip_subexpression(code: str, start: int) -> int:
    """Skip past a ``$(...)`` sub-expression starting at *start*.

    Returns the index *after* the closing ``)``.
    """
    assert code[start] == "$"
    i = start + 2
    depth = 1
    n = len(code)
    while i < n and depth:
        c = code[i]
        if c == "(":
            depth += 1
            i += 1
        elif c == ")":
            depth -= 1
            i += 1
        elif c == "'":
            i = _scan_single_quoted(code, i)
        elif c == '"':
            i = _scan_double_quoted(code, i)
        elif c == "$" and i + 1 < n and code[i + 1] == "(":
            i = _skip_subexpression(code, i)
        else:
            i += 1
    return i


# ===========================================================================
# Region finders  (strings, comments, here-strings)
# ===========================================================================

def _scan_here_string(code: str, start: int) -> int:
    """Skip a here-string (``@'...'@`` or ``@"..."@``) starting at *start*.

    *start* points to the opening ``@``.
    Returns the index *after* the closing delimiter.
    """
    n = len(code)
    quote = code[start + 1]
    i = start + 2
    seen_newline = False
    line_begin = start + 2  # start of current line
    while i < n:
        if code[i] == "\n":
            seen_newline = True
            line_begin = i + 1
        elif (
            code[i] == quote
            and i + 1 < n
            and code[i + 1] == "@"
            and seen_newline
        ):
            # Closing delimiter must be at the beginning of its own line
            # (only whitespace may precede it on that line)
            if code[line_begin:i].strip() == "":
                return i + 2
        i += 1
    return i


def _build_region_mask(code: str, *, here_strings: bool = True) -> bytearray:
    """Build a region mask for *code* in a single pass.

    Returns a bytearray where 1 means outside regions (code) and 0 means
    inside (strings, comments, here-strings).

    When *here_strings* is False, here-strings are not detected (used for
    line-level scanning where here-strings cannot be reliably identified).
    """
    n = len(code)
    mask = bytearray(b"\x01" * n)
    i = 0
    while i < n:
        c = code[i]
        if c == "<" and i + 1 < n and code[i + 1] == "#":
            start = i
            i = _scan_block_comment(code, i)
            mask[start:i] = b"\x00" * (i - start)
        elif c == "#":
            start = i
            while i < n and code[i] != "\n":
                i += 1
            mask[start:i] = b"\x00" * (i - start)
        elif here_strings and c == "@" and i + 1 < n and code[i + 1] in ("'", '"'):
            # PowerShell here-strings require the closing delimiter at the
            # beginning of its own line (only whitespace may precede it).
            j = i + 2
            while j < n and code[j] in " \t\r":
                j += 1
            if j < n and code[j] != "\n":
                i += 1  # Not a here-string: @' or @" not followed by newline
                continue
            start = i
            i = _scan_here_string(code, i)
            mask[start:i] = b"\x00" * (i - start)
        elif c == "'":
            start = i
            i = _scan_single_quoted(code, i)
            mask[start:i] = b"\x00" * (i - start)
        elif c == '"':
            start = i
            i = _scan_double_quoted(code, i)
            mask[start:i] = b"\x00" * (i - start)
        else:
            i += 1
    return mask


def _line_mask(line: str) -> bytearray:
    """Return a region mask for *line* (here-strings disabled)."""
    return _build_region_mask(line, here_strings=False)


# ===========================================================================
# Depth tracking  (for matching ternary colon)
# ===========================================================================

def _compute_depths(line: str, mask: bytearray) -> list[int]:
    """Return nesting depth of ``()``, ``{}``, ``[]`` before each character."""
    depths: list[int] = []
    depth = 0
    for i, ch in enumerate(line):
        depths.append(depth)
        if mask[i]:
            if ch in _DEPTH_OPEN:
                depth += 1
            elif ch in _DEPTH_CLOSE:
                depth -= 1
    depths.append(depth)
    return depths


# ===========================================================================
# Pre-processing: backtick line continuation
# ===========================================================================

def _join_continuation_lines(code: str) -> str:
    """Collapse backtick line-continuations into single logical lines."""
    mask = _build_region_mask(code)
    n = len(code)
    result: list[str] = []
    i = 0
    while i < n:
        if code[i] == "`" and mask[i]:
            j = i + 1
            while j < n and code[j] in " \t\r":
                j += 1
            if j < n and code[j] == "\n":
                j += 1
                while j < n and code[j] in " \t\r":
                    j += 1
                result.append(" ")
                i = j
                continue
        result.append(code[i])
        i += 1
    return "".join(result)


# ===========================================================================
# Assignment detection
# ===========================================================================

_ASSIGN_RE = re.compile(r"(.*?)(\$\w+(?::\w+)?(?:\.\w+)*)\s*=\s*$")
_COMMAND_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*\s+")


def _match_assignment(before: str) -> tuple[str, str] | None:
    """Match an assignment prefix like ``$var = `` at the end of *before*."""
    m = _ASSIGN_RE.match(before.rstrip())
    if m:
        return m.group(1), m.group(2)
    return None


def _build_replacement(prefix: str, inner: str) -> str:
    """Build replacement string, preserving an assignment if one is detected."""
    assign = _match_assignment(prefix)
    if assign:
        p, var = assign
        return f"{p}{var} = {inner}"
    return f"{prefix}{inner}"


def _strip_command_prefix(expr: str, start: int, *, check_keywords: bool = True) -> tuple[str, int]:
    """Strip a leading command name (e.g. ``Write-Output ``) from *expr*.

    Returns ``(stripped_expr, adjusted_start)``.
    When *check_keywords* is True (default), PowerShell keywords (if, foreach, …)
    are never stripped. When False, any command prefix is stripped.
    """
    m = _COMMAND_PREFIX_RE.match(expr)
    if m:
        cmd = m.group(0).strip().lower()
        if not check_keywords or cmd not in _PS_KEYWORDS:
            expr_part = expr[m.end():]
            if expr_part and expr_part[0] in '$(["\'@0123456789':
                return expr_part, start + m.end()
    return expr, start


# ===========================================================================
# $? adjacency check — shared by all transforms
# ===========================================================================

def _separate_trailing_comment(line: str, start: int, mask: bytearray
                         ) -> tuple[int, str]:
    """Split off a trailing line comment starting at *start*, if present.

    Returns ``(content_end, comment_str)`` where *content_end* is the index
    at which the content before the comment ends, and *comment_str* is the
    comment text (including the ``#``) or ``""`` if none.
    """
    for ri in range(max(start, 1), len(line)):
        if mask[ri] == 0 and mask[ri - 1] == 1 and line[ri] == "#":
            return ri, line[ri:]
    return len(line), ""


def _after_dollar_question(line: str, op_idx: int) -> bool:
    """Return True if *op_idx* is immediately after ``$?``.

    When True the first ``?`` of the operator at *op_idx* is actually the
    ``?`` of the ``$?`` automatic variable, so the operator should be skipped.

    Does NOT match ``$$?.`` — ``$$`` is a separate automatic variable.
    """
    return (
        op_idx > 0
        and line[op_idx - 1] == "$"
        and not (op_idx > 1 and line[op_idx - 2] == "$")
    )


# ===========================================================================
# Shared colon-context check
# ===========================================================================

def _is_scope_colon(line: str, i: int) -> bool:
    """Return ``True`` if the colon at *i* belongs to a ``$scope:var`` prefix."""
    # Scan backwards from the colon to find the preceding ``$`` variable prefix.
    j = i - 1
    while j >= 0 and (line[j].isalnum() or line[j] == "_"):
        j -= 1
    return j >= 0 and line[j] == "$"


# ===========================================================================
# Expression boundary helpers
# ===========================================================================

def _is_null_conditional_qmark(line: str, i: int) -> bool:
    """Return True if ``?`` at *i* is part of ``?.`` (null-conditional)."""
    return i + 1 < len(line) and line[i + 1] == "."


def _is_double_colon(line: str, i: int) -> bool:
    """Return True if ``:`` at *i* is part of ``::`` (static member access)."""
    return (i + 1 < len(line) and line[i + 1] == ":") or (i > 0 and line[i - 1] == ":")


def _find_expr_start(line: str, end: int, mask: bytearray,
                     extra_stop: str = "") -> int:
    """Scan backwards from *end* to locate the start of the expression.

    *extra_stop* can contain additional delimiter characters (e.g. ``"?:``
    for null-conditional base scanning).
    """
    depth = 0
    stop_set = _EXPR_STOP + extra_stop if extra_stop else _EXPR_STOP
    for i in range(end - 1, -1, -1):
        if not mask[i]:
            continue
        c = line[i]
        if c in _DEPTH_CLOSE:
            depth += 1
        elif c in _DEPTH_OPEN:
            depth -= 1
            if depth < 0:
                return i + 1
        elif depth == 0 and c in stop_set:
            if c == "?":
                if _is_null_conditional_qmark(line, i):
                    continue
                if _after_dollar_question(line, i):
                    continue
            elif c == ":":
                if _is_double_colon(line, i) or _is_scope_colon(line, i):
                    continue
            return i + 1
    return 0


def _find_expr_end(line: str, start: int, mask: bytearray) -> int:
    """Scan forwards from *start* to locate the end of the expression."""
    depth = 0
    # Track mask value at the previous position to detect region boundaries.
    prev_mask = 1 if start == 0 else mask[start - 1]
    for i in range(start, len(line)):
        c = line[i]
        cur_mask = mask[i]
        if not cur_mask:
            # If we just stepped from outside (mask=1) into a region and the
            # character is '#', it is the start of a line comment — stop here.
            if prev_mask and c == "#":
                return i
            prev_mask = cur_mask
            continue
        prev_mask = cur_mask
        if c in _DEPTH_OPEN:
            depth += 1
        elif c in _DEPTH_CLOSE:
            depth -= 1
            if depth < 0:
                return i
        elif depth == 0 and c in _EXPR_STOP:
            return i
    return len(line)


def _expr_left(line: str, pos: int, mask: bytearray,
               extra_stop: str = "") -> tuple[int, int]:
    """Return (start, end) of the expression immediately left of *pos*."""
    end = pos
    while end > 0 and line[end - 1] == " ":
        end -= 1
    start = _find_expr_start(line, end, mask, extra_stop)
    return start, end


def _expr_right(line: str, pos: int, mask: bytearray) -> tuple[int, int]:
    """Return (start, end) of the expression immediately right of *pos*."""
    start = pos
    while start < len(line) and line[start] == " ":
        start += 1
    end = _find_expr_end(line, start, mask)
    return start, end


# ===========================================================================
# Generic operator transform engine
# ===========================================================================

def _find_next_op(line: str, op: str, mask: bytearray,
                  skip_dollar_q: bool = False, start: int = 0,
                  reverse: bool = False) -> int:
    """Find the next (or rightmost when *reverse* is True) occurrence of *op*
    in *line* that is outside regions.

    If *skip_dollar_q* is True, skip positions immediately after ``$?``.
    *start* restricts the search to indices >= *start* (ignored when *reverse*).
    """
    search = line.rfind if reverse else line.find
    idx = search(op, start if not reverse else None)
    while idx != -1:
        if mask[idx] and (not skip_dollar_q or not _after_dollar_question(line, idx)):
            return idx
        idx = search(op, idx + 1) if not reverse else line.rfind(op, 0, idx)
    return -1


def _transform_operator(
    line: str,
    op: str,
    builder: Callable,
    *,
    skip_dollar_q: bool = True,
) -> tuple[str, list[str]]:
    """Generic single-operator line transformer.

    *op*          — the operator string to search for (e.g. ``"??"``).
    *builder*     — callable(left_expr, right_expr, right_extra) → inner_str.
    *skip_dollar_q* — when True, skip positions immediately after ``$?``.

    Returns ``(transformed_line, warnings)``.
    """
    warnings: list[str] = []
    op_len = len(op)
    search = 0
    while True:
        mask = _line_mask(line)
        idx = _find_next_op(line, op, mask, skip_dollar_q, search)
        if idx == -1:
            break

        left_start, left_end = _expr_left(line, idx, mask)
        left_expr = line[left_start:left_end].strip()
        left_expr, left_start = _strip_command_prefix(left_expr, left_start)

        right_start, right_end = _expr_right(line, idx + op_len, mask)
        right_expr = line[right_start:right_end].strip()
        right_extra = None

        if not left_expr or not right_expr:
            search = idx + 1
            continue

        inner = builder(left_expr, right_expr, right_extra)
        warnings.append(f"{op} operator `{left_expr} {op} {right_expr}` rewritten to `{inner}`")
        line = _build_replacement(line[:left_start], inner) + line[right_end:]
        search = left_start
    return line, warnings


# ===========================================================================
# Transform: null-coalescing assignment  (??=)
# ===========================================================================

def _transform_nca_line(line: str) -> tuple[str, list[str]]:
    """Rewrite null-coalescing assignment ``$var ??= value``.

    Uses the generic expression scanner so braced variables (``${foo}``),
    property chains (``$obj.Name``) and indexed targets (``$arr[0]``) are
    all handled correctly.
    """
    warnings: list[str] = []
    search = 0
    while True:
        mask = _line_mask(line)
        idx = _find_next_op(line, "??=", mask, True, search)
        if idx == -1:
            break

        left_start, left_end = _expr_left(line, idx, mask)
        var = line[left_start:left_end].strip()
        if not var:
            search = idx + 1
            continue

        val_start, val_end = _expr_right(line, idx + 3, mask)
        value = line[val_start:val_end].strip()
        new_inner = f"if ($null -eq {var}) {{ {var} = {value} }}"
        warnings.append(
            f"null-coalescing assignment `{var} ??= {value}` "
            f"rewritten to `{new_inner}`"
        )
        line = _build_replacement(line[:left_start].rstrip(), new_inner) + line[val_end:]
        search = left_start
    return line, warnings


# ===========================================================================
# Transform: null-coalescing  (??)
# ===========================================================================

def _transform_nc_line(line: str) -> tuple[str, list[str]]:
    """Transform every ``??`` on *line* into PS 5.1 compatible ``if`` form."""

    def _builder(left: str, right: str, _extra: None) -> str:
        return f"if ($null -ne {left}) {{ {left} }} else {{ {right} }}"

    return _transform_operator(line, "??", _builder)


# ===========================================================================
# Transform: ternary  (? :)
# ===========================================================================

def _find_matching_colon(line: str, start: int, mask: bytearray,
                         depth_arr: list[int]) -> int:
    """Find the colon that separates the true/false branches of a ternary."""
    for i in range(start, len(line)):
        if line[i] != ":" or depth_arr[i] != 0 or not mask[i]:
            continue
        if _is_double_colon(line, i) or _is_scope_colon(line, i):
            continue
        return i
    return -1


def _transform_ternary_line(line: str) -> tuple[str, list[str]]:
    """Rewrite ternary ``$cond ? $true : $false`` into an ``if`` statement."""
    warnings: list[str] = []
    mask = _line_mask(line)
    depth_arr = _compute_depths(line, mask)
    pos = 0
    while pos < len(line):
        if (
            line[pos] == "?"
            and mask[pos]
            and not _after_dollar_question(line, pos)
        ):
            colon_pos = _find_matching_colon(line, pos + 1, mask, depth_arr)
            if colon_pos != -1:
                cond_start, cond_end = _expr_left(line, pos, mask)
                condition = line[cond_start:cond_end].strip()
                true_expr = line[pos + 1:colon_pos].strip()
                false_start, false_end = _expr_right(line, colon_pos + 1, mask)
                false_expr = line[false_start:false_end].strip()
                if not _match_assignment(line[:cond_start]):
                    condition, cond_start = _strip_command_prefix(condition, cond_start, check_keywords=False)
                inner = f"if ({condition}) {{ {true_expr} }} else {{ {false_expr} }}"
                warnings.append(
                    f"ternary operator `{condition} ? {true_expr} : {false_expr}` "
                    f"rewritten to `{inner}`"
                )
                suffix = line[false_end:]
                line = _build_replacement(line[:cond_start], inner) + suffix
                mask = _line_mask(line)
                depth_arr = _compute_depths(line, mask)
                pos = len(line) - len(suffix)
                continue
        pos += 1
    return line, warnings


# ===========================================================================
# Transform: pipeline chain operators  (&& / ||)
# ===========================================================================

def _transform_chain_line(line: str) -> tuple[str, list[str]]:
    """Rewrite pipeline chain operators ``&&`` and ``||``.

    Uses rightmost-first to maintain correct right-associative semantics.
    """
    warnings: list[str] = []
    while True:
        mask = _line_mask(line)
        and_pos = _find_next_op(line, "&&", mask, reverse=True)
        or_pos = _find_next_op(line, "||", mask, reverse=True)
        if and_pos == -1 and or_pos == -1:
            break
        if and_pos > or_pos:
            best_pos, best_op = and_pos, "&&"
        else:
            best_pos, best_op = or_pos, "||"
        condition = "$?" if best_op == "&&" else "-not $?"
        left = line[:best_pos].strip()
        right_start = best_pos + 2
        # Separate a trailing line comment from the right-hand expression
        # so it stays outside the generated ``{ }`` block (a ``#`` inside
        # braces would comment out the closing ``}``).
        right_raw_end, comment = _separate_trailing_comment(line, right_start, mask)
        right = line[right_start:right_raw_end].strip()
        new_line = f"{left}; if ({condition}) {{ {right} }}{comment}"
        warnings.append(
            f"pipeline chain `{left} {best_op} {right}` "
            f"rewritten to `{new_line}`"
        )
        line = new_line
    return line, warnings


# ===========================================================================
# Null-conditional helpers
# ===========================================================================

def _scan_member_name(line: str, ms: int, mask: bytearray) -> int:
    """Scan a member name starting at *ms*; return the index after it."""
    if ms >= len(line):
        return ms
    c0 = line[ms]
    if c0 == "'":
        return _scan_single_quoted(line, ms)
    if c0 == '"':
        return _scan_double_quoted(line, ms)
    if c0 != "$":
        me = ms
        while me < len(line) and (line[me].isalnum() or line[me] == "_"):
            me += 1
        return me
    # Variable member ($var, ${var}, $? etc.)
    me = ms + 1
    if me >= len(line):
        return me
    ch = line[me]
    if ch == "{":
        bd = 1
        me += 1
        while me < len(line) and bd > 0:
            if line[me] == "{":
                bd += 1
            elif line[me] == "}":
                bd -= 1
            me += 1
    elif ch in "?$^":
        me += 1  # single-char automatic variables ($? $$ $^)
    else:
        while me < len(line) and (line[me].isalnum() or line[me] in "_:"):
            me += 1
    return me


def _scan_method_args(line: str, start: int, mask: bytearray) -> tuple[str, int]:
    """If *start* points to ``(``, scan the method argument list.

    Returns ``(args_string, index_after_closing_paren)``.
    """
    j = start
    while j < len(line) and line[j] == " ":
        j += 1
    if j >= len(line) or line[j] != "(":
        return "", start
    d = 1
    k = j + 1
    while k < len(line) and d > 0:
        if mask[k]:
            if line[k] == "(":
                d += 1
            elif line[k] == ")":
                d -= 1
        k += 1
    return line[j:k], k


# ===========================================================================
# Transform: null-conditional  (?. and ?[)
# ===========================================================================

def _transform_null_conditional_dot(line: str) -> tuple[str, list[str]]:
    """Rewrite null-conditional member access (``?.``)."""
    return _transform_null_conditional_line(line, "?.")


def _transform_null_conditional_bracket(line: str) -> tuple[str, list[str]]:
    """Rewrite null-conditional index access (``?[``)."""
    return _transform_null_conditional_line(line, "?[")


def _transform_null_conditional_line(line: str, op: str) -> tuple[str, list[str]]:
    """Rewrite null-conditional member access (``?.``) or index access (``?[``)."""
    warnings: list[str] = []
    is_dot = op == "?."
    op_len = len(op)
    search = 0
    while True:
        mask = _line_mask(line)
        idx = _find_next_op(line, op, mask, True, search)
        if idx == -1:
            return line, warnings

        expr_start, expr_end = _expr_left(line, idx, mask, "?:")
        base = line[expr_start:expr_end].strip()
        base, expr_start = _strip_command_prefix(base, expr_start)
        if not base:
            search = idx + op_len
            continue

        if is_dot:
            segments: list[tuple[str, int]] = []  # (member_expr, end_pos)
            prefixes = [base]  # accumulated dotted prefixes
            cur = idx
            while cur < len(line) - 1 and line[cur:cur + 2] == "?.":
                ms = cur + 2
                while ms < len(line) and line[ms] == " ":
                    ms += 1
                me = _scan_member_name(line, ms, mask)
                if me == ms:
                    break
                mem = line[ms:me]
                args, me = _scan_method_args(line, me, mask)
                segments.append((f".{mem}{args}", me))
                prefixes.append(prefixes[-1] + segments[-1][0])
                cur = me
            if not segments:
                search = idx + op_len
                continue
            # Build nested if chain from innermost to outermost
            full_expr = prefixes[-1]  # full dotted expression
            for pfx in reversed(prefixes[:-1]):
                full_expr = f"if ($null -ne {pfx}) {{ {full_expr} }}"
            inner = f"$({full_expr})"
            end_pos = segments[-1][1]
            orig_expr = base + "?." + "?.".join(seg[0][1:] for seg in segments)
        else:
            bracket_depth = 1
            bracket_end = idx + 2
            while bracket_end < len(line) and bracket_depth > 0:
                c = line[bracket_end]
                if mask[bracket_end]:
                    if c == "[":
                        bracket_depth += 1
                    elif c == "]":
                        bracket_depth -= 1
                bracket_end += 1
            index_expr = line[idx + 2:bracket_end - 1]
            inner = f"$(if ($null -ne {base}) {{ {base}[{index_expr}] }})"
            end_pos = bracket_end
            orig_expr = f"{base}?[{index_expr}]"

        kind = "member access" if is_dot else "index"
        warnings.append(
            f"null-conditional {kind} `{orig_expr}` "
            f"rewritten to `{inner}`"
        )
        line = _build_replacement(line[:expr_start], inner) + line[end_pos:]
        search = 0


# ===========================================================================
# Transform dispatch
# ===========================================================================

_TRANSFORMS = (
    _transform_nca_line,
    _transform_null_conditional_dot,
    _transform_null_conditional_bracket,
    _transform_nc_line,
    _transform_ternary_line,
    _transform_chain_line,
)


# ===========================================================================
# Public API
# ===========================================================================

def pwsh_transform(code: str) -> tuple[str, list[str]]:
    """Transform PowerShell 7.x syntax into PowerShell 5.1 compatible syntax.

    Returns ``(transformed_code, warnings)`` where *warnings* is a list of
    human-readable messages describing each transformation that was applied.
    """
    code = _join_continuation_lines(code)
    lines = code.split("\n")
    mask = _build_region_mask(code)
    multi = _find_multiline_regions(code, mask, lines)

    result: list[str] = []
    all_warnings: list[str] = []

    for i, line in enumerate(lines):
        if i in multi:
            result.append(line)
            continue
        for xform in _TRANSFORMS:
            line, w = xform(line)
            if w:
                all_warnings.extend(f"Line {i + 1}: {msg}" for msg in w)
        result.append(line)

    return "\n".join(result), all_warnings


def _find_multiline_regions(code: str, mask: bytearray, lines: list[str]) -> set[int]:
    """Return a set of line indices that are inside multi-line regions.

    Multi-line regions are strings, comments, or here-strings that span
    across two or more lines. Lines wholly inside such regions must be
    skipped by line-level transforms.
    """
    multi: set[int] = set()
    i = 0
    n = len(code)
    line_idx = 0
    while i < n:
        if mask[i] == 0:
            start_line = line_idx
            while i < n and mask[i] == 0:
                if code[i] == "\n":
                    line_idx += 1
                i += 1
            if line_idx > start_line:
                multi.update(range(start_line, line_idx + 1))
        else:
            if code[i] == "\n":
                line_idx += 1
            i += 1
    return multi


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read()
    result, warnings = pwsh_transform(text)
    for w in warnings:
        print(f"[WARNING] {w}", file=sys.stderr)
    print(result)
