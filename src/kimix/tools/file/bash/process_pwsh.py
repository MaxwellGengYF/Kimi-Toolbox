"""Transform PowerShell 7.x syntax to PowerShell 5.1 compatible syntax.

PowerShell 7 introduced several expression-level operators that do not exist in
PowerShell 5.1:

  * Ternary:          $cond ? $true_expr : $false_expr
  * Null-coalescing:  $a ?? $fallback
  * Null-assign:      $a ??= $default
  * Pipeline chains:  cmd1 && cmd2   /   cmd1 || cmd2
  * Null-conditional: $obj?.Property / $obj?[0]

This module performs a *source-to-source* transformation.  It operates on raw
text rather than an AST because the target environment (5.1) cannot parse the
new syntax in the first place.

The transformer implementation is the canonical pure-Python reference that
lives in ``bin/kimix_native/_shell_compat.py`` (the ``kimix_native`` shim);
this module only re-exports the public API so the transformation logic exists
in exactly one place.
"""

from __future__ import annotations

from kimi_cli.native_loader import get_compat as _native_get_compat

# The canonical pure-Python implementation (the historical body of this
# module) lives in the kimix_native shim so there is exactly one copy of the
# transformation logic.
_shell = _native_get_compat("_shell_compat")
if _shell is None:  # pragma: no cover - shim missing (unbundled install)
    raise ImportError(
        "kimix_native shim unavailable: the pure-Python PowerShell transformer "
        "lives in bin/kimix_native/_shell_compat.py and must be importable. "
        "Install the kimix package with its bundled shim or run from the "
        "repository checkout."
    )

# Re-export the reference implementation's public surface (single source of
# truth in the shim).
pwsh_transform = _shell.pwsh_transform

# Keep the historical module namespace complete for consumers that import the
# transform helpers directly.
_PS_KEYWORDS = _shell._PS_KEYWORDS
_EXPR_STOP = _shell._EXPR_STOP
_DEPTH_OPEN = _shell._DEPTH_OPEN
_DEPTH_CLOSE = _shell._DEPTH_CLOSE
_TRANSFORMS = _shell._TRANSFORMS
_scan_single_quoted = _shell._scan_single_quoted
_scan_double_quoted = _shell._scan_double_quoted
_scan_block_comment = _shell._scan_block_comment
_skip_subexpression = _shell._skip_subexpression
_scan_here_string = _shell._scan_here_string
_build_region_mask = _shell._build_region_mask
_line_mask = _shell._line_mask
_compute_depths = _shell._compute_depths
_join_continuation_lines = _shell._join_continuation_lines
_match_assignment = _shell._match_assignment
_build_replacement = _shell._build_replacement
_strip_command_prefix = _shell._strip_command_prefix
_separate_trailing_comment = _shell._separate_trailing_comment
_after_dollar_question = _shell._after_dollar_question
_is_scope_colon = _shell._is_scope_colon
_is_null_conditional_qmark = _shell._is_null_conditional_qmark
_is_double_colon = _shell._is_double_colon
_find_expr_start = _shell._find_expr_start
_find_expr_end = _shell._find_expr_end
_expr_left = _shell._expr_left
_expr_right = _shell._expr_right
_find_next_op = _shell._find_next_op
_transform_operator = _shell._transform_operator
_transform_nca_line = _shell._transform_nca_line
_transform_nc_line = _shell._transform_nc_line
_find_matching_colon = _shell._find_matching_colon
_transform_ternary_line = _shell._transform_ternary_line
_transform_chain_line = _shell._transform_chain_line
_scan_member_name = _shell._scan_member_name
_scan_method_args = _shell._scan_method_args
_transform_null_conditional_dot = _shell._transform_null_conditional_dot
_transform_null_conditional_bracket = _shell._transform_null_conditional_bracket
_transform_null_conditional_line = _shell._transform_null_conditional_line
_find_multiline_regions = _shell._find_multiline_regions

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
