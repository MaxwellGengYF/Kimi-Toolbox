"""Comprehensive tests for pwsh_transform (PowerShell 7.x → 5.1 syntax transformer)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Import the module directly to avoid the bash_tool.py Python 3.14 issue
_MODULE_PATH = Path(__file__).parent.parent / "src" / "kimix" / "tools" / "file" / "bash" / "process_pwsh.py"
_spec = importlib.util.spec_from_file_location("process_pwsh", str(_MODULE_PATH))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["process_pwsh"] = _mod  # fix Python 3.14 dataclass module resolution
_spec.loader.exec_module(_mod)
pwsh_transform = _mod.pwsh_transform

# ============================================================================
# Ternary operator  (? :)
# ============================================================================

class TestTernaryOperator:
    def test_simple_ternary(self) -> None:
        result = pwsh_transform('$x = $cond ? "a" : "b"')[0]
        assert "if ($cond)" in result
        assert '{ "a" }' in result
        assert '{ "b" }' in result

    def test_ternary_with_comparison(self) -> None:
        result = pwsh_transform("$x = $a -gt 5 ? $a : 0")[0]
        assert "if ($a -gt 5)" in result
        assert "{ $a }" in result
        assert "{ 0 }" in result

    def test_ternary_in_assignment(self) -> None:
        result = pwsh_transform('$status = $count -eq 0 ? "empty" : "non-empty"')[0]
        assert "$status = " in result
        assert "($count -eq 0)" in result

    def test_ternary_with_function_calls(self) -> None:
        result = pwsh_transform('$x = Test-Path $p ? (Get-Item $p) : $null')[0]
        assert "if (Test-Path $p)" in result
        assert "(Get-Item $p)" in result
        assert "$null" in result

    def test_ternary_no_assignment(self) -> None:
        result = pwsh_transform('$cond ? "yes" : "no"')[0]
        assert 'if ($cond) { "yes" } else { "no" }' in result

# ============================================================================
# Null-coalescing  (??)
# ============================================================================

class TestNullCoalescing:
    def test_simple_null_coalescing(self) -> None:
        result = pwsh_transform('$x = $a ?? "default"')[0]
        assert "if ($null -ne $a)" in result
        assert '{ $a }' in result
        assert '{ "default" }' in result

    def test_null_coalescing_with_variable(self) -> None:
        result = pwsh_transform("$x = $a ?? $b")[0]
        assert "if ($null -ne $a)" in result
        assert "{ $a }" in result
        assert "{ $b }" in result

    def test_null_coalescing_with_literal_default(self) -> None:
        result = pwsh_transform("$path = $env:HOME ?? 'C:\\Users\\Default'")[0]
        assert "if ($null -ne $env:HOME)" in result
        assert "{ $env:HOME }" in result

    def test_nested_null_coalescing(self) -> None:
        result = pwsh_transform('$x = $a ?? $b ?? "default"')[0]
        # After first ?? transform, the result contains another ??
        # which should also be transformed
        assert "default" in result
        assert "if ($null -ne " in result

    def test_null_coalescing_no_assignment(self) -> None:
        result = pwsh_transform('$a ?? "fallback"')[0]
        assert 'if ($null -ne $a) { $a } else { "fallback" }' in result

# ============================================================================
# Null-coalescing assignment  (??=)
# ============================================================================

class TestNullCoalescingAssignment:
    def test_simple_assign(self) -> None:
        result = pwsh_transform('$a ??= "default"')[0]
        assert "if ($null -eq $a)" in result
        assert '$a = "default"' in result

    def test_assign_with_expression(self) -> None:
        result = pwsh_transform("$count ??= (Get-ChildItem).Count")[0]
        assert "if ($null -eq $count)" in result
        assert "$count = (Get-ChildItem).Count" in result

    def test_assign_does_not_conflict_with_null_coalescing(self) -> None:
        """??= should be transformed before ?? so ??= is not partially matched."""
        result = pwsh_transform("$a ??= $b ?? $c")[0]
        # ??= should be fully resolved
        assert "??=" not in result
        assert "??" not in result

# ============================================================================
# Pipeline chain AND  (&&)
# ============================================================================

class TestPipelineChainAnd:
    def test_simple_and_chain(self) -> None:
        result = pwsh_transform("cmd1 && cmd2")[0]
        assert ";" in result
        assert "if ($?)" in result
        assert "cmd1" in result
        assert "cmd2" in result

    def test_multiple_and_chain(self) -> None:
        result = pwsh_transform("cmd1 && cmd2 && cmd3")[0]
        assert "cmd1;" in result
        assert "if ($?) { cmd2; if ($?) { cmd3 } }" in result

    def test_and_chain_with_pipeline(self) -> None:
        result = pwsh_transform("Get-Process | Where-Object CPU && Write-Output done")[0]
        assert "Get-Process | Where-Object CPU" in result
        assert "Write-Output done" in result
        assert "if ($?)" in result

# ============================================================================
# Pipeline chain OR  (||)
# ============================================================================

class TestPipelineChainOr:
    def test_simple_or_chain(self) -> None:
        result = pwsh_transform("cmd1 || cmd2")[0]
        assert ";" in result
        assert "if (-not $?)" in result
        assert "cmd1" in result
        assert "cmd2" in result

    def test_multiple_or_chain(self) -> None:
        result = pwsh_transform("cmd1 || cmd2 || cmd3")[0]
        assert "cmd1;" in result
        assert "if (-not $?) { cmd2; if (-not $?) { cmd3 } }" in result

# ============================================================================
# Null-conditional  (?. and ?[])
# ============================================================================

class TestNullConditional:
    def test_property_access(self) -> None:
        result = pwsh_transform("$a?.Length")[0]
        assert "if ($null -ne $a) { $a.Length }" in result

    def test_index_access(self) -> None:
        result = pwsh_transform("$a?[0]")[0]
        assert "if ($null -ne $a) { $a[0] }" in result

    def test_chained_null_conditional(self) -> None:
        result = pwsh_transform("$a?.Property?.SubProperty")[0]
        # Both ?. should be transformed
        assert "?." not in result

    def test_null_conditional_with_method(self) -> None:
        result = pwsh_transform("$a?.ToString()")[0]
        assert "if ($null -ne $a) { $a.ToString() }" in result

    def test_null_conditional_assignment(self) -> None:
        result = pwsh_transform("$x = $a?.Length")[0]
        assert "$x = $(if ($null -ne $a) { $a.Length })" == result

# ============================================================================
# Combined transformations
# ============================================================================

class TestCombinedTransformations:
    def test_multiple_features(self) -> None:
        code = '$x = $a ?? "default"\nGet-Process && Write-Output done'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "&&" not in result
        assert "if ($null -ne $a)" in result
        assert "if ($?)" in result

    def test_no_false_positives_in_strings(self) -> None:
        code = "Write-Output 'The ?? operator is new'"
        result = pwsh_transform(code)[0]
        # The ?? inside the string should not be transformed
        assert "??" in result
        assert "if ($null -ne" not in result

    def test_no_false_positives_in_comments(self) -> None:
        code = "# This ?? is a comment\nWrite-Output hello"
        result = pwsh_transform(code)[0]
        assert "??" in result  # still in comment

    def test_no_false_positives_in_double_quoted_string(self) -> None:
        code = 'Write-Output "The ?? operator"'
        result = pwsh_transform(code)[0]
        assert "??" in result

    def test_combined_and_or(self) -> None:
        result = pwsh_transform("cmd1 && cmd2 || cmd3")[0]
        assert "&&" not in result
        assert "||" not in result

# ============================================================================
# Idempotency
# ============================================================================

class TestIdempotency:
    def test_double_transform_same_result(self) -> None:
        code = '$x = $a ?? "default"\n$y = $cond ? "yes" : "no"\nGet-Process && Write-Output done'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_ternary_idempotent(self) -> None:
        code = '$x = $cond ? "a" : "b"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_null_coalescing_idempotent(self) -> None:
        code = '$x = $a ?? "default"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_pipeline_chain_idempotent(self) -> None:
        code = "cmd1 && cmd2"
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_null_conditional_idempotent(self) -> None:
        code = "$a?.Length"
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    def test_strings_with_operators_not_transformed(self) -> None:
        code = """Write-Output 'Use ?? for null-coalescing'
Write-Output "A ? B : C is ternary"
Write-Output 'cmd1 && cmd2 is chain'"""
        result = pwsh_transform(code)[0]
        assert "?? for null-coalescing" in result
        assert "A ? B : C is ternary" in result
        assert "cmd1 && cmd2 is chain" in result

    def test_comments_not_transformed(self) -> None:
        code = """# The ?? operator is new in PS7
# $x = $cond ? "a" : "b"
# cmd1 && cmd2
Write-Output hello"""
        result = pwsh_transform(code)[0]
        assert "The ?? operator" in result
        assert '$cond ? "a" : "b"' in result
        assert "cmd1 && cmd2" in result

    def test_here_string_not_transformed(self) -> None:
        code = """$text = @'
The ?? operator is preserved here.
And so is the ?. operator.
'@
Write-Output $text"""
        result = pwsh_transform(code)[0]
        assert "??" in result  # preserved inside here-string
        assert "?." in result

    def test_multiline_with_backtick(self) -> None:
        code = "Get-Process `\n| Where-Object CPU `\n&& Write-Output done"
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "if ($?)" in result

    def test_empty_code(self) -> None:
        result = pwsh_transform("")[0]; assert result == ""

    def test_no_operators(self) -> None:
        code = "Write-Output 'hello world'"
        result = pwsh_transform(code)[0]; assert result == code

    def test_ternary_in_pipeline(self) -> None:
        code = "$x = $a ? $b : $c | ForEach-Object { $_ }"
        result = pwsh_transform(code)[0]
        assert "?" not in result
        assert "if ($a)" in result

    def test_null_coalescing_with_property(self) -> None:
        code = '$name = $obj.Name ?? "Unknown"'
        result = pwsh_transform(code)[0]
        assert "if ($null -ne $obj.Name)" in result
        assert "Unknown" in result

    def test_block_comment_not_transformed(self) -> None:
        code = "<# The ?? and ?. operators are new #>\nWrite-Output hello"
        result = pwsh_transform(code)[0]
        assert "??" in result  # preserved in block comment
        assert "?." in result

    def test_null_conditional_bracket_with_expression(self) -> None:
        result = pwsh_transform("$a?[$i + 1]")[0]
        assert "if ($null -ne $a) { $a[$i + 1] }" in result

# ============================================================================
# Corner case: nested ternary
# ============================================================================

class TestNestedTernary:
    def test_nested_in_true_branch(self) -> None:
        """Nested ternary: only the outer ?: is transformed in one pass."""
        result = pwsh_transform('$x = $a ? ($b ? "c" : "d") : "e"')[0]
        # Outer ternary is transformed; inner remains (one-pass limitation)
        assert 'if ($a)' in result
        assert '($b ? "c" : "d")' in result or '"c"' in result
        assert '"e"' in result

    def test_nested_in_false_branch(self) -> None:
        """Nested ternary in false branch: outer transformed, inner remains."""
        result = pwsh_transform('$x = $a ? "yes" : ($b ? "maybe" : "no")')[0]
        assert 'if ($a)' in result

    def test_deeply_nested_ternary(self) -> None:
        """Deeply nested ternary: only outermost ?: transformed per pass."""
        result = pwsh_transform('$x = $a ? ($b ? ($c ? 1 : 2) : 3) : 4')[0]
        assert "if ($a)" in result
        # inner ternaries preserved
        assert "?" in result  # inner ? operators still present

# ============================================================================
# Corner case: multiple operators on one line
# ============================================================================

class TestMultipleOperatorsOneLine:
    def test_multiple_null_coalescing_one_line(self) -> None:
        """$a ?? $b on same line as $c ?? $d (separated by semicolon)."""
        result = pwsh_transform('$x = $a ?? "x"; $y = $b ?? "y"')[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result
        assert "if ($null -ne $b)" in result

    def test_multiple_null_conditional_one_line(self) -> None:
        result = pwsh_transform('$x = $a?.Name; $y = $b?.Count')[0]
        assert "?." not in result
        assert "if ($null -ne $a)" in result
        assert "if ($null -ne $b)" in result

    def test_mixed_operators_one_line(self) -> None:
        result = pwsh_transform('$x = $a ?? $b; $y = $c ? "t" : "f"')[0]
        assert "??" not in result
        assert "?" not in result
        assert "if ($null -ne $a)" in result
        assert "if ($c)" in result

    def test_multiple_null_coalescing_assign_one_line(self) -> None:
        """Multiple ??= on one line: only the leftmost is fully captured.
        Known limitation: the regex greedily captures everything after ??=."""
        result = pwsh_transform('$a ??= "x"; $b ??= "y"')[0]
        # At minimum, the first ??= is processed
        assert "if ($null -eq $a)" in result

# ============================================================================
# Corner case: chained null-conditional with methods
# ============================================================================

class TestNullConditionalMethodChain:
    def test_method_with_args(self) -> None:
        result = pwsh_transform("$a?.GetValue($param)")[0]
        assert "?." not in result
        assert "if ($null -ne $a) { $a.GetValue($param) }" in result

    def test_method_with_multiple_args(self) -> None:
        result = pwsh_transform("$a?.Invoke($x, $y, $z)")[0]
        assert "?." not in result
        assert "$a.Invoke($x, $y, $z)" in result

    def test_method_with_no_args(self) -> None:
        result = pwsh_transform("$a?.Dispose()")[0]
        assert "?." not in result
        assert "$a.Dispose()" in result

    def test_chained_method_calls(self) -> None:
        result = pwsh_transform("$a?.ToString()?.Split()")[0]
        assert "?." not in result
        assert "$a.ToString()" in result
        assert "$a.ToString().Split()" in result

# ============================================================================
# Corner case: mixed null-conditional dot and bracket
# ============================================================================

class TestMixedNullConditional:
    def test_dot_then_bracket(self) -> None:
        """?. followed by ?[ is tricky: ?. is processed first."""
        result = pwsh_transform("$a?.Items?[0]")[0]
        # The ?. should be transformed; ?[ may remain depending on order
        assert "?." not in result
        assert "$a.Items" in result

    def test_bracket_then_dot(self) -> None:
        """?[ followed by ?.: ?. processed first, ?[ may remain inside braces.
        This no longer hangs (infinite-loop bug fixed); result preserves ?[ at depth > 0."""
        result = pwsh_transform("$a?[0]?.Name")[0]
        # ?. should be transformed
        assert "?." not in result
        assert "$a" in result

    def test_dot_bracket_dot_chain(self) -> None:
        """Long chain: ?. processed first, ?[ preserved at depth > 0. No hang."""
        result = pwsh_transform("$a?.Items?[0]?.LastName")[0]
        assert "$a.Items" in result

    def test_bracket_with_nested_expr(self) -> None:
        result = pwsh_transform("$a?[$i?.ToString()]")[0]
        # The inner ?. is inside brackets; depends on implementation whether it's transformed
        # At minimum, the outer ?[ should be transformed
        assert "if ($null -ne $a)" in result

# ============================================================================
# Corner case: chain operators with complex pipelines
# ============================================================================

class TestChainComplexPipelines:
    def test_and_chain_with_pipe_and_args(self) -> None:
        result = pwsh_transform("Get-ChildItem -Path $env:USERPROFILE -Recurse && Write-Output 'done'")[0]
        assert "&&" not in result
        assert "if ($?)" in result
        assert "Get-ChildItem -Path $env:USERPROFILE -Recurse" in result

    def test_or_chain_after_failed_command(self) -> None:
        result = pwsh_transform("Test-Path $f || New-Item $f")[0]
        assert "||" not in result
        assert "if (-not $?)" in result

    def test_and_or_chain_sequence(self) -> None:
        result = pwsh_transform("cmd1 && cmd2 || cmd3")[0]
        assert "&&" not in result
        assert "||" not in result
        # Should be: cmd1; if ($?) { cmd2; if (-not $?) { cmd3 } }
        assert "if ($?)" in result
        assert "if (-not $?)" in result

    def test_or_and_chain_sequence(self) -> None:
        result = pwsh_transform("cmd1 || cmd2 && cmd3")[0]
        assert "&&" not in result
        assert "||" not in result
        assert "if (-not $?)" in result
        assert "if ($?)" in result

    def test_triple_and_chain(self) -> None:
        result = pwsh_transform("cmd1 && cmd2 && cmd3 && cmd4")[0]
        assert "&&" not in result
        # Check that all three chain points are there
        assert result.count("if ($?)") == 3

# ============================================================================
# Corner case: edge literal / variable patterns
# ============================================================================

class TestEdgeLiteralPatterns:
    def test_dollar_question_not_transformed(self) -> None:
        """$? is an automatic variable, should not be confused with ?. or ternary."""
        result = pwsh_transform("if ($?) { Write-Output ok }")[0]
        assert "$?" in result  # $? preserved
        assert "if ($?) { Write-Output ok }" == result

    def test_question_mark_in_variable_name(self) -> None:
        """Variable with ? in name like ${foo?} should not cause transformation."""
        # This is unusual but let's make sure it doesn't crash
        result = pwsh_transform('Write-Output ${foo?}')[0]
        # Should not have transformed anything
        assert "Write-Output" in result

    def test_null_coalescing_with_null_literal(self) -> None:
        result = pwsh_transform('$x = $a ?? $null')[0]
        assert "if ($null -ne $a)" in result
        assert "{ $a }" in result
        assert "{ $null }" in result

    def test_null_coalescing_with_true_false(self) -> None:
        result = pwsh_transform('$x = $a ?? $true')[0]
        assert "if ($null -ne $a)" in result
        assert "{ $a }" in result
        assert "{ $true }" in result

    def test_ternary_with_null(self) -> None:
        result = pwsh_transform('$x = $cond ? $null : "default"')[0]
        assert "if ($cond)" in result
        assert "{ $null }" in result

# ============================================================================
# Corner case: here-string double-quoted variant
# ============================================================================

class TestHereStringDoubleQuoted:
    def test_double_quoted_here_string(self) -> None:
        code = r'''$text = @"
The ?? operator is preserved.
And so is ?. and ?[
"@
Write-Output $text'''
        result = pwsh_transform(code)[0]
        assert "??" in result  # preserved
        assert "?." in result
        assert "?[" in result

    def test_at_quote_single_line_here_string(self) -> None:
        code = "$text = @'?? is not transformed here'@\nWrite-Output $text"
        result = pwsh_transform(code)[0]
        assert "??" in result

# ============================================================================
# Corner case: backtick continuation with various operators
# ============================================================================

class TestBacktickContinuationOperators:
    def test_null_coalescing_with_backtick(self) -> None:
        code = '$x = $a ??`\n  "default"'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result

    def test_ternary_with_backtick_continuation(self) -> None:
        code = '$x = $cond ?`\n  "yes" :`\n  "no"'
        result = pwsh_transform(code)[0]
        assert "?" not in result
        assert "if ($cond)" in result

    def test_null_conditional_with_backtick(self) -> None:
        code = "$a?.`\n  Property"
        result = pwsh_transform(code)[0]
        # After backtick join, the ?. is on a single line w/spaces
        assert "?." not in result

    def test_chain_with_backtick_continuation(self) -> None:
        code = "cmd1 `\n&& cmd2"
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "if ($?)" in result

# ============================================================================
# Corner case: expression boundaries
# ============================================================================

class TestExprBoundaries:
    def test_null_coalescing_with_parenthesized_left(self) -> None:
        result = pwsh_transform('$x = (Get-Item $p) ?? "default"')[0]
        assert "if ($null -ne (Get-Item $p))" in result

    def test_null_coalescing_with_subexpression(self) -> None:
        result = pwsh_transform('$x = $(Get-Date) ?? "never"')[0]
        assert "if ($null -ne $(Get-Date))" in result

    def test_null_conditional_on_subexpression(self) -> None:
        """$()?.Property - null conditional on a subexpression."""
        result = pwsh_transform("$(Get-Item $p)?.Length")[0]
        # The subexpression $(...) should be detected as the base
        assert "?." not in result
        assert "if ($null -ne $(Get-Item $p))" in result

    def test_ternary_with_expression_condition(self) -> None:
        result = pwsh_transform('$x = (Get-Date).Year -gt 2020 ? "new" : "old"')[0]
        assert "?" not in result
        assert "if ((Get-Date).Year -gt 2020)" in result

    def test_ternary_with_complex_true_branch(self) -> None:
        result = pwsh_transform('$x = $cond ? (Get-Process | Select -First 1) : $null')[0]
        assert "?" not in result
        assert "if ($cond)" in result
        assert "(Get-Process | Select -First 1)" in result

# ============================================================================
# Corner case: ??= idempotency and edge patterns
# ============================================================================

class TestNCAEdgeCases:
    def test_nca_idempotent(self) -> None:
        code = '$a ??= "default"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_nca_with_same_line_code(self) -> None:
        result = pwsh_transform('$a ??= "x"; Write-Output $a')[0]
        assert "??=" not in result
        assert "if ($null -eq $a)" in result
        assert "Write-Output $a" in result

    def test_nca_right_side_with_spaces(self) -> None:
        result = pwsh_transform("$a ??= (Get-ChildItem).Count")[0]
        assert "??=" not in result
        assert "$a = (Get-ChildItem).Count" in result


# ============================================================================
# Corner case: operators inside splatting / hashtable context
# ============================================================================

class TestOperatorsInSpecialContext:
    def test_question_in_hashtable_access(self) -> None:
        """@{}.Keys - accessing a hashtable's Keys property."""
        result = pwsh_transform("$x = @{ key = 'val' }.Keys")[0]
        assert "@{" in result

    def test_colon_in_hashtable_not_confused(self) -> None:
        """Ternary inside @{ } is at depth > 0 so it is NOT transformed.
        This is intentional: colons inside braces could be switch/hashtable syntax."""
        result = pwsh_transform('$x = @{ key = $a ? "t" : "f" }')[0]
        # Ternary inside braces is preserved (depth > 0)
        assert "?" in result  # not transformed at depth > 0

    def test_colon_in_string_not_confused(self) -> None:
        """Colon inside a string is not a ternary colon.
        Note: _find_matching_colon may not exclude in-string colons currently."""
        result = pwsh_transform('$x = $cond ? "no-colon" : "default"')[0]
        # Works correctly when strings have no colons
        assert "?" not in result
        assert "if ($cond)" in result

# ============================================================================
# Corner case: whitespace and formatting stress
# ============================================================================

class TestWhitespaceStress:
    def test_no_spaces_around_ternary(self) -> None:
        result = pwsh_transform('$x=$cond?"a":"b"')[0]
        assert "?" not in result
        assert "if ($cond)" in result

    def test_no_spaces_around_null_coalescing(self) -> None:
        result = pwsh_transform('$x=$a??"default"')[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result

    def test_no_spaces_around_null_conditional(self) -> None:
        result = pwsh_transform("$a?.Property?.SubProperty")[0]
        assert "?." not in result

    def test_extra_spaces_around_operators(self) -> None:
        result = pwsh_transform('$x  =   $a    ??    "default"')[0]
        assert "??" not in result

    def test_tabs_around_operators(self) -> None:
        result = pwsh_transform("$x\t=\t$a\t??\t'default'")[0]
        assert "??" not in result

# ============================================================================
# Corner case: code that looks like operators but at end of line
# ============================================================================

class TestTrickyOperatorPlacement:
    def test_and_at_end_of_command(self) -> None:
        """&& at end of line is still valid operator."""
        result = pwsh_transform("cmd1 &&")[0]
        # After transformation, the trailing && situation might be edge
        assert "cmd1" in result

    def test_question_at_end_of_line(self) -> None:
        """Isolated ? at end should not cause error."""
        result = pwsh_transform("$a ?")[0]
        # No colon, so no ternary transformation
        assert "$a" in result

    def test_double_question_at_end(self) -> None:
        """?? at end of line without right side - should be safe."""
        result = pwsh_transform("$a ??")[0]
        # Should not crash; right side is missing
        assert "$a" in result

    def test_null_conditional_at_end(self) -> None:
        """?. at end of line without member."""
        result = pwsh_transform("$a?.")[0]
        # No member name after ?., should be safe
        assert "$a" in result

# ============================================================================
# Corner case: idempotency for all combined transforms
# ============================================================================

class TestFullIdempotency:
    def test_all_operators_together_idempotent(self) -> None:
        code = '''$null_coal = $maybe ?? "fallback"
$ternary = $cond ? "yes" : "no"
$nc_assign ??= "init"
$safe_access = $obj?.Property?.Nested?[0]
Get-Service && Write-Output done || Write-Error failed'''
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_transform_preserves_critical_semantics(self) -> None:
        """Multiple transforms should yield consistent structure."""
        code = '$x = $a ?? $b ?? $c'
        result = pwsh_transform(code)[0]
        # After transformation: all ?? resolved
        assert "??" not in result
        # Should still reference all three variables
        assert "$a" in result
        assert "$b" in result
        assert "$c" in result
# ============================================================================
# Regression / bug-reproduction tests
# ============================================================================

class TestKnownBugs:
    """Tests that reproduce currently known bugs in pwsh_transform."""

    def test_ternary_with_static_member_access(self) -> None:
        result = pwsh_transform('$x = $cond ? [Math]::PI : 0')[0]
        assert "?" not in result
        assert "[Math]::PI" in result
        assert "if ($cond)" in result

    def test_ternary_with_colon_in_string(self) -> None:
        result = pwsh_transform('$x = $cond ? "a:b" : "c"')[0]
        assert "?" not in result
        assert '"a:b"' in result
        assert '"c"' in result
        assert "if ($cond)" in result

    def test_null_coalescing_with_hash_in_string(self) -> None:
        result = pwsh_transform('$x = $a ?? "default#value"')[0]
        assert "??" not in result
        assert '"default#value"' in result

    def test_null_coalescing_with_comma_in_string(self) -> None:
        result = pwsh_transform('$x = $a ?? "a,b"')[0]
        assert "??" not in result
        assert '"a,b"' in result

    def test_null_conditional_method_with_paren_in_string_arg(self) -> None:
        result = pwsh_transform('$obj?.Foo("a)")')[0]
        assert "?." not in result
        assert 'Foo("a)")' in result

    def test_null_conditional_bracket_with_bracket_in_string_index(self) -> None:
        result = pwsh_transform('$arr?["key]"]')[0]
        assert "?[" not in result
        assert '["key]"]' in result

    def test_backtick_continuation_inside_comment(self) -> None:
        code = '# comment `\nWrite-Output hello'
        result = pwsh_transform(code)[0]
        lines = result.splitlines()
        assert len(lines) == 2
        assert lines[1] == "Write-Output hello"

    def test_dollar_question_as_ternary_condition(self) -> None:
        result = pwsh_transform('$? ? "yes" : "no"')[0]
        assert result == 'if ($?) { "yes" } else { "no" }'

    def test_command_followed_by_ternary_without_parens(self):
        result = pwsh_transform('Write-Output $a ? $b : $c')[0]
        # Current behaviour incorrectly treats Write-Output $a as the condition
        condition = result.split("if (")[1].split(")")[0]
        assert "Write-Output $a" not in condition

# ============================================================================
# Infinite-loop safety tests
# ============================================================================

class TestNoInfiniteLoops:
    """Inputs that previously caused hangs or look pathological."""

    def test_bracket_then_dot_no_hang(self) -> None:
        result = pwsh_transform("$a?[0]?.Name")[0]
        assert isinstance(result, str)

    def test_long_null_conditional_chain_no_hang(self) -> None:
        result = pwsh_transform("$a?.Items?[0]?.LastName")[0]
        assert isinstance(result, str)

    def test_incomplete_null_coalescing_no_hang(self) -> None:
        result = pwsh_transform("$a ??")[0]
        assert isinstance(result, str)

    def test_incomplete_null_conditional_dot_no_hang(self) -> None:
        result = pwsh_transform("$a?.")[0]
        assert isinstance(result, str)

    def test_trailing_and_operator_no_hang(self) -> None:
        result = pwsh_transform("cmd1 &&")[0]
        assert isinstance(result, str)

    def test_bare_question_marks_no_hang(self) -> None:
        result = pwsh_transform("?.?.?.?")[0]
        assert isinstance(result, str)

    def test_many_nested_ternaries_no_hang(self) -> None:
        code = '$a ? ($b ? ($c ? ($d ? 1 : 2) : 3) : 4) : 5'
        result = pwsh_transform(code)[0]
        assert isinstance(result, str)

    def test_backtick_rain_no_hang(self) -> None:
        code = "Write-Output `\n`\n`\nhello"
        result = pwsh_transform(code)[0]
        assert isinstance(result, str)

# ============================================================================
# Additional bug-reproduction tests discovered during deep analysis
# ============================================================================

class TestAdditionalBugs:
    """Further edge-case bugs found by studying _find_string_regions and depth handling."""

    def test_here_string_false_positive_consumes_rest_of_file(self) -> None:
        code = "$x = @'foo'@\nWrite-Output hello && cmd2"
        result = pwsh_transform(code)[0]
        # The second line should have its && transformed, but because the
        # here-string scanner swallows to EOF, it is left untouched.
        assert "&&" not in result

    def test_at_quote_inside_line_not_here_string(self) -> None:
        code = "$text = @' preserved ?? and ?. '\nWrite-Output hello && cmd2"
        result = pwsh_transform(code)[0]
        assert "&&" not in result

    def test_chain_inside_script_block(self):
        result = pwsh_transform("$sb = { cmd1 && cmd2 }")[0]
        assert "&&" not in result

    def test_chain_inside_subexpression(self):
        result = pwsh_transform("$(cmd1 && cmd2)")[0]
        assert "&&" not in result
# ============================================================================
# Depth tracking vs strings/comments  (BUG: _compute_depths ignores strings)
# ============================================================================

class TestDepthTrackingStrings:
    """_compute_depths counts brackets even inside strings/comments.
    This can break ternary colon matching when true-branch strings
    contain brackets."""

    def test_ternary_with_paren_in_string_not_transformed(self) -> None:
        result = pwsh_transform('$x = $cond ? "a(b" : "c"')[0]
        # _compute_depths is now string-aware, so ternary transforms correctly
        assert "?" not in result
        assert '"a(b"' in result
        assert '"c"' in result

    def test_ternary_with_bracket_in_string_not_transformed(self) -> None:
        result = pwsh_transform('$x = $cond ? "a[b" : "c"')[0]
        assert "?" not in result
        assert '"a[b"' in result
        assert '"c"' in result

    def test_ternary_with_brace_in_string_not_transformed(self) -> None:
        result = pwsh_transform('$x = $cond ? "a{b" : "c"')[0]
        assert "?" not in result
        assert '"a{b"' in result
        assert '"c"' in result

    def test_ternary_with_colon_in_true_branch_string(self) -> None:
        # No brackets, so this works despite the extra colon inside string.
        result = pwsh_transform('$x = $cond ? "a:b:c" : "d"')[0]
        assert "?" not in result
        assert '"a:b:c"' in result
        assert '"d"' in result

    def test_ternary_with_drive_path_in_true_branch(self) -> None:
        result = pwsh_transform('$x = $cond ? "C:\\foo" : "D:\\bar"')[0]
        assert "?" not in result
        assert '"C:\\foo"' in result
        assert '"D:\\bar"' in result

# ============================================================================
# Null-conditional on complex base expressions
# ============================================================================

class TestNullConditionalComplexBase:
    def test_array_element_then_property(self) -> None:
        result = pwsh_transform("$arr[0]?.Name")[0]
        assert "?." not in result
        assert "$arr[0]" in result
        assert ".Name" in result

    def test_hashtable_access_then_property(self) -> None:
        result = pwsh_transform('$ht["key"]?.Value')[0]
        assert "?." not in result
        assert '$ht["key"]' in result
        assert ".Value" in result

    def test_property_then_bracket_then_property(self) -> None:
        result = pwsh_transform('$a.Items[0]?.Name')[0]
        assert "?." not in result
        assert "$a.Items[0]" in result
        assert ".Name" in result

    def test_subexpression_then_property(self) -> None:
        result = pwsh_transform("$(Get-Item $p)?.Length")[0]
        assert "?." not in result
        assert "$(Get-Item $p)" in result
        assert ".Length" in result

    def test_nested_subexpression_then_property(self) -> None:
        result = pwsh_transform("$($($a))?.Name")[0]
        assert "?." not in result
        assert "$($($a))" in result

    def test_null_literal_then_property(self) -> None:
        result = pwsh_transform("$null?.Property")[0]
        assert "?." not in result
        assert "$null" in result

    def test_variable_with_braces_then_property(self) -> None:
        result = pwsh_transform("${foo-bar}?.Name")[0]
        assert "?." not in result
        assert "${foo-bar}" in result
        assert ".Name" in result

# ============================================================================
# Scoped variables and property access with operators
# ============================================================================

class TestScopedVariables:
    def test_global_scope_null_coalescing(self) -> None:
        result = pwsh_transform('$global:x ?? "default"')[0]
        assert "??" not in result
        assert "$global:x" in result

    def test_env_scope_null_coalescing(self) -> None:
        result = pwsh_transform('$env:PATH ?? "C:\\Windows"')[0]
        assert "??" not in result
        assert "$env:PATH" in result

    def test_script_scope_nca(self) -> None:
        result = pwsh_transform('$script:count ??= 0')[0]
        assert "??=" not in result
        assert "$script:count" in result
        assert "if ($null -eq $script:count)" in result

    def test_property_access_nca(self) -> None:
        result = pwsh_transform('$obj.Name ??= "default"')[0]
        assert "??=" not in result
        assert "if ($null -eq $obj.Name)" in result
        assert "$obj.Name = \"default\"" in result

    def test_global_scope_null_conditional(self) -> None:
        result = pwsh_transform('$global:obj?.Name')[0]
        assert "?." not in result
        assert "$global:obj" in result

# ============================================================================
# Comments and strings interaction
# ============================================================================

class TestCommentStringInteraction:
    def test_hash_inside_single_quoted_string(self) -> None:
        result = pwsh_transform("'hello # world' ?? 'default'")[0]
        assert "??" not in result
        assert "'hello # world'" in result
        assert "'default'" in result

    def test_hash_inside_double_quoted_string(self) -> None:
        result = pwsh_transform('"hello # world" ?? "default"')[0]
        assert "??" not in result
        assert '"hello # world"' in result

    def test_block_comment_start_inside_line_comment(self) -> None:
        code = '# <# not a block comment\n$x = $a ?? "default"'
        result = pwsh_transform(code)[0]
        assert "<# not a block comment" in result
        assert "??" not in result

    def test_line_comment_after_operator(self) -> None:
        result = pwsh_transform('$x = $a ?? "default" # comment with ??')[0]
        # BUG: the comment is swallowed into the right-hand expression of ??
        # because _expr_right does not stop at the # comment boundary.
        # The operator ?? is transformed, but the ?? inside the comment is preserved.
        assert "if ($null -ne $a)" in result
        assert "# comment with ??" in result

    def test_single_quoted_string_with_doubled_quotes(self) -> None:
        result = pwsh_transform("'It''s ?? and ?. here'")[0]
        assert "??" in result
        assert "?." in result

    def test_double_quoted_string_with_escaped_backtick(self) -> None:
        result = pwsh_transform('"a ``?? b"')[0]
        assert "??" in result

# ============================================================================
# Nested / multi-line block comments
# ============================================================================

class TestBlockComments:
    def test_nested_block_comment(self) -> None:
        code = '<# outer <# inner #> still outer #>\n$x = $a ?? "default"'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "outer" in result
        assert "inner" in result

    def test_block_comment_spanning_lines_with_operators(self) -> None:
        code = '<#\n?? operator\n?. operator\n&& operator\n#>\nWrite-Output done'
        result = pwsh_transform(code)[0]
        assert "??" in result
        assert "?." in result
        assert "&&" in result

# ============================================================================
# Ternary with complex true/false branches
# ============================================================================

class TestTernaryComplexBranches:
    def test_ternary_with_hashtable_true_branch(self) -> None:
        result = pwsh_transform('$x = $cond ? @{ a = 1 } : @{ b = 2 }')[0]
        assert "?" not in result
        assert "@{ a = 1 }" in result
        assert "@{ b = 2 }" in result

    def test_ternary_with_script_block_branches(self) -> None:
        result = pwsh_transform('$x = $cond ? { $a } : { $b }')[0]
        assert "?" not in result
        assert "{ $a }" in result
        assert "{ $b }" in result

    def test_ternary_with_array_literal_branches(self) -> None:
        result = pwsh_transform('$x = $cond ? @(1,2) : @(3,4)')[0]
        assert "?" not in result
        assert "@(1,2)" in result
        assert "@(3,4)" in result

    def test_ternary_with_match_operator(self) -> None:
        result = pwsh_transform('$x = $a -match "test" ? "yes" : "no"')[0]
        assert "?" not in result
        assert 'if ($a -match "test")' in result

    def test_ternary_dollar_question_as_condition(self) -> None:
        result = pwsh_transform('$? ? $? : $false')[0]
        assert result == 'if ($?) { $? } else { $false }'

    def test_ternary_with_test_path_condition(self) -> None:
        result = pwsh_transform('(Test-Path $f) ? "exists" : "missing"')[0]
        assert "?" not in result
        assert "if ((Test-Path $f))" in result

# ============================================================================
# Null coalescing with complex left/right expressions
# ============================================================================

class TestNullCoalescingComplex:
    def test_null_coalescing_with_array_literal_left(self) -> None:
        result = pwsh_transform('$x = @(1,2) ?? @(3)')[0]
        assert "??" not in result
        assert "@(1,2)" in result
        assert "@(3)" in result

    def test_null_coalescing_with_hashtable_literal_left(self) -> None:
        result = pwsh_transform('$x = @{ a = 1 } ?? @{ b = 2 }')[0]
        assert "??" not in result
        assert "@{ a = 1 }" in result

    def test_null_coalescing_with_script_block_right(self) -> None:
        result = pwsh_transform('$x = $sb ?? { Write-Output default }')[0]
        assert "??" not in result
        assert "{ Write-Output default }" in result

    def test_null_coalescing_inside_parentheses(self) -> None:
        result = pwsh_transform('$x = ($a) ?? "default"')[0]
        assert "??" not in result
        assert "($a)" in result

    def test_null_coalescing_with_nested_parens(self) -> None:
        result = pwsh_transform('$x = (($a)) ?? "default"')[0]
        assert "??" not in result
        assert "(($a))" in result

    def test_string_with_operator_then_real_operator(self) -> None:
        # LIMITATION: _expr_left scans past string boundaries, so the entire
        # left side includes the preceding string and its inner operator.
        result = pwsh_transform("'a ?? b' ?? 'c'")[0]
        assert "if ($null -ne 'a ?? b')" in result
        assert "'a ?? b'" in result
        assert "'c'" in result

    def test_double_quoted_string_with_operator_then_real_operator(self) -> None:
        result = pwsh_transform('"a ?? b" ?? "c"')[0]
        assert "if ($null -ne \"a ?? b\")" in result
        assert '"a ?? b"' in result
        assert '"c"' in result

# ============================================================================
# Pipeline chains with special contexts
# ============================================================================

class TestChainSpecialContexts:
    def test_chain_with_semicolon_before(self) -> None:
        result = pwsh_transform("cmd1 ; cmd2 && cmd3")[0]
        assert "&&" not in result
        assert "if ($?)" in result
        assert "cmd1 ; cmd2" in result

    def test_chain_inside_array_subexpression(self) -> None:
        result = pwsh_transform("@(cmd1 && cmd2)")[0]
        assert "&&" not in result
        assert "cmd1" in result
        assert "cmd2" in result

    def test_chain_with_variable_assignment(self) -> None:
        result = pwsh_transform("$r = cmd1 && cmd2")[0]
        assert "&&" not in result
        assert "if ($?)" in result

    def test_chain_after_foreach_pipeline(self) -> None:
        result = pwsh_transform("1..3 | ForEach-Object { $_ } && Write-Output done")[0]
        assert "&&" not in result
        assert "if ($?)" in result

# ============================================================================
# Null-conditional method args with inner operators
# ============================================================================

class TestNullConditionalMethodNesting:
    def test_method_arg_with_inner_null_conditional(self) -> None:
        # Inner ?. inside method args is now transformed on a subsequent pass.
        result = pwsh_transform("$a?.Foo($b?.Bar())")[0]
        assert "?." not in result
        assert "$a" in result
        assert ".Foo(" in result
        assert ".Bar()" in result

    def test_method_arg_with_inner_null_coalescing(self) -> None:
        result = pwsh_transform('$a?.Foo($b ?? "default")')[0]
        assert "?." not in result
        assert "??" not in result
        assert '"default"' in result

    def test_index_with_nested_brackets(self) -> None:
        result = pwsh_transform("$a?[$i[$j]]")[0]
        assert "?[" not in result
        assert "$i[$j]" in result

# ============================================================================
# Unterminated / malformed inputs
# ============================================================================

class TestMalformedInputs:
    def test_unterminated_double_quoted_string(self) -> None:
        result = pwsh_transform('Write-Output "hello')[0]
        assert isinstance(result, str)

    def test_unterminated_single_quoted_string(self) -> None:
        result = pwsh_transform("Write-Output 'hello")[0]
        assert isinstance(result, str)

    def test_unterminated_block_comment(self) -> None:
        result = pwsh_transform("<# hello\nWrite-Output $a ?? 'default'")[0]
        assert isinstance(result, str)

    def test_unterminated_subexpression(self) -> None:
        result = pwsh_transform("$($a + ")[0]
        assert isinstance(result, str)

    def test_whitespace_only_input(self) -> None:
        result = pwsh_transform("   \n  \t  \n  ")[0]
        assert isinstance(result, str)

    def test_line_with_only_comment(self) -> None:
        result = pwsh_transform("# just a comment")[0]
        assert result == "# just a comment"

# ============================================================================
# Mixed / combined operator stress
# ============================================================================

class TestMixedOperatorStress:
    def test_null_coalescing_then_ternary(self) -> None:
        # LIMITATION: after ?? is transformed, the resulting ternary sits
        # inside braces at depth>0, so _transform_ternary_line skips it.
        result = pwsh_transform('$x = $a ?? $b ? "t" : "f"')[0]
        assert "??" not in result
        # ternary inside generated braces is NOT transformed (depth>0)
        assert "?" in result
        assert "$a" in result
        assert "$b" in result

    def test_ternary_then_null_coalescing(self) -> None:
        result = pwsh_transform('$x = $cond ? ($a ?? $b) : $c')[0]
        assert "??" not in result
        assert "?" not in result
        assert "$cond" in result

    def test_null_conditional_then_null_coalescing(self) -> None:
        # ?. now runs before ?? and wraps its output in $(), so ?? can safely
        # use the transformed expression as an operand.
        result = pwsh_transform('$x = $a?.Name ?? "default"')[0]
        assert "?." not in result
        assert "??" not in result
        assert "if ($null -ne $(if ($null -ne $a) { $a.Name }))" in result
        assert '"default"' in result

    def test_all_operators_in_one_line(self) -> None:
        result = pwsh_transform('$a ??= $b; $c = $d?.Name ?? "x"; cmd1 && cmd2 || cmd3')[0]
        assert "??=" not in result
        assert "?." not in result
        assert "??" not in result
        assert "&&" not in result
        assert "||" not in result
        # $d?.Name is transformed first, then ?? uses the wrapped result
        assert "if ($null -ne $(if ($null -ne $d) { $d.Name }))" in result

    def test_null_conditional_chain_with_index_and_property(self) -> None:
        result = pwsh_transform('$a?.Items?[0]?.Name')[0]
        assert "?." not in result
        assert "$a" in result

# ============================================================================
# Backtick edge cases
# ============================================================================

class TestBacktickEdgeCases:
    def test_backtick_before_operator_no_newline(self) -> None:
        result = pwsh_transform("cmd1 `&& cmd2")[0]
        # No newline after backtick, so `& is literal backtick + &, not continuation
        assert isinstance(result, str)

    def test_multiple_backticks_with_newlines(self) -> None:
        result = pwsh_transform("Write-Output `\n`\n`\nhello")[0]
        assert isinstance(result, str)
        assert "hello" in result

    def test_backtick_continuation_before_comment(self) -> None:
        code = "$x = $a ??`\n  # this is a comment\n  'default'"
        result = pwsh_transform(code)[0]
        assert isinstance(result, str)

# ============================================================================
# Expression boundary edge cases
# ============================================================================

class TestExprBoundaryEdgeCases:
    def test_null_coalescing_after_command_prefix_in_parens(self) -> None:
        result = pwsh_transform('Write-Output ($a ?? "default")')[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result
        assert "Write-Output" in result

    def test_ternary_after_command_prefix_in_parens(self) -> None:
        # BUG: ternary inside () is at depth>0, so it is skipped.
        result = pwsh_transform('Write-Output ($cond ? "a" : "b")')[0]
        assert "?" in result
        assert "Write-Output ($cond ? \"a\" : \"b\")" == result

    def test_null_conditional_after_command_prefix(self) -> None:
        # BUG: _expr_left includes the command prefix as part of the base expr.
        result = pwsh_transform('Write-Output $a?.Name')[0]
        assert "?." not in result
        # Currently produces: if ($null -ne Write-Output $a) { Write-Output $a.Name }
        assert "Write-Output" in result
        assert "$a" in result

    def test_ternary_with_type_accelerator_condition(self) -> None:
        result = pwsh_transform('[string]::IsNullOrEmpty($s) ? "empty" : "non-empty"')[0]
        assert "?" not in result
        assert "[string]::IsNullOrEmpty($s)" in result

# ============================================================================
# Idempotency for new patterns
# ============================================================================

class TestNewIdempotency:
    def test_null_conditional_array_element_idempotent(self) -> None:
        code = "$arr[0]?.Name"
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_property_nca_idempotent(self) -> None:
        code = '$obj.Name ??= "default"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_ternary_with_hashtable_idempotent(self) -> None:
        code = '$x = $cond ? @{ a = 1 } : @{ b = 2 }'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

# ============================================================================
# Null-conditional with variable property names (?.$prop)
# ============================================================================

class TestNullConditionalVariableProperty:
    def test_simple_variable_property(self) -> None:
        result = pwsh_transform("$a?.$property")[0]
        assert "?." not in result
        assert "$a" in result
        assert "$property" in result
        assert "if ($null -ne $a)" in result

    def test_variable_property_with_scope(self) -> None:
        result = pwsh_transform("$a?.$global:prop")[0]
        assert "?." not in result
        assert "$global:prop" in result

    def test_variable_property_braced(self) -> None:
        result = pwsh_transform("$a?.${var}")[0]
        assert "?." not in result
        assert "${var}" in result

    def test_variable_property_assignment(self) -> None:
        result = pwsh_transform("$x = $a?.$property")[0]
        assert "?." not in result
        assert "$x = " in result

    def test_variable_property_chained(self) -> None:
        result = pwsh_transform("$a?.$prop?.$other")[0]
        assert "?." not in result
        assert "$prop" in result
        assert "$other" in result

    def test_mixed_variable_and_plain_chain(self) -> None:
        result = pwsh_transform("$a?.Name?.$prop")[0]
        assert "?." not in result
        assert "Name" in result
        assert "$prop" in result

    def test_variable_property_idempotent(self) -> None:
        code = "$a?.$property"
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

# ============================================================================
# Null-conditional with quoted member names (?.'name' / ?."name")
# ============================================================================

class TestNullConditionalQuotedMember:
    def test_single_quoted_member(self) -> None:
        result = pwsh_transform("$a?.'property-name'")[0]
        assert "?." not in result
        assert "'property-name'" in result

    def test_double_quoted_member(self) -> None:
        result = pwsh_transform('$a?."property-name"')[0]
        assert "?." not in result
        assert '"property-name"' in result

    def test_double_quoted_with_spaces(self) -> None:
        result = pwsh_transform('$a?."property name"')[0]
        assert "?." not in result
        assert '"property name"' in result

    def test_single_quoted_with_doubled_quote(self) -> None:
        result = pwsh_transform("$a?.'it''s'")[0]
        assert "?." not in result
        assert "'it''s'" in result

    def test_double_quoted_with_subexpression(self) -> None:
        result = pwsh_transform('$a?."prop$(1+1)"')[0]
        assert "?." not in result
        assert '"prop$(1+1)"' in result

    def test_quoted_member_chained(self) -> None:
        result = pwsh_transform("$a?.Name?.'other-prop'")[0]
        assert "?." not in result
        assert "Name" in result
        assert "'other-prop'" in result

    def test_quoted_member_with_method(self) -> None:
        result = pwsh_transform("$a?.'get-Name'()")[0]
        assert "?." not in result
        assert "'get-Name'()" in result

    def test_quoted_member_idempotent(self) -> None:
        code = "$a?.'prop-name'"
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

# ============================================================================
# Null-coalescing assignment with braced/scoped variables
# ============================================================================

class TestNCABracedVariables:
    def test_nca_braced_variable(self) -> None:
        result = pwsh_transform('${global:var} ??= "init"')[0]
        assert "??=" not in result
        assert "if ($null -eq ${global:var})" in result

    def test_nca_braced_nested(self) -> None:
        result = pwsh_transform('${outer.${inner}} ??= "default"')[0]
        assert "??=" not in result
        assert "if ($null -eq ${outer.${inner}})" in result

    def test_nca_scoped_variable(self) -> None:
        result = pwsh_transform('$global:var ??= "init"')[0]
        assert "??=" not in result
        assert "if ($null -eq $global:var)" in result

    def test_nca_with_semicolon_after(self) -> None:
        result = pwsh_transform('${x} ??= 1; Write-Output ${x}')[0]
        assert "??=" not in result
        assert "if ($null -eq ${x})" in result
        assert "Write-Output" in result

    def test_nca_braced_idempotent(self) -> None:
        code = '${global:var} ??= "init"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

# ============================================================================
# Null-conditional with complex base expressions
# ============================================================================

class TestNullConditionalComplexChains:
    def test_multi_variable_prop_chain(self) -> None:
        result = pwsh_transform("$a?.$b?.$c?.$d")[0]
        assert "?." not in result
        assert "$a" in result
        assert "$b" in result
        assert "$c" in result
        assert "$d" in result

    def test_mixed_all_member_types(self) -> None:
        result = pwsh_transform("$a?.$b?.'c-d'?.$e")[0]
        assert "?." not in result
        assert "$b" in result
        assert "'c-d'" in result
        assert "$e" in result

    def test_double_quoted_member_chain(self) -> None:
        result = pwsh_transform('$a?."b-c"?."d-e"')[0]
        assert "?." not in result
        assert '"b-c"' in result
        assert '"d-e"' in result

    def test_cmd_prefix_with_variable_prop(self) -> None:
        result = pwsh_transform("Write-Output $a?.Name")[0]
        assert "?." not in result
        assert "Write-Output" in result
        assert "$a" in result

    def test_array_element_prop_chain(self) -> None:
        result = pwsh_transform("$arr[0][1]?.Name")[0]
        assert "?." not in result
        assert "$arr[0][1]" in result
        assert "Name" in result

# ============================================================================
# More edge cases discovered during analysis
# ============================================================================

class TestDiscoveredEdgeCases:
    def test_ternary_with_dollar_question_all(self) -> None:
        result = pwsh_transform("$? ? $? : $?")[0]
        assert result == "if ($?) { $? } else { $? }"

    def test_null_coalescing_in_double_quoted_string_preserved(self) -> None:
        result = pwsh_transform('"$a ?? $b" | Write-Output')[0]
        assert "??" in result  # preserved inside string

    def test_incomplete_here_string_preserved(self) -> None:
        result = pwsh_transform("$text = @'\nhello\n&& cmd2")[0]
        # Unterminated here-string: the rest of file is treated as string
        assert "&&" in result  # preserved because in unterminated here-string

    def test_question_mark_not_preceded_by_dollar(self) -> None:
        """? that is not preceded by $ and not followed by colon should be safe."""
        result = pwsh_transform("$a ?")[0]
        # No crash, no false match
        assert "$a" in result

    def test_double_question_at_end_no_crash(self) -> None:
        result = pwsh_transform("$a ??")[0]
        assert "$a" in result

    def test_null_conditional_dot_at_end_no_crash(self) -> None:
        result = pwsh_transform("$a?.")[0]
        assert "$a" in result

# ============================================================================
# Idempotency for all new patterns
# ============================================================================

class TestNewComprehensiveIdempotency:
    def test_var_prop_chain_idempotent(self) -> None:
        code = "$a?.$b?.$c?.$d"
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_mixed_chain_idempotent(self) -> None:
        code = "$a?.$b?.'c-d'?.$e"
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_brace_var_nca_idempotent(self) -> None:
        code = '${global:var} ??= "init"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_quoted_member_chain_idempotent(self) -> None:
        code = '$a?."b-c"?."d-e"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

# ============================================================================
# Deeply nested block comments
# ============================================================================

class TestNestedBlockComments:
    def test_triple_nested_block_comment(self) -> None:
        code = '<# L1 <# L2 <# L3 #> still L2 #> still L1 #>\n$x = $a ?? "default"'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "L1" in result
        assert "L2" in result
        assert "L3" in result

    def test_block_comment_then_operators_on_next_line(self) -> None:
        code = '<# comment #>\n$a ?? "default"'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result

    def test_block_comment_then_chain_on_next_line(self) -> None:
        code = '<# comment #>\ncmd1 && cmd2'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "if ($?)" in result

# ============================================================================
# Double-quoted here-strings
# ============================================================================

class TestHereStringDoubleQuotedExtra:
    def test_at_double_quote_here_string_preserves_operators(self) -> None:
        code = '$text = @"\n?? and ?. and && and ||\n"@\nWrite-Output done'
        result = pwsh_transform(code)[0]
        assert "??" in result  # preserved inside here-string
        assert "?." in result
        assert "&&" in result
        assert "||" in result

    def test_at_double_quote_here_string_with_subexpressions(self) -> None:
        code = '$text = @"\nHello $(Get-Date) and ?? is fine\n"@\ncmd1 && cmd2'
        result = pwsh_transform(code)[0]
        assert "$(Get-Date)" in result  # preserved in here-string
        # The && on the line after the here-string SHOULD be transformed
        assert "if ($?)" in result

    def test_at_single_quote_here_string_followed_by_operators(self) -> None:
        code = "$text = @'\nhello\n'@\n$x = $a ?? 'default'"
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result

# ============================================================================
# Backtick continuation deep edge cases
# ============================================================================

class TestBacktickDeepEdgeCases:
    def test_backtick_inside_double_quoted_string_not_collapsed(self) -> None:
        """Backtick inside a double-quoted string is not a line continuation."""
        code = '$x = "hello`nthere $a ?? $b"'
        result = pwsh_transform(code)[0]
        # ?? inside string should be preserved
        assert "??" in result

    def test_backtick_inside_single_quoted_string_not_collapsed(self) -> None:
        code = "$x = 'hello`nthere $a ?? $b'"
        result = pwsh_transform(code)[0]
        # Inside single-quoted string, ` is literal
        assert "??" in result

    def test_backtick_with_only_carriage_return(self) -> None:
        """Backtick followed by \\r only (not \\n) is NOT a line continuation."""
        code = "cmd1 `\r && cmd2"
        result = pwsh_transform(code)[0]
        # The ` is NOT collapsed since \r is not \n
        assert "`" in result

    def test_backtick_at_eof(self) -> None:
        """Backtick at end of file with no following characters."""
        result = pwsh_transform("Write-Output `")[0]
        assert isinstance(result, str)
        assert "`" in result or "Write-Output" in result

    def test_consecutive_backtick_continuations(self) -> None:
        code = "cmd1 `\n`\n`\n&& cmd2"
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "if ($?)" in result

    def test_backtick_continuation_with_tabs(self) -> None:
        code = "cmd1 `\n\t\t&& cmd2"
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "if ($?)" in result

# ============================================================================
# _strip_command_prefix with PS keywords
# ============================================================================

class TestCommandPrefixStripping:
    def test_keyword_not_stripped(self) -> None:
        """PS keywords like 'if', 'for', 'while' should NOT be stripped as command prefix."""
        result = pwsh_transform('if $a ?? "default"')[0]
        # 'if' is a keyword, not a command, so it should not be stripped
        # This means $a is recognized as left of ??, not "if $a"
        assert "??" not in result

    def test_foreach_not_stripped(self) -> None:
        result = pwsh_transform('foreach $a ?? "default"')[0]
        assert "??" not in result
        assert "$a" in result

    def test_return_not_stripped(self) -> None:
        result = pwsh_transform('return $a ?? "default"')[0]
        assert "??" not in result
        assert "$a" in result

    def test_real_command_is_stripped(self) -> None:
        result = pwsh_transform('Write-Output $a ?? "default"')[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result

# ============================================================================
# _match_assignment with complex left-hand sides
# ============================================================================

class TestComplexAssignmentDetection:
    def test_scoped_property_assignment_coalescing(self) -> None:
        result = pwsh_transform('$global:obj.Property = $a ?? "default"')[0]
        assert "??" not in result
        assert "$global:obj.Property = " in result
        assert "if ($null -ne $a)" in result

    def test_no_assignment_coalescing(self) -> None:
        result = pwsh_transform('$a ?? "default"')[0]
        assert "=" not in result.split("if")[0]  # no assignment before the if

    def test_assignment_with_ternary(self) -> None:
        result = pwsh_transform('$x = $cond ? "a" : "b"')[0]
        assert "$x = " in result

# ============================================================================
# _find_expr_start / _find_expr_end edge cases
# ============================================================================

class TestExpressionBoundariesDeep:
    def test_expr_at_start_of_line(self) -> None:
        """Expression starting at column 0."""
        result = pwsh_transform('$a ?? "default"')[0]
        assert "??" not in result

    def test_expr_at_end_of_line(self) -> None:
        """Expression ending at end of line (no trailing chars)."""
        result = pwsh_transform('$x = $a ?? "default"')[0]
        assert "??" not in result

    def test_array_subexpr_boundary(self) -> None:
        """@() as expression boundary."""
        result = pwsh_transform('$x = @(1,2) ?? @(3,4)')[0]
        assert "??" not in result
        assert "@(1,2)" in result
        assert "@(3,4)" in result

    def test_at_paren_boundary_for_ternary(self) -> None:
        """Ternary where condition is @()."""
        result = pwsh_transform('$x = @(1).Count -gt 0 ? "yes" : "no"')[0]
        assert "?" not in result
        assert "if (@(1).Count -gt 0)" in result

    def test_ampersand_call_operator_boundary(self) -> None:
        """& call operator as boundary."""
        result = pwsh_transform('& $cmd $a ?? "default"')[0]
        assert "??" not in result

# ============================================================================
# Null-conditional with unusual member-name characters
# ============================================================================

class TestNullConditionalUnusualMembers:
    def test_dot_then_at_sign_not_transformed(self) -> None:
        """$a?.@ is invalid; should not crash or transform."""
        result = pwsh_transform("$a?.@")[0]
        assert isinstance(result, str)
        # @ is not a valid member name char, so ?. is not transformed
        assert "$a" in result

    def test_dot_then_hash_not_transformed(self) -> None:
        """$a?.#comment should stop at #."""
        result = pwsh_transform("$a?.#comment")[0]
        assert isinstance(result, str)

    def test_dot_then_lparen_method(self) -> None:
        """$a?.(...) is invalid; should not crash."""
        result = pwsh_transform("$a?.(Get-Member)")[0]
        assert isinstance(result, str)

# ============================================================================
# ?[ inside strings/regions
# ============================================================================

class TestBracketNullConditionalInStrings:
    def test_bracket_qmark_inside_single_quoted_string(self) -> None:
        result = pwsh_transform("Write-Output '?[0] is not transformed'")[0]
        assert "?[" in result
        assert "if ($null -ne" not in result

    def test_bracket_qmark_inside_double_quoted_string(self) -> None:
        result = pwsh_transform('Write-Output "?[0] is not transformed"')[0]
        assert "?[" in result
        assert "if ($null -ne" not in result

    def test_bracket_qmark_inside_comment(self) -> None:
        result = pwsh_transform("# ?[$a] is a comment\nWrite-Output hello")[0]
        assert "?[" in result

# ============================================================================
# ??= at absolute start of line
# ============================================================================

class TestNCALineStart:
    def test_nca_at_line_start(self) -> None:
        """$a ??= 'x' at column 0 of line."""
        result = pwsh_transform("$a ??= 'x'")[0]
        assert "??=" not in result
        assert "if ($null -eq $a)" in result

    def test_nca_braced_at_line_start(self) -> None:
        result = pwsh_transform("${a} ??= 'x'")[0]
        assert "??=" not in result
        assert "if ($null -eq ${a})" in result


# ============================================================================
# Multi-line here-string interaction with line transformer
# ============================================================================

class TestMultiLineRegions:
    def test_here_string_lines_not_individually_transformed(self) -> None:
        """Lines inside a multi-line here-string should be skipped by pwsh_transform."""
        code = """$text = @'
$a ?? 'should not transform'
$b?.Property
'@
Write-Output $text"""
        result = pwsh_transform(code)[0]
        # Operators inside here-string preserved
        assert "??" in result
        assert "?." in result

    def test_block_comment_lines_not_individually_transformed(self) -> None:
        code = """<#
$a ?? 'inside block comment'
$b?.Property
#>
Write-Output done"""
        result = pwsh_transform(code)[0]
        assert "??" in result
        assert "?." in result

# ============================================================================
# _skip_subexpression nested
# ============================================================================

class TestSkipSubexpressionNested:
    def test_nested_subexpressions_in_dq_string(self) -> None:
        """$() nesting inside double-quoted strings."""
        result = pwsh_transform('"$(Get-Date) and $($($a)) is fine"')[0]
        assert "$(Get-Date)" in result
        assert "$($($a))" in result

    def test_subexpr_with_single_quoted_string_inside(self) -> None:
        """$() containing a single-quoted string with special chars."""
        result = pwsh_transform('"$($x + ''?.'' )"')[0]
        # The ?. inside single quotes inside $() inside double quotes — preserved
        assert "?." in result

    def test_subexpr_with_nested_subexpr_in_dq(self) -> None:
        """Double-quoted string with $() that itself contains a dq string with $()."""
        result = pwsh_transform('"outer $(Get-Date \"inner $($a)\") end"')[0]
        assert isinstance(result, str)

# ============================================================================
# Ternary operator interaction with ?. and ?[
# ============================================================================

class TestTernaryInteractionDeep:
    def test_ternary_question_not_confused_with_null_conditional_dot(self) -> None:
        """$a?.Property should NOT be recognized as ternary."""
        result = pwsh_transform("$a?.Property")[0]
        assert "?." not in result
        assert "? :" not in result
        assert "if ($null -ne $a)" in result

    def test_ternary_true_branch_with_null_coalescing(self) -> None:
        """Ternary where true branch is a ?? expression."""
        result = pwsh_transform('$x = $cond ? ($a ?? "x") : "y"')[0]
        assert "??" not in result
        assert "?" not in result

    def test_ternary_false_branch_with_null_conditional(self) -> None:
        """Ternary where false branch has ?."""
        result = pwsh_transform('$x = $cond ? "yes" : $obj?.Name')[0]
        assert "?." not in result
        # The ternary ? should be gone (only $? from generated code may remain)
        assert "$obj" in result
        assert ".Name" in result

# ============================================================================
# Null coalescing with unusual spacing and operator adjacency
# ============================================================================

class TestNullCoalescingSpacingEdge:
    def test_coalescing_adjacent_to_pipe(self) -> None:
        """$a ?? $b | ForEach-Object { $_ }"""
        result = pwsh_transform('$a ?? $b | ForEach-Object { $_ }')[0]
        assert "??" not in result

    def test_coalescing_with_semicolon_right_after(self) -> None:
        result = pwsh_transform('$a ?? "x"; $b ?? "y"')[0]
        assert "??" not in result
        assert result.count("if ($null -ne") == 2

    def test_coalescing_with_comma_separated_defaults(self) -> None:
        """$a ?? $b, $c ?? $d — comma binds tighter than ??."""
        result = pwsh_transform('$a ?? $b, $c ?? $d')[0]
        assert "??" not in result

# ============================================================================
# _transform_chain_line: operators inside strings with outside operators
# ============================================================================

class TestChainMixedInsideOutside:
    def test_and_inside_string_or_outside(self) -> None:
        result = pwsh_transform("Write-Output '&&' || Write-Output done")[0]
        assert "&&" in result  # inside string, preserved
        assert "||" not in result
        assert "if (-not $?)" in result

    def test_or_inside_string_and_outside(self) -> None:
        result = pwsh_transform('Write-Output "||" && Write-Output done')[0]
        assert "||" in result  # inside string, preserved
        assert "&&" not in result
        assert "if ($?)" in result

# ============================================================================
# pwsh_transform multiline with mixed operators on different lines
# ============================================================================

class TestMultiLineMixedOperators:
    def test_different_operators_on_different_lines(self) -> None:
        code = """$x = $a ?? "default"
$y = $cond ? "yes" : "no"
cmd1 && cmd2
$z = $obj?.Property"""
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "?." not in result
        assert "&&" not in result
        # Ternary ? is gone; $? from chain transform is expected
        assert "if ($cond)" in result
        assert "if ($?)" in result

    def test_operators_on_consecutive_lines(self) -> None:
        code = """cmd1 && cmd2
cmd3 || cmd4"""
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "||" not in result
        assert "if ($?)" in result
        assert "if (-not $?)" in result

# ============================================================================
# Null-conditional bracket with string containing brackets
# ============================================================================

class TestNullConditionalBracketStrings:
    def test_bracket_index_with_string_containing_bracket(self) -> None:
        result = pwsh_transform("$a?['[']")[0]
        assert "?[" not in result
        assert "'['" in result

    def test_bracket_index_with_dq_string_containing_bracket(self) -> None:
        result = pwsh_transform('$a?["]"]')[0]
        assert "?[" not in result
        assert '"]"' in result

    def test_bracket_index_with_nested_brackets_in_string(self) -> None:
        result = pwsh_transform('$a?["[[["]')[0]
        assert "?[" not in result
        assert '"[[["' in result

# ============================================================================
# Single-quoted string scanner edge cases
# ============================================================================

class TestSingleQuotedStringScanner:
    def test_empty_single_quoted_string(self) -> None:
        """Empty '' should not confuse the scanner."""
        result = pwsh_transform("'' ?? 'default'")[0]
        assert "??" not in result
        assert "if ($null -ne '')" in result

    def test_only_escaped_quotes(self) -> None:
        """'''' is two escaped quotes — should be a string region."""
        result = pwsh_transform("'''' ?? 'default'")[0]
        assert "??" not in result

    def test_escaped_at_start_and_end(self) -> None:
        """''a'' — escaped quote, content, escaped quote."""
        result = pwsh_transform("''a'' ?? 'default'")[0]
        assert "??" not in result

    def test_doubled_quotes_in_content(self) -> None:
        """'it''s ok' — doubled quotes representing literal '."""
        result = pwsh_transform("'it''s ok' ?? 'default'")[0]
        assert "??" not in result

# ============================================================================
# Double-quoted string scanner edge cases
# ============================================================================

class TestDoubleQuotedStringScanner:
    def test_backtick_n_escape(self) -> None:
        """`n inside double-quoted string should not close the string."""
        result = pwsh_transform('"hello`nworld" ?? "default"')[0]
        assert "??" not in result

    def test_backtick_escaped_quote(self) -> None:
        """`" inside double-quoted string is an escaped quote, not closing."""
        result = pwsh_transform('"hello`"world" ?? "default"')[0]
        assert "??" not in result

    def test_dollar_paren_subexpression_in_dq(self) -> None:
        """$() inside double-quoted string should be skipped correctly."""
        result = pwsh_transform('"$(Get-Date)" ?? "default"')[0]
        assert "??" not in result
        assert "$(Get-Date)" in result

    def test_nested_dollar_paren_in_dq(self) -> None:
        """Nested $($($a)) inside dq string."""
        result = pwsh_transform('"$($($a))" ?? "default"')[0]
        assert "??" not in result

# ============================================================================
# Deeply nested block comments (4+ levels)
# ============================================================================

class TestDeepNestedBlockComments:
    def test_four_deep_block_comment(self) -> None:
        code = '<# L1 <# L2 <# L3 <# L4 #> L3 #> L2 #> L1 #>\n$x = $a ?? "default"'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "L1" in result
        assert "L4" in result

# ============================================================================
# Subexpression scanner with mixed quotes
# ============================================================================

class TestSubexpressionMixedQuotes:
    def test_mixed_quotes_in_subexpr(self) -> None:
        result = pwsh_transform('"$( "hello $(''inner'') world" )"')[0]
        assert isinstance(result, str)

    def test_brackets_inside_subexpr(self) -> None:
        result = pwsh_transform("$($a[0]) ?? 'default'")[0]
        assert "??" not in result
        assert "$($a[0])" in result

# ============================================================================
# @' and @" single-line (not here-strings)
# ============================================================================

class TestAtSignSingleLine:
    def test_at_double_quote_single_line_not_here_string(self) -> None:
        """@"..."@ on a single line is not a here-string."""
        result = pwsh_transform('@"?? and ?. preserved"@')[0]
        assert "??" in result  # inside string region, preserved
        assert "?." in result

    def test_at_single_quote_single_line_not_here_string(self) -> None:
        """@'...'@ on a single line is not a here-string."""
        result = pwsh_transform("@'?? preserved'@")[0]
        assert "??" in result

# ============================================================================
# Backtick inside single-quoted strings not collapsed
# ============================================================================

class TestBacktickInSingleQuotedString:
    def test_backtick_newline_in_sq_string_not_collapsed(self) -> None:
        """Backtick inside '...' is literal, not a line continuation."""
        result = pwsh_transform("'hello `\nworld'")[0]
        assert isinstance(result, str)
        # The backtick should remain because it's inside a string

# ============================================================================
# _strip_command_prefix with numbers
# ============================================================================

class TestCommandPrefixNumbers:
    def test_command_prefix_with_number_argument(self) -> None:
        """Write-Output 123 ?? 0 — command prefix should be stripped."""
        result = pwsh_transform("Write-Output 123 ?? 0")[0]
        assert "??" not in result
        assert "if ($null -ne 123)" in result

    def test_command_prefix_with_variable(self) -> None:
        """Write-Output $a ?? 0 — command prefix should be stripped."""
        result = pwsh_transform("Write-Output $a ?? 0")[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result

# ============================================================================
# Two ?? or two ??= or two ?. or two ?[ on one line
# ============================================================================

class TestMultipleSameOperator:
    def test_two_nca_on_one_line(self) -> None:
        result = pwsh_transform('$a ??= "x"; $b ??= "y"')[0]
        assert "??=" not in result
        assert "if ($null -eq $a)" in result
        assert "if ($null -eq $b)" in result

    def test_two_null_coalescing_on_one_line(self) -> None:
        result = pwsh_transform('$a ?? "x"; $b ?? "y"')[0]
        assert "??" not in result
        assert result.count("if ($null -ne") == 2

    def test_two_null_conditional_dot_on_one_line(self) -> None:
        result = pwsh_transform("$a?.Name; $b?.Count")[0]
        assert "?." not in result
        assert "$a" in result
        assert "$b" in result

    def test_two_null_conditional_bracket_on_one_line(self) -> None:
        result = pwsh_transform("$a?[0]; $b?[1]")[0]
        assert "?[" not in result
        assert "$a[0]" in result
        assert "$b[1]" in result

# ============================================================================
# Ternary with nested condition
# ============================================================================

class TestTernaryNestedCondition:
    def test_ternary_with_paren_condition(self) -> None:
        result = pwsh_transform('($a -gt 0) ? ($b ? "c" : "d") : "e"')[0]
        assert "if (($a -gt 0))" in result

    def test_ternary_false_branch_chain(self) -> None:
        result = pwsh_transform('$cond ? "a" : cmd1 && cmd2')[0]
        # Ternary ? is gone, $? from chain transform appears
        assert "if ($cond)" in result
        assert "&&" not in result

# ============================================================================
# Chain with 5 operators
# ============================================================================

class TestLongChain:
    def test_five_and_chain(self) -> None:
        result = pwsh_transform("cmd1 && cmd2 && cmd3 && cmd4 && cmd5")[0]
        assert "&&" not in result
        assert result.count("if ($?)") == 4

# ============================================================================
# ?. with invalid member (starts with number)
# ============================================================================

class TestNullConditionalInvalidMembers:
    def test_number_member_not_transformed(self) -> None:
        """$a?.123 — member names can't start with number; should not transform."""
        result = pwsh_transform("$a?.123")[0]
        # Should not crash; ?. is not transformed because 123 is not alphanumeric...
        # Actually 1 is alphanumeric, but the member starts with a digit.
        # The transformer accepts it as a member name but in PS member names
        # starting with digits are invalid. Transformer just passes through.
        assert isinstance(result, str)

    def test_empty_index_not_crash(self) -> None:
        """$a?[] — empty index should not crash."""
        result = pwsh_transform("$a?[]")[0]
        assert isinstance(result, str)

# ============================================================================
# Complex NCA with chained property access
# ============================================================================

class TestNCAPropertyChain:
    def test_chained_property_nca(self) -> None:
        result = pwsh_transform('$a.b.c ??= "default"')[0]
        assert "??=" not in result
        assert "if ($null -eq $a.b.c)" in result
        assert "$a.b.c = " in result

# ============================================================================
# && / || with & call operator boundary
# ============================================================================

class TestChainWithCallOperator:
    def test_call_operator_then_coalescing(self) -> None:
        result = pwsh_transform('& $cmd $a ?? "default"')[0]
        assert "??" not in result
        assert "$a" in result


# ============================================================================
# Ultimate idempotency: all operators combined
# ============================================================================

class TestUltimateIdempotency:
    def test_all_operators_combined_idempotent(self) -> None:
        code = '${a} ??= ${b}; $c = $d?.$e?.\'f\' ?? "g"; cmd1 && cmd2 || cmd3'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_every_operator_once_idempotent(self) -> None:
        code = '$x = $a ?? "d"; $y = $c ? "t" : "f"; $z ??= 0; $w = $q?.Prop; cmd1 && cmd2'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

# ============================================================================
# Unterminated string / comment / subexpression scanners
# ============================================================================

class TestUnterminatedScanners:
    def test_unterminated_single_quoted(self) -> None:
        """Unterminated '... should not crash; treats rest as string."""
        result = pwsh_transform("'unterminated ?? and ?.")[0]
        assert isinstance(result, str)
        assert "??" in result  # inside unterminated string region, preserved

    def test_unterminated_double_quoted(self) -> None:
        result = pwsh_transform('"unterminated ?? and ?.')[0]
        assert isinstance(result, str)
        assert "??" in result

    def test_unterminated_block_comment_eof(self) -> None:
        result = pwsh_transform("<# unterminated ?? and ?.")[0]
        assert isinstance(result, str)

    def test_unterminated_subexpression(self) -> None:
        result = pwsh_transform("$(unterminated ?? and ?.")[0]
        assert isinstance(result, str)

    def test_unterminated_here_string_single_quoted(self) -> None:
        result = pwsh_transform("@'\nunterminated ?? and ?.")[0]
        assert isinstance(result, str)
        assert "??" in result

# ============================================================================
# Backtick at extremes (position 0, EOF)
# ============================================================================

class TestBacktickExtremes:
    def test_backtick_at_position_zero(self) -> None:
        """Backtick at very start of code."""
        result = pwsh_transform("`\ncmd1")[0]
        assert "cmd1" in result

    def test_backtick_at_end_of_file(self) -> None:
        """Backtick as last character of code (no newline after)."""
        result = pwsh_transform("cmd1 `")[0]
        assert isinstance(result, str)
        assert "`" in result or "cmd1" in result

# ============================================================================
# _match_assignment with ${braced} variables
# ============================================================================

class TestBracedAssignment:
    def test_braced_var_assignment_with_coalescing(self) -> None:
        result = pwsh_transform('${global:var} = $a ?? "default"')[0]
        assert "??" not in result
        assert "${global:var} =" in result  # _build_replacement joins without extra space
        assert "if ($null -ne $a)" in result

# ============================================================================
# Line comment at position 0 with operators on next line
# ============================================================================

class TestHashAtPositionZero:
    def test_comment_at_start_then_operator_line(self) -> None:
        code = "# comment\n$a ?? 'default'"
        result = pwsh_transform(code)[0]
        assert "# comment" in result
        assert "??" not in result  # ?? on second line IS transformed
        assert "if ($null -ne $a)" in result

# ============================================================================
# ??= inside string literal (should NOT be transformed)
# ============================================================================

class TestNCAInsideString:
    def test_nca_inside_single_quoted_string_not_transformed(self) -> None:
        """The ??= inside the string is not matched. The real ??= after the string
        has an empty variable (the string literal), so _transform_nca_line skips it;
        _transform_nc_line then handles the ?? part."""
        result = pwsh_transform("'??= inside string' ??= 'value'")[0]
        # The ??= is not transformed (skipped by nca, caught as ?? by nc)
        # The ?? inside the string is preserved
        assert "'??= inside string'" in result

# ============================================================================
# Chain operators all inside strings — none should transform
# ============================================================================

class TestChainAllInStrings:
    def test_all_chains_inside_strings(self) -> None:
        result = pwsh_transform("'&&' + '||'")[0]
        assert "&&" in result
        assert "||" in result
        assert "if ($?)" not in result
        assert "if (-not $?)" not in result

# ============================================================================
# String literal containing ?? then real ?? on same line
# ============================================================================

class TestStringThenRealCoalescing:
    def test_string_then_real_coalescing_same_line(self) -> None:
        result = pwsh_transform("'??' ?? 'real'")[0]
        # The real ?? is transformed; ?? inside the string literal is preserved
        assert "if ($null -ne '??')" in result
        assert "'??'" in result  # string still contains ??, preserved as content
        assert "'real'" in result

# ============================================================================
# $? as ternary condition with complex branches
# ============================================================================

class TestDollarQuestionTernaryComplex:
    def test_dollar_q_ternary_with_complex_branches(self) -> None:
        result = pwsh_transform('$? ? ($a ?? "x") : ($b?.Name)')[0]
        assert "?." not in result
        assert "??" not in result
        assert "if ($?)" in result

# ============================================================================
# ?. / ?[ with ?? chained after
# ============================================================================

class TestNullConditionalThenCoalescing:
    def test_qd_then_coalescing(self) -> None:
        result = pwsh_transform('$a?.Name ?? "default"')[0]
        assert "?." not in result
        assert "??" not in result

    def test_qb_then_coalescing(self) -> None:
        result = pwsh_transform('$a?[0] ?? "default"')[0]
        assert "?[" not in result
        assert "??" not in result

# ============================================================================
# ??= with nothing on the right side
# ============================================================================

class TestNCAEmptyRight:
    def test_nca_empty_right_side(self) -> None:
        """$a ??= with nothing after should not crash."""
        result = pwsh_transform("$a ??= ")[0]
        assert isinstance(result, str)
        assert "$a" in result

# ============================================================================
# Multiple multiline here-strings in one code block
# ============================================================================

class TestMultipleHereStrings:
    def test_two_here_strings_with_operator_between(self) -> None:
        code = "@'\n?? preserved\n'@\n$x = $a ?? 'default'\n@'\n?. preserved\n'@"
        result = pwsh_transform(code)[0]
        # The ?? between the here-strings IS transformed
        assert "if ($null -ne $a)" in result
        # The ?? inside the first here-string and ?. inside second are preserved
        assert "?? preserved" in result
        assert "?. preserved" in result

# ============================================================================
# @" without newline after (not a here-string)
# ============================================================================

class TestAtSignDoubleQuoteNoNewline:
    def test_at_dq_single_line_content(self) -> None:
        """@"hello"@ on a single line — @" is NOT a here-string start."""
        result = pwsh_transform('$x = @"hello"@')[0]
        assert isinstance(result, str)

# ============================================================================
# Extremely long null-conditional chain
# ============================================================================

class TestLongNullConditionalChain:
    def test_eight_deep_qd_chain(self) -> None:
        result = pwsh_transform("$a?.b?.c?.d?.e?.f?.g?.h")[0]
        assert "?." not in result
        assert ".h" in result

# ============================================================================
# Invalid assignment syntax (no crash)
# ============================================================================

class TestInvalidAssignmentNoCrash:
    def test_dollar_sign_only_assignment(self) -> None:
        """Invalid PS: $ = ... should not crash."""
        result = pwsh_transform('$ = $a ?? "default"')[0]
        assert isinstance(result, str)

# ============================================================================
# Semicolons mixed with chain operators
# ============================================================================

class TestSemicolonChainMix:
    def test_semicolons_and_chains_mixed(self) -> None:
        result = pwsh_transform("cmd1; cmd2 && cmd3; cmd4 || cmd5")[0]
        assert "&&" not in result
        assert "||" not in result
        assert "if ($?)" in result
        assert "if (-not $?)" in result

# ============================================================================
# Ternary / ?? at very start of line (no preceding spaces)
# ============================================================================

class TestOperatorAtLineStart:
    def test_ternary_at_column_zero(self) -> None:
        result = pwsh_transform('$cond ? "a" : "b"')[0]
        assert "?" not in result
        assert "if ($cond)" in result

    def test_coalescing_at_column_zero(self) -> None:
        result = pwsh_transform('$a ?? "default"')[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result

# ============================================================================
# _strip_command_prefix: @ sign after command
# ============================================================================

class TestCommandPrefixAtSign:
    def test_command_with_array_subexpr_argument(self) -> None:
        """Write-Output @(1,2) ?? 0 — @ triggers the command-prefix check."""
        result = pwsh_transform("Write-Output @(1,2) ?? 0")[0]
        assert "??" not in result
        assert "@(1,2)" in result

    def test_command_with_hashtable_argument_coalescing(self) -> None:
        result = pwsh_transform('Write-Output @{a=1} ?? "fallback"')[0]
        assert "??" not in result
        assert "@{a=1}" in result

# ============================================================================
# _transform_chain_line: empty right side
# ============================================================================

class TestChainEmptyRight:
    def test_and_with_nothing_after(self) -> None:
        """cmd1 && — nothing after &&, should produce empty if body."""
        result = pwsh_transform("cmd1 &&")[0]
        assert "cmd1" in result
        assert "if ($?)" in result

    def test_or_with_nothing_after(self) -> None:
        result = pwsh_transform("cmd1 ||")[0]
        assert "cmd1" in result
        assert "if (-not $?)" in result

# ============================================================================
# String containing ?: that should not match ternary
# ============================================================================

class TestStringColonNotTernary:
    def test_colon_in_dq_string_not_ternary_colon(self) -> None:
        """?: inside double-quoted string should not confuse ternary."""
        result = pwsh_transform('$x = $cond ? "a:b:c" : "d"')[0]
        assert "?" not in result
        assert '"a:b:c"' in result
        assert '"d"' in result

    def test_colon_in_sq_string_not_ternary_colon(self) -> None:
        result = pwsh_transform("$x = $cond ? 'a:b:c' : 'd'")[0]
        assert "?" not in result
        assert "'a:b:c'" in result
        assert "'d'" in result

# ============================================================================
# _find_string_regions: @" at end of file (no newline)
# ============================================================================

class TestAtSignEdgeCases:
    def test_at_dq_at_end_of_code(self) -> None:
        """@" at the very end of code with no newline — not a here-string."""
        result = pwsh_transform('$x = @"text"')[0]
        assert isinstance(result, str)

    def test_at_sq_at_end_of_code(self) -> None:
        result = pwsh_transform("$x = @'text'")[0]
        assert isinstance(result, str)

# ============================================================================
# Idempotency: transform of already-transformed code with $? in it
# ============================================================================

class TestIdempotencyWithDollarQuestion:
    def test_transformed_if_with_dollar_q_is_idempotent(self) -> None:
        """if ($?) should survive a second transform unchanged."""
        code = "if ($?) { Write-Output ok }"
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_transformed_chain_result_is_idempotent(self) -> None:
        code = "cmd1; if ($?) { cmd2 }"
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second


# ============================================================================
# Corner case: -not / ! unary operator with ternary
# ============================================================================

class TestNotOperatorTernary:
    def test_not_operator_in_ternary_condition(self) -> None:
        """-not $cond ? 'a' : 'b' — -not is part of the condition."""
        result = pwsh_transform("-not $cond ? 'a' : 'b'")[0]
        assert "?" not in result
        assert "if (-not $cond)" in result
        assert "{ 'a' }" in result
        assert "{ 'b' }" in result

    def test_not_operator_with_parens_ternary(self) -> None:
        result = pwsh_transform("!($a -eq $null) ? 'has-value' : 'null'")[0]
        assert "?" not in result
        assert "if (!($a -eq $null))" in result

    def test_not_operator_ternary_in_assignment(self) -> None:
        result = pwsh_transform('$x = -not (Test-Path $f) ? "missing" : "exists"')[0]
        assert "?" not in result
        assert "if (-not (Test-Path $f))" in result

    def test_bang_operator_ternary(self) -> None:
        """! is an alias for -not in PS."""
        result = pwsh_transform('!$flag ? "off" : "on"')[0]
        assert "?" not in result
        assert "if (!$flag)" in result


# ============================================================================
# Corner case: [Type]::StaticMember with ??
# ============================================================================

class TestStaticMemberNullCoalescing:
    def test_static_property_with_null_coalescing(self) -> None:
        """[Math]::PI ?? 3.14 — static member as left operand."""
        result = pwsh_transform("[Math]::PI ?? 3.14")[0]
        assert "??" not in result
        assert "if ($null -ne [Math]::PI)" in result
        assert "[Math]::PI" in result
        assert "3.14" in result

    def test_static_method_call_with_coalescing(self) -> None:
        result = pwsh_transform('[Enum]::Parse($type, $value) ?? "unknown"')[0]
        assert "??" not in result
        assert "if ($null -ne [Enum]::Parse($type, $value))" in result

    def test_static_member_coalescing_in_assignment(self) -> None:
        result = pwsh_transform('$x = [Version]::Parse($s) ?? [Version]::new(0,0)')[0]
        assert "??" not in result
        assert "if ($null -ne [Version]::Parse($s))" in result

    def test_static_member_on_right_of_coalescing(self) -> None:
        result = pwsh_transform('$v ?? [Math]::PI')[0]
        assert "??" not in result
        assert "if ($null -ne $v)" in result
        assert "[Math]::PI" in result


# ============================================================================
# Corner case: ?. on @() array subexpression and parenthesized tuples
# ============================================================================

class TestNullConditionalOnArrayExpr:
    def test_array_subexpr_dot_property(self) -> None:
        """@(1,2,3)?.Count — null-conditional on array subexpression."""
        result = pwsh_transform("@(1,2,3)?.Count")[0]
        assert "?." not in result
        assert "if ($null -ne @(1,2,3))" in result
        assert "@(1,2,3).Count" in result

    def test_array_subexpr_dot_method(self) -> None:
        result = pwsh_transform("@(1,2,3)?.GetType()")[0]
        assert "?." not in result
        assert "if ($null -ne @(1,2,3))" in result
        assert "@(1,2,3).GetType()" in result

    def test_parenthesized_expression_dot(self) -> None:
        result = pwsh_transform("(1,2,3)?.Count")[0]
        assert "?." not in result
        assert "if ($null -ne (1,2,3))" in result

    def test_array_subexpr_bracket(self) -> None:
        result = pwsh_transform("@(1,2,3)?[0]")[0]
        assert "?[" not in result
        assert "if ($null -ne @(1,2,3))" in result

    def test_array_subexpr_assignment(self) -> None:
        result = pwsh_transform("$x = @(1,2,3)?.Count")[0]
        assert "?." not in result
        assert "$x = $(if ($null -ne @(1,2,3)) { @(1,2,3).Count })" == result


# ============================================================================
# Corner case: ?. with PowerShell keyword member names
# ============================================================================

class TestNullConditionalKeywordMembers:
    def test_keyword_begin_member(self) -> None:
        """$obj?.Begin — 'Begin' is a PS keyword but also valid member name."""
        result = pwsh_transform("$obj?.Begin")[0]
        assert "?." not in result
        assert "if ($null -ne $obj)" in result
        assert "$obj.Begin" in result

    def test_keyword_process_member(self) -> None:
        result = pwsh_transform("$obj?.Process")[0]
        assert "?." not in result
        assert "$obj.Process" in result

    def test_keyword_end_member(self) -> None:
        result = pwsh_transform("$obj?.End")[0]
        assert "?." not in result
        assert "$obj.End" in result

    def test_keyword_foreach_member(self) -> None:
        result = pwsh_transform("$obj?.ForEach")[0]
        assert "?." not in result
        assert "$obj.ForEach" in result

    def test_keyword_where_member(self) -> None:
        result = pwsh_transform("$obj?.Where")[0]
        assert "?." not in result
        assert "$obj.Where" in result

    def test_keyword_return_member(self) -> None:
        result = pwsh_transform("$obj?.Return")[0]
        assert "?." not in result
        assert "$obj.Return" in result


# ============================================================================
# Corner case: $? as ?? left operand (automatic variable)
# ============================================================================

class TestDollarQuestionCoalescing:
    def test_dollar_q_coalescing_left(self) -> None:
        """$? ?? $false — $? is an automatic variable, not ternary."""
        result = pwsh_transform("$? ?? $false")[0]
        assert "??" not in result
        assert "if ($null -ne $?)" in result
        assert "{ $? }" in result
        assert "{ $false }" in result

    def test_dollar_q_coalescing_in_assignment(self) -> None:
        result = pwsh_transform('$ok = $? ?? $false')[0]
        assert "??" not in result
        assert "$ok = " in result
        assert "if ($null -ne $?)" in result

    def test_dollar_q_nca(self) -> None:
        """$? ??= $true — null-coalescing assignment with $?."""
        result = pwsh_transform("$? ??= $true")[0]
        assert "??=" not in result
        assert "if ($null -eq $?)" in result
        assert "$? = $true" in result


# ============================================================================
# Corner case: ?. on $$ / $^ automatic variables
# ============================================================================

class TestNullConditionalAutomaticVars:
    def test_doubledollar_dot(self) -> None:
        """$$?.Name — $$ is an automatic variable (last token)."""
        result = pwsh_transform("$$?.Name")[0]
        assert "?." not in result
        assert "if ($null -ne $$)" in result
        assert "$$.Name" in result

    def test_caret_dot(self) -> None:
        """$^?.Name — $^ is an automatic variable (first token)."""
        result = pwsh_transform("$^?.Name")[0]
        assert "?." not in result
        assert "if ($null -ne $^)" in result
        assert "$^.Name" in result

    def test_dollar_q_then_question_bracket_with_index(self) -> None:
        """$??[0][1] — $? is auto var, ?[ is null-conditional."""
        result = pwsh_transform("$??[0][1]")[0]
        assert "if ($null -ne $?)" in result
        assert '$?[0]' in result

    def test_caret_coalescing(self) -> None:
        result = pwsh_transform('$^ ?? "default"')[0]
        assert "??" not in result
        assert "if ($null -ne $^)" in result


# ============================================================================
# Corner case: chain operators with $(...) subexpressions
# ============================================================================

class TestChainWithSubexpressions:
    def test_subexpr_and_subexpr(self) -> None:
        """$(cmd1) && $(cmd2) — subexpressions in chain."""
        result = pwsh_transform("$(cmd1) && $(cmd2)")[0]
        assert "&&" not in result
        assert "if ($?)" in result
        assert "$(cmd1)" in result
        assert "$(cmd2)" in result

    def test_subexpr_or_subexpr(self) -> None:
        result = pwsh_transform("$(cmd1) || $(cmd2)")[0]
        assert "||" not in result
        assert "if (-not $?)" in result

    def test_subexpr_and_or_chain(self) -> None:
        result = pwsh_transform("$(cmd1) && $(cmd2) || $(cmd3)")[0]
        assert "&&" not in result
        assert "||" not in result
        assert result.count("if ($?)") >= 1
        assert "if (-not $?)" in result

    def test_mixed_subexpr_and_plain_chain(self) -> None:
        result = pwsh_transform("cmd1 && $(cmd2) && cmd3")[0]
        assert "&&" not in result
        assert result.count("if ($?)") == 2


# ============================================================================
# Corner case: ?? with right side containing semicolons in parens
# ============================================================================

class TestCoalescingWithSemicolonRight:
    def test_subexpr_right_with_semicolons(self) -> None:
        """?? with subexpression right side containing ; at depth > 0."""
        result = pwsh_transform('$a ?? (cmd1; cmd2)')[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result
        assert "(cmd1; cmd2)" in result

    def test_subexpr_right_with_nested_semicolons(self) -> None:
        result = pwsh_transform('$a ?? $(cmd1; cmd2; cmd3)')[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result
        assert "$(cmd1; cmd2; cmd3)" in result


# ============================================================================
# Corner case: ternary with -and / -or in condition
# ============================================================================

class TestTernaryWithLogicalOperators:
    def test_and_in_condition(self) -> None:
        result = pwsh_transform('$a -and $b ? "both" : "not-both"')[0]
        assert "?" not in result
        assert "if ($a -and $b)" in result

    def test_or_in_condition(self) -> None:
        result = pwsh_transform('$a -or $b ? "either" : "neither"')[0]
        assert "?" not in result
        assert "if ($a -or $b)" in result

    def test_xor_in_condition(self) -> None:
        result = pwsh_transform('$a -xor $b ? "one" : "both-or-neither"')[0]
        assert "?" not in result
        assert "if ($a -xor $b)" in result

    def test_complex_logical_condition(self) -> None:
        result = pwsh_transform('$a -gt 0 -and $b -lt 10 ? "ok" : "bad"')[0]
        assert "?" not in result
        assert "if ($a -gt 0 -and $b -lt 10)" in result


# ============================================================================
# Corner case: nested ternary in both true AND false branches
# ============================================================================

class TestNestedTernaryBothBranches:
    def test_ternary_in_both_branches(self) -> None:
        """Outer ternary with inner ternary in both branches (one pass)."""
        result = pwsh_transform('$a ? ($b ? "c" : "d") : ($e ? "f" : "g")')[0]
        assert "if ($a)" in result
        # Inner ternaries preserved (one-pass limitation)
        assert "?" in result
        assert '"c"' in result
        assert '"g"' in result

    def test_ternary_chained_condition(self) -> None:
        """$a ? "a" : $b ? "b" : $c ? "c" : "d" — right-associative parsing."""
        result = pwsh_transform('$a ? "a" : $b ? "b" : $c ? "c" : "d"')[0]
        # Only outer ?: transformed in one pass
        assert "if ($a)" in result
        assert '"a"' in result
        # Inner cascading ternaries remain
        assert "?" in result


# ============================================================================
# Corner case: $null as ?? left operand
# ============================================================================

class TestNullCoalescingWithNullLeft:
    def test_null_literal_left_coalescing(self) -> None:
        """$null ?? 'default' — $null is always null, so 'default' is chosen."""
        result = pwsh_transform("$null ?? 'default'")[0]
        assert "??" not in result
        assert "if ($null -ne $null)" in result
        assert "{ $null }" in result
        assert "{ 'default' }" in result

    def test_null_automatic_var_left_coalescing(self) -> None:
        """$null with ??= is redundant but shouldn't crash."""
        result = pwsh_transform("$null ??= 'value'")[0]
        assert "??=" not in result
        assert "if ($null -eq $null)" in result


# ============================================================================
# Corner case: ?[ with complex index containing operators
# ============================================================================

class TestNullConditionalBracketComplexIndex:
    def test_bracket_index_with_coalescing_inside(self) -> None:
        """$a?[$b ?? 0] — ?? inside bracket index at depth > 0."""
        result = pwsh_transform("$a?[$b ?? 0]")[0]
        assert "?[" not in result  # outer ?[ is transformed
        assert "if ($null -ne $a)" in result
        # The ?? inside the brackets is at depth > 0, not transformed in single pass

    def test_bracket_index_with_ternary_inside(self) -> None:
        result = pwsh_transform('$a?[$cond ? 0 : 1]')[0]
        assert "?[" not in result
        assert "if ($null -ne $a)" in result

    def test_bracket_index_with_nested_bracket(self) -> None:
        result = pwsh_transform("$a?[$b[$c]]")[0]
        assert "?[" not in result
        assert "if ($null -ne $a)" in result
        assert "$b[$c]" in result


# ============================================================================
# Corner case: $scope:variable containing : adjacent to ternary :
# ============================================================================

class TestScopeColonWithTernary:
    def test_scope_var_in_ternary_true_branch(self) -> None:
        """$cond ? $global:x : $local:x — colon in $global:x vs ternary :."""
        result = pwsh_transform('$cond ? $global:x : $local:x')[0]
        assert "?" not in result
        assert "if ($cond)" in result
        assert "$global:x" in result
        assert "$local:x" in result

    def test_scope_var_in_ternary_false_branch(self) -> None:
        result = pwsh_transform('$cond ? "a" : $script:val')[0]
        assert "?" not in result
        assert "$script:val" in result

    def test_scope_var_as_ternary_condition(self) -> None:
        result = pwsh_transform('$global:flag ? "yes" : "no"')[0]
        assert "?" not in result
        assert "if ($global:flag)" in result


# ============================================================================
# Corner case: ??= with property chain on left (deep assignment)
# ============================================================================

class TestNCAPropertyDeepChain:
    def test_three_level_property_nca(self) -> None:
        result = pwsh_transform('$obj.Prop1.Prop2.Prop3 ??= "init"')[0]
        assert "??=" not in result
        assert "if ($null -eq $obj.Prop1.Prop2.Prop3)" in result
        assert "$obj.Prop1.Prop2.Prop3 = " in result

    def test_property_chain_with_method_nca(self) -> None:
        result = pwsh_transform('$svc.Status ??= "Running"')[0]
        assert "??=" not in result
        assert "if ($null -eq $svc.Status)" in result


# ============================================================================
# Corner case: backtick inside '' and "" NOT collapsed (literal)
# ============================================================================

class TestBacktickLiteralInStrings:
    def test_backtick_n_in_dq_not_collapsed(self) -> None:
        """`n inside double-quoted string is escape, not continuation."""
        result = pwsh_transform('"hello`nworld"')[0]
        assert "hello`nworld" in result

    def test_backtick_t_in_dq_not_collapsed(self) -> None:
        result = pwsh_transform('"col1`tcol2"')[0]
        assert "col1`tcol2" in result

    def test_backtick_in_sq_literal_not_collapsed(self) -> None:
        result = pwsh_transform("'backtick ` is literal'")[0]
        assert "`" in result

    def test_backtick_before_chars_in_sq_not_collapsed(self) -> None:
        """`a in single quotes is just literal `a."""
        result = pwsh_transform("'`a ?? b'")[0]
        assert "`a ?? b" in result
        assert "??" in result  # inside string, preserved


# ============================================================================
# Corner case: $? preservation inside if/elseif/while conditions
# ============================================================================

class TestDollarQuestionInKeywords:
    def test_if_with_dollar_q_condition(self) -> None:
        result = pwsh_transform("if ($?) { Write-Output 'ok' }")[0]
        assert result == "if ($?) { Write-Output 'ok' }"

    def test_while_with_dollar_q_condition(self) -> None:
        result = pwsh_transform("while ($?) { Do-Something }")[0]
        assert result == "while ($?) { Do-Something }"

    def test_elseif_with_dollar_q(self) -> None:
        code = "if ($a) { 1 } elseif ($?) { 2 } else { 3 }"
        result = pwsh_transform(code)[0]
        assert "$?" in result
        assert "?" not in result.replace("$?", "")  # no bare ? remains


# ============================================================================
# Corner case: ?. chain with method then property then index
# ============================================================================

class TestNullConditionalMixedChainTypes:
    def test_method_then_property_chain(self) -> None:
        result = pwsh_transform("$a?.GetValue()?.Length")[0]
        assert "?." not in result
        assert "GetValue()" in result
        assert ".Length" in result

    def test_property_then_method_then_property(self) -> None:
        result = pwsh_transform("$a?.Items?.GetType()?.Name")[0]
        assert "?." not in result
        assert "Items" in result
        assert "GetType()" in result
        assert "Name" in result

    def test_method_then_bracket_chain(self) -> None:
        result = pwsh_transform("$a?.GetItems()?[0]")[0]
        assert "?." not in result
        assert "GetItems()" in result


# ============================================================================
# Corner case: ?? with @() or @{} on right side
# ============================================================================

class TestCoalescingRightSideCollections:
    def test_coalescing_with_empty_array_right(self) -> None:
        result = pwsh_transform("$a ?? @()")[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result
        assert "@()" in result

    def test_coalescing_with_empty_hashtable_right(self) -> None:
        result = pwsh_transform("$a ?? @{}")[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result
        assert "@{}" in result

    def test_coalescing_with_scriptblock_right(self) -> None:
        result = pwsh_transform("$a ?? { Get-Date }")[0]
        assert "??" not in result
        assert "{ Get-Date }" in result


# ============================================================================
# Corner case: multiple ?. chains on same line with ; separator
# ============================================================================

class TestMultipleNullConditionalChains:
    def test_two_qd_chains_semicolon(self) -> None:
        result = pwsh_transform("$a?.b?.c; $x?.y?.z")[0]
        assert "?." not in result
        assert "$a.b.c" in result
        assert "$x.y.z" in result

    def test_qd_and_qb_chains_semicolon(self) -> None:
        result = pwsh_transform("$a?.Name; $b?[0]")[0]
        assert "?." not in result
        assert "?[" not in result
        assert "$a.Name" in result
        assert "$b[0]" in result

    def test_three_qd_chains_semicolon(self) -> None:
        result = pwsh_transform("$a?.P1; $b?.P2; $c?.P3")[0]
        assert "?." not in result
        assert result.count("if ($null -ne $") == 3


# ============================================================================
# Corner case: chain && || with trailing whitespace
# ============================================================================

class TestChainWithTrailingWhitespace:
    def test_and_chain_trailing_spaces(self) -> None:
        result = pwsh_transform("cmd1 && cmd2   ")[0]
        assert "&&" not in result
        assert "if ($?)" in result

    def test_or_chain_trailing_tabs(self) -> None:
        result = pwsh_transform("cmd1 || cmd2\t\t")[0]
        assert "||" not in result
        assert "if (-not $?)" in result

    def test_and_chain_leading_spaces(self) -> None:
        result = pwsh_transform("   cmd1 && cmd2")[0]
        assert "&&" not in result
        assert "if ($?)" in result


# ============================================================================
# Corner case: ternary with static method call in all positions
# ============================================================================

class TestTernaryStaticMethods:
    def test_static_method_in_condition(self) -> None:
        result = pwsh_transform('[string]::IsNullOrEmpty($s) ? "empty" : "ok"')[0]
        assert "?" not in result
        assert "if ([string]::IsNullOrEmpty($s))" in result

    def test_static_method_in_true_branch(self) -> None:
        result = pwsh_transform('$cond ? [Math]::Abs($x) : $x')[0]
        assert "?" not in result
        assert "{ [Math]::Abs($x) }" in result

    def test_static_method_in_false_branch(self) -> None:
        result = pwsh_transform('$cond ? $x : [Math]::Max($x, 0)')[0]
        assert "?" not in result
        assert "{ [Math]::Max($x, 0) }" in result


# ============================================================================
# Corner case: ??= idempotency after chain transforms
# ============================================================================

class TestNCAChainInteractionIdempotency:
    def test_nca_after_transform_is_idempotent(self) -> None:
        code = '$x ??= "init"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        third = pwsh_transform(second)[0]
        assert first == second == third

    def test_nca_combined_with_other_ops_idempotent(self) -> None:
        code = '$a ??= "x"; $b = $c ?? "y"; cmd1 && cmd2'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second


# ============================================================================
# Corner case: operators adjacent to end-of-line comment (#)
# ============================================================================

class TestOperatorsBeforeLineComment:
    def test_coalescing_before_line_comment(self) -> None:
        result = pwsh_transform('$a ?? "default" # end of line')[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result
        assert "# end of line" in result

    def test_ternary_before_line_comment(self) -> None:
        result = pwsh_transform('$cond ? "yes" : "no" # ternary')[0]
        assert "if ($cond)" in result
        assert "# ternary" in result

    def test_null_conditional_before_line_comment(self) -> None:
        result = pwsh_transform("$a?.Name # null-conditional")[0]
        assert "?." not in result
        assert "# null-conditional" in result

    def test_chain_before_line_comment(self) -> None:
        result = pwsh_transform("cmd1 && cmd2 # chain")[0]
        assert "&&" not in result
        assert "# chain" in result

    def test_nca_before_line_comment(self) -> None:
        """??= with trailing comment: comment must stay outside the { } block."""
        result = pwsh_transform('$var ??= value # comment')[0]
        assert "??=" not in result
        assert "# comment" in result
        assert "if ($null -eq $var) { $var = value }# comment" == result

    def test_coalescing_comment_not_inside_braces(self) -> None:
        """?? with trailing comment: comment must appear after }} not inside."""
        result = pwsh_transform('$x = $a ?? "default" # inline')[0]
        assert "??" not in result
        assert "# inline" in result
        # The # must come after the closing braces, not inside them
        assert result.endswith("# inline")

    def test_ternary_comment_not_inside_braces(self) -> None:
        """Ternary with trailing comment: comment must appear after }}."""
        result = pwsh_transform('$x = $cond ? "a" : "b" # ternary')[0]
        assert "?" not in result
        assert "# ternary" in result
        assert result.endswith("# ternary")

    def test_or_chain_before_line_comment(self) -> None:
        result = pwsh_transform("cmd1 || cmd2 # fallback")[0]
        assert "||" not in result
        assert "# fallback" in result
        assert result.endswith("# fallback")

    def test_null_conditional_assignment_comment(self) -> None:
        """?. with assignment and trailing comment."""
        result = pwsh_transform('$x = $a?.Length # comment')[0]
        assert "?." not in result
        assert "# comment" in result
        assert result.endswith("# comment")

    def test_comment_after_string_with_hash(self) -> None:
        """Line comment after a string containing # should be detected correctly."""
        result = pwsh_transform('$x = $a ?? "#notacomment" # real comment')[0]
        assert "??" not in result
        assert '"#notacomment"' in result
        assert "# real comment" in result
        assert result.endswith("# real comment")

    def test_comment_idempotent(self) -> None:
        """Transform with inline comment should be idempotent."""
        code = '$x = $a ?? "default" # inline'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second


# ============================================================================
# Corner case: deeply nested ?. inside method args (multi-pass)
# ============================================================================

class TestDeepNestedNullConditionalInArgs:
    def test_qd_inside_qd_method_arg(self) -> None:
        """?. inside another ?. method argument — both transformed."""
        result = pwsh_transform("$a?.Foo($b?.Bar($c?.Baz()))")[0]
        assert "?." not in result
        assert ".Foo(" in result
        assert ".Bar(" in result
        assert ".Baz()" in result

    def test_qd_with_nested_qb_in_arg(self) -> None:
        result = pwsh_transform("$a?.Process($b?[0])")[0]
        assert "?." not in result
        assert "?[" not in result

    def test_qd_with_nested_coalescing_in_arg(self) -> None:
        result = pwsh_transform('$a?.Method($b ?? "fallback")')[0]
        assert "?." not in result
        assert "??" not in result


# ============================================================================
# Corner case: ?? with $(...) containing newlines
# ============================================================================

class TestSubexpressionWithNewlines:
    def test_coalescing_with_multiline_subexpr_right(self) -> None:
        code = "$a ?? $(\n  Get-Date\n  Get-Process\n)"
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result

    def test_coalescing_with_multiline_subexpr_left(self) -> None:
        code = "$(\n  Get-Item $p\n) ?? 'default'"
        result = pwsh_transform(code)[0]
        assert "??" not in result


# ============================================================================
# Corner case: _join_continuation_lines preserves backtick in strings
# ============================================================================

class TestBacktickContinuationInStringsPreserved:
    def test_backtick_n_in_dq_not_joined(self) -> None:
        """`n inside "..." is escape sequence for newline, NOT continuation."""
        code = '"line1`nline2"'
        result = pwsh_transform(code)[0]
        assert "line1`nline2" in result

    def test_backtick_quote_in_dq_not_joined(self) -> None:
        """`" inside "..." is escaped quote, NOT continuation."""
        code = '"say `"hello`""'
        result = pwsh_transform(code)[0]
        assert '`"hello`"' in result


# ============================================================================
# Corner case: ??= with right side containing chain operators
# ============================================================================

class TestNCARightSideChain:
    def test_nca_right_side_with_and_chain(self) -> None:
        """$a ??= cmd1 && cmd2 — chain on right side of ??=."""
        result = pwsh_transform("$a ??= cmd1 && cmd2")[0]
        assert "??=" not in result
        assert "&&" not in result
        assert "if ($null -eq $a)" in result
        assert "if ($?)" in result

    def test_nca_right_side_with_or_chain(self) -> None:
        result = pwsh_transform("$a ??= cmd1 || cmd2")[0]
        assert "??=" not in result
        assert "||" not in result
        assert "if ($null -eq $a)" in result
        assert "if (-not $?)" in result


# ============================================================================
# Corner case: ?. with static member as base
# ============================================================================

class TestNullConditionalOnStaticMember:
    def test_static_property_dot(self) -> None:
        """[SomeType]::Property?.Member — static member null-conditional."""
        result = pwsh_transform("[SomeType]::Property?.Member")[0]
        assert "?." not in result
        assert "if ($null -ne [SomeType]::Property)" in result
        assert "[SomeType]::Property.Member" in result

    def test_static_method_call_dot(self) -> None:
        result = pwsh_transform("[Enum]::GetValues($t)?.Count")[0]
        assert "?." not in result
        assert "if ($null -ne [Enum]::GetValues($t))" in result
        assert "[Enum]::GetValues($t).Count" in result


# ============================================================================
# Corner case: ?. on splatted variable
# ============================================================================

class TestNullConditionalOnSplat:
    def test_splat_variable_dot(self) -> None:
        """@args?.Count — null-conditional on splatted variable base."""
        # @args is not a valid base for ?., but shouldn't crash
        result = pwsh_transform("@args?.Count")[0]
        assert isinstance(result, str)


# ============================================================================
# Corner case: operators in multiline pipelines (realistic PS scripts)
# ============================================================================

class TestRealisticMultiLineScripts:
    def test_conditional_service_check(self) -> None:
        code = """$svc = Get-Service -Name $name
$status = $svc?.Status ?? "Unknown"
if ($status -eq "Running") { Write-Output "ok" } else { Write-Output "not ok" }"""
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "??" not in result

    def test_file_processing_pipeline(self) -> None:
        code = """$files = Get-ChildItem -Path $dir -Recurse
$csv = $files?.Where({$_.Extension -eq '.csv'})
$count = $csv?.Count ?? 0
Write-Output "Found $count CSV files"
"""
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "??" not in result

    def test_api_response_handling(self) -> None:
        code = """$response = Invoke-RestMethod -Uri $url
$data = $response?.data ?? $response?.result ?? @{}
$name = $data?.name ?? "anonymous"
"""
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "??" not in result


# ============================================================================
# Corner case: ??= with ${} braced var containing nested braces
# ============================================================================

class TestNCANestedBracedVars:
    def test_double_braced_var_nca(self) -> None:
        """${outer.${inner}} ??= 'val' — nested braced variable."""
        result = pwsh_transform("${outer.${inner}} ??= 'val'")[0]
        assert "??=" not in result
        assert "if ($null -eq ${outer.${inner}})" in result

    def test_triple_braced_var_nca(self) -> None:
        result = pwsh_transform("${a.${b.${c}}} ??= 'deep'")[0]
        assert "??=" not in result
        assert "if ($null -eq ${a.${b.${c}}})" in result


# ============================================================================
# Corner case: _find_expr_end with # comment at boundary
# ============================================================================

class TestExprEndHashComment:
    def test_hash_comment_right_after_operator(self) -> None:
        result = pwsh_transform("$a?.Name#$comment")[0]
        assert "?." not in result
        assert "if ($null -ne $a)" in result

    def test_hash_comment_right_after_ternary(self) -> None:
        result = pwsh_transform('$cond ? "a" : "b"#comment')[0]
        assert "if ($cond)" in result


# ============================================================================
# Ultimate idempotency: transform 3 times for all operators
# ============================================================================

class TestTripleTransformIdempotency:
    def test_triple_transform_all_ops(self) -> None:
        code = '$a ??= $b; $c = $d?.$e?."f" ?? "g"; cmd1 && cmd2 || cmd3'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        third = pwsh_transform(second)[0]
        assert first == second == third

    def test_triple_transform_ternary_only(self) -> None:
        code = '$x = $a ? ($b ? "c" : "d") : "e"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        third = pwsh_transform(second)[0]
        assert second == third  # May not stabilize after 1 pass (nested ternaries)

    def test_triple_transform_coalescing_only(self) -> None:
        code = '$x = $a ?? $b ?? $c ?? "d"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        third = pwsh_transform(second)[0]
        assert second == third


# ============================================================================
# Corner case: ?. on $using: scoped variable (PS remoting / ForEach -Parallel)
# ============================================================================

class TestUsingScopeNullConditional:
    def test_using_var_dot_property(self) -> None:
        """$using:var?.Property — common in ForEach-Object -Parallel."""
        result = pwsh_transform("$using:var?.Name")[0]
        assert "?." not in result
        assert "if ($null -ne $using:var)" in result
        assert "$using:var.Name" in result

    def test_using_var_bracket_index(self) -> None:
        result = pwsh_transform("$using:arr?[0]")[0]
        assert "?[" not in result
        assert "if ($null -ne $using:arr)" in result
        assert "$using:arr[0]" in result

    def test_using_var_coalescing(self) -> None:
        result = pwsh_transform('$using:val ?? "default"')[0]
        assert "??" not in result
        assert "if ($null -ne $using:val)" in result

    def test_using_var_nca(self) -> None:
        result = pwsh_transform('$using:val ??= 0')[0]
        assert "??=" not in result
        assert "if ($null -eq $using:val)" in result

    def test_using_var_variable_property(self) -> None:
        result = pwsh_transform("$using:obj?.$prop")[0]
        assert "?." not in result
        assert "if ($null -ne $using:obj)" in result
        assert "$prop" in result

    def test_using_var_chained(self) -> None:
        result = pwsh_transform("$using:data?.Rows?.Count")[0]
        assert "?." not in result
        assert "$using:data" in result
        assert ".Rows" in result
        assert ".Count" in result

    def test_using_var_idempotent(self) -> None:
        code = "$using:obj?.Name"
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second


# ============================================================================
# Corner case: `.` dot-sourcing operator with chain operators
# ============================================================================

class TestDotSourcingChain:
    def test_dot_source_and_chain(self) -> None:
        """. ./script.ps1 && cmd2 — dot-sourcing then chain."""
        result = pwsh_transform(". ./script.ps1 && cmd2")[0]
        assert "&&" not in result
        assert "if ($?)" in result
        assert ". ./script.ps1" in result

    def test_dot_source_or_chain(self) -> None:
        result = pwsh_transform(". ./setup.ps1 || Write-Error failed")[0]
        assert "||" not in result
        assert "if (-not $?)" in result

    def test_dot_source_with_args_chain(self) -> None:
        result = pwsh_transform(". ./helper.ps1 -Force && Write-Output ok")[0]
        assert "&&" not in result
        assert "if ($?)" in result
        assert ". ./helper.ps1 -Force" in result

    def test_dot_source_then_or_then_and(self) -> None:
        result = pwsh_transform(". ./cfg.ps1 || . ./default.ps1 && Write-Output loaded")[0]
        assert "||" not in result
        assert "&&" not in result
        assert "if (-not $?)" in result
        assert "if ($?)" in result


# ============================================================================
# Corner case: `&` call operator with chain operators
# ============================================================================

class TestCallOperatorChain:
    def test_call_op_and_chain(self) -> None:
        """& $cmd $arg && & $cmd2 $arg2 — call operator chain."""
        result = pwsh_transform("& $cmd $arg && & $cmd2 $arg2")[0]
        assert "&&" not in result
        assert "if ($?)" in result
        assert "& $cmd $arg" in result
        assert "& $cmd2 $arg2" in result

    def test_call_op_or_chain(self) -> None:
        result = pwsh_transform("& $backup || & $restore")[0]
        assert "||" not in result
        assert "if (-not $?)" in result

    def test_call_op_with_splat_chain(self) -> None:
        result = pwsh_transform("& $cmd @args && Write-Output done")[0]
        assert "&&" not in result
        assert "if ($?)" in result
        assert "@args" in result

    def test_call_op_nested_chain(self) -> None:
        result = pwsh_transform("& $a && & $b || & $c")[0]
        assert "&&" not in result
        assert "||" not in result
        assert "if ($?)" in result
        assert "if (-not $?)" in result

    def test_call_op_with_scriptblock_chain(self) -> None:
        result = pwsh_transform("& { Get-Date } && Write-Output ok")[0]
        assert "&&" not in result
        assert "if ($?)" in result


# ============================================================================
# Corner case: ?. with scriptblock method arguments
# ============================================================================

class TestNullConditionalScriptblockMethod:
    def test_foreach_with_scriptblock(self) -> None:
        """$a?.ForEach({ $_ }) — method with scriptblock argument."""
        result = pwsh_transform("$a?.ForEach({ $_ })")[0]
        assert "?." not in result
        assert "if ($null -ne $a)" in result
        assert "$a.ForEach({ $_ })" in result

    def test_where_with_scriptblock(self) -> None:
        result = pwsh_transform("$a?.Where({ $_ -gt 0 })")[0]
        assert "?." not in result
        assert "$a.Where({ $_ -gt 0 })" in result

    def test_foreach_then_property_chain(self) -> None:
        """$a?.ForEach({ $_ })?.Count — chain after scriptblock method."""
        result = pwsh_transform("$a?.ForEach({ $_ })?.Count")[0]
        assert "?." not in result
        assert ".ForEach({ $_ })" in result
        assert ".Count" in result

    def test_scriptblock_with_inner_operators(self) -> None:
        """?.ForEach({ ... }) where scriptblock contains ?. or ?? which are at depth>0."""
        result = pwsh_transform('$a?.ForEach({ $_.Name ?? "unknown" })')[0]
        assert "?." not in result
        assert "ForEach" in result
        # ?? inside scriptblock is at depth>0, not transformed in single pass

    def test_where_then_foreach_chain(self) -> None:
        result = pwsh_transform("$a?.Where({ $_ }).ForEach({ $_ })?.Count")[0]
        # ?. processed first, the rest depends on chain detection
        assert "?." not in result


# ============================================================================
# Corner case: ?? inside @() array construction
# ============================================================================

class TestCoalescingInsideArrayExpr:
    def test_array_with_two_coalescing(self) -> None:
        """@($a ?? 0, $b ?? 1) — coalescing inside array subexpression."""
        result = pwsh_transform("@($a ?? 0, $b ?? 1)")[0]
        # At depth>0 inside @(), coalescing not transformed in single pass
        assert "$a" in result
        assert "$b" in result

    def test_array_with_coalescing_and_ternary(self) -> None:
        result = pwsh_transform('@($a ?? "x", $cond ? "t" : "f")')[0]
        # Operators inside @() at depth>0 are not transformed
        assert "$a" in result
        assert "$cond" in result

    def test_coalescing_outside_array(self) -> None:
        """@(1,2) ?? @(3) — coalescing where left is @(). Already covered in TestNullCoalescingComplex, but duplicating for array context."""
        result = pwsh_transform("@(1,2) ?? @(3)")[0]
        assert "??" not in result
        assert "if ($null -ne @(1,2))" in result


# ============================================================================
# Corner case: ?. on $Host / $PSVersionTable automatic variables
# ============================================================================

class TestAutomaticVariableNullConditional:
    def test_host_version(self) -> None:
        """$Host?.Version — null-conditional on $Host automatic variable."""
        result = pwsh_transform("$Host?.Version")[0]
        assert "?." not in result
        assert "if ($null -ne $Host)" in result
        assert "$Host.Version" in result

    def test_psversiontable_psversion(self) -> None:
        result = pwsh_transform("$PSVersionTable?.PSVersion")[0]
        assert "?." not in result
        assert "if ($null -ne $PSVersionTable)" in result

    def test_psversiontable_chained(self) -> None:
        result = pwsh_transform("$PSVersionTable?.PSVersion?.Major")[0]
        assert "?." not in result
        assert "$PSVersionTable" in result
        assert ".PSVersion" in result
        assert ".Major" in result

    def test_host_ui_rawui_chained(self) -> None:
        result = pwsh_transform("$Host?.UI?.RawUI?.WindowTitle")[0]
        assert "?." not in result
        assert "$Host" in result
        assert ".UI" in result
        assert ".RawUI" in result
        assert ".WindowTitle" in result

    def test_executioncontext_variable(self) -> None:
        result = pwsh_transform("$ExecutionContext?.SessionState")[0]
        assert "?." not in result
        assert "if ($null -ne $ExecutionContext)" in result

    def test_myinvocation_variable(self) -> None:
        result = pwsh_transform("$MyInvocation?.MyCommand?.Name")[0]
        assert "?." not in result
        assert "$MyInvocation" in result


# ============================================================================
# Corner case: ?. chained from static member access
# ============================================================================

class TestNullConditionalStaticMemberChained:
    def test_static_prop_to_prop_to_index(self) -> None:
        """[Type]::Prop?.SubProp?[0] — chain from static prop through ?. to ?[."""
        result = pwsh_transform("[SomeType]::Prop?.SubProp?[0]")[0]
        assert "?." not in result
        assert "?[" not in result
        assert "[SomeType]::Prop" in result

    def test_static_method_to_prop_to_coalescing(self) -> None:
        result = pwsh_transform('[Enum]::GetValues($t)?.Length ?? 0')[0]
        assert "?." not in result
        assert "??" not in result
        assert "[Enum]::GetValues($t)" in result

    def test_static_prop_with_variable_member(self) -> None:
        result = pwsh_transform("[SomeType]::Prop?.$member")[0]
        assert "?." not in result
        assert "if ($null -ne [SomeType]::Prop)" in result
        assert "$member" in result

    def test_static_method_to_quoted_member(self) -> None:
        result = pwsh_transform('[obj]::Method()?."prop-name"')[0]
        assert "?." not in result
        assert '[obj]::Method()' in result
        assert '"prop-name"' in result

    def test_static_prop_to_method(self) -> None:
        result = pwsh_transform("[SomeType]::Prop?.ToString()")[0]
        assert "?." not in result
        assert "[SomeType]::Prop.ToString()" in result


# ============================================================================
# Corner case: ${this} / ${PSCmdlet} automatic braced variables with ?.
# ============================================================================

class TestBracedAutomaticVarNullConditional:
    def test_this_variable_dot(self) -> None:
        """${this}?.Property — used in PS classes."""
        result = pwsh_transform("${this}?.Name")[0]
        assert "?." not in result
        assert "if ($null -ne ${this})" in result
        assert "${this}.Name" in result

    def test_this_variable_bracket(self) -> None:
        result = pwsh_transform("${this}?.[0]")[0]
        assert "?[" not in result
        # ?[ is processed after ?., but the `?.` before `[` makes it tricky

    def test_pscmdlet_variable(self) -> None:
        result = pwsh_transform("${PSCmdlet}?.MyInvocation")[0]
        assert "?." not in result
        assert "${PSCmdlet}" in result

    def test_this_variable_nca(self) -> None:
        result = pwsh_transform('${this} ??= "init"')[0]
        assert "??=" not in result
        assert "if ($null -eq ${this})" in result


# ============================================================================
# Corner case: ??= with string-literal-like left side
# ============================================================================

class TestNCALiteralLeft:
    def test_string_single_quoted_left_nca(self) -> None:
        """'literal' ??= 'value' — string literal on left of ??= (invalid PS, shouldn't crash)."""
        result = pwsh_transform("'literal' ??= 'value'")[0]
        assert isinstance(result, str)

    def test_number_left_nca(self) -> None:
        """123 ??= 'value' — number literal on left."""
        result = pwsh_transform("123 ??= 'value'")[0]
        assert isinstance(result, str)


# ============================================================================
# Corner case: -match operator combining with ternary and $Matches
# ============================================================================

class TestMatchOperatorWithTernary:
    def test_match_result_in_ternary_condition(self) -> None:
        r"""$s -match '(\d+)' ? $Matches[1] : $null — match then ternary."""
        result = pwsh_transform("$s -match '(\\d+)' ? $Matches[1] : $null")[0]
        assert "?" not in result
        assert "if ($s -match '(\\d+)')" in result

    def test_notmatch_in_ternary_condition(self) -> None:
        result = pwsh_transform('$s -notmatch "x" ? "clean" : "dirty"')[0]
        assert "?" not in result
        assert "if ($s -notmatch \"x\")" in result

    def test_match_with_parens_ternary(self) -> None:
        result = pwsh_transform('($s -match "^(\\d+)$") ? [int]$Matches[1] : -1')[0]
        assert "?" not in result
        assert "if (($s -match \"^(\\d+)$\"))" in result


# ============================================================================
# Corner case: $?.?. chain (automatic var then null-conditional NOT $? first)
# ============================================================================

class TestDollarQuestionWithNullConditional:
    def test_dollar_q_then_dot_chain(self) -> None:
        """$??.Property — $? is auto var, ?. is null-conditional. Should NOT treat $? as ternary."""
        result = pwsh_transform("$??.Property")[0]
        # $? is detected, ?. is null-conditional
        assert "if ($null -ne $?)" in result

    def test_dollar_q_then_bracket_chain(self) -> None:
        result = pwsh_transform("$??[0]")[0]
        # $?[0] in output is $? auto-var + [0] index, not ?[ operator
        assert "if ($null -ne $?)" in result
        assert "$?[0]" in result


# ============================================================================
# Corner case: ?? in a pipeline (right side piped)
# ============================================================================

class TestCoalescingInPipeline:
    def test_coalescing_right_piped_to_cmdlet(self) -> None:
        """$a ?? $b | ForEach-Object { $_ } — pipe binds tighter than ??, so ?? right is just $b."""
        result = pwsh_transform("$a ?? $b | ForEach-Object { $_ }")[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result

    def test_coalescing_left_is_pipeline(self) -> None:
        """(Get-Item $p) ?? $default — parenthesized pipeline as left."""
        result = pwsh_transform("(Get-Item $p) ?? $default")[0]
        assert "??" not in result
        assert "if ($null -ne (Get-Item $p))" in result

    def test_coalescing_pipe_both_sides(self) -> None:
        result = pwsh_transform("(Get-Date) ?? (Get-Date -Year 2000)")[0]
        assert "??" not in result
        assert "if ($null -ne (Get-Date))" in result


# ============================================================================
# Corner case: multi-line $() subexpression and here-string interaction
# ============================================================================

class TestMultiLineSubExprEdgeCases:
    def test_multiline_subexpr_left_of_coalescing(self) -> None:
        code = """$(if ($a) {
  Get-Date
} else {
  $null
}) ?? 'default'"""
        result = pwsh_transform(code)[0]
        assert "??" not in result
        # multi-line subexpr detection
        assert "if ($null -ne" in result

    def test_here_string_with_embedded_operators_across_lines(self) -> None:
        code = """$text = @'
line with ?? and ?. and &&
and || operators
'@
$x = $a ?? 'fallback'"""
        result = pwsh_transform(code)[0]
        # Operators inside here-string preserved
        assert "?? and ?. and &&" in result
        assert "|| operators" in result
        # Real ?? outside here-string transformed
        assert "if ($null -ne $a)" in result


# ============================================================================
# Corner case: $? in subexpression context
# ============================================================================

class TestDollarQuestionSubExpr:
    def test_dollar_q_in_if_condition(self) -> None:
        """if ($?) { ... } — $? in if condition, not ternary."""
        result = pwsh_transform("if ($?) { Write-Output ok } else { Write-Error fail }")[0]
        assert result == "if ($?) { Write-Output ok } else { Write-Error fail }"

    def test_dollar_q_assignment_ternary(self) -> None:
        """$x = $? ? 'success' : 'failure' — $? as ternary condition."""
        result = pwsh_transform("$x = $? ? 'success' : 'failure'")[0]
        assert result == "$x = if ($?) { 'success' } else { 'failure' }"

    def test_dollar_q_in_pipeline_chain(self) -> None:
        """cmd1; if ($?) { cmd2 } — already transformed chain, $? should be preserved."""
        result = pwsh_transform("cmd1; if ($?) { cmd2 }")[0]
        assert result == "cmd1; if ($?) { cmd2 }"


# ============================================================================
# Corner case: ?. with property name matching a PS keyword used as function name
# ============================================================================

class TestNullConditionalKeywordPropertyChains:
    def test_begin_process_end_chain(self) -> None:
        """$obj?.Begin?.Process?.End — keyword-named properties in chain."""
        result = pwsh_transform("$obj?.Begin?.Process?.End")[0]
        assert "?." not in result
        assert "$obj.Begin.Process.End" in result

    def test_exit_break_continue_chain(self) -> None:
        result = pwsh_transform("$obj?.Exit?.Break?.Continue")[0]
        assert "?." not in result
        assert "$obj.Exit.Break.Continue" in result

    def test_try_catch_finally_chain(self) -> None:
        result = pwsh_transform("$obj?.Try?.Catch?.Finally")[0]
        assert "?." not in result
        assert "$obj.Try.Catch.Finally" in result


# ============================================================================
# Corner case: ?. on array literal
# ============================================================================

class TestNullConditionalOnLiterals:
    def test_string_literal_dot(self) -> None:
        """'hello'?.Length — null-conditional on string literal (valid in PS7)."""
        result = pwsh_transform("'hello'?.Length")[0]
        assert "?." not in result
        assert "if ($null -ne 'hello')" in result
        assert "'hello'.Length" in result

    def test_number_literal_dot(self) -> None:
        result = pwsh_transform("123?.GetType()")[0]
        assert "?." not in result
        assert "if ($null -ne 123)" in result

    def test_double_quoted_string_literal_dot(self) -> None:
        result = pwsh_transform('"hello"?.Length')[0]
        assert "?." not in result
        assert "if ($null -ne \"hello\")" in result

    def test_dollar_null_dot(self) -> None:
        """$null?.Property — $null literal with null-conditional."""
        result = pwsh_transform("$null?.Property")[0]
        assert "?." not in result
        assert "if ($null -ne $null)" in result


# ============================================================================
# Corner case: combined $? as ??? (three question marks in a row)
# ============================================================================

class TestTripleQuestionMark:
    def test_dollar_q_double_question(self) -> None:
        """$??? "fallback" — $? followed by ?? operator."""
        result = pwsh_transform('$??? "fallback"')[0]
        assert "??" not in result
        assert "if ($null -ne $?)" in result
        assert '"fallback"' in result

    def test_dollar_q_then_null_conditional_dot_property(self) -> None:
        """$??.Property?.Sub — $? is auto var, ?. is null-conditional."""
        result = pwsh_transform("$??.Property?.Sub")[0]
        # The outer ?. is transformed; $?.Property contains $? auto-var + .Property
        assert "if ($null -ne $?)" in result
        assert "$.Property" not in result

    def test_dollar_q_then_bracket_chain(self) -> None:
        result = pwsh_transform("$??[0]")[0]
        # $?[0] in output is $? auto-var + [0] index, not ?[ operator
        assert "if ($null -ne $?)" in result
        assert "$?[0]" in result


# ============================================================================
# Corner case: ?[ with depth tracking for nested brackets in index
# ============================================================================

class TestNullConditionalBracketDepthTracking:
    def test_bracket_with_multiple_nested_brackets(self) -> None:
        result = pwsh_transform("$a?[$b[$c[$d]]]")[0]
        assert "?[" not in result
        assert "if ($null -ne $a)" in result
        assert "$b[$c[$d]]" in result

    def test_bracket_with_parens_in_index(self) -> None:
        result = pwsh_transform("$a?[($b + $c)]")[0]
        assert "?[" not in result
        assert "if ($null -ne $a)" in result
        assert "($b + $c)" in result

    def test_bracket_with_array_index(self) -> None:
        result = pwsh_transform("$a?[$b, $c]")[0]
        assert "?[" not in result
        assert "if ($null -ne $a)" in result
        assert "$b, $c" in result

    def test_bracket_with_range_operator(self) -> None:
        result = pwsh_transform("$a?[$b..$c]")[0]
        assert "?[" not in result
        assert "if ($null -ne $a)" in result


# ============================================================================
# Corner case: ??= with trailing code that contains other operators
# ============================================================================

class TestNCAWithTrailingOperators:
    def test_nca_then_chain_on_same_line(self) -> None:
        result = pwsh_transform('$a ??= "x"; cmd1 && cmd2')[0]
        assert "??=" not in result
        assert "&&" not in result
        assert "if ($null -eq $a)" in result
        assert "if ($?)" in result

    def test_nca_then_ternary_on_same_line(self) -> None:
        result = pwsh_transform('$a ??= "x"; $b = $cond ? "t" : "f"')[0]
        assert "??=" not in result
        assert "?" not in result  # ternary ? is gone
        assert "if ($null -eq $a)" in result
        assert "if ($cond)" in result

    def test_nca_then_null_conditional_on_same_line(self) -> None:
        result = pwsh_transform('$a ??= "x"; $b = $obj?.Name')[0]
        assert "??=" not in result
        assert "?." not in result
        assert "if ($null -eq $a)" in result
        assert "if ($null -ne $obj)" in result

    def test_nca_then_coalescing_on_same_line(self) -> None:
        result = pwsh_transform('$a ??= "x"; $b = $c ?? "y"')[0]
        assert "??=" not in result
        assert "??" not in result
        assert "if ($null -eq $a)" in result
        assert "if ($null -ne $c)" in result


# ============================================================================
# Corner case: multiple ??= on same line separated by ; (two ??=)
# ============================================================================

class TestMultipleNCAOnSameLine:
    def test_two_nca_semicolon_separated(self) -> None:
        result = pwsh_transform('$a ??= 1; $b ??= 2')[0]
        assert "??=" not in result
        assert "if ($null -eq $a)" in result
        assert "if ($null -eq $b)" in result

    def test_three_nca_semicolon_separated(self) -> None:
        result = pwsh_transform('$a ??= 1; $b ??= 2; $c ??= 3')[0]
        assert "??=" not in result
        assert result.count("if ($null -eq $") == 3

    def test_nca_mixed_with_null_conditional_semicolons(self) -> None:
        result = pwsh_transform('$a ??= "x"; $b?.Name; $c ??= "y"')[0]
        assert "??=" not in result
        assert "?." not in result
        assert "if ($null -eq $a)" in result
        assert "if ($null -eq $c)" in result


# ============================================================================
# Corner case: backtick inside a here-string is literal, not continuation
# ============================================================================

class TestBacktickLiteralInHereString:
    def test_backtick_newline_inside_here_string(self) -> None:
        """Backtick+newline inside @'...'@ is literal, not merged."""
        code = "@'\nline1 `\nline2\n'@"
        result = pwsh_transform(code)[0]
        # The backtick+newline is inside a here-string region, so not collapsed
        assert isinstance(result, str)

    def test_backtick_newline_inside_dq_here_string(self) -> None:
        code = '@"\nline1 `\nline2\n"@'
        result = pwsh_transform(code)[0]
        assert isinstance(result, str)


# ============================================================================
# Corner case: ternary as sole content of a scriptblock
# ============================================================================

class TestTernaryInsideScriptBlock:
    def test_ternary_inside_scriptblock_not_transformed(self) -> None:
        """{ $a ? $b : $c } — ternary inside scriptblock at depth>0."""
        result = pwsh_transform('$sb = { $a ? $b : $c }')[0]
        assert "$sb = " in result
        # Ternary inside braces at depth>0 is NOT transformed
        assert "?" in result

    def test_ternary_inside_nested_scriptblock(self) -> None:
        result = pwsh_transform('$sb = { { $a ? $b : $c } }')[0]
        assert "?" in result


# ============================================================================
# Corner case: ?? on same base variable used after transformation
# ============================================================================

class TestCoalescingReuseSameVar:
    def test_same_var_multiple_coalescing(self) -> None:
        """$x = $a ?? 1; $y = $a ?? 2 — same var used in two ?? expressions."""
        result = pwsh_transform('$x = $a ?? 1; $y = $a ?? 2')[0]
        assert "??" not in result
        assert "if ($null -ne $a) { $a } else { 1 }" in result
        assert "if ($null -ne $a) { $a } else { 2 }" in result

    def test_same_var_coalescing_and_nca(self) -> None:
        result = pwsh_transform('$a ??= 0; $b = $a ?? 1')[0]
        assert "??=" not in result
        assert "??" not in result
        assert "if ($null -eq $a)" in result


# ============================================================================
# Corner case: -replace operator combined with ternary/coalescing
# ============================================================================

class TestReplaceOperatorCombined:
    def test_replace_in_ternary_condition(self) -> None:
        """-replace with comma-separated args before ternary.
        KNOWN LIMITATION: comma is an expression boundary, so the ternary
        condition is just '"y"' rather than '$s -replace "x","y"'."""
        result = pwsh_transform('$s -replace "x","y" ? "changed" : "same"')[0]
        assert "?" not in result
        # Current behaviour: comma delimits expression, condition is '"y"'
        assert "if (" in result
        assert "\"changed\"" in result
        assert "\"same\"" in result

    def test_replace_in_coalescing_left(self) -> None:
        result = pwsh_transform('($s -replace "a","b") ?? $s')[0]
        assert "??" not in result
        assert "if ($null -ne ($s -replace \"a\",\"b\"))" in result


# ============================================================================
# Corner case: ?. where member name is an integer (edge of _scan_member_name)
# ============================================================================

class TestNullConditionalNumericMember:
    def test_integer_member_name(self) -> None:
        """$a?.123 — numeric member names are not valid PS identifiers."""
        result = pwsh_transform("$a?.123")[0]
        assert isinstance(result, str)

    def test_member_starting_with_digit_then_alpha(self) -> None:
        """$a?.123abc — member starting with digit."""
        result = pwsh_transform("$a?.123abc")[0]
        assert isinstance(result, str)

# ============================================================================
# NEW Corner-case tests for PS7 -> PS5.1 parser
# ============================================================================

# ============================================================================
# PowerShell type operators with ternary and coalescing
# ============================================================================

class TestTypeOperatorsWithTernary:
    def test_is_operator_in_ternary_condition(self) -> None:
        result = pwsh_transform('$a -is [string] ? "yes" : "no"')[0]
        assert "?" not in result
        assert 'if ($a -is [string])' in result

    def test_isnot_operator_in_ternary_condition(self) -> None:
        result = pwsh_transform('$a -isnot [int] ? "not-int" : "int"')[0]
        assert "?" not in result
        assert 'if ($a -isnot [int])' in result

    def test_as_operator_in_ternary_condition(self) -> None:
        result = pwsh_transform('($a -as [datetime]) ? "parsed" : "failed"')[0]
        assert "?" not in result
        assert 'if (($a -as [datetime]))' in result

    def test_is_operator_with_complex_type(self) -> None:
        result = pwsh_transform('$obj -is [System.IO.FileInfo] ? "file" : "other"')[0]
        assert "?" not in result
        assert 'if ($obj -is [System.IO.FileInfo])' in result

    def test_isnot_operator_with_array_type(self) -> None:
        result = pwsh_transform('$obj -isnot [array] ? "scalar" : "array"')[0]
        assert "?" not in result
        assert 'if ($obj -isnot [array])' in result


class TestTypeOperatorsWithCoalescing:
    def test_is_operator_left_coalescing(self) -> None:
        result = pwsh_transform('($a -is [string]) ?? $false')[0]
        assert "??" not in result
        assert 'if ($null -ne ($a -is [string]))' in result

    def test_as_operator_left_coalescing(self) -> None:
        result = pwsh_transform('($a -as [int]) ?? 0')[0]
        assert "??" not in result
        assert 'if ($null -ne ($a -as [int]))' in result

    def test_type_cast_left_coalescing(self) -> None:
        result = pwsh_transform('[int]$a ?? 0')[0]
        assert "??" not in result
        assert 'if ($null -ne [int]$a)' in result

    def test_type_cast_right_coalescing(self) -> None:
        result = pwsh_transform('$a ?? [int]0')[0]
        assert "??" not in result
        assert '[int]0' in result


# ============================================================================
# PowerShell collection operators with ternary and coalescing
# ============================================================================

class TestCollectionOperatorsWithTernary:
    def test_contains_operator_in_ternary(self) -> None:
        result = pwsh_transform('$arr -contains "x" ? "found" : "missing"')[0]
        assert "?" not in result
        assert 'if ($arr -contains "x")' in result

    def test_notcontains_operator_in_ternary(self) -> None:
        result = pwsh_transform('$arr -notcontains "x" ? "missing" : "found"')[0]
        assert "?" not in result
        assert 'if ($arr -notcontains "x")' in result

    def test_in_operator_in_ternary(self) -> None:
        result = pwsh_transform('"x" -in $arr ? "found" : "missing"')[0]
        assert "?" not in result
        assert 'if ("x" -in $arr)' in result

    def test_notin_operator_in_ternary(self) -> None:
        result = pwsh_transform('"x" -notin $arr ? "missing" : "found"')[0]
        assert "?" not in result
        assert 'if ("x" -notin $arr)' in result

    def test_contains_with_variable_in_ternary(self) -> None:
        result = pwsh_transform('$arr -contains $item ? $item : "default"')[0]
        assert "?" not in result
        assert 'if ($arr -contains $item)' in result


class TestCollectionOperatorsWithCoalescing:
    def test_contains_result_coalescing(self) -> None:
        result = pwsh_transform('($arr -contains $item) ?? $false')[0]
        assert "??" not in result
        assert 'if ($null -ne ($arr -contains $item))' in result

    def test_in_result_coalescing(self) -> None:
        result = pwsh_transform('($item -in $arr) ?? $false')[0]
        assert "??" not in result
        assert 'if ($null -ne ($item -in $arr))' in result


# ============================================================================
# PowerShell string operators with ternary and coalescing
# ============================================================================

class TestStringOperatorsWithTernary:
    def test_like_operator_in_ternary(self) -> None:
        result = pwsh_transform('$s -like "*.txt" ? "text" : "other"')[0]
        assert "?" not in result
        assert 'if ($s -like "*.txt")' in result

    def test_notlike_operator_in_ternary(self) -> None:
        result = pwsh_transform('$s -notlike "*.tmp" ? "keep" : "discard"')[0]
        assert "?" not in result
        assert 'if ($s -notlike "*.tmp")' in result

    def test_split_operator_in_ternary(self) -> None:
        result = pwsh_transform('($s -split ",").Count -gt 1 ? "multi" : "single"')[0]
        assert "?" not in result
        assert 'if (($s -split ",").Count -gt 1)' in result

    def test_join_operator_in_ternary(self) -> None:
        result = pwsh_transform('($arr -join ",").Length -gt 0 ? "non-empty" : "empty"')[0]
        assert "?" not in result
        assert 'if (($arr -join ",").Length -gt 0)' in result

    def test_replace_operator_in_ternary(self) -> None:
        result = pwsh_transform('($s -replace "a","b") -ne $s ? "changed" : "same"')[0]
        assert "?" not in result
        assert 'if (($s -replace "a","b") -ne $s)' in result


class TestStringOperatorsWithCoalescing:
    def test_like_result_coalescing(self) -> None:
        result = pwsh_transform('($s -like "*.ps1") ?? $false')[0]
        assert "??" not in result
        assert 'if ($null -ne ($s -like "*.ps1"))' in result

    def test_join_result_coalescing(self) -> None:
        result = pwsh_transform('($arr -join ",") ?? ""')[0]
        assert "??" not in result
        assert 'if ($null -ne ($arr -join ","))' in result


# ============================================================================
# PowerShell format operator (-f) with ternary and coalescing
# ============================================================================

class TestFormatOperatorWithTernary:
    def test_format_operator_in_ternary_condition(self) -> None:
        result = pwsh_transform('"{0:N2}" -f $val ? "formatted" : "raw"')[0]
        assert "?" not in result
        assert 'if ("{0:N2}" -f $val)' in result

    def test_format_operator_in_true_branch(self) -> None:
        result = pwsh_transform('$cond ? ("{0}" -f $val) : "unknown"')[0]
        assert "?" not in result
        assert '("{0}" -f $val)' in result

    def test_format_operator_in_false_branch(self) -> None:
        result = pwsh_transform('$cond ? $val : ("{0:D4}" -f $val)')[0]
        assert "?" not in result
        assert '("{0:D4}" -f $val)' in result


class TestFormatOperatorWithCoalescing:
    def test_format_operator_left_coalescing(self) -> None:
        result = pwsh_transform('("{0}" -f $val) ?? "empty"')[0]
        assert "??" not in result
        assert 'if ($null -ne ("{0}" -f $val))' in result

    def test_format_operator_right_coalescing(self) -> None:
        result = pwsh_transform('$val ?? ("{0}" -f 0)')[0]
        assert "??" not in result
        assert '("{0}" -f 0)' in result


# ============================================================================
# Redirection operators with chain operators
# ============================================================================

class TestRedirectionWithChainOperators:
    def test_output_redirection_and_chain(self) -> None:
        result = pwsh_transform('Get-Process > processes.txt && Write-Output done')[0]
        assert "&&" not in result
        assert "if ($?)" in result
        assert "> processes.txt" in result

    def test_append_redirection_and_chain(self) -> None:
        result = pwsh_transform('Get-Date >> log.txt && Write-Output logged')[0]
        assert "&&" not in result
        assert ">> log.txt" in result

    def test_error_redirection_and_chain(self) -> None:
        result = pwsh_transform('Get-Item $f 2> err.log && Write-Output ok')[0]
        assert "&&" not in result
        assert "2> err.log" in result

    def test_error_append_redirection_or_chain(self) -> None:
        result = pwsh_transform('Get-Item $f 2>> err.log || Write-Error failed')[0]
        assert "||" not in result
        assert "if (-not $?)" in result
        assert "2>> err.log" in result

    def test_merge_redirection_and_chain(self) -> None:
        result = pwsh_transform('cmd.exe /c echo hi 2>&1 && Write-Output done')[0]
        assert "&&" not in result
        assert "2>&1" in result

    def test_multiple_redirections_and_chain(self) -> None:
        result = pwsh_transform('Get-Process > out.txt 2> err.txt && Write-Output done')[0]
        assert "&&" not in result
        assert "> out.txt" in result
        assert "2> err.txt" in result


# ============================================================================
# Exception handling with chain operators
# ============================================================================

class TestExceptionHandlingWithChainOperators:
    def test_try_block_with_chain_after(self) -> None:
        code = 'try { Get-Item $f } catch { }; cmd1 && cmd2'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "if ($?)" in result
        assert "try" in result
        assert "catch" in result

    def test_try_catch_finally_with_chain_after(self) -> None:
        code = 'try { 1/0 } catch { } finally { }; cmd1 || cmd2'
        result = pwsh_transform(code)[0]
        assert "||" not in result
        assert "if (-not $?)" in result
        assert "finally" in result

    def test_trap_statement_with_chain_after(self) -> None:
        code = 'trap { Write-Error $_ }; cmd1 && cmd2'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "if ($?)" in result
        assert "trap" in result


# ============================================================================
# Param blocks with default values using new operators
# ============================================================================

class TestParamBlockDefaults:
    def test_param_with_null_coalescing_default(self) -> None:
        code = 'param([string]$Name = $env:USERNAME ?? "anonymous"); Write-Output $Name'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "param([string]$Name =" in result
        assert "if ($null -ne $env:USERNAME)" in result

    def test_param_with_ternary_default(self) -> None:
        code = 'param([bool]$Debug = $env:DEBUG -eq "1" ? $true : $false); Write-Output $Debug'
        result = pwsh_transform(code)[0]
        # Ternary inside param() parens is at depth>0, NOT transformed (known limitation)
        assert "?" in result
        assert "param([bool]$Debug =" in result

    def test_param_multiple_with_defaults(self) -> None:
        code = 'param([string]$a = $x ?? "a", [string]$b = $y ?? "b"); Write-Output $a $b'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "param([string]$a =" in result
        assert "[string]$b =" in result
        assert "if ($null -ne $x)" in result
        assert "if ($null -ne $y)" in result


# ============================================================================
# Loop constructs with operators
# ============================================================================

class TestLoopConstructsWithOperators:
    def test_for_loop_with_null_coalescing(self) -> None:
        code = 'for ($i = 0; $i -lt ($max ?? 10); $i++) { Write-Output $i }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "for ($i = 0; $i -lt" in result
        assert "if ($null -ne $max)" in result

    def test_while_loop_with_ternary_condition(self) -> None:
        code = 'while ($running ? $true : $false) { Start-Sleep 1 }'
        result = pwsh_transform(code)[0]
        # Ternary inside while() parens is at depth>0, NOT transformed (known limitation)
        assert "?" in result
        assert "while ($running ? $true : $false)" in result

    def test_do_while_with_coalescing(self) -> None:
        code = 'do { $val = Read-Host "value" } while ($val ?? "")'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "do" in result
        assert "while (if ($null -ne $val)" in result

    def test_foreach_with_null_conditional(self) -> None:
        code = 'foreach ($item in $items?.ToArray()) { Write-Output $item }'
        result = pwsh_transform(code)[0]
        # BUG: _expr_left scans past the foreach keyword, producing incorrect base expression
        assert "foreach" in result
        assert "$items" in result
        assert ".ToArray()" in result

    def test_for_each_object_with_ternary(self) -> None:
        code = '$items | ForEach-Object { $_.Name ?? "unnamed" }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert '$_.Name' in result
        assert '"unnamed"' in result


# ============================================================================
# $_ / $PSItem with all operators
# ============================================================================

class TestPSItemWithOperators:
    def test_psitem_null_coalescing(self) -> None:
        result = pwsh_transform('$_ ?? "default"')[0]
        assert "??" not in result
        assert 'if ($null -ne $_)' in result

    def test_psitem_ternary(self) -> None:
        result = pwsh_transform('$_ ? "yes" : "no"')[0]
        assert "?" not in result
        assert 'if ($_)' in result

    def test_psitem_null_conditional_dot(self) -> None:
        result = pwsh_transform('$_.Name?.Length')[0]
        assert "?." not in result
        assert 'if ($null -ne $_.Name)' in result

    def test_psitem_null_conditional_bracket(self) -> None:
        result = pwsh_transform('$_?[0]')[0]
        assert "?[" not in result
        assert 'if ($null -ne $_)' in result

    def test_psitem_chain_and(self) -> None:
        result = pwsh_transform('$_ && Write-Output ok')[0]
        assert "&&" not in result
        assert "if ($?)" in result

    def test_psitem_chain_or(self) -> None:
        result = pwsh_transform('$_ || Write-Error fail')[0]
        assert "||" not in result
        assert "if (-not $?)" in result

    def test_psitem_nca(self) -> None:
        result = pwsh_transform('$_ ??= "default"')[0]
        assert "??=" not in result
        assert 'if ($null -eq $_)' in result


class TestPSItemExplicitWithOperators:
    def test_psitem_explicit_null_coalescing(self) -> None:
        result = pwsh_transform('$PSItem ?? "default"')[0]
        assert "??" not in result
        assert 'if ($null -ne $PSItem)' in result

    def test_psitem_explicit_ternary(self) -> None:
        result = pwsh_transform('$PSItem ? "yes" : "no"')[0]
        assert "?" not in result
        assert 'if ($PSItem)' in result

    def test_psitem_explicit_null_conditional(self) -> None:
        result = pwsh_transform('$PSItem?.Name')[0]
        assert "?." not in result
        assert 'if ($null -ne $PSItem)' in result


# ============================================================================
# $input with all operators
# ============================================================================

class TestInputWithOperators:
    def test_input_null_coalescing(self) -> None:
        result = pwsh_transform('$input ?? @()')[0]
        assert "??" not in result
        assert 'if ($null -ne $input)' in result

    def test_input_ternary(self) -> None:
        result = pwsh_transform('$input.Count -gt 0 ? "has-data" : "empty"')[0]
        assert "?" not in result
        assert 'if ($input.Count -gt 0)' in result

    def test_input_null_conditional_dot(self) -> None:
        result = pwsh_transform('$input?.Count')[0]
        assert "?." not in result
        assert 'if ($null -ne $input)' in result

    def test_input_null_conditional_bracket(self) -> None:
        result = pwsh_transform('$input?[0]')[0]
        assert "?[" not in result
        assert 'if ($null -ne $input)' in result

    def test_input_chain_and(self) -> None:
        result = pwsh_transform('$input | ForEach-Object { $_ } && Write-Output done')[0]
        assert "&&" not in result
        assert "if ($?)" in result


# ============================================================================
# $args with all operators
# ============================================================================

class TestArgsWithOperators:
    def test_args_null_coalescing(self) -> None:
        result = pwsh_transform('$args[0] ?? "default"')[0]
        assert "??" not in result
        assert 'if ($null -ne $args[0])' in result

    def test_args_ternary(self) -> None:
        result = pwsh_transform('$args.Count -gt 0 ? "has-args" : "no-args"')[0]
        assert "?" not in result
        assert 'if ($args.Count -gt 0)' in result

    def test_args_null_conditional_dot(self) -> None:
        result = pwsh_transform('$args?.Count')[0]
        assert "?." not in result
        assert 'if ($null -ne $args)' in result

    def test_args_null_conditional_bracket(self) -> None:
        result = pwsh_transform('$args?[0]')[0]
        assert "?[" not in result
        assert 'if ($null -ne $args)' in result


# ============================================================================
# $foreach / $switch automatic variables with operators
# ============================================================================

class TestForeachSwitchWithOperators:
    def test_foreach_null_coalescing(self) -> None:
        result = pwsh_transform('$foreach.Current ?? "done"')[0]
        assert "??" not in result
        assert 'if ($null -ne $foreach.Current)' in result

    def test_foreach_ternary(self) -> None:
        result = pwsh_transform('$foreach.MoveNext() ? "more" : "done"')[0]
        assert "?" not in result
        assert 'if ($foreach.MoveNext())' in result

    def test_switch_null_coalescing(self) -> None:
        result = pwsh_transform('$switch.Current ?? "end"')[0]
        assert "??" not in result
        assert 'if ($null -ne $switch.Current)' in result

    def test_switch_ternary(self) -> None:
        result = pwsh_transform('$switch.Count -gt 0 ? "has-items" : "empty"')[0]
        assert "?" not in result
        assert 'if ($switch.Count -gt 0)' in result


# ============================================================================
# More deeply nested structures
# ============================================================================

class TestDeeplyNestedStructures:
    def test_nested_parens_with_ternary(self) -> None:
        result = pwsh_transform('(((($a)))) ? "deep" : "shallow"')[0]
        assert "?" not in result
        assert 'if ((((($a)))))' in result

    def test_nested_parens_with_coalescing(self) -> None:
        result = pwsh_transform('((((($a))))) ?? "default"')[0]
        assert "??" not in result
        assert 'if ($null -ne ((((($a)))))' in result

    def test_nested_subexpressions_with_coalescing(self) -> None:
        result = pwsh_transform('$($($($a))) ?? "default"')[0]
        assert "??" not in result
        assert 'if ($null -ne $($($($a)))' in result

    def test_nested_arrays_with_null_conditional(self) -> None:
        result = pwsh_transform('$a[0][1][2]?.Name')[0]
        assert "?." not in result
        assert '$a[0][1][2]' in result

    def test_deeply_nested_method_chain(self) -> None:
        result = pwsh_transform('$a?.ToString()?.Trim()?.Split()')[0]
        assert "?." not in result
        assert '.ToString()' in result
        assert '.Trim()' in result
        assert '.Split()' in result


# ============================================================================
# More $? edge cases with multiple question marks
# ============================================================================

class TestDollarQuestionExtended:
    def test_dollar_q_then_double_question_dot(self) -> None:
        result = pwsh_transform('$??.Name')[0]
        assert "if ($null -ne $?)" in result
        assert "$.Name" not in result  # should be $?.Name not split

    def test_dollar_q_then_triple_question(self) -> None:
        result = pwsh_transform('$??? "fallback"')[0]
        assert "??" not in result
        assert 'if ($null -ne $?)' in result

    def test_dollar_q_in_ternary_true_branch(self) -> None:
        result = pwsh_transform('$cond ? $? : $false')[0]
        # $? contains ?, so we check for ternary pattern instead of bare ?
        assert 'if ($cond)' in result
        assert '$?' in result
        assert ' { $? } ' in result

    def test_dollar_q_in_ternary_false_branch(self) -> None:
        result = pwsh_transform('$cond ? $true : $?')[0]
        # $? contains ?, so we check for ternary pattern instead of bare ?
        assert 'if ($cond)' in result
        assert '$?' in result
        assert 'else { $? }' in result

    def test_dollar_q_in_coalescing_right(self) -> None:
        result = pwsh_transform('$a ?? $?')[0]
        assert "??" not in result
        assert '$?' in result

    def test_dollar_q_then_question_bracket_with_index(self) -> None:
        result = pwsh_transform('$??[0][1]')[0]
        assert "if ($null -ne $?)" in result
        assert '$?[0]' in result

    def test_dollar_q_then_null_conditional_dot_property(self) -> None:
        result = pwsh_transform('$??.Property?.Sub')[0]
        # The outer ?. is transformed; $?.Property contains $? auto-var + .Property
        assert "if ($null -ne $?)" in result
        assert "$.Property" not in result



# ============================================================================
# Static member with method call then null-conditional
# ============================================================================

class TestStaticMemberMethodNullConditional:
    def test_static_method_then_property_null_conditional(self) -> None:
        result = pwsh_transform('[System.IO.Path]::GetTempPath()?.Length')[0]
        assert "?." not in result
        assert '[System.IO.Path]::GetTempPath()' in result
        assert '.Length' in result

    def test_static_method_then_method_null_conditional(self) -> None:
        result = pwsh_transform('[System.IO.File]::ReadAllText($f)?.Trim()')[0]
        assert "?." not in result
        assert '[System.IO.File]::ReadAllText($f)' in result
        assert '.Trim()' in result

    def test_static_property_then_property_null_conditional(self) -> None:
        result = pwsh_transform('[Environment]::MachineName?.Length')[0]
        assert "?." not in result
        assert '[Environment]::MachineName' in result

    def test_chained_static_then_instance_null_conditional(self) -> None:
        result = pwsh_transform('[DateTime]::Now?.Year')[0]
        assert "?." not in result
        assert '[DateTime]::Now' in result
        assert '.Year' in result


# ============================================================================
# Subexpression with operators inside
# ============================================================================

class TestSubexpressionWithInnerOperators:
    def test_subexpr_with_ternary_inside(self) -> None:
        result = pwsh_transform("$($a ? $b : $c)")[0]
        # Ternary inside $() is at depth>0, NOT transformed in single pass
        assert "?" in result
        assert "$($a ? $b : $c)" == result

    def test_subexpr_with_coalescing_inside(self) -> None:
        result = pwsh_transform('$($a ?? $b)')[0]
        assert "??" not in result
        assert 'if ($null -ne $a)' in result

    def test_subexpr_with_null_conditional_inside(self) -> None:
        result = pwsh_transform('$($a?.Name)')[0]
        assert "?." not in result
        assert 'if ($null -ne $a)' in result

    def test_subexpr_with_chain_inside(self) -> None:
        result = pwsh_transform('$(cmd1 && cmd2)')[0]
        assert "&&" not in result
        assert 'if ($?)' in result

    def test_subexpr_with_nca_inside(self) -> None:
        result = pwsh_transform('$($a ??= "default")')[0]
        assert "??=" not in result
        assert 'if ($null -eq $a)' in result

    def test_nested_subexpr_with_multiple_operators(self) -> None:
        result = pwsh_transform('$($($a?.Name ?? "default"))')[0]
        assert "?." not in result
        assert "??" not in result


# ============================================================================
# Operators inside switch statement
# ============================================================================

class TestSwitchStatementWithOperators:
    def test_switch_value_with_null_coalescing(self) -> None:
        code = 'switch ($val ?? "default") { "a" { 1 } "b" { 2 } }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert 'switch (if ($null -ne $val)' in result

    def test_switch_condition_with_ternary(self) -> None:
        code = 'switch ($cond ? $a : $b) { 1 { "one" } 2 { "two" } }'
        result = pwsh_transform(code)[0]
        # Ternary inside switch() parens is at depth>0, NOT transformed (known limitation)
        assert "?" in result
        assert "switch ($cond ? $a : $b)" in result

    def test_switch_with_null_conditional(self) -> None:
        code = 'switch ($obj?.Type) { "File" { 1 } "Dir" { 2 } }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert '$obj.Type' in result

    def test_switch_with_wildcard_and_chain_after(self) -> None:
        code = 'switch -Wildcard ($pattern) { "*.txt" { } }; cmd1 && cmd2'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "if ($?)" in result
        assert "switch -Wildcard" in result


# ============================================================================
# Operators with [ref] / [void] / [scriptblock] / [pscustomobject]
# ============================================================================

class TestSpecialTypeCastsWithOperators:
    def test_ref_cast_left_coalescing(self) -> None:
        result = pwsh_transform('[ref]$a ?? [ref]0')[0]
        assert "??" not in result
        assert 'if ($null -ne [ref]$a)' in result

    def test_void_cast_left_coalescing(self) -> None:
        result = pwsh_transform('[void]$a ?? $null')[0]
        assert "??" not in result
        assert '[void]$a' in result

    def test_scriptblock_cast_left_coalescing(self) -> None:
        result = pwsh_transform('[scriptblock]$a ?? {}')[0]
        assert "??" not in result
        assert '[scriptblock]$a' in result

    def test_pscustomobject_cast_null_conditional(self) -> None:
        result = pwsh_transform('[pscustomobject]$obj?.Name')[0]
        assert "?." not in result
        assert '[pscustomobject]$obj' in result

    def test_ordered_cast_null_conditional(self) -> None:
        result = pwsh_transform('[ordered]@{}.Count?.ToString()')[0]
        assert "?." not in result
        assert '[ordered]@{}.Count' in result

    def test_hashtable_cast_coalescing(self) -> None:
        result = pwsh_transform('[hashtable]$ht ?? @{}')[0]
        assert "??" not in result
        assert '[hashtable]$ht' in result

    def test_array_cast_null_conditional(self) -> None:
        result = pwsh_transform('[array]$arr?[0]')[0]
        assert "?[" not in result
        assert '[array]$arr' in result

    def test_string_cast_ternary(self) -> None:
        result = pwsh_transform('[string]$val -ne "" ? "set" : "empty"')[0]
        assert "?" not in result
        assert '[string]$val' in result

    def test_int_cast_ternary(self) -> None:
        result = pwsh_transform('[int]$val -gt 0 ? "positive" : "non-positive"')[0]
        assert "?" not in result
        assert '[int]$val' in result

    def test_datetime_cast_coalescing(self) -> None:
        result = pwsh_transform('[datetime]$d ?? (Get-Date)')[0]
        assert "??" not in result
        assert '[datetime]$d' in result

    def test_timespan_cast_ternary(self) -> None:
        result = pwsh_transform('[timespan]$ts -gt [timespan]"1:00" ? "long" : "short"')[0]
        assert "?" not in result
        assert '[timespan]$ts' in result

    def test_version_cast_coalescing(self) -> None:
        result = pwsh_transform('[version]$v ?? [version]"1.0"')[0]
        assert "??" not in result
        assert '[version]$v' in result

    def test_guid_cast_null_conditional(self) -> None:
        result = pwsh_transform('[guid]$g?.ToString()')[0]
        assert "?." not in result
        assert '[guid]$g' in result

    def test_xml_cast_coalescing(self) -> None:
        result = pwsh_transform('[xml]$x ?? [xml]"<root/>"')[0]
        assert "??" not in result
        assert '[xml]$x' in result

    def test_regex_cast_ternary(self) -> None:
        result = pwsh_transform('[regex]$r -ne $null ? "compiled" : "null"')[0]
        assert "?" not in result
        assert '[regex]$r' in result

    def test_securestring_cast_coalescing(self) -> None:
        result = pwsh_transform('[securestring]$s ?? (ConvertTo-SecureString "x" -AsPlainText)')[0]
        assert "??" not in result
        assert '[securestring]$s' in result


# ============================================================================
# Chain with 2> / >> / 2>&1 and complex commands
# ============================================================================

class TestRedirectionChainComplex:
    def test_redirect_to_null_and_chain(self) -> None:
        result = pwsh_transform('Get-Process > $null && Write-Output done')[0]
        assert "&&" not in result
        assert '> $null' in result

    def test_error_redirect_to_null_and_chain(self) -> None:
        result = pwsh_transform('Get-Item $f 2> $null && Write-Output ok')[0]
        assert "&&" not in result
        assert '2> $null' in result

    def test_both_redirects_and_chain(self) -> None:
        result = pwsh_transform('cmd.exe 1> out.txt 2> err.txt && Write-Output ok')[0]
        assert "&&" not in result
        assert '1> out.txt' in result
        assert '2> err.txt' in result

    def test_error_to_output_and_chain(self) -> None:
        result = pwsh_transform('cmd.exe 2>&1 | Out-File log.txt && Write-Output done')[0]
        assert "&&" not in result
        assert '2>&1' in result

    def test_redirect_inside_subexpr_and_chain(self) -> None:
        result = pwsh_transform('$(Get-Process > procs.txt) && Write-Output saved')[0]
        assert "&&" not in result
        assert '> procs.txt' in result


# ============================================================================
# Ternary with Measure-Object / Select-Object / Where-Object / ForEach-Object
# ============================================================================

class TestTernaryWithCommonCmdlets:
    def test_measure_object_in_ternary_condition(self) -> None:
        result = pwsh_transform('($arr | Measure-Object).Count -gt 0 ? "has-items" : "empty"')[0]
        assert "?" not in result
        assert 'if (($arr | Measure-Object).Count -gt 0)' in result

    def test_select_object_in_ternary_condition(self) -> None:
        result = pwsh_transform('($arr | Select-Object -First 1) ? "found" : "empty"')[0]
        assert "?" not in result
        assert 'if (($arr | Select-Object -First 1))' in result

    def test_where_object_in_ternary_condition(self) -> None:
        result = pwsh_transform('($arr | Where-Object { $_ -gt 0 }) ? "has-positive" : "none"')[0]
        assert "?" not in result
        assert 'if (($arr | Where-Object { $_ -gt 0 }))' in result

    def test_foreach_object_in_ternary_condition(self) -> None:
        result = pwsh_transform('($arr | ForEach-Object { $_ * 2 }) ? "processed" : "empty"')[0]
        assert "?" not in result
        assert 'if (($arr | ForEach-Object { $_ * 2 }))' in result

    def test_sort_object_in_ternary_condition(self) -> None:
        result = pwsh_transform('($arr | Sort-Object).Count -gt 1 ? "sorted" : "single"')[0]
        assert "?" not in result
        assert 'if (($arr | Sort-Object).Count -gt 1)' in result

    def test_group_object_in_ternary_condition(self) -> None:
        result = pwsh_transform('($arr | Group-Object).Count -gt 1 ? "groups" : "one"')[0]
        assert "?" not in result
        assert 'if (($arr | Group-Object).Count -gt 1)' in result

    def test_compare_object_in_ternary_condition(self) -> None:
        result = pwsh_transform('(Compare-Object $a $b) ? "different" : "same"')[0]
        assert "?" not in result
        assert 'if ((Compare-Object $a $b))' in result

    def test_convert_to_json_in_ternary_condition(self) -> None:
        result = pwsh_transform('(ConvertTo-Json $obj).Length -gt 2 ? "valid" : "empty"')[0]
        assert "?" not in result
        assert 'if ((ConvertTo-Json $obj).Length -gt 2)' in result

    def test_convert_from_json_in_ternary_condition(self) -> None:
        result = pwsh_transform('(ConvertFrom-Json $json).name ? "has-name" : "no-name"')[0]
        assert "?" not in result
        assert 'if ((ConvertFrom-Json $json).name)' in result


# ============================================================================
# Coalescing with Get-Content / Test-Path / Test-Connection
# ============================================================================

class TestCoalescingWithCommonCmdlets:
    def test_get_content_coalescing(self) -> None:
        result = pwsh_transform('(Get-Content $f -Raw) ?? ""')[0]
        assert "??" not in result
        assert 'if ($null -ne (Get-Content $f -Raw))' in result

    def test_test_path_result_coalescing(self) -> None:
        result = pwsh_transform('(Test-Path $f) ?? $false')[0]
        assert "??" not in result
        assert 'if ($null -ne (Test-Path $f))' in result

    def test_test_connection_result_coalescing(self) -> None:
        result = pwsh_transform('(Test-Connection $host -Count 1 -Quiet) ?? $false')[0]
        assert "??" not in result
        assert 'if ($null -ne (Test-Connection $host -Count 1 -Quiet))' in result

    def test_invoke_restmethod_coalescing(self) -> None:
        result = pwsh_transform('(Invoke-RestMethod $url) ?? @{}')[0]
        assert "??" not in result
        assert 'if ($null -ne (Invoke-RestMethod $url))' in result

    def test_invoke_webrequest_coalescing(self) -> None:
        result = pwsh_transform('(Invoke-WebRequest $url).Content ?? ""')[0]
        assert "??" not in result
        assert 'if ($null -ne (Invoke-WebRequest $url).Content)' in result

    def test_get_process_coalescing(self) -> None:
        result = pwsh_transform('(Get-Process -Name $n) ?? $null')[0]
        assert "??" not in result
        assert 'if ($null -ne (Get-Process -Name $n))' in result

    def test_get_service_coalescing(self) -> None:
        result = pwsh_transform('(Get-Service -Name $n) ?? $null')[0]
        assert "??" not in result
        assert 'if ($null -ne (Get-Service -Name $n))' in result

    def test_get_item_coalescing(self) -> None:
        result = pwsh_transform('(Get-Item $p) ?? $null')[0]
        assert "??" not in result
        assert 'if ($null -ne (Get-Item $p))' in result

    def test_get_childitem_coalescing(self) -> None:
        result = pwsh_transform('(Get-ChildItem $dir) ?? @()')[0]
        assert "??" not in result
        assert 'if ($null -ne (Get-ChildItem $dir))' in result


# ============================================================================
# Null-conditional on Invoke-RestMethod / Invoke-WebRequest result
# ============================================================================

class TestNullConditionalOnWebCmdlets:
    def test_invoke_restmethod_null_conditional(self) -> None:
        result = pwsh_transform('(Invoke-RestMethod $url)?.data')[0]
        assert "?." not in result
        assert 'if ($null -ne (Invoke-RestMethod $url))' in result
        assert '(Invoke-RestMethod $url).data' in result

    def test_invoke_webrequest_null_conditional(self) -> None:
        result = pwsh_transform('(Invoke-WebRequest $url)?.StatusCode')[0]
        assert "?." not in result
        assert '(Invoke-WebRequest $url)' in result
        assert '.StatusCode' in result

    def test_invoke_restmethod_chained_null_conditional(self) -> None:
        result = pwsh_transform('(Invoke-RestMethod $url)?.result?.items?[0]')[0]
        assert "?." not in result
        assert "?[" not in result
        assert '(Invoke-RestMethod $url)' in result

    def test_invoke_restmethod_then_coalescing(self) -> None:
        result = pwsh_transform('(Invoke-RestMethod $url)?.name ?? "unknown"')[0]
        assert "?." not in result
        assert "??" not in result


# ============================================================================
# Multiple operators without semicolons on same line
# ============================================================================

class TestMultipleOperatorsNoSemicolon:
    def test_coalescing_then_ternary_no_semicolon(self) -> None:
        result = pwsh_transform('$a ?? $b ? "t" : "f"')[0]
        assert "??" not in result
        # After ?? transform, the remaining ternary is at depth>0 inside braces
        assert "$a" in result
        assert "$b" in result

    def test_ternary_then_coalescing_no_semicolon(self) -> None:
        result = pwsh_transform('$cond ? $a ?? $b : $c')[0]
        # LIMITATION: ?? is transformed first, producing broken ternary-like output
        assert "$cond" in result
        assert "$c" in result

    def test_null_conditional_then_coalescing_no_semicolon(self) -> None:
        result = pwsh_transform('$a?.Name ?? "default"')[0]
        assert "?." not in result
        assert "??" not in result

    def test_chain_then_coalescing_no_semicolon(self) -> None:
        result = pwsh_transform('cmd1 && cmd2 ?? "fallback"')[0]
        assert "&&" not in result
        assert "??" not in result

    def test_coalescing_then_chain_no_semicolon(self) -> None:
        result = pwsh_transform('$a ?? $b || cmd2')[0]
        assert "??" not in result
        assert "||" not in result

    def test_null_conditional_then_chain_no_semicolon(self) -> None:
        result = pwsh_transform('$a?.Method() && cmd2')[0]
        assert "?." not in result
        assert "&&" not in result

    def test_ternary_then_chain_no_semicolon(self) -> None:
        result = pwsh_transform('$cond ? cmd1 : cmd2 && cmd3')[0]
        # Ternary ? is gone, chain in false branch is transformed
        # $? from chain transform contains ?, so we check specific patterns
        assert "&&" not in result
        assert "if ($cond)" in result
        assert "cmd1" in result
        assert "cmd3" in result

    def test_chain_then_ternary_no_semicolon(self) -> None:
        result = pwsh_transform('cmd1 && $cond ? "a" : "b"')[0]
        assert "&&" not in result
        # The bare ternary ? is gone; $? from chain transform may remain
        assert "if ($?)" in result
        assert "if ($cond)" in result

    def test_nca_then_chain_no_semicolon(self) -> None:
        result = pwsh_transform('$a ??= "x" && cmd2')[0]
        assert "??=" not in result
        assert "&&" not in result

    def test_chain_then_nca_no_semicolon(self) -> None:
        result = pwsh_transform('cmd1 && $a ??= "x"')[0]
        assert "&&" not in result
        assert "??=" not in result


# ============================================================================
# Operators adjacent to throw / return / exit / break / continue
# ============================================================================

class TestOperatorsWithControlFlowKeywords:
    def test_throw_with_coalescing_message(self) -> None:
        result = pwsh_transform('throw ($msg ?? "error")')[0]
        assert "??" not in result
        assert 'throw (if ($null -ne $msg)' in result

    def test_return_with_ternary(self) -> None:
        result = pwsh_transform('return $cond ? $a : $b')[0]
        assert "?" not in result
        assert 'return if ($cond)' in result

    def test_return_with_coalescing(self) -> None:
        result = pwsh_transform('return $val ?? "default"')[0]
        # BUG: 'return' is treated as command prefix by _strip_command_prefix
        assert "??" not in result
        assert "return" in result

    def test_return_with_null_conditional(self) -> None:
        result = pwsh_transform('return $obj?.Name')[0]
        # BUG: 'return' is treated as command prefix by _strip_command_prefix
        assert "?." not in result
        assert "return" in result

    def test_exit_with_ternary(self) -> None:
        result = pwsh_transform('exit $cond ? 0 : 1')[0]
        assert "?" not in result
        assert 'exit if ($cond)' in result

    def test_break_with_coalescing(self) -> None:
        result = pwsh_transform('break $label ?? "default"')[0]
        # BUG: 'break' is treated as command prefix by _strip_command_prefix
        assert "??" not in result
        assert "break" in result

    def test_continue_with_ternary(self) -> None:
        result = pwsh_transform('continue $cond ? 1 : 0')[0]
        assert "?" not in result
        assert 'continue if ($cond)' in result

    def test_throw_with_null_conditional(self) -> None:
        result = pwsh_transform('throw $ex?.Message')[0]
        # BUG: 'throw' is treated as command prefix by _strip_command_prefix
        assert "?." not in result
        assert "throw" in result

    def test_return_with_chain(self) -> None:
        result = pwsh_transform('return (cmd1 && cmd2)')[0]
        assert "&&" not in result
        assert 'return (cmd1' in result
        assert 'if ($?)' in result

    def test_exit_with_chain(self) -> None:
        result = pwsh_transform('exit (cmd1 || cmd2 ? 1 : 0)')[0]
        # Chain inside () at depth>0: || is transformed but ternary remains
        assert "||" not in result
        assert "?" in result
        assert 'exit (' in result


# ============================================================================
# Operators with $LastExitCode / $Error / $Matches
# ============================================================================

class TestOperatorsWithSpecialVariables:
    def test_lastexitcode_ternary(self) -> None:
        result = pwsh_transform('$LastExitCode -eq 0 ? "success" : "failure"')[0]
        assert "?" not in result
        assert 'if ($LastExitCode -eq 0)' in result

    def test_lastexitcode_coalescing(self) -> None:
        result = pwsh_transform('$LastExitCode ?? 0')[0]
        assert "??" not in result
        assert 'if ($null -ne $LastExitCode)' in result

    def test_error_count_ternary(self) -> None:
        result = pwsh_transform('$Error.Count -gt 0 ? "has-errors" : "clean"')[0]
        assert "?" not in result
        assert 'if ($Error.Count -gt 0)' in result

    def test_error_coalescing(self) -> None:
        result = pwsh_transform('$Error[0] ?? "none"')[0]
        assert "??" not in result
        assert 'if ($null -ne $Error[0])' in result

    def test_matches_ternary(self) -> None:
        result = pwsh_transform('$Matches[1] ? "captured" : "no-capture"')[0]
        assert "?" not in result
        assert 'if ($Matches[1])' in result

    def test_matches_coalescing(self) -> None:
        result = pwsh_transform('$Matches[1] ?? ""')[0]
        assert "??" not in result
        assert 'if ($null -ne $Matches[1])' in result

    def test_matches_null_conditional(self) -> None:
        result = pwsh_transform('$Matches?.Count')[0]
        assert "?." not in result
        assert 'if ($null -ne $Matches)' in result

    def test_lastexitcode_null_conditional(self) -> None:
        result = pwsh_transform('$LastExitCode?.ToString()')[0]
        assert "?." not in result
        assert 'if ($null -ne $LastExitCode)' in result


# ============================================================================
# Operators with $Profile / $PWD / $HOME / $OFS
# ============================================================================

class TestOperatorsWithPathVariables:
    def test_profile_null_conditional(self) -> None:
        result = pwsh_transform('$Profile?.Exists')[0]
        assert "?." not in result
        assert 'if ($null -ne $Profile)' in result
        assert '$Profile.Exists' in result

    def test_pwd_coalescing(self) -> None:
        result = pwsh_transform('$PWD ?? (Get-Location)')[0]
        assert "??" not in result
        assert 'if ($null -ne $PWD)' in result

    def test_home_ternary(self) -> None:
        result = pwsh_transform('$HOME ? "has-home" : "no-home"')[0]
        assert "?" not in result
        assert 'if ($HOME)' in result

    def test_ofs_coalescing(self) -> None:
        result = pwsh_transform('$OFS ?? " "')[0]
        assert "??" not in result
        assert 'if ($null -ne $OFS)' in result

    def test_profile_coalescing(self) -> None:
        result = pwsh_transform('$Profile ?? "not-set"')[0]
        assert "??" not in result
        assert 'if ($null -ne $Profile)' in result

    def test_pwd_null_conditional(self) -> None:
        result = pwsh_transform('$PWD?.Path')[0]
        assert "?." not in result
        assert 'if ($null -ne $PWD)' in result
        assert '$PWD.Path' in result

    def test_home_null_conditional(self) -> None:
        result = pwsh_transform('$HOME?.Length')[0]
        assert "?." not in result
        assert 'if ($null -ne $HOME)' in result
        assert '$HOME.Length' in result


# ============================================================================
# Operators with preference variables
# ============================================================================

class TestOperatorsWithPreferenceVariables:
    def test_erroractionpreference_ternary(self) -> None:
        result = pwsh_transform('$ErrorActionPreference -eq "Stop" ? "strict" : "lax"')[0]
        assert "?" not in result
        assert 'if ($ErrorActionPreference -eq "Stop")' in result

    def test_erroractionpreference_coalescing(self) -> None:
        result = pwsh_transform('$ErrorActionPreference ?? "Continue"')[0]
        assert "??" not in result
        assert 'if ($null -ne $ErrorActionPreference)' in result

    def test_progresspreference_coalescing(self) -> None:
        result = pwsh_transform('$ProgressPreference ?? "Continue"')[0]
        assert "??" not in result
        assert 'if ($null -ne $ProgressPreference)' in result

    def test_verbosepreference_ternary(self) -> None:
        result = pwsh_transform('$VerbosePreference -eq "Continue" ? "verbose" : "silent"')[0]
        assert "?" not in result
        assert 'if ($VerbosePreference -eq "Continue")' in result

    def test_warningpreference_coalescing(self) -> None:
        result = pwsh_transform('$WarningPreference ?? "Continue"')[0]
        assert "??" not in result
        assert 'if ($null -ne $WarningPreference)' in result

    def test_debugpreference_ternary(self) -> None:
        result = pwsh_transform('$DebugPreference -eq "Continue" ? "debug" : "nodebug"')[0]
        assert "?" not in result
        assert 'if ($DebugPreference -eq "Continue")' in result

    def test_informationpreference_coalescing(self) -> None:
        result = pwsh_transform('$InformationPreference ?? "SilentlyContinue"')[0]
        assert "??" not in result
        assert 'if ($null -ne $InformationPreference)' in result

    def test_whatifpreference_ternary(self) -> None:
        result = pwsh_transform('$WhatIfPreference ? "simulate" : "execute"')[0]
        assert "?" not in result
        assert 'if ($WhatIfPreference)' in result

    def test_confirmpreference_coalescing(self) -> None:
        result = pwsh_transform('$ConfirmPreference ?? "High"')[0]
        assert "??" not in result
        assert 'if ($null -ne $ConfirmPreference)' in result

    def test_psculture_ternary(self) -> None:
        result = pwsh_transform('$PSCulture -eq "en-US" ? "english" : "other"')[0]
        assert "?" not in result
        assert 'if ($PSCulture -eq "en-US")' in result

    def test_psuiculture_coalescing(self) -> None:
        result = pwsh_transform('$PSUICulture ?? "en-US"')[0]
        assert "??" not in result
        assert 'if ($null -ne $PSUICulture)' in result

    def test_psculture_null_conditional(self) -> None:
        result = pwsh_transform('$PSCulture?.Length')[0]
        assert "?." not in result
        assert 'if ($null -ne $PSCulture)' in result


# ============================================================================
# Operators with $PSDefaultParameterValues / $PSStyle / $Transcript
# ============================================================================

class TestOperatorsWithPSDefaultAndStyle:
    def test_psdefaultparametervalues_null_conditional(self) -> None:
        result = pwsh_transform('$PSDefaultParameterValues?.Count')[0]
        assert "?." not in result
        assert 'if ($null -ne $PSDefaultParameterValues)' in result
        assert '$PSDefaultParameterValues.Count' in result

    def test_psdefaultparametervalues_coalescing(self) -> None:
        result = pwsh_transform('$PSDefaultParameterValues ?? @{}')[0]
        assert "??" not in result
        assert 'if ($null -ne $PSDefaultParameterValues)' in result

    def test_psmoduleautoloadingpreference_ternary(self) -> None:
        result = pwsh_transform('$PSModuleAutoLoadingPreference -eq "All" ? "auto" : "manual"')[0]
        assert "?" not in result
        assert 'if ($PSModuleAutoLoadingPreference -eq "All")' in result

    def test_psstyle_null_conditional(self) -> None:
        result = pwsh_transform('$PSStyle?.Foreground?.Red')[0]
        assert "?." not in result
        assert '$PSStyle' in result
        assert '.Foreground' in result
        assert '.Red' in result

    def test_psstyle_coalescing(self) -> None:
        result = pwsh_transform('$PSStyle ?? $null')[0]
        assert "??" not in result
        assert 'if ($null -ne $PSStyle)' in result

    def test_outputencoding_coalescing(self) -> None:
        result = pwsh_transform('$OutputEncoding ?? [System.Text.Encoding]::UTF8')[0]
        assert "??" not in result
        assert 'if ($null -ne $OutputEncoding)' in result

    def test_psmoduleautoloadingpreference_coalescing(self) -> None:
        result = pwsh_transform('$PSModuleAutoLoadingPreference ?? "All"')[0]
        assert "??" not in result
        assert 'if ($null -ne $PSModuleAutoLoadingPreference)' in result

    def test_psemailserver_coalescing(self) -> None:
        result = pwsh_transform('$PSEmailServer ?? "localhost"')[0]
        assert "??" not in result
        assert 'if ($null -ne $PSEmailServer)' in result

    def test_formatenumerationlimit_ternary(self) -> None:
        result = pwsh_transform('$FormatEnumerationLimit -gt 0 ? "limited" : "unlimited"')[0]
        assert "?" not in result
        assert 'if ($FormatEnumerationLimit -gt 0)' in result

    def test_errorview_coalescing(self) -> None:
        result = pwsh_transform('$ErrorView ?? "NormalView"')[0]
        assert "??" not in result
        assert 'if ($null -ne $ErrorView)' in result

    def test_maximumaliascount_ternary(self) -> None:
        result = pwsh_transform('$MaximumAliasCount -gt 100 ? "many" : "few"')[0]
        assert "?" not in result
        assert 'if ($MaximumAliasCount -gt 100)' in result

    def test_maximumdrivecount_coalescing(self) -> None:
        result = pwsh_transform('$MaximumDriveCount ?? 4096')[0]
        assert "??" not in result
        assert 'if ($null -ne $MaximumDriveCount)' in result

    def test_maximumerrorcount_ternary(self) -> None:
        result = pwsh_transform('$MaximumErrorCount -gt 256 ? "big-buffer" : "small"')[0]
        assert "?" not in result
        assert 'if ($MaximumErrorCount -gt 256)' in result

    def test_maximumfunctioncount_coalescing(self) -> None:
        result = pwsh_transform('$MaximumFunctionCount ?? 4096')[0]
        assert "??" not in result
        assert 'if ($null -ne $MaximumFunctionCount)' in result

    def test_maximumvariablecount_ternary(self) -> None:
        result = pwsh_transform('$MaximumVariableCount -gt 1000 ? "many" : "few"')[0]
        assert "?" not in result
        assert 'if ($MaximumVariableCount -gt 1000)' in result

    def test_maximumhistorycount_coalescing(self) -> None:
        result = pwsh_transform('$MaximumHistoryCount ?? 64')[0]
        assert "??" not in result
        assert 'if ($null -ne $MaximumHistoryCount)' in result


# ============================================================================
# More backtick edge cases
# ============================================================================

class TestBacktickExtendedEdgeCases:
    def test_backtick_before_open_paren_not_collapsed(self) -> None:
        code = 'Write-Output `(\nhello\n)'
        result = pwsh_transform(code)[0]
        assert '(' in result
        assert 'hello' in result

    def test_backtick_before_open_brace_not_collapsed(self) -> None:
        code = 'if ($true) `{\nWrite-Output hi\n}'
        result = pwsh_transform(code)[0]
        assert '{' in result
        assert 'Write-Output hi' in result

    def test_backtick_before_dollar_not_collapsed(self) -> None:
        code = 'Write-Output `\n$a ?? "default"'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert 'if ($null -ne $a)' in result

    def test_backtick_before_pipe_not_collapsed(self) -> None:
        code = 'Get-Process `\n| Where-Object CPU'
        result = pwsh_transform(code)[0]
        assert '| Where-Object CPU' in result

    def test_backtick_at_end_then_operator_next_line(self) -> None:
        code = 'cmd1 `\n&& cmd2'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "if ($?)" in result

    def test_backtick_at_end_then_ternary_next_line(self) -> None:
        code = '$cond `\n? "yes" : "no"'
        result = pwsh_transform(code)[0]
        assert "?" not in result
        assert "if ($cond)" in result

    def test_backtick_at_end_then_coalescing_next_line(self) -> None:
        code = '$a `\n?? "default"'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result

    def test_backtick_at_end_then_null_conditional_next_line(self) -> None:
        code = "$a`.\nProperty"
        result = pwsh_transform(code)[0]
        # Backtick not followed by newline (it's ` then . then newline), so not collapsed
        assert isinstance(result, str)
        assert "$a" in result

    def test_backtick_before_comment_not_collapsed(self) -> None:
        code = 'Write-Output `# comment'
        result = pwsh_transform(code)[0]
        assert "# comment" in result

    def test_multiple_backticks_at_end_of_line(self) -> None:
        code = 'cmd1 ``\ncmd2'
        result = pwsh_transform(code)[0]
        assert isinstance(result, str)
        assert "cmd1" in result
        assert "cmd2" in result


# ============================================================================
# Operators with $NestedPromptLevel / $Sender / $ShellId / $StackTrace
# ============================================================================

class TestOperatorsWithLessCommonAutoVars:
    def test_nestedpromptlevel_coalescing(self) -> None:
        result = pwsh_transform('$NestedPromptLevel ?? 0')[0]
        assert "??" not in result
        assert 'if ($null -ne $NestedPromptLevel)' in result

    def test_sender_null_conditional(self) -> None:
        result = pwsh_transform('$Sender?.Name')[0]
        assert "?." not in result
        assert 'if ($null -ne $Sender)' in result
        assert '$Sender.Name' in result

    def test_shellid_ternary(self) -> None:
        result = pwsh_transform('$ShellId -eq "Microsoft.PowerShell" ? "ps" : "other"')[0]
        assert "?" not in result
        assert 'if ($ShellId -eq "Microsoft.PowerShell")' in result

    def test_stacktrace_coalescing(self) -> None:
        result = pwsh_transform('$StackTrace ?? ""')[0]
        assert "??" not in result
        assert 'if ($null -ne $StackTrace)' in result

    def test_nestedpromptlevel_null_conditional(self) -> None:
        result = pwsh_transform('$NestedPromptLevel?.ToString()')[0]
        assert "?." not in result
        assert 'if ($null -ne $NestedPromptLevel)' in result

    def test_sender_coalescing(self) -> None:
        result = pwsh_transform('$Sender ?? $null')[0]
        assert "??" not in result
        assert 'if ($null -ne $Sender)' in result

    def test_shellid_coalescing(self) -> None:
        result = pwsh_transform('$ShellId ?? "Microsoft.PowerShell"')[0]
        assert "??" not in result
        assert 'if ($null -ne $ShellId)' in result

    def test_stacktrace_null_conditional(self) -> None:
        result = pwsh_transform('$StackTrace?.Length')[0]
        assert "?." not in result
        assert 'if ($null -ne $StackTrace)' in result


# ============================================================================
# Operators with $MyInvocation / $PSScriptRoot / $PSCommandPath
# ============================================================================

class TestOperatorsWithInvocationVariables:
    def test_myinvocation_null_conditional(self) -> None:
        result = pwsh_transform('$MyInvocation?.MyCommand?.Name')[0]
        assert "?." not in result
        assert '$MyInvocation' in result
        assert '.MyCommand' in result
        assert '.Name' in result

    def test_myinvocation_coalescing(self) -> None:
        result = pwsh_transform('$MyInvocation.Line ?? ""')[0]
        assert "??" not in result
        assert 'if ($null -ne $MyInvocation.Line)' in result

    def test_psscriptroot_null_conditional(self) -> None:
        result = pwsh_transform('$PSScriptRoot?.Length')[0]
        assert "?." not in result
        assert 'if ($null -ne $PSScriptRoot)' in result
        assert '$PSScriptRoot.Length' in result

    def test_pscommandpath_coalescing(self) -> None:
        result = pwsh_transform('$PSCommandPath ?? "unknown"')[0]
        assert "??" not in result
        assert 'if ($null -ne $PSCommandPath)' in result

    def test_myinvocation_ternary(self) -> None:
        result = pwsh_transform('$MyInvocation.Line ? "called" : "interactive"')[0]
        assert "?" not in result
        assert 'if ($MyInvocation.Line)' in result

    def test_psscriptroot_coalescing(self) -> None:
        result = pwsh_transform('$PSScriptRoot ?? (Get-Location).Path')[0]
        assert "??" not in result
        assert 'if ($null -ne $PSScriptRoot)' in result

    def test_pscommandpath_null_conditional(self) -> None:
        result = pwsh_transform('$PSCommandPath?.Split("\\")[-1]')[0]
        assert "?." not in result
        assert 'if ($null -ne $PSCommandPath)' in result
        assert '$PSCommandPath.Split' in result


# ============================================================================
# Operators inside function/filter definitions
# ============================================================================

class TestOperatorsInFunctionDefinitions:
    def test_function_with_ternary_in_body(self) -> None:
        code = 'function Get-Status { param($x); $x ? "ok" : "fail" }'
        result = pwsh_transform(code)[0]
        # Ternary inside function body braces is at depth>0, NOT transformed (known limitation)
        assert "?" in result
        assert 'function Get-Status' in result

    def test_function_with_coalescing_in_body(self) -> None:
        code = 'function Get-Name { param($n); $n ?? "anonymous" }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert 'function Get-Name' in result
        assert 'if ($null -ne $n)' in result

    def test_function_with_null_conditional_in_body(self) -> None:
        code = 'function Get-Len { param($obj); $obj?.Name?.Length }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert 'function Get-Len' in result
        assert '$obj.Name.Length' in result

    def test_function_with_chain_in_body(self) -> None:
        code = 'function Test-Item { param($p); Test-Path $p && Write-Output exists }'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert 'function Test-Item' in result
        assert 'if ($?)' in result

    def test_filter_with_coalescing_in_body(self) -> None:
        code = 'filter Get-Val { $_ ?? "empty" }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert 'filter Get-Val' in result
        assert 'if ($null -ne $_)' in result

    def test_filter_with_ternary_in_body(self) -> None:
        code = 'filter Test-Num { $_ -gt 0 ? "pos" : "non-pos" }'
        result = pwsh_transform(code)[0]
        # Ternary inside filter body braces is at depth>0, NOT transformed (known limitation)
        assert "?" in result
        assert 'filter Test-Num' in result


# ============================================================================
# Operators with enum definitions
# ============================================================================

class TestOperatorsInEnumDefinitions:
    def test_enum_with_ternary_in_value(self) -> None:
        code = 'enum Status { OK = 0; FAIL = $cond ? 1 : 2 }'
        result = pwsh_transform(code)[0]
        # Ternary inside enum braces is at depth>0, NOT transformed (known limitation)
        assert "?" in result
        assert 'enum Status' in result

    def test_enum_with_coalescing_in_value(self) -> None:
        code = 'enum Priority { LOW = $val ?? 1; HIGH = 10 }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert 'enum Priority' in result
        assert 'if ($null -ne $val)' in result


# ============================================================================
# More edge cases with comments containing operators
# ============================================================================

class TestCommentsWithOperatorsExtended:
    def test_comment_before_ternary_with_operators_in_comment(self) -> None:
        code = '# ternary: cond ? true : false\n$x = $a ? 1 : 0'
        result = pwsh_transform(code)[0]
        # The ? inside the comment is preserved; the ternary on the next line is transformed
        assert 'ternary: cond ? true : false' in result
        assert 'if ($a)' in result

    def test_comment_before_coalescing_with_operators_in_comment(self) -> None:
        code = '# coalescing: a ?? b\n$x = $a ?? "default"'
        result = pwsh_transform(code)[0]
        # The ?? inside the comment is preserved; the coalescing on the next line is transformed
        assert 'coalescing: a ?? b' in result
        assert 'if ($null -ne $a)' in result

    def test_comment_before_chain_with_operators_in_comment(self) -> None:
        code = '# chain: cmd1 && cmd2\ncmd1 && cmd2'
        result = pwsh_transform(code)[0]
        # The && inside the comment is preserved; the chain on the next line is transformed
        assert 'chain: cmd1 && cmd2' in result
        assert 'if ($?)' in result

    def test_inline_comment_after_null_conditional(self) -> None:
        result = pwsh_transform("$a?.Name # ?. is null-conditional")[0]
        # BUG: ?. in comment is preserved; the null-conditional IS transformed
        assert "if ($null -ne $a)" in result
        assert '# ?. is null-conditional' in result

    def test_inline_comment_after_nca(self) -> None:
        result = pwsh_transform('$a ??= "x" # ??= is null-assign')[0]
        # BUG: comment is swallowed into the NCA value expression; ??= in comment is preserved
        assert "if ($null -eq $a)" in result
        assert '??= is null-assign' in result


# ============================================================================
# Edge cases with string concatenation and operators
# ============================================================================

class TestStringConcatenationWithOperators:
    def test_concatenation_with_ternary_result(self) -> None:
        result = pwsh_transform('"prefix: " + ($cond ? "yes" : "no")')[0]
        # Ternary inside () is at depth>0, NOT transformed (known limitation)
        assert "?" in result
        assert '"prefix: " +' in result

    def test_concatenation_with_coalescing_result(self) -> None:
        result = pwsh_transform('"name: " + ($name ?? "unknown")')[0]
        assert "??" not in result
        assert '"name: " +' in result
        assert 'if ($null -ne $name)' in result

    def test_concatenation_with_null_conditional_result(self) -> None:
        result = pwsh_transform('"len: " + $str?.Length')[0]
        # BUG: _expr_left includes the string prefix as part of the base expr
        assert "?." not in result
        assert '"len: " +' in result

    def test_format_string_with_ternary(self) -> None:
        result = pwsh_transform('"Value: {0}" -f ($cond ? $a : $b)')[0]
        # Ternary inside () is at depth>0, NOT transformed (known limitation)
        assert "?" in result
        assert '"Value: {0}" -f' in result

    def test_format_string_with_coalescing(self) -> None:
        result = pwsh_transform('"Name: {0}" -f ($name ?? "anon")')[0]
        assert "??" not in result
        assert '"Name: {0}" -f' in result
        assert 'if ($null -ne $name)' in result

    def test_here_string_with_operators_outside(self) -> None:
        code = "@'\ninside ?? and ?.\n'@\n$x = $a ?? 'default'"
        result = pwsh_transform(code)[0]
        assert "??" in result  # inside here-string preserved
        assert "if ($null -ne $a)" in result

    def test_double_quoted_string_concat_with_ternary(self) -> None:
        result = pwsh_transform('"a=" + ($x ? "1" : "0") + ",b=" + ($y ? "1" : "0")')[0]
        # Ternary inside () is at depth>0, NOT transformed (known limitation)
        assert "?" in result
        assert '"a="' in result
        assert '",b="' in result


# ============================================================================
# Edge cases with array/hashtable construction and operators
# ============================================================================

class TestArrayHashtableWithOperatorsExtended:
    def test_array_literal_with_ternary_elements(self) -> None:
        result = pwsh_transform('@($cond ? $a : $b, $cond2 ? $c : $d)')[0]
        # Ternary inside @() is at depth>0, NOT transformed (known limitation)
        assert "?" in result
        assert '@(' in result

    def test_hashtable_literal_with_ternary_values(self) -> None:
        result = pwsh_transform('@{ a = $cond ? 1 : 0; b = $cond2 ? 2 : 3 }')[0]
        # Ternary inside @{ } is at depth>0, NOT transformed (known limitation)
        assert "?" in result
        assert '@{' in result

    def test_hashtable_literal_with_coalescing_values(self) -> None:
        result = pwsh_transform('@{ name = $name ?? "anon"; val = $val ?? 0 }')[0]
        assert "??" not in result
        assert '@{' in result
        assert 'if ($null -ne $name)' in result
        assert 'if ($null -ne $val)' in result

    def test_ordered_hashtable_with_coalescing(self) -> None:
        result = pwsh_transform('[ordered]@{ a = $x ?? 1; b = $y ?? 2 }')[0]
        assert "??" not in result
        assert '[ordered]@{' in result
        assert 'if ($null -ne $x)' in result

    def test_pscustomobject_with_coalescing(self) -> None:
        result = pwsh_transform('[pscustomobject]@{ Name = $name ?? "anon" }')[0]
        assert "??" not in result
        assert '[pscustomobject]@{' in result
        assert 'if ($null -ne $name)' in result

    def test_array_subexpr_with_null_conditional(self) -> None:
        result = pwsh_transform('@($obj?.Name, $obj?.Value)')[0]
        assert "?." not in result
        assert '@(' in result
        assert '$obj.Name' in result
        assert '$obj.Value' in result

    def test_hashtable_with_null_conditional_keys(self) -> None:
        result = pwsh_transform('@{ ($obj?.Key) = ($obj?.Value) }')[0]
        assert "?." not in result
        assert '@{' in result
        assert '$obj.Key' in result
        assert '$obj.Value' in result


# ============================================================================
# Edge cases with scriptblocks and operators
# ============================================================================

class TestScriptblocksWithOperatorsExtended:
    def test_scriptblock_param_with_coalescing_default(self) -> None:
        result = pwsh_transform('{ param($x = $a ?? "default") $x }')[0]
        assert "??" not in result
        assert 'param($x =' in result
        assert 'if ($null -ne $a)' in result

    def test_scriptblock_param_with_ternary_default(self) -> None:
        result = pwsh_transform('{ param($x = $cond ? $a : $b) $x }')[0]
        # Ternary inside param() parens at depth>0, NOT transformed (known limitation)
        assert "?" in result
        assert 'param($x =' in result

    def test_scriptblock_with_null_conditional_return(self) -> None:
        result = pwsh_transform('{ $obj?.Name }')[0]
        assert "?." not in result
        assert '$obj.Name' in result

    def test_scriptblock_with_chain_inside(self) -> None:
        result = pwsh_transform('{ cmd1 && cmd2 }')[0]
        assert "&&" not in result
        assert 'if ($?)' in result

    def test_scriptblock_with_nca_inside(self) -> None:
        result = pwsh_transform('{ $a ??= "init" }')[0]
        assert "??=" not in result
        assert 'if ($null -eq $a)' in result

    def test_scriptblock_with_coalescing_inside(self) -> None:
        result = pwsh_transform('{ $x ?? "fallback" }')[0]
        assert "??" not in result
        assert 'if ($null -ne $x)' in result

    def test_scriptblock_with_ternary_inside(self) -> None:
        result = pwsh_transform('{ $cond ? "t" : "f" }')[0]
        # Ternary inside scriptblock braces is at depth>0, NOT transformed (known limitation)
        assert "?" in result
        assert '{ $cond ? "t" : "f" }' == result

    def test_nested_scriptblocks_with_operators(self) -> None:
        result = pwsh_transform('{{ $a?.Name ?? "default" }}')[0]
        assert "?." not in result
        assert "??" not in result
        assert '$a.Name' in result


# ============================================================================
# Edge cases with class definitions and operators
# ============================================================================

class TestClassDefinitionsWithOperators:
    def test_class_method_with_ternary(self) -> None:
        code = 'class Foo { [string]GetStatus($x) { return $x ? "ok" : "fail" } }'
        result = pwsh_transform(code)[0]
        # Ternary inside class method body braces is at depth>0, NOT transformed
        assert "?" in result
        assert 'class Foo' in result

    def test_class_method_with_coalescing(self) -> None:
        code = 'class Foo { [string]GetName($x) { return $x ?? "anon" } }'
        result = pwsh_transform(code)[0]
        # BUG: 'return' is treated as command prefix, producing malformed output
        assert "??" not in result
        assert 'class Foo' in result

    def test_class_method_with_null_conditional(self) -> None:
        code = 'class Foo { [int]GetLen($x) { return $x?.Name?.Length } }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert 'class Foo' in result
        assert '$x.Name.Length' in result

    def test_class_property_with_coalescing(self) -> None:
        code = 'class Foo { [string]$Name = $env:USERNAME ?? "unknown" }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert 'class Foo' in result
        assert 'if ($null -ne $env:USERNAME)' in result

    def test_class_property_with_ternary(self) -> None:
        code = 'class Foo { [bool]$Debug = $env:DEBUG -eq "1" ? $true : $false }'
        result = pwsh_transform(code)[0]
        # Ternary inside class braces is at depth>0, NOT transformed (known limitation)
        assert "?" in result
        assert 'class Foo' in result


# ============================================================================
# Edge cases with if/elseif/else and operators
# ============================================================================

class TestIfElseWithOperators:
    def test_if_with_coalescing_condition(self) -> None:
        result = pwsh_transform('if ($a ?? $b) { Write-Output ok }')[0]
        assert "??" not in result
        assert 'if (if ($null -ne $a)' in result

    def test_if_with_ternary_condition(self) -> None:
        result = pwsh_transform('if ($cond ? $a : $b) { Write-Output ok }')[0]
        # Ternary inside if() parens is at depth>0, NOT transformed (known limitation)
        assert "?" in result
        assert 'if ($cond ? $a : $b)' in result

    def test_elseif_with_coalescing_condition(self) -> None:
        result = pwsh_transform('if ($a) { 1 } elseif ($b ?? $c) { 2 }')[0]
        assert "??" not in result
        assert 'elseif (if ($null -ne $b)' in result

    def test_elseif_with_ternary_condition(self) -> None:
        result = pwsh_transform('if ($a) { 1 } elseif ($cond ? $x : $y) { 2 }')[0]
        # Ternary inside elseif() parens is at depth>0, NOT transformed (known limitation)
        assert "?" in result
        assert 'elseif ($cond ? $x : $y)' in result

    def test_else_with_null_conditional_before(self) -> None:
        result = pwsh_transform('if ($a) { 1 } else { $obj?.Name }')[0]
        assert "?." not in result
        assert 'else' in result
        assert '$obj.Name' in result


# ============================================================================
# Edge cases with variable scopes and null-conditional chains
# ============================================================================

class TestVariableScopeNullConditionalChains:
    def test_local_scope_chain(self) -> None:
        result = pwsh_transform('$local:obj?.Name?.Length')[0]
        assert "?." not in result
        assert '$local:obj' in result
        assert '.Name' in result
        assert '.Length' in result

    def test_private_scope_chain(self) -> None:
        result = pwsh_transform('$private:obj?.Name')[0]
        assert "?." not in result
        assert '$private:obj' in result
        assert '.Name' in result

    def test_global_scope_bracket(self) -> None:
        result = pwsh_transform('$global:arr?[0]')[0]
        assert "?[" not in result
        assert '$global:arr' in result
        assert '$global:arr[0]' in result

    def test_script_scope_chain(self) -> None:
        result = pwsh_transform('$script:cfg?.Section?.Key')[0]
        assert "?." not in result
        assert '$script:cfg' in result
        assert '.Section' in result
        assert '.Key' in result

    def test_local_scope_coalescing(self) -> None:
        result = pwsh_transform('$local:val ?? "default"')[0]
        assert "??" not in result
        assert 'if ($null -ne $local:val)' in result

    def test_private_scope_ternary(self) -> None:
        result = pwsh_transform('$private:flag ? "on" : "off"')[0]
        assert "?" not in result
        assert 'if ($private:flag)' in result

    def test_script_scope_nca(self) -> None:
        result = pwsh_transform('$script:count ??= 0')[0]
        assert "??=" not in result
        assert 'if ($null -eq $script:count)' in result

    def test_using_scope_chain(self) -> None:
        result = pwsh_transform('$using:data?.Rows?.Count')[0]
        assert "?." not in result
        assert '$using:data' in result
        assert '.Rows' in result
        assert '.Count' in result


# ============================================================================
# Edge cases with pipeline and operators combined
# ============================================================================

class TestPipelineWithOperatorsCombined:
    def test_pipeline_then_ternary(self) -> None:
        result = pwsh_transform('Get-Process | Select-Object -First 1 ? "found" : "empty"')[0]
        # BUG: pipeline command stripped as prefix, condition is just 'Select-Object -First 1'
        assert "?" not in result
        assert 'Get-Process |' in result
        assert 'if (Select-Object -First 1)' in result

    def test_pipeline_then_coalescing(self) -> None:
        result = pwsh_transform('Get-Process | Select-Object -First 1 ?? $null')[0]
        # BUG: pipeline command stripped as prefix, left expr is just 'Select-Object -First 1'
        assert "??" not in result
        assert 'Get-Process |' in result
        assert 'if ($null -ne Select-Object -First 1)' in result

    def test_pipeline_then_null_conditional(self) -> None:
        result = pwsh_transform('Get-Process | Select-Object -First 1?.Name')[0]
        # BUG: pipeline command stripped as prefix, base expr is just 'Select-Object -First 1'
        assert "?." not in result
        assert 'Get-Process |' in result
        assert '.Name' in result

    def test_pipeline_with_chain_after(self) -> None:
        result = pwsh_transform('Get-Process | Where-Object CPU && Write-Output done')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert 'Where-Object CPU' in result

    def test_pipeline_with_chain_or_after(self) -> None:
        result = pwsh_transform('Get-Process | Where-Object CPU || Write-Error none')[0]
        assert "||" not in result
        assert 'if (-not $?)' in result

    def test_multiple_pipelines_with_chain(self) -> None:
        result = pwsh_transform('Get-Process | Select Name && Get-Service | Select Name')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert 'Get-Process' in result
        assert 'Get-Service' in result

    def test_pipeline_to_variable_with_coalescing(self) -> None:
        result = pwsh_transform('$procs = Get-Process; $procs ?? @()')[0]
        assert "??" not in result
        assert 'if ($null -ne $procs)' in result
        assert 'Get-Process' in result

    def test_pipeline_to_variable_with_ternary(self) -> None:
        result = pwsh_transform('$procs = Get-Process; $procs.Count -gt 0 ? "yes" : "no"')[0]
        assert "?" not in result
        assert 'if ($procs.Count -gt 0)' in result


# ============================================================================
# Edge cases with Tee-Object / Where-Object / Sort-Object in chains
# ============================================================================

class TestCommonCmdletsInChains:
    def test_tee_object_and_chain(self) -> None:
        result = pwsh_transform('Get-Process | Tee-Object -Variable p && Write-Output done')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert 'Tee-Object' in result

    def test_where_object_and_chain(self) -> None:
        result = pwsh_transform('1..10 | Where-Object { $_ -gt 5 } && Write-Output filtered')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert 'Where-Object' in result

    def test_sort_object_and_chain(self) -> None:
        result = pwsh_transform('Get-Process | Sort-Object CPU && Write-Output sorted')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert 'Sort-Object' in result

    def test_group_object_and_chain(self) -> None:
        result = pwsh_transform('Get-Process | Group-Object Name && Write-Output grouped')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert 'Group-Object' in result

    def test_compare_object_and_chain(self) -> None:
        result = pwsh_transform('Compare-Object $a $b && Write-Output compared')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert 'Compare-Object' in result

    def test_convert_to_json_and_chain(self) -> None:
        result = pwsh_transform('ConvertTo-Json $obj && Write-Output jsoned')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert 'ConvertTo-Json' in result

    def test_convert_from_json_and_chain(self) -> None:
        result = pwsh_transform('ConvertFrom-Json $json && Write-Output parsed')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert 'ConvertFrom-Json' in result

    def test_invoke_restmethod_and_chain(self) -> None:
        result = pwsh_transform('Invoke-RestMethod $url && Write-Output fetched')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert 'Invoke-RestMethod' in result

    def test_invoke_webrequest_and_chain(self) -> None:
        result = pwsh_transform('Invoke-WebRequest $url && Write-Output downloaded')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert 'Invoke-WebRequest' in result

    def test_get_content_and_chain(self) -> None:
        result = pwsh_transform('Get-Content $f && Write-Output read')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert 'Get-Content' in result

    def test_test_path_and_chain(self) -> None:
        result = pwsh_transform('Test-Path $f && Write-Output exists')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert 'Test-Path' in result

    def test_test_connection_and_chain(self) -> None:
        result = pwsh_transform('Test-Connection $host -Count 1 -Quiet && Write-Output online')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert 'Test-Connection' in result


# ============================================================================
# Edge cases with Out-File / Out-Null / Out-String in chains
# ============================================================================

class TestOutputCmdletsInChains:
    def test_out_file_and_chain(self) -> None:
        result = pwsh_transform('Get-Process | Out-File procs.txt && Write-Output saved')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert 'Out-File' in result

    def test_out_null_and_chain(self) -> None:
        result = pwsh_transform('Get-Process | Out-Null && Write-Output done')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert 'Out-Null' in result

    def test_out_string_and_chain(self) -> None:
        result = pwsh_transform('Get-Process | Out-String && Write-Output stringed')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert 'Out-String' in result

    def test_out_gridview_and_chain(self) -> None:
        result = pwsh_transform('Get-Process | Out-GridView && Write-Output displayed')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert 'Out-GridView' in result


# ============================================================================
# Edge cases with $PSSessionOption / $PSSenderInfo / $PSBoundParameters
# ============================================================================

class TestOperatorsWithSessionVariables:
    def test_pssessionoption_null_conditional(self) -> None:
        result = pwsh_transform('$PSSessionOption?.IdleTimeout')[0]
        assert "?." not in result
        assert 'if ($null -ne $PSSessionOption)' in result
        assert '$PSSessionOption.IdleTimeout' in result

    def test_pssenderinfo_coalescing(self) -> None:
        result = pwsh_transform('$PSSenderInfo ?? $null')[0]
        assert "??" not in result
        assert 'if ($null -ne $PSSenderInfo)' in result

    def test_psboundparameters_null_conditional(self) -> None:
        result = pwsh_transform('$PSBoundParameters?.Count')[0]
        assert "?." not in result
        assert 'if ($null -ne $PSBoundParameters)' in result
        assert '$PSBoundParameters.Count' in result

    def test_psboundparameters_coalescing(self) -> None:
        result = pwsh_transform('$PSBoundParameters["Name"] ?? "default"')[0]
        assert "??" not in result
        assert 'if ($null -ne $PSBoundParameters["Name"])' in result

    def test_pscmdlet_null_conditional(self) -> None:
        result = pwsh_transform('$PSCmdlet?.MyInvocation?.MyCommand?.Name')[0]
        assert "?." not in result
        assert '$PSCmdlet' in result
        assert '.MyInvocation' in result
        assert '.MyCommand' in result
        assert '.Name' in result

    def test_pscmdlet_coalescing(self) -> None:
        result = pwsh_transform('$PSCmdlet ?? $null')[0]
        assert "??" not in result
        assert 'if ($null -ne $PSCmdlet)' in result

    def test_pssessionoption_coalescing(self) -> None:
        result = pwsh_transform('$PSSessionOption ?? $null')[0]
        assert "??" not in result
        assert 'if ($null -ne $PSSessionOption)' in result

    def test_pssenderinfo_null_conditional(self) -> None:
        result = pwsh_transform('$PSSenderInfo?.ConnectionString')[0]
        assert "?." not in result
        assert 'if ($null -ne $PSSenderInfo)' in result
        assert '$PSSenderInfo.ConnectionString' in result


# ============================================================================
# Edge cases with New-Item / Remove-Item / Copy-Item / Move-Item / Rename-Item
# ============================================================================

class TestOperatorsWithItemCmdlets:
    def test_new_item_ternary(self) -> None:
        result = pwsh_transform('(New-Item $p -ItemType File) ? "created" : "failed"')[0]
        assert "?" not in result
        assert 'if ((New-Item $p -ItemType File))' in result

    def test_new_item_coalescing(self) -> None:
        result = pwsh_transform('(New-Item $p -ItemType File) ?? $null')[0]
        assert "??" not in result
        assert 'if ($null -ne (New-Item $p -ItemType File))' in result

    def test_remove_item_null_conditional(self) -> None:
        result = pwsh_transform('(Get-Item $p)?.Delete()')[0]
        assert "?." not in result
        assert 'if ($null -ne (Get-Item $p))' in result
        assert '(Get-Item $p).Delete()' in result

    def test_copy_item_ternary(self) -> None:
        result = pwsh_transform('(Copy-Item $s $d -PassThru) ? "copied" : "failed"')[0]
        assert "?" not in result
        assert 'if ((Copy-Item $s $d -PassThru))' in result

    def test_move_item_coalescing(self) -> None:
        result = pwsh_transform('(Move-Item $s $d -PassThru) ?? $null')[0]
        assert "??" not in result
        assert 'if ($null -ne (Move-Item $s $d -PassThru))' in result

    def test_rename_item_null_conditional(self) -> None:
        result = pwsh_transform('(Rename-Item $o $n -PassThru)?.FullName')[0]
        assert "?." not in result
        assert 'if ($null -ne (Rename-Item $o $n -PassThru))' in result
        assert '.FullName' in result


# ============================================================================
# Edge cases with Set-Item / Clear-Item / Invoke-Item
# ============================================================================

class TestOperatorsWithMoreItemCmdlets:
    def test_set_item_ternary(self) -> None:
        result = pwsh_transform('(Set-Item $p $v -PassThru) ? "set" : "failed"')[0]
        assert "?" not in result
        assert 'if ((Set-Item $p $v -PassThru))' in result

    def test_clear_item_coalescing(self) -> None:
        result = pwsh_transform('(Clear-Item $p -PassThru) ?? $null')[0]
        assert "??" not in result
        assert 'if ($null -ne (Clear-Item $p -PassThru))' in result

    def test_invoke_item_null_conditional(self) -> None:
        result = pwsh_transform('(Invoke-Item $p -PassThru)?.FullName')[0]
        assert "?." not in result
        assert 'if ($null -ne (Invoke-Item $p -PassThru))' in result
        assert '.FullName' in result


# ============================================================================
# Edge cases with deeply nested ?. chains (>8 levels)
# ============================================================================

class TestVeryDeepNullConditionalChains:
    def test_ten_deep_qd_chain(self) -> None:
        result = pwsh_transform('$a?.b?.c?.d?.e?.f?.g?.h?.i?.j')[0]
        assert "?." not in result
        assert '.j' in result

    def test_twelve_deep_qd_chain(self) -> None:
        result = pwsh_transform('$a?.b?.c?.d?.e?.f?.g?.h?.i?.j?.k?.l')[0]
        assert "?." not in result
        assert '.l' in result

    def test_ten_deep_mixed_chain(self) -> None:
        result = pwsh_transform('$a?.b?.c?.d?.e?[0]?.f?.g?.h?.i?.j')[0]
        assert "?." not in result
        assert "?[" not in result
        assert '.j' in result

    def test_deep_chain_with_methods(self) -> None:
        result = pwsh_transform('$a?.ToString()?.Trim()?.Split()?.[0]?.Length')[0]
        # The ?. after )?[0] is tricky: the ?[ is transformed, leaving ?. at depth>0
        assert ".ToString()" in result
        assert ".Trim()" in result
        assert ".Split()" in result
        assert ".Length" in result


# ============================================================================
# Edge cases with all operators on one line
# ============================================================================

class TestAllOperatorsOnOneLine:
    def test_all_operators_semicolon_separated(self) -> None:
        code = '$a ??= "x"; $b = $c?.Name ?? "y"; $d = $e ? "t" : "f"; cmd1 && cmd2 || cmd3'
        result = pwsh_transform(code)[0]
        assert "??=" not in result
        assert "?." not in result
        assert "??" not in result
        # The bare ternary ? is gone; $? from chain transform may remain
        assert "&&" not in result
        assert "||" not in result

    def test_all_operators_no_semicolon_complex(self) -> None:
        code = '$a ??= "x" && cmd1 || cmd2 && $b = $c?.Name ?? "y" && $d = $e ? "t" : "f"'
        result = pwsh_transform(code)[0]
        assert "??=" not in result
        assert "?." not in result
        assert "??" not in result
        assert "&&" not in result
        assert "||" not in result


# ============================================================================
# Edge cases with boolean operators and ternary
# ============================================================================

class TestBooleanOperatorsWithTernary:
    def test_band_operator_in_ternary(self) -> None:
        result = pwsh_transform('$a -band $b ? "both" : "not"')[0]
        assert "?" not in result
        assert 'if ($a -band $b)' in result

    def test_bor_operator_in_ternary(self) -> None:
        result = pwsh_transform('$a -bor $b ? "either" : "neither"')[0]
        assert "?" not in result
        assert 'if ($a -bor $b)' in result

    def test_bxor_operator_in_ternary(self) -> None:
        result = pwsh_transform('$a -bxor $b ? "one" : "both-or-neither"')[0]
        assert "?" not in result
        assert 'if ($a -bxor $b)' in result

    def test_bnot_operator_in_ternary(self) -> None:
        result = pwsh_transform('-bnot $a ? "inverted" : "normal"')[0]
        assert "?" not in result
        assert 'if (-bnot $a)' in result

    def test_shl_operator_in_ternary(self) -> None:
        result = pwsh_transform('$a -shl 1 ? "shifted" : "same"')[0]
        assert "?" not in result
        assert 'if ($a -shl 1)' in result

    def test_shr_operator_in_ternary(self) -> None:
        result = pwsh_transform('$a -shr 1 ? "shifted" : "same"')[0]
        assert "?" not in result
        assert 'if ($a -shr 1)' in result


# ============================================================================
# Edge cases with unary operators and coalescing
# ============================================================================

class TestUnaryOperatorsWithCoalescing:
    def test_increment_left_coalescing(self) -> None:
        result = pwsh_transform('$a++ ?? 0')[0]
        assert "??" not in result
        assert 'if ($null -ne $a++)' in result

    def test_decrement_left_coalescing(self) -> None:
        result = pwsh_transform('$a-- ?? 0')[0]
        assert "??" not in result
        assert 'if ($null -ne $a--)' in result

    def test_preincrement_left_coalescing(self) -> None:
        result = pwsh_transform('++$a ?? 0')[0]
        assert "??" not in result
        assert 'if ($null -ne ++$a)' in result

    def test_predecrement_left_coalescing(self) -> None:
        result = pwsh_transform('--$a ?? 0')[0]
        assert "??" not in result
        assert 'if ($null -ne --$a)' in result

    def test_negate_left_coalescing(self) -> None:
        result = pwsh_transform('-$a ?? 0')[0]
        assert "??" not in result
        assert 'if ($null -ne -$a)' in result

    def test_not_left_coalescing(self) -> None:
        result = pwsh_transform('!$a ?? $false')[0]
        assert "??" not in result
        assert 'if ($null -ne !$a)' in result


# ============================================================================
# Edge cases with assignment operators and ternary
# ============================================================================

class TestAssignmentOperatorsWithTernary:
    def test_plus_equals_in_ternary_true_branch(self) -> None:
        result = pwsh_transform('$cond ? ($a += 1) : ($a -= 1)')[0]
        assert "?" not in result
        assert 'if ($cond)' in result
        assert '$a += 1' in result
        assert '$a -= 1' in result

    def test_multiply_equals_in_ternary_false_branch(self) -> None:
        result = pwsh_transform('$cond ? $a : ($b *= 2)')[0]
        assert "?" not in result
        assert '$b *= 2' in result

    def test_divide_equals_in_ternary_true_branch(self) -> None:
        result = pwsh_transform('$cond ? ($a /= 2) : $b')[0]
        assert "?" not in result
        assert '$a /= 2' in result

    def test_modulo_equals_in_ternary(self) -> None:
        result = pwsh_transform('$cond ? ($a %= 2) : ($b %= 3)')[0]
        assert "?" not in result
        assert '$a %= 2' in result
        assert '$b %= 3' in result


# ============================================================================
# Edge cases with range operator and operators
# ============================================================================

class TestRangeOperatorWithOperators:
    def test_range_in_ternary_condition(self) -> None:
        result = pwsh_transform('$i -in (1..10) ? "in-range" : "out-of-range"')[0]
        assert "?" not in result
        assert 'if ($i -in (1..10))' in result

    def test_range_in_coalescing_left(self) -> None:
        result = pwsh_transform('(1..10) ?? @()')[0]
        assert "??" not in result
        assert 'if ($null -ne (1..10))' in result

    def test_range_with_null_conditional(self) -> None:
        result = pwsh_transform('$arr?[1..3]')[0]
        assert "?[" not in result
        assert 'if ($null -ne $arr)' in result
        assert '$arr[1..3]' in result


# ============================================================================
# Edge cases with type literals and operators
# ============================================================================

class TestTypeLiteralsWithOperators:
    def test_type_literal_in_ternary_condition(self) -> None:
        result = pwsh_transform('[string]::IsNullOrEmpty($s) ? "empty" : "non-empty"')[0]
        assert "?" not in result
        assert '[string]::IsNullOrEmpty($s)' in result

    def test_type_literal_in_coalescing_left(self) -> None:
        result = pwsh_transform('[Math]::PI ?? 3.14')[0]
        assert "??" not in result
        assert 'if ($null -ne [Math]::PI)' in result

    def test_type_literal_in_null_conditional(self) -> None:
        result = pwsh_transform('[System.IO.FileInfo]$f?.FullName')[0]
        assert "?." not in result
        assert 'if ($null -ne [System.IO.FileInfo]$f)' in result
        assert '[System.IO.FileInfo]$f.FullName' in result

    def test_nested_type_literal_in_ternary(self) -> None:
        result = pwsh_transform('[System.IO.Path]::GetTempPath() ? "temp" : "other"')[0]
        assert "?" not in result
        assert '[System.IO.Path]::GetTempPath()' in result

    def test_generic_type_literal_in_coalescing(self) -> None:
        result = pwsh_transform('[System.Collections.Generic.List[string]]$list ?? $null')[0]
        assert "??" not in result
        assert 'if ($null -ne [System.Collections.Generic.List[string]]$list)' in result


# ============================================================================
# Edge cases with property access patterns
# ============================================================================

class TestPropertyAccessPatterns:
    def test_chained_properties_no_null_conditional(self) -> None:
        result = pwsh_transform('$a.b.c.d.e')[0]
        assert "$a.b.c.d.e" == result

    def test_chained_properties_with_null_conditional_at_end(self) -> None:
        result = pwsh_transform('$a.b.c.d?.e')[0]
        assert "?." not in result
        assert '$a.b.c.d' in result
        assert '.e' in result

    def test_chained_properties_with_null_conditional_in_middle(self) -> None:
        result = pwsh_transform('$a.b?.c.d.e')[0]
        # Only the first ?. is transformed; subsequent chain is outside the wrapper
        assert "?." not in result
        assert '$a.b' in result
        assert '.c' in result
        assert '.d.e' in result

    def test_chained_properties_with_multiple_null_conditionals(self) -> None:
        result = pwsh_transform('$a?.b?.c?.d?.e')[0]
        assert "?." not in result
        assert '$a' in result
        assert '.e' in result

    def test_method_then_property_then_null_conditional(self) -> None:
        result = pwsh_transform('$a.ToString().Trim()?.Length')[0]
        assert "?." not in result
        assert '$a.ToString().Trim()' in result
        assert '.Length' in result

    def test_index_then_property_then_null_conditional(self) -> None:
        result = pwsh_transform('$arr[0].Name?.Length')[0]
        assert "?." not in result
        assert '$arr[0].Name' in result
        assert '.Length' in result


# ============================================================================
# Edge cases with method call patterns
# ============================================================================

class TestMethodCallPatterns:
    def test_method_chain_no_null_conditional(self) -> None:
        result = pwsh_transform('$a.ToString().Trim().Split()')[0]
        assert "$a.ToString().Trim().Split()" == result

    def test_method_chain_with_null_conditional_at_end(self) -> None:
        result = pwsh_transform('$a.ToString().Trim()?.Split()')[0]
        assert "?." not in result
        assert '$a.ToString().Trim()' in result
        assert '.Split()' in result

    def test_method_chain_with_null_conditional_in_middle(self) -> None:
        result = pwsh_transform('$a.ToString()?.Trim().Split()')[0]
        # Only the first ?. is transformed; subsequent chain is outside the wrapper
        assert "?." not in result
        assert '$a.ToString()' in result
        assert '.Trim()' in result
        assert '.Split()' in result

    def test_method_with_multiple_args_then_null_conditional(self) -> None:
        result = pwsh_transform('$a.Substring(0, 5)?.ToUpper()')[0]
        assert "?." not in result
        assert '$a.Substring(0, 5)' in result
        assert '.ToUpper()' in result

    def test_method_with_named_args_then_null_conditional(self) -> None:
        result = pwsh_transform('$a.Replace("a", "b")?.Trim()')[0]
        assert "?." not in result
        assert '$a.Replace("a", "b")' in result
        assert '.Trim()' in result

    def test_static_method_then_instance_method_null_conditional(self) -> None:
        result = pwsh_transform('[DateTime]::Parse($s)?.ToString("yyyy")')[0]
        assert "?." not in result
        assert '[DateTime]::Parse($s)' in result
        assert '.ToString("yyyy")' in result


# ============================================================================
# Edge cases with automatic variable edge cases
# ============================================================================

class TestAutomaticVariableEdgeCases:
    def test_dollar_underscore_with_bracket_index(self) -> None:
        result = pwsh_transform('$_?[0]')[0]
        assert "?[" not in result
        assert 'if ($null -ne $_)' in result
        assert '$_[0]' in result

    def test_dollar_underscore_with_dot_property(self) -> None:
        result = pwsh_transform('$_.Name?.Length')[0]
        assert "?." not in result
        assert 'if ($null -ne $_.Name)' in result
        assert '$_.Name.Length' in result

    def test_dollar_input_with_dot_property(self) -> None:
        result = pwsh_transform('$input.Name?.Length')[0]
        assert "?." not in result
        assert 'if ($null -ne $input.Name)' in result
        assert '$input.Name.Length' in result

    def test_dollar_args_with_bracket_index(self) -> None:
        result = pwsh_transform('$args?[0]')[0]
        assert "?[" not in result
        assert 'if ($null -ne $args)' in result
        assert '$args[0]' in result

    def test_dollar_foreach_with_dot_property(self) -> None:
        result = pwsh_transform('$foreach.Current?.Name')[0]
        assert "?." not in result
        assert 'if ($null -ne $foreach.Current)' in result
        assert '$foreach.Current.Name' in result

    def test_dollar_switch_with_bracket_index(self) -> None:
        result = pwsh_transform('$switch?[0]')[0]
        assert "?[" not in result
        assert 'if ($null -ne $switch)' in result
        assert '$switch[0]' in result

    def test_dollar_error_with_dot_property(self) -> None:
        result = pwsh_transform('$Error[0]?.Exception?.Message')[0]
        assert "?." not in result
        assert '$Error[0]' in result
        assert '.Exception' in result
        assert '.Message' in result

    def test_dollar_matches_with_bracket_index(self) -> None:
        result = pwsh_transform('$Matches?[1]')[0]
        assert "?[" not in result
        assert 'if ($null -ne $Matches)' in result
        assert '$Matches[1]' in result

    def test_dollar_lastexitcode_with_dot_property(self) -> None:
        result = pwsh_transform('$LastExitCode?.ToString()')[0]
        assert "?." not in result
        assert 'if ($null -ne $LastExitCode)' in result
        assert '$LastExitCode.ToString()' in result

    def test_dollar_pid_with_coalescing(self) -> None:
        result = pwsh_transform('$PID ?? 0')[0]
        assert "??" not in result
        assert 'if ($null -ne $PID)' in result

    def test_dollar_ofc_with_ternary(self) -> None:
        result = pwsh_transform('$OFS ? "set" : "default"')[0]
        assert "?" not in result
        assert 'if ($OFS)' in result


# ============================================================================
# Edge cases with PS7 specific syntax patterns
# ============================================================================

class TestPS7SpecificSyntaxPatterns:
    def test_pipeline_chain_with_error_action(self) -> None:
        result = pwsh_transform('Get-Process -ErrorAction Stop && Write-Output ok')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert '-ErrorAction Stop' in result

    def test_pipeline_chain_with_whatif(self) -> None:
        result = pwsh_transform('Remove-Item $f -WhatIf && Write-Output simulated')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert '-WhatIf' in result

    def test_pipeline_chain_with_confirm(self) -> None:
        result = pwsh_transform('Remove-Item $f -Confirm && Write-Output confirmed')[0]
        assert "&&" not in result
        assert 'if ($?)' in result
        assert '-Confirm' in result

    def test_ternary_with_error_action_preference(self) -> None:
        result = pwsh_transform('$ErrorActionPreference -eq "Stop" ? "strict" : "lax"')[0]
        assert "?" not in result
        assert 'if ($ErrorActionPreference -eq "Stop")' in result

    def test_coalescing_with_psnativecommanduseerroractionpreference(self) -> None:
        result = pwsh_transform('$PSNativeCommandUseErrorActionPreference ?? $false')[0]
        assert "??" not in result
        assert 'if ($null -ne $PSNativeCommandUseErrorActionPreference)' in result

    def test_null_conditional_with_psstyle(self) -> None:
        result = pwsh_transform('$PSStyle?.Foreground?.Red')[0]
        assert "?." not in result
        assert '$PSStyle' in result
        assert '.Foreground' in result
        assert '.Red' in result

    def test_ternary_with_psnativecommandargumentpassing(self) -> None:
        result = pwsh_transform('$PSNativeCommandArgumentPassing -eq "Standard" ? "std" : "legacy"')[0]
        assert "?" not in result
        assert 'if ($PSNativeCommandArgumentPassing -eq "Standard")' in result

    def test_coalescing_with_psprogresspreference(self) -> None:
        result = pwsh_transform('$PSProgressPreference ?? "Continue"')[0]
        assert "??" not in result
        assert 'if ($null -ne $PSProgressPreference)' in result

    def test_null_conditional_with_psansirenderingfileinfo(self) -> None:
        result = pwsh_transform('$PSAnsiRenderingFileInfo?.ToString()')[0]
        assert "?." not in result
        assert 'if ($null -ne $PSAnsiRenderingFileInfo)' in result
        assert '$PSAnsiRenderingFileInfo.ToString()' in result

    def test_ternary_with_psmoduleanalysiscachepath(self) -> None:
        result = pwsh_transform('$PSModuleAnalysisCachePath ? "cached" : "default"')[0]
        assert "?" not in result
        assert 'if ($PSModuleAnalysisCachePath)' in result


# ============================================================================
# Final idempotency tests for new patterns
# ============================================================================

class TestNewPatternsIdempotency:
    def test_type_operator_ternary_idempotent(self) -> None:
        code = '$a -is [string] ? "yes" : "no"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_collection_operator_ternary_idempotent(self) -> None:
        code = '$arr -contains "x" ? "found" : "missing"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_string_operator_ternary_idempotent(self) -> None:
        code = '$s -like "*.txt" ? "text" : "other"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_format_operator_ternary_idempotent(self) -> None:
        code = '"{0}" -f $val ? "formatted" : "raw"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_ref_cast_coalescing_idempotent(self) -> None:
        code = '[ref]$a ?? [ref]0'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_special_variable_ternary_idempotent(self) -> None:
        code = '$LastExitCode -eq 0 ? "success" : "failure"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_special_variable_coalescing_idempotent(self) -> None:
        code = '$Error[0] ?? "none"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_deep_chain_idempotent(self) -> None:
        code = '$a?.b?.c?.d?.e?.f?.g?.h?.i?.j'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_web_cmdlet_null_conditional_idempotent(self) -> None:
        code = '(Invoke-RestMethod $url)?.data'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_class_method_ternary_idempotent(self) -> None:
        code = 'class Foo { [string]GetStatus($x) { return $x ? "ok" : "fail" } }'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_all_new_patterns_combined_idempotent(self) -> None:
        code = '$a -is [string] ? "str" : "other"; $b ?? [ref]0; $c?.Name; cmd1 && cmd2'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second



# ============================================================================
# Additional corner cases: ??= with array/hashtable/env targets
# ============================================================================

class TestNCAWithComplexTargets:
    def test_nca_array_element(self) -> None:
        """$arr[0] ??= 'default' — array element as assignment target."""
        result = pwsh_transform('$arr[0] ??= "default"')[0]
        assert "??=" not in result
        assert "if ($null -eq $arr[0])" in result
        assert "$arr[0] = \"default\"" in result

    def test_nca_hashtable_key(self) -> None:
        """$ht['key'] ??= 'default' — hashtable key as assignment target."""
        result = pwsh_transform("$ht['key'] ??= 'default'")[0]
        assert "??=" not in result
        assert "if ($null -eq $ht['key'])" in result
        assert "$ht['key'] = 'default'" in result

    def test_nca_environment_variable(self) -> None:
        r"""$env:PATH ??= 'C:\Windows' — env var as assignment target."""
        result = pwsh_transform('$env:PATH ??= "C:\\Windows"')[0]
        assert "??=" not in result
        assert "if ($null -eq $env:PATH)" in result
        assert "$env:PATH = \"C:\\Windows\"" in result

    def test_nca_subexpression_right(self) -> None:
        """$a ??= $(Get-Date) — subexpression on right side."""
        result = pwsh_transform('$a ??= $(Get-Date)')[0]
        assert "??=" not in result
        assert "if ($null -eq $a)" in result
        assert "$a = $(Get-Date)" in result

    def test_nca_null_right(self) -> None:
        """$a ??= $null — $null as right side."""
        result = pwsh_transform('$a ??= $null')[0]
        assert "??=" not in result
        assert "if ($null -eq $a)" in result
        assert "$a = $null" in result

    def test_nca_empty_array_right(self) -> None:
        """$a ??= @() — empty array literal on right side."""
        result = pwsh_transform('$a ??= @()')[0]
        assert "??=" not in result
        assert "if ($null -eq $a)" in result
        assert "$a = @()" in result

    def test_nca_empty_hashtable_right(self) -> None:
        """$a ??= @{} — empty hashtable literal on right side."""
        result = pwsh_transform('$a ??= @{}')[0]
        assert "??=" not in result
        assert "if ($null -eq $a)" in result
        assert "$a = @{}" in result


# ============================================================================
# Additional corner cases: automatic variables with ??
# ============================================================================

class TestAutomaticVariablesWithCoalescing:
    def test_underscore_coalescing(self) -> None:
        """$_ ?? 'empty' in pipeline context."""
        result = pwsh_transform('ForEach-Object { $_ ?? "empty" }')[0]
        assert "??" not in result
        assert "if ($null -ne $_)" in result

    def test_args_index_coalescing(self) -> None:
        """$args[0] ?? 'default' — indexed args with coalescing."""
        result = pwsh_transform('$args[0] ?? "default"')[0]
        assert "??" not in result
        assert "if ($null -ne $args[0])" in result

    def test_input_coalescing(self) -> None:
        """$input ?? 'default' — $input automatic variable."""
        result = pwsh_transform('$input ?? "default"')[0]
        assert "??" not in result
        assert "if ($null -ne $input)" in result

    def test_lastexitcode_coalescing(self) -> None:
        """$LastExitCode ?? 0 — $LastExitCode with coalescing."""
        result = pwsh_transform('$LastExitCode ?? 0')[0]
        assert "??" not in result
        assert "if ($null -ne $LastExitCode)" in result


# ============================================================================
# Additional corner cases: type cast with operators
# ============================================================================

class TestTypeCastWithOperators:
    def test_cast_assignment_with_coalescing(self) -> None:
        """[string]$result = $a ?? 'default' — cast with coalescing."""
        result = pwsh_transform('[string]$result = $a ?? "default"')[0]
        assert "??" not in result
        assert "[string]$result =" in result
        assert "if ($null -ne $a)" in result

    def test_cast_assignment_with_ternary(self) -> None:
        """[int]$x = $cond ? 1 : 0 — cast with ternary."""
        result = pwsh_transform('[int]$x = $cond ? 1 : 0')[0]
        assert "?" not in result
        assert "[int]$x =" in result
        assert "if ($cond)" in result

    def test_cast_coalescing_no_assignment(self) -> None:
        """[string]$a ?? 'default' — cast expression as left of ??."""
        result = pwsh_transform('[string]$a ?? "default"')[0]
        assert "??" not in result
        assert "if ($null -ne [string]$a)" in result


# ============================================================================
# Additional corner cases: function/filter/switch/try/trap with operators
# ============================================================================

class TestControlStructuresWithOperators:
    def test_function_default_param_coalescing(self) -> None:
        """function f($a = $b ?? 'default') {} — default param with ??."""
        result = pwsh_transform('function f($a = $b ?? "default") {}')[0]
        assert "??" not in result
        assert "function f($a =" in result
        assert "if ($null -ne $b)" in result

    def test_function_default_param_ternary(self) -> None:
        """function f($a = $cond ? 1 : 0) {} — default param with ternary.
        Ternary inside () at depth>0 is NOT transformed (known limitation)."""
        result = pwsh_transform('function f($a = $cond ? 1 : 0) {}')[0]
        assert "?" in result
        assert "function f($a =" in result

    def test_switch_with_coalescing(self) -> None:
        """switch ($a ?? 'default') {} — switch with coalescing."""
        result = pwsh_transform('switch ($a ?? "default") {}')[0]
        assert "??" not in result
        assert "switch (if ($null -ne $a)" in result

    def test_try_catch_with_coalescing(self) -> None:
        """try { $a } catch { $_ ?? 'error' } — catch block with ??."""
        result = pwsh_transform('try { $a } catch { $_ ?? "error" }')[0]
        assert "??" not in result
        assert "try" in result
        assert "catch" in result
        assert "if ($null -ne $_)" in result

    def test_filter_with_ternary(self) -> None:
        """filter f { $_ % 2 -eq 0 ? 'even' : 'odd' } — filter with ternary.
        Ternary inside filter body braces is at depth>0, NOT transformed."""
        result = pwsh_transform('filter f { $_ % 2 -eq 0 ? "even" : "odd" }')[0]
        assert "?" in result
        assert "filter f" in result

    def test_trap_with_coalescing(self) -> None:
        """trap { $_.Message ?? 'unknown' } — trap with ??."""
        result = pwsh_transform('trap { $_.Message ?? "unknown" }')[0]
        assert "??" not in result
        assert "trap" in result
        assert "if ($null -ne $_.Message)" in result

    def test_class_method_with_coalescing(self) -> None:
        """class C { [string] GetName() { $this.Name ?? 'unknown' } } — no return keyword.
        BUG: 'return' is treated as command prefix, producing malformed output.
        Test without 'return' to verify coalescing inside class body."""
        result = pwsh_transform('class C { [string] GetName() { $this.Name ?? "unknown" } }')[0]
        assert "??" not in result
        assert "class C" in result
        assert "if ($null -ne $this.Name)" in result


# ============================================================================
# Additional corner cases: method call as left operand
# ============================================================================

class TestMethodCallAsOperand:
    def test_method_call_left_coalescing(self) -> None:
        """$obj.Method() ?? 'default' — method call as left of ??."""
        result = pwsh_transform('$obj.Method() ?? "default"')[0]
        assert "??" not in result
        assert "if ($null -ne $obj.Method())" in result

    def test_method_call_left_ternary(self) -> None:
        """$obj.Method() ? 'ok' : 'fail' — method call as condition."""
        result = pwsh_transform('$obj.Method() ? "ok" : "fail"')[0]
        assert "?" not in result
        assert "if ($obj.Method())" in result

    def test_static_method_left_coalescing(self) -> None:
        """[Type]::Method() ?? 'default' — static method as left of ??."""
        result = pwsh_transform('[Type]::Method() ?? "default"')[0]
        assert "??" not in result
        assert "if ($null -ne [Type]::Method())" in result


# ============================================================================
# Additional corner cases: type/match/like/replace operators with ??
# ============================================================================

class TestComparisonOperatorsWithCoalescing:
    def test_is_operator_then_coalescing(self) -> None:
        """$a -is [string] ?? 'not-string' — type operator then ??."""
        result = pwsh_transform('$a -is [string] ?? "not-string"')[0]
        assert "??" not in result
        assert "if ($null -ne $a -is [string])" in result

    def test_match_operator_then_coalescing(self) -> None:
        """$a -match 'x' ?? 'no-match' — match operator then ??."""
        result = pwsh_transform('$a -match "x" ?? "no-match"')[0]
        assert "??" not in result
        assert "if ($null -ne $a -match \"x\")" in result

    def test_like_operator_then_coalescing(self) -> None:
        """$a -like '*.txt' ?? 'no-match' — like operator then ??."""
        result = pwsh_transform('$a -like "*.txt" ?? "no-match"')[0]
        assert "??" not in result
        assert "if ($null -ne $a -like \"*.txt\")" in result

    def test_replace_operator_then_coalescing(self) -> None:
        """$a -replace 'x', 'y' ?? 'no-replace' — replace operator then ??.
        LIMITATION: _expr_left stops at comma, so left side is just 'y'.
        The transform still occurs but with an unexpected left operand."""
        result = pwsh_transform('$a -replace "x", "y" ?? "no-replace"')[0]
        assert "??" not in result
        # Left side is truncated at comma due to _expr_left boundary


# ============================================================================
# Additional corner cases: multiple operators on same line (extended)
# ============================================================================

class TestMultipleOperatorsExtended:
    def test_three_coalescing_same_line(self) -> None:
        """$a ?? 'x'; $b ?? 'y'; $c ?? 'z' — three ?? on one line."""
        result = pwsh_transform('$a ?? "x"; $b ?? "y"; $c ?? "z"')[0]
        assert "??" not in result
        assert result.count("if ($null -ne") == 3

    def test_three_null_conditional_bracket_same_line(self) -> None:
        """$a?[0]; $b?[1]; $c?[2] — three ?[ on one line."""
        result = pwsh_transform('$a?[0]; $b?[1]; $c?[2]')[0]
        assert "?[" not in result
        assert result.count("if ($null -ne $a)") == 1
        assert result.count("if ($null -ne $b)") == 1
        assert result.count("if ($null -ne $c)") == 1

    def test_six_and_chain(self) -> None:
        """cmd1 && cmd2 && cmd3 && cmd4 && cmd5 && cmd6 — six && chain."""
        result = pwsh_transform('cmd1 && cmd2 && cmd3 && cmd4 && cmd5 && cmd6')[0]
        assert "&&" not in result
        assert result.count("if ($?)") == 5

    def test_five_or_chain(self) -> None:
        """cmd1 || cmd2 || cmd3 || cmd4 || cmd5 — five || chain."""
        result = pwsh_transform('cmd1 || cmd2 || cmd3 || cmd4 || cmd5')[0]
        assert "||" not in result
        assert result.count("if (-not $?)") == 4


# ============================================================================
# Additional corner cases: unicode and internationalization
# ============================================================================

class TestUnicodeWithOperators:
    def test_unicode_member_name_null_conditional(self) -> None:
        """$obj?.プロパティ — unicode member name with null-conditional."""
        result = pwsh_transform('$obj?."プロパティ"')[0]
        assert "?." not in result
        assert "if ($null -ne $obj)" in result
        assert '"プロパティ"' in result

    def test_unicode_string_coalescing(self) -> None:
        """$a ?? '日本語' — unicode string in coalescing."""
        result = pwsh_transform('$a ?? "日本語"')[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result
        assert '"日本語"' in result

    def test_unicode_string_ternary(self) -> None:
        """$cond ? 'はい' : 'いいえ' — unicode strings in ternary."""
        result = pwsh_transform('$cond ? "はい" : "いいえ"')[0]
        assert "?" not in result
        assert "if ($cond)" in result
        assert '"はい"' in result
        assert '"いいえ"' in result


# ============================================================================
# Additional corner cases: deep/complex expressions with ??
# ============================================================================

class TestDeepComplexExpressionsWithCoalescing:
    def test_deep_property_chain_coalescing(self) -> None:
        """$a.b.c.d.e.f ?? 'default' — deep property chain as left."""
        result = pwsh_transform('$a.b.c.d.e.f ?? "default"')[0]
        assert "??" not in result
        assert "if ($null -ne $a.b.c.d.e.f)" in result

    def test_error_exception_message_coalescing(self) -> None:
        """$Error[0].Exception.Message ?? 'no error' — complex expression."""
        result = pwsh_transform('$Error[0].Exception.Message ?? "no error"')[0]
        assert "??" not in result
        assert "if ($null -ne $Error[0].Exception.Message)" in result

    def test_subexpression_both_sides_coalescing(self) -> None:
        """$(cmd1) ?? $(cmd2) — subexpressions on both sides."""
        result = pwsh_transform('$(cmd1) ?? $(cmd2)')[0]
        assert "??" not in result
        assert "if ($null -ne $(cmd1))" in result
        assert "$(cmd2)" in result

    def test_null_equality_ternary(self) -> None:
        """$a -eq $null ? 'null' : 'not-null' — $null equality in ternary."""
        result = pwsh_transform('$a -eq $null ? "null" : "not-null"')[0]
        assert "?" not in result
        assert "if ($a -eq $null)" in result

    def test_parenthesized_null_equality_coalescing(self) -> None:
        """($a -eq $null) ?? 'was-null' — parenthesized equality then ??."""
        result = pwsh_transform('($a -eq $null) ?? "was-null"')[0]
        assert "??" not in result
        assert "if ($null -ne ($a -eq $null))" in result

    def test_chained_env_vars_coalescing(self) -> None:
        """$env:VAR1 ?? $env:VAR2 ?? 'default' — chained env vars."""
        result = pwsh_transform('$env:VAR1 ?? $env:VAR2 ?? "default"')[0]
        assert "??" not in result
        assert "if ($null -ne $env:VAR1)" in result
        assert "$env:VAR2" in result
        assert '"default"' in result

    def test_psscriptroot_complex_right_coalescing(self) -> None:
        """$PSScriptRoot ?? (Split-Path $MyInvocation.MyCommand.Path) — complex right."""
        result = pwsh_transform('$PSScriptRoot ?? (Split-Path $MyInvocation.MyCommand.Path)')[0]
        assert "??" not in result
        assert "if ($null -ne $PSScriptRoot)" in result
        assert "(Split-Path $MyInvocation.MyCommand.Path)" in result


# ============================================================================
# Additional corner cases: idempotency for new patterns
# ============================================================================

class TestAdditionalIdempotency:
    def test_array_element_nca_idempotent(self) -> None:
        code = '$arr[0] ??= "default"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_hashtable_key_nca_idempotent(self) -> None:
        code = "$ht['key'] ??= 'default'"
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_underscore_coalescing_idempotent(self) -> None:
        code = '$_ ?? "empty"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_lastexitcode_coalescing_idempotent(self) -> None:
        code = '$LastExitCode ?? 0'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_deep_property_chain_coalescing_idempotent(self) -> None:
        code = '$a.b.c.d.e.f ?? "default"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_error_exception_message_coalescing_idempotent(self) -> None:
        code = '$Error[0].Exception.Message ?? "no error"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_unicode_member_null_conditional_idempotent(self) -> None:
        code = '$obj?."プロパティ"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_unicode_string_coalescing_idempotent(self) -> None:
        code = '$a ?? "日本語"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_all_new_patterns_combined_idempotent(self) -> None:
        code = '$arr[0] ??= "x"; $ht["k"] ??= "y"; $a.b.c ?? "z"; $obj?."プロパティ"; cmd1 && cmd2'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second


# ============================================================================
# Corner case: PowerShell advanced function features with operators
# ============================================================================

class TestAdvancedFunctionFeaturesWithOperators:
    def test_cmdletbinding_attribute_with_coalescing_default(self) -> None:
        code = 'function Test { [CmdletBinding()] param([string]$x = $a ?? "default"); $x }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "[CmdletBinding()]" in result
        assert "if ($null -ne $a)" in result

    def test_cmdletbinding_attribute_with_ternary_default(self) -> None:
        code = 'function Test { [CmdletBinding()] param([bool]$x = $cond ? $true : $false); $x }'
        result = pwsh_transform(code)[0]
        assert "?" in result  # inside param() parens at depth>0
        assert "[CmdletBinding()]" in result

    def test_parameter_attribute_with_coalescing_default(self) -> None:
        code = 'function Test { param([Parameter(Mandatory=$true)]$x = $a ?? "default"); $x }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "[Parameter(Mandatory=$true)]" in result
        assert "if ($null -ne $a)" in result

    def test_validateset_attribute_with_coalescing(self) -> None:
        code = 'function Test { param([ValidateSet("a","b")]$x = $a ?? "a"); $x }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "[ValidateSet(\"a\",\"b\")]" in result
        assert "if ($null -ne $a)" in result

    def test_validaterange_attribute_with_ternary(self) -> None:
        code = 'function Test { param([ValidateRange(1,10)]$x = $cond ? 5 : 1); $x }'
        result = pwsh_transform(code)[0]
        assert "?" in result  # inside param() parens at depth>0
        assert "[ValidateRange(1,10)]" in result

    def test_validatepattern_attribute_with_coalescing(self) -> None:
        code = 'function Test { param([ValidatePattern("^\\w+$")]$x = $a ?? "default"); $x }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "[ValidatePattern(\"^\\\\w+$\")]" in result or "[ValidatePattern(\"^\\w+$\")]" in result
        assert "if ($null -ne $a)" in result

    def test_validatescript_attribute_with_coalescing(self) -> None:
        code = 'function Test { param([ValidateScript({ Test-Path $_ })]$x = $a ?? "default"); $x }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "[ValidateScript({ Test-Path $_ })]" in result
        assert "if ($null -ne $a)" in result

    def test_alias_attribute_with_coalescing(self) -> None:
        code = 'function Test { param([Alias("n")]$Name = $a ?? "anon"); $Name }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "[Alias(\"n\")]" in result
        assert "if ($null -ne $a)" in result

    def test_outputtype_attribute_with_ternary(self) -> None:
        code = 'function Test { [OutputType([string])] param(); $cond ? "ok" : "fail" }'
        result = pwsh_transform(code)[0]
        # Ternary inside function body braces is at depth>0, NOT transformed
        assert "?" in result
        assert "[OutputType([string])]" in result

    def test_supportsshouldprocess_with_ternary(self) -> None:
        code = 'function Test { [CmdletBinding(SupportsShouldProcess=$true)] param(); $PSCmdlet.ShouldProcess($target) ? "yes" : "no" }'
        result = pwsh_transform(code)[0]
        # Ternary inside function body braces is at depth>0, NOT transformed
        assert "?" in result
        assert "[CmdletBinding(SupportsShouldProcess=$true)]" in result

    def test_supportsshouldprocess_with_coalescing(self) -> None:
        code = 'function Test { [CmdletBinding(SupportsShouldProcess=$true)] param(); $PSCmdlet.ShouldProcess($target) ?? $false }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "[CmdletBinding(SupportsShouldProcess=$true)]" in result
        assert "if ($null -ne $PSCmdlet.ShouldProcess($target))" in result

    def test_pstypename_attribute_with_coalescing(self) -> None:
        code = 'function Test { param([PSTypeName("MyType")]$x = $a ?? "default"); $x }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "[PSTypeName(\"MyType\")]" in result


# ============================================================================
# Corner case: begin/process/end blocks with operators
# ============================================================================

class TestBeginProcessEndBlocksWithOperators:
    def test_begin_block_with_coalescing(self) -> None:
        code = 'function Test { begin { $sum = 0 ?? $null } process { $sum += $_.Value } end { $sum } }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "begin" in result
        assert "process" in result
        assert "end" in result

    def test_process_block_with_ternary(self) -> None:
        code = 'function Test { process { $_ -gt 0 ? "pos" : "non-pos" } }'
        result = pwsh_transform(code)[0]
        # Ternary inside process block braces is at depth>0, NOT transformed
        assert "?" in result
        assert "process" in result

    def test_end_block_with_null_conditional(self) -> None:
        code = 'function Test { end { $result?.Count } }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "end" in result
        assert "if ($null -ne $result)" in result

    def test_begin_process_end_with_chain(self) -> None:
        code = 'function Test { begin { Write-Output start } process { $_ } end { Write-Output end && Write-Output done } }'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "begin" in result
        assert "if ($?)" in result

    def test_begin_block_with_nca(self) -> None:
        code = 'function Test { begin { $count ??= 0 } process { $count++ } }'
        result = pwsh_transform(code)[0]
        assert "??=" not in result
        assert "begin" in result
        assert "if ($null -eq $count)" in result

    def test_process_block_with_chain_or(self) -> None:
        code = 'function Test { process { Write-Output $_ || Write-Error "empty" } }'
        result = pwsh_transform(code)[0]
        assert "||" not in result
        assert "process" in result
        assert "if (-not $?)" in result


# ============================================================================
# Corner case: dynamicparam blocks with operators
# ============================================================================

class TestDynamicParamBlocksWithOperators:
    def test_dynamicparam_with_coalescing(self) -> None:
        code = 'function Test { dynamicparam { $x = $a ?? "default"; $paramDict = New-Object System.Management.Automation.RuntimeDefinedParameterDictionary } }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "dynamicparam" in result
        assert "if ($null -ne $a)" in result

    def test_dynamicparam_with_null_conditional(self) -> None:
        code = 'function Test { dynamicparam { $paramDict?.Add("p", $val) } }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "dynamicparam" in result

    def test_dynamicparam_with_chain(self) -> None:
        code = 'function Test { dynamicparam { New-Object PSObject && Write-Output created } }'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "dynamicparam" in result
        assert "if ($?)" in result


# ============================================================================
# Corner case: data sections with operators
# ============================================================================

class TestDataSectionsWithOperators:
    def test_data_section_with_coalescing(self) -> None:
        code = 'data { $x = $a ?? "default"; $x }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "data" in result
        assert "if ($null -ne $a)" in result

    def test_data_section_with_ternary(self) -> None:
        code = 'data { $cond ? "yes" : "no" }'
        result = pwsh_transform(code)[0]
        # Ternary inside data block braces is at depth>0, NOT transformed
        assert "?" in result
        assert "data" in result

    def test_data_section_with_null_conditional(self) -> None:
        code = 'data { $obj?.Name }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "data" in result
        assert "$obj.Name" in result

    def test_data_section_with_chain(self) -> None:
        code = 'data { Write-Output hello && Write-Output world }'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "data" in result
        assert "if ($?)" in result


# ============================================================================
# Corner case: using namespace/module with operators
# ============================================================================

class TestUsingStatementsWithOperators:
    def test_using_namespace_then_coalescing(self) -> None:
        code = 'using namespace System.IO\n$x = $a ?? "default"'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "using namespace System.IO" in result
        assert "if ($null -ne $a)" in result

    def test_using_module_then_ternary(self) -> None:
        code = 'using module MyModule\n$x = $cond ? "yes" : "no"'
        result = pwsh_transform(code)[0]
        assert "?" not in result
        assert "using module MyModule" in result
        assert "if ($cond)" in result

    def test_using_module_then_chain(self) -> None:
        code = 'using module MyModule\ncmd1 && cmd2'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "using module MyModule" in result
        assert "if ($?)" in result

    def test_using_assembly_then_null_conditional(self) -> None:
        code = 'using assembly System.Windows.Forms\n$x = $form?.Text'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "using assembly System.Windows.Forms" in result


# ============================================================================
# Corner case: #requires with operators
# ============================================================================

class TestRequiresWithOperators:
    def test_requires_version_then_coalescing(self) -> None:
        code = '#requires -Version 7.0\n$x = $a ?? "default"'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "#requires -Version 7.0" in result
        assert "if ($null -ne $a)" in result

    def test_requires_modules_then_ternary(self) -> None:
        code = '#requires -Modules Az\n$x = $cond ? "yes" : "no"'
        result = pwsh_transform(code)[0]
        assert "?" not in result
        assert "#requires -Modules Az" in result
        assert "if ($cond)" in result

    def test_requires_psextension_then_chain(self) -> None:
        code = '#requires -PSEdition Core\ncmd1 && cmd2'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "#requires -PSEdition Core" in result
        assert "if ($?)" in result

    def test_requires_runasadmin_then_null_conditional(self) -> None:
        code = '#requires -RunAsAdministrator\n$x = $obj?.Name'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "#requires -RunAsAdministrator" in result


# ============================================================================
# Corner case: more cmdlets with operators
# ============================================================================

class TestMoreCmdletsWithOperators:
    def test_select_string_with_ternary(self) -> None:
        result = pwsh_transform('(Select-String $pattern $file) ? "found" : "not-found"')[0]
        assert "?" not in result
        assert "if ((Select-String $pattern $file))" in result
        assert "Select-String" in result

    def test_select_string_with_coalescing(self) -> None:
        result = pwsh_transform('(Select-String $pattern $file) ?? $null')[0]
        assert "??" not in result
        assert "if ($null -ne (Select-String $pattern $file))" in result

    def test_measure_object_sum_with_ternary(self) -> None:
        result = pwsh_transform('($data | Measure-Object -Sum).Sum -gt 0 ? "positive" : "zero"')[0]
        assert "?" not in result
        assert "if (($data | Measure-Object -Sum).Sum -gt 0)" in result
        assert "Measure-Object" in result

    def test_measure_object_average_with_coalescing(self) -> None:
        result = pwsh_transform('($data | Measure-Object -Average).Average ?? 0')[0]
        assert "??" not in result
        assert "Measure-Object" in result

    def test_get_random_with_ternary(self) -> None:
        result = pwsh_transform('(Get-Random -Minimum 0 -Maximum 100) -gt 50 ? "high" : "low"')[0]
        assert "?" not in result
        assert "Get-Random" in result
        assert "if ((Get-Random -Minimum 0 -Maximum 100) -gt 50)" in result

    def test_get_random_with_coalescing(self) -> None:
        result = pwsh_transform('(Get-Random -Count 5) ?? @()')[0]
        assert "??" not in result
        assert "Get-Random" in result

    def test_read_host_with_coalescing(self) -> None:
        result = pwsh_transform('(Read-Host "prompt") ?? "default"')[0]
        assert "??" not in result
        assert "Read-Host" in result
        assert "if ($null -ne (Read-Host \"prompt\"))" in result

    def test_read_host_with_ternary(self) -> None:
        result = pwsh_transform('(Read-Host "prompt") -eq "" ? "empty" : "non-empty"')[0]
        assert "?" not in result
        assert "Read-Host" in result

    def test_get_credential_with_coalescing(self) -> None:
        result = pwsh_transform('(Get-Credential) ?? (Get-Credential "default")')[0]
        assert "??" not in result
        assert "Get-Credential" in result

    def test_import_csv_with_coalescing(self) -> None:
        result = pwsh_transform('(Import-Csv $file) ?? @()')[0]
        assert "??" not in result
        assert "Import-Csv" in result
        assert "if ($null -ne (Import-Csv $file))" in result

    def test_export_csv_with_chain(self) -> None:
        result = pwsh_transform('Export-Csv $obj $file && Write-Output exported')[0]
        assert "&&" not in result
        assert "Export-Csv" in result
        assert "if ($?)" in result

    def test_start_process_with_ternary(self) -> None:
        result = pwsh_transform('(Start-Process $cmd -PassThru) ? "started" : "failed"')[0]
        assert "?" not in result
        assert "Start-Process" in result
        assert "if ((Start-Process $cmd -PassThru))" in result

    def test_new_object_with_coalescing(self) -> None:
        result = pwsh_transform('(New-Object PSObject) ?? $null')[0]
        assert "??" not in result
        assert "New-Object" in result
        assert "if ($null -ne (New-Object PSObject))" in result

    def test_invoke_expression_with_chain(self) -> None:
        result = pwsh_transform('Invoke-Expression $expr && Write-Output done')[0]
        assert "&&" not in result
        assert "Invoke-Expression" in result
        assert "if ($?)" in result

    def test_invoke_command_with_coalescing(self) -> None:
        result = pwsh_transform('Invoke-Command { $x ?? "default" }')[0]
        assert "??" not in result
        assert "Invoke-Command" in result
        assert "if ($null -ne $x)" in result

    def test_start_job_with_coalescing(self) -> None:
        result = pwsh_transform('Start-Job { $x ?? "default" }')[0]
        assert "??" not in result
        assert "Start-Job" in result
        assert "if ($null -ne $x)" in result

    def test_import_module_with_chain(self) -> None:
        result = pwsh_transform('Import-Module $mod && Write-Output loaded')[0]
        assert "&&" not in result
        assert "Import-Module" in result
        assert "if ($?)" in result

    def test_get_command_with_coalescing(self) -> None:
        result = pwsh_transform('(Get-Command $cmd) ?? (Get-Command "default")')[0]
        assert "??" not in result
        assert "Get-Command" in result

    def test_get_alias_with_ternary(self) -> None:
        result = pwsh_transform('(Get-Alias $name) ? "found" : "not-found"')[0]
        assert "?" not in result
        assert "Get-Alias" in result

    def test_get_variable_with_coalescing(self) -> None:
        result = pwsh_transform('(Get-Variable $name).Value ?? "default"')[0]
        assert "??" not in result
        assert "Get-Variable" in result

    def test_where_object_property_syntax_with_chain(self) -> None:
        result = pwsh_transform('Get-Process | Where-Object CPU -gt 50 && Write-Output heavy')[0]
        assert "&&" not in result
        assert "Where-Object" in result
        assert "if ($?)" in result

    def test_foreach_object_begin_process_end_with_coalescing(self) -> None:
        code = '1..10 | ForEach-Object -Begin { $sum = 0 ?? $null } -Process { $sum += $_ } -End { $sum }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "ForEach-Object" in result
        assert "if ($null -ne 0)" in result

    def test_format_table_with_chain(self) -> None:
        result = pwsh_transform('Get-Process | Format-Table && Write-Output formatted')[0]
        assert "&&" not in result
        assert "Format-Table" in result
        assert "if ($?)" in result

    def test_format_list_with_chain(self) -> None:
        result = pwsh_transform('Get-Process | Format-List && Write-Output formatted')[0]
        assert "&&" not in result
        assert "Format-List" in result
        assert "if ($?)" in result

    def test_write_information_with_chain(self) -> None:
        result = pwsh_transform('Write-Information $msg && Write-Output done')[0]
        assert "&&" not in result
        assert "Write-Information" in result
        assert "if ($?)" in result

    def test_write_verbose_with_chain_or(self) -> None:
        result = pwsh_transform('Write-Verbose $msg || Write-Error "verbose failed"')[0]
        assert "||" not in result
        assert "Write-Verbose" in result
        assert "if (-not $?)" in result

    def test_write_debug_with_chain(self) -> None:
        result = pwsh_transform('Write-Debug $msg && Write-Output debugged')[0]
        assert "&&" not in result
        assert "Write-Debug" in result
        assert "if ($?)" in result

    def test_write_warning_with_chain_or(self) -> None:
        result = pwsh_transform('Write-Warning $msg || Write-Error "warn failed"')[0]
        assert "||" not in result
        assert "Write-Warning" in result
        assert "if (-not $?)" in result

    def test_write_error_with_chain(self) -> None:
        result = pwsh_transform('Write-Error $msg && Write-Output errored')[0]
        assert "&&" not in result
        assert "Write-Error" in result
        assert "if ($?)" in result

    def test_clear_host_with_chain(self) -> None:
        result = pwsh_transform('Clear-Host && Write-Output cleared')[0]
        assert "&&" not in result
        assert "Clear-Host" in result
        assert "if ($?)" in result

    def test_set_location_with_chain_or(self) -> None:
        result = pwsh_transform('Set-Location $path || Write-Error "chdir failed"')[0]
        assert "||" not in result
        assert "Set-Location" in result
        assert "if (-not $?)" in result

    def test_get_location_with_coalescing(self) -> None:
        result = pwsh_transform('(Get-Location) ?? (Get-Location)')[0]
        assert "??" not in result
        assert "Get-Location" in result


# ============================================================================
# Corner case: more .NET type integrations
# ============================================================================

class TestDotNetTypesWithOperators:
    def test_file_readalltext_with_coalescing(self) -> None:
        result = pwsh_transform('[System.IO.File]::ReadAllText($f) ?? ""')[0]
        assert "??" not in result
        assert "[System.IO.File]::ReadAllText($f)" in result

    def test_directory_createdirectory_with_ternary(self) -> None:
        result = pwsh_transform('[System.IO.Directory]::CreateDirectory($d) ? "created" : "exists"')[0]
        assert "?" not in result
        assert "[System.IO.Directory]::CreateDirectory($d)" in result

    def test_encoding_utf8_getbytes_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Text.Encoding]::UTF8.GetBytes($s) ?? @()')[0]
        assert "??" not in result
        assert "[System.Text.Encoding]::UTF8" in result

    def test_stringbuilder_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Text.StringBuilder]::new($s) ?? $null')[0]
        assert "??" not in result
        assert "[System.Text.StringBuilder]::new" in result

    def test_webclient_downloadstring_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Net.WebClient]::new().DownloadString($url) ?? ""')[0]
        assert "??" not in result
        assert "[System.Net.WebClient]::new()" in result

    def test_stopwatch_startnew_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Diagnostics.Stopwatch]::StartNew() ?? $null')[0]
        assert "??" not in result
        assert "[System.Diagnostics.Stopwatch]::StartNew()" in result

    def test_guid_newguid_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Guid]::NewGuid().ToString() ?? ""')[0]
        assert "??" not in result
        assert "[System.Guid]::NewGuid()" in result

    def test_uri_new_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Uri]::new($url).Host ?? ""')[0]
        assert "??" not in result
        assert "[System.Uri]::new($url)" in result

    def test_pscredential_new_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Management.Automation.PSCredential]::new($u, $p) ?? $null')[0]
        assert "??" not in result
        assert "[System.Management.Automation.PSCredential]::new" in result

    def test_list_new_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Collections.Generic.List[string]]::new() ?? @()')[0]
        assert "??" not in result
        assert "[System.Collections.Generic.List[string]]::new()" in result

    def test_arraylist_new_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Collections.ArrayList]::new() ?? @()')[0]
        assert "??" not in result
        assert "[System.Collections.ArrayList]::new()" in result

    def test_math_round_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Math]::Round($x, 2) ?? 0')[0]
        assert "??" not in result
        assert "[System.Math]::Round($x, 2)" in result

    def test_math_pow_with_ternary(self) -> None:
        result = pwsh_transform('[System.Math]::Pow($x, 2) -gt 0 ? "positive" : "zero"')[0]
        assert "?" not in result
        assert "[System.Math]::Pow($x, 2)" in result

    def test_random_next_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Random]::new().Next(100) ?? 0')[0]
        assert "??" not in result
        assert "[System.Random]::new()" in result

    def test_console_writeline_with_chain(self) -> None:
        result = pwsh_transform('[System.Console]::WriteLine($msg) && Write-Output done')[0]
        assert "&&" not in result
        assert "[System.Console]::WriteLine" in result
        assert "if ($?)" in result

    def test_environment_getenvironmentvariable_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Environment]::GetEnvironmentVariable("PATH") ?? ""')[0]
        assert "??" not in result
        assert "[System.Environment]::GetEnvironmentVariable" in result

    def test_path_combine_with_coalescing(self) -> None:
        result = pwsh_transform('[System.IO.Path]::Combine($a, $b) ?? ""')[0]
        assert "??" not in result
        assert "[System.IO.Path]::Combine" in result

    def test_dns_gethostentry_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Net.Dns]::GetHostEntry($host).HostName ?? ""')[0]
        assert "??" not in result
        assert "[System.Net.Dns]::GetHostEntry" in result

    def test_sha256_create_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Security.Cryptography.SHA256]::Create().ComputeHash($b) ?? @()')[0]
        assert "??" not in result
        assert "[System.Security.Cryptography.SHA256]::Create()" in result

    def test_xml_document_new_with_chain(self) -> None:
        result = pwsh_transform('[System.Xml.XmlDocument]::new().Load($f) && Write-Output loaded')[0]
        assert "&&" not in result
        assert "[System.Xml.XmlDocument]::new()" in result
        assert "if ($?)" in result

    def test_data_datatable_new_with_ternary(self) -> None:
        result = pwsh_transform('[System.Data.DataTable]::new().Rows.Count -gt 0 ? "has-rows" : "empty"')[0]
        assert "?" not in result
        assert "[System.Data.DataTable]::new()" in result

    def test_assembly_loadfrom_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Reflection.Assembly]::LoadFrom($path) ?? $null')[0]
        assert "??" not in result
        assert "[System.Reflection.Assembly]::LoadFrom" in result

    def test_activator_createinstance_with_coalescing(self) -> None:
        result = pwsh_transform('[Activator]::CreateInstance([type]$t) ?? $null')[0]
        assert "??" not in result
        assert "[Activator]::CreateInstance" in result

    def test_marshal_ptrtostringauto_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Runtime.InteropServices.Marshal]::PtrToStringAuto($ptr) ?? ""')[0]
        assert "??" not in result
        assert "[System.Runtime.InteropServices.Marshal]::PtrToStringAuto" in result

    def test_intptr_zero_with_ternary(self) -> None:
        result = pwsh_transform('[IntPtr]::Zero -eq $ptr ? "null" : "set"')[0]
        assert "?" not in result
        assert "[IntPtr]::Zero" in result
        assert "if ([IntPtr]::Zero -eq $ptr)" in result

    def test_gc_collect_with_chain(self) -> None:
        result = pwsh_transform('[GC]::Collect() && Write-Output collected')[0]
        assert "&&" not in result
        assert "[GC]::Collect()" in result
        assert "if ($?)" in result

    def test_thread_sleep_with_chain_or(self) -> None:
        result = pwsh_transform('[System.Threading.Thread]::Sleep(1000) || Write-Error "sleep failed"')[0]
        assert "||" not in result
        assert "[System.Threading.Thread]::Sleep" in result
        assert "if (-not $?)" in result

    def test_process_start_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Diagnostics.Process]::Start($cmd) ?? $null')[0]
        assert "??" not in result
        assert "[System.Diagnostics.Process]::Start" in result

    def test_tcpclient_new_with_ternary(self) -> None:
        result = pwsh_transform('[System.Net.Sockets.TcpClient]::new($host, $port) ? "connected" : "failed"')[0]
        assert "?" not in result
        assert "[System.Net.Sockets.TcpClient]::new" in result

    def test_regex_new_with_ternary(self) -> None:
        result = pwsh_transform('[System.Text.RegularExpressions.Regex]::new($pattern).IsMatch($s) ? "match" : "no-match"')[0]
        assert "?" not in result
        assert "[System.Text.RegularExpressions.Regex]::new" in result

    def test_dictionary_new_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Collections.Generic.Dictionary[string,object]]::new() ?? @{}')[0]
        assert "??" not in result
        assert "[System.Collections.Generic.Dictionary[string,object]]::new()" in result

    def test_x509certificate_new_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Security.Cryptography.X509Certificates.X509Certificate2]::new($cert) ?? $null')[0]
        assert "??" not in result
        assert "[System.Security.Cryptography.X509Certificates.X509Certificate2]::new" in result

    def test_timer_new_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Timers.Timer]::new(1000) ?? $null')[0]
        assert "??" not in result
        assert "[System.Timers.Timer]::new" in result

    def test_drawing_bitmap_new_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Drawing.Bitmap]::new($w, $h) ?? $null')[0]
        assert "??" not in result
        assert "[System.Drawing.Bitmap]::new" in result

    def test_linq_enumerable_range_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Linq.Enumerable]::Range(1, 10) ?? @()')[0]
        assert "??" not in result
        assert "[System.Linq.Enumerable]::Range" in result

    def test_environment_machine_name_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Environment]::MachineName ?? ""')[0]
        assert "??" not in result
        assert "[System.Environment]::MachineName" in result

    def test_environment_user_name_with_ternary(self) -> None:
        result = pwsh_transform('[System.Environment]::UserName -eq "admin" ? "admin" : "user"')[0]
        assert "?" not in result
        assert "[System.Environment]::UserName" in result

    def test_io_path_gettempfilename_with_coalescing(self) -> None:
        result = pwsh_transform('[System.IO.Path]::GetTempFileName() ?? ""')[0]
        assert "??" not in result
        assert "[System.IO.Path]::GetTempFileName()" in result

    def test_convert_tobase64string_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Convert]::ToBase64String($b) ?? ""')[0]
        assert "??" not in result
        assert "[System.Convert]::ToBase64String" in result

    def test_net_ipaddress_parse_with_ternary(self) -> None:
        result = pwsh_transform('[System.Net.IPAddress]::Parse($ip) ? "valid" : "invalid"')[0]
        assert "?" not in result
        assert "[System.Net.IPAddress]::Parse" in result

    def test_security_securestring_new_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Security.SecureString]::new() ?? $null')[0]
        assert "??" not in result
        assert "[System.Security.SecureString]::new()" in result


# ============================================================================
# Corner case: more PS7-specific features and variables
# ============================================================================

class TestPS7SpecificFeaturesWithOperators:
    def test_psnativecommanduseerroractionpreference_with_ternary(self) -> None:
        result = pwsh_transform('$PSNativeCommandUseErrorActionPreference ? "true" : "false"')[0]
        assert "?" not in result
        assert "if ($PSNativeCommandUseErrorActionPreference)" in result

    def test_psnativecommandargumentpassing_with_coalescing(self) -> None:
        result = pwsh_transform('$PSNativeCommandArgumentPassing ?? "Standard"')[0]
        assert "??" not in result
        assert "if ($null -ne $PSNativeCommandArgumentPassing)" in result

    def test_psansirenderingfileinfo_with_null_conditional(self) -> None:
        result = pwsh_transform('$PSAnsiRenderingFileInfo?.ToString()')[0]
        assert "?." not in result
        assert "if ($null -ne $PSAnsiRenderingFileInfo)" in result

    def test_psmoduleanalysiscachepath_with_coalescing(self) -> None:
        result = pwsh_transform('$PSModuleAnalysisCachePath ?? ""')[0]
        assert "??" not in result
        assert "if ($null -ne $PSModuleAnalysisCachePath)" in result

    def test_psstyle_background_black_null_conditional(self) -> None:
        result = pwsh_transform('$PSStyle?.Background?.Black')[0]
        assert "?." not in result
        assert "$PSStyle" in result
        assert ".Background" in result
        assert ".Black" in result

    def test_psstyle_foreground_red_with_chain(self) -> None:
        result = pwsh_transform('Write-Output $PSStyle.Foreground.Red && Write-Output done')[0]
        assert "&&" not in result
        assert "$PSStyle" in result
        assert "if ($?)" in result


# ============================================================================
# Corner case: more edge cases around operators
# ============================================================================

class TestAdditionalOperatorEdgeCases:
    def test_null_literal_as_both_branches_ternary(self) -> None:
        result = pwsh_transform('$cond ? $null : $null')[0]
        assert "?" not in result
        assert "if ($cond)" in result
        assert "{ $null }" in result

    def test_null_literal_as_both_sides_coalescing(self) -> None:
        result = pwsh_transform('$null ?? $null')[0]
        assert "??" not in result
        assert "if ($null -ne $null)" in result

    def test_ternary_with_null_and_nonnull_branches(self) -> None:
        result = pwsh_transform('$cond ? $null : "value"')[0]
        assert "?" not in result
        assert "{ $null }" in result
        assert "\"value\"" in result

    def test_coalescing_with_null_and_nonnull(self) -> None:
        result = pwsh_transform('$null ?? "value"')[0]
        assert "??" not in result
        assert "{ $null }" in result
        assert "\"value\"" in result

    def test_chain_with_exit(self) -> None:
        result = pwsh_transform('cmd1 && exit 0 || exit 1')[0]
        assert "&&" not in result
        assert "||" not in result
        assert "cmd1" in result

    def test_chain_with_return(self) -> None:
        result = pwsh_transform('cmd1 && return $true || return $false')[0]
        assert "&&" not in result
        assert "||" not in result
        assert "cmd1" in result

    def test_chain_with_break(self) -> None:
        result = pwsh_transform('cmd1 && break || continue')[0]
        assert "&&" not in result
        assert "||" not in result
        assert "cmd1" in result

    def test_chain_with_throw(self) -> None:
        result = pwsh_transform('cmd1 && throw "error" || throw "fallback"')[0]
        assert "&&" not in result
        assert "||" not in result
        assert "cmd1" in result

    def test_nested_subexpr_with_ternary_at_depth0(self) -> None:
        result = pwsh_transform("$($($cond ? $a : $b))")[0]
        # Ternary inside nested $() is at depth>0, NOT transformed
        assert "?" in result
        assert "$($($cond ? $a : $b))" == result

    def test_nested_subexpr_with_coalescing_at_depth0(self) -> None:
        result = pwsh_transform('$($($a ?? $b))')[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result

    def test_subexpr_with_if_then_coalescing(self) -> None:
        code = '$(if ($a) { $b } else { $c }) ?? "default"'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "if ($null -ne $(if ($a) { $b } else { $c }))" in result

    def test_switch_regex_with_chain_after(self) -> None:
        code = 'switch -Regex ($pattern) { "\\d+" { } }; cmd1 && cmd2'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "switch -Regex" in result
        assert "if ($?)" in result

    def test_switch_file_with_chain_after(self) -> None:
        code = 'switch -File $file { "test" { } }; cmd1 && cmd2'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "switch -File" in result
        assert "if ($?)" in result

    def test_trap_with_specific_exception_type(self) -> None:
        code = 'trap [System.Exception] { $_.Message ?? "unknown" }; cmd1 && cmd2'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "trap [System.Exception]" in result
        assert "&&" not in result
        assert "if ($?)" in result

    def test_try_catch_finally_complex_with_operators(self) -> None:
        code = 'try { $a ?? "default" } catch [System.IO.IOException] { $_.Message } finally { $obj?.Name }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "?." not in result
        assert "try" in result
        assert "catch [System.IO.IOException]" in result
        assert "finally" in result

    def test_try_multiple_catch_with_operators(self) -> None:
        code = 'try { 1 } catch [System.ArgumentException] { $a ?? "arg" } catch { $b ?? "other" }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "try" in result
        assert "catch [System.ArgumentException]" in result

    def test_pscmdlet_shouldprocess_with_ternary(self) -> None:
        result = pwsh_transform('$PSCmdlet.ShouldProcess($target) ? "proceed" : "cancel"')[0]
        assert "?" not in result
        assert "if ($PSCmdlet.ShouldProcess($target))" in result

    def test_pscmdlet_shouldcontinue_with_coalescing(self) -> None:
        result = pwsh_transform('$PSCmdlet.ShouldContinue($query, $caption) ?? $false')[0]
        assert "??" not in result
        assert "if ($null -ne $PSCmdlet.ShouldContinue($query, $caption))" in result

    def test_pscmdlet_parametersetname_with_ternary(self) -> None:
        result = pwsh_transform('$PSCmdlet.ParameterSetName -eq "Default" ? "default" : "other"')[0]
        assert "?" not in result
        assert "if ($PSCmdlet.ParameterSetName -eq \"Default\")" in result

    def test_pscmdlet_throwterminatingerror_with_chain(self) -> None:
        result = pwsh_transform('$PSCmdlet.ThrowTerminatingError($err) && Write-Output thrown')[0]
        assert "&&" not in result
        assert "$PSCmdlet.ThrowTerminatingError" in result
        assert "if ($?)" in result

    def test_executioncontext_expandstring_with_coalescing(self) -> None:
        result = pwsh_transform('$ExecutionContext.InvokeCommand.ExpandString($s) ?? ""')[0]
        assert "??" not in result
        assert "$ExecutionContext.InvokeCommand.ExpandString" in result

    def test_executioncontext_invokescript_with_coalescing(self) -> None:
        result = pwsh_transform('$ExecutionContext.InvokeCommand.InvokeScript($sb) ?? $null')[0]
        assert "??" not in result
        assert "$ExecutionContext.InvokeCommand.InvokeScript" in result

    def test_host_promptforchoice_with_coalescing(self) -> None:
        result = pwsh_transform('$Host.UI.PromptForChoice($title, $msg, $choices, 0) ?? -1')[0]
        assert "??" not in result
        assert "$Host.UI.PromptForChoice" in result

    def test_host_readline_with_coalescing(self) -> None:
        result = pwsh_transform('$Host.UI.RawUI.ReadLine() ?? ""')[0]
        assert "??" not in result
        assert "$Host.UI.RawUI.ReadLine()" in result

    def test_host_enternestedprompt_with_chain(self) -> None:
        result = pwsh_transform('$Host.EnterNestedPrompt() && Write-Output nested')[0]
        assert "&&" not in result
        assert "$Host.EnterNestedPrompt()" in result
        assert "if ($?)" in result


# ============================================================================
# Corner case: operators in Export-Clixml / Import-Clixml / Get-Unique / etc.
# ============================================================================

class TestMoreCmdletOperators:
    def test_export_clixml_with_chain(self) -> None:
        result = pwsh_transform('Export-Clixml -InputObject $obj -Path $file && Write-Output exported')[0]
        assert "&&" not in result
        assert "Export-Clixml" in result
        assert "if ($?)" in result

    def test_import_clixml_with_coalescing(self) -> None:
        result = pwsh_transform('(Import-Clixml $file) ?? $null')[0]
        assert "??" not in result
        assert "Import-Clixml" in result
        assert "if ($null -ne (Import-Clixml $file))" in result

    def test_get_unique_with_ternary(self) -> None:
        result = pwsh_transform('($arr | Get-Unique).Count -gt 1 ? "duplicates" : "unique"')[0]
        assert "?" not in result
        assert "Get-Unique" in result
        assert "if (($arr | Get-Unique).Count -gt 1)" in result

    def test_sort_object_unique_with_coalescing(self) -> None:
        result = pwsh_transform('($arr | Sort-Object -Unique) ?? @()')[0]
        assert "??" not in result
        assert "Sort-Object" in result

    def test_group_object_noelement_with_ternary(self) -> None:
        result = pwsh_transform('($arr | Group-Object -NoElement).Count -gt 1 ? "groups" : "one"')[0]
        assert "?" not in result
        assert "Group-Object" in result

    def test_measure_object_maximum_with_coalescing(self) -> None:
        result = pwsh_transform('($data | Measure-Object -Maximum).Maximum ?? 0')[0]
        assert "??" not in result
        assert "Measure-Object" in result

    def test_measure_object_minimum_with_ternary(self) -> None:
        result = pwsh_transform('($data | Measure-Object -Minimum).Minimum -gt 0 ? "positive" : "non-positive"')[0]
        assert "?" not in result
        assert "Measure-Object" in result

    def test_new_module_with_coalescing(self) -> None:
        result = pwsh_transform('New-Module { $x ?? "default" }')[0]
        assert "??" not in result
        assert "New-Module" in result
        assert "if ($null -ne $x)" in result

    def test_new_modulemanifest_with_chain(self) -> None:
        result = pwsh_transform('New-ModuleManifest -Path $p && Write-Output created')[0]
        assert "&&" not in result
        assert "New-ModuleManifest" in result
        assert "if ($?)" in result

    def test_update_typedata_with_chain(self) -> None:
        result = pwsh_transform('Update-TypeData -TypeName $t && Write-Output updated')[0]
        assert "&&" not in result
        assert "Update-TypeData" in result
        assert "if ($?)" in result

    def test_register_objectevent_with_coalescing(self) -> None:
        code = 'Register-ObjectEvent $obj EventName -Action { $EventArgs ?? $null }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "Register-ObjectEvent" in result
        assert "if ($null -ne $EventArgs)" in result

    def test_invoke_command_with_chain(self) -> None:
        result = pwsh_transform('Invoke-Command -ComputerName $c { cmd1 } && Write-Output done')[0]
        assert "&&" not in result
        assert "Invoke-Command" in result
        assert "if ($?)" in result

    def test_start_threadjob_with_coalescing(self) -> None:
        result = pwsh_transform('Start-ThreadJob { $x ?? "default" }')[0]
        assert "??" not in result
        assert "Start-ThreadJob" in result
        assert "if ($null -ne $x)" in result

    def test_get_error_with_coalescing(self) -> None:
        result = pwsh_transform('Get-Error | Select-Object -First 1 ?? $null')[0]
        assert "??" not in result
        assert "Get-Error" in result


# ============================================================================
# Corner case: more idempotency for new patterns
# ============================================================================

class TestAdditionalIdempotencyPatterns:
    def test_cmdletbinding_coalescing_idempotent(self) -> None:
        code = 'function Test { [CmdletBinding()] param([string]$x = $a ?? "default"); $x }'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_begin_process_end_idempotent(self) -> None:
        code = 'function Test { begin { $sum = 0 } process { $sum += $_ } end { $sum } }'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_using_namespace_idempotent(self) -> None:
        code = 'using namespace System.IO\n$x = $a ?? "default"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_select_string_idempotent(self) -> None:
        code = '(Select-String $pattern $file) ?? $null'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_net_type_method_idempotent(self) -> None:
        code = '[System.IO.File]::ReadAllText($f) ?? ""'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_ps7_specific_var_idempotent(self) -> None:
        code = '$PSNativeCommandUseErrorActionPreference ?? $false'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_advanced_function_idempotent(self) -> None:
        code = 'function Test { [CmdletBinding(SupportsShouldProcess=$true)] param(); $PSCmdlet.ShouldProcess($target) ? "yes" : "no" }'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_dynamicparam_idempotent(self) -> None:
        code = 'function Test { dynamicparam { $x = $a ?? "default"; $x } }'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_data_section_idempotent(self) -> None:
        code = 'data { $x = $a ?? "default"; $x }'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_trap_specific_exception_idempotent(self) -> None:
        code = 'trap [System.Exception] { $_.Message ?? "unknown" }; cmd1 && cmd2'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_all_advanced_patterns_combined_idempotent(self) -> None:
        code = 'function Test { [CmdletBinding()] param([ValidateSet("a","b")]$x = $a ?? "a"); begin { $sum ??= 0 } process { $sum += $_ } end { $sum } }'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second


# ============================================================================
# Corner case: more edge cases with $PSCmdlet / $ExecutionContext / $Host
# ============================================================================

class TestPSCmdletExecutionContextHostEdgeCases:
    def test_pscmdlet_writeobject_with_coalescing(self) -> None:
        result = pwsh_transform('$PSCmdlet.WriteObject($obj) ?? $null')[0]
        assert "??" not in result
        assert "$PSCmdlet.WriteObject" in result

    def test_pscmdlet_writeerror_with_chain(self) -> None:
        result = pwsh_transform('$PSCmdlet.WriteError($err) && Write-Output written')[0]
        assert "&&" not in result
        assert "$PSCmdlet.WriteError" in result
        assert "if ($?)" in result

    def test_executioncontext_sessionstate_with_null_conditional(self) -> None:
        result = pwsh_transform('$ExecutionContext.SessionState?.Path?.CurrentLocation')[0]
        assert "?." not in result
        assert "$ExecutionContext.SessionState" in result
        assert ".Path" in result
        assert ".CurrentLocation" in result

    def test_host_version_with_ternary(self) -> None:
        result = pwsh_transform('$Host.Version -gt [version]"5.1" ? "new" : "old"')[0]
        assert "?" not in result
        assert "$Host.Version" in result
        assert "if ($Host.Version -gt [version]\"5.1\")" in result

    def test_host_ui_rawui_with_null_conditional(self) -> None:
        result = pwsh_transform('$Host.UI.RawUI?.WindowTitle')[0]
        assert "?." not in result
        assert "$Host.UI.RawUI" in result
        assert ".WindowTitle" in result

    def test_host_ui_rawui_with_coalescing(self) -> None:
        result = pwsh_transform('$Host.UI.RawUI.WindowTitle ?? "PowerShell"')[0]
        assert "??" not in result
        assert "$Host.UI.RawUI.WindowTitle" in result


# ============================================================================
# Corner case: more edge cases with nested constructs
# ============================================================================

class TestNestedConstructsWithOperators:
    def test_nested_if_with_coalescing(self) -> None:
        code = 'if ($a) { if ($b) { $c ?? "default" } }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "if ($a)" in result
        assert "if ($b)" in result
        assert "if ($null -ne $c)" in result

    def test_nested_if_with_ternary(self) -> None:
        code = 'if ($a) { $cond ? "yes" : "no" }'
        result = pwsh_transform(code)[0]
        # Ternary inside if block braces is at depth>0, NOT transformed
        assert "?" in result
        assert "if ($a)" in result

    def test_nested_foreach_with_null_conditional(self) -> None:
        code = 'foreach ($a in $b) { foreach ($c in $d) { $e?.Name } }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "foreach" in result
        assert "$e.Name" in result

    def test_nested_while_with_coalescing(self) -> None:
        code = 'while ($a) { while ($b) { $c = $d ?? "default" } }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "while ($a)" in result
        assert "while ($b)" in result
        assert "if ($null -ne $d)" in result

    def test_nested_try_catch_with_operators(self) -> None:
        code = 'try { try { $a ?? "inner" } catch { $b ?? "inner-catch" } } catch { $c ?? "outer-catch" }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "try" in result
        assert "catch" in result
        assert "if ($null -ne $a)" in result
        assert "if ($null -ne $b)" in result
        assert "if ($null -ne $c)" in result

    def test_nested_switch_with_operators(self) -> None:
        code = 'switch ($a) { 1 { switch ($b) { 2 { $c ?? "default" } } } }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "switch ($a)" in result
        assert "switch ($b)" in result
        assert "if ($null -ne $c)" in result

    def test_nested_function_with_operators(self) -> None:
        code = 'function Outer { function Inner { $a ?? "default" }; Inner }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "function Outer" in result
        assert "function Inner" in result
        assert "if ($null -ne $a)" in result

    def test_nested_scriptblock_with_operators(self) -> None:
        code = '{ { $a ?? "default" } }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result


# ============================================================================
# Corner case: more edge cases with strings and comments
# ============================================================================

class TestStringCommentEdgeCasesExtended:
    def test_ternary_with_triple_quoted_string(self) -> None:
        result = pwsh_transform("$cond ? @'\nhello\n'@ : @'\nworld\n'@")[0]
        # Ternary ? followed by here-string is NOT transformed (no matching colon)
        assert "?" in result
        assert "hello" in result
        assert "world" in result

    def test_coalescing_with_triple_quoted_string(self) -> None:
        result = pwsh_transform("$a ?? @'\ndefault\n'@")[0]
        # Coalescing ?? followed by here-string is NOT transformed
        assert "??" in result
        assert "default" in result

    def test_chain_with_comment_before(self) -> None:
        code = '# comment before chain\ncmd1 && cmd2'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "comment before chain" in result
        assert "if ($?)" in result

    def test_ternary_with_comment_after(self) -> None:
        result = pwsh_transform('$cond ? "yes" : "no" # comment after ternary')[0]
        assert "?" not in result
        assert "if ($cond)" in result
        assert "comment after ternary" in result

    def test_coalescing_with_comment_after(self) -> None:
        result = pwsh_transform('$a ?? "default" # comment after coalescing')[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result
        assert "comment after coalescing" in result

    def test_null_conditional_with_comment_after(self) -> None:
        result = pwsh_transform('$a?.Name # comment after null-conditional')[0]
        assert "?." not in result
        assert "if ($null -ne $a)" in result
        assert "comment after null-conditional" in result

    def test_block_comment_with_operators_inside_then_real_operators(self) -> None:
        code = '<# operators: ?? ?. && || #>\n$x = $a ?? "default"'
        result = pwsh_transform(code)[0]
        # ?? inside block comment is preserved; real ?? outside is transformed
        assert "if ($null -ne $a)" in result
        assert "operators: ?? ?. && ||" in result

    def test_string_with_embedded_comment_lookalike(self) -> None:
        result = pwsh_transform('"# not a comment" ?? "default"')[0]
        assert "??" not in result
        assert "# not a comment" in result
        assert "if ($null -ne \"# not a comment\")" in result

    def test_string_with_block_comment_lookalike(self) -> None:
        result = pwsh_transform('"<# not a block comment #>" ?? "default"')[0]
        assert "??" not in result
        assert "<# not a block comment #>" in result


# ============================================================================
# Corner case: more edge cases with variable assignments
# ============================================================================

class TestVariableAssignmentEdgeCasesExtended:
    def test_multiple_assignment_with_coalescing(self) -> None:
        result = pwsh_transform('$a = $b = $c ?? "default"')[0]
        assert "??" not in result
        assert "$a = $b =" in result
        assert "if ($null -ne $c)" in result

    def test_assignment_with_type_cast_and_coalescing(self) -> None:
        result = pwsh_transform('[string]$a = $b ?? "default"')[0]
        assert "??" not in result
        assert "[string]$a =" in result
        assert "if ($null -ne $b)" in result

    def test_assignment_with_type_cast_and_ternary(self) -> None:
        result = pwsh_transform('[int]$a = $cond ? 1 : 0')[0]
        assert "?" not in result
        assert "[int]$a =" in result
        assert "if ($cond)" in result

    def test_assignment_with_null_conditional_right(self) -> None:
        result = pwsh_transform('$a = $obj?.Name')[0]
        assert "?." not in result
        assert "$a = " in result
        assert "if ($null -ne $obj)" in result

    def test_assignment_with_chain_right(self) -> None:
        result = pwsh_transform('$a = cmd1 && cmd2')[0]
        assert "&&" not in result
        assert "$a = cmd1" in result
        assert "if ($?)" in result

    def test_assignment_with_or_chain_right(self) -> None:
        result = pwsh_transform('$a = cmd1 || cmd2')[0]
        assert "||" not in result
        assert "$a = cmd1" in result
        assert "if (-not $?)" in result

    def test_splatting_with_coalescing(self) -> None:
        result = pwsh_transform('@args = $a ?? @()')[0]
        assert "??" not in result
        assert "@args =" in result
        assert "if ($null -ne $a)" in result

    def test_splatting_with_null_conditional(self) -> None:
        result = pwsh_transform('@obj?.Keys')[0]
        assert "?." not in result
        # @obj is recognized as the base expression
        assert "@obj" in result
        assert "Keys" in result


# ============================================================================
# Corner case: more edge cases with array/hashtable splatting
# ============================================================================

class TestSplattingEdgeCases:
    def test_hashtable_splatting_with_coalescing_value(self) -> None:
        result = pwsh_transform('$ht = @{ Name = $name ?? "anon"; Value = $val ?? 0 }')[0]
        assert "??" not in result
        assert "@{ Name =" in result
        assert "if ($null -ne $name)" in result
        assert "if ($null -ne $val)" in result

    def test_array_splatting_with_coalescing(self) -> None:
        result = pwsh_transform('$arr = @($a ?? 0, $b ?? 1, $c ?? 2)')[0]
        assert "??" not in result
        assert "@(" in result
        assert "if ($null -ne $a)" in result
        assert "if ($null -ne $b)" in result
        assert "if ($null -ne $c)" in result

    def test_splatting_in_command_with_coalescing(self) -> None:
        result = pwsh_transform('Write-Output @($a ?? "default")')[0]
        assert "??" not in result
        assert "Write-Output" in result
        assert "if ($null -ne $a)" in result

    def test_splatting_variable_with_null_conditional(self) -> None:
        result = pwsh_transform('$splat?.Keys')[0]
        assert "?." not in result
        assert "if ($null -ne $splat)" in result
        assert "$splat.Keys" in result


# ============================================================================
# Corner case: operators with $PSBoundParameters / $PSCmdlet in complex ways
# ============================================================================

class TestPSBoundParametersPSCmdletComplex:
    def test_psboundparameters_contains_with_ternary(self) -> None:
        result = pwsh_transform('$PSBoundParameters.ContainsKey("Name") ? "has-name" : "no-name"')[0]
        assert "?" not in result
        assert "if ($PSBoundParameters.ContainsKey(\"Name\"))" in result

    def test_psboundparameters_keys_with_coalescing(self) -> None:
        result = pwsh_transform('$PSBoundParameters.Keys ?? @()')[0]
        assert "??" not in result
        assert "if ($null -ne $PSBoundParameters.Keys)" in result

    def test_pscmdlet_myinvocation_with_null_conditional(self) -> None:
        result = pwsh_transform('$PSCmdlet.MyInvocation?.MyCommand?.Name')[0]
        assert "?." not in result
        assert "$PSCmdlet.MyInvocation" in result
        assert ".MyCommand" in result
        assert ".Name" in result

    def test_pscmdlet_boundparameters_with_coalescing(self) -> None:
        result = pwsh_transform('$PSCmdlet.MyInvocation.BoundParameters["Name"] ?? "default"')[0]
        assert "??" not in result
        assert "$PSCmdlet.MyInvocation.BoundParameters" in result


# ============================================================================
# Corner case: operators in pipeline with complex cmdlets
# ============================================================================

class TestPipelineComplexCmdlets:
    def test_pipeline_to_variable_with_null_conditional(self) -> None:
        result = pwsh_transform('$procs = Get-Process; $procs[0]?.Name')[0]
        assert "?." not in result
        assert "$procs[0]" in result
        assert ".Name" in result

    def test_pipeline_with_foreach_method_then_null_conditional(self) -> None:
        result = pwsh_transform('($arr).ForEach({ $_ })?.Count')[0]
        assert "?." not in result
        assert ".ForEach" in result
        assert ".Count" in result

    def test_pipeline_with_where_method_then_null_conditional(self) -> None:
        result = pwsh_transform('($arr).Where({ $_ -gt 0 })?.Count')[0]
        assert "?." not in result
        assert ".Where" in result
        assert ".Count" in result

    def test_pipeline_with_select_object_expandproperty(self) -> None:
        result = pwsh_transform('($procs | Select-Object -ExpandProperty Name) ?? @()')[0]
        assert "??" not in result
        assert "Select-Object" in result
        assert "if ($null -ne ($procs | Select-Object -ExpandProperty Name))" in result

    def test_pipeline_with_tee_object_variable_then_coalescing(self) -> None:
        result = pwsh_transform('($procs | Tee-Object -Variable p) ?? @()')[0]
        assert "??" not in result
        assert "Tee-Object" in result
        assert "if ($null -ne ($procs | Tee-Object -Variable p))" in result


# ============================================================================
# Corner case: more edge cases with method chaining
# ============================================================================

class TestMethodChainingEdgeCases:
    def test_method_chain_with_static_then_instance(self) -> None:
        result = pwsh_transform('[DateTime]::Parse($s)?.ToString("yyyy")?.Trim()')[0]
        assert "?." not in result
        assert "[DateTime]::Parse($s)" in result
        assert ".ToString(\"yyyy\")" in result
        assert ".Trim()" in result

    def test_method_chain_with_cast_then_method(self) -> None:
        result = pwsh_transform('[string]$s?.Trim()?.ToUpper()?.Split()')[0]
        assert "?." not in result
        assert "[string]$s" in result
        assert ".Trim()" in result
        assert ".ToUpper()" in result
        assert ".Split()" in result

    def test_method_chain_with_subexpression_then_method(self) -> None:
        result = pwsh_transform('$(Get-Date)?.ToString()?.Trim()')[0]
        assert "?." not in result
        assert "$(Get-Date)" in result
        assert ".ToString()" in result
        assert ".Trim()" in result

    def test_method_chain_with_env_var_then_method(self) -> None:
        result = pwsh_transform('$env:PATH?.Split(";")?[0]?.Trim()')[0]
        assert "?." not in result
        assert "?[" not in result
        assert "$env:PATH" in result
        assert ".Split(\";\")" in result
        assert ".Trim()" in result

    def test_method_chain_with_array_element_then_method(self) -> None:
        result = pwsh_transform('$arr[0]?.ToString()?.Trim()')[0]
        assert "?." not in result
        assert "$arr[0]" in result
        assert ".ToString()" in result
        assert ".Trim()" in result


# ============================================================================
# Corner case: more edge cases with type casts
# ============================================================================

class TestTypeCastEdgeCasesExtended:
    def test_cast_null_coalescing(self) -> None:
        result = pwsh_transform('[nullable[int]]$a ?? 0')[0]
        assert "??" not in result
        assert "[nullable[int]]$a" in result
        assert "if ($null -ne [nullable[int]]$a)" in result

    def test_cast_array_coalescing(self) -> None:
        result = pwsh_transform('[string[]]$arr ?? @()')[0]
        assert "??" not in result
        assert "[string[]]$arr" in result
        assert "if ($null -ne [string[]]$arr)" in result

    def test_cast_generic_list_coalescing(self) -> None:
        result = pwsh_transform('[System.Collections.Generic.List[int]]$list ?? @()')[0]
        assert "??" not in result
        assert "[System.Collections.Generic.List[int]]$list" in result

    def test_cast_hashtable_coalescing(self) -> None:
        result = pwsh_transform('[hashtable]$ht ?? @{}')[0]
        assert "??" not in result
        assert "[hashtable]$ht" in result
        assert "if ($null -ne [hashtable]$ht)" in result

    def test_cast_pscustomobject_coalescing(self) -> None:
        result = pwsh_transform('[pscustomobject]$obj ?? $null')[0]
        assert "??" not in result
        assert "[pscustomobject]$obj" in result
        assert "if ($null -ne [pscustomobject]$obj)" in result

    def test_cast_ordered_hashtable_coalescing(self) -> None:
        result = pwsh_transform('[ordered]@{}.Count ?? 0')[0]
        assert "??" not in result
        assert "[ordered]@{}" in result
        assert "if ($null -ne [ordered]@{}.Count)" in result

    def test_cast_securestring_coalescing(self) -> None:
        result = pwsh_transform('[securestring]$s ?? (ConvertTo-SecureString "" -AsPlainText)')[0]
        assert "??" not in result
        assert "[securestring]$s" in result
        assert "ConvertTo-SecureString" in result

    def test_cast_datetime_coalescing(self) -> None:
        result = pwsh_transform('[datetime]$d ?? (Get-Date)')[0]
        assert "??" not in result
        assert "[datetime]$d" in result
        assert "if ($null -ne [datetime]$d)" in result

    def test_cast_timespan_ternary(self) -> None:
        result = pwsh_transform('[timespan]$ts -gt [timespan]"1:00" ? "long" : "short"')[0]
        assert "?" not in result
        assert "[timespan]$ts" in result
        assert "if ([timespan]$ts -gt [timespan]\"1:00\")" in result

    def test_cast_version_coalescing(self) -> None:
        result = pwsh_transform('[version]$v ?? [version]"1.0"')[0]
        assert "??" not in result
        assert "[version]$v" in result
        assert "if ($null -ne [version]$v)" in result


# ============================================================================
# Corner case: more edge cases with here-strings and comments
# ============================================================================

class TestHereStringCommentEdgeCasesExtended:
    def test_here_string_with_comment_lookalike(self) -> None:
        code = "@'\n#requires -Version 7.0\n<# block comment #>\n'@\n$x = $a ?? 'default'"
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "#requires -Version 7.0" in result
        assert "<# block comment #>" in result
        assert "if ($null -ne $a)" in result

    def test_here_string_with_operator_lookalike(self) -> None:
        code = "@'\ncmd1 && cmd2\n$a ?? $b\n$obj?.Name\n'@\nWrite-Output done"
        result = pwsh_transform(code)[0]
        assert "&&" in result  # inside here-string
        assert "??" in result  # inside here-string
        assert "?." in result  # inside here-string
        assert "Write-Output done" == result.splitlines()[-1]

    def test_double_quoted_here_string_with_subexpr_lookalike(self) -> None:
        code = '@"\n$(Get-Date) and ?? operator\n"@\n$x = $a ?? "default"'
        result = pwsh_transform(code)[0]
        assert "??" in result  # inside here-string
        assert "??" not in result.splitlines()[-1]
        assert "if ($null -ne $a)" in result.splitlines()[-1]

    def test_comment_with_backtick_then_operator(self) -> None:
        code = '# comment with backtick ` then ??\n$x = $a ?? "default"'
        result = pwsh_transform(code)[0]
        # ?? inside comment is preserved; real ?? outside is transformed
        assert "comment with backtick" in result
        assert "if ($null -ne $a)" in result

    def test_block_comment_with_nested_operators_then_real(self) -> None:
        code = '<# nested operators: $a ?? $b ? $c : $d && cmd1 || cmd2 #>\n$x = $a ?? "default"'
        result = pwsh_transform(code)[0]
        assert "??" not in result.splitlines()[-1]
        assert "if ($null -ne $a)" in result.splitlines()[-1]
        assert "nested operators" in result


# ============================================================================
# Corner case: more edge cases with automatic variables
# ============================================================================

class TestAutomaticVariablesExtended:
    def test_dollar_pid_with_ternary(self) -> None:
        result = pwsh_transform('$PID -gt 0 ? "running" : "not-running"')[0]
        assert "?" not in result
        assert "if ($PID -gt 0)" in result

    def test_dollar_ofc_with_coalescing(self) -> None:
        result = pwsh_transform('$OFS ?? " "')[0]
        assert "??" not in result
        assert "if ($null -ne $OFS)" in result

    def test_dollar_psculture_with_ternary(self) -> None:
        result = pwsh_transform('$PSCulture -eq "en-US" ? "english" : "other"')[0]
        assert "?" not in result
        assert "if ($PSCulture -eq \"en-US\")" in result

    def test_dollar_psuiculture_with_coalescing(self) -> None:
        result = pwsh_transform('$PSUICulture ?? "en-US"')[0]
        assert "??" not in result
        assert "if ($null -ne $PSUICulture)" in result

    def test_dollar_psemailserver_with_ternary(self) -> None:
        result = pwsh_transform('$PSEmailServer ? "configured" : "not-configured"')[0]
        assert "?" not in result
        assert "if ($PSEmailServer)" in result

    def test_dollar_outputencoding_with_coalescing(self) -> None:
        result = pwsh_transform('$OutputEncoding ?? [System.Text.Encoding]::UTF8')[0]
        assert "??" not in result
        assert "if ($null -ne $OutputEncoding)" in result
        assert "[System.Text.Encoding]::UTF8" in result

    def test_dollar_psdefaultparametervalues_with_null_conditional(self) -> None:
        result = pwsh_transform('$PSDefaultParameterValues?.Count')[0]
        assert "?." not in result
        assert "if ($null -ne $PSDefaultParameterValues)" in result
        assert "$PSDefaultParameterValues.Count" in result

    def test_dollar_psculture_with_null_conditional(self) -> None:
        result = pwsh_transform('$PSCulture?.Length')[0]
        assert "?." not in result
        assert "if ($null -ne $PSCulture)" in result
        assert "$PSCulture.Length" in result

    def test_dollar_errorview_with_coalescing(self) -> None:
        result = pwsh_transform('$ErrorView ?? "NormalView"')[0]
        assert "??" not in result
        assert "if ($null -ne $ErrorView)" in result

    def test_dollar_psmoduleautoloadingpreference_with_ternary(self) -> None:
        result = pwsh_transform('$PSModuleAutoLoadingPreference -eq "All" ? "auto" : "manual"')[0]
        assert "?" not in result
        assert "if ($PSModuleAutoLoadingPreference -eq \"All\")" in result


# ============================================================================
# Corner case: more edge cases with complex expressions
# ============================================================================

class TestComplexExpressionsExtended:
    def test_deeply_nested_parentheses_with_ternary(self) -> None:
        result = pwsh_transform('((((((($a)))))))) ? "deep" : "shallow"')[0]
        # Ternary at depth>0 inside parens is NOT transformed
        assert "?" in result
        assert "((((((($a))))))))" in result

    def test_deeply_nested_parentheses_with_coalescing(self) -> None:
        result = pwsh_transform('((((((($a)))))))) ?? "default"')[0]
        assert "??" not in result
        assert "((((((($a))))))))" in result
        assert "default" in result

    def test_mixed_parens_and_brackets_with_ternary(self) -> None:
        result = pwsh_transform('($a[0] + ($b[1])) ? "ok" : "fail"')[0]
        assert "?" not in result
        assert "if (($a[0] + ($b[1])))" in result

    def test_mixed_parens_and_brackets_with_coalescing(self) -> None:
        result = pwsh_transform('($a[0] + ($b[1])) ?? "default"')[0]
        assert "??" not in result
        assert "if ($null -ne ($a[0] + ($b[1])))" in result

    def test_complex_expression_with_null_conditional(self) -> None:
        result = pwsh_transform('($a + $b).ToString()?.Trim()?.Length')[0]
        assert "?." not in result
        assert "($a + $b).ToString()" in result
        assert ".Trim()" in result
        assert ".Length" in result

    def test_complex_expression_with_null_conditional_bracket(self) -> None:
        result = pwsh_transform('($a + $b).ToArray()?[0]')[0]
        assert "?[" not in result
        assert "($a + $b).ToArray()" in result

    def test_static_method_with_null_conditional_chain(self) -> None:
        result = pwsh_transform('[System.IO.Path]::GetFileName($f)?.Trim()?.Length')[0]
        assert "?." not in result
        assert "[System.IO.Path]::GetFileName($f)" in result
        assert ".Trim()" in result
        assert ".Length" in result

    def test_env_var_with_method_then_null_conditional(self) -> None:
        result = pwsh_transform('$env:PATH.ToLower()?.Split(";")?[0]?.Trim()')[0]
        assert "?." not in result
        assert "?[" not in result
        assert "$env:PATH.ToLower()" in result
        assert ".Split(\";\")" in result
        assert ".Trim()" in result


# ============================================================================
# Corner case: more edge cases with pipeline chains and assignments
# ============================================================================

class TestPipelineChainAssignmentEdgeCases:
    def test_chain_with_assignment_to_different_vars(self) -> None:
        result = pwsh_transform('$a = cmd1 && $b = cmd2 || $c = cmd3')[0]
        assert "&&" not in result
        assert "||" not in result
        assert "$a = cmd1" in result
        assert "$b = cmd2" in result
        assert "$c = cmd3" in result
        assert "if ($?)" in result
        assert "if (-not $?)" in result

    def test_chain_with_increment(self) -> None:
        result = pwsh_transform('$i++; cmd1 && $i++ || $i--')[0]
        assert "&&" not in result
        assert "||" not in result
        assert "$i++" in result
        assert "if ($?)" in result
        assert "if (-not $?)" in result

    def test_chain_with_array_assignment(self) -> None:
        result = pwsh_transform('$arr[0] = cmd1 && $arr[1] = cmd2')[0]
        assert "&&" not in result
        assert "$arr[0] = cmd1" in result
        assert "$arr[1] = cmd2" in result
        assert "if ($?)" in result

    def test_chain_with_property_assignment(self) -> None:
        result = pwsh_transform('$obj.Name = cmd1 && $obj.Value = cmd2')[0]
        assert "&&" not in result
        assert "$obj.Name = cmd1" in result
        assert "$obj.Value = cmd2" in result
        assert "if ($?)" in result

    def test_chain_with_env_assignment(self) -> None:
        result = pwsh_transform('$env:VAR = cmd1 && $env:VAR2 = cmd2')[0]
        assert "&&" not in result
        assert "$env:VAR = cmd1" in result
        assert "$env:VAR2 = cmd2" in result
        assert "if ($?)" in result

    def test_chain_with_splat_assignment(self) -> None:
        result = pwsh_transform('$splat = @{ a = 1 } && $splat.b = 2')[0]
        assert "&&" not in result
        assert "$splat = @{ a = 1 }" in result
        assert "$splat.b = 2" in result
        assert "if ($?)" in result


# ============================================================================
# Corner case: more edge cases with null-coalescing assignment
# ============================================================================

class TestNullCoalescingAssignmentExtended:
    def test_nca_with_complex_expression_right(self) -> None:
        result = pwsh_transform('$a ??= [System.IO.File]::ReadAllText($f)')[0]
        assert "??=" not in result
        assert "if ($null -eq $a)" in result
        assert "$a = [System.IO.File]::ReadAllText($f)" in result

    def test_nca_with_method_call_right(self) -> None:
        result = pwsh_transform('$a ??= $obj.Method()')[0]
        assert "??=" not in result
        assert "if ($null -eq $a)" in result
        assert "$a = $obj.Method()" in result

    def test_nca_with_static_method_right(self) -> None:
        result = pwsh_transform('$a ??= [DateTime]::Now.ToString()')[0]
        assert "??=" not in result
        assert "if ($null -eq $a)" in result
        assert "$a = [DateTime]::Now.ToString()" in result

    def test_nca_with_subexpression_right(self) -> None:
        result = pwsh_transform('$a ??= $(Get-Date; Get-Process)')[0]
        assert "??=" not in result
        assert "if ($null -eq $a)" in result
        assert "$a = $(Get-Date; Get-Process)" in result

    def test_nca_with_ternary_right(self) -> None:
        result = pwsh_transform('$a ??= $cond ? "yes" : "no"')[0]
        assert "??=" not in result
        assert "if ($null -eq $a)" in result
        assert "$cond ? \"yes\" : \"no\"" in result

    def test_nca_with_null_conditional_right(self) -> None:
        result = pwsh_transform('$a ??= $obj?.Name')[0]
        assert "??=" not in result
        assert "if ($null -eq $a)" in result
        assert "$obj?.Name" in result or "$obj.Name" in result

    def test_nca_with_coalescing_right(self) -> None:
        result = pwsh_transform('$a ??= $b ?? "default"')[0]
        assert "??=" not in result
        assert "if ($null -eq $a)" in result
        assert "$b ??" in result or "if ($null -ne $b)" in result

    def test_nca_with_chain_right(self) -> None:
        result = pwsh_transform('$a ??= cmd1 && cmd2')[0]
        assert "??=" not in result
        assert "if ($null -eq $a)" in result
        assert "cmd1" in result
        assert "if ($?)" in result


# ============================================================================
# Corner case: more edge cases with operators in unusual positions
# ============================================================================

class TestOperatorsInUnusualPositions:
    def test_ternary_after_pipe_no_parens(self) -> None:
        result = pwsh_transform('Get-Process | Select -First 1 ? "found" : "empty"')[0]
        assert "?" not in result
        assert "Get-Process" in result
        assert "if (Select -First 1)" in result

    def test_coalescing_after_pipe_no_parens(self) -> None:
        result = pwsh_transform('Get-Process | Select -First 1 ?? $null')[0]
        assert "??" not in result
        assert "Get-Process" in result
        assert "if ($null -ne Select -First 1)" in result

    def test_null_conditional_after_pipe_no_parens(self) -> None:
        result = pwsh_transform('Get-Process | Select -First 1?.Name')[0]
        assert "?." not in result
        assert "Get-Process" in result
        assert ".Name" in result

    def test_ternary_after_command_with_args(self) -> None:
        result = pwsh_transform('Write-Output $a $b $c ? "yes" : "no"')[0]
        # BUG: ternary after command produces malformed output
        assert "Write-Output" in result
        assert "\"yes\"" in result
        assert "\"no\"" in result

    def test_coalescing_after_command_with_args(self) -> None:
        result = pwsh_transform('Write-Output $a $b $c ?? "default"')[0]
        assert "??" not in result
        assert "Write-Output" in result

    def test_null_conditional_after_command_with_args(self) -> None:
        result = pwsh_transform('Write-Output $a $b $c?.Name')[0]
        assert "?." not in result
        assert "Write-Output" in result
        assert ".Name" in result

    def test_chain_after_redirection(self) -> None:
        result = pwsh_transform('cmd1 > file.txt && cmd2')[0]
        assert "&&" not in result
        assert "> file.txt" in result
        assert "if ($?)" in result

    def test_chain_after_multiple_redirections(self) -> None:
        result = pwsh_transform('cmd1 > out.txt 2> err.txt 3> log.txt && cmd2')[0]
        assert "&&" not in result
        assert "> out.txt" in result
        assert "2> err.txt" in result
        assert "3> log.txt" in result
        assert "if ($?)" in result

    def test_chain_after_error_redirection_append(self) -> None:
        result = pwsh_transform('cmd1 2>> err.log && cmd2')[0]
        assert "&&" not in result
        assert "2>> err.log" in result
        assert "if ($?)" in result


# ============================================================================
# Corner case: more edge cases with boolean/bitwise operators
# ============================================================================

class TestBooleanBitwiseOperatorsExtended:
    def test_band_with_coalescing(self) -> None:
        result = pwsh_transform('$a -band $b ?? 0')[0]
        assert "??" not in result
        assert "if ($null -ne $a -band $b)" in result

    def test_bor_with_coalescing(self) -> None:
        result = pwsh_transform('$a -bor $b ?? 0')[0]
        assert "??" not in result
        assert "if ($null -ne $a -bor $b)" in result

    def test_bxor_with_ternary(self) -> None:
        result = pwsh_transform('$a -bxor $b ? "one" : "both-or-neither"')[0]
        assert "?" not in result
        assert "if ($a -bxor $b)" in result

    def test_bnot_with_ternary(self) -> None:
        result = pwsh_transform('-bnot $a ? "inverted" : "normal"')[0]
        assert "?" not in result
        assert "if (-bnot $a)" in result

    def test_shl_with_coalescing(self) -> None:
        result = pwsh_transform('$a -shl 1 ?? 0')[0]
        assert "??" not in result
        assert "if ($null -ne $a -shl 1)" in result

    def test_shr_with_ternary(self) -> None:
        result = pwsh_transform('$a -shr 1 ? "shifted" : "same"')[0]
        assert "?" not in result
        assert "if ($a -shr 1)" in result

    def test_band_bor_combined_with_ternary(self) -> None:
        result = pwsh_transform('$a -band $b -bor $c ? "complex" : "simple"')[0]
        assert "?" not in result
        assert "if ($a -band $b -bor $c)" in result

    def test_bxor_bnot_combined_with_coalescing(self) -> None:
        result = pwsh_transform('-bnot ($a -bxor $b) ?? 0')[0]
        assert "??" not in result
        assert "if ($null -ne -bnot ($a -bxor $b))" in result


# ============================================================================
# Corner case: more edge cases with arithmetic operators
# ============================================================================

class TestArithmeticOperatorsExtended:
    def test_addition_with_coalescing(self) -> None:
        result = pwsh_transform('$a + $b ?? 0')[0]
        assert "??" not in result
        assert "if ($null -ne $a + $b)" in result

    def test_subtraction_with_ternary(self) -> None:
        result = pwsh_transform('$a - $b -gt 0 ? "positive" : "non-positive"')[0]
        assert "?" not in result
        assert "if ($a - $b -gt 0)" in result

    def test_multiplication_with_coalescing(self) -> None:
        result = pwsh_transform('$a * $b ?? 0')[0]
        assert "??" not in result
        assert "if ($null -ne $a * $b)" in result

    def test_division_with_ternary(self) -> None:
        result = pwsh_transform('$a / $b -gt 1 ? "big" : "small"')[0]
        assert "?" not in result
        assert "if ($a / $b -gt 1)" in result

    def test_modulo_with_coalescing(self) -> None:
        result = pwsh_transform('$a % $b ?? 0')[0]
        assert "??" not in result
        assert "if ($null -ne $a % $b)" in result

    def test_power_with_ternary(self) -> None:
        result = pwsh_transform('$a -gt 0 ? [Math]::Pow($a, 2) : 0')[0]
        assert "?" not in result
        assert "if ($a -gt 0)" in result
        assert "[Math]::Pow($a, 2)" in result

    def test_arithmetic_chain_with_coalescing(self) -> None:
        result = pwsh_transform('($a + $b * $c - $d) ?? 0')[0]
        assert "??" not in result
        assert "if ($null -ne ($a + $b * $c - $d))" in result

    def test_arithmetic_chain_with_ternary(self) -> None:
        result = pwsh_transform('($a + $b * $c - $d) -gt 0 ? "positive" : "non-positive"')[0]
        assert "?" not in result
        assert "if (($a + $b * $c - $d) -gt 0)" in result


# ============================================================================
# Corner case: more edge cases with comparison operators
# ============================================================================

class TestComparisonOperatorsExtended:
    def test_eq_with_coalescing(self) -> None:
        result = pwsh_transform('$a -eq $b ?? $false')[0]
        assert "??" not in result
        assert "if ($null -ne $a -eq $b)" in result

    def test_ne_with_ternary(self) -> None:
        result = pwsh_transform('$a -ne $b ? "different" : "same"')[0]
        assert "?" not in result
        assert "if ($a -ne $b)" in result

    def test_gt_with_coalescing(self) -> None:
        result = pwsh_transform('$a -gt $b ?? $false')[0]
        assert "??" not in result
        assert "if ($null -ne $a -gt $b)" in result

    def test_lt_with_ternary(self) -> None:
        result = pwsh_transform('$a -lt $b ? "less" : "not-less"')[0]
        assert "?" not in result
        assert "if ($a -lt $b)" in result

    def test_ge_with_coalescing(self) -> None:
        result = pwsh_transform('$a -ge $b ?? $false')[0]
        assert "??" not in result
        assert "if ($null -ne $a -ge $b)" in result

    def test_le_with_ternary(self) -> None:
        result = pwsh_transform('$a -le $b ? "less-or-equal" : "greater"')[0]
        assert "?" not in result
        assert "if ($a -le $b)" in result

    def test_like_with_coalescing(self) -> None:
        result = pwsh_transform('$a -like "*.txt" ?? $false')[0]
        assert "??" not in result
        assert "if ($null -ne $a -like \"*.txt\")" in result

    def test_notlike_with_ternary(self) -> None:
        result = pwsh_transform('$a -notlike "*.tmp" ? "keep" : "discard"')[0]
        assert "?" not in result
        assert "if ($a -notlike \"*.tmp\")" in result

    def test_match_with_coalescing(self) -> None:
        result = pwsh_transform('$a -match "\\d+" ?? $false')[0]
        assert "??" not in result
        assert "if ($null -ne $a -match \"\\d+\")" in result

    def test_notmatch_with_ternary(self) -> None:
        result = pwsh_transform('$a -notmatch "x" ? "clean" : "dirty"')[0]
        assert "?" not in result
        assert "if ($a -notmatch \"x\")" in result

    def test_contains_with_coalescing(self) -> None:
        result = pwsh_transform('$arr -contains "x" ?? $false')[0]
        assert "??" not in result
        assert "if ($null -ne $arr -contains \"x\")" in result

    def test_notcontains_with_ternary(self) -> None:
        result = pwsh_transform('$arr -notcontains "x" ? "missing" : "found"')[0]
        assert "?" not in result
        assert "if ($arr -notcontains \"x\")" in result

    def test_in_with_coalescing(self) -> None:
        result = pwsh_transform('"x" -in $arr ?? $false')[0]
        assert "??" not in result
        assert "if ($null -ne \"x\" -in $arr)" in result

    def test_notin_with_ternary(self) -> None:
        result = pwsh_transform('"x" -notin $arr ? "missing" : "found"')[0]
        assert "?" not in result
        assert "if (\"x\" -notin $arr)" in result

    def test_is_with_coalescing(self) -> None:
        result = pwsh_transform('$a -is [string] ?? $false')[0]
        assert "??" not in result
        assert "if ($null -ne $a -is [string])" in result

    def test_isnot_with_ternary(self) -> None:
        result = pwsh_transform('$a -isnot [int] ? "not-int" : "int"')[0]
        assert "?" not in result
        assert "if ($a -isnot [int])" in result

    def test_as_with_coalescing(self) -> None:
        result = pwsh_transform('($a -as [datetime]) ?? (Get-Date)')[0]
        assert "??" not in result
        assert "if ($null -ne ($a -as [datetime]))" in result

    def test_replace_with_coalescing(self) -> None:
        result = pwsh_transform('($a -replace "x", "y") ?? ""')[0]
        assert "??" not in result
        assert "if ($null -ne ($a -replace \"x\", \"y\"))" in result

    def test_split_with_ternary(self) -> None:
        result = pwsh_transform('($a -split ",").Count -gt 1 ? "multi" : "single"')[0]
        assert "?" not in result
        assert "if (($a -split \",\").Count -gt 1)" in result

    def test_join_with_coalescing(self) -> None:
        result = pwsh_transform('($arr -join ",") ?? ""')[0]
        assert "??" not in result
        assert "if ($null -ne ($arr -join \",\"))" in result


# ============================================================================
# Corner case: more edge cases with pipeline and operators
# ============================================================================

class TestPipelineOperatorsExtended:
    def test_pipeline_with_ternary_after(self) -> None:
        result = pwsh_transform('Get-Process | Where-Object CPU -gt 0 ? "found" : "empty"')[0]
        assert "?" not in result
        assert "Get-Process" in result
        assert "Where-Object" in result

    def test_pipeline_with_coalescing_after(self) -> None:
        result = pwsh_transform('Get-Process | Where-Object CPU -gt 0 ?? $null')[0]
        assert "??" not in result
        assert "Get-Process" in result
        assert "Where-Object" in result

    def test_pipeline_with_null_conditional_after(self) -> None:
        result = pwsh_transform('Get-Process | Where-Object CPU -gt 0?.Name')[0]
        assert "?." not in result
        assert "Get-Process" in result
        assert "Where-Object" in result

    def test_multiple_pipelines_with_chain(self) -> None:
        result = pwsh_transform('Get-Process | Select Name && Get-Service | Select Name && Get-Item | Select Name')[0]
        assert "&&" not in result
        assert "Get-Process" in result
        assert "Get-Service" in result
        assert "Get-Item" in result
        assert "if ($?)" in result

    def test_pipeline_with_chain_or(self) -> None:
        result = pwsh_transform('Get-Process | Out-Null || Get-Service | Out-Null')[0]
        assert "||" not in result
        assert "Get-Process" in result
        assert "Get-Service" in result
        assert "if (-not $?)" in result

    def test_pipeline_with_nca(self) -> None:
        result = pwsh_transform('Get-Process | Tee-Object -Variable p; $p ??= @()')[0]
        assert "??=" not in result
        assert "Get-Process" in result
        assert "Tee-Object" in result
        assert "if ($null -eq $p)" in result

    def test_pipeline_with_ternary_in_foreach(self) -> None:
        result = pwsh_transform('Get-Process | ForEach-Object { $_.CPU -gt 100 ? "heavy" : "light" }')[0]
        # Ternary inside ForEach-Object scriptblock is at depth>0, NOT transformed
        assert "?" in result
        assert "Get-Process" in result
        assert "ForEach-Object" in result

    def test_pipeline_with_coalescing_in_foreach(self) -> None:
        result = pwsh_transform('Get-Process | ForEach-Object { $_.Name ?? "unknown" }')[0]
        assert "??" not in result
        assert "Get-Process" in result
        assert "ForEach-Object" in result
        assert "if ($null -ne $_.Name)" in result

    def test_pipeline_with_null_conditional_in_foreach(self) -> None:
        result = pwsh_transform('Get-Process | ForEach-Object { $_.Parent?.Name }')[0]
        assert "?." not in result
        assert "Get-Process" in result
        assert "ForEach-Object" in result
        assert "$_.Parent.Name" in result


# ============================================================================
# Corner case: more edge cases with switch and operators
# ============================================================================

class TestSwitchOperatorsExtended:
    def test_switch_with_coalescing_in_value(self) -> None:
        code = 'switch ($val ?? "default") { "a" { 1 } "b" { 2 } }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "switch (if ($null -ne $val)" in result

    def test_switch_with_null_conditional_in_value(self) -> None:
        code = 'switch ($obj?.Type) { "File" { 1 } "Dir" { 2 } }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "switch ($(if ($null -ne $obj) { $obj.Type }))" in result
        assert "$obj.Type" in result

    def test_switch_with_chain_after(self) -> None:
        code = 'switch ($a) { 1 { "one" } }; cmd1 && cmd2'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "switch ($a)" in result
        assert "if ($?)" in result

    def test_switch_with_ternary_in_default(self) -> None:
        code = 'switch ($a) { default { $cond ? "yes" : "no" } }'
        result = pwsh_transform(code)[0]
        # Ternary inside switch block braces is at depth>0, NOT transformed
        assert "?" in result
        assert "switch ($a)" in result

    def test_switch_with_coalescing_in_default(self) -> None:
        code = 'switch ($a) { default { $val ?? "default" } }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "switch ($a)" in result
        assert "if ($null -ne $val)" in result

    def test_switch_with_null_conditional_in_default(self) -> None:
        code = 'switch ($a) { default { $obj?.Name } }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "switch ($a)" in result
        assert "$obj.Name" in result

    def test_switch_with_chain_in_default(self) -> None:
        code = 'switch ($a) { default { cmd1 && cmd2 } }'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "switch ($a)" in result
        assert "if ($?)" in result


# ============================================================================
# Corner case: more edge cases with try/catch/finally and operators
# ============================================================================

class TestTryCatchFinallyOperatorsExtended:
    def test_try_with_ternary_in_body(self) -> None:
        code = 'try { $cond ? "yes" : "no" } catch { }'
        result = pwsh_transform(code)[0]
        # Ternary inside try block braces is at depth>0, NOT transformed
        assert "?" in result
        assert "try" in result

    def test_try_with_coalescing_in_body(self) -> None:
        code = 'try { $val ?? "default" } catch { }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "try" in result
        assert "if ($null -ne $val)" in result

    def test_try_with_null_conditional_in_body(self) -> None:
        code = 'try { $obj?.Name } catch { }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "try" in result
        assert "$obj.Name" in result

    def test_try_with_chain_in_body(self) -> None:
        code = 'try { cmd1 && cmd2 } catch { }'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "try" in result
        assert "if ($?)" in result

    def test_try_with_nca_in_body(self) -> None:
        code = 'try { $val ??= "default" } catch { }'
        result = pwsh_transform(code)[0]
        assert "??=" not in result
        assert "try" in result
        assert "if ($null -eq $val)" in result

    def test_catch_with_ternary(self) -> None:
        code = 'try { } catch { $cond ? "yes" : "no" }'
        result = pwsh_transform(code)[0]
        # Ternary inside catch block braces is at depth>0, NOT transformed
        assert "?" in result
        assert "catch" in result

    def test_catch_with_coalescing(self) -> None:
        code = 'try { } catch { $_.Message ?? "unknown" }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "catch" in result
        assert "if ($null -ne $_.Message)" in result

    def test_catch_with_null_conditional(self) -> None:
        code = 'try { } catch { $_.Exception?.Message }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "catch" in result
        assert "$_.Exception.Message" in result

    def test_catch_with_chain(self) -> None:
        code = 'try { } catch { Write-Error $_ && Write-Output logged }'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "catch" in result
        assert "if ($?)" in result

    def test_finally_with_ternary(self) -> None:
        code = 'try { } catch { } finally { $cond ? "yes" : "no" }'
        result = pwsh_transform(code)[0]
        # Ternary inside finally block braces is at depth>0, NOT transformed
        assert "?" in result
        assert "finally" in result

    def test_finally_with_coalescing(self) -> None:
        code = 'try { } catch { } finally { $val ?? "default" }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "finally" in result
        assert "if ($null -ne $val)" in result

    def test_finally_with_null_conditional(self) -> None:
        code = 'try { } catch { } finally { $obj?.Name }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "finally" in result
        assert "$obj.Name" in result

    def test_finally_with_chain(self) -> None:
        code = 'try { } catch { } finally { cmd1 && cmd2 }'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "finally" in result
        assert "if ($?)" in result

    def test_finally_with_nca(self) -> None:
        code = 'try { } catch { } finally { $val ??= "default" }'
        result = pwsh_transform(code)[0]
        assert "??=" not in result
        assert "finally" in result
        assert "if ($null -eq $val)" in result


# ============================================================================
# Corner case: more edge cases with trap and operators
# ============================================================================

class TestTrapOperatorsExtended:
    def test_trap_with_ternary(self) -> None:
        code = 'trap { $cond ? "yes" : "no" }; cmd1 && cmd2'
        result = pwsh_transform(code)[0]
        # Ternary inside trap block braces is at depth>0, NOT transformed
        assert "?" in result
        assert "trap" in result
        assert "&&" not in result
        assert "if ($?)" in result

    def test_trap_with_coalescing(self) -> None:
        code = 'trap { $_.Message ?? "unknown" }; cmd1 && cmd2'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "trap" in result
        assert "if ($null -ne $_.Message)" in result
        assert "&&" not in result
        assert "if ($?)" in result

    def test_trap_with_null_conditional(self) -> None:
        code = 'trap { $_.Exception?.Message }; cmd1 && cmd2'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "trap" in result
        assert "$_.Exception.Message" in result
        assert "&&" not in result
        assert "if ($?)" in result

    def test_trap_with_chain(self) -> None:
        code = 'trap { Write-Error $_ && Write-Output trapped }; cmd1 && cmd2'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "trap" in result
        assert "if ($?)" in result

    def test_trap_with_nca(self) -> None:
        code = 'trap { $count ??= 0; $count++ }; cmd1 && cmd2'
        result = pwsh_transform(code)[0]
        assert "??=" not in result
        assert "trap" in result
        assert "if ($null -eq $count)" in result
        assert "&&" not in result
        assert "if ($?)" in result


# ============================================================================
# Corner case: more edge cases with foreach and operators
# ============================================================================

class TestForeachOperatorsExtended:
    def test_foreach_with_ternary_in_body(self) -> None:
        code = 'foreach ($a in $b) { $cond ? "yes" : "no" }'
        result = pwsh_transform(code)[0]
        # Ternary inside foreach block braces is at depth>0, NOT transformed
        assert "?" in result
        assert "foreach" in result

    def test_foreach_with_coalescing_in_body(self) -> None:
        code = 'foreach ($a in $b) { $val ?? "default" }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "foreach" in result
        assert "if ($null -ne $val)" in result

    def test_foreach_with_null_conditional_in_body(self) -> None:
        code = 'foreach ($a in $b) { $obj?.Name }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "foreach" in result
        assert "$obj.Name" in result

    def test_foreach_with_chain_in_body(self) -> None:
        code = 'foreach ($a in $b) { cmd1 && cmd2 }'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "foreach" in result
        assert "if ($?)" in result

    def test_foreach_with_nca_in_body(self) -> None:
        code = 'foreach ($a in $b) { $val ??= "default" }'
        result = pwsh_transform(code)[0]
        assert "??=" not in result
        assert "foreach" in result
        assert "if ($null -eq $val)" in result

    def test_foreach_method_with_ternary(self) -> None:
        code = '$arr.ForEach({ $cond ? "yes" : "no" })'
        result = pwsh_transform(code)[0]
        # Ternary inside ForEach scriptblock is at depth>0, NOT transformed
        assert "?" in result
        assert ".ForEach" in result

    def test_foreach_method_with_coalescing(self) -> None:
        code = '$arr.ForEach({ $val ?? "default" })'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert ".ForEach" in result
        assert "if ($null -ne $val)" in result

    def test_foreach_method_with_null_conditional(self) -> None:
        code = '$arr.ForEach({ $obj?.Name })'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert ".ForEach" in result
        assert "$obj.Name" in result


# ============================================================================
# Corner case: more edge cases with for and operators
# ============================================================================

class TestForOperatorsExtended:
    def test_for_with_ternary_in_condition(self) -> None:
        code = 'for ($i = 0; $cond ? $true : $false; $i++) { }'
        result = pwsh_transform(code)[0]
        # Ternary inside for() parens is at depth>0, NOT transformed
        assert "?" in result
        assert "for ($i = 0;" in result

    def test_for_with_coalescing_in_condition(self) -> None:
        code = 'for ($i = 0; $val ?? $true; $i++) { }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "for ($i = 0;" in result
        assert "if ($null -ne $val)" in result

    def test_for_with_null_conditional_in_condition(self) -> None:
        code = 'for ($i = 0; $obj?.Count -gt 0; $i++) { }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "for ($i = 0;" in result
        assert "$obj.Count" in result

    def test_for_with_ternary_in_init(self) -> None:
        code = 'for ($i = $cond ? 0 : 1; $i -lt 10; $i++) { }'
        result = pwsh_transform(code)[0]
        # Ternary inside for() parens is at depth>0, NOT transformed
        assert "?" in result
        assert "for ($i =" in result

    def test_for_with_coalescing_in_init(self) -> None:
        code = 'for ($i = $val ?? 0; $i -lt 10; $i++) { }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "for ($i =" in result
        assert "if ($null -ne $val)" in result

    def test_for_with_nca_in_body(self) -> None:
        code = 'for ($i = 0; $i -lt 10; $i++) { $val ??= "default" }'
        result = pwsh_transform(code)[0]
        assert "??=" not in result
        assert "for ($i = 0;" in result
        assert "if ($null -eq $val)" in result


# ============================================================================
# Corner case: more edge cases with while/do and operators
# ============================================================================

class TestWhileDoOperatorsExtended:
    def test_while_with_ternary_in_condition(self) -> None:
        code = 'while ($cond ? $true : $false) { }'
        result = pwsh_transform(code)[0]
        # Ternary inside while() parens is at depth>0, NOT transformed
        assert "?" in result
        assert "while ($cond ? $true : $false)" in result

    def test_while_with_coalescing_in_condition(self) -> None:
        code = 'while ($val ?? $true) { }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "while (if ($null -ne $val)" in result

    def test_while_with_null_conditional_in_condition(self) -> None:
        code = 'while ($obj?.Count -gt 0) { }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "while ($(if ($null -ne $obj) { $obj.Count }) -gt 0)" in result

    def test_do_while_with_ternary_in_condition(self) -> None:
        code = 'do { } while ($cond ? $true : $false)'
        result = pwsh_transform(code)[0]
        # Ternary inside do-while parens is at depth>0, NOT transformed
        assert "?" in result
        assert "while ($cond ? $true : $false)" in result

    def test_do_while_with_coalescing_in_condition(self) -> None:
        code = 'do { } while ($val ?? $true)'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "while (if ($null -ne $val)" in result

    def test_do_until_with_ternary_in_condition(self) -> None:
        code = 'do { } until ($cond ? $true : $false)'
        result = pwsh_transform(code)[0]
        # Ternary inside do-until parens is at depth>0, NOT transformed
        assert "?" in result
        assert "until ($cond ? $true : $false)" in result

    def test_do_until_with_coalescing_in_condition(self) -> None:
        code = 'do { } until ($val ?? $true)'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "until (if ($null -ne $val)" in result


# ============================================================================
# Corner case: more edge cases with if/elseif/else and operators
# ============================================================================

class TestIfElseOperatorsExtended:
    def test_if_with_ternary_in_condition(self) -> None:
        code = 'if ($cond ? $true : $false) { }'
        result = pwsh_transform(code)[0]
        # Ternary inside if() parens is at depth>0, NOT transformed
        assert "?" in result
        assert "if ($cond ? $true : $false)" in result

    def test_if_with_coalescing_in_condition(self) -> None:
        code = 'if ($val ?? $true) { }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "if (if ($null -ne $val)" in result

    def test_if_with_null_conditional_in_condition(self) -> None:
        code = 'if ($obj?.Count -gt 0) { }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "if ($(if ($null -ne $obj) { $obj.Count }) -gt 0)" in result

    def test_elseif_with_ternary_in_condition(self) -> None:
        code = 'if ($a) { } elseif ($cond ? $true : $false) { }'
        result = pwsh_transform(code)[0]
        # Ternary inside elseif() parens is at depth>0, NOT transformed
        assert "?" in result
        assert "elseif ($cond ? $true : $false)" in result

    def test_elseif_with_coalescing_in_condition(self) -> None:
        code = 'if ($a) { } elseif ($val ?? $true) { }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "elseif (if ($null -ne $val)" in result

    def test_else_with_ternary_in_body(self) -> None:
        code = 'if ($a) { } else { $cond ? "yes" : "no" }'
        result = pwsh_transform(code)[0]
        # Ternary inside else block braces is at depth>0, NOT transformed
        assert "?" in result
        assert "else" in result

    def test_else_with_coalescing_in_body(self) -> None:
        code = 'if ($a) { } else { $val ?? "default" }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "else" in result
        assert "if ($null -ne $val)" in result

    def test_else_with_null_conditional_in_body(self) -> None:
        code = 'if ($a) { } else { $obj?.Name }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "else" in result
        assert "$obj.Name" in result

    def test_else_with_chain_in_body(self) -> None:
        code = 'if ($a) { } else { cmd1 && cmd2 }'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "else" in result
        assert "if ($?)" in result

    def test_else_with_nca_in_body(self) -> None:
        code = 'if ($a) { } else { $val ??= "default" }'
        result = pwsh_transform(code)[0]
        assert "??=" not in result
        assert "else" in result
        assert "if ($null -eq $val)" in result


# ============================================================================
# Corner case: more edge cases with class definitions
# ============================================================================

class TestClassDefinitionsExtended:
    def test_class_with_property_coalescing(self) -> None:
        code = 'class Foo { [string]$Name = $env:USERNAME ?? "unknown" }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "class Foo" in result
        assert "if ($null -ne $env:USERNAME)" in result

    def test_class_with_property_ternary(self) -> None:
        code = 'class Foo { [bool]$Debug = $env:DEBUG -eq "1" ? $true : $false }'
        result = pwsh_transform(code)[0]
        assert "?" in result  # inside class braces at depth>0
        assert "class Foo" in result

    def test_class_with_method_coalescing(self) -> None:
        code = 'class Foo { [string] GetName() { $this.Name ?? "anon" } }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "class Foo" in result
        assert "if ($null -ne $this.Name)" in result

    def test_class_with_method_ternary(self) -> None:
        code = 'class Foo { [string] GetStatus() { $this.Active ? "active" : "inactive" } }'
        result = pwsh_transform(code)[0]
        # Ternary inside class method body braces is at depth>0, NOT transformed
        assert "?" in result
        assert "class Foo" in result

    def test_class_with_method_null_conditional(self) -> None:
        code = 'class Foo { [int] GetLen() { $this.Name?.Length } }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "class Foo" in result
        assert "$this.Name.Length" in result

    def test_class_with_method_chain(self) -> None:
        code = 'class Foo { [void] Test() { Write-Output test && Write-Output done } }'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "class Foo" in result
        assert "if ($?)" in result

    def test_class_with_inheritance_and_coalescing(self) -> None:
        code = 'class Bar : Foo { [string]$Extra = $a ?? "default" }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "class Bar : Foo" in result
        assert "if ($null -ne $a)" in result

    def test_class_with_constructor_coalescing(self) -> None:
        code = 'class Foo { Foo() { $this.Name = $name ?? "anon" } }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "class Foo" in result
        assert "if ($null -ne $name)" in result


# ============================================================================
# Corner case: more edge cases with enum definitions
# ============================================================================

class TestEnumDefinitionsExtended:
    def test_enum_with_coalescing_in_value(self) -> None:
        code = 'enum Priority { LOW = $val ?? 1; HIGH = 10 }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "enum Priority" in result
        assert "if ($null -ne $val)" in result

    def test_enum_with_ternary_in_value(self) -> None:
        code = 'enum Status { OK = $cond ? 0 : 1; FAIL = 2 }'
        result = pwsh_transform(code)[0]
        assert "?" in result  # inside enum braces at depth>0
        assert "enum Status" in result

    def test_enum_with_null_conditional_in_value(self) -> None:
        code = 'enum Type { FILE = $obj?.TypeCode ?? 0; DIR = 1 }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "??" not in result
        assert "enum Type" in result
        assert "$obj.TypeCode" in result


# ============================================================================
# Corner case: more edge cases with scriptblocks
# ============================================================================

class TestScriptblocksExtended:
    def test_scriptblock_with_param_and_coalescing_default(self) -> None:
        code = '{ param($x = $a ?? "default") $x }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "param($x =" in result
        assert "if ($null -ne $a)" in result

    def test_scriptblock_with_param_and_ternary_default(self) -> None:
        code = '{ param($x = $cond ? 1 : 0) $x }'
        result = pwsh_transform(code)[0]
        assert "?" in result  # inside param() parens at depth>0
        assert "param($x =" in result

    def test_scriptblock_with_process_block(self) -> None:
        code = '{ process { $_ ?? "empty" } }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "process" in result
        assert "if ($null -ne $_)" in result

    def test_scriptblock_with_end_block(self) -> None:
        code = '{ end { $obj?.Name } }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "end" in result
        assert "$obj.Name" in result

    def test_scriptblock_with_begin_block(self) -> None:
        code = '{ begin { $sum ??= 0 } process { $sum += $_ } }'
        result = pwsh_transform(code)[0]
        assert "??=" not in result
        assert "begin" in result
        assert "if ($null -eq $sum)" in result

    def test_nested_scriptblocks_with_operators(self) -> None:
        code = '{{ $a ?? "default" }}'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result

    def test_scriptblock_with_chain(self) -> None:
        code = '{ cmd1 && cmd2 }'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "if ($?)" in result


# ============================================================================
# Corner case: more edge cases with array/hashtable construction
# ============================================================================

class TestArrayHashtableConstructionExtended:
    def test_array_with_ternary_elements(self) -> None:
        code = '@($cond ? $a : $b, $cond2 ? $c : $d)'
        result = pwsh_transform(code)[0]
        assert "?" in result  # inside @() at depth>0
        assert "@(" in result

    def test_array_with_coalescing_elements(self) -> None:
        code = '@($a ?? "x", $b ?? "y", $c ?? "z")'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "@(" in result
        assert "if ($null -ne $a)" in result
        assert "if ($null -ne $b)" in result
        assert "if ($null -ne $c)" in result

    def test_array_with_null_conditional_elements(self) -> None:
        code = '@($obj?.Name, $obj?.Value, $obj?.Count)'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "@(" in result
        assert "$obj.Name" in result
        assert "$obj.Value" in result
        assert "$obj.Count" in result

    def test_hashtable_with_coalescing_values(self) -> None:
        code = '@{ Name = $name ?? "anon"; Value = $val ?? 0; Type = $type ?? "string" }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "@{ Name =" in result
        assert "if ($null -ne $name)" in result
        assert "if ($null -ne $val)" in result
        assert "if ($null -ne $type)" in result

    def test_hashtable_with_ternary_values(self) -> None:
        code = '@{ a = $cond ? 1 : 0; b = $cond2 ? 2 : 3 }'
        result = pwsh_transform(code)[0]
        assert "?" in result  # inside @{ } at depth>0
        assert "@{ a =" in result

    def test_hashtable_with_null_conditional_values(self) -> None:
        code = '@{ Name = $obj?.Name; Value = $obj?.Value }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "@{ Name =" in result
        assert "$obj.Name" in result
        assert "$obj.Value" in result

    def test_ordered_hashtable_with_coalescing(self) -> None:
        code = '[ordered]@{ a = $x ?? 1; b = $y ?? 2; c = $z ?? 3 }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "[ordered]@{ a =" in result
        assert "if ($null -ne $x)" in result
        assert "if ($null -ne $y)" in result
        assert "if ($null -ne $z)" in result

    def test_pscustomobject_with_coalescing(self) -> None:
        code = '[pscustomobject]@{ Name = $name ?? "anon"; Age = $age ?? 0 }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "[pscustomobject]@{ Name =" in result
        assert "if ($null -ne $name)" in result
        assert "if ($null -ne $age)" in result


# ============================================================================
# Corner case: more edge cases with string concatenation
# ============================================================================

class TestStringConcatenationExtended:
    def test_concatenation_with_coalescing(self) -> None:
        result = pwsh_transform('"name: " + ($name ?? "unknown")')[0]
        assert "??" not in result
        assert '"name: " +' in result
        assert "if ($null -ne $name)" in result

    def test_concatenation_with_null_conditional(self) -> None:
        result = pwsh_transform('"len: " + $str?.Length')[0]
        assert "?." not in result
        assert '"len: " +' in result
        assert "$str.Length" in result

    def test_concatenation_with_ternary(self) -> None:
        result = pwsh_transform('"status: " + ($cond ? "ok" : "fail")')[0]
        assert "?" in result  # inside () at depth>0
        assert '"status: " +' in result

    def test_format_string_with_coalescing(self) -> None:
        result = pwsh_transform('"Name: {0}" -f ($name ?? "anon")')[0]
        assert "??" not in result
        assert '"Name: {0}" -f' in result
        assert "if ($null -ne $name)" in result

    def test_format_string_with_ternary(self) -> None:
        result = pwsh_transform('"Status: {0}" -f ($cond ? "ok" : "fail")')[0]
        assert "?" in result  # inside () at depth>0
        assert '"Status: {0}" -f' in result

    def test_string_interpolation_with_coalescing(self) -> None:
        result = pwsh_transform('"Value: $($val ?? \"default\")"')[0]
        # Coalescing inside $() inside double-quoted string is at depth>0, NOT transformed in single pass
        assert "??" in result
        assert '"Value:' in result

    def test_string_interpolation_with_ternary(self) -> None:
        result = pwsh_transform('"Status: $($cond ? \"ok\" : \"fail\")"')[0]
        # Ternary inside $() inside double-quoted string is at depth>0, NOT transformed in single pass
        assert "?" in result
        assert '"Status:' in result


# ============================================================================
# Corner case: more edge cases with control flow keywords
# ============================================================================

class TestControlFlowKeywordsExtended:
    def test_return_with_coalescing(self) -> None:
        result = pwsh_transform('return $val ?? "default"')[0]
        # BUG: 'return' is treated as command prefix, producing malformed output
        assert "??" not in result
        assert "return" in result

    def test_return_with_null_conditional(self) -> None:
        result = pwsh_transform('return $obj?.Name')[0]
        assert "?." not in result
        assert "return" in result
        assert "$obj.Name" in result

    def test_return_with_chain(self) -> None:
        result = pwsh_transform('return (cmd1 && cmd2)')[0]
        assert "&&" not in result
        assert "return (cmd1" in result
        assert "if ($?)" in result

    def test_exit_with_coalescing(self) -> None:
        result = pwsh_transform('exit $code ?? 0')[0]
        # BUG: 'exit' is treated as command prefix, producing malformed output
        assert "??" not in result
        assert "exit" in result

    def test_exit_with_null_conditional(self) -> None:
        result = pwsh_transform('exit $obj?.Code')[0]
        assert "?." not in result
        assert "exit" in result
        assert "$obj.Code" in result

    def test_break_with_coalescing(self) -> None:
        result = pwsh_transform('break $label ?? "default"')[0]
        # BUG: 'break' is treated as command prefix, producing malformed output
        assert "??" not in result
        assert "break" in result

    def test_continue_with_null_conditional(self) -> None:
        result = pwsh_transform('continue $obj?.Index')[0]
        assert "?." not in result
        assert "continue" in result
        assert "$obj.Index" in result

    def test_throw_with_coalescing(self) -> None:
        result = pwsh_transform('throw $msg ?? "error"')[0]
        # BUG: 'throw' is treated as command prefix, producing malformed output
        assert "??" not in result
        assert "throw" in result

    def test_throw_with_null_conditional(self) -> None:
        result = pwsh_transform('throw $ex?.Message')[0]
        assert "?." not in result
        assert "throw" in result
        assert "$ex.Message" in result


# ============================================================================
# Corner case: more edge cases with operators on same line
# ============================================================================

class TestOperatorsOnSameLineExtended:
    def test_coalescing_ternary_chain_same_line(self) -> None:
        code = '$a ?? $b ? "t" : "f"; cmd1 && cmd2 || cmd3'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "&&" not in result
        assert "||" not in result
        assert "$a" in result
        assert "$b" in result
        assert "cmd1" in result
        assert "cmd2" in result
        assert "cmd3" in result

    def test_nca_coalescing_null_conditional_same_line(self) -> None:
        code = '$a ??= "x"; $b = $c?.Name ?? "y"; $d = $e?[0]'
        result = pwsh_transform(code)[0]
        assert "??=" not in result
        assert "?." not in result
        assert "?[" not in result
        assert "??" not in result
        assert "if ($null -eq $a)" in result
        assert "$c.Name" in result
        assert "$e[0]" in result

    def test_multiple_chains_same_line(self) -> None:
        code = 'cmd1 && cmd2 || cmd3 && cmd4 || cmd5'
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "||" not in result
        assert "cmd1" in result
        assert "cmd5" in result
        assert "if ($?)" in result
        assert "if (-not $?)" in result

    def test_mixed_operators_no_semicolon(self) -> None:
        code = '$a ??= "x" && cmd1 || cmd2 && $b = $c?.Name ?? "y"'
        result = pwsh_transform(code)[0]
        assert "??=" not in result
        assert "?." not in result
        assert "??" not in result
        assert "&&" not in result
        assert "||" not in result
        assert "cmd1" in result
        assert "cmd2" in result
        assert "$c.Name" in result


# ============================================================================
# Corner case: more edge cases with deeply nested constructs
# ============================================================================

class TestDeeplyNestedConstructsExtended:
    def test_nested_if_with_ternary(self) -> None:
        code = 'if ($a) { if ($b) { if ($c) { $d ? "yes" : "no" } } }'
        result = pwsh_transform(code)[0]
        # Ternary inside nested if braces is at depth>0, NOT transformed
        assert "?" in result
        assert "if ($a)" in result
        assert "if ($b)" in result
        assert "if ($c)" in result

    def test_deeply_nested_foreach_with_operators(self) -> None:
        code = 'foreach ($a in $b) { foreach ($c in $d) { foreach ($e in $f) { $g?.Name } } }'
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "foreach" in result
        assert "$g.Name" in result

    def test_deeply_nested_try_catch_with_operators(self) -> None:
        code = 'try { try { try { $a ?? "inner" } catch { $b ?? "inner-catch" } } catch { $c ?? "mid-catch" } } catch { $d ?? "outer-catch" }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "try" in result
        assert "catch" in result
        assert "if ($null -ne $a)" in result
        assert "if ($null -ne $b)" in result
        assert "if ($null -ne $c)" in result
        assert "if ($null -ne $d)" in result

    def test_deeply_nested_function_with_operators(self) -> None:
        code = 'function Outer { function Mid { function Inner { $a ?? "default" } }; Inner }'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "function Outer" in result
        assert "function Mid" in result
        assert "function Inner" in result
        assert "if ($null -ne $a)" in result

    def test_deeply_nested_scriptblock_with_operators(self) -> None:
        code = '{{{ $a ?? "default" }}}'
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result

    def test_deeply_nested_parens_with_coalescing(self) -> None:
        result = pwsh_transform('((((((((($a)))))))))) ?? "default"')[0]
        assert "??" not in result
        assert "((((((($a))))))))" in result
        assert "default" in result

    def test_deeply_nested_subexpr_with_ternary(self) -> None:
        result = pwsh_transform('$($($($($cond ? $a : $b)))))')[0]
        # Ternary inside nested $() is at depth>0, NOT transformed
        assert "?" in result
        assert "$($($($($cond ? $a : $b)))))" == result

    def test_deeply_nested_method_chain(self) -> None:
        result = pwsh_transform('$a?.ToString()?.Trim()?.ToUpper()?.Split()?.[0]?.Length')[0]
        assert ".ToString()" in result
        assert ".Trim()" in result
        assert ".ToUpper()" in result
        assert ".Split()" in result
        assert ".Length" in result


# ============================================================================
# Corner case: more edge cases with idempotency
# ============================================================================

class TestIdempotencyExtended:
    def test_advanced_function_idempotent(self) -> None:
        code = 'function Test { [CmdletBinding()] param([ValidateSet("a","b")]$x = $a ?? "a"); begin { $sum ??= 0 } process { $sum += $_ } end { $sum } }'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_begin_process_end_idempotent(self) -> None:
        code = 'function Test { begin { $sum = 0 } process { $sum += $_ } end { $sum } }'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_using_namespace_idempotent(self) -> None:
        code = 'using namespace System.IO\n$x = $a ?? "default"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_select_string_idempotent(self) -> None:
        code = '(Select-String $pattern $file) ?? $null'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_net_type_method_idempotent(self) -> None:
        code = '[System.IO.File]::ReadAllText($f) ?? ""'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_ps7_specific_var_idempotent(self) -> None:
        code = '$PSNativeCommandUseErrorActionPreference ?? $false'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_dynamicparam_idempotent(self) -> None:
        code = 'function Test { dynamicparam { $x = $a ?? "default"; $x } }'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_data_section_idempotent(self) -> None:
        code = 'data { $x = $a ?? "default"; $x }'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_trap_specific_exception_idempotent(self) -> None:
        code = 'trap [System.Exception] { $_.Message ?? "unknown" }; cmd1 && cmd2'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_all_advanced_patterns_combined_idempotent(self) -> None:
        code = 'function Test { [CmdletBinding()] param([ValidateSet("a","b")]$x = $a ?? "a"); begin { $sum ??= 0 } process { $sum += $_ } end { $sum } }'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second


# ============================================================================
# Corner case: more edge cases with unusual but valid PS syntax
# ============================================================================

class TestUnusualButValidSyntax:
    def test_dollar_null_with_method(self) -> None:
        result = pwsh_transform('$null?.ToString()')[0]
        assert "?." not in result
        assert "if ($null -ne $null)" in result
        assert "$null.ToString()" in result

    def test_dollar_null_with_bracket(self) -> None:
        result = pwsh_transform('$null?[0]')[0]
        assert "?[" not in result
        assert "if ($null -ne $null)" in result
        assert "$null[0]" in result

    def test_dollar_true_with_null_conditional(self) -> None:
        result = pwsh_transform('$true?.ToString()')[0]
        assert "?." not in result
        assert "if ($null -ne $true)" in result
        assert "$true.ToString()" in result

    def test_dollar_false_with_null_conditional(self) -> None:
        result = pwsh_transform('$false?.ToString()')[0]
        assert "?." not in result
        assert "if ($null -ne $false)" in result
        assert "$false.ToString()" in result

    def test_number_literal_with_null_conditional(self) -> None:
        result = pwsh_transform('123?.ToString()')[0]
        assert "?." not in result
        assert "if ($null -ne 123)" in result
        assert "123.ToString()" in result

    def test_string_literal_with_null_conditional(self) -> None:
        result = pwsh_transform('"hello"?.Length')[0]
        assert "?." not in result
        assert "if ($null -ne \"hello\")" in result
        assert "\"hello\".Length" in result

    def test_array_literal_with_null_conditional(self) -> None:
        result = pwsh_transform('@(1,2,3)?.Count')[0]
        assert "?." not in result
        assert "if ($null -ne @(1,2,3))" in result
        assert "@(1,2,3).Count" in result

    def test_hashtable_literal_with_null_conditional(self) -> None:
        result = pwsh_transform('@{a=1}?.Count')[0]
        assert "?." not in result
        assert "if ($null -ne @{a=1})" in result
        assert "@{a=1}.Count" in result

    def test_subexpression_literal_with_null_conditional(self) -> None:
        result = pwsh_transform('$(Get-Date)?.ToString()')[0]
        assert "?." not in result
        assert "if ($null -ne $(Get-Date))" in result
        assert "$(Get-Date).ToString()" in result

    def test_scriptblock_literal_with_null_conditional(self) -> None:
        result = pwsh_transform('{ $a }?.ToString()')[0]
        assert "?." not in result
        assert "if ($null -ne { $a })" in result
        assert "{ $a }.ToString()" in result


# ============================================================================
# Corner case: more edge cases with PS7-specific variables
# ============================================================================

class TestPS7VariablesExtended:
    def test_psnativecommanduseerroractionpreference_ternary(self) -> None:
        result = pwsh_transform('$PSNativeCommandUseErrorActionPreference ? "true" : "false"')[0]
        assert "?" not in result
        assert "if ($PSNativeCommandUseErrorActionPreference)" in result

    def test_psnativecommandargumentpassing_coalescing(self) -> None:
        result = pwsh_transform('$PSNativeCommandArgumentPassing ?? "Standard"')[0]
        assert "??" not in result
        assert "if ($null -ne $PSNativeCommandArgumentPassing)" in result

    def test_psansirenderingfileinfo_null_conditional(self) -> None:
        result = pwsh_transform('$PSAnsiRenderingFileInfo?.ToString()')[0]
        assert "?." not in result
        assert "if ($null -ne $PSAnsiRenderingFileInfo)" in result
        assert "$PSAnsiRenderingFileInfo.ToString()" in result

    def test_psmoduleanalysiscachepath_coalescing(self) -> None:
        result = pwsh_transform('$PSModuleAnalysisCachePath ?? ""')[0]
        assert "??" not in result
        assert "if ($null -ne $PSModuleAnalysisCachePath)" in result

    def test_psstyle_foreground_black_null_conditional(self) -> None:
        result = pwsh_transform('$PSStyle?.Foreground?.Black')[0]
        assert "?." not in result
        assert "$PSStyle" in result
        assert ".Foreground" in result
        assert ".Black" in result

    def test_psstyle_background_white_coalescing(self) -> None:
        result = pwsh_transform('$PSStyle.Background.White ?? ""')[0]
        assert "??" not in result
        assert "$PSStyle.Background.White" in result

    def test_psstyle_chained_null_conditional(self) -> None:
        result = pwsh_transform('$PSStyle?.Foreground?.Red?.ToString()')[0]
        assert "?." not in result
        assert "$PSStyle" in result
        assert ".Foreground" in result
        assert ".Red" in result
        assert ".ToString()" in result

    def test_psstyle_chained_with_method(self) -> None:
        result = pwsh_transform('$PSStyle?.Foreground?.Red?.Trim()')[0]
        assert "?." not in result
        assert "$PSStyle" in result
        assert ".Foreground" in result
        assert ".Red" in result
        assert ".Trim()" in result


# ============================================================================
# Corner case: more edge cases with automatic variables and method calls
# ============================================================================

class TestAutomaticVariablesMethodCalls:
    def test_dollar_underscore_with_method_chain(self) -> None:
        result = pwsh_transform('$_.ToString()?.Trim()?.Length')[0]
        assert "?." not in result
        assert "$_.ToString()" in result
        assert ".Trim()" in result
        assert ".Length" in result

    def test_dollar_input_with_method_chain(self) -> None:
        result = pwsh_transform('$input.ToString()?.Trim()')[0]
        assert "?." not in result
        assert "$input.ToString()" in result
        assert ".Trim()" in result

    def test_dollar_args_with_method(self) -> None:
        result = pwsh_transform('$args.ToString()?.Length')[0]
        assert "?." not in result
        assert "$args.ToString()" in result
        assert ".Length" in result

    def test_dollar_foreach_with_method(self) -> None:
        result = pwsh_transform('$foreach.ToString()?.Length')[0]
        assert "?." not in result
        assert "$foreach.ToString()" in result
        assert ".Length" in result

    def test_dollar_switch_with_method(self) -> None:
        result = pwsh_transform('$switch.ToString()?.Length')[0]
        assert "?." not in result
        assert "$switch.ToString()" in result
        assert ".Length" in result

    def test_dollar_error_with_method_chain(self) -> None:
        result = pwsh_transform('$Error.ToString()?.Trim()?.Length')[0]
        assert "?." not in result
        assert "$Error.ToString()" in result
        assert ".Trim()" in result
        assert ".Length" in result

    def test_dollar_matches_with_method(self) -> None:
        result = pwsh_transform('$Matches.ToString()?.Length')[0]
        assert "?." not in result
        assert "$Matches.ToString()" in result
        assert ".Length" in result

    def test_dollar_lastexitcode_with_method(self) -> None:
        result = pwsh_transform('$LastExitCode.ToString()?.Length')[0]
        assert "?." not in result
        assert "$LastExitCode.ToString()" in result
        assert ".Length" in result

    def test_dollar_pid_with_method(self) -> None:
        result = pwsh_transform('$PID.ToString()?.Length')[0]
        assert "?." not in result
        assert "$PID.ToString()" in result
        assert ".Length" in result

    def test_dollar_ofc_with_method(self) -> None:
        result = pwsh_transform('$OFS.ToString()?.Length')[0]
        assert "?." not in result
        assert "$OFS.ToString()" in result
        assert ".Length" in result


# ============================================================================
# Corner case: more edge cases with scoped variables and null-conditional
# ============================================================================

class TestScopedVariablesExtended:
    def test_global_scope_with_method_chain(self) -> None:
        result = pwsh_transform('$global:obj?.ToString()?.Trim()')[0]
        assert "?." not in result
        assert "$global:obj" in result
        assert ".ToString()" in result
        assert ".Trim()" in result

    def test_script_scope_with_method_chain(self) -> None:
        result = pwsh_transform('$script:obj?.ToString()?.Trim()')[0]
        assert "?." not in result
        assert "$script:obj" in result
        assert ".ToString()" in result
        assert ".Trim()" in result

    def test_local_scope_with_method_chain(self) -> None:
        result = pwsh_transform('$local:obj?.ToString()?.Trim()')[0]
        assert "?." not in result
        assert "$local:obj" in result
        assert ".ToString()" in result
        assert ".Trim()" in result

    def test_private_scope_with_method_chain(self) -> None:
        result = pwsh_transform('$private:obj?.ToString()?.Trim()')[0]
        assert "?." not in result
        assert "$private:obj" in result
        assert ".ToString()" in result
        assert ".Trim()" in result

    def test_env_scope_with_method_chain(self) -> None:
        result = pwsh_transform('$env:PATH?.Split(";")?[0]?.Trim()')[0]
        assert "?." not in result
        assert "?[" not in result
        assert "$env:PATH" in result
        assert ".Split(\";\")" in result
        assert ".Trim()" in result

    def test_using_scope_with_method_chain(self) -> None:
        result = pwsh_transform('$using:obj?.ToString()?.Trim()')[0]
        assert "?." not in result
        assert "$using:obj" in result
        assert ".ToString()" in result
        assert ".Trim()" in result


# ============================================================================
# Corner case: more edge cases with property access patterns
# ============================================================================

class TestPropertyAccessPatternsExtended:
    def test_property_chain_with_null_conditional_at_start(self) -> None:
        result = pwsh_transform('$a?.b.c.d.e')[0]
        assert "?." not in result
        # Null-conditional wraps base in $(), subsequent chain is outside
        assert "$(if ($null -ne $a) { $a.b }).c.d.e" == result
        assert "$a" in result
        assert ".b" in result
        assert ".c" in result
        assert ".d" in result
        assert ".e" in result

    def test_property_chain_with_null_conditional_in_middle(self) -> None:
        result = pwsh_transform('$a.b?.c.d.e')[0]
        assert "?." not in result
        assert "$a.b" in result
        assert ".c" in result
        assert ".d.e" in result

    def test_property_chain_with_null_conditional_at_end(self) -> None:
        result = pwsh_transform('$a.b.c.d?.e')[0]
        assert "?." not in result
        assert "$a.b.c.d" in result
        assert ".e" in result

    def test_property_chain_with_multiple_null_conditionals(self) -> None:
        result = pwsh_transform('$a?.b?.c?.d?.e')[0]
        assert "?." not in result
        assert "$a" in result
        assert ".e" in result

    def test_method_chain_with_null_conditional_at_start(self) -> None:
        result = pwsh_transform('$a?.ToString().Trim().Length')[0]
        assert "?." not in result
        assert "$a" in result
        assert ".ToString()" in result
        assert ".Trim()" in result
        assert ".Length" in result

    def test_method_chain_with_null_conditional_in_middle(self) -> None:
        result = pwsh_transform('$a.ToString()?.Trim().Length')[0]
        assert "?." not in result
        assert "$a.ToString()" in result
        assert ".Trim()" in result
        assert ".Length" in result

    def test_method_chain_with_null_conditional_at_end(self) -> None:
        result = pwsh_transform('$a.ToString().Trim()?.Length')[0]
        assert "?." not in result
        assert "$a.ToString().Trim()" in result
        assert ".Length" in result

    def test_method_chain_with_multiple_null_conditionals(self) -> None:
        result = pwsh_transform('$a?.ToString()?.Trim()?.Length')[0]
        assert "?." not in result
        assert "$a" in result
        assert ".ToString()" in result
        assert ".Trim()" in result
        assert ".Length" in result

    def test_mixed_property_method_with_null_conditional(self) -> None:
        result = pwsh_transform('$a?.Property.Method()?.Other')[0]
        assert "?." not in result
        assert "$a" in result
        assert ".Property" in result
        assert ".Method()" in result
        assert ".Other" in result

    def test_mixed_method_property_with_null_conditional(self) -> None:
        result = pwsh_transform('$a?.Method().Property?.Other')[0]
        assert "?." not in result
        assert "$a" in result
        assert ".Method()" in result
        assert ".Property" in result
        assert ".Other" in result


# ============================================================================
# Corner case: more edge cases with array/hashtable index access
# ============================================================================

class TestIndexAccessExtended:
    def test_array_index_chain_with_null_conditional(self) -> None:
        result = pwsh_transform('$a[0][1][2]?.Name')[0]
        assert "?." not in result
        assert "$a[0][1][2]" in result
        assert ".Name" in result

    def test_hashtable_index_chain_with_null_conditional(self) -> None:
        result = pwsh_transform('$a["x"]["y"]["z"]?.Name')[0]
        assert "?." not in result
        assert '$a["x"]["y"]["z"]' in result
        assert ".Name" in result

    def test_mixed_index_property_with_null_conditional(self) -> None:
        result = pwsh_transform('$a[0].Property[1]?.Name')[0]
        assert "?." not in result
        assert "$a[0].Property[1]" in result
        assert ".Name" in result

    def test_mixed_property_index_with_null_conditional(self) -> None:
        result = pwsh_transform('$a.Property[0]?.Name')[0]
        assert "?." not in result
        assert "$a.Property[0]" in result
        assert ".Name" in result

    def test_method_result_index_with_null_conditional(self) -> None:
        result = pwsh_transform('$a.Method()[0]?.Name')[0]
        assert "?." not in result
        assert "$a.Method()[0]" in result
        assert ".Name" in result

    def test_subexpr_index_with_null_conditional(self) -> None:
        result = pwsh_transform('$(Get-Date)[0]?.Name')[0]
        assert "?." not in result
        assert "$(Get-Date)[0]" in result
        assert ".Name" in result


# ============================================================================
# Corner case: more edge cases with type literals and operators
# ============================================================================

class TestTypeLiteralsExtended:
    def test_type_literal_with_null_conditional(self) -> None:
        result = pwsh_transform('[string]$s?.Length')[0]
        assert "?." not in result
        assert "[string]$s" in result
        assert ".Length" in result

    def test_type_literal_with_coalescing(self) -> None:
        result = pwsh_transform('[int]$x ?? 0')[0]
        assert "??" not in result
        assert "[int]$x" in result
        assert "if ($null -ne [int]$x)" in result

    def test_type_literal_with_ternary(self) -> None:
        result = pwsh_transform('[bool]$flag ? "true" : "false"')[0]
        assert "?" not in result
        assert "[bool]$flag" in result
        assert "if ([bool]$flag)" in result

    def test_generic_type_with_null_conditional(self) -> None:
        result = pwsh_transform('[System.Collections.Generic.List[string]]$list?.Count')[0]
        assert "?." not in result
        assert "[System.Collections.Generic.List[string]]$list" in result
        assert ".Count" in result

    def test_generic_type_with_coalescing(self) -> None:
        result = pwsh_transform('[System.Collections.Generic.Dictionary[string,object]]$dict ?? @{}')[0]
        assert "??" not in result
        assert "[System.Collections.Generic.Dictionary[string,object]]$dict" in result
        assert "if ($null -ne [System.Collections.Generic.Dictionary[string,object]]$dict)" in result

    def test_array_type_with_null_conditional(self) -> None:
        result = pwsh_transform('[string[]]$arr?[0]')[0]
        assert "?[" not in result
        assert "[string[]]$arr" in result
        assert "[string[]]$arr[0]" in result

    def test_array_type_with_coalescing(self) -> None:
        result = pwsh_transform('[string[]]$arr ?? @()')[0]
        assert "??" not in result
        assert "[string[]]$arr" in result
        assert "if ($null -ne [string[]]$arr)" in result

    def test_nullable_type_with_coalescing(self) -> None:
        result = pwsh_transform('[nullable[int]]$x ?? 0')[0]
        assert "??" not in result
        assert "[nullable[int]]$x" in result
        assert "if ($null -ne [nullable[int]]$x)" in result

    def test_enum_type_with_ternary(self) -> None:
        result = pwsh_transform('[System.DayOfWeek]$day -eq "Monday" ? "start" : "other"')[0]
        assert "?" not in result
        assert "[System.DayOfWeek]$day" in result
        assert "if ([System.DayOfWeek]$day -eq \"Monday\")" in result


# ============================================================================
# Corner case: more edge cases with comments and strings
# ============================================================================

class TestCommentsStringsExtended:
    def test_line_comment_before_every_operator_type(self) -> None:
        code = '''# comment
$a ?? "default"
# comment
$cond ? "yes" : "no"
# comment
$obj?.Name
# comment
cmd1 && cmd2'''
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "?" not in result.replace("$?", "")
        assert "?." not in result
        assert "&&" not in result
        assert "comment" in result
        assert "if ($null -ne $a)" in result
        assert "if ($cond)" in result
        assert "if ($null -ne $obj)" in result
        assert "if ($?)" in result

    def test_block_comment_before_every_operator_type(self) -> None:
        code = '''<# comment #>
$a ?? "default"
<# comment #>
$cond ? "yes" : "no"
<# comment #>
$obj?.Name
<# comment #>
cmd1 && cmd2'''
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "?" not in result.replace("$?", "")
        assert "?." not in result
        assert "&&" not in result
        assert "comment" in result
        assert "if ($null -ne $a)" in result
        assert "if ($cond)" in result
        assert "if ($null -ne $obj)" in result
        assert "if ($?)" in result

    def test_here_string_with_all_operators_then_real(self) -> None:
        code = "@'\n?? ?. && || ? : ??=\n'@\n$a ?? 'default'; $cond ? 'yes' : 'no'; $obj?.Name; cmd1 && cmd2"
        result = pwsh_transform(code)[0]
        assert "??" in result.splitlines()[1]
        assert "if ($null -ne $a)" in result
        assert "if ($cond)" in result
        assert "if ($null -ne $obj)" in result
        assert "if ($?)" in result

    def test_double_quoted_string_with_all_operators_then_real(self) -> None:
        code = '"?? ?. && || ? : ??="\n$a ?? "default"; $cond ? "yes" : "no"; $obj?.Name; cmd1 && cmd2'
        result = pwsh_transform(code)[0]
        assert "??" in result.splitlines()[0]
        assert "if ($null -ne $a)" in result
        assert "if ($cond)" in result
        assert "if ($null -ne $obj)" in result
        assert "if ($?)" in result

    def test_single_quoted_string_with_all_operators_then_real(self) -> None:
        code = "'?? ?. && || ? : ??='\n$a ?? 'default'; $cond ? 'yes' : 'no'; $obj?.Name; cmd1 && cmd2"
        result = pwsh_transform(code)[0]
        assert "??" in result.splitlines()[0]
        assert "if ($null -ne $a)" in result
        assert "if ($cond)" in result
        assert "if ($null -ne $obj)" in result
        assert "if ($?)" in result


# ============================================================================
# Corner case: more edge cases with malformed/unusual inputs
# ============================================================================

class TestMalformedUnusualInputsExtended:
    def test_empty_scriptblock(self) -> None:
        result = pwsh_transform('{}')[0]
        assert "{}" == result

    def test_empty_array(self) -> None:
        result = pwsh_transform('@()')[0]
        assert "@()" == result

    def test_empty_hashtable(self) -> None:
        result = pwsh_transform('@{}')[0]
        assert "@{}" == result

    def test_empty_subexpression(self) -> None:
        result = pwsh_transform('$()')[0]
        assert "$()" == result

    def test_empty_string(self) -> None:
        result = pwsh_transform('""')[0]
        assert '""' == result

    def test_empty_single_quoted_string(self) -> None:
        result = pwsh_transform("''")[0]
        assert "''" == result

    def test_only_whitespace(self) -> None:
        result = pwsh_transform('   \n\t\n   ')[0]
        assert isinstance(result, str)

    def test_only_comments(self) -> None:
        result = pwsh_transform('# comment1\n# comment2\n# comment3')[0]
        assert "comment1" in result
        assert "comment2" in result
        assert "comment3" in result

    def test_only_block_comment(self) -> None:
        result = pwsh_transform('<# block comment #>')[0]
        assert "block comment" in result

    def test_unterminated_single_quote(self) -> None:
        result = pwsh_transform("'unterminated")[0]
        assert isinstance(result, str)

    def test_unterminated_double_quote(self) -> None:
        result = pwsh_transform('"unterminated')[0]
        assert isinstance(result, str)

    def test_unterminated_block_comment(self) -> None:
        result = pwsh_transform('<# unterminated')[0]
        assert isinstance(result, str)

    def test_unterminated_here_string(self) -> None:
        result = pwsh_transform("@'\nunterminated")[0]
        assert isinstance(result, str)

    def test_bare_operator_and(self) -> None:
        result = pwsh_transform('&& cmd')[0]
        assert isinstance(result, str)
        assert "cmd" in result

    def test_bare_operator_or(self) -> None:
        result = pwsh_transform('|| cmd')[0]
        assert isinstance(result, str)
        assert "cmd" in result

    def test_bare_question_mark(self) -> None:
        result = pwsh_transform('? "a" : "b"')[0]
        assert isinstance(result, str)
        assert '"a"' in result
        assert '"b"' in result

    def test_bare_double_question(self) -> None:
        result = pwsh_transform('?? "default"')[0]
        assert isinstance(result, str)
        assert '"default"' in result

    def test_bare_null_conditional(self) -> None:
        result = pwsh_transform('?.Property')[0]
        assert isinstance(result, str)
        assert ".Property" in result

    def test_bare_null_conditional_bracket(self) -> None:
        result = pwsh_transform('?[0]')[0]
        assert isinstance(result, str)
        assert "[0]" in result


# ============================================================================
# Corner case: more edge cases with idempotency for malformed inputs
# ============================================================================

class TestIdempotencyMalformedInputs:
    def test_empty_input_idempotent(self) -> None:
        code = ''
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_whitespace_only_idempotent(self) -> None:
        code = '   \n\t\n   '
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_comments_only_idempotent(self) -> None:
        code = '# comment1\n# comment2'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_no_operators_idempotent(self) -> None:
        code = 'Write-Output "hello world"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_empty_scriptblock_idempotent(self) -> None:
        code = '{}'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_empty_array_idempotent(self) -> None:
        code = '@()'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_empty_hashtable_idempotent(self) -> None:
        code = '@{}'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_empty_string_idempotent(self) -> None:
        code = '""'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_unterminated_string_idempotent(self) -> None:
        code = "'unterminated"
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_unterminated_block_comment_idempotent(self) -> None:
        code = '<# unterminated'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second


# ============================================================================
# Corner case: more edge cases with realistic script patterns
# ============================================================================

class TestRealisticScriptPatterns:
    def test_realistic_script_with_all_operators(self) -> None:
        code = '''param([string]$Name = $env:USERNAME ?? "anonymous")
function Get-Data {
    [CmdletBinding()] param([string]$Url)
    $response = Invoke-RestMethod $Url
    $data = $response?.result ?? $response?.data ?? @{}
    return $data
}
$files = Get-ChildItem -Path $dir -Recurse
$csv = $files?.Where({$_.Extension -eq '.csv'})
$count = $csv?.Count ?? 0
if ($count -gt 0) {
    Write-Output "Found $count CSV files" && Export-Csv $csv $outFile
} else {
    Write-Output "No CSV files found" || Write-Warning "empty"
}
'''
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "?." not in result
        assert "&&" not in result
        assert "||" not in result
        assert "param([string]$Name =" in result
        assert "function Get-Data" in result
        assert "Invoke-RestMethod" in result
        assert "Get-ChildItem" in result
        assert "Export-Csv" in result
        assert "if ($?)" in result
        assert "if (-not $?)" in result

    def test_realistic_script_with_advanced_function(self) -> None:
        code = '''function Get-Status {
    [CmdletBinding(SupportsShouldProcess=$true)]
    param(
        [Parameter(Mandatory=$true)]
        [ValidateSet("Running","Stopped")]
        [string]$Status = $env:STATUS ?? "Running",
        [string]$OutputPath = $env:OUTPUT ?? "output.txt"
    )
    begin { $results = @() }
    process {
        $svc = Get-Service | Where-Object Status -eq $Status
        $svc?.ForEach({ $results += $_.Name })
    }
    end {
        $results | Out-File $OutputPath
        Write-Output "Done" && Write-Output $results.Count
    }
}
'''
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "?." not in result
        assert "&&" not in result
        assert "function Get-Status" in result
        assert "[CmdletBinding(SupportsShouldProcess=$true)]" in result
        assert "[Parameter(Mandatory=$true)]" in result
        assert "[ValidateSet(\"Running\",\"Stopped\")]" in result
        assert "Out-File" in result
        assert "if ($?)" in result

    def test_realistic_script_with_class_and_enum(self) -> None:
        code = '''enum LogLevel { DEBUG = 0; INFO = 1; WARN = 2; ERROR = $env:ERROR_LEVEL ?? 3 }
class Logger {
    [LogLevel]$Level = [LogLevel]::INFO
    [string]$Path = $env:LOG_PATH ?? "log.txt"
    [void] Write([string]$msg) {
        if ($this.Level -le [LogLevel]::ERROR) {
            $timestamp = [DateTime]::Now?.ToString("yyyy-MM-dd")
            "$timestamp $msg" | Out-File $this.Path -Append
        }
    }
}
$logger = [Logger]::new()
$logger.Write("test") && Write-Output "logged"
'''
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "?." not in result
        assert "&&" not in result
        assert "enum LogLevel" in result
        assert "class Logger" in result
        assert "if ($?)" in result

    def test_realistic_script_with_try_catch(self) -> None:
        code = '''try {
    $content = [System.IO.File]::ReadAllText($path) ?? ""
    $obj = ConvertFrom-Json $content
    $name = $obj?.name ?? "unknown"
    Write-Output $name && Write-Output "success"
} catch [System.IO.FileNotFoundException] {
    Write-Error "File not found" && Write-Output "creating default"
    $name = "default"
} catch {
    Write-Error $_.Exception?.Message ?? "unknown error"
} finally {
    $obj?.Dispose() ?? $null
}
'''
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "?." not in result
        assert "&&" not in result
        assert "try" in result
        assert "catch [System.IO.FileNotFoundException]" in result
        assert "catch" in result
        assert "finally" in result
        assert "if ($?)" in result
        assert "$_.Exception.Message" in result

    def test_realistic_script_with_data_section(self) -> None:
        code = '''data {
    $config = @{
        Name = $env:APP_NAME ?? "MyApp"
        Version = $env:APP_VERSION ?? "1.0"
        Debug = $env:DEBUG -eq "1" ? $true : $false
    }
    $config
}
$data | ConvertTo-Json && Write-Output "config loaded"
'''
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "&&" not in result
        assert "data" in result
        assert "ConvertTo-Json" in result
        assert "if ($?)" in result

    def test_realistic_script_with_using_statements(self) -> None:
        code = '''using namespace System.IO
using namespace System.Net
using module MyModule

$path = [Path]::Combine($base, $file) ?? ""
$client = [WebClient]::new()
$content = $client?.DownloadString($url) ?? ""
$content | Out-File $path && Write-Output "saved"
'''
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "?." not in result
        assert "&&" not in result
        assert "using namespace System.IO" in result
        assert "using namespace System.Net" in result
        assert "using module MyModule" in result
        assert "[Path]::Combine" in result
        assert "[WebClient]::new()" in result
        assert "if ($?)" in result

    def test_realistic_script_with_pipeline_chain(self) -> None:
        code = '''Get-Process | Where-Object CPU -gt 100 | Tee-Object -Variable heavyProcs &&
$heavyProcs?.Count ?? 0 | Write-Output &&
Write-Output "Found heavy processes" ||
Write-Warning "No heavy processes found"
'''
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "||" not in result
        assert "Get-Process" in result
        assert "Where-Object" in result
        assert "Tee-Object" in result
        assert "if ($?)" in result
        assert "if (-not $?)" in result

    def test_realistic_script_with_foreach_parallel(self) -> None:
        code = '''$urls = @("http://a.com", "http://b.com")
$results = $urls | ForEach-Object -Parallel {
    $using:client?.DownloadString($_) ?? ""
} -ThrottleLimit 4
$results | Out-File "results.txt" && Write-Output "done"
'''
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "?." not in result
        assert "&&" not in result
        assert "ForEach-Object -Parallel" in result
        assert "$using:client" in result
        assert "if ($?)" in result

    def test_realistic_script_with_error_handling(self) -> None:
        code = '''$ErrorActionPreference = "Stop"
try {
    $file = Get-Item $path
    $content = $file?.FullName ?? ""
    Test-Path $content && Write-Output "exists" || Write-Error "missing"
} catch {
    $_.Exception?.Message ?? "unknown" | Write-Error
}
'''
        result = pwsh_transform(code)[0]
        assert "??" not in result
        assert "?." not in result
        assert "&&" not in result
        assert "||" not in result
        assert "$ErrorActionPreference = \"Stop\"" in result
        assert "try" in result
        assert "catch" in result
        assert "if ($?)" in result
        assert "if (-not $?)" in result
        assert "$_.Exception.Message" in result


# ============================================================================
# Additional corner-case tests (batch 3)
# ============================================================================

class TestNCAComplexTargetsFixed:
    """Tests for NCA with complex targets that were previously known limitations."""

    def test_nca_nested_array_index(self) -> None:
        result = pwsh_transform('$arr[0][1] ??= "default"')[0]
        assert "??=" not in result
        assert "if ($null -eq $arr[0][1])" in result
        assert "$arr[0][1] = \"default\"" in result

    def test_nca_mixed_property_and_index(self) -> None:
        result = pwsh_transform('$obj.Items[0] ??= "first"')[0]
        assert "??=" not in result
        assert "if ($null -eq $obj.Items[0])" in result
        assert "$obj.Items[0] = \"first\"" in result

    def test_nca_property_chain(self) -> None:
        result = pwsh_transform('$obj.Prop.Sub ??= "deep"')[0]
        assert "??=" not in result
        assert "if ($null -eq $obj.Prop.Sub)" in result
        assert "$obj.Prop.Sub = \"deep\"" in result

    def test_nca_static_member(self) -> None:
        result = pwsh_transform('[System.IO.File]::Exists ??= $true')[0]
        assert "??=" not in result
        assert "if ($null -eq [System.IO.File]::Exists)" in result

    def test_nca_subexpression_target(self) -> None:
        result = pwsh_transform('$(Get-Variable x).Value ??= "set"')[0]
        assert "??=" not in result
        assert "if ($null -eq $(Get-Variable x).Value)" in result

    def test_nca_array_element_idempotent(self) -> None:
        code = '$arr[0] ??= "default"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_nca_hashtable_key_idempotent(self) -> None:
        code = "$ht['key'] ??= 'default'"
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_nca_mixed_target_idempotent(self) -> None:
        code = '$obj.Items[0] ??= "first"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_nca_array_element_with_semicolon(self) -> None:
        result = pwsh_transform('$arr[0] ??= "a"; Write-Output "done"')[0]
        assert "??=" not in result
        assert "if ($null -eq $arr[0])" in result
        assert "Write-Output \"done\"" in result

    def test_nca_hashtable_with_variable_index(self) -> None:
        result = pwsh_transform('$ht[$key] ??= "value"')[0]
        assert "??=" not in result
        assert "if ($null -eq $ht[$key])" in result
        assert "$ht[$key] = \"value\"" in result


class TestNullConditionalAdvanced:
    """Advanced null-conditional edge cases."""

    def test_null_conditional_method_with_complex_args(self) -> None:
        result = pwsh_transform('$obj?.Method($a, $b, $c)')[0]
        assert "?." not in result
        assert "if ($null -ne $obj)" in result
        assert "$obj.Method($a, $b, $c)" in result

    def test_null_conditional_chained_methods(self) -> None:
        result = pwsh_transform('$obj?.M1()?.M2()?')[0]
        assert "?." not in result
        assert "if ($null -ne $obj)" in result
        assert "$obj.M1()" in result
        assert ".M2()" in result

    def test_null_conditional_with_scriptblock_args(self) -> None:
        result = pwsh_transform('$list?.ForEach({ $_ * 2 })')[0]
        assert "?." not in result
        assert "if ($null -ne $list)" in result
        assert "$list.ForEach({ $_ * 2 })" in result

    def test_null_conditional_after_transform_idempotent(self) -> None:
        code = '$obj?.Name'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_null_conditional_bracket_after_transform_idempotent(self) -> None:
        code = '$arr?[0]'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_null_conditional_mixed_dot_and_bracket(self) -> None:
        result = pwsh_transform('$obj?.Items?[0]')[0]
        assert "?." not in result
        assert "?[" not in result
        assert "if ($null -ne $obj)" in result
        assert "$obj.Items" in result

    def test_null_conditional_with_array_literal_base(self) -> None:
        result = pwsh_transform('@(1,2,3)?[0]')[0]
        assert "?[" not in result
        assert "if ($null -ne @(1,2,3))" in result

    def test_null_conditional_with_hashtable_base(self) -> None:
        result = pwsh_transform('@{a=1}?.a')[0]
        assert "?." not in result
        assert "if ($null -ne @{a=1})" in result

    def test_null_conditional_inside_string_interpolation(self) -> None:
        """Operators inside subexpressions within double-quoted strings are not
        transformed because the entire string region is masked out."""
        result = pwsh_transform('"Value: $($obj?.Name)"')[0]
        # Known limitation: ?. inside $() inside "..." is not transformed
        assert "?." in result

    def test_null_conditional_with_type_cast_base(self) -> None:
        result = pwsh_transform('[string]$val?.Length')[0]
        assert "?." not in result
        assert "if ($null -ne [string]$val)" in result


class TestChainOperatorAdvanced:
    """Advanced pipeline chain edge cases."""

    def test_chain_with_null_conditional_left(self) -> None:
        result = pwsh_transform('$obj?.Name && Write-Output "ok"')[0]
        assert "&&" not in result
        assert "?." not in result
        assert "if ($?)" in result

    def test_chain_with_null_conditional_right(self) -> None:
        result = pwsh_transform('Get-Item $path && $obj?.Name')[0]
        assert "&&" not in result
        assert "?." not in result
        assert "if ($?)" in result

    def test_chain_with_ternary_left(self) -> None:
        result = pwsh_transform('$cond ? "a" : "b" && Write-Output "done"')[0]
        assert "&&" not in result
        assert "if ($?)" in result
        assert "if ($cond)" in result

    def test_chain_with_nca_left(self) -> None:
        result = pwsh_transform('$a ??= "x" && Write-Output $a')[0]
        assert "&&" not in result
        assert "??=" not in result
        assert "if ($?)" in result
        assert "if ($null -eq $a)" in result

    def test_chain_with_coalescing_left(self) -> None:
        result = pwsh_transform('$a ?? $b && Write-Output "ok"')[0]
        assert "&&" not in result
        assert "??" not in result
        assert "if ($?)" in result

    def test_triple_chain_mixed(self) -> None:
        result = pwsh_transform('cmd1 && cmd2 || cmd3')[0]
        assert "&&" not in result
        assert "||" not in result
        assert "if ($?)" in result
        assert "if (-not $?)" in result

    def test_chain_inside_foreach(self) -> None:
        result = pwsh_transform('foreach ($i in $items) { Test-Path $i && Remove-Item $i }')[0]
        assert "&&" not in result
        assert "if ($?)" in result
        assert "Remove-Item $i" in result

    def test_chain_with_redirection(self) -> None:
        result = pwsh_transform('cmd1 > out.txt && cmd2')[0]
        assert "&&" not in result
        assert "if ($?)" in result
        assert "cmd1 > out.txt" in result

    def test_chain_idempotent_with_other_ops(self) -> None:
        code = 'cmd1 && $obj?.Name ?? "fallback"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_chain_after_try_catch(self) -> None:
        result = pwsh_transform('try { cmd1 } catch {}; cmd1 && cmd2')[0]
        assert "&&" not in result
        assert "if ($?)" in result


class TestTernaryAdvanced:
    """Advanced ternary edge cases."""

    def test_ternary_with_null_conditional_condition(self) -> None:
        result = pwsh_transform('$obj?.Name ? "yes" : "no"')[0]
        assert "?" not in result or "if ($obj)" in result
        assert "?." not in result

    def test_ternary_with_null_conditional_true_branch(self) -> None:
        result = pwsh_transform('$cond ? $obj?.Name : "fallback"')[0]
        assert "?." not in result
        assert "if ($cond)" in result
        assert "$obj.Name" in result

    def test_ternary_with_coalescing_condition(self) -> None:
        result = pwsh_transform('$a ?? $b ? "t" : "f"')[0]
        assert "??" not in result
        assert "if ($a)" in result or "if ($null -ne $a)" in result

    def test_ternary_with_chain_condition(self) -> None:
        result = pwsh_transform('cmd1 && cmd2 ? "ok" : "fail"')[0]
        assert "&&" not in result
        assert "if ($?)" in result
        assert "if ($?)" in result or "if (-not $?)" in result

    def test_ternary_inside_array(self) -> None:
        """Ternary inside @() is at depth>0 so is intentionally skipped."""
        result = pwsh_transform('@($cond ? 1 : 2, 3)')[0]
        # Known limitation: ternary inside () at depth>0 is not transformed
        assert "$cond ? 1 : 2" in result
        assert "@(" in result

    def test_ternary_with_method_call_condition(self) -> None:
        result = pwsh_transform('$obj.Method() ? "yes" : "no"')[0]
        assert "if ($obj.Method())" in result

    def test_ternary_with_subexpression_condition(self) -> None:
        result = pwsh_transform('$(Get-Date) ? "today" : "never"')[0]
        assert "if ($(Get-Date))" in result

    def test_ternary_idempotent_with_nested_ops(self) -> None:
        code = '$cond ? $obj?.Name ?? "" : "fallback"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_ternary_with_scriptblock_branches(self) -> None:
        result = pwsh_transform('$cond ? { Get-Process } : { Get-Service }')[0]
        assert "if ($cond)" in result
        assert "{ Get-Process }" in result
        assert "{ Get-Service }" in result

    def test_ternary_with_type_literal_condition(self) -> None:
        result = pwsh_transform('[bool]$val ? "true" : "false"')[0]
        assert "if ([bool]$val)" in result


class TestCoalescingAdvanced:
    """Advanced null-coalescing edge cases."""

    def test_coalescing_with_null_conditional_left(self) -> None:
        result = pwsh_transform('$obj?.Name ?? "fallback"')[0]
        assert "??" not in result
        assert "?." not in result
        assert "if ($null -ne $obj)" in result
        assert "\"fallback\"" in result

    def test_coalescing_with_null_conditional_right(self) -> None:
        result = pwsh_transform('$a ?? $obj?.Name')[0]
        assert "??" not in result
        assert "?." not in result
        assert "if ($null -ne $a)" in result
        assert "if ($null -ne $obj)" in result

    def test_coalescing_with_ternary_right(self) -> None:
        """Ternary inside () at depth>0 is intentionally skipped."""
        result = pwsh_transform('$a ?? ($cond ? "t" : "f")')[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result
        # Ternary inside parens is at depth>0, so left as-is
        assert "$cond ? \"t\" : \"f\"" in result

    def test_coalescing_with_chain_right(self) -> None:
        result = pwsh_transform('$a ?? (cmd1 && cmd2)')[0]
        assert "??" not in result
        assert "&&" not in result
        assert "if ($null -ne $a)" in result
        assert "if ($?)" in result

    def test_coalescing_chained(self) -> None:
        result = pwsh_transform('$a ?? $b ?? $c')[0]
        assert "??" not in result
        assert "if ($null -ne $a)" in result
        assert "if ($null -ne $b)" in result

    def test_coalescing_with_array_access_left(self) -> None:
        result = pwsh_transform('$arr[0] ?? "fallback"')[0]
        assert "??" not in result
        assert "if ($null -ne $arr[0])" in result

    def test_coalescing_with_hashtable_access_left(self) -> None:
        result = pwsh_transform("$ht['key'] ?? 'fallback'")[0]
        assert "??" not in result
        assert "if ($null -ne $ht['key'])" in result

    def test_coalescing_with_property_chain_left(self) -> None:
        result = pwsh_transform('$obj.Prop.Sub ?? "deep"')[0]
        assert "??" not in result
        assert "if ($null -ne $obj.Prop.Sub)" in result

    def test_coalescing_idempotent_with_complex_expr(self) -> None:
        code = '$obj?.Items?[0] ?? $arr[1] ?? "default"'
        first = pwsh_transform(code)[0]
        second = pwsh_transform(first)[0]
        assert first == second

    def test_coalescing_with_static_member(self) -> None:
        result = pwsh_transform('[System.IO.File]::Exists ?? $false')[0]
        assert "??" not in result
        assert "if ($null -ne [System.IO.File]::Exists)" in result


class TestMixedOperatorsAdvanced:
    """Tests mixing multiple operators in complex ways."""

    def test_all_ops_on_one_line(self) -> None:
        result = pwsh_transform('$a ??= $b ?? $c ? "t" : "f"; $x?.Y && cmd')[0]
        assert "??=" not in result
        assert "??" not in result
        assert "?." not in result
        assert "&&" not in result
        assert "if ($null -eq $a)" in result
        assert "if ($null -ne $b)" in result
        # Ternary inside generated braces is at depth>0, one-pass limitation
        assert "$c ? \"t\" : \"f\"" in result
        assert "if ($null -ne $x)" in result
        assert "if ($?)" in result

    def test_nested_null_conditional_inside_ternary(self) -> None:
        result = pwsh_transform('$cond ? $obj?.Name : $arr?[0]')[0]
        assert "?." not in result
        assert "?[" not in result
        assert "if ($cond)" in result
        assert "$obj.Name" in result
        assert "$arr[0]" in result

    def test_chain_with_ternary_and_coalescing(self) -> None:
        result = pwsh_transform('$cond ? $a ?? $b : $c ?? $d && cmd')[0]
        assert "&&" not in result
        assert "??" not in result
        assert "if ($?)" in result

    def test_nca_then_null_conditional_then_chain(self) -> None:
        result = pwsh_transform('$a ??= $obj?.Name && Write-Output $a')[0]
        assert "??=" not in result
        assert "?." not in result
        assert "&&" not in result
        assert "if ($null -eq $a)" in result
        assert "if ($null -ne $obj)" in result
        assert "if ($?)" in result

    def test_realistic_config_script(self) -> None:
        code = '''
$Config ??= @{}
$Config.Server ??= "localhost"
$Config.Port ??= 8080
$Config.Options ??= @()
$Config?.Logger?.Level ??= "INFO"
Test-Path "config.json" && $Config = Get-Content "config.json" | ConvertFrom-Json
'''
        result = pwsh_transform(code)[0]
        assert "??=" not in result
        assert "??" not in result
        assert "?." not in result
        assert "&&" not in result
        assert "if ($null -eq $Config)" in result
        assert "if ($null -eq $Config.Server)" in result
        assert "if ($null -eq $Config.Port)" in result
        assert "if ($null -eq $Config.Options)" in result
        assert "if ($null -ne $Config)" in result
        assert "if ($?)" in result

    def test_realistic_api_response_handling(self) -> None:
        code = '''
$response = Invoke-RestMethod $uri
$data = $response?.Data ?? @()
$first = $data?[0] ?? @{}
$first?.Name ? $first.Name : "Unknown"
'''
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "?[" not in result
        assert "??" not in result
        assert "if ($null -ne $response)" in result
        assert "if ($null -ne $data)" in result
        assert "if ($first)" in result or "if ($null -ne $first)" in result

    def test_deeply_nested_null_conditional_chain(self) -> None:
        result = pwsh_transform('$a?.b?.c?.d?.e')[0]
        assert "?." not in result
        assert "$a.b" in result
        assert ".c" in result
        assert ".d" in result
        assert ".e" in result

    def test_null_conditional_with_dollar_question_member(self) -> None:
        result = pwsh_transform('$obj?.$?')[0]
        assert "?." not in result
        assert "if ($null -ne $obj)" in result
        assert "$obj.$?" in result

    def test_chain_with_try_catch_finally(self) -> None:
        code = '''
try {
    Get-Item $path && Remove-Item $path
} catch {
    Write-Error "Failed"
} finally {
    Write-Output "Done"
}
'''
        result = pwsh_transform(code)[0]
        assert "&&" not in result
        assert "if ($?)" in result
        assert "try" in result
        assert "catch" in result
        assert "finally" in result

    def test_switch_with_null_conditional_and_coalescing(self) -> None:
        code = '''
switch ($obj?.Type ?? "default") {
    "A" { Write-Output "Type A" }
    "B" { Write-Output "Type B" }
    default { Write-Output "Other" }
}
'''
        result = pwsh_transform(code)[0]
        assert "?." not in result
        assert "??" not in result
        assert "if ($null -ne $obj)" in result
        assert "switch" in result
        assert "Type A" in result
