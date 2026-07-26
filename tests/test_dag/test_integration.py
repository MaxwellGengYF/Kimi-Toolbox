"""Integration tests for Executor with various DAG topologies."""
from __future__ import annotations

import time
from typing import Any

import pytest

from kimix.dag import DAG, Context, ExecutionError, Executor, TaskNode

from tests.test_dag.conftest import make_adder, make_slow_task, async_add_one


def make_executor(max_workers: int | None = None) -> Executor:
    return Executor(max_workers=max_workers)


# ============================================================================
# Basic execution
# ============================================================================
class TestBasicExecution:
    def test_empty_dag(self) -> None:
        dag = DAG()
        exe = make_executor()
        results = exe.execute(dag)
        assert results == {}

    def test_single_node(self) -> None:
        dag = DAG()
        dag.add_node(TaskNode("a", make_adder("x", 5)))
        exe = make_executor()
        results = exe.execute(dag)
        assert results == {"a": 5}

    def test_linear_chain(self, linear_dag: DAG) -> None:
        exe = make_executor()
        results = exe.execute(linear_dag)
        assert results == {"a": 1, "b": 11, "c": 111}

    def test_diamond(self, diamond_dag: DAG) -> None:
        exe = make_executor()
        results = exe.execute(diamond_dag)
        # a=1; b and c run in parallel reading x=1, so whichever finishes last wins the ctx race.
        # d reads whatever b/c left and adds 1000.
        assert results["a"] == 1
        assert results["b"] in (11, 111)  # 1+10 or 11+100 depending on race
        assert results["c"] in (101, 111)  # 1+100 or 11+100 depending on race
        assert results["d"] in (1011, 1111)  # 11+1000 or 111+1000

    def test_fan_out(self, fan_out_dag: DAG) -> None:
        exe = make_executor()
        results = exe.execute(fan_out_dag)
        # a=1; b,c,d all read x=1 concurrently, last write wins ctx race.
        assert results["a"] == 1
        assert results["b"] in (11, 111, 1111)
        assert results["c"] in (101, 111, 1111)
        assert results["d"] in (1001, 1011, 1111)

    def test_context_shared(self) -> None:
        dag = DAG()
        dag.add_node(TaskNode("a", make_adder("counter", 1)))
        dag.add_node(TaskNode("b", make_adder("counter", 10), dependencies={"a"}))
        dag.add_node(TaskNode("c", make_adder("counter", 100), dependencies={"a"}))
        ctx = Context()
        exe = make_executor()
        exe.execute(dag, ctx)
        # b and c both see a's increment; whichever runs second wins the race
        # but counter will be at least 101 (1 + 100) or 11 (1 + 10)
        final = ctx.get("counter")
        assert final in (11, 101, 111)

    def test_isolated_nodes_run_in_parallel(self) -> None:
        """Two slow isolated nodes should finish faster than sequential."""
        dag = DAG()
        dag.add_node(TaskNode("a", make_slow_task("a", 0.15, 1)))
        dag.add_node(TaskNode("b", make_slow_task("b", 0.15, 2)))
        exe = make_executor(max_workers=2)
        start = time.perf_counter()
        results = exe.execute(dag)
        elapsed = time.perf_counter() - start
        assert results == {"a": 1, "b": 2}
        assert elapsed < 0.30  # parallel, not sequential

    def test_async_task(self) -> None:
        dag = DAG()
        dag.add_node(TaskNode("a", async_add_one))
        exe = make_executor()
        results = exe.execute(dag)
        assert results["a"] == 1

    def test_mixed_sync_async(self) -> None:
        dag = DAG()
        dag.add_node(TaskNode("sync", make_adder("x", 1)))
        dag.add_node(TaskNode("async", async_add_one, dependencies={"sync"}))
        exe = make_executor()
        results = exe.execute(dag)
        assert results["sync"] == 1
        assert results["async"] == 1  # counter starts at 0, +1


# ============================================================================
# Retry during execution
# ============================================================================
class TestRetryIntegration:
    def test_retry_eventually_succeeds(self) -> None:
        from tests.test_dag.conftest import make_fail_once
        dag = DAG()
        dag.add_node(TaskNode("a", make_fail_once("k"), retries=2))
        exe = make_executor()
        results = exe.execute(dag)
        assert results["a"] == 2  # second call succeeds


# ============================================================================
# Deep chains
# ============================================================================
class TestDeepChains:
    def test_chain_of_100(self) -> None:
        dag = DAG()
        prev: str | None = None
        for i in range(100):
            name = f"n{i}"
            deps = {prev} if prev else set()
            dag.add_node(TaskNode(name, make_adder("x", 1), dependencies=deps))
            prev = name
        exe = make_executor()
        results = exe.execute(dag)
        # Each task increments x by 1; the 100th task sees x=99 and returns 100.
        assert results["n99"] == 100
        for i in range(100):
            assert dag.get_node(f"n{i}").done

    def test_binary_tree(self) -> None:
        """Build a small binary tree: each node depends on two children."""
        dag = DAG()
        # leaves
        dag.add_node(TaskNode("leaf0", lambda c: 0))
        dag.add_node(TaskNode("leaf1", lambda c: 1))
        dag.add_node(TaskNode("leaf2", lambda c: 2))
        dag.add_node(TaskNode("leaf3", lambda c: 3))
        # internal
        dag.add_node(TaskNode("i0", lambda c: 10, dependencies={"leaf0", "leaf1"}))
        dag.add_node(TaskNode("i1", lambda c: 11, dependencies={"leaf2", "leaf3"}))
        dag.add_node(TaskNode("root", lambda c: 100, dependencies={"i0", "i1"}))
        exe = make_executor()
        results = exe.execute(dag)
        assert results["root"] == 100


# ============================================================================
# Multiple executors / reuse
# ============================================================================
class TestExecutorReuse:
    def test_execute_twice_fresh(self) -> None:
        """Executor.execute should be callable multiple times."""
        exe = make_executor()
        dag1 = DAG()
        dag1.add_node(TaskNode("a", lambda c: 1))
        dag2 = DAG()
        dag2.add_node(TaskNode("b", lambda c: 2))
        assert exe.execute(dag1) == {"a": 1}
        assert exe.execute(dag2) == {"b": 2}
