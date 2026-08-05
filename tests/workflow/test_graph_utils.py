import pytest
import networkx as nx
from dsbo.workflow import (
    assert_no_self_loops,
    assert_no_disconnected_nodes,
    assert_no_duplicate_nodes,
    task_levels,
    is_output_task,
    get_dependent_task_groups,
)
from tests.workflow.conftest import SimpleModule


class TestAssertNoSelfLoops:
    def test_passes_for_acyclic_graph(self):
        G = nx.DiGraph()
        G.add_edge("a", "b")
        assert_no_self_loops(G)

    def test_raises_on_self_loop(self):
        G = nx.DiGraph()
        G.add_edge("a", "a")
        with pytest.raises(AssertionError, match="self-loop"):
            assert_no_self_loops(G)


class TestAssertNoDisconnectedNodes:
    def test_passes_when_all_nodes_connected(self):
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("b", "c")])
        assert_no_disconnected_nodes(G)

    def test_raises_on_disconnected_node(self):
        G = nx.DiGraph()
        a = SimpleModule()
        b = SimpleModule()
        c = SimpleModule()
        G.add_edge(a, b)
        G.add_node(c)
        with pytest.raises(AssertionError, match="disconnected"):
            assert_no_disconnected_nodes(G)

    def test_skips_check_for_single_node(self):
        G = nx.DiGraph()
        G.add_node("a")
        assert_no_disconnected_nodes(G)


class TestTaskLevels:
    def test_single_node(self):
        G = nx.DiGraph()
        G.add_node("a")
        assert task_levels(G) == {"a": 0}

    def test_chain(self):
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("b", "c")])
        levels = task_levels(G)
        assert levels["a"] == 0
        assert levels["b"] == 1
        assert levels["c"] == 2

    def test_diamond(self):
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
        levels = task_levels(G)
        assert levels["a"] == 0
        assert levels["b"] == 1
        assert levels["c"] == 1
        assert levels["d"] == 2


class TestGetDependentTaskGroups:
    def test_single_node(self):
        G = nx.DiGraph()
        G.add_node("a")
        groups = get_dependent_task_groups(G)
        assert groups == [["a"]]

    def test_chain(self):
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("b", "c")])
        groups = get_dependent_task_groups(G)
        assert len(groups) == 3
        assert "a" in groups[0]
        assert "b" in groups[1]
        assert "c" in groups[2]

    def test_diamond_groups_same_level(self):
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
        groups = get_dependent_task_groups(G)
        assert len(groups) == 3
        assert groups[0] == ["a"]
        assert groups[1] == ["b", "c"]
        assert groups[2] == ["d"]
