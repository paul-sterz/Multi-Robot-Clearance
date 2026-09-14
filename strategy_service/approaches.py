"""Dynamic loader for the 4 approach folders (DP / DP BLabel Order / Greedy /
Greedy BLabel Order).

Those folders are not Python packages -- two of the names contain spaces --
and each defines its own Graph.py / GraphBuilder.py / TrajectoryPlanning.py /
Searcher.py with plain, unqualified imports (e.g. "from Graph import Graph").
Graph.py and TrajectoryPlanning.py are byte-identical across all four
folders, only Searcher.py differs (allocation/scheduling logic). So the
shared modules are loaded once from the DP folder, and each approach's own
Searcher.py is loaded separately under a unique module name.

Same pattern as "Monte Carlo Simulation/approachTest.py", reused here so the
scene-backed service and the existing Monte Carlo harness stay independent
(neither imports the other).
"""

import importlib.util
import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

APPROACH_DIRS = {
    "dp": os.path.join(REPO_DIR, "DP"),
    "dp-blabel": os.path.join(REPO_DIR, "DP BLabel Order"),
    "greedy": os.path.join(REPO_DIR, "Greedy"),
    "greedy-blabel": os.path.join(REPO_DIR, "Greedy BLabel Order"),
}

APPROACH_LABELS = {
    "dp": "DP",
    "dp-blabel": "DP (B-label order)",
    "greedy": "Greedy",
    "greedy-blabel": "Greedy (B-label order)",
}

_SHARED_MODULES_DIR = APPROACH_DIRS["dp"]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_graph_search(key, approach_dir):
    searcher_name = "Searcher_" + key.replace("-", "_")
    searcher = _load_module(searcher_name, os.path.join(approach_dir, "Searcher.py"))
    return searcher.graphSearch


# Shared modules (identical for every approach) -- loaded once from DP/.
# TrajectoryPlanning has to load before any Searcher.py, since every
# Searcher.py does `from TrajectoryPlanning import aStar` unqualified.
_graph_module = _load_module("Graph", os.path.join(_SHARED_MODULES_DIR, "Graph.py"))
_trajectory_planning_module = _load_module(
    "TrajectoryPlanning", os.path.join(_SHARED_MODULES_DIR, "TrajectoryPlanning.py")
)

Graph = _graph_module.Graph
Node = _graph_module.Node
Edge = _graph_module.Edge

# Approach-specific graphSearch, loaded in the same fixed key order.
_GRAPH_SEARCH = {
    key: _load_graph_search(key, directory) for key, directory in APPROACH_DIRS.items()
}


def graph_search_for(key):
    if key not in _GRAPH_SEARCH:
        raise KeyError(f"unknown approach {key!r}, expected one of {list(_GRAPH_SEARCH)}")
    return _GRAPH_SEARCH[key]
