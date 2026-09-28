import os
import sys
import importlib.util
import time

import numpy as np

from approachTest import (
    REPO_DIR,
    buildEnvironment,
    detectionFnc,
    graphBuilder,
    computeObstacleDistance,
    APPROACH_NAMES,
    APPROACHES,
    START_REGION,
    GRAPH_PRIOR_L,
    GRAPH_PRIOR_SIGMA,
    GRAPH_ALPHA,
)


# ==================================================
# LOADING THE 2 BASELINE METHODS
# ==================================================
# "Baseline" and "Baseline Modified" are not Python packages and define
# their own Graph.py / TrajectoryPlanning.py / Searcher.py using plain,
# unqualified imports (e.g. "from Graph import Graph"). Their Graph.py and
# TrajectoryPlanning.py are identical across both folders, so the shared
# modules are loaded once (from the "Baseline" folder) and only Searcher.py
# is loaded separately for each method - same pattern as approachTest.py.
# ("Baseline2"/"Baseline2 Modified" used to be loaded here too, but are no
# longer part of this comparison.)
#
# IMPORTANT: none of these folders' own GraphBuilderV2.py is ever loaded
# here. It builds its own graph via a random free-space partitioning that is
# independent of (and structurally different from - different node/edge
# count) the one approachTest.py's graphBuilder() builds for the compared
# approaches (Greedy BLabel Order, DP BLabel Order). Comparing the baseline
# methods against them on different graphs would be meaningless, so instead
# all methods run on the exact same graph (built once via approachTest.py's
# graphBuilder(), see baselineTest() below). This works because neither
# baseline Searcher.py reads anything graph-specific beyond G.nodes[i].pos /
# G.adj[...] and (only "Baseline Modified", only for its 3D-scene node-prior
# tiebreak - see its Searcher.py) G.nodes[i].prior, and each internally
# rebuilds its own spanning-tree copy using its own bound Graph() class
# regardless of what G it was given - they only need a Graph object shaped
# like their own, not literally built by GraphBuilderV2.
#
# Because approachTest.py has already registered its own "Graph" and
# "TrajectoryPlanning" modules under those exact names in sys.modules (so
# that DP/Searcher.py's "from Graph import Graph" resolves correctly), those
# names are temporarily overridden with the Baseline versions while loading
# these Searcher.py files, and restored again immediately afterwards - so
# nothing outside this loading block ever observes the swap.

BASELINE_SHARED_DIR = os.path.join(REPO_DIR, "baseline")

BASELINE_METHOD_DIRS = {
    "Baseline": os.path.join(REPO_DIR, "baseline"),
    "Baseline Modified": os.path.join(REPO_DIR, "baseline-modified"),
}

BASELINE_METHOD_NAMES = list(BASELINE_METHOD_DIRS.keys())

# "Baseline Modified"'s graphSearch() additionally accepts cellpriors/D to
# break ties (same neededRobots) by a prior-based objective instead of
# keeping whichever tree was found first - see the "CELL-PRIOR vs.
# NODE-PRIOR" comments in its Searcher.py. "Baseline" doesn't accept these
# kwargs.
METHODS_WITH_CELL_PRIOR_TIEBREAK = {"Baseline Modified"}

# The 2 fixed DP/Greedy-style approaches "Baseline"/"Baseline Modified" are
# compared against - always both, no longer a single user-selectable "3rd
# approach". All 4 methods now run with the exact same fixed availableRobots
# budget (see baselineTest() below) - "Baseline"/"Baseline Modified" no
# longer self-determine their own minimum robot count (their Searcher.py's
# availableRobots is a hard cap, not a target to search for - see the
# "CELL-PRIOR vs. NODE-PRIOR" comments in their Searcher.py), so there is no
# baseline-derived count left to scale the compared approaches' budget from.
COMPARED_APPROACH_NAMES = ["Greedy BLabel Order", "DP BLabel Order"]


def _loadModule(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _loadWithTemporarySysModules(overrides, name, path):
    saved = {}

    for overrideName, overrideModule in overrides.items():
        saved[overrideName] = sys.modules.get(overrideName)
        sys.modules[overrideName] = overrideModule

    try:
        return _loadModule(name, path)
    finally:
        for overrideName, previousModule in saved.items():
            if previousModule is not None:
                sys.modules[overrideName] = previousModule
            else:
                del sys.modules[overrideName]


# Shared modules (identical across both baseline folders) - loaded once
# from "Baseline", only so that each Searcher.py's own "from Graph import
# Graph" / "from TrajectoryPlanning import aStar" resolve correctly.
_baselineGraphModule = _loadModule(
    "Graph_Baseline", os.path.join(BASELINE_SHARED_DIR, "Graph.py")
)

_baselineTrajectoryModule = _loadWithTemporarySysModules(
    {"Graph": _baselineGraphModule},
    "TrajectoryPlanning_Baseline",
    os.path.join(BASELINE_SHARED_DIR, "TrajectoryPlanning.py"),
)


def _loadBaselineSearcher(methodDir, uniqueSuffix):
    module = _loadWithTemporarySysModules(
        {"Graph": _baselineGraphModule, "TrajectoryPlanning": _baselineTrajectoryModule},
        "Searcher_" + uniqueSuffix,
        os.path.join(methodDir, "Searcher.py"),
    )
    return module.graphSearch


# The 2 baseline search methods, in the same fixed order as
# BASELINE_METHOD_NAMES
BASELINE_SEARCHERS = [
    _loadBaselineSearcher(BASELINE_METHOD_DIRS[name], name.replace(" ", "_"))
    for name in BASELINE_METHOD_NAMES
]


# ==================================================
# CELL PRIOR OBJECTIVE FOR THE 2 BASELINE METHODS
#
# Unlike the compared approaches loaded from approachTest.py (whose
# Searcher.py already returns its own objective directly as bestFitness,
# computed internally against the graph's priors/detection sets), "Baseline"
# knows nothing about priors at all, and "Baseline Modified" only uses them
# internally to pick its own best tree (see its Searcher.py) - neither
# returns a prior-based objective value directly, so the cell-prior
# objective for a winning strategy has to be computed here instead, by walking the
# strategy exactly like DP's internal computeExpTime()/approachTest.py's
# computeNodePriorObjective(): every node's first-visit time is determined
# from the strategy's moves, nodes are processed in ascending visit-time
# order, and each detected cell only ever contributes its prior once - to
# whichever node reaches it first.
#
# NOTE: this is the expensive cell-sum version, only safe to call on the 2D
# synthetic grid (small cellpriors/D). The 3D-scene equivalent
# (3DTest.py's baselineTest3D()) uses approachTest.py's
# computeNodePriorObjective() instead, to avoid ever summing over a real
# scene's (potentially huge) cell count - see its own comment.
# ==================================================


def computeCellPriorObjective(strategy, startNodes, D, cellpriors):

    if strategy is None:
        return np.inf

    targetedNodes = {move[1] for move in strategy if len(move) == 5}

    nodeVisitedTime = {}

    # Start-region nodes that are never the target of a move already have
    # robots on them at t = 0.
    for i in range(startNodes):
        if i not in targetedNodes:
            nodeVisitedTime[i] = 0

    for move in strategy:

        if len(move) != 5:
            continue

        _, target, _, _, tArrival = move

        if target not in nodeVisitedTime or tArrival < nodeVisitedTime[target]:
            nodeVisitedTime[target] = tArrival

    foundPriors = np.copy(cellpriors)

    objective = 0.0

    for nodeIdx in sorted(nodeVisitedTime, key=lambda idx: nodeVisitedTime[idx]):

        for cell in D[nodeIdx]:

            objective += foundPriors[cell[0], cell[1]] * nodeVisitedTime[nodeIdx]
            foundPriors[cell[0], cell[1]] = 0.0

    return objective


def baselineTest(
    detecRad: int,
    numOfRuns: int,
    availableRobots: int,
    availableTime,
    maxTrees,
    epsilon=0.05,
    probBudget=200,
):

    #-------------------------------------------------------------------
    # STEP 1: ALLOCATION
    #-------------------------------------------------------------------
    obstacles, hotspots = buildEnvironment()

    # ONLY ONE GRAPH: built once (DP-style, with priors) and reused for every
    # run and every one of the 4 compared methods below (Baseline, Baseline
    # Modified, Greedy BLabel Order, DP BLabel Order) - the 2 baseline
    # methods run on it exactly like the compared approaches in
    # approachTest.py do, see the module-level comment above for why that is
    # safe.
    G, edges_shady, D, startNodes, priors = graphBuilder(
        obstacles,
        hotspots,
        lambda p, obs: detectionFnc(p, obs, detecRad),
        START_REGION,
        GRAPH_PRIOR_L,
        GRAPH_PRIOR_SIGMA,
        GRAPH_ALPHA,
        epsilon,
    )

    distanceMap = computeObstacleDistance(obstacles)

    comparedApproachSearchFns = [
        APPROACHES[APPROACH_NAMES.index(name)] for name in COMPARED_APPROACH_NAMES
    ]

    methodNames = BASELINE_METHOD_NAMES + COMPARED_APPROACH_NAMES
    numBaselineMethods = len(BASELINE_METHOD_NAMES)
    numMethods = len(methodNames)

    resultsCellPrior = np.zeros((numOfRuns, numMethods))
    resultsRobots = np.full((numOfRuns, numMethods), availableRobots, dtype=float)
    resultsTrees = np.zeros((numOfRuns, numMethods))
    resultsTime = np.zeros((numOfRuns, numMethods))

    #------------------------------------------------------------------
    # STEP 2: EVALUATE THE METHODS AND STORE THE RESULTS
    #
    # All 4 methods below get the exact same fixed availableRobots budget -
    # resultsRobots is filled with that constant up front (above) rather
    # than per-method here, since it no longer varies by method or run.
    #------------------------------------------------------------------

    for i in range(numOfRuns):

        # ---- Both baseline methods: availableRobots is a hard cap for
        # ---- them (see their Searcher.py) - a run without a feasible tree
        # ---- within maxTrees/availableTime comes back with strategy=None.
        for methodIdx, methodName in enumerate(BASELINE_METHOD_NAMES):

            searchFn = BASELINE_SEARCHERS[methodIdx]

            tiebreakKwargs = (
                {"cellpriors": priors, "D": D}
                if methodName in METHODS_WITH_CELL_PRIOR_TIEBREAK
                else {}
            )

            startTime = time.time()

            strategy, tree, checkedTrees, robotsNeeded = searchFn(
                G,
                availableTime,
                availableRobots,
                startNodes,
                obstacles,
                distanceMap,
                GRAPH_ALPHA,
                maxTrees=maxTrees,
                **tiebreakKwargs,
            )

            resultsTime[i, methodIdx] = time.time() - startTime
            resultsTrees[i, methodIdx] = checkedTrees

            resultsCellPrior[i, methodIdx] = computeCellPriorObjective(
                strategy, startNodes, D, priors
            )

            print(methodName + " beendet")

        # ---- COMPARED APPROACHES (Greedy BLabel Order, DP BLabel Order):
        # ---- same fixed availableRobots budget as the baseline methods.
        for approachOffset, (approachName, searchFn) in enumerate(
            zip(COMPARED_APPROACH_NAMES, comparedApproachSearchFns)
        ):

            columnIdx = numBaselineMethods + approachOffset

            startTime = time.time()

            strategy, tree, checkedTrees, cellPriorFitness = searchFn(
                G,
                availableTime,
                availableRobots,
                startNodes,
                obstacles,
                distanceMap,
                GRAPH_ALPHA,
                priors,
                D,
                maxTrees=maxTrees,
                probBudget=probBudget,
            )

            resultsTime[i, columnIdx] = time.time() - startTime
            resultsTrees[i, columnIdx] = checkedTrees
            resultsCellPrior[i, columnIdx] = cellPriorFitness

            print(approachName + " beendet")

        print("Run Nummer " + str(i) + " beendet!")

    return {
        "G": G,
        "edges_shady": edges_shady,
        "priors": priors,
        "methodNames": methodNames,
        "numGraphNodes": len(G.nodes),
        "numGraphEdges": len(G.edges),
        "resultsCellPrior": resultsCellPrior,
        "resultsRobots": resultsRobots,
        "resultsTrees": resultsTrees,
        "resultsTime": resultsTime,
    }
