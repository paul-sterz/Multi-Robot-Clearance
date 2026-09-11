import os
import sys
import math
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
# LOADING THE 4 BASELINE METHODS
# ==================================================
# "Baseline", "Baseline Modified", "Baseline2" and "Baseline2 Modified" are
# not Python packages and define their own Graph.py / TrajectoryPlanning.py
# / Searcher.py using plain, unqualified imports (e.g. "from Graph import
# Graph"). Their Graph.py and TrajectoryPlanning.py are identical across all
# four folders, so the shared modules are loaded once (from the "Baseline"
# folder) and only Searcher.py is loaded separately for each method - same
# pattern as approachTest.py.
#
# IMPORTANT: none of these folders' own GraphBuilderV2.py is ever loaded
# here. It builds its own graph via a random free-space partitioning that is
# independent of (and structurally different from - different node/edge
# count) the one approachTest.py's graphBuilder() builds for the 4
# DP/Greedy-style approaches. Comparing the baseline methods against the 3rd
# approach on different graphs would be meaningless, so instead all methods
# run on the exact same graph (built once via approachTest.py's
# graphBuilder(), see baselineTest() below). This works because none of the
# baseline Searcher.py files read anything graph-specific beyond
# G.nodes[i].pos / G.adj[...] (checked directly in Searcher.py: none of them
# touch a node's .prior or an edge's .time/.robotType, and each internally
# rebuilds its own spanning-tree copy using its own bound Graph() class
# regardless of what G it was given) - they only need a Graph object shaped
# like their own, not literally built by GraphBuilderV2.
#
# Because approachTest.py has already registered its own "Graph" and
# "TrajectoryPlanning" modules under those exact names in sys.modules (so
# that DP/Searcher.py's "from Graph import Graph" resolves correctly), those
# names are temporarily overridden with the Baseline versions while loading
# these Searcher.py files, and restored again immediately afterwards - so
# nothing outside this loading block ever observes the swap.

BASELINE_SHARED_DIR = os.path.join(REPO_DIR, "Baseline")

BASELINE_METHOD_DIRS = {
    "Baseline": os.path.join(REPO_DIR, "Baseline"),
    "Baseline Modified": os.path.join(REPO_DIR, "Baseline Modified"),
    "Baseline2": os.path.join(REPO_DIR, "Baseline2"),
    "Baseline2 Modified": os.path.join(REPO_DIR, "Baseline2 Modified"),
}

BASELINE_METHOD_NAMES = list(BASELINE_METHOD_DIRS.keys())

# The "Modified" variants' graphSearch() additionally accepts cellpriors/D
# to break ties (same neededRobots) by cell-prior objective instead of
# keeping whichever tree was found first - see the "TEST" comments in their
# Searcher.py files. "Baseline" and "Baseline2" don't accept these kwargs.
METHODS_WITH_CELL_PRIOR_TIEBREAK = {"Baseline Modified", "Baseline2 Modified"}


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


# Shared modules (identical across all 4 baseline folders) - loaded once
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


# The 4 baseline search methods, in the same fixed order as
# BASELINE_METHOD_NAMES
BASELINE_SEARCHERS = [
    _loadBaselineSearcher(BASELINE_METHOD_DIRS[name], name.replace(" ", "_"))
    for name in BASELINE_METHOD_NAMES
]


# ==================================================
# CELL PRIOR OBJECTIVE FOR THE 4 BASELINE METHODS
#
# Unlike the 4 approaches loaded in approachTest.py (whose Searcher.py
# already returns the cell-prior objective directly as bestFitness, computed
# internally against the graph's priors/detection sets), none of the 4
# baseline Searcher.py files know anything about priors - they only ever
# track/minimize the number of robots needed - so the cell-prior objective
# for a winning strategy has to be computed here instead, by walking the
# strategy exactly like DP's internal computeExpTime()/approachTest.py's
# computeNodePriorObjective(): every node's first-visit time is determined
# from the strategy's moves, nodes are processed in ascending visit-time
# order, and each detected cell only ever contributes its prior once - to
# whichever node reaches it first.
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
    robotIncreasePercent: float,
    thirdApproachName: str,
    availableTime,
    maxTrees,
):

    #-------------------------------------------------------------------
    # STEP 1: ALLOCATION
    #-------------------------------------------------------------------
    obstacles, hotspots = buildEnvironment()

    # ONLY ONE GRAPH: built once (DP-style, with priors) and reused for every
    # run and every one of the 5 compared methods below - the 4 baseline
    # methods run on it exactly like the 4 approaches in approachTest.py do,
    # see the module-level comment above for why that is safe.
    G, edges_shady, D, startNodes, priors = graphBuilder(
        obstacles,
        hotspots,
        lambda p, obs: detectionFnc(p, obs, detecRad),
        START_REGION,
        GRAPH_PRIOR_L,
        GRAPH_PRIOR_SIGMA,
        GRAPH_ALPHA,
    )

    distanceMap = computeObstacleDistance(obstacles)

    thirdApproachIdx = APPROACH_NAMES.index(thirdApproachName)
    thirdApproachSearchFn = APPROACHES[thirdApproachIdx]

    methodNames = BASELINE_METHOD_NAMES + [thirdApproachName]
    numMethods = len(methodNames)

    resultsCellPrior = np.zeros((numOfRuns, numMethods))
    resultsRobots = np.zeros((numOfRuns, numMethods))
    resultsTrees = np.zeros((numOfRuns, numMethods))
    resultsTime = np.zeros((numOfRuns, numMethods))

    #------------------------------------------------------------------
    # STEP 2: EVALUATE THE METHODS AND STORE THE RESULTS
    #------------------------------------------------------------------

    for i in range(numOfRuns):

        runRobotCounts = {}

        # ---- All 4 baseline methods: always use their own minimum number
        # ---- of robots, computed internally.
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
                startNodes,
                obstacles,
                distanceMap,
                GRAPH_ALPHA,
                maxTrees=maxTrees,
                **tiebreakKwargs,
            )

            resultsTime[i, methodIdx] = time.time() - startTime
            resultsTrees[i, methodIdx] = checkedTrees
            resultsRobots[i, methodIdx] = robotsNeeded

            resultsCellPrior[i, methodIdx] = computeCellPriorObjective(
                strategy, startNodes, D, priors
            )

            runRobotCounts[methodName] = robotsNeeded

            print(methodName + " beendet")

        # ---- 3rd approach: gets the (rounded up) minimum robots needed by
        # ---- the baselines this run, scaled by the chosen percentage.
        baseRobots = max(runRobotCounts.values())
        thirdApproachRobots = max(
            1, math.ceil(baseRobots * (1 + robotIncreasePercent / 100))
        )

        startTime = time.time()

        strategy, tree, checkedTrees, cellPriorFitness = thirdApproachSearchFn(
            G,
            availableTime,
            thirdApproachRobots,
            startNodes,
            obstacles,
            distanceMap,
            GRAPH_ALPHA,
            priors,
            D,
            maxTrees=maxTrees,
        )

        resultsTime[i, -1] = time.time() - startTime
        resultsTrees[i, -1] = checkedTrees
        resultsRobots[i, -1] = thirdApproachRobots
        resultsCellPrior[i, -1] = cellPriorFitness

        print(thirdApproachName + " beendet")

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
