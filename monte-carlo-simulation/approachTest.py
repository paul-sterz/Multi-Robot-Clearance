import os
import sys
import importlib.util
import json
import time

import numpy as np


# ==================================================
# LOADING THE 4 APPROACHES
# ==================================================
# The approach folders (on disk: "dp", "dp-blabel-order", "greedy",
# "greedy-blabel-order" - the keys below are the display/approach names
# used everywhere else in this codebase, not the folder names) are not
# Python packages and each one defines its own Graph.py / GraphBuilder.py /
# TrajectoryPlanning.py / Searcher.py using plain, unqualified imports
# (e.g. "from Graph import Graph"). Graph.py, GraphBuilder.py and
# TrajectoryPlanning.py are identical across all approaches, only
# Searcher.py differs. So the shared modules are loaded once (from the dp
# folder) and only Searcher.py is loaded separately - and under a unique
# module name - for every approach.

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

APPROACH_DIRS = {
    "DP": os.path.join(REPO_DIR, "dp"),
    "DP BLabel Order": os.path.join(REPO_DIR, "dp-blabel-order"),
    "Greedy": os.path.join(REPO_DIR, "greedy"),
    "Greedy BLabel Order": os.path.join(REPO_DIR, "greedy-blabel-order"),
}

# Shared modules (Graph.py, GraphBuilder.py, TrajectoryPlanning.py) are
# loaded once from the dp folder; dp's own Searcher.py is still loaded
# separately below, alongside the other approaches.
_SHARED_MODULES_DIR = os.path.join(REPO_DIR, "dp")


def _loadModule(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _loadGraphSearch(approachDir):
    searcherName = "Searcher_" + os.path.basename(approachDir).replace(" ", "_")
    searcher = _loadModule(searcherName, os.path.join(approachDir, "Searcher.py"))
    return searcher.graphSearch


# Shared modules (identical for every approach) - loaded once from "DP"
_graphModule = _loadModule("Graph", os.path.join(_SHARED_MODULES_DIR, "Graph.py"))
_trajectoryPlanningModule = _loadModule("TrajectoryPlanning", os.path.join(_SHARED_MODULES_DIR, "TrajectoryPlanning.py"))
_graphBuilderModule = _loadModule("GraphBuilder", os.path.join(_SHARED_MODULES_DIR, "GraphBuilder.py"))

graphBuilder = _graphBuilderModule.graphBuilder
computeObstacleDistance = _trajectoryPlanningModule.computeObstacleDistance

Graph = _graphModule.Graph
Node = _graphModule.Node
Edge = _graphModule.Edge

# Fixed column order used throughout the Monte Carlo comparison (Step 2/3)
APPROACH_NAMES = list(APPROACH_DIRS.keys())

# Approach specific search methods, loaded in the same fixed order
APPROACHES = [_loadGraphSearch(APPROACH_DIRS[name]) for name in APPROACH_NAMES]


# ==================================================
# ENVIRONMENT
# ==================================================
# Shared between all 2D tests and the Streamlit UI (test.py), so the
# environment (without a graph) can be visualized independently of running
# a search.
#
# If an environment was saved via the "Save environment" button in one of
# the approaches' test.py UIs, that one is used for every 2D Monte Carlo
# test: its grid, priors, start region, its GraphBuilder parameters
# (detection radius, prior l / sigma, alpha, epsilon - these override the
# Monte Carlo UI's own values) and its node positions, so the graph always
# has exactly the same nodes instead of new random ones. Otherwise the fixed
# default grid below is used (with random node positions, as before). The
# file is re-read on every call (not cached at import), so a newly saved
# environment is picked up without restarting Streamlit.

SAVED_ENVIRONMENT_PATH = os.path.join(
    REPO_DIR, "monte-carlo-simulation", "saved_environment.json"
)


def _defaultEnvironment():

    obstacles = np.zeros((10, 10))
    obstacles[0, 4] = 1
    obstacles[1, 4] = 1
    obstacles[2, 4] = 1
    obstacles[3, 4] = 1

    obstacles[ 7, 0] = 1
    obstacles[ 7, 1] = 1
    obstacles[ 7, 2] = 1
    obstacles[ 7, 3] = 1
    obstacles[ 7, 4] = 1
    obstacles[ 9, 4] = 1

    obstacles[9, 7] = 1
    obstacles[7, 7] = 1
    obstacles[7, 8] = 1
    obstacles[7, 9] = 1

    hotspots = [((8,1),2), ((9,3),2), ((9,9),1), ((1,5),0), ((2,9),0)]

    startRegion = {(0, 0)}

    return obstacles, hotspots, startRegion


def hasSavedEnvironment():
    return os.path.exists(SAVED_ENVIRONMENT_PATH)


def loadEnvironment(detecRad, epsilon=0.05):
    """Returns the environment every 2D test runs on, as a dict. detecRad and
    epsilon are only used for the default environment - a saved one brings
    its own."""

    if not hasSavedEnvironment():

        obstacles, hotspots, startRegion = _defaultEnvironment()

        return {
            "saved": False,
            "obstacles": obstacles,
            "hotspots": hotspots,
            "startRegion": startRegion,
            "detecRad": detecRad,
            "priorL": GRAPH_PRIOR_L,
            "priorSigma": GRAPH_PRIOR_SIGMA,
            "alpha": GRAPH_ALPHA,
            "epsilon": epsilon,
            "nodePositions": None,
            "numStartNodes": None,
        }

    with open(SAVED_ENVIRONMENT_PATH) as f:
        data = json.load(f)

    return {
        "saved": True,
        "obstacles": np.array(data["obstacles"], dtype=float),
        "hotspots": [((r, c), category) for r, c, category in data["hotspots"]],
        "startRegion": {(r, c) for r, c in data["start_region"]} or {(0, 0)},
        "detecRad": data["detection_radius"],
        "priorL": data["prior_l"],
        "priorSigma": data["prior_sigma"],
        "alpha": data["alpha"],
        "epsilon": data["epsilon"],
        "nodePositions": [tuple(pos) for pos in data["node_positions"]],
        "numStartNodes": data["num_start_nodes"],
    }


def buildEnvironmentGraph(env):
    """graphBuilder() on a loadEnvironment() environment - with its fixed
    node positions if it is a saved one."""

    return graphBuilder(
        env["obstacles"],
        env["hotspots"],
        lambda p, obs: detectionFnc(p, obs, env["detecRad"]),
        env["startRegion"],
        env["priorL"],
        env["priorSigma"],
        env["alpha"],
        env["epsilon"],
        nodePositions=env["nodePositions"],
        numStartNodes=env["numStartNodes"],
    )


# ==================================================
# SIMPLE DETECTION FUNCTION
# ==================================================


def detectionFnc(p, obstacles, radius):

    H, W = obstacles.shape
    px, py = p

    # --------------------------------------------------
    # Simplified Bresenham line-of-sight test
    # --------------------------------------------------

    def has_clear_line_of_sight(x0, y0, x1, y1):

        xold = x0
        yold = y0

        dx = x1 - x0
        dy = y1 - y0

        steps = max(abs(dx), abs(dy))

        if steps == 0:
            return True

        for i in range(1, steps + 1):

            x = round(x0 + dx * i / steps)
            y = round(y0 + dy * i / steps)

            # If it is a diagonal/cross move,
            # also check the cells crossed on the way.
            if x != xold and y != yold:

                if abs(dy) == abs(dx):

                    if (
                        obstacles[x, yold] == 1
                        or obstacles[xold, y] == 1
                    ):
                        return False

                elif steps == abs(dx):

                    if obstacles[x, yold] == 1:
                        return False

                else:

                    if obstacles[xold, y] == 1:
                        return False

            # Check the new position
            if obstacles[x, y] == 1:
                return False

            xold = x
            yold = y

        return True

    visible = set()

    for x in range(H):

        for y in range(W):

            if obstacles[x, y] == 1:
                continue

            if (
                np.sqrt(
                    (x - px) ** 2
                    + (y - py) ** 2
                )
                <= radius
            ):

                if has_clear_line_of_sight(
                    px,
                    py,
                    x,
                    y,
                ):

                    visible.add((x, y))

    return visible



def computeLabels(T : Graph, root, parent):

    #Saving labels in a dictionary of the form: (x,y) | lambda((x,y))
    edgeLabels = {} 

    #Saving pi meaning all children of the root
    childLabels = [] 

    # Calculating lables recursive for children
    for y in T.adj[T.nodes[root]]:
        if y.idx != parent:

            subLabels = computeLabels(T, y.idx, root)

            edgeLabels.update(subLabels)

            childLabels.append(subLabels[(root, y.idx)])


    # Check if leaf
    if len(childLabels) == 0:
        edgeLabels[(parent, root)] = 1

    else:
        #formula from the paper
        childLabels.sort(reverse=True)

        p1 = childLabels[0]
        if len(childLabels) > 1:  
            p2 = childLabels[1]
        else: p2 = 0

        if p1 == 1:
            currentLabel = p1 + 1
        else:
            currentLabel = max(p1, p2 + 1)

        edgeLabels[(parent, root)] = currentLabel

    return edgeLabels


# ==================================================
# NODE PRIOR OBJECTIVE
#
# Unlike the cell-prior objective (computeExpTime() inside each Searcher.py,
# returned directly as bestFitness), there is no existing node-based
# equivalent - so it is computed here by walking the winning strategy:
# every node's first-visit time is determined from the strategy moves, and
# the objective sums node.prior * firstVisitTime over all nodes.
# ==================================================


def computeNodePriorObjective(strategy, G : Graph, startNodes):

    targetedNodes = {move[1] for move in strategy if len(move) == 5}

    nodeVisitedTime = {}

    # Start-region nodes that are never the target of a move already have
    # robots on them at t = 0 (see computeRobotCounts() in TrajectoryPlanning.py)
    for i in range(startNodes):
        if i not in targetedNodes:
            nodeVisitedTime[i] = 0

    for move in strategy:

        if len(move) != 5:
            continue

        _, target, _, _, tArrival = move

        if target not in nodeVisitedTime or tArrival < nodeVisitedTime[target]:
            nodeVisitedTime[target] = tArrival

    objective = 0.0

    for node in G.nodes:
        if node.idx in nodeVisitedTime:
            objective += node.prior * nodeVisitedTime[node.idx]

    return objective


# Fixed GraphBuilder prior/A* parameters, shared between approachTest() and
# the Streamlit UI (test.py) so both use the exact same prior computation.
GRAPH_PRIOR_L = 2
GRAPH_PRIOR_SIGMA = 2.860054369471078
GRAPH_ALPHA = 1


def approachTest(detecRad : int, numOfRuns : int, availableRobots : int, availableTime, maxTrees, epsilon=0.05, probBudget=200):

    #-------------------------------------------------------------------
    # STEP 1: ALLOCATION
    #-------------------------------------------------------------------
    env = loadEnvironment(detecRad, epsilon)
    obstacles = env["obstacles"]
    alpha = env["alpha"]

    G, edges_shady, D, startNodes, priors = buildEnvironmentGraph(env)

    distanceMap = computeObstacleDistance(obstacles)

    # NUR EIN GRAPH: the graph above is built once and reused for every run
    # and every approach below - only the (randomized) search itself differs
    # between runs.
    # 100 runs auf dem GRPH PLOTTE MIN, MAX AVG und ändere die gegebene Rechenzeit bzw. Trees ist intuitiver

    numApproaches = len(APPROACHES)

    resultsCellPrior = np.zeros((numOfRuns, numApproaches))
    resultsNodePrior = np.zeros((numOfRuns, numApproaches))
    resultsTrees = np.zeros((numOfRuns, numApproaches))
    resultsTime = np.zeros((numOfRuns, numApproaches))

    #------------------------------------------------------------------
    # STEP 2: EVALUATE THE METHODS ON THE GRAPH AND STORE THE RESULTS
    #------------------------------------------------------------------

    for i in range(numOfRuns):

        for approachIdx, graphSearchFn in enumerate(APPROACHES):

            # calculate the best strategy with the searcher method for the compared approaches.
            # the best fittnes value according to the cell prior is immidieatly returned and can be stored in the according numpy array
            # the node prior value has to be calculated with a sperate function that goes along the strategy and everythime a node is visited for the first time to the objective function is added prior of node * visited time

            startTime = time.time()

            strategy, tree, checkedTrees, cellPriorFitness = graphSearchFn(
                G,
                availableTime,
                availableRobots,
                startNodes,
                obstacles,
                distanceMap,
                alpha,
                priors,
                D,
                maxTrees=maxTrees,
                probBudget=probBudget,
            )

            resultsTime[i, approachIdx] = time.time() - startTime
            resultsTrees[i, approachIdx] = checkedTrees

            print("Approach " + str(approachIdx) + " beendet")
            resultsCellPrior[i, approachIdx] = cellPriorFitness

            if cellPriorFitness == np.inf:
                resultsNodePrior[i, approachIdx] = np.inf
            else:
                resultsNodePrior[i, approachIdx] = computeNodePriorObjective(strategy, G, startNodes)
        print("Run Nummer " + str(i) + " beendet!")

    return G, edges_shady, D, startNodes, priors, resultsCellPrior, resultsNodePrior, resultsTrees, resultsTime
