import os
import sys
import importlib.util
import time

import numpy as np


# ==================================================
# LOADING THE 4 APPROACHES
# ==================================================
# The approach folders ("DP", "DP badge slpitting", "Worst Case Labels
# Greedy", "Constrained Tree Optimal") are not Python packages (their names
# contain spaces) and each one defines its own Graph.py / GraphBuilder.py /
# TrajectoryPlanning.py / Searcher.py using plain, unqualified imports
# (e.g. "from Graph import Graph"). Graph.py, GraphBuilder.py and
# TrajectoryPlanning.py are identical across all 4 approaches, only
# Searcher.py differs. So the shared modules are loaded once (from the DP
# folder) and only Searcher.py is loaded separately - and under a unique
# module name - for every approach.

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

APPROACH_DIRS = {
    "DP": os.path.join(REPO_DIR, "DP"),
    "DP Badge Splitting": os.path.join(REPO_DIR, "DP badge slpitting"),
    "Worst Case Labels Greedy": os.path.join(REPO_DIR, "Worst Case Labels Greedy"),
    "Constrained Tree Optimal": os.path.join(REPO_DIR, "Constrained Tree Optimal"),
}


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
_graphModule = _loadModule("Graph", os.path.join(APPROACH_DIRS["DP"], "Graph.py"))
_trajectoryPlanningModule = _loadModule("TrajectoryPlanning", os.path.join(APPROACH_DIRS["DP"], "TrajectoryPlanning.py"))
_graphBuilderModule = _loadModule("GraphBuilder", os.path.join(APPROACH_DIRS["DP"], "GraphBuilder.py"))

graphBuilder = _graphBuilderModule.graphBuilder
computeObstacleDistance = _trajectoryPlanningModule.computeObstacleDistance

Graph = _graphModule.Graph
Node = _graphModule.Node
Edge = _graphModule.Edge

# Approach specific search methods
graphSearchDP = _loadGraphSearch(APPROACH_DIRS["DP"])
graphSearchDPBadgeSplitting = _loadGraphSearch(APPROACH_DIRS["DP Badge Splitting"])
graphSearchWorstCaseLabels = _loadGraphSearch(APPROACH_DIRS["Worst Case Labels Greedy"])
graphSearchConstrainedTreeOptimal = _loadGraphSearch(APPROACH_DIRS["Constrained Tree Optimal"])

# Fixed column order used throughout the Monte Carlo comparison (Step 2/3)
APPROACHES = [
    graphSearchDP,
    graphSearchDPBadgeSplitting,
    graphSearchWorstCaseLabels,
    graphSearchConstrainedTreeOptimal,
]

APPROACH_NAMES = list(APPROACH_DIRS.keys())


# ==================================================
# ENVIRONMENT
# ==================================================
# Shared between approachTest() and the Streamlit UI (test.py), so the
# environment (without a graph) can be visualized independently of running
# a search.


def buildEnvironment():

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

    return obstacles, hotspots


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
START_REGION = {(0, 0)}
GRAPH_PRIOR_L = 2
GRAPH_PRIOR_SIGMA = 2.860054369471078
GRAPH_ALPHA = 1


def approachTest(detecRad : int, numOfRuns : int, availableRobots : int, availableTime, maxTrees):

    #-------------------------------------------------------------------
    # STEP 1: ALLOCATION
    #-------------------------------------------------------------------
    obstacles, hotspots = buildEnvironment()

    # Graphbuilder method with obstacles, hotspots, detectionFnc, startRegion = (0,0), l = 2, sigma=2.860054369471078 , alpha = 1

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

    # NUR EIN GRAPH: the graph above is built once and reused for every run
    # and every approach below - only the (randomized) search itself differs
    # between runs.
    # 100 runs auf dem GRPH PLOTTE MIN, MAX AVG und ändere die gegebene Rechenzeit bzw. Trees ist intuitiver

    resultsCellPrior = np.zeros((numOfRuns, 4))
    resultsNodePrior = np.zeros((numOfRuns, 4))
    resultsTrees = np.zeros((numOfRuns, 4))
    resultsTime = np.zeros((numOfRuns, 4))

    #------------------------------------------------------------------
    # STEP 2: EVALUATE THE 4 METHODS ON THE GRAPH AND STORE THE RESULTS
    #------------------------------------------------------------------

    for i in range(numOfRuns):

        for approachIdx, graphSearchFn in enumerate(APPROACHES):

            # calculate the best strategy with the searcher method for the 4 approaches.
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
                GRAPH_ALPHA,
                priors,
                D,
                maxTrees=maxTrees,
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
