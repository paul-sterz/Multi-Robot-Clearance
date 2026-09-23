

#--------------------------------------------------------
# SPANNING TREE EVOLUTION
#
# Runs a single one of the 4 approaches (DP, DP BLabel Order, Greedy,
# Greedy BLabel Order) on the same environment/graph used by
# approachTest() twice: once with the normal evolutionary steady-state
# search (graphSearch()'s default searchMode="evolutionary") and once
# with a pure random-spanning-tree baseline (searchMode="random", see
# Searcher.py). Both runs share the same graph, robots and stopping
# budget (computation time / number of spanning trees), so they are
# directly comparable. For each run, tree by tree, the objective value of
# the spanning tree that was just checked and the best objective value
# found so far are tracked via graphSearch()'s historyCallback hook, so
# the Streamlit UI (test.py) can plot how each search converges.
#--------------------------------------------------------

import numpy as np

from approachTest import (
    APPROACH_NAMES,
    APPROACHES,
    buildEnvironment,
    detectionFnc,
    graphBuilder,
    computeObstacleDistance,
    START_REGION,
    GRAPH_PRIOR_L,
    GRAPH_PRIOR_SIGMA,
    GRAPH_ALPHA,
)


# Approach name -> its graphSearch function, so a single approach can be
# picked by name (the same fixed order as APPROACH_NAMES/APPROACHES).
APPROACH_FUNCS = dict(zip(APPROACH_NAMES, APPROACHES))


def _runSingleSearch(
    graphSearchFn,
    G,
    startNodes,
    obstacles,
    distanceMap,
    priors,
    D,
    availableRobots,
    availableTime,
    maxTrees,
    populationSize,
    searchMode,
    probBudget,
):

    history = []

    def historyCallback(checkedTrees, fitness, bestFitness):
        history.append((checkedTrees, fitness, bestFitness))

    strategy, tree, checkedTrees, bestFitness = graphSearchFn(
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
        populationSize=populationSize,
        historyCallback=historyCallback,
        searchMode=searchMode,
        probBudget=probBudget,
    )

    return {
        "strategy": strategy,
        "tree": tree,
        "checkedTrees": checkedTrees,
        "bestFitness": bestFitness,
        "history": history,
    }


def runSpanningTreeEvolution(
    approachName: str,
    detecRad: int,
    availableRobots: int,
    availableTime,
    maxTrees,
    populationSize: int,
    epsilon=0.05,
    probBudget=200,
):

    #-------------------------------------------------------------------
    # STEP 1: ALLOCATION (identical to approachTest())
    #-------------------------------------------------------------------
    obstacles, hotspots = buildEnvironment()

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

    graphSearchFn = APPROACH_FUNCS[approachName]

    #-------------------------------------------------------------------
    # STEP 2: RUN THE CHOSEN APPROACH TWICE ON THE SAME GRAPH - ONCE WITH
    # THE EVOLUTIONARY SEARCH, ONCE WITH PURE RANDOM SPANNING TREES
    #-------------------------------------------------------------------
    evolutionary = _runSingleSearch(
        graphSearchFn,
        G,
        startNodes,
        obstacles,
        distanceMap,
        priors,
        D,
        availableRobots,
        availableTime,
        maxTrees,
        populationSize,
        searchMode="evolutionary",
        probBudget=probBudget,
    )

    random_ = _runSingleSearch(
        graphSearchFn,
        G,
        startNodes,
        obstacles,
        distanceMap,
        priors,
        D,
        availableRobots,
        availableTime,
        maxTrees,
        populationSize,
        searchMode="random",
        probBudget=probBudget,
    )

    return {
        "G": G,
        "edgesShady": edges_shady,
        "priors": priors,
        "evolutionary": evolutionary,
        "random": random_,
    }


#--------------------------------------------------------
# SPANNING TREE EVOLUTION - REPEATED (100 RUNS) COMPARISON
#
# Like runSpanningTreeEvolution() above, but instead of tracking one run's
# tree-by-tree history, it repeats both searchModes ("evolutionary" and
# "random") numOfRuns times each - all on the exact same graph (built once,
# same convention as approachTest()/baselineTest()) so the two methods are
# directly comparable under identical conditions. For each run only the
# final outcome (checked trees + best objective found) is kept, not the
# full history, since only the aggregate best/mean/variance across runs is
# needed here.
#--------------------------------------------------------


def _runSearchModeRepeated(
    graphSearchFn,
    G,
    startNodes,
    obstacles,
    distanceMap,
    priors,
    D,
    availableRobots,
    availableTime,
    maxTrees,
    populationSize,
    searchMode,
    probBudget,
    numOfRuns,
):

    bestFitness = np.zeros(numOfRuns)
    checkedTrees = np.zeros(numOfRuns)

    for i in range(numOfRuns):

        _, _, treesChecked, fitness = graphSearchFn(
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
            populationSize=populationSize,
            searchMode=searchMode,
            probBudget=probBudget,
        )

        bestFitness[i] = fitness
        checkedTrees[i] = treesChecked

    return {
        "bestFitness": bestFitness,
        "checkedTrees": checkedTrees,
    }


def runSpanningTreeEvolutionRepeated(
    approachName: str,
    detecRad: int,
    availableRobots: int,
    availableTime,
    maxTrees,
    populationSize: int,
    numOfRuns: int = 100,
    epsilon=0.05,
    probBudget=200,
):

    #-------------------------------------------------------------------
    # STEP 1: ALLOCATION (identical to approachTest()/runSpanningTreeEvolution())
    #-------------------------------------------------------------------
    obstacles, hotspots = buildEnvironment()

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

    graphSearchFn = APPROACH_FUNCS[approachName]

    #-------------------------------------------------------------------
    # STEP 2: RUN BOTH SEARCH MODES numOfRuns TIMES EACH, ON THE SAME GRAPH
    #-------------------------------------------------------------------
    evolutionary = _runSearchModeRepeated(
        graphSearchFn,
        G,
        startNodes,
        obstacles,
        distanceMap,
        priors,
        D,
        availableRobots,
        availableTime,
        maxTrees,
        populationSize,
        searchMode="evolutionary",
        probBudget=probBudget,
        numOfRuns=numOfRuns,
    )

    random_ = _runSearchModeRepeated(
        graphSearchFn,
        G,
        startNodes,
        obstacles,
        distanceMap,
        priors,
        D,
        availableRobots,
        availableTime,
        maxTrees,
        populationSize,
        searchMode="random",
        probBudget=probBudget,
        numOfRuns=numOfRuns,
    )

    return {
        "G": G,
        "edgesShady": edges_shady,
        "priors": priors,
        "evolutionary": evolutionary,
        "random": random_,
    }
