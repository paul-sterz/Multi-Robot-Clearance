

#--------------------------------------------------------
# SPANNING TREE EVOLUTION
#
# Runs a single one of the 4 approaches (DP, DP BLabel Order, Greedy,
# Greedy BLabel Order) on the same environment/graph used by
# approachTest(), but instead of averaging over many independent runs it
# tracks, tree by tree, the objective value of the spanning tree that was
# just checked and the best objective value found so far. This is exactly
# what graphSearch() already computes internally (fitness / bestFitness)
# - it is only exposed here via the historyCallback hook so the Streamlit
# UI (test.py) can plot how the search converges.
#--------------------------------------------------------

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


def runSpanningTreeEvolution(
    approachName: str,
    detecRad: int,
    availableRobots: int,
    availableTime,
    maxTrees,
    populationSize: int,
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
    )

    distanceMap = computeObstacleDistance(obstacles)

    graphSearchFn = APPROACH_FUNCS[approachName]

    #-------------------------------------------------------------------
    # STEP 2: RUN THE CHOSEN APPROACH ONCE, RECORDING PER-TREE HISTORY
    #-------------------------------------------------------------------
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
    )

    return {
        "G": G,
        "edgesShady": edges_shady,
        "priors": priors,
        "strategy": strategy,
        "tree": tree,
        "checkedTrees": checkedTrees,
        "bestFitness": bestFitness,
        "history": history,
    }
