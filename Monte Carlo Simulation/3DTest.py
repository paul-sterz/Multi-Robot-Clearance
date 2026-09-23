"""3D-scene equivalents of approachTest()/baselineTest()/runSpanningTreeEvolution()
in this same folder -- the same three Monte Carlo comparisons, run against one
of the six real clearing-scenes scenes (christ-church, keble-college, ...)
instead of the synthetic 10x10 obstacle grid `buildEnvironment()` builds.

Nothing about the 2D grid path changes: the approach/baseline graphSearch
functions this module calls are the very same ones approachTest.py and
baselineTest.py already loaded (imported from them below, not reloaded), and
this module only supplies a different graph/priors/travel-time table for them
to run on -- built from a real Scene via strategy_service.adapter, the same
adapter the 3D viewer's backend (strategy_service/server.py) uses. Comparing
mixed Graph/Node/Edge module instances this way is safe: none of the loaded
Searcher.py files ever check the class identity of the G they're given, only
read plain attributes off it (G.nodes[i].pos/.prior, G.adj[...], G.edges[...]),
and any spanning tree they build internally uses their own bound Graph/Node/Edge
classes regardless of what built G - baselineTest.py's own module docstring
about mixing Graph instances explains this in more detail.
"""

import json
import math
import os
import sys
import time

import numpy as np

from approachTest import (
    REPO_DIR,
    APPROACH_NAMES,
    APPROACHES,
    computeNodePriorObjective,
)
from baselineTest import (
    BASELINE_METHOD_NAMES,
    BASELINE_SEARCHERS,
    METHODS_WITH_CELL_PRIOR_TIEBREAK,
    computeCellPriorObjective,
)

CLEARING_SCENES_DIR = os.path.join(REPO_DIR, "clearing-scenes")
SCENES_DIR = os.path.join(CLEARING_SCENES_DIR, "scenes")

for _p in (CLEARING_SCENES_DIR, REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import clearing  # noqa: E402
from strategy_service import adapter as scene_adapter  # noqa: E402

SCENE_NAMES = clearing.available(scenes_dir=SCENES_DIR)

# Fallback hotspot weight/spread, the 3D-scene equivalent of approachTest.py's
# GRAPH_PRIOR_L/GRAPH_PRIOR_SIGMA constants - not exposed as sliders there
# either, so not added here (the request was to carry over the *existing*
# sliders, not add new ones; only the environment itself becomes selectable).
# Used only for a scene with no saved hotspot config (see loadHotspotConfig
# below) - once one is saved from the viewer, its own prior_l/prior_radius_m
# take over.
PRIOR_L_3D = 3.0
PRIOR_RADIUS_M_3D = 8.0
PRIOR_EPSILON_3D = 0.05

# strategy_service/server.py's POST/GET /api/hotspots/<scene> read and write
# exactly this directory - saving a hotspot layout from the viewer is what
# fills it in.
SCENE_PRIORS_DIR = os.path.join(REPO_DIR, "strategy_service", "scene_priors")

_scene_cache: dict[str, "clearing.Scene"] = {}


def loadScene3D(sceneName: str) -> "clearing.Scene":
    if sceneName not in _scene_cache:
        _scene_cache[sceneName] = clearing.load(sceneName, scenes_dir=SCENES_DIR)
    return _scene_cache[sceneName]


def default3DHotspots(scene) -> list[tuple[int, int]]:
    """A small, fixed, deterministic hotspot layout for a scene - the
    fallback used only when nothing has been saved for it yet from the
    viewer (see loadHotspotConfig below). Picked at fixed fractions of the
    vertex list (vertices already cover the walkable area by construction,
    so their host cells are always valid, on-surface hotspot locations)
    rather than fixed cell ids, since cell counts vary a lot between scenes.
    """
    n = scene.n_vertices
    picks = sorted({0, n // 3, (2 * n) // 3, n - 1})
    categories = [2, 1, 1, 0]
    return [(int(scene.vertex_cell[v]), categories[i % len(categories)])
            for i, v in enumerate(picks)]


def loadHotspotConfig(scene, sceneName: str):
    """Whatever was last saved for this scene from the viewer's "Save
    hotspots for this scene" button (strategy_service/server.py's
    POST /api/hotspots/<scene>, written to SCENE_PRIORS_DIR), converted from
    the clicked 3D positions it was saved as into real cell ids via the same
    nearest-cell lookup the live viewer uses - or None if nothing has been
    saved for this scene yet, so the caller can fall back to
    default3DHotspots().

    Returns (hotspots, startVertices, priorL, priorRadiusM) or None.
    """
    path = os.path.join(SCENE_PRIORS_DIR, f"{sceneName}.json")
    if not os.path.isfile(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    savedHotspots = config.get("hotspots") or []
    if not savedHotspots:
        return None

    hotspots = [
        (scene_adapter.nearest_cell(scene, (h["x"], h["y"], h["z"])), h["category"])
        for h in savedHotspots
    ]
    startVertices = config.get("start_vertices") or [0]
    priorL = config.get("prior_l") or PRIOR_L_3D
    priorRadiusM = config.get("prior_radius_m") or PRIOR_RADIUS_M_3D

    return hotspots, startVertices, priorL, priorRadiusM


def build3DEnvironment(sceneName: str, startVertices=None, epsilon=None):
    """The 3D equivalent of calling graphBuilder() in the 2D tests: builds
    the graph/detection sets/priors once, to be reused for every run and
    every method compared - exactly like "NUR EIN GRAPH" in approachTest.py
    and baselineTest.py.

    epsilon: GraphBuilder uncertainty floor passed to compute_priors(),
        defaulting to PRIOR_EPSILON_3D when not given (None).

    Returns (scene, G, D, startNodes, cellpriors, travelTime).
    """
    scene = loadScene3D(sceneName)

    saved = loadHotspotConfig(scene, sceneName)
    if saved is not None:
        hotspots, savedStartVertices, priorL, priorRadiusM = saved
        if startVertices is None:
            startVertices = savedStartVertices
    else:
        hotspots = default3DHotspots(scene)
        priorL, priorRadiusM = PRIOR_L_3D, PRIOR_RADIUS_M_3D
        if startVertices is None:
            startVertices = [0]

    if epsilon is None:
        epsilon = PRIOR_EPSILON_3D

    sigma = priorRadiusM / math.sqrt(-2 * math.log(0.5))
    priors = scene_adapter.compute_priors(
        scene, hotspots, priorL, sigma, epsilon
    )
    G, D, startNodes, idxToVertex = scene_adapter.build_graph(scene, startVertices, priors)
    travelTime = scene_adapter.travel_time_matrix(scene, idxToVertex)
    cellpriors = priors.reshape(-1, 1)

    return scene, G, D, startNodes, cellpriors, travelTime


# ==================================================
# APPROACH TEST (3D)
# ==================================================


def approachTest3D(sceneName: str, numOfRuns: int, availableRobots: int, availableTime, maxTrees, epsilon=None, probBudget=200):

    scene, G, D, startNodes, cellpriors, travelTime = build3DEnvironment(sceneName, epsilon=epsilon)

    numApproaches = len(APPROACHES)

    resultsCellPrior = np.zeros((numOfRuns, numApproaches))
    resultsNodePrior = np.zeros((numOfRuns, numApproaches))
    resultsTrees = np.zeros((numOfRuns, numApproaches))
    resultsTime = np.zeros((numOfRuns, numApproaches))

    for i in range(numOfRuns):

        for approachIdx, graphSearchFn in enumerate(APPROACHES):

            startTime = time.time()

            strategy, tree, checkedTrees, cellPriorFitness = graphSearchFn(
                G,
                availableTime,
                availableRobots,
                startNodes,
                None,
                None,
                None,
                cellpriors,
                D,
                maxTrees=maxTrees,
                travelTime=travelTime,
                probBudget=probBudget,
            )

            resultsTime[i, approachIdx] = time.time() - startTime
            resultsTrees[i, approachIdx] = checkedTrees
            resultsCellPrior[i, approachIdx] = cellPriorFitness

            if cellPriorFitness == np.inf:
                resultsNodePrior[i, approachIdx] = np.inf
            else:
                resultsNodePrior[i, approachIdx] = computeNodePriorObjective(strategy, G, startNodes)

            print(APPROACH_NAMES[approachIdx] + " beendet")

        print("Run Nummer " + str(i) + " beendet!")

    return {
        "scene": scene,
        "G": G,
        "D": D,
        "priors": cellpriors,
        "numGraphNodes": len(G.nodes),
        "numGraphEdges": len(G.edges),
        "resultsCellPrior": resultsCellPrior,
        "resultsNodePrior": resultsNodePrior,
        "resultsTrees": resultsTrees,
        "resultsTime": resultsTime,
    }


# ==================================================
# BASELINE TEST (3D)
# ==================================================


def baselineTest3D(
    sceneName: str,
    numOfRuns: int,
    robotIncreasePercent: float,
    thirdApproachName: str,
    availableTime,
    maxTrees,
    epsilon=None,
    probBudget=200,
):

    scene, G, D, startNodes, cellpriors, travelTime = build3DEnvironment(sceneName, epsilon=epsilon)

    thirdApproachIdx = APPROACH_NAMES.index(thirdApproachName)
    thirdApproachSearchFn = APPROACHES[thirdApproachIdx]

    methodNames = BASELINE_METHOD_NAMES + [thirdApproachName]
    numMethods = len(methodNames)

    resultsCellPrior = np.zeros((numOfRuns, numMethods))
    resultsRobots = np.zeros((numOfRuns, numMethods))
    resultsTrees = np.zeros((numOfRuns, numMethods))
    resultsTime = np.zeros((numOfRuns, numMethods))

    for i in range(numOfRuns):

        runRobotCounts = {}

        for methodIdx, methodName in enumerate(BASELINE_METHOD_NAMES):

            searchFn = BASELINE_SEARCHERS[methodIdx]

            tiebreakKwargs = (
                {"cellpriors": cellpriors, "D": D}
                if methodName in METHODS_WITH_CELL_PRIOR_TIEBREAK
                else {}
            )

            startTime = time.time()

            strategy, tree, checkedTrees, robotsNeeded = searchFn(
                G,
                availableTime,
                startNodes,
                None,
                None,
                None,
                maxTrees=maxTrees,
                travelTime=travelTime,
                **tiebreakKwargs,
            )

            resultsTime[i, methodIdx] = time.time() - startTime
            resultsTrees[i, methodIdx] = checkedTrees
            resultsRobots[i, methodIdx] = robotsNeeded

            resultsCellPrior[i, methodIdx] = computeCellPriorObjective(
                strategy, startNodes, D, cellpriors
            )

            runRobotCounts[methodName] = robotsNeeded

            print(methodName + " beendet")

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
            None,
            None,
            None,
            cellpriors,
            D,
            maxTrees=maxTrees,
            travelTime=travelTime,
            probBudget=probBudget,
        )

        resultsTime[i, -1] = time.time() - startTime
        resultsTrees[i, -1] = checkedTrees
        resultsRobots[i, -1] = thirdApproachRobots
        resultsCellPrior[i, -1] = cellPriorFitness

        print(thirdApproachName + " beendet")

        print("Run Nummer " + str(i) + " beendet!")

    return {
        "scene": scene,
        "G": G,
        "D": D,
        "priors": cellpriors,
        "methodNames": methodNames,
        "numGraphNodes": len(G.nodes),
        "numGraphEdges": len(G.edges),
        "resultsCellPrior": resultsCellPrior,
        "resultsRobots": resultsRobots,
        "resultsTrees": resultsTrees,
        "resultsTime": resultsTime,
    }


# ==================================================
# SPANNING TREE EVOLUTION (3D)
# ==================================================

APPROACH_FUNCS_3D = dict(zip(APPROACH_NAMES, APPROACHES))


def _runSingleSearch3D(
    graphSearchFn,
    G,
    startNodes,
    cellpriors,
    D,
    travelTime,
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
        None,
        None,
        None,
        cellpriors,
        D,
        maxTrees=maxTrees,
        populationSize=populationSize,
        historyCallback=historyCallback,
        searchMode=searchMode,
        travelTime=travelTime,
        probBudget=probBudget,
    )

    return {
        "strategy": strategy,
        "tree": tree,
        "checkedTrees": checkedTrees,
        "bestFitness": bestFitness,
        "history": history,
    }


def runSpanningTreeEvolution3D(
    sceneName: str,
    approachName: str,
    availableRobots: int,
    availableTime,
    maxTrees,
    populationSize: int,
    epsilon=None,
    probBudget=200,
):

    scene, G, D, startNodes, cellpriors, travelTime = build3DEnvironment(sceneName, epsilon=epsilon)

    graphSearchFn = APPROACH_FUNCS_3D[approachName]

    evolutionary = _runSingleSearch3D(
        graphSearchFn, G, startNodes, cellpriors, D, travelTime,
        availableRobots, availableTime, maxTrees, populationSize,
        searchMode="evolutionary",
        probBudget=probBudget,
    )

    random_ = _runSingleSearch3D(
        graphSearchFn, G, startNodes, cellpriors, D, travelTime,
        availableRobots, availableTime, maxTrees, populationSize,
        searchMode="random",
        probBudget=probBudget,
    )

    return {
        "scene": scene,
        "G": G,
        "D": D,
        "priors": cellpriors,
        "evolutionary": evolutionary,
        "random": random_,
    }


#--------------------------------------------------------
# SPANNING TREE EVOLUTION (3D) - REPEATED (100 RUNS) COMPARISON
#
# Same idea as runSpanningTreeEvolutionRepeated() in treeTest.py: repeats
# both searchModes numOfRuns times each, on the exact same 3D scene graph
# (built once), keeping only each run's final outcome (checked trees + best
# objective found).
#--------------------------------------------------------


def _runSearchModeRepeated3D(
    graphSearchFn,
    G,
    startNodes,
    cellpriors,
    D,
    travelTime,
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
            None,
            None,
            None,
            cellpriors,
            D,
            maxTrees=maxTrees,
            populationSize=populationSize,
            searchMode=searchMode,
            travelTime=travelTime,
            probBudget=probBudget,
        )

        bestFitness[i] = fitness
        checkedTrees[i] = treesChecked

    return {
        "bestFitness": bestFitness,
        "checkedTrees": checkedTrees,
    }


def runSpanningTreeEvolution3DRepeated(
    sceneName: str,
    approachName: str,
    availableRobots: int,
    availableTime,
    maxTrees,
    populationSize: int,
    numOfRuns: int = 100,
    epsilon=None,
    probBudget=200,
):

    scene, G, D, startNodes, cellpriors, travelTime = build3DEnvironment(sceneName, epsilon=epsilon)

    graphSearchFn = APPROACH_FUNCS_3D[approachName]

    evolutionary = _runSearchModeRepeated3D(
        graphSearchFn, G, startNodes, cellpriors, D, travelTime,
        availableRobots, availableTime, maxTrees, populationSize,
        searchMode="evolutionary",
        probBudget=probBudget,
        numOfRuns=numOfRuns,
    )

    random_ = _runSearchModeRepeated3D(
        graphSearchFn, G, startNodes, cellpriors, D, travelTime,
        availableRobots, availableTime, maxTrees, populationSize,
        searchMode="random",
        probBudget=probBudget,
        numOfRuns=numOfRuns,
    )

    return {
        "scene": scene,
        "G": G,
        "D": D,
        "priors": cellpriors,
        "evolutionary": evolutionary,
        "random": random_,
    }
