"""Monte Carlo comparison: Greedy BLabel Order's "Closing Exits" strategy
(a normal graphSearch() run - full clearance if one is found, otherwise its
computeClosingExits() fallback: local clearances + guards + a final FHPE_SA
patrol) against a plain FHPE_SA baseline that does no local clearing at all,
started from the exact same root.

Both strategies are scored by simulating a randomly moving target: at every
simulated time step its cell is redrawn from the SAME, never-updated prior
distribution (graphBuilder's per-cell priors), unless it has already been
"hurt" (probability q per step, checked once each step until it happens),
at which point it freezes in its current cell forever. Priors themselves are
never touched by this simulation - repeated draws are always i.i.d. from the
original distribution.

The target counts as found the instant its cell lies in the detection set of
a node a robot is AT (this includes a robot that arrives and immediately
continues on to its next move without lingering - an instantaneous touch
still counts, exactly like the rest of this codebase treats a node's own
first-arrival time as the moment it is "found" - not a continuous sweep
while mid-move between two nodes), OR - Closing Exits only - inside the
detection set of a node belonging to a REGION that has, by now, been fully
secured (graphSearch()'s new clearedRegionsOut: one region covering the
whole graph on a full clearance, or one entry per successfully-secured local
region from computeClosingExits otherwise). A region's cells only count as
safe from the moment its OWN clearance is entirely finished - the time its
own LAST node was first reached, not any single node's own visit time and
never from t=0 - and then stay safe for the rest of the run. The plain
FHPE_SA baseline never clears any region, so that second condition never
applies to it.

Both strategies are evaluated against the SAME drawn target trajectory each
run (paired sampling), so the comparison isn't muddied by also having to
average out independent target randomness between the two arms.
"""

import bisect

import numpy as np

from approachTest import (
    APPROACH_NAMES,
    APPROACHES,
    GRAPH_ALPHA,
    loadEnvironment,
    buildEnvironmentGraph,
    computeObstacleDistance,
)

_GREEDY_BLABEL_ORDER_SEARCH = APPROACHES[APPROACH_NAMES.index("Greedy BLabel Order")]


# ==================================================
# ENVIRONMENT (identical 2D synthetic grid every other test in this folder
# uses - see approachTest.py's buildEnvironment()/graphBuilder() call)
# ==================================================


def buildClosingExitsEnvironment(detecRad: int):

    env = loadEnvironment(detecRad)
    obstacles = env["obstacles"]
    alpha = env["alpha"]

    G, edges_shady, D, startNodes, priors = buildEnvironmentGraph(env)

    distanceMap = computeObstacleDistance(obstacles)

    return obstacles, G, edges_shady, D, startNodes, priors, distanceMap, alpha


# ==================================================
# ROBOT OCCUPANCY TIMELINE
# Turns a strategy's flat (source, target, count, departure, arrival) move
# list into a step function: at any time t, which nodes currently have at
# least one robot physically standing on them.
# ==================================================


def _buildOccupancyTimeline(strategy):

    events = []

    _, rootNode, count = strategy[0]
    events.append((0.0, count, rootNode))

    for (src, tgt, cnt, dep, arr) in strategy[1:]:
        events.append((dep, -cnt, src))
        events.append((arr, cnt, tgt))

    events.sort(key=lambda e: e[0])

    startTimes = []
    occupiedSets = []
    counts = {}

    idx = 0
    n = len(events)

    while idx < n:

        t = events[idx][0]

        while idx < n and events[idx][0] == t:
            _, delta, node = events[idx]
            counts[node] = counts.get(node, 0) + delta
            idx += 1

        occupied = frozenset(node for node, c in counts.items() if c > 0)

        startTimes.append(t)
        occupiedSets.append(occupied)

    return startTimes, occupiedSets


def _unionCells(D, nodeSet):

    cells = set()

    for node in nodeSet:
        cells.update(D[node])

    return frozenset(cells)


def _liveDetectionTimeline(strategy, D):
    """Per-interval cells covered by a robot's LIVE, SUSTAINED detection
    (i.e. continuing to stand at a node between an arrival and its next
    later departure) - this can start AND stop over time as robots move
    on. Deliberately does NOT capture an instantaneous touch (a robot whose
    arrival and next departure share the very same timestamp - the two
    events cancel out and never make the node's count go above zero for
    any positive-duration interval); _buildArrivalEvents below covers that
    case separately."""

    startTimes, occupiedSets = _buildOccupancyTimeline(strategy)

    detectionCells = [_unionCells(D, occ) for occ in occupiedSets]

    return startTimes, detectionCells


def _buildArrivalEvents(strategy):
    """Sorted (arrivalTime, node) for every move's arrival, plus the root at
    t=0 - a POINT event per arrival, independent of how long (if at all)
    the robot then stays. This is what actually catches a robot that
    arrives and immediately continues on to its next planned move (a very
    common FHPE_SA pattern): _liveDetectionTimeline's occupancy-COUNT model
    can't represent a zero-duration stay at all, since the matching arrival
    and departure events at the same timestamp cancel out.

    Returns (sortedTimes, events) - sortedTimes is just [t for t, _ in
    events], kept separate for bisect.
    """

    events = []

    _, rootNode, _ = strategy[0]
    events.append((0.0, rootNode))

    for (src, tgt, cnt, dep, arr) in strategy[1:]:
        events.append((arr, tgt))

    events.sort(key=lambda e: e[0])

    return [t for t, _ in events], events


def _clearedCellsTimeline(D, clearedRegions):
    """A region's detection cells only become - and then stay - safe from
    the moment its OWN clearance is entirely finished (its completionTime,
    the time its last node was first reached), never earlier and never
    per-node: the whole region has to actually finish clearing first.
    Monotonically growing over time (a cleared region never becomes unsafe
    again). clearedRegions is graphSearch()'s clearedRegionsOut -
    [(nodeSet, completionTime), ...]."""

    if not clearedRegions:
        return [0.0], [frozenset()]

    ordered = sorted(clearedRegions, key=lambda region: region[1])

    startTimes = []
    cumulativeCells = []
    accumulated = set()

    for nodeSet, completionTime in ordered:
        accumulated.update(_unionCells(D, nodeSet))
        startTimes.append(completionTime)
        cumulativeCells.append(frozenset(accumulated))

    return startTimes, cumulativeCells


def _cellsAt(startTimes, cellsPerInterval, t):
    """The state that applies AT time t - i.e. the last entry whose own
    time is <= t. Before the first recorded time (relevant for
    _clearedCellsTimeline, whose first entry can be well after t=0), NONE
    of the recorded states apply yet, so this returns an empty set rather
    than incorrectly reusing the first entry (see _liveDetectionTimeline,
    whose startTimes[0] is always 0.0, so this branch never triggers
    there)."""

    i = bisect.bisect_right(startTimes, t) - 1

    if i < 0:
        return frozenset()

    return cellsPerInterval[i]


# ==================================================
# TARGET SIMULATION
# ==================================================


def _simulateTargetTrajectory(cellCumWeights, hurtProbability, probBudget, dt, rng):
    """[(t, cellIdx), ...] for t = 0, dt, 2*dt, ... up to probBudget. Once
    "hurt" triggers (probability hurtProbability, checked once per step
    until it fires), the target freezes at its current cell for every
    remaining step.

    cellCumWeights: cellWeights.cumsum() - precomputed ONCE by the caller
    (not per draw, and not even once per run) so drawing a cell here is a
    single np.searchsorted() (O(log n)), not rng.choice(..., p=cellWeights)
    recomputing its own cumulative sum from scratch on every single call
    (O(n) each). For a 2D grid's ~100 cells that redundant O(n) is free, but
    a real 3D scene can have hundreds of thousands of cells and this runs
    every dt of simulated time, every run - i.e. exactly the kind of
    per-iteration O(cells) cost this codebase has repeatedly had to hunt
    down elsewhere.
    """

    trajectory = []

    def draw():
        return int(np.searchsorted(cellCumWeights, rng.random(), side="right"))

    hurt = False
    currentCellIdx = draw()

    t = 0.0
    while t <= probBudget:

        trajectory.append((t, currentCellIdx))

        t += dt

        if not hurt:
            if rng.random() < hurtProbability:
                hurt = True
            else:
                currentCellIdx = draw()

    return trajectory


def _foundDuringWindow(
    cell, D,
    liveStartTimes, liveCells,
    arrivalTimesSorted, arrivalEvents,
    clearedStartTimes, clearedCells,
    windowStart, windowEnd,
):
    """Whether `cell` is detected at ANY point during [windowStart,
    windowEnd) - the target's cell is constant across this whole window
    (it only redraws at window boundaries), so any overlap counts:
      - already under SUSTAINED live detection right when the window
        starts (covers a robot present since before this window), OR
      - any arrival (however brief) happens strictly inside the window
        (covers a robot reaching/passing through mid-window - including a
        zero-duration touch _liveDetectionTimeline can't represent), OR
      - the cell is inside an already-secured region at the window's
        start (a region's own safety only ever turns on, never off, so
        checking the start is enough - it's already been safe for a
        while by construction whenever it's safe at all here)."""

    if cell in _cellsAt(liveStartTimes, liveCells, windowStart):
        return True

    lo = bisect.bisect_left(arrivalTimesSorted, windowStart)
    hi = bisect.bisect_left(arrivalTimesSorted, windowEnd)

    for i in range(lo, hi):
        if cell in D[arrivalEvents[i][1]]:
            return True

    if cell in _cellsAt(clearedStartTimes, clearedCells, windowStart):
        return True

    return False


def _findTime(
    trajectory, dt, cellIndices,
    D,
    liveStartTimes, liveCells,
    arrivalTimesSorted, arrivalEvents,
    clearedStartTimes, clearedCells,
):

    for t, cellIdx in trajectory:

        cell = cellIndices[cellIdx]

        if _foundDuringWindow(
            cell, D,
            liveStartTimes, liveCells,
            arrivalTimesSorted, arrivalEvents,
            clearedStartTimes, clearedCells,
            t, t + dt,
        ):
            return t

    return None  # timeout - never found within probBudget


# ==================================================
# ONE (Closing Exits, plain FHPE_SA) COMPARISON RUN
# ==================================================


def _runOneComparison(
    G, obstacles, distanceMap, startNodes, priors, D,
    availableRobots, availableTime, maxTrees,
    horizon, recencyLambda, probBudget, maxHops,
    cellIndices, cellCumWeights, hurtProbability, dt, rng,
    travelTime=None,
    alpha=GRAPH_ALPHA,
):

    clearedRegions = []

    strategyCE, _, checkedTreesCE, fitnessCE = _GREEDY_BLABEL_ORDER_SEARCH(
        G, availableTime, availableRobots, startNodes,
        obstacles, distanceMap, alpha, priors, D,
        maxTrees=maxTrees,
        horizon=horizon, recencyLambda=recencyLambda,
        probBudget=probBudget, maxHops=maxHops,
        clearedRegionsOut=clearedRegions,
        travelTime=travelTime,
    )

    root = strategyCE[0][1]

    # Plain FHPE_SA baseline - same root, no local clearing at all first.
    strategyFHPE, _, _, _ = _GREEDY_BLABEL_ORDER_SEARCH(
        G, availableTime, availableRobots, startNodes,
        obstacles, distanceMap, alpha, priors, D,
        horizon=horizon, recencyLambda=recencyLambda,
        probBudget=probBudget, maxHops=maxHops,
        skipTreeSearch=True, startRootOverride=root,
        travelTime=travelTime,
    )

    liveStartTimesCE, liveCellsCE = _liveDetectionTimeline(strategyCE, D)
    liveStartTimesFHPE, liveCellsFHPE = _liveDetectionTimeline(strategyFHPE, D)

    arrivalTimesCE, arrivalEventsCE = _buildArrivalEvents(strategyCE)
    arrivalTimesFHPE, arrivalEventsFHPE = _buildArrivalEvents(strategyFHPE)

    clearedStartTimesCE, clearedCellsCE = _clearedCellsTimeline(D, clearedRegions)
    # Plain FHPE_SA never clears anything - an empty, permanently-empty
    # timeline, so this check never fires for it.
    clearedStartTimesFHPE, clearedCellsFHPE = [0.0], [frozenset()]

    trajectory = _simulateTargetTrajectory(
        cellCumWeights, hurtProbability, probBudget, dt, rng
    )

    findTimeCE = _findTime(
        trajectory, dt, cellIndices, D,
        liveStartTimesCE, liveCellsCE,
        arrivalTimesCE, arrivalEventsCE,
        clearedStartTimesCE, clearedCellsCE,
    )
    findTimeFHPE = _findTime(
        trajectory, dt, cellIndices, D,
        liveStartTimesFHPE, liveCellsFHPE,
        arrivalTimesFHPE, arrivalEventsFHPE,
        clearedStartTimesFHPE, clearedCellsFHPE,
    )

    totalClearedNodes = set()
    for nodeSet, _completionTime in clearedRegions:
        totalClearedNodes |= nodeSet
    clearanceMass = len(totalClearedNodes) / len(G.nodes) if len(G.nodes) else 0.0

    return findTimeCE, findTimeFHPE, clearanceMass, checkedTreesCE, fitnessCE


# ==================================================
# FULL MONTE CARLO COMPARISON
# ==================================================


def _summarize(findTimes):

    found = [t for t in findTimes if t is not None]

    timeoutRate = 1.0 - len(found) / len(findTimes) if findTimes else 0.0
    meanTime = float(np.mean(found)) if found else None
    varianceTime = float(np.var(found)) if found else None
    p90Time = float(np.percentile(found, 90)) if found else None

    return meanTime, varianceTime, p90Time, timeoutRate


def runClosingExitsComparisonOnGraph(
    G, obstacles, distanceMap, startNodes, priors, D,
    cellIndices, cellWeights,
    numOfRuns: int,
    availableRobots: int,
    availableTime,
    maxTrees,
    horizon: int,
    probBudget: float,
    hurtProbability: float,
    maxHops: int = 4,
    recencyLambda=None,
    dt: float = 1.0,
    seed=None,
    travelTime=None,
    alpha=GRAPH_ALPHA,
):
    """The shared Monte Carlo comparison loop, given an already-built graph
    (2D synthetic grid or a real 3D scene - see runClosingExitsComparison()
    below for the 2D entry point, and 3DTest.py's
    runClosingExitsComparison3D() for the 3D one). Both just build their own
    environment/cellIndices/cellWeights and hand them to this.

    cellIndices: every valid cell, as whatever 2-tuple D's detection sets
        use - (x, y) grid coordinates for the 2D grid, (cellId, 0) for a 3D
        scene (matching cellpriors[cell[0], cell[1]]'s convention - see any
        Searcher.py). cellWeights: matching per-cell prior mass, summing to
        1 (the target's redraw distribution).
    """

    cellCumWeights = np.cumsum(cellWeights)

    rng = np.random.default_rng(seed)

    #------------------------------------------------------------------
    # RUN BOTH STRATEGIES ON THE SAME GRAPH, numOfRuns TIMES, EACH TIME
    # AGAINST THE SAME DRAWN TARGET TRAJECTORY
    #------------------------------------------------------------------
    findTimesCE = []
    findTimesFHPE = []
    clearanceMasses = []
    resultsTrees = []

    for i in range(numOfRuns):

        findTimeCE, findTimeFHPE, clearanceMass, checkedTrees, _fitnessCE = _runOneComparison(
            G, obstacles, distanceMap, startNodes, priors, D,
            availableRobots, availableTime, maxTrees,
            horizon, recencyLambda, probBudget, maxHops,
            cellIndices, cellCumWeights, hurtProbability, dt, rng,
            travelTime=travelTime,
            alpha=alpha,
        )

        findTimesCE.append(findTimeCE)
        findTimesFHPE.append(findTimeFHPE)
        clearanceMasses.append(clearanceMass)
        resultsTrees.append(checkedTrees)

        print("Run Nummer " + str(i) + " beendet!")

    meanCE, varianceCE, p90CE, timeoutCE = _summarize(findTimesCE)
    meanFHPE, varianceFHPE, p90FHPE, timeoutFHPE = _summarize(findTimesFHPE)

    summary = {
        "Closing Exits": {
            "mean_find_time": meanCE,
            "variance_find_time": varianceCE,
            "p90_find_time": p90CE,
            "timeout_rate": timeoutCE,
            "clearance_mass": float(np.mean(clearanceMasses)) if clearanceMasses else 0.0,
        },
        "Plain FHPE_SA": {
            "mean_find_time": meanFHPE,
            "variance_find_time": varianceFHPE,
            "p90_find_time": p90FHPE,
            "timeout_rate": timeoutFHPE,
            "clearance_mass": 0.0,
        },
    }

    return {
        "G": G,
        "priors": priors,
        "numGraphNodes": len(G.nodes),
        "numGraphEdges": len(G.edges),
        "findTimesClosingExits": findTimesCE,
        "findTimesFHPE": findTimesFHPE,
        "clearanceMasses": clearanceMasses,
        "resultsTrees": resultsTrees,
        "summary": summary,
    }


def runClosingExitsComparison(
    detecRad: int,
    numOfRuns: int,
    availableRobots: int,
    availableTime,
    maxTrees,
    horizon: int,
    probBudget: float,
    hurtProbability: float,
    maxHops: int = 4,
    recencyLambda=None,
    dt: float = 1.0,
    seed=None,
):

    #-------------------------------------------------------------------
    # ALLOCATION (identical 2D grid every other test here uses - NUR EIN
    # GRAPH: built once, reused for every run)
    #-------------------------------------------------------------------
    obstacles, G, edges_shady, D, startNodes, priors, distanceMap, alpha = buildClosingExitsEnvironment(detecRad)

    H, W = obstacles.shape
    cellIndices = [(r, c) for r in range(H) for c in range(W) if obstacles[r, c] == 0]
    cellWeights = np.array([priors[r, c] for (r, c) in cellIndices], dtype=float)
    cellWeights = cellWeights / cellWeights.sum()

    return runClosingExitsComparisonOnGraph(
        G, obstacles, distanceMap, startNodes, priors, D,
        cellIndices, cellWeights,
        numOfRuns,
        availableRobots, availableTime, maxTrees,
        horizon, probBudget, hurtProbability,
        maxHops=maxHops, recencyLambda=recencyLambda, dt=dt, seed=seed,
        alpha=alpha,
    )
