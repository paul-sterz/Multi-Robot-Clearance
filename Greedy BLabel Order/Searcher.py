import numpy as np
import math
import random
import sys
from Graph import Graph, Node, Edge
from itertools import combinations
from TrajectoryPlanning import aStar
from collections import defaultdict
import copy
import time


def graphSearch(G : Graph, availableTime, availableRobots, startNodes, obstacles, distanceMap, alpha, cellpriors, D, maxTrees=None, populationSize=10, historyCallback=None, searchMode="evolutionary", travelTime=None, horizon=5, recencyLambda=None, probBudget = 200, maxHops=4):
    #INPUT:
    # G: a Graph object repesenting the merged navigationgraph
    # availableTime: the available computation time budget in seconds (ignored if maxTrees is given)
    # availableRobots: an integer representing the amount of robots available
    # maxTrees: if given (not None), stop after evaluating exactly this many spanning trees instead of using availableTime
    # startNodes:  an integer representing that all nodes from 0 to startNode-1 are valid startNodes for our Algorithim
    # obstacles:
    # distanceMap: 
    # alpha:
    # cellpriors: All prior values for each cell for calculating the expected searchtime function
    # D: Detection set for each node for calculating the expected searchtime function
    # horizon: max number of EDGES a candidate path may chain in one of FHPE_SA's
    #          finite-horizon path enumeration calls (the fallback strategy when no
    #          clearance is found) - a hop COUNT, not a time budget, since edges can take
    #          very different amounts of time to travel
    # recencyLambda: recency decay constant used by FHPE_SA, in the same time units as
    #          edge travel times (default None -> probBudget / 10)
    # probBudget: simulated-time budget used by FHPE_SA - every move it plans arrives at its
    #          target by t = probBudget at the latest (default 200); NOT a wall-clock limit
    # maxHops: additional hard cap on how many edges FHPE_SA's brute-force search may chain
    #          in one horizon-planning call, on top of horizon itself (default 4) - whichever
    #          of horizon/maxHops is smaller effectively wins
    # aerialSpeed: TO-DO
    # groundSpeed: TO-DO
    # aerialBattery: TO-DO
    # groundBattery: TO-DO

    #OUTPUT:
    #bestStrategy: A list containing the best strategy where each entry is in the form (source node,target node, amount of Robots)

    # ---------------------------------------------------
    # PRECOMPUTING ALL-PAIRS FLAG DISTANCES
    # obstacles/distanceMap/alpha and the node positions in G never change
    # for the duration of this graphSearch() call, so the aStar step-count
    # between any two node indices is the same for every spanning tree
    # evaluated below. Computing it once here - instead of re-running aStar
    # inside findNearestFlag() for every flag/node pair on every tree -
    # turns the dominant cost of the search into a single upfront O(n^2)
    # pass over a lookup table.
    # ---------------------------------------------------
    numGraphNodes = len(G.nodes)

    # A caller that already has an all-pairs travel-time table (e.g. a real
    # scene's shortest-path matrix) can pass it in directly and skip the aStar
    # sweep below entirely -- it depends only on the node positions and never
    # changes during this call either way.
    if travelTime is not None:
        flagDistance = travelTime
    else:
        flagDistance = [[0] * numGraphNodes for _ in range(numGraphNodes)]

        for i in range(numGraphNodes):
            for j in range(numGraphNodes):
                if i == j:
                    continue
                path = aStar(G.nodes[i].pos, G.nodes[j].pos, obstacles, distanceMap, alpha)
                flagDistance[i][j] = len(path) - 1

    # horizon is now a hop COUNT (max edges a candidate path may chain, see
    # bestPathForSearcher below), not a time budget - so it can no longer
    # anchor a TIME decay constant. lam still needs a genuine time scale, so
    # it now defaults off probBudget (the actual mission timescale) instead.
    lam = recencyLambda if recencyLambda is not None else probBudget / 10

    # ---------------------------------------------------
    # COMPUTING EDGE LABLES FOR THE TREE SEARCH THAT REPRESENT THE AMOUNT OF NEEDED ROBOTS AND THE EFFICIENCY OF EACH SUBTREE
    # Remark: Lables represent the amount of robots needed for this path
    # ---------------------------------------------------

    def computeWorstCaseLabels(T : Graph, root, parent):

        #Saving labels in a dictionary of the form: ((x,y) | lambda((x,y))
        edgeLabelsRobotCost = {}
        edgeLabelsEfficiency = {}
        # Same keys as edgeLabelsEfficiency, but the raw (totalTime, totalPrior)
        # sum pair the ratio was built from, instead of just their ratio -
        # needed wherever both quantities are required separately (e.g. the
        # local clearance candidate scoring in computeClosingExits).
        edgeLabelsTotals = {}


        # Keep track of total prior and time in subtree
        totalTime = 0
        totalPrior = 0

        #Saving pi meaning all children of the root
        childLabels = [] 
        children = []
        subTreePriors = {} #in form (child | subTreePrior)
        subTreeTimes = {} #in form (child | subTreeTime)

        # Calculating lables recursive for children
        for y in T.adj[T.nodes[root]]:
            if y.idx != parent:
                subLabelsRobotCost, subLablesEfficiency, subLabelsTotals, sumTime, sumPrior = computeWorstCaseLabels(T, y.idx, root)
                totalTime += sumTime
                totalPrior += sumPrior

                subTreePriors[y.idx] = sumPrior
                subTreeTimes[y.idx] =  sumTime

                edgeLabelsEfficiency.update(subLablesEfficiency)
                edgeLabelsRobotCost.update(subLabelsRobotCost)
                edgeLabelsTotals.update(subLabelsTotals)
                childLabels.append(subLabelsRobotCost[(root, y.idx)])
                children.append(y.idx)
  
        # Check if leaf
        if len(childLabels) == 0:
            edgeLabelsRobotCost[(parent, root)] = 1
            
            if parent != None:
                totalPrior += T.nodes[root].prior 
                totalTime +=  T.edges[(parent,root)].time
            else:
                totalPrior += T.nodes[root].prior 
                totalTime +=  1

            edgeLabelsEfficiency[(parent, root)] = totalPrior / totalTime
            edgeLabelsTotals[(parent, root)] = (totalTime, totalPrior)
        else:

            #formula from the paper for robot cost
            childLabels.sort(reverse=True)

            p1 = childLabels[0]
            if len(childLabels) > 1:  
                p2 = childLabels[1]
            else: p2 = 0
    
            if p1 == 1:
                currentLabel = p1 + 1
            else:
                currentLabel = max(p1, p2 + 1)

            edgeLabelsRobotCost[(parent, root)] = currentLabel

            #setting the efficiency edgelable
            totalPrior += T.nodes[root].prior 
            if parent != None:
                totalTime = T.edges[(parent,root)].time + totalTime
            else:
                totalTime = 1
            edgeLabelsEfficiency[(parent, root)] = totalPrior / totalTime
            edgeLabelsTotals[(parent, root)] = (totalTime, totalPrior)


        return edgeLabelsRobotCost, edgeLabelsEfficiency, edgeLabelsTotals, totalTime, totalPrior

    
     # ---------------------------------------------------
    # CALCULATING A STRATEGY FOR TREES AND TRANSFORMING IT TO THE GRAPH
    # Note: A Strategy is safed in the format [(source node idx,target node idx, amount of robots, time_Depature, time_Arrival),...]
    # Approach (same two-stage split as Baseline's Searcher):
    #   1) Build the abstract strategy purely as [(node, releaseIndex), ...]
    #      - the order the tree is walked in, and the strategy index(es)
    #      each guard depends on before it may leave. Children of a node
    #      are ordered by descending efficiency label, except the single
    #      costliest (by BLabel/robot cost) child, which always goes last
    #      - so it is still guarded by its parent while every cheaper
    #      sibling subtree gets cleared first.
    #   2) Feed that abstract strategy into the same multi-level LBAP
    #      assignment Baseline uses, except the robot count is fixed
    #      up-front to availableRobots instead of starting from the
    #      B-label bound and retrying with one extra robot on failure -
    #      if availableRobots does not suffice, clearance simply fails.
    # ---------------------------------------------------
    def treeSearch(T : Graph, root, availableRobots, G : Graph, startState=None, recordCandidates=True):
        # ------------------------------------------------
        # INPUT:
        # T - The Tree on which the strategy is calculated (Graph object)
        # root - The starting point of the strategy (index)
        # availableRobots - The amount of robots that can be used for exploration (non negative Integer)
        # G - The Navigation Graph (Graph object)
        # startState - per-robot (node, freeFromTime) list, if the
        #   committed robots don't simply all start together at root at
        #   t=0 (e.g. a chain of local clearance candidates inside
        #   computeClosingExits, where each candidate's robots are really
        #   whatever robots are currently free, scattered wherever their
        #   previous job left them). Must have exactly availableRobots
        #   entries when given. Defaults to None (all at root, t=0),
        #   matching prior behaviour - the LBAP then also transparently
        #   handles routing directly from wherever each robot really is.
        # recordCandidates - whether a failing tree here should feed
        #   computeLocalClearanceCandidates() into the outer
        #   localClearanceCandidates list. Disabled when treeSearch is
        #   itself being used to evaluate local-candidate subtrees inside
        #   computeClosingExits, where recording sub-candidates would just
        #   be wasted work on trees that are already local subtrees.
        # ------------------------------------------------
        # OUTPUT:
        # strategy, clearance, visitedTimes, finalRobotStates - clearance
        # is False (with the other two None) whenever availableRobots does
        # not suffice. finalRobotStates is the same per-robot (node,
        # freeFromTime) shape as startState, reflecting where every
        # committed robot ends up - the natural input to the NEXT chained
        # treeSearch/FHPE_SA call.

        BLabels, effLables, subtreeTotals, _, _ = computeWorstCaseLabels(T, root, None)

        # ------------------------------------------------
        # LOCAL CLEARANCE CANDIDATES
        # A local clearance candidate is a subtree of T (identified by its
        # root node) whose own B-label plus its "Gegner" fits within
        # availableRobots. Gegner ("opponents") of a subtree are every node
        # of G outside the subtree that has an edge into a node of the
        # subtree - i.e. every source contamination could re-enter the
        # subtree from once its guards leave. A node bordering several
        # nodes of the subtree is still only counted once.
        # Only computed on demand (see call site below), since it is only
        # needed while no clearance strategy has been found yet.
        # ------------------------------------------------
        def computeLocalClearanceCandidates():

            candidates = []
            subtreeNodes = {}

            def visit(node, parent):
                nodes = {node}
                for child in T.adj[T.nodes[node]]:
                    if child.idx != parent:
                        visit(child.idx, node)
                        nodes |= subtreeNodes[child.idx]
                subtreeNodes[node] = nodes

                opponents = {
                    neighbour.idx
                    for u in nodes
                    for neighbour in G.adj[G.nodes[u]]
                    if neighbour.idx not in nodes
                }

                bLabel = BLabels[(parent, node)]

                if bLabel + len(opponents) <= availableRobots:
                    totalTime, totalPrior = subtreeTotals[(parent, node)]
                    candidates.append({
                        "root": node,
                        "parent": parent,
                        "nodes": nodes,
                        "bLabel": bLabel,
                        "opponents": len(opponents),
                        # The actual opponent node indices (not just their
                        # count) - needed later to tell whether an opponent
                        # of THIS candidate is already guarded because it is
                        # also an opponent of some other, already-cleared
                        # candidate (see computeClosingExits).
                        "opponentNodes": opponents,
                        "totalEdgeTime": totalTime,
                        "totalPrior": totalPrior,
                    })

            visit(root, None)
            return candidates

        visited = [0] * len(T.nodes)
        visited[root] = 1

        # Strategy is built in-place, in true final traversal order, so
        # that a "releaseIndex" resolved to len(strategy) always matches
        # the index that entry will actually end up at.
        strategy = []
        nodeStrategyIndex = {}

        def addEntry(node):
            # enemys = graph-neighbours of node not yet reached by the
            # search strategy at this point -> node's guard must stay
            # until all of them have been visited.
            #
            # A neighbour already visited at this point is NOT
            # automatically safe: the LBAP later assigns whichever robot
            # is physically fastest to each task, so an
            # earlier-scheduled ("already visited") neighbour can still
            # end up with a LATER real arrival time than this node. Such
            # neighbours go straight into "deps" as known strategy
            # indices, to be resolved against their real execution time
            # once it is known; still-unscheduled neighbours go into
            # "pending", resolved once they are added later.
            pendingEnemys = []
            knownDeps = []

            for neighbour in G.adj[G.nodes[node]]:
                if visited[neighbour.idx] == 0:
                    pendingEnemys.append(neighbour.idx)
                else:
                    knownDeps.append(nodeStrategyIndex[neighbour.idx])

            myIndex = len(strategy)

            if len(pendingEnemys) > 0:
                strategy.append([node, {"pending": set(pendingEnemys), "deps": knownDeps}])
            elif len(knownDeps) > 0:
                strategy.append([node, knownDeps])
            else:
                strategy.append([node, [myIndex]])

            nodeStrategyIndex[node] = myIndex

        addEntry(root)

        def explorePath(node, parent):

            neighbours = [
                neighbour
                for neighbour in T.adj[T.nodes[node]]
                if neighbour.idx != parent
            ]

            if len(neighbours) == 0:
                return

            maxBLabel = max(BLabels[(node, neighbour.idx)] for neighbour in neighbours)
            maxBLabelNeighbours = [neighbour for neighbour in neighbours if BLabels[(node, neighbour.idx)] == maxBLabel]

            if len(maxBLabelNeighbours) >= 2:
                #Ambiguous largest BLabel: no order can guarantee the "largest
                #last" property, so fall back to a plain efficiency order instead
                #of failing outright.
                orderedNeighbours = sorted(
                    neighbours,
                    key=lambda neighbour: effLables[(node, neighbour.idx)],
                    reverse=True
                )
            else:
                largestNeighbour = maxBLabelNeighbours[0]
                orderedNeighbours = sorted(
                    (neighbour for neighbour in neighbours if neighbour.idx != largestNeighbour.idx),
                    key=lambda neighbour: effLables[(node, neighbour.idx)],
                    reverse=True
                )
                orderedNeighbours.append(largestNeighbour)

            for neighbour in orderedNeighbours:

                for entry in strategy:
                    rel = entry[1]
                    if isinstance(rel, dict) and neighbour.idx in rel["pending"]:
                        rel["pending"].discard(neighbour.idx)
                        rel["deps"].append(len(strategy))
                        if not rel["pending"]:
                            # Fully resolved: from now on this is the
                            # complete list of strategy indices whose real
                            # completion time this guard must wait for.
                            entry[1] = rel["deps"]

                visited[neighbour.idx] = 1

                addEntry(neighbour.idx)

                explorePath(neighbour.idx, node)

        explorePath(root, None)

        strategy, clearance, visitedTimes, finalRobotStates = transformStrategy(strategy, availableRobots, root, startState=startState)

        # Only worth computing while we still have no clearance strategy at
        # all, and only for a tree that just failed - a successful tree
        # makes the fallback (which consumes these candidates) moot.
        # Candidates accumulate across every failing tree seen so far
        # (never cleared here) - each one is stored purely as a set of node
        # indices, so it stays usable later without needing that tree
        # around any more.
        if recordCandidates and not foundClearance and not clearance:
            localClearanceCandidates.extend(computeLocalClearanceCandidates())

        return strategy, clearance, visitedTimes, finalRobotStates


    def transformStrategy(strategy, availableRobots, root, startState=None):
        """
        strategy:
            [(node, releaseIndex), ...]

            releaseIndex = last strategy index up to which the robot on
            this node is still needed as a guard.

        availableRobots:
            fixed robot budget. Unlike Baseline this never grows: if
            availableRobots does not suffice, clearance fails outright
            instead of retrying with one more robot.

        startState:
            optional [(node, freeFromTime), ...] with exactly
            availableRobots entries, one per robot, if they don't all
            simply start together at root at t=0 (see treeSearch's
            docstring). None (default) means all availableRobots robots
            start at root at t=0, and the returned executionPlan gets the
            usual (None, root, availableRobots) sentinel as its first
            entry; when startState is given explicitly, that sentinel is
            omitted instead (the caller already knows where its robots are
            - it is the one chaining several such calls together and
            keeping its own single overarching sentinel).

        Assumption:
            flagDistance[u][v] is available in the surrounding scope.

        Returns:
            executionPlan, clearance, visitedTimes, finalRobotStates

            finalRobotStates is the same [(node, freeFromTime), ...] shape
            as startState - where every one of the availableRobots robots
            ends up, ready to be threaded into the next chained call.

        executionPlan format (same as Baseline):
            First entry is (None, root, availableRobots) UNLESS startState
            was given explicitly (see above); all following entries are
            (source, target, robots, t_departure, t_arrival).
        """

        usingDefaultStart = startState is None
        if usingDefaultStart:
            startState = [(root, 0)] * availableRobots

        # ============================================================
        # LBAP
        # ============================================================

        def LBAPSolver(level, robots):
            """
            level:
                [(taskIndex, node, releaseIndex), ...]

            robots:
                complete robot list.

            Only currently free robots are considered.

            Cost:
                robot["freeFrom"]
                + flagDistance[robot["node"]][taskNode]

            Returns:
                assignment:
                    taskIndex -> robotId
            """

            freeRobots = [
                robot for robot in robots
                if robot["guardUntil"] is None
            ]

            if len(freeRobots) < len(level):
                raise RuntimeError(
                    f"LBAP impossible: {len(level)} tasks but only "
                    f"{len(freeRobots)} free robots."
                )

            # --------------------------------------------------------
            # Cost matrix
            #
            # rows    = robots
            # columns = tasks
            # --------------------------------------------------------

            costs = []

            for robot in freeRobots:

                row = []

                for taskIndex, node, releaseIndex in level:

                    travelTime = flagDistance[robot["node"]][node]

                    earliestArrival = (
                        robot["freeFrom"] + travelTime
                    )

                    row.append(earliestArrival)

                costs.append(row)

            numRobots = len(freeRobots)
            numTasks = len(level)

            # --------------------------------------------------------
            # Test whether matching with all edges <= threshold exists
            # --------------------------------------------------------

            def matchingForThreshold(threshold):

                # taskToRobot[j] = robot-row currently assigned to task j
                taskToRobot = [-1] * numTasks

                def augment(robotIdx, visitedTasks):

                    for taskIdx in range(numTasks):

                        if visitedTasks[taskIdx]:
                            continue

                        if costs[robotIdx][taskIdx] > threshold:
                            continue

                        visitedTasks[taskIdx] = True

                        # Task not yet assigned
                        if taskToRobot[taskIdx] == -1:
                            taskToRobot[taskIdx] = robotIdx
                            return True

                        # Try to move currently assigned robot elsewhere
                        oldRobot = taskToRobot[taskIdx]

                        if augment(oldRobot, visitedTasks):
                            taskToRobot[taskIdx] = robotIdx
                            return True

                    return False

                matched = 0

                for robotIdx in range(numRobots):

                    visitedTasks = [False] * numTasks

                    if augment(robotIdx, visitedTasks):
                        matched += 1

                        if matched == numTasks:
                            break

                if matched != numTasks:
                    return None

                return taskToRobot

            # --------------------------------------------------------
            # Possible bottleneck values
            # --------------------------------------------------------

            thresholds = sorted({
                costs[r][t]
                for r in range(numRobots)
                for t in range(numTasks)
            })

            # --------------------------------------------------------
            # Binary search for smallest feasible bottleneck
            # --------------------------------------------------------

            left = 0
            right = len(thresholds) - 1

            bestMatching = None
            bestThreshold = None

            while left <= right:

                mid = (left + right) // 2
                threshold = thresholds[mid]

                matching = matchingForThreshold(threshold)

                if matching is not None:
                    bestMatching = matching
                    bestThreshold = threshold
                    right = mid - 1
                else:
                    left = mid + 1

            if bestMatching is None:
                raise RuntimeError("No feasible LBAP assignment found.")

            # --------------------------------------------------------
            # Convert matching to taskIndex -> robotId
            # --------------------------------------------------------

            assignment = {}

            for localTaskIdx, robotRow in enumerate(bestMatching):

                globalTaskIndex = level[localTaskIdx][0]
                robotId = freeRobots[robotRow]["id"]

                assignment[globalTaskIndex] = robotId

            return assignment, bestThreshold


        # ============================================================
        # LEVEL DISTRIBUTION
        # Same bookkeeping as Baseline, except once the fixed
        # availableRobots budget can't fit a level, this reports failure
        # (None) instead of recursing with one more robot.
        # ============================================================

        def levelDistribution(strategy, robotBudget):

            levels = []
            currentLevel = []

            # Only release indices matter here.
            # Concrete robot IDs are determined later by the LBAP.
            activeGuards = []

            levelCapacity = None

            for i, (node, releaseIndex) in enumerate(strategy):

                # releaseIndex is the full list of strategy indices this
                # guard depends on; for capacity bookkeeping only the
                # LAST one to occur (highest index) matters, since that
                # is when the guard structurally becomes releasable.
                maxRelease = max(releaseIndex)

                # ----------------------------------------------------
                # Start new level
                # ----------------------------------------------------

                if not currentLevel:

                    levelCapacity = robotBudget - len(activeGuards)

                    if levelCapacity < 1:
                        return None

                # ----------------------------------------------------
                # No more robots available inside this level
                # ----------------------------------------------------

                if len(currentLevel) >= levelCapacity:

                    levels.append(currentLevel)
                    currentLevel = []

                    levelCapacity = robotBudget - len(activeGuards)

                    if levelCapacity < 1:
                        return None

                # Store global strategy index as well!
                currentLevel.append(
                    (i, node, releaseIndex)
                )

                # ----------------------------------------------------
                # Old guards released by reaching task i
                # ----------------------------------------------------

                freedNow = [
                    release
                    for release in activeGuards
                    if release == i
                ]

                if freedNow:

                    activeGuards = [
                        release
                        for release in activeGuards
                        if release != i
                    ]

                # ----------------------------------------------------
                # Current task creates new guard
                #
                # With backward dependencies now possible (a neighbour
                # visited earlier in strategy order, see addEntry above),
                # maxRelease can legitimately be <= i - that still means
                # no guard persists past this task, exactly like maxRelease
                # == i used to.
                # ----------------------------------------------------

                if maxRelease > i:
                    activeGuards.append(maxRelease)

                # ----------------------------------------------------
                # Change in free robot set -> level ends
                # ----------------------------------------------------

                if freedNow or maxRelease <= i:

                    levels.append(currentLevel)

                    currentLevel = []
                    levelCapacity = None

            if currentLevel:
                levels.append(currentLevel)

            return levels


        # ============================================================
        # CREATE LEVELS
        # ============================================================

        levels = levelDistribution(strategy, availableRobots)

        if levels is None:
            # availableRobots does not suffice for this tree/order - no
            # retry with more robots: clearance simply fails.
            return None, False, None, None


        # ============================================================
        # ROBOT STATE
        # ============================================================

        # Robots may physically start somewhere other than the tree's own
        # root, and not even all at the same place/time (see startState
        # doc above) - the LBAP naturally routes each of them via the
        # shortest path from wherever it actually is to wherever it is
        # first needed, task 0 (root itself) included.
        robots = [
            {
                "id": robotId,

                # Current physical position
                "node": startState[robotId][0],

                # Earliest time at which this robot may depart
                "freeFrom": startState[robotId][1],

                # None => currently free
                # list  => guarding, still waiting on these strategy
                #          indices (the guard's full dependency set)
                "guardUntil": None,

                # Time at which this robot started its current guard
                # duty (its own physical arrival there) - a guard can
                # never be released before that, no matter how fast
                # the task(s) satisfying its dependency finish.
                "guardSince": 0,

                # Remaining dependency indices not yet reached, and the
                # latest execution time seen among the ones that have.
                "guardPending": set(),
                "guardMaxDepTime": 0
            }
            for robotId in range(availableRobots)
        ]


        # Fast access by ID
        robotById = {
            robot["id"]: robot
            for robot in robots
        }


        # ============================================================
        # EXECUTION
        # ============================================================

        executionPlan = []

        # Real execution time of every task, filled in as tasks run (in
        # strict strategy-index order) - needed so a guard's dependency on
        # an earlier-indexed (already executed) task can be resolved
        # immediately against its real arrival time, instead of only ever
        # waiting for later-indexed tasks via the broadcast below.
        taskExecutionTime = {}

        for level in levels:

            # --------------------------------------------------------
            # Solve LBAP using robots currently free
            # --------------------------------------------------------

            assignment, bottleneck = LBAPSolver(level, robots)

            # --------------------------------------------------------
            # Compute movement information BEFORE modifying robots
            # --------------------------------------------------------

            movementInfo = {}

            for taskIndex, node, releaseIndex in level:

                robotId = assignment[taskIndex]
                robot = robotById[robotId]

                source = robot["node"]

                departure = robot["freeFrom"]

                physicalArrival = (
                    departure
                    + flagDistance[source][node]
                )

                movementInfo[taskIndex] = {
                    "robot": robotId,
                    "source": source,
                    "departure": departure,
                    "arrival": physicalArrival
                }

            # --------------------------------------------------------
            # Execute tasks in SEARCH STRATEGY order
            # --------------------------------------------------------

            for taskIndex, node, releaseIndex in level:

                info = movementInfo[taskIndex]

                robotId = info["robot"]
                robot = robotById[robotId]

                # Levels only decide which robot is assigned to which
                # node - each task completes as soon as its own robot
                # physically gets there, independent of other tasks.
                executionTime = info["arrival"]

                # taskIndex 0 is always the root itself: the assigned
                # robot is already standing there at t=0, so there is
                # no actual movement to record.
                if info["source"] != node:

                    executionPlan.append((
                        info["source"],
                        node,
                        1,
                        info["departure"],
                        executionTime
                    ))

                # Robot is physically located at its assigned node.
                robot["node"] = node

                # This task's own real execution time is now fixed - make
                # it available for any later dependency lookup.
                taskExecutionTime[taskIndex] = executionTime

                # ----------------------------------------------------
                # Current robot becomes guard or immediately free
                #
                # releaseIndex may mix dependencies whose real time is
                # already known (their taskIndex already executed -
                # possibly even taskIndex itself, or an
                # earlier-in-strategy-order neighbour whose LBAP
                # assignment nonetheless made it arrive later than this
                # node) with dependencies still ahead (resolved later via
                # the broadcast below).
                # ----------------------------------------------------

                knownDepTimes = [
                    taskExecutionTime[dep]
                    for dep in releaseIndex
                    if dep in taskExecutionTime
                ]
                pendingDeps = [
                    dep for dep in releaseIndex
                    if dep not in taskExecutionTime
                ]
                knownMaxDepTime = max(knownDepTimes) if knownDepTimes else 0

                if not pendingDeps:

                    robot["guardUntil"] = None
                    robot["freeFrom"] = max(executionTime, knownMaxDepTime)

                else:

                    robot["guardUntil"] = pendingDeps
                    robot["guardSince"] = executionTime
                    robot["guardPending"] = set(pendingDeps)
                    robot["guardMaxDepTime"] = knownMaxDepTime

                    # Exact release time is not yet known.
                    # It will be set once every remaining dependency executes.
                    robot["freeFrom"] = None


                # ----------------------------------------------------
                # Reaching taskIndex may satisfy one dependency of
                # OTHER guards - a guard is only actually released once
                # ALL of its dependencies have fired, using the LATEST
                # of their real completion times (never just whichever
                # one happens to be the highest strategy index).
                # ----------------------------------------------------

                for otherRobot in robots:

                    if taskIndex in otherRobot["guardPending"]:

                        otherRobot["guardPending"].discard(taskIndex)

                        otherRobot["guardMaxDepTime"] = max(
                            otherRobot["guardMaxDepTime"],
                            executionTime
                        )

                        if not otherRobot["guardPending"]:

                            otherRobot["guardUntil"] = None

                            # The guard can't leave before it itself
                            # got there, even if every dependency it
                            # was waiting on cleared faster elsewhere.
                            otherRobot["freeFrom"] = max(
                                otherRobot["guardSince"],
                                otherRobot["guardMaxDepTime"]
                            )


        if usingDefaultStart:
            executionPlan.insert(0, (None, root, availableRobots))

        # ============================================================
        # VISITED TIMES
        # Read directly off taskExecutionTime/strategy (every task index
        # has both, always) rather than off executionPlan's move entries -
        # task 0 (root) has no move recorded whenever its assigned robot
        # already happens to start exactly there, which - with a
        # heterogeneous startState - can happen at a nonzero freeFrom time,
        # not necessarily t=0.
        # ============================================================

        visitedTimes = [-1] * len(T.nodes)
        for taskIndex, (node, _) in enumerate(strategy):
            visitedTimes[node] = taskExecutionTime[taskIndex]

        finalRobotStates = [(robot["node"], robot["freeFrom"]) for robot in robots]

        return executionPlan, True, visitedTimes, finalRobotStates




    # ---------------------------------------------------
    # COMPUTING A SPANNING TREE WITH DFS
    # The generation uses a greedy choice for the node with the highest prior
    # ---------------------------------------------------

    def computeGreedySpanningTree(G : Graph, root):
        
        T = Graph()
        for i in range(len(G.nodes)):
            T.add_node(G.nodes[i].idx, G.nodes[i].pos, G.nodes[i].prior)
        
        visited = set()

        def dfs(node):

            visited.add(node)

            neighbours = list(G.adj[G.nodes[node]])

            # Sorting the neighbours in respect to their prior
            neighbours.sort(key=lambda x: x.prior, reverse = True)
            
            for neighbour in neighbours:

                if neighbour.idx in visited:
                    continue

                T.add_edge(T.nodes[node], T.nodes[neighbour.idx], G.edges[(node,neighbour.idx)].time, 2)

                dfs(neighbour.idx)

        dfs(root)
        return T
    
    # ---------------------------------------------------
    # COMPUTING A RANDOM SPANNING TREE WITH DFS
    # So that each iteration considers a new spanning tree
    # ---------------------------------------------------
    
    def computeRandomSpanningTree(G : Graph, root, preferredEdges=None):

        # preferredEdges: an optional set of directed (u,v) edge keys that should
        # always be picked over a non-preferred edge whenever both are available
        # at a given step of the DFS (used to favour edges shared by both parents)
        if preferredEdges is None:
            preferredEdges = set()

        T = Graph()
        for i in range(len(G.nodes)):
            T.add_node2(G.nodes[i])

        visited = set()

        def dfs(node):

            visited.add(node)

            neighbours = list(G.adj[G.nodes[node]])

            preferred = [n for n in neighbours if (node, n.idx) in preferredEdges]
            others = [n for n in neighbours if (node, n.idx) not in preferredEdges]
            random.shuffle(preferred)
            random.shuffle(others)
            neighbours = preferred + others

            for neighbour in neighbours:

                if neighbour.idx in visited:
                    continue

                T.add_edge(T.nodes[node], T.nodes[neighbour.idx], G.edges[(node,neighbour.idx)].time, G.edges[(node,neighbour.idx)].robotType)

                dfs(neighbour.idx)

        dfs(root)
        return T
    

    # ---------------------------------------------------
    # COMPUTING THE CLOSING EXITS STRATEGY IF NOT ENOUGH ROBOTS ARE GIVEN
    # Dedupe/score/filter the local clearance candidates gathered while
    # searching (see computeLocalClearanceCandidates), commit robots to
    # whichever succeed best-first, and search anything left over with
    # FHPE_SA on the remaining time/robots/graph.
    # ---------------------------------------------------

    def computeClosingExits(root, localCandidates, G):

        # ------------------------------------------------
        # 1) DEDUPLICATE
        # Two candidates covering the exact same node set (found via
        # different failing trees) are the same local clearance option -
        # keep only the first occurrence.
        # ------------------------------------------------
        seenNodeSets = set()
        dedupedCandidates = []
        for candidate in localCandidates:
            key = frozenset(candidate["nodes"])
            if key in seenNodeSets:
                continue
            seenNodeSets.add(key)
            dedupedCandidates.append(candidate)

        if not dedupedCandidates:
            return FHPE_SA(root, G)

        # ------------------------------------------------
        # 2) GLOBAL BASELINE RATIO
        # Expected prior mass collectable over the whole remaining budget
        # (probBudget) when searching at the graph's average efficiency
        # (average prior per node / average travel time per edge) with
        # ALL availableRobots searching in parallel - matching the same
        # robot-count factor the per-candidate score below applies via
        # sparRobots, so the two stay on a comparable footing.
        # ------------------------------------------------
        totalPriorAll = sum(node.prior for node in G.nodes)
        avgPriorAll = totalPriorAll / len(G.nodes)
        allEdgeTimes = [edge.time for edge in G.edges.values()]
        avgEdgeTimeAll = sum(allEdgeTimes) / len(allEdgeTimes)

        globalRatio = probBudget * availableRobots * avgPriorAll / avgEdgeTimeAll 

        # ------------------------------------------------
        # 3) PER-CANDIDATE SCORING
        # ------------------------------------------------
        EPS = 1e-9

        for candidate in dedupedCandidates:

            candidateNodes = candidate["nodes"]
            localMass = candidate["totalPrior"]
            clearanceTime = candidate["totalEdgeTime"]

            # Prior redistribution: once this local area is handled
            # separately, its prior mass is already accounted for - the
            # remaining nodes' priors are proportionally renormalised so
            # they still sum to the original total (a "given this area is
            # already cleared" conditional redistribution).
            removedFraction = localMass / totalPriorAll if totalPriorAll > 0 else 0
            renormFactor = 1 / max(EPS, 1 - removedFraction)

            remainingNodes = [node for node in G.nodes if node.idx not in candidateNodes]

            avgRedistributedPrior = (
                sum(node.prior * renormFactor for node in remainingNodes) / len(remainingNodes)
                if remainingNodes else 0
            )

            # Average travel time recomputed on the remaining region only
            # (edges whose both endpoints lie outside this candidate) -
            # falls back to the global average on the degenerate case of
            # no such edge existing at all.
            remainingEdgeTimes = [
                edge.time
                for (u, v), edge in G.edges.items()
                if u not in candidateNodes and v not in candidateNodes
            ]
            avgEdgeTimeRemaining = (
                sum(remainingEdgeTimes) / len(remainingEdgeTimes)
                if remainingEdgeTimes else avgEdgeTimeAll
            )

            redistributedRatio = avgRedistributedPrior / avgEdgeTimeRemaining
            sparRobots = availableRobots - candidate["opponents"]

            score = (
                localMass
                + (probBudget - clearanceTime) * redistributedRatio * sparRobots + (probBudget - clearanceTime) * localMass * (1 - np.exp(-1 / lam))
            )

            candidate["globalBaselineRatio"] = globalRatio
            candidate["redistributedRatio"] = redistributedRatio
            candidate["ratio"] = score

        dedupedCandidates.sort(key=lambda candidate: candidate["ratio"], reverse=True)
        # ------------------------------------------------
        # 4) DROP CANDIDATES WORSE THAN THE BASELINE
        # A candidate that would not even keep pace with plain
        # average-efficiency search over the whole remaining budget isn't
        # worth carving out as a dedicated local clearance.
        # ------------------------------------------------
        feasibleCandidates = [
            candidate for candidate in dedupedCandidates
            if candidate["ratio"] >= globalRatio
        ]

        if not feasibleCandidates:
            return FHPE_SA(root, G)

        # ------------------------------------------------
        # Helpers for the consumption loop below
        # ------------------------------------------------

        def buildInducedSubgraph(nodeSet):
            # Same node array as G (so idx-based indexing stays valid),
            # but only the edges whose both endpoints lie inside nodeSet -
            # every node outside it ends up isolated, which is exactly
            # what's needed both for a local candidate's own subgraph and
            # for the "rest of the graph" handed to FHPE_SA afterwards.
            localG = Graph()
            for node in G.nodes:
                localG.add_node2(node)
            for (u, v), edge in G.edges.items():
                if u in nodeSet and v in nodeSet:
                    localG.add_edge(localG.nodes[u], localG.nodes[v], edge.time, edge.robotType)
            return localG

        def computeLocalExpTime(visitedTimes, nodeSet):
            # Same idea as computeExpTime, restricted to nodeSet - used
            # only to rank the (up to) 10 random local spanning trees
            # tried per candidate against each other, so nodes outside
            # nodeSet (which stay at visitedTimes == -1, never actually
            # reached by this local tree) must not be included.
            eff = 0
            foundPriors = np.copy(cellpriors)
            orderedNodes = sorted(nodeSet, key=lambda idx: visitedTimes[idx])
            for idx in orderedNodes:
                for cell in D[idx]:
                    eff += foundPriors[cell[0], cell[1]] * visitedTimes[idx]
                    foundPriors[cell[0], cell[1]] = 0
            return eff

        # ------------------------------------------------
        # 5) CONSUME CANDIDATES BEST-FIRST
        # A pool of (node, freeFromTime) pairs tracks the REAL current
        # state of every still-mobile robot - after each successfully
        # cleared candidate its robots are scattered wherever their own
        # local plan left them, not back at root, so every subsequent
        # candidate (and the final FHPE_SA continuation) must route from
        # there, never from a fresh, phantom "root" respawn.
        # ------------------------------------------------
        overallStrategy = [(None, root, availableRobots)]
        overallVisitedTime = {}

        robotPool = [(root, 0)] * availableRobots
        guardedOpponents = set()
        clearedNodes = set()

        def assignNearestRobot(targetNode, pool):
            # Greedy nearest-available match: pops whichever pooled robot
            # reaches targetNode soonest and returns its dispatch move.
            bestIdx, bestArrival = None, None
            for i, (node, freeTime) in enumerate(pool):
                arrival = freeTime + flagDistance[node][targetNode]
                if bestArrival is None or arrival < bestArrival:
                    bestArrival = arrival
                    bestIdx = i
            sourceNode, sourceFree = pool.pop(bestIdx)
            return sourceNode, sourceFree, bestArrival

        pending = feasibleCandidates

        while pending:

            candidate = pending.pop(0)

            # Opponents already guarded because some other, already
            # cleared candidate happens to border the same node don't need
            # a second dedicated guard.
            newOpponents = candidate["opponentNodes"] - guardedOpponents
            neededGuards = len(newOpponents)

            if candidate["bLabel"] + neededGuards > len(robotPool):
                continue

            # Work on a scratch copy of the pool so a candidate that ends
            # up failing all 10 local attempts leaves robotPool untouched -
            # those robots were never actually dispatched anywhere.
            scratchPool = list(robotPool)
            guardMoves = []
            guardVisited = {}
            for opponentNode in newOpponents:
                sourceNode, sourceFree, arrival = assignNearestRobot(opponentNode, scratchPool)
                guardMoves.append((sourceNode, opponentNode, 1, sourceFree, arrival))
                guardVisited[opponentNode] = arrival

            localRobots = len(scratchPool)
            localGraph = buildInducedSubgraph(candidate["nodes"])

            bestLocal = None
            for _ in range(10):
                localTree = computeRandomSpanningTree(localGraph, candidate["root"])
                localStrategy, localClearance, localVisitedTimes, localFinalStates = treeSearch(
                    localTree, candidate["root"], localRobots, localGraph,
                    startState=scratchPool, recordCandidates=False
                )
                if not localClearance:
                    continue
                localFitness = computeLocalExpTime(localVisitedTimes, candidate["nodes"])
                if bestLocal is None or localFitness < bestLocal[0]:
                    bestLocal = (localFitness, localStrategy, localVisitedTimes, localFinalStates)

            if bestLocal is None:
                # All 10 random local trees failed with this robot budget -
                # more robots won't free up later (the pool only shrinks),
                # so this candidate is permanently unworkable. scratchPool
                # (and its speculative guard picks) is simply discarded.
                continue

            _, chosenStrategy, chosenVisitedTimes, chosenFinalStates = bestLocal

            # chosenStrategy has no sentinel of its own (startState was
            # given explicitly) - append its moves and the guard dispatch
            # moves directly.
            overallStrategy.extend(guardMoves)
            overallStrategy.extend(chosenStrategy)
            overallVisitedTime.update(guardVisited)
            for node in candidate["nodes"]:
                overallVisitedTime[node] = chosenVisitedTimes[node]

            # Opponent guards must stay put permanently - they are the
            # only thing preventing recontamination of this now-cleared
            # region from outside, so they never rejoin the mobile pool.
            # The robots that did the INTERNAL clearing are a different
            # matter: treeSearch already resolves every one of their guard
            # duties by the time the local region is fully traversed (its
            # returned finalRobotStates always has a real freeFrom, never
            # a still-pending guard), so they are correctly free to help
            # with FHPE_SA right away.
            robotPool = list(chosenFinalStates)
            guardedOpponents |= newOpponents
            clearedNodes |= candidate["nodes"]

            # Drop every remaining candidate that shares so much as one
            # node with a region already cleared - candidates come from
            # many different (and possibly nested/overlapping) failed
            # trees, so a still-pending candidate can easily be a subtree
            # already contained in, or overlapping, what was just cleared.
            # Attempting it anyway would re-clear already-safe nodes and,
            # worse, dispatch a fresh permanent guard for one of its
            # "opponents" (computed against the ORIGINAL tree) that is now
            # actually sitting safely inside the just-cleared region -
            # exactly the kind of guard that then looks permanently stuck
            # in the middle of already-cleared territory.
            # Order by ratio is unaffected by dropping entries, so no
            # re-sort is needed.
            coveredNodes = clearedNodes | guardedOpponents
            pending = [
                c for c in pending
                if c["nodes"].isdisjoint(coveredNodes)
                and c["bLabel"] + len(c["opponentNodes"] - guardedOpponents) <= len(robotPool)
            ]

        # ------------------------------------------------
        # 6) MOVE MOBILE ROBOTS OFF EVERY NODE FHPE_SA WON'T SEARCH ON
        # A mobile robot can be physically standing anywhere inside a
        # just-cleared region (that's simply where its own local clearance
        # left it) - FHPE_SA must not be tempted to wander back in there,
        # so each such robot is routed - shortest path - to the nearest
        # node that will actually still be part of FHPE_SA's graph, before
        # that graph is built below. A guarded opponent node also isn't a
        # valid destination even though it stays reachable for OTHERS
        # (kept as a pass-through when it's a cut vertex): a guard is
        # already there, and landing a mobile robot on a node that might
        # end up dropped from the graph entirely would strand it exactly
        # like standing in the cleared region would.
        # ------------------------------------------------
        avoidNodes = clearedNodes | guardedOpponents

        def relocateOutOfClearedRegion(pool):
            if not avoidNodes:
                return pool

            relocated = []
            for node, freeTime in pool:

                if node not in avoidNodes:
                    relocated.append((node, freeTime))
                    continue

                nearestNode, nearestDist = None, None
                for candidateNode in range(len(G.nodes)):
                    if candidateNode in avoidNodes:
                        continue
                    dist = flagDistance[node][candidateNode]
                    if nearestDist is None or dist < nearestDist:
                        nearestDist = dist
                        nearestNode = candidateNode

                if nearestNode is None:
                    # Nowhere left outside cleared/guarded nodes to send
                    # this robot, so it just stays put.
                    relocated.append((node, freeTime))
                    continue

                arrival = freeTime + nearestDist
                overallStrategy.append((node, nearestNode, 1, freeTime, arrival))
                if nearestNode not in overallVisitedTime or arrival < overallVisitedTime[nearestNode]:
                    overallVisitedTime[nearestNode] = arrival
                relocated.append((nearestNode, arrival))

            return relocated

        robotPool = relocateOutOfClearedRegion(robotPool)

        # ------------------------------------------------
        # 7) SEARCH WHATEVER IS LEFT WITH FHPE_SA
        # Every permanently guarded opponent node is dropped from the
        # graph unconditionally now, even one that is the only link
        # between two otherwise separate areas - a guard is already
        # stationed there, so FHPE_SA gains nothing by stopping there
        # again, and letting it "bridge" the gap just means every robot
        # keeps treating both sides as one shared pool instead of properly
        # splitting up to cover the actual disconnected territories.
        # ------------------------------------------------
        baseRemainingNodes = {node.idx for node in G.nodes} - clearedNodes
        remainingNodeIndices = baseRemainingNodes - guardedOpponents
        remainingGraph = buildInducedSubgraph(remainingNodeIndices)

        # ------------------------------------------------
        # 8) SPLIT ACROSS WHATEVER SEPARATE AREAS THIS CREATES
        # Dropping every guard can leave remainingGraph as several mutually
        # unreachable components (not just two) - each is searched by its
        # own share of the mobile robots, apportioned by that component's
        # share of the remaining prior mass (10 robots, two components each
        # holding 50% of the prior -> 5 each), using largest-remainder
        # rounding so the shares always add back up to exactly len(robotPool)
        # robots - never fewer (lost) or more (invented).
        # ------------------------------------------------
        def connectedComponents(nodeSet, graph):
            seen = set()
            comps = []
            for start in nodeSet:
                if start in seen:
                    continue
                comp = {start}
                stack = [start]
                while stack:
                    cur = stack.pop()
                    for neighbour in graph.adj[graph.nodes[cur]]:
                        if neighbour.idx in nodeSet and neighbour.idx not in comp:
                            comp.add(neighbour.idx)
                            stack.append(neighbour.idx)
                comps.append(comp)
                seen |= comp
            return comps

        def apportionByLargestRemainder(total, weights):
            # Guarantees sum(result) == total exactly, regardless of
            # floating-point rounding - the plain share every component is
            # entitled to (its floor) plus, one each, to whichever
            # components have the largest leftover fraction, until the
            # rounding gap to `total` is used up.
            totalWeight = sum(weights)
            if totalWeight <= 0:
                base, extra = divmod(total, len(weights))
                return [base + (1 if i < extra else 0) for i in range(len(weights))]

            exact = [total * w / totalWeight for w in weights]
            counts = [int(math.floor(x)) for x in exact]
            shortfall = total - sum(counts)
            order = sorted(range(len(weights)), key=lambda i: exact[i] - counts[i], reverse=True)
            for i in range(shortfall):
                counts[order[i]] += 1
            return counts

        if robotPool:

            components = connectedComponents(remainingNodeIndices, remainingGraph)

            if len(components) <= 1:
                fhpeExecutionPlan, _, _, _ = FHPE_SA(
                    root, remainingGraph,
                    startState=robotPool,
                    seedLastVisitTime=overallVisitedTime,
                )
                overallStrategy.extend(fhpeExecutionPlan)

            else:

                componentPriors = [
                    sum(G.nodes[i].prior for i in comp)
                    for comp in components
                ]
                targetCounts = apportionByLargestRemainder(len(robotPool), componentPriors)

                # Greedy nearest-first (robot, component) matching: sum of
                # targetCounts always equals len(robotPool), so every robot
                # is guaranteed to end up assigned somewhere, even though
                # this isn't a globally-optimal (min-total-travel) matching.
                costs = []
                for robotIdx, (node, _) in enumerate(robotPool):
                    for compIdx, comp in enumerate(components):
                        if targetCounts[compIdx] == 0:
                            continue
                        nearestDist = min(flagDistance[node][n] for n in comp)
                        costs.append((nearestDist, robotIdx, compIdx))
                costs.sort(key=lambda c: c[0])

                assignedComponent = [None] * len(robotPool)
                remainingCapacity = list(targetCounts)
                assignedCount = 0
                for dist, robotIdx, compIdx in costs:
                    if assignedComponent[robotIdx] is not None or remainingCapacity[compIdx] == 0:
                        continue
                    assignedComponent[robotIdx] = compIdx
                    remainingCapacity[compIdx] -= 1
                    assignedCount += 1
                    if assignedCount == len(robotPool):
                        break

                # Relocate every robot - shortest path - to the nearest
                # node inside whichever component it was actually assigned
                # to (a no-op if it already happens to stand there).
                splitPool = [[] for _ in components]
                for robotIdx, (node, freeTime) in enumerate(robotPool):
                    comp = components[assignedComponent[robotIdx]]

                    if node in comp:
                        splitPool[assignedComponent[robotIdx]].append((node, freeTime))
                        continue

                    nearestNode, nearestDist = None, None
                    for candidateNode in comp:
                        dist = flagDistance[node][candidateNode]
                        if nearestDist is None or dist < nearestDist:
                            nearestDist = dist
                            nearestNode = candidateNode

                    arrival = freeTime + nearestDist
                    overallStrategy.append((node, nearestNode, 1, freeTime, arrival))
                    if nearestNode not in overallVisitedTime or arrival < overallVisitedTime[nearestNode]:
                        overallVisitedTime[nearestNode] = arrival
                    splitPool[assignedComponent[robotIdx]].append((nearestNode, arrival))

                # One shared FHPE_SA call over the whole (multi-component)
                # remainingGraph: since the components have no edges
                # between them at all, each robot's own search naturally
                # stays confined to whichever component it was placed in -
                # no need to invoke FHPE_SA separately per component.
                combinedPool = [state for pool in splitPool for state in pool]

                fhpeExecutionPlan, _, _, _ = FHPE_SA(
                    root, remainingGraph,
                    startState=combinedPool,
                    seedLastVisitTime=overallVisitedTime,
                )
                overallStrategy.extend(fhpeExecutionPlan)

        return overallStrategy, None, len(feasibleCandidates), None


    # ---------------------------------------------------
    # FINITE HORIZON PATH ENUMERATION WITH SEQUENTIAL ALLOCATION (FHPE_SA)
    #
    # F, per candidate path, sums one term per node w in G:
    #   w reached within the horizon (by this path or an earlier searcher
    #   this round) at the shortest such time t:   priorAt(w, t) * t
    #   w not reached at all:                       priorAt(w, deadline) * deadline
    # priorAt(w, t) decays w's static prior by how long it has been since
    # w was actually last visited (lastVisitTime, updated only once a
    # path is chosen - never during the search itself), so a node found
    # long ago is worth almost as much as an unfound one again.
    # ---------------------------------------------------

    def bestPathForSearcher(startNode, startTime, horizonVisited, lastVisitTime, horizon, lam, G : Graph, maxHops):

        # horizon is now the max number of EDGES a candidate path may chain
        # (not a time budget) - "not reached at all" therefore no longer has
        # a horizon-relative time to compare candidates against, since two
        # candidate paths of the same hop count can finish at very
        # different real times once edges vary in travel time. The
        # absolute mission deadline (probBudget) is the only genuine,
        # path-independent reference time left, so it takes over as the
        # common comparison point every candidate path is scored against.
        deadline = probBudget + 1

        #Calculating the Prior at a node with respect to the recency bias if a node was already visited
        def priorAt(node, t):
            prior = G.nodes[node].prior
            if node not in lastVisitTime:
                return prior
            delta = max(0, t - lastVisitTime[node])
            return prior * (1 - math.exp(-delta / lam))

        def objective(path, times):
            merged = dict(horizonVisited)
            for node, t in zip(path, times):
                if node not in merged or t < merged[node]:
                    merged[node] = t

            F = 0
            for node in G.nodes:
                t = merged.get(node.idx, deadline)
                F += priorAt(node.idx, t) * t
            return F

        # Staying put is never a candidate: with every prior > 0, moving to
        # any reachable neighbour strictly lowers F (its contribution goes
        # from priorAt(w, deadline) * deadline down to priorAt(w, t) * t,
        # t < deadline). Only exception is a genuine dead end (no
        # neighbour reachable within horizon/probBudget/maxHops at all).
        best = {"F": None, "path": None, "times": None}

        # horizon directly caps how many edges a candidate path may chain;
        # maxHops remains an independent, additional tractability cap on
        # top of it (whichever of the two is smaller effectively wins) -
        # brute-forcing every path only stays tractable with a hard limit
        # on branching depth regardless of how horizon itself is set.
        def visit(node, t, path, times, hopsLeft):
            if hopsLeft <= 0:
                return
            for neighbour in G.adj[G.nodes[node]]:
                newT = t + G.edges[(node, neighbour.idx)].time
                if newT > probBudget:
                    continue
                path.append(neighbour.idx)
                times.append(newT)

                F = objective(path, times)
                # Equal F: prefer whichever finishes sooner. Still tied
                # (same F, same finish time): keep the current best.
                if best["F"] is None or F < best["F"] or (F == best["F"] and newT < best["times"][-1]):
                    best["F"] = F
                    best["path"] = list(path)
                    best["times"] = list(times)

                visit(neighbour.idx, newT, path, times, hopsLeft - 1)

                path.pop()
                times.pop()

        visit(startNode, startTime, [startNode], [startTime], min(horizon, maxHops))

        if best["path"] is None:
            return [startNode], [startTime]

        return best["path"], best["times"]


    def FHPE_SA(root, G : Graph, startState=None, seedLastVisitTime=None):
        # startState: per-robot [(node, freeFromTime), ...], if robots
        #   don't all simply start together at root at t=0 - e.g.
        #   continuing after computeClosingExits has already run some
        #   robots through local clearances and/or committed others as
        #   permanent opponent guards, leaving the rest scattered at
        #   various nodes/times instead of freshly spawned at root.
        #   probBudget stays the same absolute deadline throughout, so a
        #   robot whose freeFromTime is already > 0 naturally gets exactly
        #   (probBudget - freeFromTime) of search time, no separate budget
        #   parameter needed. None (default): availableRobots robots all
        #   start at root at t=0, exactly as before, sentinel included.
        # seedLastVisitTime: lastVisitTime entries for nodes already
        #   handled before this call (cleared locally or under permanent
        #   guard) so their decayed prior correctly reflects that instead
        #   of looking completely unvisited.

        usingDefaultStart = startState is None
        if usingDefaultStart:
            startState = [(root, 0)] * availableRobots
        robots_ = len(startState)


        lastVisitTime = {root: 0, **(seedLastVisitTime or {})} #Format Node: lastVisitTime
        executionPlan = [(None, root, robots_)] if usingDefaultStart else []

        robotNode = [node for node, _ in startState] #Position of where the nodes are after the horizon iteration
        robotTime = [t for _, t in startState] #Time when horizon iteration ends

        roundCounter = 0

        # Robots rarely finish their planned horizon path at the same
        # time. Cuts a robot's still-running path down to what actually
        # gets executed before time T: every move fully done by T is kept
        # as is, the one move straddling T is finished (a robot can't be
        # pulled off mid-edge), and everything planned after that is
        # dropped.
        def truncateToTime(path, times, T):
            k = 0
            while k < len(times) and times[k] <= T:
                k += 1
            if k == len(times):
                return path, times
            if k == 0:
                return path[:1], times[:1]
            if times[k - 1] == T:
                return path[:k], times[:k]
            return path[:k + 1], times[:k + 1]

        while min(robotTime) < probBudget: #As long as there is time

            horizonVisited = {} #Format Node: horizonVisitedTime
            roundPaths = [] #Every robot's full planned (path, times) this horizon iteration

            for k in range(robots_):

                path, times = bestPathForSearcher(
                    robotNode[k], robotTime[k], horizonVisited, lastVisitTime, horizon, lam, G, maxHops
                )

                for node, t in zip(path, times):
                    if node not in horizonVisited or t < horizonVisited[node]: #New Node or visited but earlier
                        horizonVisited[node] = t

                roundPaths.append((path, times))

            # As soon as the first (shortest) planned path finishes,
            # everyone replans - so this round only actually runs until
            # that earliest finish time T. Robots stuck at a genuine dead
            # end (no move fits within horizon/probBudget at all) don't
            # count towards T; if every robot is stuck, none of them will
            # ever move again, so they are done for good.
            nonTrivialFinishTimes = [times[-1] for path, times in roundPaths if len(path) > 1]

            if not nonTrivialFinishTimes:
                for k in range(robots_):
                    robotTime[k] = probBudget
                roundCounter += 1
                continue

            T = min(nonTrivialFinishTimes)

            for k in range(robots_):

                path, times = roundPaths[k]

                if len(path) == 1:
                    robotTime[k] = probBudget
                    continue

                execPath, execTimes = truncateToTime(path, times, T)

                for i in range(1, len(execPath)):
                    executionPlan.append((execPath[i - 1], execPath[i], 1, execTimes[i - 1], execTimes[i]))

                # Adjust the lastVisitTime only for the part of the path
                # that is actually executed - a node planned but never
                # reached (cut off by the truncation above) must not have
                # its lastVisitTime touched.
                for node, t in zip(execPath, execTimes):
                    if node not in lastVisitTime or t > lastVisitTime[node]: # New Node or new last visit
                        lastVisitTime[node] = t

                robotNode[k] = execPath[-1]
                robotTime[k] = execTimes[-1]

            roundCounter += 1

        # FHPE_SA's moves are a patrol path (revisits, cycles) rather than
        # a tree, so - unlike the tree-search paths above - there is no
        # meaningful "spanning tree" to hand the visualization here.
        return executionPlan, None, roundCounter, None


    # ---------------------------------------------------
    # COMPUTING EFFICENCY WITH WHICH TWO STRATEGYS ARE COMPARED
    # ---------------------------------------------------
    def computeExpTime(visitedTimes, G: Graph):
        #Computing the expected search time
        eff = 0
        foundPriors = np.copy(cellpriors)
        sortedNodes = sorted(G.nodes, key=lambda node: visitedTimes[node.idx])
        for node in sortedNodes:
            for cell in D[node.idx]:
                eff += foundPriors[cell[0], cell[1]] * visitedTimes[node.idx]
                foundPriors[cell[0], cell[1]] = 0

        return eff



    # ---------------------------------------------------
    # Splicing a mutation edge (contained in neither parent) into an
    # already complete spanning tree while keeping it a valid tree:
    # the edge on the existing path between u and v is removed and
    # replaced by the new (u,v) edge, oriented away from the root
    # ---------------------------------------------------
    def applyMutation(tree, root, u, v, G):

        undirectedAdj = defaultdict(set)
        for a, b in tree.edges.keys():
            undirectedAdj[a].add(b)
            undirectedAdj[b].add(a)

        # BFS from the root to know each node's parent in the current tree
        parent = {root: None}
        frontier = [root]
        while frontier:
            nextFrontier = []
            for node in frontier:
                for neighbour in undirectedAdj[node]:
                    if neighbour not in parent:
                        parent[neighbour] = node
                        nextFrontier.append(neighbour)
            frontier = nextFrontier

        # Walk from u and v up to their lowest common ancestor to get the
        # path between them
        def ancestors(node):
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return path

        ancestorsUSet = set(ancestors(u))
        lca = next(node for node in ancestors(v) if node in ancestorsUSet)

        pathEdges = []
        node = u
        while node != lca:
            pathEdges.append((parent[node], node))
            node = parent[node]
        node = v
        while node != lca:
            pathEdges.append((parent[node], node))
            node = parent[node]

        if not pathEdges:
            return

        # Removing one edge on the u-v path splits the tree into a part
        # containing u and a part containing v
        removedParent, removedChild = random.choice(pathEdges)

        subtree = {removedChild}
        stack = [removedChild]
        while stack:
            node = stack.pop()
            for neighbour in undirectedAdj[node]:
                if neighbour != removedParent and neighbour not in subtree:
                    subtree.add(neighbour)
                    stack.append(neighbour)

        rootSide, childSide = (v, u) if u in subtree else (u, v)

        # childSide is not necessarily removedChild itself but possibly one
        # of its descendants - every edge on the old path from removedChild
        # down to childSide has to flip direction so the reattached subtree
        # stays oriented away from the (unchanged) overall root
        reorientPath = []
        node = childSide
        while node != removedChild:
            reorientPath.append((parent[node], node))
            node = parent[node]

        for a, b in reorientPath:
            del tree.edges[(a, b)]
            tree.adj[tree.nodes[a]].remove(tree.nodes[b])
            edgeInfo = G.edges[(b, a)] if (b, a) in G.edges else G.edges[(a, b)]
            tree.add_edge(tree.nodes[b], tree.nodes[a], edgeInfo.time, edgeInfo.robotType)

        del tree.edges[(removedParent, removedChild)]
        tree.adj[tree.nodes[removedParent]].remove(tree.nodes[removedChild])

        newEdge = G.edges[(rootSide, childSide)] if (rootSide, childSide) in G.edges else G.edges[(childSide, rootSide)]
        tree.add_edge(tree.nodes[rootSide], tree.nodes[childSide], newEdge.time, newEdge.robotType)


    # ---------------------------------------------------
    # Computing the cross over of two trees so for the Genetic Algorithim
    # ---------------------------------------------------
    def crossOver(parentA, parentB, G):

        # parent = (Tree, root)
        treeA, rootA = parentA
        treeB, rootB = parentB


        # 1. Choose root from one of the parents
        root = random.choice([rootA, rootB])

        # 2. Union of edges from both parents, ignoring direction: for every
        #    A-B edge of a parent, the B-A edge is added as well
        def undirectedEdgeSet(tree):
            edges = set()
            for u, v in tree.edges.keys():
                edges.add((u, v))
                edges.add((v, u))
            return edges

        undirectedA = undirectedEdgeSet(treeA)
        undirectedB = undirectedEdgeSet(treeB)
        parentEdges = undirectedA | undirectedB

        # Edges contained in both parents (direction-independent) are always
        # preferred when building the child's spanning tree
        commonEdges = undirectedA & undirectedB

        # 3. Build temporary graph containing exactly the parents' edges
        candidateGraph = Graph()

        for node in G.nodes:
            candidateGraph.add_node2(node)

        for u, v in parentEdges:

            # Get original edge information from G, falling back to the
            # opposite direction in case G does not have this exact one
            edge = G.edges[(u, v)] if (u, v) in G.edges else G.edges[(v, u)]

            candidateGraph.add_edge(
                candidateGraph.nodes[u],
                candidateGraph.nodes[v],
                edge.time,
                edge.robotType
            )

        # 4. Generate a spanning tree from the candidate graph, always
        #    preferring edges that are contained in both parents
        childTree = computeRandomSpanningTree(
            candidateGraph,
            root,
            preferredEdges=commonEdges
        )

        # 5. Mutation: only once the child's spanning tree is complete,
        #    splice in up to two edges that are contained in neither parent
        possibleRandomEdges = [
            edgeKey
            for edgeKey in G.edges.keys()
            if edgeKey not in parentEdges
        ]

        numberOfRandomEdges = min(2, len(possibleRandomEdges))

        randomEdges = random.sample(
            possibleRandomEdges,
            numberOfRandomEdges
        )

        for u, v in randomEdges:
            applyMutation(childTree, root, u, v, G)

        return childTree, root


    # ---------------------------------------------------
    # Weighted selection of two distinct parents for the Genetic Algorithim,
    # used by the steady state evaluation loop
    # ---------------------------------------------------
    def selectParents(population, fitnesses):
        sortedGen = [individual for fitness, individual in sorted(zip(fitnesses, population), key=lambda x: x[0])]
        del sortedGen[3:]
        P = [0.6, 0.25, 0.15]
        ParentA = random.choices(sortedGen, weights= P,k=1)[0]
        ParentB = random.choices(sortedGen, weights= P,k=1)[0]
        while ParentB == ParentA:
            ParentB = random.choices(sortedGen, weights= P,k=1)[0]

        return ParentA, ParentB

    # ------------------------------------------------------------
    # THE REAL GRAPH SEARCH ALGORITHIM USING EVERYTHING FROM ABOVE
    # ------------------------------------------------------------

    startingTime = time.monotonic()
    bestFitness = np.inf
    bestStrategy = None
    bestTree = None
    counter = [0] * startNodes
    currGen = []
    checkedTreesCounter = 0

    # Tracks whether ANY clearance strategy has been found so far - not just
    # whether the current bestFitness is finite - so treeSearch (see below)
    # knows when local clearance candidates are still worth computing.
    foundClearance = False
    localClearanceCandidates = []

    def shouldStop():
        if maxTrees is not None:
            return checkedTreesCounter >= maxTrees
        return time.monotonic() - startingTime >= availableTime

    if searchMode == "random":

        #------------------------------------------------------------
        # PURE RANDOM SPANNING TREE GENERATION (baseline for comparison
        # against the evolutionary steady-state search below)
        #------------------------------------------------------------
        while not shouldStop():

            root = random.randint(0, startNodes - 1)
            T = computeRandomSpanningTree(G, root)

            strategy, clearance, visitedTimes, _ = treeSearch(T, root, availableRobots, G)
            checkedTreesCounter += 1
            if clearance:
                foundClearance = True
                fitness = computeExpTime(visitedTimes, G)
            else: fitness = np.inf
            if fitness < bestFitness:
                bestFitness = fitness
                bestTree = T
                bestStrategy = strategy
            if historyCallback is not None:
                historyCallback(checkedTreesCounter, fitness, bestFitness)

    else:

        #------------------------------------------------------------
        # INITALIZING THE STARTING POPULATION
        #------------------------------------------------------------
        for i in range(0, populationSize):
            if counter != [1] * startNodes:
                for i in range(0,len(counter)):
                    if counter[i] == 0:
                        counter[i] = 1
                        root = i
                        T = computeGreedySpanningTree(G,root)
                        currGen.append((T,root))
                        break
            else: 
                root = random.randint(0,startNodes-1)
                T = computeRandomSpanningTree(G,root)
                currGen.append((T,root))


        #------------------------------------------------------------
        # ANYTIME ALGORITHIM (STEADY STATE EVALUATION)
        #------------------------------------------------------------
        #NOTE: If the available Robots is greater or equal than the amount of nodes the result will always be the trivial strategy that immidiatly sends a robot to every node

        # Evaluating the starting population once
        fitnesses = []
        for indivium in currGen:
            if shouldStop():
                break

            strategy, clearance, visitedTimes, _ = treeSearch(indivium[0], indivium[1], availableRobots, G)
            checkedTreesCounter += 1
            if clearance:
                foundClearance = True
                fitness = computeExpTime(visitedTimes, G)
            else: fitness = np.inf
            fitnesses.append(fitness)
            if fitness < bestFitness:
                bestFitness = fitness
                bestTree = indivium[0]
                bestStrategy = strategy
            if historyCallback is not None:
                historyCallback(checkedTreesCounter, fitness, bestFitness)

        # Steady state reproduction: always select parents via weighted selection,
        # add one evaluated child to the population and remove its worst member
        while not shouldStop():

            ParentA, ParentB = selectParents(currGen, fitnesses)
            child = crossOver(ParentA, ParentB, G)

            strategy, clearance, visitedTimes, _ = treeSearch(child[0], child[1], availableRobots, G)
            checkedTreesCounter += 1
            if clearance:
                foundClearance = True
                fitness = computeExpTime(visitedTimes, G)
            else: fitness = np.inf
            if fitness < bestFitness:
                bestFitness = fitness
                bestTree = child[0]
                bestStrategy = strategy
            if historyCallback is not None:
                historyCallback(checkedTreesCounter, fitness, bestFitness)

            currGen.append(child)
            fitnesses.append(fitness)

            worstIdx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            del currGen[worstIdx]
            del fitnesses[worstIdx]

    if bestFitness == np.inf:
        bestStrategy, bestTree, _, _ = computeClosingExits(root, localClearanceCandidates, G)

    return bestStrategy, bestTree, checkedTreesCounter, bestFitness

