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


def graphSearch(G : Graph, availableTime, availableRobots, startNodes, obstacles, distanceMap, alpha, cellpriors, D, maxTrees=None, populationSize=10, historyCallback=None, searchMode="evolutionary"):
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
    flagDistance = [[0] * numGraphNodes for _ in range(numGraphNodes)]

    for i in range(numGraphNodes):
        for j in range(numGraphNodes):
            if i == j:
                continue
            path = aStar(G.nodes[i].pos, G.nodes[j].pos, obstacles, distanceMap, alpha)
            flagDistance[i][j] = len(path) - 1

    # ---------------------------------------------------
    # COMPUTING B-LABELS and efficiency labels depending on available robots
    # ---------------------------------------------------
    def computeLabelsWithBudget(T: Graph, root, parent, maxRobots: int):

        bLabels = {}
        robotTable = {}
        policyTable = {}

        children = [y.idx for y in T.adj[T.nodes[root]] if y.idx != parent]

        # ---------------- LEAF ----------------
        if len(children) == 0:
            bLabels[(parent, root)] = 1

            edgeTime = T.edges[(parent, root)].time if parent is not None else 1
            prior = T.nodes[root].prior

            robotTable[(parent, root)] = {
                r: (edgeTime, prior / edgeTime)
                for r in range(1, maxRobots + 1)
            }

            policyTable[(parent, root)] = {
                r: [] for r in range(1, maxRobots + 1)
            }

            return bLabels, robotTable, policyTable, prior, edgeTime

        # ---------------- INNER NODE ----------------
        childBLabel = {}
        childPrior = {}
        childTimeTable = {}

        totalPrior = T.nodes[root].prior

        for child in children:
            subB, subTable, subPolicy, subPrior, _ = computeLabelsWithBudget(
                T, child, root, maxRobots
            )

            bLabels.update(subB)
            robotTable.update(subTable)
            policyTable.update(subPolicy)

            childBLabel[child] = subB[(root, child)]
            childPrior[child] = subPrior
            childTimeTable[child] = {
                r: t for r, (t, _eff) in subTable[(root, child)].items()
            }

            totalPrior += subPrior

        # ---------------- B-LABEL ----------------
        childLabelsSorted = sorted(childBLabel.values(), reverse=True)

        p1 = childLabelsSorted[0]
        p2 = childLabelsSorted[1] if len(childLabelsSorted) > 1 else 0

        bLabelRoot = p1 + 1 if p1 == 1 else max(p1, p2 + 1)
        bLabels[(parent, root)] = bLabelRoot

        # ---------------- FEASIBILITY ----------------
        if maxRobots < bLabelRoot:
            robotTable[(parent, root)] = {}
            policyTable[(parent, root)] = {}

            return bLabels, robotTable, policyTable, totalPrior, float('inf')

        edgeTime = T.edges[(parent, root)].time if parent is not None else 1

        # ---------------- BEST POLICY FOR EACH ROBOT BUDGET ----------------
        table = {}
        policies = {}

        for r in range(bLabelRoot, maxRobots + 1):
            bestChildrenTime, bestSchedule = _bestChildrenSchedule(
                children, childBLabel, childTimeTable, childPrior, r
            )

            time_r = edgeTime + bestChildrenTime

            table[r] = (time_r, totalPrior / time_r)
            policies[r] = bestSchedule

        robotTable[(parent, root)] = table
        policyTable[(parent, root)] = policies

        return bLabels, robotTable, policyTable, totalPrior, table[maxRobots][0]


    def _bestChildrenSchedule(children, childBLabel, childTimeTable, childPrior, r):
        """
        Returns:
            bestTime:
                Minimal total clearance time of all child subtrees.

            schedule:
                Ordered list of sequential batches.
                Children inside one batch run in parallel.

                [
                    {
                        "children": [...],
                        "robots": {child: robots, ...},
                        "time": batchTime,
                        "prior": batchPrior,
                        "efficiency": batchPrior / batchTime
                    },
                    ...
                ]
        """

        k = len(children)

        if k == 0:
            return 0.0, []

        fullMask = (1 << k) - 1

        # ---------------- COST OF EVERY POSSIBLE PARALLEL BATCH ----------------
        costOfSubset = [float('inf')] * (1 << k)
        allocationOfSubset = [None] * (1 << k)

        costOfSubset[0] = 0.0
        allocationOfSubset[0] = {}

        for mask in range(1, 1 << k):
            subset = [children[i] for i in range(k) if mask & (1 << i)]

            if sum(childBLabel[c] for c in subset) > r:
                continue

            batchTime, allocation = _parallelBatchCost(
                subset, childTimeTable, r
            )

            costOfSubset[mask] = batchTime
            allocationOfSubset[mask] = allocation

        # ---------------- DP OVER PARTITIONS ----------------
        dp = [float('inf')] * (1 << k)
        choice = [None] * (1 << k)

        dp[0] = 0.0

        for mask in range(1, 1 << k):
            low = mask & (-mask)
            sub = mask

            while sub > 0:
                if sub & low:
                    candidate = costOfSubset[sub] + dp[mask ^ sub]

                    if candidate < dp[mask]:
                        dp[mask] = candidate
                        choice[mask] = sub

                sub = (sub - 1) & mask

        if dp[fullMask] == float('inf'):
            return float('inf'), []

        # ---------------- RECONSTRUCT OPTIMAL PARTITION ----------------
        schedule = []
        mask = fullMask

        while mask != 0:
            chosenMask = choice[mask]

            if chosenMask is None:
                return float('inf'), []

            subset = [
                children[i]
                for i in range(k)
                if chosenMask & (1 << i)
            ]

            batchTime = costOfSubset[chosenMask]
            batchPrior = sum(childPrior[c] for c in subset)
            batchEfficiency = batchPrior / batchTime

            schedule.append({
                "children": subset,
                "robots": allocationOfSubset[chosenMask],
                "time": batchTime,
                "prior": batchPrior,
                "efficiency": batchEfficiency
            })

            mask ^= chosenMask

        # ---------------- ORDER BATCHES BY PRIOR PER TIME ----------------
        schedule.sort(key=lambda batch: batch["efficiency"], reverse=True)

        return dp[fullMask], schedule


    def _parallelBatchCost(subset, childTimeTable, r):
        """
        Returns:
            batchTime:
                Minimal time in which all children of the subset can be
                cleared in parallel.

            allocation:
                child -> required number of robots
        """

        if len(subset) == 0:
            return 0.0, {}

        candidateTimes = set()

        for c in subset:
            candidateTimes.update(childTimeTable[c].values())

        for T in sorted(candidateTimes):
            allocation = {}
            totalRobots = 0
            feasible = True

            for c in subset:
                minRi = None

                for ri in sorted(childTimeTable[c]):
                    if childTimeTable[c][ri] <= T:
                        minRi = ri
                        break

                if minRi is None:
                    feasible = False
                    break

                allocation[c] = minRi
                totalRobots += minRi

            if feasible and totalRobots <= r:
                return T, allocation

        return float('inf'), None
    
    # ---------------------------------------------------
    # CALCULATING A STRATEGY FOR TREES AND TRANSFORMING IT TO THE GRAPH
    # Note: A Strategy is safed in the format [(source node idx,target node idx, amount of robots, time_Depature, time_Arrival),...]
    # Approach (same two-stage split as Baseline's Searcher):
    #   1) Build the abstract strategy purely as [(node, releaseIndex), ...]
    #      - the order the tree is walked in, and the strategy index(es)
    #      each guard depends on before it may leave. Children of a node
    #      are ordered by descending efficiency (from the DP's
    #      robotTable, at each child's own minimal/BLabel budget), except
    #      the single costliest (by BLabel) child, which always goes last
    #      - so it is still guarded by its parent while every cheaper
    #      sibling subtree gets cleared first. This is exactly the order
    #      DP BLabel Order already used.
    #   2) Feed that abstract strategy into the same multi-level LBAP
    #      assignment Baseline uses, except the robot count is fixed
    #      up-front to availableRobots instead of starting from the
    #      B-label bound and retrying with one extra robot on failure -
    #      if availableRobots does not suffice, clearance simply fails.
    # ---------------------------------------------------
    def treeSearch(T : Graph, root, availableRobots, G : Graph):
        # ------------------------------------------------
        # INPUT:
        # T - The Tree on which the strategy is calculated (Graph object)
        # root - The starting point of the strategy (index)
        # availableRobots - The amount of robots that can be used for exploration (non negative Integer)
        # G - The Navigation Graph (Graph object)
        # ------------------------------------------------
        # OUTPUT:
        # strategy, clearance, visitedTimes - clearance is False (with the
        # other two None) whenever availableRobots does not suffice.

        BLabels, robotTable, policyTable, _, _ = computeLabelsWithBudget(T, root, None, availableRobots)

        # BLabels only ever increase going from a leaf up to the root (a
        # parent's BLabel is always >= its largest child's), so root's own
        # BLabel is the tree-wide maximum. Checking it once here guarantees
        # robotTable[(node, child)][BLabels[(node, child)]] is populated for
        # every edge visited below - without this check, an edge whose own
        # BLabel exceeds availableRobots would have an EMPTY robotTable
        # entry (see computeLabelsWithBudget's feasibility branch) and blow
        # up with a KeyError during ordering, instead of cleanly failing.
        if BLabels[(None, root)] > availableRobots:
            return None, False, None

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

            # Run the child with the largest BLabel last, the rest in
            # efficiency order (efficiency at its own minimal robot
            # budget, i.e. robotTable at r = BLabel). If the largest
            # BLabel occurs more than once, no order can free up enough
            # robots for it, so fall back to a plain efficiency order.
            maxBLabel = max(BLabels[(node, neighbour.idx)] for neighbour in neighbours)
            maxBLabelNeighbours = [neighbour for neighbour in neighbours if BLabels[(node, neighbour.idx)] == maxBLabel]

            if len(maxBLabelNeighbours) >= 2:
                orderedNeighbours = sorted(
                    neighbours,
                    key=lambda neighbour: robotTable[(node, neighbour.idx)][BLabels[(node, neighbour.idx)]][1],
                    reverse=True
                )
            else:
                largestNeighbour = maxBLabelNeighbours[0]
                orderedNeighbours = sorted(
                    (neighbour for neighbour in neighbours if neighbour.idx != largestNeighbour.idx),
                    key=lambda neighbour: robotTable[(node, neighbour.idx)][BLabels[(node, neighbour.idx)]][1],
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

        return transformStrategy(strategy, availableRobots, root)


    def transformStrategy(strategy, availableRobots, root):
        """
        strategy:
            [(node, releaseIndex), ...]

            releaseIndex = last strategy index up to which the robot on
            this node is still needed as a guard.

        availableRobots:
            fixed robot budget. Unlike Baseline this never grows: if
            availableRobots does not suffice, clearance fails outright
            instead of retrying with one more robot.

        Assumption:
            flagDistance[u][v] is available in the surrounding scope.

        Returns:
            executionPlan, clearance, visitedTimes

        executionPlan format (same as Baseline):
            First entry is (None, root, availableRobots), all following
            entries are (source, target, robots, t_departure, t_arrival).
        """

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
            return None, False, None


        # ============================================================
        # ROBOT STATE
        # ============================================================

        robots = [
            {
                "id": robotId,

                # Current physical position
                "node": root,

                # Earliest time at which this robot may depart
                "freeFrom": 0,

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


        executionPlan.insert(0, (None, root, availableRobots))

        # ============================================================
        # VISITED TIMES
        # Every non-root node appears exactly once as a target in
        # executionPlan, since the strategy walks a spanning tree.
        # ============================================================

        visitedTimes = [-1] * len(T.nodes)
        visitedTimes[root] = 0

        for move in executionPlan[1:]:
            _, target, _, _, arrival = move
            visitedTimes[target] = arrival

        return executionPlan, True, visitedTimes

            


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
    # COMPUTING THE CLOSING EXISTS STRATEGY IF NOT ENOUGH ROBOTS ARE GIVEN
    # TODO: Implement method
    # ---------------------------------------------------

    def computeClosingExits(graphStrategy, G):
        print("Hello World!")
        return []


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

            strategy, clearance, visitedTimes = treeSearch(T, root, availableRobots, G)
            checkedTreesCounter += 1
            if clearance: fitness = computeExpTime(visitedTimes, G)
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

            strategy, clearance, visitedTimes = treeSearch(indivium[0], indivium[1], availableRobots, G)
            checkedTreesCounter += 1
            if clearance: fitness = computeExpTime(visitedTimes, G)
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

            strategy, clearance, visitedTimes = treeSearch(child[0], child[1], availableRobots, G)
            checkedTreesCounter += 1
            if clearance: fitness = computeExpTime(visitedTimes, G)
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
        bestStrategy = computeClosingExits(bestStrategy,G)
        print(availableRobots)
        print("HAALLLO")
        bestTree = G

    return bestStrategy, bestTree, checkedTreesCounter, bestFitness

