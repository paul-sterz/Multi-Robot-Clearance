import numpy as np
import random
from Graph import Graph, Node, Edge
from TrajectoryPlanning import aStar
import time
import copy

def graphSearch(G : Graph, availableTime, startNodes, obstacles, distanceMap, alpha, maxTrees=None):
    #INPUT:
    # G: a Graph object repesenting the given Graph
    # availableTime: the available computation time budget in seconds (ignored if maxTrees is given)
    # startNodes: an integer representing that all nodes from 0 to startNodes-1 are valid roots
    # maxTrees: if given (not None), stop after evaluating exactly this many spanning trees instead of using availableTime

    #OUTPUT:
    #bestStrategy: list of moves. First entry is (None, root, totalRobots), all following
    #              entries are (source, target, robots, t_departure, t_arrival)
    #bestTree: the spanning tree (as a Graph) that produced bestStrategy
    #checkedTreesCounter: how many random spanning trees were evaluated
    #minCost: the number of robots needed by bestStrategy


    #Calculating the distance matrix for all nodes
    numGraphNodes = len(G.nodes)
    flagDistance = [[0] * numGraphNodes for _ in range(numGraphNodes)]

    for i in range(numGraphNodes):
        for j in range(numGraphNodes):
            if i == j:
                continue
            path = aStar(G.nodes[i].pos, G.nodes[j].pos, obstacles, distanceMap, alpha)
            flagDistance[i][j] = len(path) - 1



    # ---------------------------------------------------
    # COMPUTING EDGE LABLES FOR THE TREE SEARCH 
    # Remark: Lables represent the amount of robots needed for this path
    # ---------------------------------------------------

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
    
    # ---------------------------------------------------
    # CALCULATING A STRATEGY FOR TREES
    # Note: A strategy is the order in which nodes are visited, following
    #       the B-labels ascending (cheapest subtree first).
    # Strategy Foramt: [(nextTargetNode, strategy Index when the guard can leave this node agains), ...]
    # ---------------------------------------------------
                    
    def treeSearch(T : Graph, root):

        labels = computeLabels(T,root, None)
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
                # While under construction: track which enemy NODES are
                # still unscheduled ("pending"), and which STRATEGY
                # INDICES this guard already depends on ("deps") - a
                # guard may have to wait on several of these, not just
                # whichever happened to be scheduled last.
                strategy.append([node, {"pending": set(pendingEnemys), "deps": knownDeps}])
            elif len(knownDeps) > 0:
                strategy.append([node, knownDeps])
            else:
                # No dependency at all: releases as soon as its own task runs.
                strategy.append([node, [myIndex]])

            nodeStrategyIndex[node] = myIndex

        addEntry(root)

        def explorePath(node, parent):

            candidateLabels = []

            for neighbour in T.adj[T.nodes[node]]:
                if neighbour.idx == parent:
                    continue

                candidateLabels.append((labels[(node,neighbour.idx)] , neighbour.idx))

            if len(candidateLabels) > 0:
                #Sorting the B-Labels ascending
                candidateLabels.sort(key =lambda x: x[0], reverse=False)

                for candidate in candidateLabels:

                    for entry in strategy:
                        rel = entry[1]
                        if isinstance(rel, dict) and candidate[1] in rel["pending"]:
                            rel["pending"].discard(candidate[1])
                            rel["deps"].append(len(strategy))
                            if not rel["pending"]:
                                # Fully resolved: from now on this is
                                # the complete list of strategy indices
                                # whose real completion time this guard
                                # must wait for (not just the last one).
                                entry[1] = rel["deps"]

                    visited[candidate[1]] = 1

                    addEntry(candidate[1])

                    explorePath(candidate[1], node)


        explorePath(root, None)
        robotCost = labels[(None, root)]

        return transformStrategy(strategy, labels, robotCost, root)
        

    def transformStrategy(strategy, labels, minRobots, root):
        """
        strategy:
            [(node, releaseIndex), ...]

            releaseIndex = letzter Strategy-Index, bis zu dem der
            Roboter auf diesem Knoten als Guard benötigt wird.

        Assumption:
            flagDistance[u][v] is available in the surrounding scope.

        Returns:
            executionPlan, robotsNeeded

        executionPlan format (same as Baseline Modified):
            First entry is (None, root, robotsNeeded), all following
            entries are (source, target, robots, t_departure, t_arrival).
            Levels only decide which robot is assigned to which node
            (via the LBAP); t_arrival is always that robot's own,
            independent, as-fast-as-possible physical arrival time -
            never delayed by other tasks in the same or an earlier level.
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
        # ============================================================

        def levelDistribution(strategy, minRobots):

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

                    levelCapacity = (
                        minRobots - len(activeGuards)
                    )

                    if levelCapacity < 1:
                        return levelDistribution(
                            strategy,
                            minRobots + 1
                        )

                # ----------------------------------------------------
                # No more robots available inside this level
                # ----------------------------------------------------

                if len(currentLevel) >= levelCapacity:

                    levels.append(currentLevel)
                    currentLevel = []

                    levelCapacity = (
                        minRobots - len(activeGuards)
                    )

                    if levelCapacity < 1:
                        return levelDistribution(
                            strategy,
                            minRobots + 1
                        )

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

            return levels, minRobots


        # ============================================================
        # CREATE LEVELS
        # ============================================================

        # If labels[(None, root)] is your B-label / minimum robot count:
        initialRobotNumber = minRobots

        levels, robotsNeeded = levelDistribution(strategy, initialRobotNumber)


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
            for robotId in range(robotsNeeded)
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


        executionPlan.insert(0, (None, root, robotsNeeded))

        return executionPlan, robotsNeeded



    # ---------------------------------------------------
    # COMPUTING A RANDOM SPANNING TREE WITH DFS
    # ---------------------------------------------------

    def computeRandomSpanningTree(G : Graph, root):
     
        T = Graph()
        for i in range(len(G.nodes)):
            T.add_node2(G.nodes[i])
        
        visited = set()

        def dfs(node):

            visited.add(node)

            neighbours = list(G.adj[G.nodes[node]])

            random.shuffle(neighbours)

            for neighbour in neighbours:

                if neighbour.idx in visited:
                    continue

                T.add_edge(T.nodes[node], T.nodes[neighbour.idx])

                dfs(neighbour.idx)

        dfs(root)
        return T


    def shouldStop():
        if maxTrees is not None:
            return checkedTreesCounter >= maxTrees
        return time.monotonic() - startingTime >= availableTime

    # ------------------------------------------------------------
    # THE REAL GRAPH SEARCH ALGORITHIM USING EVERYTHING FROM ABOVE
    # ------------------------------------------------------------
    startingTime = time.monotonic()
    minCost = np.inf
    bestStrategy = None
    bestTree = None
    checkedTreesCounter = 0

    while not shouldStop():
        checkedTreesCounter += 1
        root = random.randint(0,startNodes-1)
        T = computeRandomSpanningTree(G,root)
        strat, neededRobots = treeSearch(T,root)
        if neededRobots < minCost:
            minCost = neededRobots
            bestStrategy = strat
            bestTree = T

    return bestStrategy, bestTree, checkedTreesCounter, minCost

