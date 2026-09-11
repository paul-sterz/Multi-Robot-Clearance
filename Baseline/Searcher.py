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

        def addEntry(node):
            # enemys = graph-neighbours of node not yet reached by the
            # search strategy at this point -> node's guard must stay
            # until all of them have been visited.
            enemys = [
                neighbour.idx
                for neighbour in G.adj[G.nodes[node]]
                if visited[neighbour.idx] == 0
            ]

            if len(enemys) > 0:
                strategy.append([node, enemys])
            else:
                strategy.append([node, len(strategy)])

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
                        if isinstance(entry[1], list) and candidate[1] in entry[1]:
                            entry[1].remove(candidate[1])
                            if len(entry[1]) == 0:
                                entry[1] = len(strategy)

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
            t_arrival is the task's executionTime (i.e. the point at which
            the strategy order guarantees the node is actually cleared),
            not the possibly-earlier physical arrival.
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
                # ----------------------------------------------------

                if releaseIndex != i:
                    activeGuards.append(releaseIndex)

                # ----------------------------------------------------
                # Change in free robot set -> level ends
                # ----------------------------------------------------

                if freedNow or releaseIndex == i:

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
        initialRobotNumber = max(
            minRobots,
            labels[(None, root)]
        )

        levels, robotsNeeded = levelDistribution(
            strategy,
            initialRobotNumber
        )


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
                # integer => guarding until this strategy index
                "guardUntil": None
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

        # Completion time of previous strategy task.
        #
        # Although robots of a level can move in parallel,
        # the SEARCH TASKS themselves still have to be executed
        # in the prescribed strategy order.
        previousExecutionTime = 0


        for level in levels:

            # --------------------------------------------------------
            # Solve LBAP using robots currently free
            # --------------------------------------------------------

            assignment, bottleneck = LBAPSolver(
                level,
                robots
            )

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

                # Robot may arrive early and wait.
                #
                # The vertex may only be cleared once all previous
                # search-strategy tasks have been completed.
                executionTime = max(
                    previousExecutionTime,
                    info["arrival"]
                )

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

                previousExecutionTime = executionTime

                # Robot is physically located at its assigned node.
                robot["node"] = node

                # ----------------------------------------------------
                # Current robot becomes guard or immediately free
                # ----------------------------------------------------

                if releaseIndex == taskIndex:

                    robot["guardUntil"] = None
                    robot["freeFrom"] = executionTime

                else:

                    robot["guardUntil"] = releaseIndex

                    # Exact release time is not yet known.
                    # It will be set once releaseIndex executes.
                    robot["freeFrom"] = None


                # ----------------------------------------------------
                # Reaching taskIndex may release OTHER guards
                # ----------------------------------------------------

                for otherRobot in robots:

                    if (
                        otherRobot["guardUntil"] == taskIndex
                        and otherRobot["id"] != robotId
                    ):

                        otherRobot["guardUntil"] = None
                        otherRobot["freeFrom"] = executionTime


            # --------------------------------------------------------
            # Safety:
            # current task itself could also have become free through
            # its release index.
            # --------------------------------------------------------

            lastTaskIndex = level[-1][0]
            lastExecutionTime = previousExecutionTime

            for robot in robots:

                if robot["guardUntil"] == lastTaskIndex:
                    robot["guardUntil"] = None
                    robot["freeFrom"] = lastExecutionTime


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

