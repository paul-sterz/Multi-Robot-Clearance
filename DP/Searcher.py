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


def graphSearch(G : Graph, availableTime, availableRobots, startNodes, obstacles, distanceMap, alpha, cellpriors, D):
    #INPUT:
    # G: a Graph object repesenting the merged navigationgraph
    # availableTime: a number which represents the available computation time
    # availableRobots: an integer representing the amount of robots available
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
        # 

        # STEP 1: ALLOCATION 
        BLabels , _, policyTable, _ ,_ = computeLabelsWithBudget(T, root, None, availableRobots)

        visitedTimes = [-1] * len(T.nodes)
        visitedTimes[root] = 0

        guards = [] # guard format: (node index, list of enemys, min guarding time)
        rootEnemys = []
        for node in  G.adj[G.nodes[root]]:
            rootEnemys.append(node.idx)
        guards.append([root,rootEnemys, 0])

        flags = [(root,0)] * (availableRobots -1) # flag format: (node index, flag begin)

        def explorePath(node,enteringTime, parent):
            #Safestades, so that if something goes wrong we can backtrack to the old states
            nonlocal guards
            nonlocal flags
            nonlocal visitedTimes

            strat = []
            minRobotsNeeded = BLabels[(parent, node)]

            for r in range(len(flags),minRobotsNeeded - 1,-1):
                snap = {
                    "visitedTimes": copy.deepcopy(visitedTimes),
                    "guards": copy.deepcopy(guards),
                    "flags": copy.deepcopy(flags),
                    "strat": copy.deepcopy(strat),
                    "enteringTime": enteringTime
                }

                schedule = policyTable[(parent, node)][r]

                successful = True
                for batch in schedule:
                    successful, newMoves, batchTime = executeBatch(batch , enteringTime, node)

                    if not successful:
                        visitedTimes = snap["visitedTimes"]
                        guards = snap["guards"]
                        flags = snap["flags"]
                        strat = snap["strat"]
                        enteringTime = snap["enteringTime"]
                        break

                    strat.extend(newMoves)
                    enteringTime = batchTime

                if successful:
                    break

            #Last Resort: Try the BLabel Order if all other orders did not work
            if len(strat) == 0:
                neighbours = [neighbour for neighbour in T.adj[T.nodes[node]] if neighbour.idx != parent]
                neighbours.sort(key=lambda neighbour: BLabels[(node, neighbour.idx)])

                for neighbour in neighbours:
                    singleBatch = {"children": [neighbour.idx]}

                    successful, newMoves, batchTime = executeBatch(singleBatch, enteringTime, node)

                    if not successful:
                        return False, [], enteringTime

                    strat.extend(newMoves)
                    enteringTime = batchTime

            return True, strat, enteringTime


        def executeBatch(batch, enteringTime, currNode):
            nonlocal guards
            nonlocal flags
            nonlocal visitedTimes


            strat = []
            welcomeBackTimes = [-1] * (len( batch["children"])) #Saving when the batch is cleared meaning the longest time that is needed for a children
            counter = 0

            for node in batch["children"]:
                successful, newMove, arrTime = findNearestFlag(node, enteringTime)

                if successful == False:
                    return False, None, None

                strat.append(newMove)

                visitedTimes[node] = arrTime

                #Remove visited node from guard lists and update minGuardTime if neicessary
                for guard in guards[:]:   # Iterate over a copy of the guard list so that when removing an item nothing is skipped
                    if node in guard[1]:
                        guard[1].remove(node) #alters the real guard list
                        guard[2] = max(guard[2], arrTime) #alters the real guard list
                        if len(guard[1]) == 0:
                            flags.append((guard[0], guard[2]))
                            guards.remove(guard) #Removes from the real guard list

                enemys = []
                minGuardTime = arrTime
                for neighbour in G.adj[G.nodes[node]]:
                    if visitedTimes[neighbour.idx] == -1:
                        enemys.append(neighbour.idx)

                    #Guard at least until this node is visited
                    minGuardTime = max(minGuardTime,visitedTimes[neighbour.idx])

                if len(enemys) == 0:
                    flags.append((node, minGuardTime))
                else:
                    guards.append([node, enemys, minGuardTime])

                successful, newMoves, subtreeFinishTime = explorePath(node, max(enteringTime, arrTime), currNode)

                if not successful:
                    return False, None, None

                strat.extend(newMoves)
                welcomeBackTimes[counter] = subtreeFinishTime
                counter += 1

            batchTime = max(welcomeBackTimes)

            return True, strat, batchTime
                    



        def findNearestFlag(node, enteringTime):
            nonlocal guards
            nonlocal flags
            nonlocal visitedTimes

            minDurr = np.inf
            nearest = None

            for flag in flags:
                path = aStar(T.nodes[flag[0]].pos, T.nodes[node].pos, obstacles, distanceMap, alpha)
                durration = len(path) -1

                if durration + flag[1] < minDurr:
                    minDurr = durration + flag[1]
                    nearest = flag

            if nearest == None:
                return False, None, None

            flags.remove(nearest)
            return True, (nearest[0], node, 1, nearest[1], minDurr), minDurr

        successful, strat, _ = explorePath(root, 0, None)
        strat.insert(0,(None, root, availableRobots))
        return strat, successful, visitedTimes

            


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
    #------------------------------------------------------------
    # INITALIZING THE STARTING POPULATION
    #------------------------------------------------------------
    for i in range(0,10):
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
        if time.monotonic() - startingTime >= availableTime:
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

    # Steady state reproduction: always select parents via weighted selection,
    # add one evaluated child to the population and remove its worst member
    while time.monotonic() - startingTime < availableTime:

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

    return bestStrategy, bestTree, checkedTreesCounter

