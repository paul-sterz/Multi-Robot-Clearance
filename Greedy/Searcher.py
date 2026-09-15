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


def graphSearch(G : Graph, availableTime, availableRobots, startNodes, obstacles, distanceMap, alpha, cellpriors, D, maxTrees=None, populationSize=10, historyCallback=None, searchMode="evolutionary", travelTime=None, horizon=15, recencyLambda=None, probBudget = 200):
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
    # horizon: time budget (same unit as edge travel time) used by FHPE_SA's finite-horizon
    #          path enumeration (the fallback strategy when no clearance is found)
    # recencyLambda: recency decay constant used by FHPE_SA (default None -> 2 * horizon)
    # probBudget: simulated-time budget used by FHPE_SA - every move it plans arrives at its
    #          target by t = probBudget at the latest (default 200); NOT a wall-clock limit
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

    # ---------------------------------------------------
    # COMPUTING EDGE LABLES FOR THE TREE SEARCH THAT REPRESENT THE AMOUNT OF NEEDED ROBOTS AND THE EFFICIENCY OF EACH SUBTREE
    # Remark: Lables represent the amount of robots needed for this path
    # ---------------------------------------------------

    def computeWorstCaseLabels(T : Graph, root, parent):

        #Saving labels in a dictionary of the form: ((x,y) | lambda((x,y))
        edgeLabelsRobotCost = {} 
        edgeLabelsEfficiency = {}


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
                subLabelsRobotCost, subLablesEfficiency, sumTime, sumPrior = computeWorstCaseLabels(T, y.idx, root)
                totalTime += sumTime
                totalPrior += sumPrior

                subTreePriors[y.idx] = sumPrior
                subTreeTimes[y.idx] =  sumTime

                edgeLabelsEfficiency.update(subLablesEfficiency)
                edgeLabelsRobotCost.update(subLabelsRobotCost)
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


        return edgeLabelsRobotCost, edgeLabelsEfficiency, totalTime, totalPrior

    
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

        # STEP 1: ALLOCATION ‚
        BLabels , effLables,  _ ,_ = computeWorstCaseLabels(T, root, None)

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

            
            candidateLabels = []

            for neighbour in T.adj[T.nodes[node]]:
                if neighbour.idx == parent:
                    continue

                candidateLabels.append((effLables[(node,neighbour.idx)] , neighbour.idx))

            candidateLabels.sort(key =lambda x: x[0], reverse=True)

            snap = {
                "visitedTimes": copy.deepcopy(visitedTimes),
                "guards": copy.deepcopy(guards),
                "flags": copy.deepcopy(flags),
                "strat": copy.deepcopy(strat),
                "enteringTime": enteringTime
            }

            successful = True
            for tuple in candidateLabels:
                singleBatch = {"children": [tuple[1]]}

                successful, newMoves, batchTime = executeBatch(singleBatch, enteringTime, node)

                if not successful: 
                    visitedTimes = snap["visitedTimes"]
                    guards = snap["guards"]
                    flags = snap["flags"]
                    strat = snap["strat"]
                    enteringTime = snap["enteringTime"]
                    break

                strat.extend(newMoves)
                enteringTime = batchTime

            #Last Resort: run the child with the largest BLabel last, the rest
            #in efficiency order. If the largest BLabel occurs more than once,
            #no order can free up enough robots for it, so fail immediately
            #instead of wasting time on a doomed attempt.
            if len(strat) == 0:
                if shouldStop():
                    return False, [], enteringTime

                neighbours = [neighbour for neighbour in T.adj[T.nodes[node]] if neighbour.idx != parent]

                if len(neighbours) > 0:
                    maxBLabel = max(BLabels[(node, neighbour.idx)] for neighbour in neighbours)
                    maxBLabelNeighbours = [neighbour for neighbour in neighbours if BLabels[(node, neighbour.idx)] == maxBLabel]

                    if len(maxBLabelNeighbours) >= 2:
                        return False, [], enteringTime

                    largestNeighbour = maxBLabelNeighbours[0]
                    orderedNeighbours = sorted(
                        (neighbour for neighbour in neighbours if neighbour.idx != largestNeighbour.idx),
                        key=lambda neighbour: effLables[(node, neighbour.idx)],
                        reverse=True
                    )
                    orderedNeighbours.append(largestNeighbour)

                    for neighbour in orderedNeighbours:
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
                durration = flagDistance[flag[0]][node]

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

    def bestPathForSearcher(startNode, startTime, horizonVisited, lastVisitTime, horizon, lam, G : Graph):

        deadline = startTime + horizon + 1

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
        # neighbour reachable within horizon/probBudget at all).
        best = {"F": None, "path": None, "times": None}

        def visit(node, t, path, times):
            for neighbour in G.adj[G.nodes[node]]:
                newT = t + G.edges[(node, neighbour.idx)].time
                if newT - startTime > horizon or newT > probBudget:
                    continue
                path.append(neighbour.idx)
                times.append(newT)

                F = objective(path, times)
                if best["F"] is None or F < best["F"]:
                    best["F"] = F
                    best["path"] = list(path)
                    best["times"] = list(times)

                visit(neighbour.idx, newT, path, times)

                path.pop()
                times.pop()

        visit(startNode, startTime, [startNode], [startTime])

        if best["path"] is None:
            return [startNode], [startTime]

        return best["path"], best["times"]


    def FHPE_SA(root, G : Graph):

        lam = recencyLambda if recencyLambda is not None else 20 * horizon

        lastVisitTime = {root: 0} #Format Node: lastVisitTime
        executionPlan = [(None, root, availableRobots)]

        robotNode = [root] * availableRobots #Position of where the nodes are after the horizon iteration
        robotTime = [0] * availableRobots #Time when horizon iteration ends

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

            for k in range(availableRobots):

                path, times = bestPathForSearcher(
                    robotNode[k], robotTime[k], horizonVisited, lastVisitTime, horizon, lam, G
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
                for k in range(availableRobots):
                    robotTime[k] = probBudget
                roundCounter += 1
                continue

            T = min(nonTrivialFinishTimes)

            for k in range(availableRobots):

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

        # A lightweight graph of every edge actually travelled, purely so
        # the "Spanning tree" visualization layer has something to draw.
        visualTree = Graph()
        for n in G.nodes:
            visualTree.add_node2(n)
        for move in executionPlan[1:]:
            src, tgt, _, dep, arr = move
            if (src, tgt) in visualTree.edges:
                continue
            edgeInfo = G.edges.get((src, tgt)) or G.edges.get((tgt, src))
            edgeTime = edgeInfo.time if edgeInfo is not None else (arr - dep)
            robotType = edgeInfo.robotType if edgeInfo is not None else 2
            visualTree.add_edge(visualTree.nodes[src], visualTree.nodes[tgt], edgeTime, robotType)

        return executionPlan, visualTree, roundCounter, None


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
        bestStrategy, T, _, _  = FHPE_SA(root ,G)
        bestTree = T

    return bestStrategy, bestTree, checkedTreesCounter, bestFitness

