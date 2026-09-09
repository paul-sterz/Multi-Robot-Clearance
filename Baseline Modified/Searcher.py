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
    # ---------------------------------------------------
                    
    def treeSearch(T : Graph, root):

        labels = computeLabels(T,root, None)
        visited = [0] * len(T.nodes)
        visited[root] = 1


        def explorePath(node, parent):
            strategy = []

                      
            candidateLabels = []

            for neighbour in T.adj[T.nodes[node]]:
                if neighbour.idx == parent:
                    continue

                candidateLabels.append((labels[(node,neighbour.idx)] , neighbour.idx))

            if len(candidateLabels) > 0:
                #Sorting the B-Labels ascending
                candidateLabels.sort(key =lambda x: x[0], reverse=False)    

                for candidate in candidateLabels:
                    strategy.append(candidate[1])
                    strategy.extend(explorePath(candidate[1], node))

            return strategy


        strategy = explorePath(root, None)
        strategy.insert(0, root)
        robotCost = labels[(None, root)]

        return transformStrategy(strategy, labels, robotCost, root)
        
            

    def transformStrategy(strategy, labels, minRobots, root):
        #NOTE: Since we do not now how many robots we have in the beginning some flag retracing won't be loacally optimal since at a later point a robot is added that could have been used earlier. 
        #      We will see what impact this has.

        visitedTimes = [-1] * len(T.nodes)
        visitedTimes[root] = 0

        guards = [] # guard format: (node index, list of enemys, min guarding time)
        rootEnemys = []
        for node in  G.adj[G.nodes[root]]:
            rootEnemys.append(node.idx)
        guards.append([root,rootEnemys, 0])

        flags = [(root,0)] * (minRobots -1) # flag format: (node index, flag begin)

        #Executionplan Format:[(source node idx,target node idx, time_Depature, time_Arrival),...]
        executionPlan = [] 
        additionalRobots = 0

        def findNearestFlag(node):
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

        # strategy[0] is always root itself, which is already placed and
        # guarded above - only the nodes it still needs to reach are
        # processed here (otherwise root would be guarded twice).
        for node in strategy[1:]:

            succesful, move, arrTime = findNearestFlag(node)

            if succesful == False:
                additionalRobots += 1
                flags.append((root, 0))

                succesful, move, arrTime = findNearestFlag(node)

            executionPlan.append(move)
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

        totalRobots = additionalRobots + minRobots
        executionPlan.insert(0, (None, root, totalRobots))

        return executionPlan, totalRobots



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

