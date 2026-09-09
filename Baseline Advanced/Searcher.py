import numpy as np
import random
from Graph import Graph, Node, Edge
from TrajectoryPlanning import aStar
import time
import copy

def graphSearch(G : Graph, shadyEdges, numOfTrees, startNodes, obstacles, distanceMap, alpha, maxTrees=None):
    #INPUT:
    # G: a Graph object repesenting the given Graph
    # edges_shady:a list containing all shady edges in the form (i,j)
    # numOfTrees: an integer which represents the number of evaluated trees

    #OUTPUT:
    #bestStrategy: A list containing the best strategy where each entry is in the form (source node,target node, amount of Robots)


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
    # TO-DO: Add Timestamps to the strategy and allow moves at the same time if amount of robots is not reached. 
    #        Note that this will only work if I merge this with the Transform to Graph method.
    # Note: A Strategy is safed in the format [(source node,target node, amount of Robots),...]
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


        strategy = explorePath(root)

        strategy.insert(0, root)

        strategy = transformStrategy(strategy, labels)

        return strategy
        
            

    def transformStrategy():

        #TODO Compute execution Schedule after Kollings Paper



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

    minCost = np.inf
    bestStrategy = None
    bestTree = None

    for i in range(numOfTrees):
        root = random.randint(0,startNodes-1)
        T = computeRandomSpanningTree(G,root)
        treeStrategy = treeSearch(T,root)
        graphStrategy = transformStrategy(G,shadyEdges, treeStrategy)
        if graphStrategy[0][2] < minCost:
            minCost = graphStrategy[0][2]
            bestStrategy = graphStrategy
            bestTree = T

    return bestStrategy, bestTree

