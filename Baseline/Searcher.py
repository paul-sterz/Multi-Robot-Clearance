import numpy as np
import random

def graphSearch(nodes, edges, root, numOfTrees):
    #INPUT:
    # nodes: a list containing all nodes of the graph
    # edges: a list containing all directed edges of the graph
    # root: index of the root node
    # numOfTrees: an integer which represents the number of evaluated trees

    #OUTPUT:
    #bestStrategy: A list containing the best strategy where each entry is in the form (source node,target node, amount of Robots)

    # ---------------------------------------------------
    # COMPUTING EDGE LABLES FOR THE TREE SEARCH 
    # Remark: Lables represent the amount of robots needed for this path
    # ---------------------------------------------------

    def computeLabels(nodes, edges, root, parent):

        #Saving labels in a dictionary of the form: (x,y) | lambda((x,y))
        edgeLabels = {} 

        #Saving pi meaning all nodes to the children of root
        childLabels = [] 

        # Calculating lables recursive for children
        for (x, y) in edges:

            if x == root and y != parent:

                subLabels = computeLabels(nodes, edges, y, root)

                edgeLabels.update(subLabels)

                childLabels.append(subLabels[(root, y)])

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
    # Note: A Strategy is safed in the format [(source node,target node, amount of Robots),...]
    # ---------------------------------------------------
                    
    def treeSearch(nodes, edges, root):

        labels = computeLabels(nodes, edges, root, None)

        def explorePath(node, parent):
            
            strategy = []
            currentLables = []
            
            for (x,y) in edges:
                if x == node:
                    currentLables.append((labels[(x,y)], y))

            if len(currentLables) > 0:
                currentLables.sort(key=lambda x: x[0]) #Sorting the lables ascending

                counter = 1
                for (robotsNeeded, y) in currentLables:
                    if counter == len(currentLables) and robotsNeeded != 1: #Don't allow slide moves!
                        strategy.append((node,y,robotsNeeded-1))
                        strategy.append((node,y,1))
                        strategy.extend(explorePath(y,node))
                        strategy.append((y,node,robotsNeeded))
                    else:
                        strategy.append((node,y,robotsNeeded))
                        strategy.extend(explorePath(y,node))
                        strategy.append((y,node,robotsNeeded))    
                        counter = counter + 1        
            
            return strategy

        strategy = explorePath(root, None)

        robotCost = labels[(None, root)]

        strategy.insert(0, (None,root,robotCost))

        return strategy
        
    
    # ---------------------------------------------------
    # TRANSFORMING THE STRATEGY FROM TREE TO GRAPH
    # Not yet implement because paper is to unspecific
    # ---------------------------------------------------

    def transformStrategy():
        print("Hello World!")


    # ---------------------------------------------------
    # COMPUTING A RANDOM SPANNING TREE WITH DFS
    # ---------------------------------------------------

    def computeRandomSpanningTree(nodes, edges, root):
        visited = set()
        treeEdges = []

        def dfs(node):

            visited.add(node)

            neighbours = []

            for (u,v) in edges:
                if u == node:
                    neighbours.append(v)


            random.shuffle(neighbours)

            for neighbour in neighbours:

                if neighbour in visited:
                    continue

                treeEdges.append((node, neighbour))

                dfs(neighbour)

        dfs(root)
        return nodes, treeEdges




    # ------------------------------------------------------------
    # THE REAL GRAPH SEARCH ALGORITHIM USING EVERYTHING FROM ABOVE
    # ------------------------------------------------------------

    minCost = np.inf
    bestStrategy = None

    for i in range(numOfTrees):
        Vi, Ei = computeRandomSpanningTree(nodes,edges,root)
        treeStrategy = treeSearch(Vi,Ei, root)
        #graphStrategy = transformStrategy(treeStrategy)
        if treeStrategy[0][2] < minCost:
            minCost = treeStrategy[0][2]
            bestStrategy = treeStrategy

    return bestStrategy

