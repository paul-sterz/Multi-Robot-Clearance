import numpy as np
import random
from Graph import Graph, Node, Edge

def graphSearch(G : Graph, shadyEdges, numOfTrees, startNodes):
    #INPUT:
    # G: a Graph object repesenting the given Graph
    # edges_shady:a list containing all shady edges in the form (i,j)
    # numOfTrees: an integer which represents the number of evaluated trees

    #OUTPUT:
    #bestStrategy: A list containing the best strategy where each entry is in the form (source node,target node, amount of Robots)

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
    # Note: A Strategy is safed in the format [(source node,target node, amount of Robots),...]
    # ---------------------------------------------------
                    
    def treeSearch(T : Graph, root):

        labels = computeLabels(T,root, None)

        def explorePath(node):
            
            strategy = []
            currentLables = []
    
            for y in T.adj[T.nodes[node]]:
                currentLables.append((labels[(node,y.idx)], y.idx))

            if len(currentLables) > 0:
                currentLables.sort(key=lambda x: x[0]) #Sorting the lables ascending

                counter = 1
                for (robotsNeeded, y) in currentLables:
                    if counter == len(currentLables) and robotsNeeded != 1: #Don't allow slide moves! Note that they can only be neicessary in the last move
                        strategy.append((node,y,robotsNeeded-1))
                        strategy.append((node,y,1))
                        strategy.extend(explorePath(y))
                        strategy.append((y,node,robotsNeeded))
                    else:
                        strategy.append((node,y,robotsNeeded))
                        strategy.extend(explorePath(y))
                        strategy.append((y,node,robotsNeeded))    
                        counter = counter + 1        
            
            return strategy

        strategy = explorePath(root)

        robotCost = labels[(None, root)]

        strategy.insert(0, (None,root,robotCost))

        return strategy
        
    
    # -------------------------------------------------------------------------------------
    # TRANSFORMING THE STRATEGY FROM TREE TO GRAPH
    # Approach from paper "The Graph Clear Problem..." by Kolling used since the baseline is to unspecific about this
    # -------------------------------------------------------------------------------------

    def transformStrategy(G : Graph, shadyEdges, strategy):

        contaminationArea = set()
        for i in range(len(G.nodes)):
            contaminationArea.add(i)

        rCounter = [0] * len(G.nodes)

        for i in range(len(strategy)):
            if strategy[i][1] in contaminationArea:
                contaminationArea.remove(strategy[i][1])
                
            if strategy[i][0] != None:
                rCounter[strategy[i][0]] = rCounter[strategy[i][0]] - strategy[i][2]
            
            rCounter[strategy[i][1]] = rCounter[strategy[i][1]] + strategy[i][2]
            

            if strategy[i][0] != None and rCounter[strategy[i][0]] == 0: #Are there still any robots left on the last node?

                exists = False

                for (u,v) in G.edges.keys(): #check regular edges
                    if u in contaminationArea and v == strategy[i][0]: #Is there an edge that leads to recontamination?
                        currNode = None
                        for j in range(i):
                            u, v, k = strategy[j]
                            if u == currNode:
                                strategy[j] = (u, v, k + 1)
                                currNode = v
                            if v == strategy[i][0]:
                                break

                        rCounter[strategy[i][0]] = 1
                        exists = True
                        break
                
                if exists == False:
                    for (u,v) in shadyEdges: #check shady edges
                        if u in contaminationArea and v == strategy[i][0]: #Is there an edge that leads to recontamination?
                            currNode = None
                            for j in range(i):
                                u, v, k = strategy[j]
                                if u == currNode:
                                    strategy[j] = (u, v, k + 1)
                                    currNode = v
                                if v == strategy[i][0]:
                                    break

                            rCounter[strategy[i][0]] = 1
                            break

        return strategy

            



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




    # ------------------------------------------------------------
    # THE REAL GRAPH SEARCH ALGORITHIM USING EVERYTHING FROM ABOVE
    # ------------------------------------------------------------

    minCost = np.inf
    bestStrategy = None

    for i in range(numOfTrees):
        root = random.randint(0,startNodes-1)
        T = computeRandomSpanningTree(G,root)
        treeStrategy = treeSearch(T,root)
        graphStrategy = transformStrategy(G,shadyEdges, treeStrategy)
        if graphStrategy[0][2] < minCost:
            minCost = graphStrategy[0][2]
            bestStrategy = graphStrategy

    return bestStrategy

