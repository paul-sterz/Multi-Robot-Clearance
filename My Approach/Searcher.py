import numpy as np
import random

def graphSearch(graph, root, numOfTrees, priors , availableRobots):
    #INPUT:
    # nodes: a list containing all nodes of the graph
    # edges: a dictionary with elements of the form {(u,v) : (robottype, travel time)}
    # root: index of the root node
    # numOfTrees: an integer which represents the number of evaluated trees
    # priors: a list containing a target detection probability for each node
    # aerialSpeed: an integer value that represents the speed of the aerial robots
    # groundSpeed: an integer value that represents the speed of the ground robots
    # aerialBattery: unsure
    # groundBattery: unsure
    # trajectoryTimes: a dictionary containg all edges

    #OUTPUT:
    #bestStrategy: A list containing the best strategy where each entry is in the form (source node,target node, amount of Robots)

    # ---------------------------------------------------
    # COMPUTING EDGE LABLES FOR THE TREE SEARCH THAT REPRESENT THE AMOUNT OF NEEDED ROBOTS
    # Remark: Lables represent the amount of robots needed for this path
    # ---------------------------------------------------

    def computeLabels(nodes, edges, root, parent, priors):

        #Saving labels in a dictionary of the form: (x,y) | lambda((x,y))
        edgeLabelsRobotCost = {} 
        edgeLabelsEfficiency = {} 
        sumI = 0
        sumT = 0

        #Saving pi meaning all nodes to the children of root
        childLabels = [] 

        # Calculating lables recursive for children
        for (x, y, a, b) in edges:

            if x == root and y != parent:
                
                #Computing all edgelables from y 
                subLabelsRobotCost, subLablesEfficiency, sumI, sumT = computeLabels(nodes, edges, y, root, priors)

                edgeLabelsRobotCost.update(subLabelsRobotCost)
                edgeLabelsEfficiency.update(subLablesEfficiency)

                childLabels.append(subLabelsRobotCost[(root, y)])

        # Check if leaf
        if len(childLabels) == 0:

            edgeLabelsRobotCost[(parent, root)] = 1

            edgeLabelsEfficiency[(parent, root)] = priors[root] / edges[(parent,root)]

        else:

            #formula from the paper for calculating the amount of robots needed 
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



        return edgeLabelsRobotCost, edgeLabelsEfficiency     
    
    
    # ---------------------------------------------------
    # CALCULATING A STRATEGY FOR TREES 
    # Note: A Strategy is safed in the format [(source node,target node, amount of Robots),...]
    # TO-DO: Compute both lables function. Define a formular for choosing a lable and then the rest should stay more or less the same.
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
        
    
    # -------------------------------------------------------------------------------------
    # TRANSFORMING THE STRATEGY FROM TREE TO GRAPH
    # TO-DO: Make sure that slide moves are only prevented if neicessary. This could be done by adding in each strategy step a 0 or a 1 which represents if this is done to prevent a slide move. Or by transforming the strategy while determining it.
    # -------------------------------------------------------------------------------------

    def transformStrategy(nodes, GraphEdges, strategy):
        contaminationArea = set(nodes)
        rCounter = [0] * len(nodes)
        for i in range(len(strategy)):
            if strategy[i][1] in contaminationArea:
                contaminationArea.remove(strategy[i][1])
                
            if strategy[i][0] != None:
                rCounter[strategy[i][0]] = rCounter[strategy[i][0]] - strategy[i][2]
            
            rCounter[strategy[i][1]] = rCounter[strategy[i][1]] + strategy[i][2]

            if strategy[i][0] != None and rCounter[strategy[i][0]] == 0: #Are there still any robots left on the last node?
                for (u,v) in GraphEdges: 
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
    # COMPUTING A SPANNING TREE WITH DFS
    # ---------------------------------------------------

    def computeRandomSpanningTree(nodes, edges, root, priors):
        visited = set()
        treeEdges = []

        def dfs(node):

            visited.add(node)

            neighbours = {}

            for (u,v) in edges:
                if u == node:
                    neighbours[v] = priors(v)
                

            sortedNeighbours = sorted(neighbours.items(), key=lambda item: item[1], reverse=True)

            for neighbour, prio in sortedNeighbours:

                if neighbour in visited:
                    continue

                treeEdges.append((node, neighbour))

                dfs(neighbour)

        dfs(root)
        return nodes, treeEdges
    

    # ---------------------------------------------------
    # COMPUTING THE CLOSING EXISTS STRATEGY IF NOT ENOUGH ROBOTS ARE GIVEN
    # TO-DO: Implement method
    # ---------------------------------------------------

    def computeClosingExits(graphStrategy, nodes, edges, priors):
        print("Hello World!")



    # ------------------------------------------------------------
    # THE REAL GRAPH SEARCH ALGORITHIM USING EVERYTHING FROM ABOVE
    # TO-DO: Using a formular that evaluates the strategys. If there are not enough robots either use the old one if for this there are enough robots or calculate closing exists for the new one and save it
    # ------------------------------------------------------------

    minCost = np.inf
    minEff = np.inf
    bestStrategy = None

    for i in range(numOfTrees):
        Vi, Ei = computeRandomSpanningTree(nodes,edges,root)
        treeStrategy = treeSearch(Vi,Ei, root)
        graphStrategy = transformStrategy(nodes, edges, treeStrategy)
        if graphStrategy[0][2] < availableRobots:
            #Check if strategy is perfomable. If not use closing exists strategy
            if graphStrategy[0][2] < minEff:
                minCost = treeStrategy[0][2]
                bestStrategy = treeStrategy
        else:
            graphStrategy, eff = computeClosingExits(graphStrategy, nodes, edges, priors)
            if eff < minEff:
                minEff = eff
                bestStrategy = graphStrategy
            
    return bestStrategy

