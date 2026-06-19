import numpy as np
import random
from Graph import Graph, Node, Edge

def graphSearch(G : Graph, shadyEdges, numOfTrees, availableRobots):
    #INPUT:
    # G: a Graph object repesenting the merged navigationgraph
    # numOfTrees: an integer which represents the number of evaluated trees
    # aerialSpeed: TO-DO
    # groundSpeed: TO-DO
    # aerialBattery: TO-DO
    # groundBattery: TO-DO

    #OUTPUT:
    #bestStrategy: A list containing the best strategy where each entry is in the form (source node,target node, amount of Robots)

    # ---------------------------------------------------
    # COMPUTING EDGE LABLES FOR THE TREE SEARCH THAT REPRESENT THE AMOUNT OF NEEDED ROBOTS
    # Remark: Lables represent the amount of robots needed for this path
    # --------------------------------------------------- 

    def computeLabels(T : Graph, root, parent):

        #Saving labels in a dictionary of the form: (x,y) | lambda((x,y))
        edgeLabelsRobotCost = {} 
        edgeLabelsEfficiency = {} 

        # Keep track of total prior and time in subtree
        totalTime = 0
        totalPrior = 0

        #Saving pi meaning all children of the root
        childLabels = [] 

        # Calculating lables recursive for children
        for y in T.adj[T.nodes[root]]:
            if y.idx != parent:
                subLabelsRobotCost, subLablesEfficiency, sumTime, sumPrior = computeLabels(T, y.idx, root)
                totalTime += sumTime
                totalPrior += sumPrior

                edgeLabelsEfficiency.update(subLablesEfficiency)

                edgeLabelsRobotCost.update(subLabelsRobotCost)
                childLabels.append(subLabelsRobotCost[(root, y.idx)])
  
        # Check if leaf
        if len(childLabels) == 0:
            edgeLabelsRobotCost[(parent, root)] = 1
               
            totalPrior += T.nodes[root].prior 
            totalTime +=  T.edges[(parent,root)].time
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
            totalPrior = T.nodes[root].prior + totalPrior
            if parent != None:
                totalTime = T.edges[(parent,root)].time + totalTime
            else:
                totalTime = 1
            edgeLabelsEfficiency[(parent, root)] = totalPrior / totalTime

        return edgeLabelsRobotCost, edgeLabelsEfficiency, totalTime, totalPrior
    
    
    # ---------------------------------------------------
    # CALCULATING A STRATEGY FOR TREES 
    # Note: A Strategy is safed in the format [(source node,target node, amount of Robots),...]
    # TO-DO: Compute both lables function. Define a formular for choosing a lable and then the rest should stay more or less the same.
    # ---------------------------------------------------
                    
    def treeSearch(T : Graph):

        labelsRobotCost, lablesEficiency, _ , _ = computeLabels(T,0, None)

        def explorePath(node):
            
            strategy = []
            currentLablesRobotCost = []
            currentLablesEfficiency = []
            combinedLables = []
    
            for y in T.adj[T.nodes[node]]:
                currentLablesRobotCost.append((labelsRobotCost[(node,y.idx)], y.idx))
                currentLablesEfficiency.append((lablesEficiency[(node, y.idx)], y.idx))
                combinedLables.append(lablesEficiency[(node, y.idx)], labelsRobotCost[(node,y.idx)], y.idx)
               

            if len(currentLablesRobotCost) > 0:
                combinedLables.sort(key=lambda x: x[0]) #Sorting the lables ascending
                currentLablesRobotCost.sort(key=lambda x: x[0], reverse = True)

                # Check if the max amount of robots is needed twice. Than the order of exploring can't effect the robot cost
                twice = False
                if currentLablesRobotCost[0][0] == currentLablesRobotCost[1][0]:
                    twice = True
                
                counter = 1
                if twice:
                    for (eff, robotsNeeded, y) in combinedLables: #Always explore the subtree with best efficiency
                        if counter == len(currentLablesRobotCost) and robotsNeeded != 1: #Don't allow slide moves! Note that they can only be neicessary in the last move
                            strategy.append((node,y,robotsNeeded-1))
                            strategy.append((node,y,1))
                            strategy.extend(explorePath(y))
                            strategy.append((y,node,robotsNeeded))
                        else:
                            strategy.append((node,y,robotsNeeded))
                            strategy.extend(explorePath(y))
                            strategy.append((y,node,robotsNeeded))    
                            counter = counter + 1       
                else:   
                    max = currentLablesRobotCost[0][0]
                    for k in range(len(combinedLables)):
                        (eff, robotsNeeded, y) = combinedLables[k]
                        if robotsNeeded == max and counter != len(currentLablesRobotCost): #Check if we want to enter the biggest subtree early
                            if eff > 2 * combinedLables[k+1]: #Score twice as good than we take the robot more
                                # adding the robot more on every move before
                                currNode = None
                                for j in range(len(strategy)):
                                    u, v, k = strategy[j]
                                    if u == currNode:
                                        strategy[j] = (u, v, k + 1)
                                        currNode = v
                                    if v == strategy[i][0]:
                                        break
                                # adding the strategy step
                                strategy.append((node,y,robotsNeeded))
                                strategy.extend(explorePath(y))
                                strategy.append((y,node,robotsNeeded))    
                                counter = counter + 1 
                                continue  
                            else:    
                                save = combinedLables[k] 
                                combinedLables[k] = combinedLables[k+1]
                                combinedLables[k+1] = save
                                (eff, robotsNeeded, y) = combinedLables[k]
            
                        if counter == len(currentLablesRobotCost) and robotsNeeded != 1: #Don't allow slide moves! Note that they can only be neicessary in the last move
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

        strategy = explorePath(0)

        robotCost = labelsRobotCost[(None, 0)]

        strategy.insert(0, (None,0,robotCost))

        return strategy
        
    
    # -------------------------------------------------------------------------------------
    # TRANSFORMING THE STRATEGY FROM TREE TO GRAPH
    # TO-DO: Make sure that slide moves are only prevented if neicessary. This could be done by adding in each strategy step a 0 or a 1 which represents if this is done to prevent a slide move. Or by transforming the strategy while determining it.
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
    # COMPUTING A SPANNING TREE WITH DFS
    # The generation uses a greedy choice for the node with the highest prior
    # ---------------------------------------------------

    def computeSpanningTree(G : Graph):
     
        T = Graph()
        for i in range(len(G.nodes)):
            T.add_node2(G.nodes[i])
        
        root = 0
        visited = set()

        def dfs(node):

            visited.add(node)

            neighbours = list(G.adj[G.nodes[node]])

            # Sorting the neighbours in respect to their prior
            neighbours.sort(key=lambda x: x.prior, reverse = True)
            
            for neighbour in neighbours:

                if neighbour.idx in visited:
                    continue

                T.add_edge(T.nodes[node], T.nodes[neighbour.idx])

                dfs(neighbour.idx)

        dfs(root)
        return T
    

    # ---------------------------------------------------
    # COMPUTING THE CLOSING EXISTS STRATEGY IF NOT ENOUGH ROBOTS ARE GIVEN
    # TO-DO: Implement method
    # ---------------------------------------------------

    def computeClosingExits(graphStrategy, G):
        print("Hello World!")


    # ---------------------------------------------------
    # COMPUTING EFFICENCY WITH WHICH TWO STRATEGYS ARE COMPARED
    # TO-DO: Implement method
    # ---------------------------------------------------
    def computeEfficiency(graphStrategy, G):
        print("Hello World!")



    # ------------------------------------------------------------
    # THE REAL GRAPH SEARCH ALGORITHIM USING EVERYTHING FROM ABOVE
    # TO-DO: Using a formular that evaluates the strategys. If there are not enough robots either use the old one if for this there are enough robots or calculate closing exists for the new one and save it
    # ------------------------------------------------------------

    minCost = np.inf
    minEff = np.inf
    bestStrategy = None

    for i in range(numOfTrees):
        T = computeSpanningTree(G)
        treeStrategy = treeSearch(T)
        graphStrategy = transformStrategy(G,shadyEdges, treeStrategy)
        #Check if strategy is perfomable. If not use closing exists strategy
        if graphStrategy[0][2] < availableRobots:
            # Compute efficiency score of strategy
            eff = computeEfficiency(graphStrategy)
            if eff < minEff:
                minEff = eff
                bestStrategy = graphStrategy
        else:
            graphStrategy, eff = computeClosingExits(graphStrategy, G)
            if eff < minEff:
                minEff = eff
                bestStrategy = graphStrategy

    return bestStrategy

