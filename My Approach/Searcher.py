import numpy as np
import math
import random
from Graph import Graph, Node, Edge

def graphSearch(G : Graph, shadyEdges, numOfTrees, availableRobots, startNodes):
    #INPUT:
    # G: a Graph object repesenting the merged navigationgraph
    # numOfTrees: an integer which represents the number of evaluated trees
    # availableRobots: an integer representing the amount of robots available
    # startNodes:  an integer representing that all nodes from 0 to startNode-1 are valid startNodes for our Algorithim
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
    # CALCULATING A STRATEGY FOR TREES 
    # Note: A Strategy is safed in the format [(source node idx,target node idx, amount of Robots, timestamp),...]
    # ---------------------------------------------------
                    
    def treeSearch(T : Graph, root):

        labelsRobotCost, lablesEfficiency, _ , _ = computeLabels(T,root, None)

        visited = [0] * len(T.nodes)
        visited[root] = 1

        def explorePath(node):
            
            strategy = []
            currentLablesRobotCost = []
            currentLablesEfficiency = []
            combinedLables = []
    
            for y in T.adj[T.nodes[node]]:
                currentLablesRobotCost.append((labelsRobotCost[(node,y.idx)], y.idx))
                currentLablesEfficiency.append((lablesEfficiency[(node, y.idx)], y.idx))
                combinedLables.append((lablesEfficiency[(node, y.idx)], labelsRobotCost[(node,y.idx)], y.idx))
               

            if len(currentLablesRobotCost) > 0:
                combinedLables.sort(key=lambda x: x[0]) #Sorting the lables ascending
                currentLablesRobotCost.sort(key=lambda x: x[0], reverse = True)

                # Check if the max amount of robots is needed twice. Than the order of exploring can't effect the robot cost
                twice = False
                if len(currentLablesRobotCost) > 1: #Is there more than one children
                    if currentLablesRobotCost[0][0] == currentLablesRobotCost[1][0]:
                        twice = True
                else:   #If there is only one children there is no order
                    twice = True 
                counter = 1
                if twice:
                    for (eff, robotsNeeded, y) in combinedLables: #Always explore the subtree with best efficiency
                        if counter == len(currentLablesRobotCost) and robotsNeeded != 1: #Don't allow slide moves! Note that they can only be neicessary in the last move
                            if len(currentLablesRobotCost) == 1: 
                                strategy.append((node,y,robotsNeeded - 1))
                                strategy.append((node,y,1))
                            else: # If same amount of robot appears twice than we have one robot in spare hence no slide move has to be prevented
                                strategy.append((node,y,robotsNeeded))

                            visited[y] = 1
                            strategy.extend(explorePath(y))
                            # Check if there is need for backtracking or if everything is visited so we are finished
                            if visited != [1] * len(visited):
                                strategy.append((y,node,robotsNeeded))
                        else:
                            strategy.append((node,y,robotsNeeded))
                            visited[y] = 1
                            strategy.extend(explorePath(y))
                            # Check if there is need for backtracking or if everything is visited so we are finished
                            if visited != [1] * len(visited):
                                strategy.append((y,node,robotsNeeded))    
                            counter = counter + 1       
                else:   
                    maximum = currentLablesRobotCost[0][0]
                    for k in range(len(combinedLables)):
                        (eff, robotsNeeded, y) = combinedLables[k]
                        if robotsNeeded == maximum and counter != len(currentLablesRobotCost): #Check if we want to enter the biggest subtree early
                            if eff > 2 * combinedLables[k+1][0] : #Score twice as good than we take the robot more
                                # adding the robot more on every move before
                                currNode = None
                                for j in range(len(strategy)):
                                    u, v, k = strategy[j]
                                    if u == currNode:
                                        strategy[j] = (u, v, k + 1)
                                        currNode = v
                                    if v == node:
                                        break
                                # adding the strategy step
                                strategy.append((node,y,robotsNeeded))
                                visited[y] = 1
                                strategy.extend(explorePath(y))
                                # Check if there is need for backtracking or if everything is visited so we are finished
                                if visited != [1] * len(visited):
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
                            visited[y] = 1
                            strategy.append((node,y,1))
                            strategy.extend(explorePath(y))
                            # Check if there is need for backtracking or if everything is visited so we are finished
                            if visited != [1] * len(visited):
                                strategy.append((y,node,robotsNeeded))
                        else:
                            strategy.append((node,y,robotsNeeded))
                            visited[y] = 1
                            strategy.extend(explorePath(y))
                            # Check if there is need for backtracking or if everything is visited so we are finished
                            if visited != [1] * len(visited):
                                strategy.append((y,node,robotsNeeded))    
                            counter = counter + 1        
            return strategy

        strategy = explorePath(root)
        robotCost = labelsRobotCost[(None, root)]

        strategy.insert(0, (None,root,robotCost))

        return strategy
        
    
    # -------------------------------------------------------------------------------------
    # TRANSFORMING THE STRATEGY FROM TREE TO GRAPH
    # -------------------------------------------------------------------------------------

    def transformStrategy(G : Graph, shadyEdges, strategy):

        contaminationArea = set()
        for i in range(len(G.nodes)):
            contaminationArea.add(i)

        rCounter = [0] * len(G.nodes)

        i = 0

        while i < len(strategy):

            if strategy[i][1] in contaminationArea:
                contaminationArea.remove(strategy[i][1])
                
            if strategy[i][0] != None:
                rCounter[strategy[i][0]] = rCounter[strategy[i][0]] - strategy[i][2]
            
            rCounter[strategy[i][1]] = rCounter[strategy[i][1]] + strategy[i][2]
            

            if strategy[i][0] != None and rCounter[strategy[i][0]] == 0: #Are there still any robots left on the last node?

                for (u,v) in G.edges.keys(): #Note that only regular edges have to be checked since shady ones are included in an regular edge
                    if u in contaminationArea and v == strategy[i][0]: #Is there an edge that leads to recontamination?

                        currNode = None
                        for j in range(i):
                            u, v, k = strategy[j]
                            if u == currNode:
                                strategy[j] = (u, v, k + 1)
                                currNode = v
                            if v == strategy[i][0]:
                                break

                        if strategy[i][2] == 1 and strategy[i-1][0] == strategy[i][0] and strategy[i-1][1] == strategy[i][1]: #Check if there is a slide move which now gets unnecessary
                            strategy.pop(i) #delete the slide move
                            strategy[i-1] = (strategy[i-1][0],strategy[i-1][1] ,strategy[i-1][2] + 1)
                            i -= 1
                        rCounter[strategy[i][0]] = 1
                        break
            i += 1

        return strategy
            

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

                T.add_edge(T.nodes[node], T.nodes[neighbour.idx], G.edges[(node,neighbour.idx)].time, G.edges[(node,neighbour.idx)].robotType)

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
    # ---------------------------------------------------
    def computeExpTime(graphStrategy, G: Graph):
        visited = set()
        eff = 0
        time = 0
        for (source, target, robotsneeded) in graphStrategy:
            if len(visited) == len(G.nodes): #Only count time until graph is cleared and not the walking back to root
                break
            if source != None:
                #Note that for walking backwords a shady edge can be used which is not saved in the edges
                if (source,target) in G.edges: #is the used edge regular?
                    time += G.edges[(source,target)].time
                else: #The other direction has to be regular edge so just use this time. TO-DO: Improve this by using the real shady edges
                    time += G.edges[(target,source)].time
            else:
                time += 1

            if target not in visited:
                visited.add(target)
                eff += G.nodes[target].prior * time # calculate the expected value of detection time

        return eff





    # ------------------------------------------------------------
    # THE REAL GRAPH SEARCH ALGORITHIM USING EVERYTHING FROM ABOVE
    # TO-DO: Using a formular that evaluates the strategys. If there are not enough robots either use the old one if for this there are enough robots or calculate closing exists for the new one and save it
    # ------------------------------------------------------------

    minExpTime = np.inf
    bestStrategy = None
    bestTree = None
    counter = [0] * startNodes

    for i in range(numOfTrees):

        root = random.randint(0,startNodes-1)

        if counter[root] == 0:
            T = computeGreedySpanningTree(G,root)
            counter[root] += 1
        else:
            T = computeRandomSpanningTree(G,root)

        treeStrategy = treeSearch(T,root)
        graphStrategy = transformStrategy(G,shadyEdges, treeStrategy)
        #Check if strategy is perfomable. If not use closing exists strategy
        if graphStrategy[0][2] <= availableRobots:
            # Compute efficiency score of strategy
            expTime = computeExpTime(graphStrategy, G)
            if expTime < minExpTime:
                minExpTime = expTime
                bestStrategy = graphStrategy
                bestTree = T
        else:
            graphStrategy, expTime = computeClosingExits(graphStrategy, G)
            if expTime < minExpTime:
                minExpTime = expTime
                bestStrategy = graphStrategy
                bestTree = T

    return bestStrategy, bestTree

