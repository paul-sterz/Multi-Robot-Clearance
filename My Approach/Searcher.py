import numpy as np
import math
import random
import sys
from Graph import Graph, Node, Edge
from itertools import combinations

def graphSearch(G : Graph, numOfTrees, availableRobots, startNodes):
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
    # COMPUTING EDGE LABLES FOR THE TREE SEARCH THAT REPRESENT THE AMOUNT OF NEEDED ROBOTS AND THE EFFICIENCY OF EACH SUBTREE
    # Remark: Lables represent the amount of robots needed for this path
    # --------------------------------------------------- 

    def computeLabels(T : Graph, root, parent):

        #Saving labels in a dictionary of the form: ((x,y) | lambda((x,y))
        edgeLabelsRobotCost = {} 
        edgeLabelsEfficiency = {}

        #All options to take multiple edges at once Saved in the Form (root | (children, robotcost, efficiency))
        multipleAtOnce = {} 

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
                subLabelsRobotCost, subLablesEfficiency, subMultipleAtOnce, sumTime, sumPrior = computeLabels(T, y.idx, root)
                totalTime += sumTime
                totalPrior += sumPrior

                subTreePriors[y.idx] = sumPrior
                subTreeTimes[y.idx] =  sumTime

                edgeLabelsEfficiency.update(subLablesEfficiency)
                multipleAtOnce.update(subMultipleAtOnce)
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

            #calculating the efficency when multiple subtrees can be explored at the same time
            multipleAtOnce[root] = []
            if len(childLabels) >= 2:
                for r in range(2,len(childLabels) + 1):
                    for perumtation in combinations(children,r):
                        robotCost = sum(edgeLabelsRobotCost[(root, child)]for child in  perumtation)

                        tmax = 0
                        for child in perumtation:
                            if subTreeTimes[child] > tmax:
                                tmax = subTreeTimes[child]

                        efficiency = sum(subTreePriors[child] for child in perumtation) / tmax
                        multipleAtOnce[root].append((perumtation,robotCost, efficiency))
                  

        return edgeLabelsRobotCost, edgeLabelsEfficiency, multipleAtOnce, totalTime, totalPrior
    
    
    # ---------------------------------------------------
    # CALCULATING A STRATEGY FOR TREES 
    # Note: A Strategy is safed in the format [(source node idx,target node idx, amount of Robots, time_Depature, time_Arrival),...]
    # ---------------------------------------------------
                    
    def treeSearch(T : Graph, root, availableRobots):

        clearance = True #Can a true graph clear be performed or is the amount of available robots to small for this tree? If not try to change the decisions towards robotcosts instead of efficiency

        labelsRobotCost, lablesEfficiency, multipleAtOnce , _ , _ = computeLabels(T,root, None)

        visited = [0] * len(T.nodes) #Saving which notes where already visited so we can stop backtracking if the graph is cleared and no unneicessary moves are done.
        visited[root] = 1

        finishTime = sys.maxsize

        robotCountPerNode = [0] * len(labelsRobotCost) #Saving how many robots are on each node at every time
        robotCountPerNode[root] = availableRobots

        def explorePath(node, enteringTime):
            
            strategy = []
            currentLablesRobotCost = []
            currentLablesEfficiency = []
            combinedLables = []
                
            for y in T.adj[T.nodes[node]]:
                currentLablesRobotCost.append((labelsRobotCost[(node,y.idx)], y.idx))
                currentLablesEfficiency.append((lablesEfficiency[(node, y.idx)], y.idx))                
                combinedLables.append((lablesEfficiency[(node, y.idx)], labelsRobotCost[(node,y.idx)], y.idx))

                                

            if len(currentLablesRobotCost) > 0: #Check if leaf because then there is nothing to do in this subtree

                if len(currentLablesRobotCost) == 1: #Check if there exists exactly one children, then there are no multiple strategys and efficiency does not matter
                    #---------------------------------------------------------------
                    # Case: There exists only one children
                    #---------------------------------------------------------------
                    (eff, robotsNeeded, y) = combinedLables[0]
                    if robotCountPerNode[node] >= robotsNeeded: #Are there enough robots to clear

                        

                        if robotCountPerNode[node] == robotsNeeded:
                            strategy.append(node, y, robotsNeeded - 1, enteringTime, enteringTime + T.edges[(node,y)].time)
                            strategy.append(node, y, 1, enteringTime + T.edges[(node,y)].time, enteringTime +2 * T.edges[(node,y)].time)
                            enteringTime += 2 * T.edges[(node,y)].time
                        else:
                            strategy.append(node, y, robotsNeeded, enteringTime + T.edges[(node,y)].time, enteringTime + T.edges[(node,y)].time)
                            enteringTime += T.edges[(node,y)].time

                        robotCountPerNode[node] -= robotsNeeded
                        robotCountPerNode[y] += robotsNeeded
                        visited[y] = 1
                        if visited == [1] * len(visited):
                            finishTime = enteringTime

                        subStrategy, feasible = explorePath(y, enteringTime)
                        if feasible == False: #Check if there are enough robots to explore the subtree
                            return strategy, False
                        
                        strategy.extend(subStrategy) #extending the Strategy with the subtree strategy
                        if visited != [1] * len(visited): #check if every node is cleared hence we are finished if not we have move our robots back to where they came from
                            strategy.append(y, node, robotsNeeded, strategy[len(strategy)-1][4],strategy[len(strategy)-1][4] + T.edges[(node,y)].time)
                            robotCountPerNode[y] -= robotsNeeded
                            robotCountPerNode[node] +=  robotsNeeded
                            enteringTime = strategy[len(strategy)-1][4]
                
                        return strategy, True
                    else:
                        # --------------TODO-------------------is it possible to get more robots from elsewhere
                        return strategy, False

                else: #There exits at least 2 children hence multiple strategys exist and an order for efficency has to be regared

                    unExplored = [e[2] for e in combinedLables] #Saving all children / Subtrees that have to be cleared and are still unexplored

                    combinedLables.sort(key=lambda x: x[0]) #Sorting the lables ascending
                    currentLablesRobotCost.sort(key=lambda x: x[0], reverse = True)
                    twice = False # Check if the max amount of robots is needed twice. Than the order of exploring can't effect the robot cost
                    if len(currentLablesRobotCost) > 1: #Is there more than one children
                        if currentLablesRobotCost[0][0] == currentLablesRobotCost[1][0]:
                            twice = True
                    else:   #If there is only one children there is no order
                            twice = True 


                    multiples = multipleAtOnce[node]
                    lastOptions = multiples #Multiple Move Strategys that are only performable as the last move, meaning robotcost = availableRobots
                    for triplets in multiples: # Deleting all Multiple Strategys which are impossible to perform, meaning robotcost > availableRobots
                        if triplets[1] > availableRobots[node]:
                            multiples.remove(triplets)
                            lastOptions.remove(triplets)
                        elif triplets[1] == availableRobots[node]:
                            multiples.remove(triplets)

                    multiples.sort(key=lambda x: x[2])
                    lastOptions.sort(key = lambda x: x[2])


                    while len(unExplored) > 0: #as long as there exists unExplored children
                        exists = False
                        #--------------------------------------------------------
                        # Check if there is multiple last move, if yes do this since it always improves efficiency               
                        #--------------------------------------------------------
                        for (candidateLast, robotsNeededLast, effLast)  in lastOptions:
                            if candidateLast == unExplored:

                                # Note that here a backtracking deciosion is opend. Furthermore the faster subtrees will be traceable robots while they wait
                                exists = True

                                # is a slide move acctually neicessary
                                if robotsNeededLast == robotCountPerNode[enteringTime][node]: slideMove = True 
                                else: slideMove = False

                                if slideMove == True:
                                    #determine the longest edge how the slider has to wait and the shortest edge where can go to make the slide move as fast as possible
                                    largest = 0
                                    cheapest = sys.maxsize
                                    for y in candidateLast:
                                        if largest < T.edges[(node,y)].time:
                                            largest =  T.edges[(node,y)].time
                                        if cheapest > T.edges[(node,y)].time:
                                            cheapest =  T.edges[(node,y)].time
                                            slideMoveGo = y

                                welcomeBackTime = [0] * len(candidateLast)
                                counter = 0
                                for y in candidateLast:

                                    for lable in combinedLables:
                                        if lable[2] == y:
                                            robotsNeeded = lable[1]

                                    if slideMove == True and slideMoveGo == y:
                                        strategy.append(node, y, robotsNeeded - 1, enteringTime, enteringTime + T.edges[(node,y)].time)
                                        #Only send the slide move preventer when the longest edge was crossed so after enteringTime + largest
                                        strategy.append(node, y, 1, enteringTime + largest, enteringTime + largest + T.edges[(node,y)].time)
                                        arrvTime = enteringTime + largest + T.edges[(node,y)].time
                                    else:
                                        strategy.append(node, y, robotsNeeded, enteringTime, enteringTime + T.edges[(node,y)].time)
                                        arrvTime = enteringTime + T.edges[(node,y)].time

                                    

                                    visited[y] = 1
                                    if visited == [1] * len(visited):
                                        finishTime = strategy[len(strategy)- 1][4]

                                    robotCountPerNode[node] -= robotsNeeded
                                    robotCountPerNode[y] += robotsNeeded 

                                    subStrategy, feasible = explorePath(y, arrvTime)

                                    #Check if during the run not enough robots are available hence this decision has to backtracked and removed
                                    if feasible == False:
                                        exists = False

                                        for i in range(0, len(strategy)):
                                            if (strategy[len(strategy)- 1][0] not in candidateLast) and (strategy[len(strategy)- 1][1] == node): 
                                                break
                                            strategy.pop()
                                        break

                                    strategy.extend(subStrategy)
                                    if visited != [1] * len(visited):
                                        #Note that these backtrack moves might go over t_finish but they are cut of outside of this method
                                        strategy.append(y,node, robotsNeeded, strategy[len(strategy)-1][4], strategy[len(strategy)-1][4] + T.edges[(node,y)].time)
                                        robotCountPerNode[node] += robotsNeeded
                                        robotCountPerNode[y] -= robotsNeeded 

                                    # Save when the robots return from this inner path
                                    welcomeBackTime[counter] = strategy[len(strategy)- 1][4]

                                # Continue with the next move when all robots have returned from there inner paths
                                enteringTime = max(welcomeBackTime)

                        #If we did a last move we are have explored all children hence are finished
                        if exists == True: 
                            break

                        counterSingle = 0
                        counterMultiple = 0

                        #find next best feasible single option
                        (effSingle, robotsNeededSingle, candidateSingle) = combinedLables[counterSingle]
                        while candidateSingle not in unExplored:
                            counterSingle += 1
                            (effSingle, robotsNeededSingle, candidateSingle) = combinedLables[counterSingle]

                        #find next best feasible multiple option if it exists
                        alreadyExplored = True
                        counterMultiple -= 1
                        while alreadyExplored == True and counterMultiple < len(multiples):
                            counterMultiple += 1
                            (candidateMultiple, robotsNeededMultiple, effMultiple) = multiples[counterMultiple]
                            alreadyExplored = False
                            for vertex in candidateMultiple:
                                if vertex not in unExplored:
                                    alreadyExplored = True

                        if alreadyExplored == True or (effSingle > effMultiple and (candidateSingle not in candidateMultiple or effSingle * 1/2 > effMultiple )) : #This implies that the single move is the better choice or that no multiple move exists
                            #--------------------------------------------------------
                            # A Single Move is the best one, Note that we have to do a case distinction here if we need slide moves & and this opens a decision or not
                            #--------------------------------------------------------
                                counterSingle += 1

                                if robotCountPerNode[node] > robotsNeededSingle: #No slide moves have to be considered
                                    strategy.append(node, candidateSingle, robotsNeededSingle, enteringTime, enteringTime + T.edges[(node,candidateSingle)].time)
                                    robotCountPerNode[node] -= robotsNeededSingle
                                    robotCountPerNode[candidateSingle] += robotsNeededSingle

                                    visited[candidateSingle] = 1
                                    if visited == [1] * len(visited):
                                        finishTime = strategy[len(strategy)- 1][4]
                                    unExplored = list(set(unExplored) - set(candidateSingle))
                                    subStrategy, feasible = explorePath(candidateSingle, strategy[len(strategy)- 1][4])
                                    if feasible == False:
                                        # Case Distinction if a decision was opend, meaning did we go early in the biggest path and this made a difference
                                        if twice or currentLablesRobotCost[0][0] != robotsNeededSingle:
                                            return strategy, False
                                        else:
                                             #Swap this with the last place and rearange accordingly, to ensure max subtree is taken at last
                                            counterSingle -= 1
                                            for i in range(counterSingle,len(combinedLables)-1):
                                                save = combinedLables[i]
                                                combinedLables[i] = combinedLables[i+1]
                                                combinedLables[i + 1] = save
                                            break  

                                    strategy.extend(subStrategy)
                                    if visited != [1] * len(visited):
                                        strategy.append(candidateSingle, node, robotsNeededSingle,strategy[len(strategy)- 1][4], strategy[len(strategy)- 1][4] + T.edges[(node, candidateSingle)].time ) 
                                        robotCountPerNode[node] += robotsNeededSingle
                                        robotCountPerNode[candidateSingle] -=  robotsNeededSingle
                                        enteringTime = strategy[len(strategy)- 1][4]
                                elif robotCountPerNode[node] == robotsNeededSingle and len(list(set(unExplored) - candidateSingle)) == 0: #Check if "==" stands, this is only feasible if we are at the last move 
                                    strategy.append(node, candidateSingle, robotsNeededSingle - 1, enteringTime, enteringTime + T.edges[(node,candidateSingle)].time)
                                    strategy.append(node, candidateSingle, 1, enteringTime + T.edges[(node,candidateSingle)].time, enteringTime + 2 * T.edges[(node,candidateSingle)].time)
                                    robotCountPerNode[node] -= robotsNeededSingle
                                    robotCountPerNode[candidateSingle] += robotsNeededSingle

                                    visited[candidateSingle] = 1
                                    if visited == [1] * len(visited):
                                        finishTime = strategy[len(strategy)- 1][4]
                                    unExplored = list(set(unExplored) - set(candidateSingle))
                                    subStrategy, feasible = explorePath(candidateSingle, strategy[len(strategy)- 1][4])
                                    if feasible == False:
                                        # Case Distinction if a decision was opend, meaning did we go early in the biggest path and this made a difference
                                        if twice or currentLablesRobotCost[0][0] != robotsNeededSingle:
                                            return strategy, False
                                        else:
                                            #Swap this with the last place and rearange accordingly, to ensure max subtree is taken at last
                                            counterSingle -= 1
                                            for i in range(counterSingle,len(combinedLables)-1):
                                                save = combinedLables[i]
                                                combinedLables[i] = combinedLables[i+1]
                                                combinedLables[i + 1] = save
                                            break
                                            
                                    strategy.extend(subStrategy)
                                    if visited != [1] * len(visited):
                                        strategy.append(candidateSingle, node, robotsNeededSingle,strategy[len(strategy)- 1][4], strategy[len(strategy)- 1][4] + T.edges[(node, candidateSingle)].time ) 
                                        robotCountPerNode[node] += robotsNeededSingle
                                        robotCountPerNode[candidateSingle] -=  robotsNeededSingle
                                        enteringTime = strategy[len(strategy)- 1][4]

                                else:
                                    # --------------TODO-------------------is it possible to get more robots from elsewhere
                                    return strategy, False

                        else: #Take the multiple move
                            #--------------------------------------------------------
                            # A Multiple Move that does not finish all children is the best one            
                            #--------------------------------------------------------
                            #Note that no slide move prevention is neicessary since we need at max availablerobots -1
                            counterMultiple += 1 
                            unExplored = list(set(unExplored) - set(candidateMultiple))

                            welcomeBackTime = [0] * len(candidateMultiple)
                            counter = 0
                            for y in candidateMultiple:
                                for label in combinedLables:
                                    if lable[2] == y:
                                        robotsNeeded = lable[1]

                                strategy.append(node, y, robotsNeeded, enteringTime, enteringTime + T.edges[(node,y)].time)
                                visited[y] = 1
                                if visited == [1] * len(visited):
                                    finishTime = strategy[len(strategy)- 1][4]

                                robotCountPerNode[node] -= robotsNeeded
                                robotCountPerNode[y] += robotsNeeded

                                subStrategy, feasible = explorePath(y, strategy[len(strategy)- 1][4])

                                #Check if during the run not enough robots are available if yes this decision has to be backtracked
                                if feasible == False:
                                    #Note that this strategy might be feasible in a later time, but this is not implemented here
                                    unExplored = list(set(unExplored) + set(candidateMultiple))

                                    for i in range(0,len(strategy)):
                                        if (strategy[len(strategy)- 1][0] not in candidateMultiple) and (strategy[len(strategy)- 1][1] == node):
                                            break
                                        strategy.pop()
                                    break


                                strategy.extend(subStrategy)
                                #Note that we always have to backtrack since this can't be the last move
                                strategy.append(y,node,robotsNeeded,strategy[len(strategy)- 1][4],strategy[len(strategy)- 1][4] + T.edges[(node,y)].time)

                                robotCountPerNode[node] += robotsNeeded
                                robotCountPerNode[y] -= robotsNeeded
                                # Save when the robots return from this inner path
                                welcomeBackTime[counter] = strategy[len(strategy)- 1][4]

                            # Continue with the next move when all robots have returned from there inner paths
                            enteringTime = max(welcomeBackTime)

            return strategy, clearance

        strategy, clearance = explorePath(root, 0)

        if clearance == False:
            strategy = computeClosingExits()


        strategy.insert(0, (None,root,availableRobots))
        #Cut of strategy moves that go over t_finish, meaning unneicessary backtrack moves
        for (source, target, amountofR, depTime, arrTime) in strategy:
            if arrTime > finishTime:
                strategy.remove((source, target, amountofR, depTime, arrTime))
        return strategy
        
    
    # -------------------------------------------------------------------------------------
    # TRANSFORMING THE STRATEGY FROM TREE TO GRAPH
    # -------------------------------------------------------------------------------------

    def transformStrategy(G : Graph, strategy):

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

        treeStrategy = treeSearch(T,root, availableRobots)
        graphStrategy = transformStrategy(G, treeStrategy) #Will be merged in treeSearch
        #Check if strategy is perfomable. If not use closing exists strategy

        #This will also have to be altered since i will always use all of the robots.
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

