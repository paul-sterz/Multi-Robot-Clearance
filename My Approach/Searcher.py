import numpy as np
import math
import random
import sys
from Graph import Graph, Node, Edge
from itertools import combinations
from TrajectoryPlanning import aStar
import copy


def graphSearch(G : Graph, numOfTrees, availableRobots, startNodes, obstacles, distanceMap, alpha):
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
    # CALCULATING A STRATEGY FOR TREES AND TRANSFORMING IT TO THE GRAPH
    # Note: A Strategy is safed in the format [(source node idx,target node idx, amount of Robots, time_Depature, time_Arrival),...]
    # ---------------------------------------------------
                    
    def treeSearch(T : Graph, root, availableRobots, G : Graph):
        #Note that root is the index of the starting node and not the starting node itself

        clearance = True #Can a true graph clear be performed or is the amount of available robots to small for this tree? If not try to change the decisions towards robotcosts instead of efficiency

        labelsRobotCost, lablesEfficiency, multipleAtOnce , _ , _ = computeLabels(T,root, None)

        #Saving which notes where already visited so we can stop backtracking if the graph is cleared and no unneicessary moves are done.
        visited = [(0,0)] * len(T.nodes) #Format: visited[node.idx] = (0 or 1, visitingTime) 
        visited[root] = (1,0)

        guards = [] #Guards that protect recontamination from Graph edges, Format: (node idx, list of enemy idx's, t_begin, minimum guard Duration)
        flags = [] #Flags that are robots with no current task that can be used to helpout if robots on current node is to low  Format: (node idx, amount, t_begin)

        finishTime = sys.maxsize

        robotCountPerNode = [0] * len(labelsRobotCost) #Saving how many robots are on each node at every time
        robotCountPerNode[root] = availableRobots

        def explorePath(node, enteringTime):

            #Safestades, so that if something goes wrong we can backtrack to the old states
            nonlocal finishTime
            nonlocal guards
            nonlocal flags

            safestadeVisited = visited.copy()
            safestadeGuards = copy.deepcopy(guards)
            safestadeFlags = copy.deepcopy(flags)
            safestadeRobotCount = robotCountPerNode.copy()

            
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
                    flagsCreated = False
                    if robotCountPerNode[node] > robotsNeeded : #Are there enough robots to clear
                        # Note that (robotsNeeded == robotCountPerNode[node] and robotsNeeded == 1) could only happen if we are in the root
                        
                        slideMove = False

                        if robotCountPerNode[node] == robotsNeeded: #This means that we leave the node completly -> Slide move and Contamination Checks are neicessary

                            #Checking if an Graph Edge leads to recontamination
                            contamination = False
                            visitedContamination = False
                            visitedContaminationIdx = []
                            enemys = []

                            for neighbour in G.adj[G.nodes[node]]:
                                if neighbour.idx != y and visited[neighbour.idx][0] == 0:
                                    contamination = True
                                    enemys.append(neighbour.idx)
                                elif neighbour.idx != y and visited[neighbour.idx][0] == 1 and visited[neighbour.idx][1] > enteringTime:
                                    visitedContamination = True
                                    visitedContaminationIdx.append(neighbour.idx)

                            if contamination or visitedContamination:
                                                            
                                if len(flags)== 0:
                                    return strategy, False
                                else:
                                    shortest = sys.maxsize
                                    vertex = 0
                                    counter = 0
                                    bestIdx = 0
                                    found = False

                                    for triplet in flags:
                                        if triplet[2] <= enteringTime:
                                            #Case: Flag_Begin before Entering Time
                                            #Assumption the robot needs 1 second for each cell travelld
                                            path = aStar(triplet[0], node,obstacles, distanceMap, alpha)
                                            if path == None: #is there a vaild path or not?
                                                continue
                                            dist = len(path) - 1
                                            if dist < shortest:
                                                found = True
                                                shortest = dist
                                                vertex = triplet[0]
                                                bestIdx = counter
                                            counter += 1

                                    if found == False:
                                        #Nothing was changed untill now so no need for going to the safestades
                                        return strategy, False

                                    #Walk the robot from the flag to the robot  
                                    strategy.append((flags[bestIdx][0], node, 1,   flags[bestIdx][2],   flags[bestIdx][2] +shortest))

                                    #Make sure that the robots wait until the guard arrives
                                    if enteringTime < flags[bestIdx][2] + shortest:
                                        enteringTime = flags[bestIdx][2] + shortest


                                    if contamination:
                                        maximum = -1
                                        for index in visitedContaminationIdx:
                                            if maximum < visited[index][1]:
                                                maximum = visited[index][1]

                                        #Guard has to stay at least untill slide move is prevented and all visited contaminations are actually visited
                                        guards.append((node, enemys, flags[bestIdx][2] + shortest, max(enteringTime + T.edges[(node,y)].time, maximum)))
                                    elif visitedContamination:
                                        maximum = -1
                                        for index in visitedContaminationIdx:
                                            if maximum < visited[index][1]:
                                                maximum = visited[index][1]

                                        flags.append((node, 1, max(maximum, enteringTime + T.edges[(node,y)].time )))

                                    strategy.append((node, y, robotsNeeded, enteringTime, enteringTime + T.edges[(node,y)].time))

                                    enteringTime += T.edges[(node,y)].time

                                    flags[bestIdx][1] -= 1

                                    if flags[bestIdx][1] == 0:
                                        flags.pop(bestIdx)

                            else:
                                #Case no edge leads to recontamination, so only a slide move has to be made
                                slideMove = True

                                strategy.append((node, y, robotsNeeded - 1, enteringTime, enteringTime + T.edges[(node,y)].time))
                                strategy.append((node, y, 1, enteringTime + T.edges[(node,y)].time, enteringTime +2 * T.edges[(node,y)].time))
                                enteringTime += 2 * T.edges[(node,y)].time

                        else:
                            #Case we have more than enough robots hence slide move prevention and contamination checks are unneicessary
                            #furthermore we have available robots - needed robots as flags but one of them has to stay so that the slide move is prevented
                            #flag that prevents the slidemove

                            contamination = False
                            visitedContamination = False
                            visitedContaminationIdx = []
                            enemys = []

                            for neighbour in G.adj[G.nodes[node]]:
                                if neighbour.idx != y and visited[neighbour.idx][0] == 0:
                                    contamination = True
                                    enemys.append(neighbour.idx)
                                elif neighbour.idx != y and visited[neighbour.idx][0] == 1 and visited[neighbour.idx][1] > enteringTime:
                                    visitedContamination = True
                                    visitedContaminationIdx.append(neighbour.idx)

                            #One robot has to stay at least untill slide moves and recontamination is no danger any more
                            if contamination:
                                maximum = -1
                                for index in visitedContaminationIdx:
                                    if maximum < visited[index][1]:
                                        maximum = visited[index][1]

                                #Guard has to stay at least untill slide move is prevented and all visited contaminations are actually visited
                                guards.append((node, enemys, flags[bestIdx][2] + shortest, max(enteringTime + T.edges[(node,y)].time, maximum)))
                            elif visitedContamination:
                                maximum = -1
                                for index in visitedContaminationIdx:
                                    if maximum < visited[index][1]:
                                        maximum = visited[index][1]
                                flags.append((node,1,max(enteringTime + T.edges[(node,y)].time, maximum)))
                            else:
                                flags.append((node,1,enteringTime + T.edges[(node,y)].time))

                            #We have more than one spare robot, these can all be used
                            if robotCountPerNode[node] - robotsNeeded > 1:
                                flags.append((node, robotCountPerNode[node] - robotsNeeded - 1, enteringTime))

                            strategy.append((node, y, robotsNeeded, enteringTime, enteringTime + T.edges[(node,y)].time))
                            enteringTime += T.edges[(node,y)].time
                            flagsCreated = True

                        robotCountPerNode[node] = 0
                        robotCountPerNode[y] += robotsNeeded
                            
                        if visited[y][0] == 0:
                            #Note that enteringTime represents the time when all robots arrive at the node but with a slide move the node is actually visited earlier
                            if slideMove == True:
                                enteringTime -= T.edges[(node,y)].time

                            visited[y] = (1,enteringTime)

                            #Convert guard into flag
                            for waechter in guards:
                                if y in waechter[1]:
                                    waechter[1].remove(y)
                                    if len(waechter[1]) == 0:
                                        #Guard turns into a flag as normal
                                        # guards[3] saves how long a guard needs at least to stay for slide move prevention
                                        flags.append((waechter[0], 1, max(enteringTime, waechter[3])))
                                        guards.remove(waechter)

                            if slideMove == True:
                                enteringTime += T.edges[(node,y)].time



                        finish = True
                        for tuple in visited:
                            if tuple[0] == 0:
                                finish = False
                                break

                        #Note that Slide moves can not change the finish time since the finishing move will only need one robot hence no slide move prevention is needed
                        if finish == True and finishTime == sys.maxsize:
                            finishTime = enteringTime

                        subStrategy, feasible = explorePath(y, enteringTime)
                        if feasible == False: #Check if there are enough robots to explore the subtree, if not go back to the safestade
                            visited = safestadeVisited
                            flags = safestadeFlags
                            guards = safestadeGuards
                            robotCountPerNode = safestadeRobotCount
                            return strategy, False
                        
                        strategy.extend(subStrategy) #extending the Strategy with the subtree strategy
                        if not finish: #check if every node is cleared hence we are finished if not we have move our robots back to where they came from
                            strategy.append((y, node, robotsNeeded, strategy[len(strategy)-1][4],strategy[len(strategy)-1][4] + T.edges[(node,y)].time))
                            robotCountPerNode[y] -= robotsNeeded
                            robotCountPerNode[node] +=  robotsNeeded
                            enteringTime = strategy[len(strategy)-1][4]
                            if flagsCreated:
                                remainingFlags = []
                                for flag in flags:
                                    if flag[0] == node:
                                        robotCountPerNode[node] += flags[index][1]
                                    else: remainingFlags.append(flags)

                                flags = remainingFlags
                
                        return strategy, True
                    else:
                        #Can we retrace robots from flags to perform this move?

                        retraceable = 0

                        for triplet in flags:
                            if triplet[2] <= enteringTime:
                                retraceable += triplet[1]

                        if not (retraceable >= robotsNeeded - robotCountPerNode[node]): #are there enough flags
                            return strategy, False


                        allreadyMoved = 0
                        movedFlagsTimes = []
                        while allreadyMoved != robotsNeeded - robotCountPerNode[node]:
                            shortest = sys.maxsize
                            vertex = 0
                            counter = 0
                            bestIdx = 0
                            for triplet in flags:
                                if triplet[2] <= enteringTime:
                                    path = aStar(triplet[0], node,obstacles, distanceMap, alpha)
                                    if path == None: #is there a vaild path or not?
                                        continue
                                    dist = len(path) - 1
                                    if triplet[2] +  dist < shortest:
                                        shortest = dist
                                        vertex = triplet[0]
                                        bestIdx = counter
                                    counter += 1

                            if allreadyMoved + flags[bestIdx][1] > robotsNeeded - robotCountPerNode[node]:
                                allreadyMoved = robotsNeeded - robotCountPerNode[node]
                                old = flags[bestIdx][1] 
                                flags[bestIdx][1] = robotsNeeded - robotCountPerNode[node] - allreadyMoved
                                strategy.append((flags[bestIdx][0], node, old - flags[bestIdx][1],   flags[bestIdx][2],   flags[bestIdx][2] + shortest))
                                movedFlagsTimes.append(flags[bestIdx][2] + shortest)
                            else: 
                                allreadyMoved += flags[bestIdx][1]
                                strategy.append((flags[bestIdx][0], node, flags[bestIdx][1],   flags[bestIdx][2],   flags[bestIdx][2] + shortest))
                                movedFlagsTimes.append(flags[bestIdx][2] + shortest)
                                flags.pop(bestIdx)

                        neededTime = max(movedFlagsTimes)
                        enteringTime = max(enteringTime, neededTime)

                        robotCountPerNode[node] = robotsNeeded

                        #Try again for the node with the fixed setup
                        subStrategy, feasible = explorePath(node, enteringTime)
                        if feasible == False:
                            visited = safestadeVisited
                            flags = safestadeFlags
                            guards = safestadeGuards
                            robotCountPerNode = safestadeRobotCount
                            return strategy, False

                        strategy.extend(subStrategy)
                        #Note no backtrack steps are neicessary here
                        return strategy,True
                            

                else: #There exits at least 2 children hence multiple strategys exist and an order for efficency has to be regared

                    unExplored = [e[2] for e in combinedLables] #Saving all children / Subtrees that have to be cleared and are still unexplored

                    combinedLables.sort(key=lambda x: x[0], reverse = True) #Sorting the lables decreasing
                    currentLablesRobotCost.sort(key=lambda x: x[0], reverse = True)
                    twice = False # Check if the max amount of robots is needed twice. Than the order of exploring can't effect the robot cost
                    if len(currentLablesRobotCost) > 1: #Is there more than one children
                        if currentLablesRobotCost[0][0] == currentLablesRobotCost[1][0]:
                            twice = True
                    else:   #If there is only one children there is no order
                            twice = True 


                    multiples = [option for option in multipleAtOnce[node] if option[1] < availableRobots] # Deleting all Multiple Strategys which are impossible to perform, meaning robotcost > availableRobots
                    lastOptions = [option for option in multipleAtOnce[node] if option[1] <= availableRobots] #Multiple Move Strategys that are only performable as the last move, meaning robotcost = availableRobots

                    multiples.sort(key=lambda x: x[2], reverse=True)
                    lastOptions.sort(key = lambda x: x[2], reverse=True)


                    while len(unExplored) > 0: #as long as there exists unExplored children
                        safestadeVisited2 = visited.copy()
                        safestadeGuards2 = copy.deepcopy(guards)
                        safestadeFlags2 = copy.deepcopy(flags)
                        safestadeRobotCount2 = robotCountPerNode.copy()
                        safestadeStrategy = strategy.copy()

                        exists = False
                        flagsCreated = False
                        #--------------------------------------------------------
                        # Check if there is multiple last move, if yes do this since it always improves efficiency               
                        #--------------------------------------------------------
                        for (candidateLast, robotsNeededLast, effLast)  in lastOptions:
                            if set(candidateLast) == set(unExplored):

                                # Note that here a backtracking deciosion is opend. Furthermore the faster subtrees will be traceable robots while they wait
                                exists = True

                                # is a slide move acctually neicessary
                                if robotsNeededLast == robotCountPerNode[node]: 
                                    slideMove = True

                                    #determine the longest edge how the slider has to wait and the shortest edge where can go to make the slide move as fast as possible
                                    largest = 0
                                    cheapest = sys.maxsize
                                    for y in candidateLast:
                                        if largest < T.edges[(node,y)].time:
                                            largest =  T.edges[(node,y)].time
                                        if cheapest > T.edges[(node,y)].time:
                                            cheapest =  T.edges[(node,y)].time
                                            slideMoveGo = y

                                    #Checking if an Graph-Edge leads to recontamination

                                    contamination = False
                                    visitedContamination = False
                                    visitedContaminationIdx = []
                                    enemys = []
        
                                    for neighbour in G.adj[G.nodes[node]]:
                                        if neighbour.idx not in candidateLast and visited[neighbour.idx][0] == 0:
                                            contamination = True
                                            enemys.append(neighbour.idx)
                                        elif neighbour.idx not in candidateLast and visited[neighbour.idx][0] == 1 and visited[neighbour.idx][1] > enteringTime:
                                            visitedContamination = True
                                            visitedContaminationIdx.append(neighbour.idx)

                                    if contamination or visitedContamination: 

                                        #retracing a robot
                                        shortest = sys.maxsize
                                        vertex = 0
                                        counter = 0
                                        bestIdx = 0
                                        found = False

                                        for triplet in flags:
                                            if triplet[2] <= enteringTime:
                                                path = aStar(triplet[0], node,obstacles, distanceMap, alpha)
                                                if path == None: #is there a vaild path or not?
                                                    continue
                                                dist = len(path) - 1
                                                if triplet[2] +  dist < shortest:
                                                    found = True
                                                    shortest = dist
                                                    vertex = triplet[0]
                                                    bestIdx = counter
                                                counter += 1

                                        if found == False:
                                            #Nothing was changed yet so no need to go back to safestades 
                                            exists = False
                                            break

                                        #Walk the robot from the flag to the robot  
                                        strategy.append((flags[bestIdx][0], node, 1,   flags[bestIdx][2],   flags[bestIdx][2] + shortest))

                                        #Make sure that the robots wait until the guard arrives
                                        if enteringTime < flags[bestIdx][2] + shortest:
                                            enteringTime = flags[bestIdx][2] + shortest

                                        flags[bestIdx][1] -= 1

                                        if contamination:
                                            maximum = -1
                                            for index in visitedContaminationIdx:
                                                if maximum < visitedContamination[index][1]:
                                                    maximum = visitedContamination[index][1]

                                            guards.append((node, enemys,flags[bestIdx][2] + shortest, max(enteringTime + largest, maximum) ))
                                        elif visitedContamination:
                                            maximum = -1
                                            for index in visitedContaminationIdx:
                                                if maximum < visitedContamination[index][1]:
                                                    maximum = visitedContamination[index][1]
                                            flags.append((node, 1,max(enteringTime + largest, maximum) ))

                                        if flags[bestIdx][1] == 0:
                                            flags.pop(bestIdx)                                           
                                                         
                                else: 
                                    #Case we have spare robots
                                    #determine the longest edge how the slider has to wait and the shortest edge where can go to make the slide move as fast as possible
                                    largest = 0
                                    cheapest = sys.maxsize
                                    for y in candidateLast:
                                        if largest < T.edges[(node,y)].time:
                                            largest =  T.edges[(node,y)].time
                                        if cheapest > T.edges[(node,y)].time:
                                            cheapest =  T.edges[(node,y)].time
                                            slideMoveGo = y

                                    contamination = False
                                    visitedContamination = False
                                    visitedContaminationIdx = []
                                    enemys = []
        
                                    for neighbour in G.adj[G.nodes[node]]:
                                        if neighbour.idx not in candidateLast and visited[neighbour.idx][0] == 0:
                                            contamination = True
                                            enemys.append(neighbour.idx)
                                        elif neighbour.idx not in candidateLast and visited[neighbour.idx][0] == 1 and visited[neighbour.idx][1] > enteringTime:
                                            visitedContamination = True
                                            visitedContaminationIdx.append(neighbour.idx)

                                    if contamination:
                                        #Case there is real contamination hence a robot has to stay
                                        maximum = -1
                                        for index in visitedContaminationIdx:
                                            if maximum < visitedContamination[index][1]:
                                                maximum = visitedContamination[index][1]
                                        guards.append((node, enemys, enteringTime, max(maximum, enteringTime + largest)))
                                    elif visitedContamination:
                                        #Case there is visited contamination one robot has to stay till a known time, thus drop a flag
                                        maximum = -1
                                        for index in visitedContaminationIdx:
                                            if maximum < visitedContamination[index][1]:
                                                maximum = visitedContamination[index][1]
                                        flags.append((node, 1, max(maximum, enteringTime + largest)))
                                        flagsCreated = True
                                    else:
                                        #Case there is no contamination, meaning the robot has to only prevent the slide move
                                        flags.append((node, 1, enteringTime + largest))
                                        flagsCreated = True

                                    if robotCountPerNode[node] - robotsNeededLast > 1:
                                        flags.append((node,robotCountPerNode[node] - robotsNeededLast - 1 , enteringTime))
                                        flagsCreated = True

                                    slideMove = False

                                    

                                welcomeBackTime = [0] * len(candidateLast)
                                counter = 0
                                for y in candidateLast:

                                    for label in combinedLables:
                                        if label[2] == y:
                                            robotsNeeded = label[1]

                                    if slideMove == True and slideMoveGo == y:
                                        strategy.append((node, y, robotsNeeded - 1, enteringTime, enteringTime + T.edges[(node,y)].time))
                                        #Only send the slide move preventer when the longest edge was crossed so after enteringTime + largest
                                        strategy.append((node, y, 1, enteringTime + largest, enteringTime + largest + T.edges[(node,y)].time))
                                        arrvTime = enteringTime + largest + T.edges[(node,y)].time
                                    else:
                                        strategy.append((node, y, robotsNeeded, enteringTime, enteringTime + T.edges[(node,y)].time))
                                        arrvTime = enteringTime + T.edges[(node,y)].time

                                    
                                    if visited[y][0] == 0:
                                        if slideMove == True and slideMoveGo == y:
                                            arrvTime -= largest

                                        visited[y] = (1,arrvTime)
                                        #Convert possible guards into flag
                                        for waechter in guards:
                                            if y in waechter[1]:
                                                waechter[1].remove(y)
                                                if len(waechter[1]) == 0:
                                                    #Guard turns into a flag as normal
                                                    # guards[3] saves how long a guard needs at least to stay for slide move prevention
                                                    flags.append((waechter[0], 1, max(enteringTime, waechter[3])))
                                                    guards.remove(waechter)

                                        if slideMove == True and slideMoveGo == y:
                                            arrvTime += largest

                                    finish = True
                                    for tuple in visited:
                                        if tuple[0] == 0:
                                            finish = False
                                            break
            
                                    if finish == True and finishTime == sys.maxsize:
                                        finishTime = arrvTime
                            

                                    robotCountPerNode[node] = 0
                                    robotCountPerNode[y] += robotsNeeded 

                                    subStrategy, feasible = explorePath(y, arrvTime)

                                    #Check if during the run not enough robots are available hence this decision has to be backtracked and removed
                                    if feasible == False:
                                        visited = safestadeVisited
                                        flags = safestadeFlags
                                        guards = safestadeGuards
                                        robotCountPerNode = safestadeRobotCount
                                        exists = False
                                        strategy = safestadeStrategy
                                        break

                                    strategy.extend(subStrategy)

                                    if not finish:
                                        #Note that these backtrack moves might go over t_finish but they are cut of outside of this method
                                        strategy.append((y,node, robotsNeeded, strategy[len(strategy)-1][4], strategy[len(strategy)-1][4] + T.edges[(node,y)].time))
                                        robotCountPerNode[node] += robotsNeeded
                                        robotCountPerNode[y] -= robotsNeeded 
                                        if flagsCreated:
                                            remainingFlags = []
                                            for flag in flags:
                                                if flag[0] == node:
                                                    robotCountPerNode[node] += flags[index][1]
                                                else: remainingFlags.append(flags)

                                            flags = remainingFlags


                                    # Save when the robots return from this inner path
                                    welcomeBackTime[counter] = strategy[len(strategy)- 1][4]

                                # Continue with the next move when all robots have returned from there inner paths
                                enteringTime = max(welcomeBackTime)

                        # End of last multiple case
                        #------------------------------------------------------------------------

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
                                    strategy.append((node, candidateSingle, robotsNeededSingle, enteringTime, enteringTime + T.edges[(node,candidateSingle)].time))
                                    enteringTime += T.edges[(node,candidateSingle)].time

                                    contamination = False
                                    visitedContamination = False
                                    visitedContaminationIdx = []
                                    enemys = []

                                    #Check for contamination to know how long at least one robot has to stay at the left node
                                    for neighbour in G.adj[G.nodes[node]]:
                                        if neighbour.idx != candidateSingle and visited[neighbour.idx][0] == 0:
                                            contamination = True
                                            enemys.append(neighbour.idx)
                                        elif neighbour.idx != candidateSingle and visited[neighbour.idx][0] == 1 and visited[neighbour.idx][1] > enteringTime:
                                            visitedContamination = True
                                            visitedContaminationIdx.append(neighbour.idx)

                                    if contamination:
                                        maximum = -1
                                        for index in enemys:
                                            if maximum < visited[index][1]:
                                                maximum = visited[index][1]

                                        guards.append((node, enemys, enteringTime, max(maximum, enteringTime + T.edges[(node,candidateSingle)].time)))
                                    elif visitedContamination:
                                        maximum = -1
                                        for index in enemys:
                                            if maximum < visited[index][1]:
                                                maximum = visited[index][1]

                                        flags.append((node, 1,  max(maximum, enteringTime + T.edges[(node,candidateSingle)].time)))
                                    else:
                                        flags.append((node, 1,  maximum, enteringTime + T.edges[(node,candidateSingle)].time))

                                    #the rest of the nodes can be used how they want
                                    if robotCountPerNode[node] - robotsNeededSingle > 1:
                                        flags.append((node, robotCountPerNode[node] - robotsNeededSingle - 1, enteringTime))
                                    robotCountPerNode[node] = 0
                                    robotCountPerNode[candidateSingle] += robotsNeededSingle


                                    if visited[candidateSingle][0] == 0:
                                        visited[candidateSingle] = (1,enteringTime)
                                        #Convert possible guards into flag
                                        for waechter in guards:
                                            if y in waechter[1]:
                                                waechter[1].remove(y)
                                                if len(waechter[1]) == 0:
                                                    #Guard turns into a flag as normal
                                                    # guards[3] saves how long a guard needs at least to stay for slide move prevention
                                                    flags.append((waechter[0], 1, max(enteringTime, waechter[3])))
                                                    guards.remove(waechter)

                                    finish = True
                                    for tuple in visited:
                                        if tuple[0] == 0:
                                            finish = False
                                            break
            
                                    if finish == True and finishTime == sys.maxsize:
                                        finishTime = enteringTime

                                    unExplored = list(set(unExplored) - set(candidateSingle))
                                    subStrategy, feasible = explorePath(candidateSingle, strategy[len(strategy)- 1][4])

                                    if feasible == False:
                                        # Case Distinction if a decision was opend, meaning did we go early in the biggest path and this made a difference
                                        if twice or currentLablesRobotCost[0][0] != robotsNeededSingle:
                                            visited = safestadeVisited
                                            flags = safestadeFlags
                                            guards = safestadeGuards
                                            robotCountPerNode = safestadeRobotCount
                                            return strategy, False
                                        else:
                                             #Swap this with the last place and rearange accordingly, to ensure max subtree is taken at last
                                            counterSingle -= 1
                                            visited = safestadeVisited2
                                            flags = safestadeFlags2
                                            guards = safestadeGuards2
                                            robotCountPerNode = safestadeRobotCount2
                                            strategy = safestadeStrategy
                                            for i in range(counterSingle,len(combinedLables)-1):
                                                save = combinedLables[i]
                                                combinedLables[i] = combinedLables[i+1]
                                                combinedLables[i + 1] = save
                                            continue

                                    strategy.extend(subStrategy)
                                    if not finish:
                                        strategy.append((candidateSingle, node, robotsNeededSingle,strategy[len(strategy)- 1][4], strategy[len(strategy)- 1][4] + T.edges[(node, candidateSingle)].time )) 
                                        robotCountPerNode[node] += robotsNeededSingle
                                        robotCountPerNode[candidateSingle] -=  robotsNeededSingle
                                        enteringTime = strategy[len(strategy)- 1][4]

                                        remainingFlags = []
                                        for flag in flags:
                                            if flag[0] == node:
                                                robotCountPerNode[node] += flags[index][1]
                                            else: remainingFlags.append(flags)

                                        flags = remainingFlags

                                elif robotCountPerNode[node] == robotsNeededSingle and len(list(set(unExplored) - candidateSingle)) == 0: #Check if "==" stands, this is only feasible if we are at the last move 

                                    #Checking if an Graph-Edge leads to recontamination
                                    contamination = False
                                    visitedContamination = False
                                    visitedContaminationIdx = []
                                    enemys = []

                                    #Check for contamination to know how long at least one robot has to stay at the left node
                                    for neighbour in G.adj[G.nodes[node]]:
                                        if neighbour.idx != candidateSingle and visited[neighbour.idx][0] == 0:
                                            contamination = True
                                            enemys.append(neighbour.idx)
                                        elif neighbour.idx != candidateSingle and visited[neighbour.idx][0] == 1 and visited[neighbour.idx][1] > enteringTime:
                                            visitedContamination = True
                                            visitedContaminationIdx.append(neighbour.idx)

                                    if contamination or visitedContamination:
                                    
                                        contamination = True
                                        shortest = sys.maxsize
                                        vertex = 0
                                        counter = 0
                                        bestIdx = 0
                                        found = False

                                        for triplet in flags:
                                            if triplet[2] <= enteringTime:
                                                path = aStar(triplet[0], node,obstacles, distanceMap, alpha)
                                                if path == None: #is there a vaild path or not?
                                                    continue
                                                dist = len(path) - 1
                                                if triplet[2] +  dist < shortest:
                                                    found = True
                                                    shortest = dist
                                                    vertex = triplet[0]
                                                    bestIdx = counter
                                                counter += 1

                                        if found == False:
                                            visited = safestadeVisited
                                            guards = safestadeGuards
                                            flags = safestadeFlags
                                            robotCountPerNode = safestadeRobotCount
                                            return strategy, False

                                        #Walk the robot from the flag to the robot  
                                        strategy.append((flags[bestIdx][0], node, 1,   flags[bestIdx][2],   flags[bestIdx][2] + shortest))

                                        #Make sure that the robots wait until the guard arrives
                                        if enteringTime < flags[bestIdx][2] + shortest:
                                            enteringTime = flags[bestIdx][2] + shortest

                                        flags[bestIdx][1] -= 1

                                        if contamination:
                                            maximum = -1
                                            for index in enemys:
                                                if maximum < visited[index][1]:
                                                    maximum = visited[index][1]

                                            guards.append((node, enemys, flags[bestIdx][2] + shortest, max(maximum, enteringTime + T.edges[(node,candidateSingle)].time)))

                                            if flags[bestIdx][1] == 0:
                                                flags.pop(bestIdx)

                                        elif visitedContamination:
                                            maximum = -1
                                            for index in enemys:
                                                if maximum < visited[index][1]:
                                                    maximum = visited[index][1]

                                            flags.append((node, 1, max(maximum, enteringTime + T.edges[(node,candidateSingle)].time)))
                                        
                                    if not contamination and not visitedContamination:
                                        strategy.append((node, candidateSingle, robotsNeededSingle - 1, enteringTime, enteringTime + T.edges[(node,candidateSingle)].time))
                                        strategy.append((node, candidateSingle, 1, enteringTime + T.edges[(node,candidateSingle)].time, enteringTime + 2 * T.edges[(node,candidateSingle)].time))
                                        enteringTime += 2 * T.edges[(node,candidateSingle)].time 
                                    else: #No slide move is neicessary since a robot will be at the left node anyway
                                        strategy.append((node, candidateSingle, robotsNeededSingle, enteringTime, enteringTime + T.edges[(node,candidateSingle)].time))
                                        enteringTime += T.edges[(node,candidateSingle)].time 


                                    robotCountPerNode[node] -= robotsNeededSingle
                                    robotCountPerNode[candidateSingle] += robotsNeededSingle

                                    if visited[candidateSingle][0] == 0:
                                        if not contamination and not visitedContamination:
                                            enteringTime -= T.edges[(node,candidateSingle)].time

                                        visited[candidateSingle] = (1,enteringTime)
                                        for waechter in guards:
                                            if candidateSingle in waechter[1]:
                                                waechter[1].remove(candidateSingle)
                                                if len(waechter[1]) == 0:
                                                    #Guard turns into a flag as normal
                                                    # guards[3] saves how long a guard needs at least to stay for slide move prevention
                                                    flags.append((waechter[0], 1, max(enteringTime, waechter[3])))
                                                    guards.remove(waechter)

                                        if not contamination and not visitedContamination:
                                            enteringTime += T.edges[(node,candidateSingle)].time

                                    finish = True
                                    for tuple in visited:
                                        if tuple[0] == 0:
                                            finish = False
                                            break
            
                                    if finish == True and finishTime == sys.maxsize:
                                        finishTime = enteringTime

                                    unExplored = list(set(unExplored) - set(candidateSingle))
                                    subStrategy, feasible = explorePath(candidateSingle, strategy[len(strategy)- 1][4])
                                    if feasible == False:
                                        # Case Distinction if a decision was opend, meaning did we go early in the biggest path and this made a difference
                                        if twice or currentLablesRobotCost[0][0] != robotsNeededSingle or contamination or visitedContamination:
                                            visited = safestadeVisited
                                            flags = safestadeFlags
                                            guards = safestadeGuards
                                            robotCountPerNode = safestadeRobotCount
                                           
                                            exists = False
                                            return strategy, False
                                        else:
                                            #Swap this with the last place and rearange accordingly, to ensure max subtree is taken at last
                                            visited = safestadeVisited2
                                            flags = safestadeFlags2
                                            guards = safestadeGuards2
                                            robotCountPerNode = safestadeRobotCount2
                                            strategy = safestadeStrategy
                                            exists = False
                                            unExplored = list(set(unExplored) + set(candidateSingle))
                                            counterSingle -= 1
                                            for i in range(counterSingle,len(combinedLables)-1):
                                                save = combinedLables[i]
                                                combinedLables[i] = combinedLables[i+1]
                                                combinedLables[i + 1] = save
                                            continue
                                            
                                    strategy.extend(subStrategy)
                                    if not finish:
                                        strategy.append((candidateSingle, node, robotsNeededSingle,strategy[len(strategy)- 1][4], strategy[len(strategy)- 1][4] + T.edges[(node, candidateSingle)].time )) 
                                        robotCountPerNode[node] += robotsNeededSingle
                                        robotCountPerNode[candidateSingle] -=  robotsNeededSingle
                                        enteringTime = strategy[len(strategy)- 1][4]

                                else:

                                    #Can we retrace robots from flags to perform this move?
                                    if len(list(set(unExplored) - candidateSingle)) == 0: 
                                        robotsNeededSingle += 1

                                    retraceable = 0

                                    for triplet in flags:
                                        if triplet[2] <= enteringTime and aStar(triplet[0], node,obstacles, distanceMap, alpha) != None:
                                            retraceable += triplet[1]

                                    if not (retraceable >= robotsNeededSingle - robotCountPerNode[node]): #are there enough flags
                                        visited = safestadeVisited
                                        guards = safestadeGuards
                                        flags = safestadeFlags
                                        robotCountPerNode = safestadeRobotCount
                                        return strategy, False


                                    allreadyMoved = 0
                                    movedFlagsTimes = []
                                    while allreadyMoved != robotsNeededSingle - robotCountPerNode[node]:
                                        shortest = sys.maxsize
                                        vertex = 0
                                        counter = 0
                                        bestIdx = 0
                                        for triplet in flags:
                                            if triplet[2] <= enteringTime:
                                                path = aStar(triplet[0], node,obstacles, distanceMap, alpha)
                                                if path == None: #is there a vaild path or not?
                                                    continue
                                                dist = len(path) - 1
                                                if triplet[2] +  dist < shortest:
                                                    shortest = dist
                                                    vertex = triplet[0]
                                                    bestIdx = counter
                                                counter += 1

                                        if allreadyMoved + flags[bestIdx][1] > robotsNeededSingle - robotCountPerNode[node]:
                                            allreadyMoved = robotsNeededSingle - robotCountPerNode[node]
                                            old = flags[bestIdx][1] 
                                            flags[bestIdx][1] = robotsNeededSingle - robotCountPerNode[node] - allreadyMoved
                                            strategy.append((flags[bestIdx][0], node, old - flags[bestIdx][1],   flags[bestIdx][2],   flags[bestIdx][2] + shortest))
                                            movedFlagsTimes.append((flags[bestIdx][2] + shortest))
                                        else: 
                                            allreadyMoved += flags[bestIdx][1]
                                            strategy.append((flags[bestIdx][0], node, flags[bestIdx][1],   flags[bestIdx][2],   flags[bestIdx][2] + shortest))
                                            movedFlagsTimes.append((flags[bestIdx][2] + shortest))
                                            flags.pop(bestIdx)

                                    neededTime = max(movedFlagsTimes)
                                    enteringTime = max(enteringTime, neededTime)

                                    robotCountPerNode[node] = robotsNeeded

                                    #Try again for the node with the fixed setup, no going back to safestades is allowed since we moved flags
                                    counterSingle -= 1
                                    unExplored = list(set(unExplored) + set(candidateSingle))
                                    continue


                        else: #Take the multiple move
                            #--------------------------------------------------------
                            # A Multiple Move that does not finish all children is the best one        
                            # -> No checking for recontamination through graph edge / Slidemoves are neicessary since we can not leave the node here    
                            #--------------------------------------------------------
                            #Note that no slide move prevention or contamination cheks are neicessary since we need at max availablerobots -1
               
                            counterMultiple += 1 
                            unExplored = list(set(unExplored) - set(candidateMultiple))

                            welcomeBackTime = [0] * len(candidateMultiple)
                            counter = 0
                            for y in candidateMultiple:
                                for label in combinedLables:
                                    if label[2] == y:
                                        robotsNeeded = label[1]

                                strategy.append((node, y, robotsNeeded, enteringTime, enteringTime + T.edges[(node,y)].time))

                                if visited[y][0] == 0:
                                    visited[y] = (1,enteringTime + T.edges[(node,y)].time)
                                    for waechter in guards:
                                        if y in waechter[1]:
                                            waechter[1].remove(y)
                                            if len(waechter[1]) == 0: 
                                                #Guard turns into a flag as normal
                                                # guards[3] saves how long a guard needs at least to stay for slide move prevention
                                                flags.append((waechter[0], 1, max(enteringTime, waechter[3])))
                                                guards.remove(waechter)



                                finish = True
                                for tuple in visited:
                                    if tuple[0] == 0:
                                        finish = False
                                        break
        
                                if finish == True and finishTime == sys.maxsize:
                                    finishTime = enteringTime + T.edges[(node,y)].time
                                

                                #TODO Implement so that the spare robots can be used as flags and 1 robot has to be a guard / flag and stay for some time     
                                #Checking if an Graph-Edge leads to recontamination
                                contamination = False
                                visitedContamination = False
                                visitedContaminationIdx = []
                                enemys = []

                                #Check for contamination to know how long at least one robot has to stay at the left node
                                for neighbour in G.adj[G.nodes[node]]:
                                    if neighbour.idx not in candidateMultiple and visited[neighbour.idx][0] == 0:
                                        contamination = True
                                        enemys.append(neighbour.idx)
                                    elif neighbour.idx not in candidateMultiple and visited[neighbour.idx][0] == 1 and visited[neighbour.idx][1] > enteringTime:
                                        visitedContamination = True
                                        visitedContaminationIdx.append(neighbour.idx) 

                                #Determine how long the one robot has to stay to prevent the slide move
                                largest = 0
                                for y in candidateMultiple:
                                    if largest < T.edges[(node,y)].time:
                                        largest =  T.edges[(node,y)].time


                                if contamination:
                                    maximum = -1
                                    for index in visitedContaminationIdx:
                                        if maximum < visited[index][1]:
                                            maximum = visited[index][1]
                                    guards.append((node, enemys, enteringTime, max(maximum, enteringTime + largest)))
                                elif visitedContamination:
                                    maximum = -1
                                    for index in visitedContaminationIdx:
                                        if maximum < visited[index][1]:
                                            maximum = visited[index][1]
                                    flags.append((node, 1, max(maximum, enteringTime + largest)))
                                else:
                                    flags.append((node, 1,  enteringTime + largest))

                                if robotCountPerNode[node] - robotsNeeded > 1:
                                    flags.append((node,robotCountPerNode[node] - robotsNeeded - 1, enteringTime))

                                robotCountPerNode[node] = 0
                                robotCountPerNode[y] += robotsNeeded

                                subStrategy, feasible = explorePath(y, strategy[len(strategy)- 1][4])

                                #Check if during the run not enough robots are available if yes this decision has to be backtracked
                                if feasible == False:
                                    #Note that this strategy might be feasible in a later time, but this is not implemented here
                                    unExplored = list(set(unExplored) + set(candidateMultiple))

                                    visited = safestadeVisited2
                                    flags = safestadeFlags2
                                    guards = safestadeGuards2
                                    robotCountPerNode = safestadeRobotCount2
                                
                                    continue

                                #Remove all flags that represent the spare rebots during the exploration of the subtree
                                remainingFlags = []
                                for flag in flags:
                                    if flag[0] == node:
                                        robotCountPerNode[node] += flags[index][1]
                                    else: remainingFlags.append(flags)

                                flags = remainingFlags
                                strategy.extend(subStrategy)
                                #Note that we always have to backtrack since this can't be the last move
                                strategy.append((y,node,robotsNeeded,strategy[len(strategy)- 1][4],strategy[len(strategy)- 1][4] + T.edges[(node,y)].time))

                                robotCountPerNode[node] += robotsNeeded
                                robotCountPerNode[y] -= robotsNeeded
                                # Save when the robots return from this inner path
                                welcomeBackTime[counter] = strategy[len(strategy)- 1][4]

                            # Continue with the next move when all robots have returned from there inner paths
                            enteringTime = max(welcomeBackTime)

            return strategy, True

        strategy, clearance = explorePath(root, 0)

        if clearance == False:
            strategy = computeClosingExits()


        strategy.insert(0, (None,root,availableRobots))
        #Cut of strategy moves that go over t_finish, meaning unneicessary backtrack moves
        strategy = [move for move in strategy if move[4] <= finishTime]

        return strategy, clearance, visited
                

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
    def computeExpTime(visited, G: Graph):
        #Computing the expected search time
        eff = 0
        for node in G.nodes:
            eff += node.prior * visited[node.idx][1]

        return eff

    # ------------------------------------------------------------
    # THE REAL GRAPH SEARCH ALGORITHIM USING EVERYTHING FROM ABOVE
    # ------------------------------------------------------------

    minExpTime = np.inf
    bestStrategy = None
    bestTree = None
    clearStrat = False
    counter = [0] * startNodes

    for i in range(numOfTrees):

        root = random.randint(0,startNodes-1)

        if counter[root] == 0:
            T = computeGreedySpanningTree(G,root)
            counter[root] += 1
        else:
            T = computeRandomSpanningTree(G,root)

        treeStrategy, clear, visited = treeSearch(T,root, availableRobots)
        #Check if strategy is perfomable. If not use closing exists strategy

        if clear == False and clearStrat == False:
            #TODO: Find a way to compare to closing-exits strategys
            print("Hello World")
        elif clear == True and clearStrat == False:
            #Always take the clearance strategy
            expTime = computeExpTime(visited, G)
            minExpTime = expTime
            bestStrategy = treeStrategy
            bestTree = T
            
        elif clear == True and clearStrat == True:
            # Take the strategy with best expected search time
            expTime = computeExpTime(visited, G)
            if expTime < minExpTime:
                minExpTime = expTime
                bestStrategy = treeStrategy
                bestTree = T

    return bestStrategy, bestTree

