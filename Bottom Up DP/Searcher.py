import numpy as np
import math
import random
import sys
from Graph import Graph, Node, Edge
from itertools import combinations
from TrajectoryPlanning import aStar
import copy
import time


def graphSearch(G : Graph, availableTime, availableRobots, startNodes, obstacles, distanceMap, alpha, cellpriors, D):
    #INPUT:
    # G: a Graph object repesenting the merged navigationgraph
    # availableTime: a number which represents the available computation time
    # availableRobots: an integer representing the amount of robots available
    # startNodes:  an integer representing that all nodes from 0 to startNode-1 are valid startNodes for our Algorithim
    # obstacles:
    # distanceMap: 
    # alpha:
    # cellpriors: All prior values for each cell for calculating the expected searchtime function
    # D: Detection set for each node for calculating the expected searchtime function
    # aerialSpeed: TO-DO
    # groundSpeed: TO-DO
    # aerialBattery: TO-DO
    # groundBattery: TO-DO

    #OUTPUT:
    #bestStrategy: A list containing the best strategy where each entry is in the form (source node,target node, amount of Robots)

    # ---------------------------------------------------
    # COMPUTING B-Lables and the Efficiency Lables in dependency of the available robots
    # Remark: Lables represent the amount of robots needed for this path
    # --------------------------------------------------- 
    def computeLabelsWithBudget(T: Graph, root, parent, maxRobots: int):
        #CHECKED

        bLabels = {}
        robotTable = {}

        children = [y.idx for y in T.adj[T.nodes[root]] if y.idx != parent]

        # ---------------- Leaf ----------------
        if len(children) == 0:
            bLabels[(parent, root)] = 1

            edgeTime = T.edges[(parent, root)].time if parent is not None else 1
            prior = T.nodes[root].prior

            # In a leaf the amount of robots can not sped up the exploration
            table = {r: (edgeTime, prior / edgeTime) for r in range(1, maxRobots + 1)}
            robotTable[(parent, root)] = table

            return bLabels, robotTable, prior, edgeTime

        # ------------- Inner node: visit children first -------------
        childBLabel = {}      # child -> BLabel(child)
        childPrior = {}       # child -> totalPrior(subtree(child))
        childTimeTable = {}   # child -> {ri: time}   (aus robotTable des Kindes)

        totalPrior = T.nodes[root].prior

        for child in children:
            subB, subTable, subPrior, _ = computeLabelsWithBudget(T, child, root, maxRobots)
            bLabels.update(subB)
            robotTable.update(subTable)

            childBLabel[child] = subB[(root, child)]
            childPrior[child] = subPrior
            childTimeTable[child] = {r: t for r, (t, _eff) in subTable[(root, child)].items()}

            totalPrior += subPrior

        # ---------- Set B-Label for inner node (formular from the Andreas Kolling paper) ----------
        childLabelsSorted = sorted(childBLabel.values(), reverse=True)
        p1 = childLabelsSorted[0]
        p2 = childLabelsSorted[1] if len(childLabelsSorted) > 1 else 0

        bLabelRoot = (p1 + 1) if p1 == 1 else max(p1, p2 + 1)
        bLabels[(parent, root)] = bLabelRoot

        # ---- Check if the bLable is feasible----
        if maxRobots < bLabelRoot:
            # No guaranteed clearance is possible
            robotTable[(parent, root)] = {}
            return bLabels, robotTable, totalPrior, float('inf')

        edgeTime = T.edges[(parent, root)].time if parent is not None else 1

        # ---------- For each r: Calculate the optimal Batch-Schedule ----------
        table = {}
        for r in range(bLabelRoot, maxRobots + 1):
            bestChildrenTime = _bestChildrenSchedule(children, childBLabel, childTimeTable, r)
            time_r = edgeTime + bestChildrenTime
            table[r] = (time_r, totalPrior / time_r)

        robotTable[(parent, root)] = table
        totalTimeAtMax = table[maxRobots][0]

        return bLabels, robotTable, totalPrior, totalTimeAtMax


    def _bestChildrenSchedule(children, childBLabel, childTimeTable, r):
        # Returns the minimal total time to fully clear all children subtrees with r robots
        # by considering all possible partitions of children into batches

        # If there exists no children then no time is needed to clear them
        k = len(children)
        if k == 0:
            return 0.0

        # Bitmask of the form 111...1 (k ones): bit i set means child i is
        # included in this subset. fullMask represents "all children".
        fullMask = (1 << k) - 1

        # ---- Step 1: precompute parallelBatchCost for every possible subset ----
        costOfSubset = [float('inf')] * (1 << k)  # cost per subset, indexed by bitmask
        costOfSubset[0] = 0.0
        for mask in range(1, 1 << k):  # loop over all possible subsets (as bitmasks)
            subset = [children[i] for i in range(k) if mask & (1 << i)]

            # If a subset needs more robots than available it is unfeasible
            minRobotsNeeded = sum(childBLabel[c] for c in subset)
            if minRobotsNeeded > r:
                costOfSubset[mask] = float('inf')
                continue

            costOfSubset[mask] = _parallelBatchCost(subset, childTimeTable, r)

        # ---- Step 2: DP over partitions ----
        # We do NOT need to check the order of the batches, since batches run
        # sequentially with robots fully reset in between -- the total time is
        # just a sum of batch times, and addition doesn't care about order.
        # We also don't need to re-check the robot budget here: that check
        # already happened inside parallelBatchCost / the B-Label skip above,
        # and any infeasible subset is simply marked as inf, which min() will
        # naturally never select.
        
        # dp[mask] = minimal total time for the best partition of the children
        # in 'mask' into any number of sequential batches.
        dp = [float('inf')] * (1 << k)
        dp[0] = 0.0  # base case: no children -> no time needed

        for mask in range(1, 1 << k):
            # Pick a fixed "anchor" child that must be part of mask: its lowest
            # set bit. Any child would work here, we just need one fixed choice
            # per mask so that every partition of mask gets counted exactly once
            # (see explanation below).
            low = mask & (-mask)

            # We want dp[mask] = the best way to split the children in 'mask'
            # into batches. Key idea: in ANY such split, the anchor child 'low'
            # belongs to exactly one batch. So instead of enumerating all
            # partitions directly, we enumerate all possible choices for "which
            # batch contains low" (call it 'sub'), and combine it with the best
            # possible partition of whatever children are left over
            # (mask ^ sub, i.e. mask with sub's bits removed).
            #
            # Because 'sub' ranges over every subset of 'mask' that contains
            # 'low', and dp[mask ^ sub] was already fully computed in an earlier
            # iteration (mask ^ sub is always a strictly smaller number than
            # mask, since it has strictly fewer bits set), dp[mask ^ sub] already
            # represents the OPTIMAL split of the remainder -- including splits
            # into two, three, or more further batches. So even though this loop
            # only ever picks "one batch (sub) + one rest (mask^sub)", the
            # recursive structure of dp means every partition into any number of
            # batches is implicitly covered.
            #
            # Enumerating only subsets containing 'low' (instead of all subsets
            # of mask) guarantees each partition of mask is considered exactly
            # once -- not zero times, not multiple times -- because the batch
            # containing 'low' is a well-defined, unique part of any partition.
            # This also cuts the total work across all masks down to O(3^k)
            # instead of a naive O(4^k).
            sub = mask
            while sub > 0:
                if sub & low:  # only consider 'sub' if it contains the anchor child
                    # Candidate: treat 'sub' as one batch, and combine with the
                    # already-optimal partition of the remaining children.
                    candidate = costOfSubset[sub] + dp[mask ^ sub]
                    if candidate < dp[mask]:
                        dp[mask] = candidate
                sub = (sub - 1) & mask  # move to the next subset of mask

        return dp[fullMask]


    def _parallelBatchCost(subset, childTimeTable, r):
        #Calculate the minimal time to clear all subtrees inside the batch subset together when r robots are available
        #Checked

        #Empty batch takes no time
        if len(subset) == 0:
            return 0.0

        #The needed time of the batch will be one of the individual times of the contained subtrees since we do all of them in parallel and the biggest determines the searchtime
        candidateTimes = set()
        for c in subset:
            candidateTimes.update(childTimeTable[c].values())


        # Sort the times ascending and then check starting with the lowest time if it is feasible to reach this time
        for T in sorted(candidateTimes):
            totalRobots = 0
            feasible = True
            for c in subset:
                #find the lowest amount of robots which can clear c under T 
                minRi = None
                for ri in sorted(childTimeTable[c].keys()):
                    if childTimeTable[c][ri] <= T:
                        minRi = ri
                        break
                if minRi is None:
                    feasible = False
                    break
                totalRobots += minRi
            if feasible and totalRobots <= r:
                #It's possible to clear batch in T
                return T

        #Batch can not be cleared together hence infinity is returned
        return float('inf')

    
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


        robotCountPerNode = [0] * len(T.nodes) #Saving how many robots are on each node at every time
        robotCountPerNode[root] = availableRobots

        def explorePath(node, enteringTime):

            #Safestades, so that if something goes wrong we can backtrack to the old states
            nonlocal guards
            nonlocal flags
            nonlocal visited
            nonlocal robotCountPerNode

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
                    if robotCountPerNode[node] >= robotsNeeded : #Are there enough robots to clear
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

                            if contamination or visitedContamination or robotsNeeded == 1:
                                                            
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
                                            path = aStar(T.nodes[triplet[0]].pos, T.nodes[node].pos,obstacles, distanceMap, alpha)
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
                                        guards.append([node, enemys, flags[bestIdx][2] + shortest, max(enteringTime + T.edges[(node,y)].time, maximum)])
                                    elif visitedContamination:
                                        maximum = -1
                                        for index in visitedContaminationIdx:
                                            if maximum < visited[index][1]:
                                                maximum = visited[index][1]

                                        flags.append([node, 1, max(maximum, enteringTime + T.edges[(node,y)].time )])
                                    else:
                                        flags.append([node, 1, enteringTime + T.edges[(node,y)].time ])

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
                                guards.append([node, enemys,enteringTime, max(enteringTime + T.edges[(node,y)].time, maximum)])
                            elif visitedContamination:
                                maximum = -1
                                for index in visitedContaminationIdx:
                                    if maximum < visited[index][1]:
                                        maximum = visited[index][1]
                                flags.append([node,1,max(enteringTime + T.edges[(node,y)].time, maximum)])
                            else:
                                flags.append([node,1,enteringTime + T.edges[(node,y)].time])

                            #We have more than one spare robot, these can all be used
                            if robotCountPerNode[node] - robotsNeeded > 1:
                                flags.append([node, robotCountPerNode[node] - robotsNeeded - 1, enteringTime])

                            strategy.append((node, y, robotsNeeded, enteringTime, enteringTime + T.edges[(node,y)].time))
                            enteringTime += T.edges[(node,y)].time

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
                                    waechter[3] = max(enteringTime, waechter[3])
                                    if len(waechter[1]) == 0:
                                        #Guard turns into a flag as normal
                                        # guards[3] saves how long a guard needs at least to stay for slide move prevention
                                        flags.append([waechter[0], 1, max(enteringTime, waechter[3])])
                                        guards.remove(waechter)

                            if slideMove == True:
                                enteringTime += T.edges[(node,y)].time



                        finish = True
                        for tuple in visited:
                            if tuple[0] == 0:
                                finish = False
                                break


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

                            remainingFlags = []
                            for flag in flags:
                                if flag[0] == node and flag[2] <= enteringTime:
                                    robotCountPerNode[node] += flag[1]
                                else: remainingFlags.append(flag)

                            flags = remainingFlags
                
                        return strategy, True
                    else:
                        #Can we retrace robots from flags to perform this move?

                        retraceable = 0

                        for triplet in flags:
                            if triplet[2] <= enteringTime and None != aStar(T.nodes[triplet[0]].pos, T.nodes[node].pos,obstacles, distanceMap, alpha):
                                retraceable += triplet[1]

                        if not (retraceable >= robotsNeeded - robotCountPerNode[node]): #are there enough flags
                            visited = safestadeVisited
                            flags = safestadeFlags
                            guards = safestadeGuards
                            robotCountPerNode = safestadeRobotCount

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
                                    path = aStar(T.nodes[triplet[0]].pos, T.nodes[node].pos,obstacles, distanceMap, alpha)
                                    if path == None: #is there a vaild path or not?
                                        continue
                                    dist = len(path) - 1
                                    if triplet[2] +  dist < shortest:
                                        shortest = dist
                                        vertex = triplet[0]
                                        bestIdx = counter
                                    counter += 1

                            if allreadyMoved + flags[bestIdx][1] > robotsNeeded - robotCountPerNode[node]:    
                                old = flags[bestIdx][1] 
                                flags[bestIdx][1] = old - (robotsNeeded - robotCountPerNode[node] - allreadyMoved)
                                strategy.append((flags[bestIdx][0], node, old - flags[bestIdx][1],   flags[bestIdx][2],   flags[bestIdx][2] + shortest))
                                movedFlagsTimes.append(flags[bestIdx][2] + shortest)
                                allreadyMoved = robotsNeeded - robotCountPerNode[node]
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


                    multiples = [option for option in multipleAtOnce[node] if option[1] < robotCountPerNode[node]] # Deleting all Multiple Strategys which are impossible to perform, meaning robotcost > availableRobots
                    lastOptions = [option for option in multipleAtOnce[node] if option[1] <= robotCountPerNode[node]] #Multiple Move Strategys that are only performable as the last move, meaning robotcost = availableRobots

                    multiples.sort(key=lambda x: x[2], reverse=True)
                    lastOptions.sort(key = lambda x: x[2], reverse=True)

                    loopGuard = 0
                    while len(unExplored) > 0: #as long as there exists unExplored children

                        loopGuard += 1
                        if loopGuard > 10000:
                            raise RuntimeError(f"Endlosschleife erkannt bei node={node}, unExplored={unExplored}")


                        safestadeVisited2 = visited.copy()
                        safestadeGuards2 = copy.deepcopy(guards)
                        safestadeFlags2 = copy.deepcopy(flags)
                        safestadeRobotCount2 = robotCountPerNode.copy()
                        safestadeStrategy = strategy.copy()
                        safestadeEnteringTime = enteringTime
                        safestadeUnexplored = unExplored

                        exists = False
 
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
                                        if cheapest > T.edges[(node,y)].time and labelsRobotCost[(node,y)] != 1:
                                            cheapest =  T.edges[(node,y)].time
                                            slideMoveGo = y

                                    #All moves take only one robot hence this multiple move can avoid a slide move and thus can not be performed
                                    if cheapest == sys.maxsize:
                                        exists = False
                                        break

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
                                                path = aStar(T.nodes[triplet[0]].pos, T.nodes[node].pos,obstacles, distanceMap, alpha)
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
                                            #Nothing was changed yet so no need to go back to safestades, doing it anywaay for testing
                                            visited = safestadeVisited2
                                            flags = safestadeFlags2
                                            guards = safestadeGuards2
                                            robotCountPerNode = safestadeRobotCount2
                                            exists = False
                                            strategy = safestadeStrategy
                                            enteringTime = safestadeEnteringTime
                                            unExplored = safestadeUnexplored
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
                                                if maximum < visited[index][1]:
                                                    maximum = visited[index][1]

                                            guards.append([node, enemys,flags[bestIdx][2] + shortest, max(enteringTime + largest, maximum) ])
                                        elif visitedContamination:
                                            maximum = -1
                                            for index in visitedContaminationIdx:
                                                if maximum < visited[index][1]:
                                                    maximum = visited[index][1]
                                            flags.append([node, 1,max(enteringTime + largest, maximum) ])

                                        if flags[bestIdx][1] == 0:
                                            flags.pop(bestIdx)  

                                        slideMove = False                                         
                                                         
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
                                            if maximum < visited[index][1]:
                                                maximum = visited[index][1]
                                        guards.append([node, enemys, enteringTime, max(maximum, enteringTime + largest)])
                                    elif visitedContamination:
                                        #Case there is visited contamination one robot has to stay till a known time, thus drop a flag
                                        maximum = -1
                                        for index in visitedContaminationIdx:
                                            if maximum < visited[index][1]:
                                                maximum = visited[index][1]
                                        flags.append([node, 1, max(maximum, enteringTime + largest)])
  
                                    else:
                                        #Case there is no contamination, meaning the robot has to only prevent the slide move
                                        flags.append([node, 1, enteringTime + largest])

                                    if robotCountPerNode[node] - robotsNeededLast > 1:
                                        flags.append([node,robotCountPerNode[node] - robotsNeededLast - 1 , enteringTime])

                                    slideMove = False

                                    

                                welcomeBackTime = [enteringTime] * len(candidateLast)
                                counter = 0

                                robotCountPerNode[node] = 0
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
                                                waechter[3] = max(arrvTime, waechter[3])
                                                if len(waechter[1]) == 0:
                                                    #Guard turns into a flag as normal
                                                    # guards[3] saves how long a guard needs at least to stay for slide move prevention
                                                    flags.append([waechter[0], 1, max(enteringTime, waechter[3])])
                                                    guards.remove(waechter)

                                        if slideMove == True and slideMoveGo == y:
                                            arrvTime += largest

                                    finish = True
                                    for tuple in visited:
                                        if tuple[0] == 0:
                                            finish = False
                                            break
            
                            

                                    robotCountPerNode[y] += robotsNeeded 

                                    subStrategy, feasible = explorePath(y, arrvTime)

                                    #Check if during the run not enough robots are available hence this decision has to be backtracked and removed
                                    if feasible == False:
                                        visited = safestadeVisited2
                                        flags = safestadeFlags2
                                        guards = safestadeGuards2
                                        robotCountPerNode = safestadeRobotCount2
                                        exists = False
                                        strategy = safestadeStrategy
                                        enteringTime = safestadeEnteringTime
                                        unExplored = safestadeUnexplored
                                        break

                                    strategy.extend(subStrategy)

                                    if not finish:
                                        #Note that these backtrack moves might go over t_finish but they are cut of outside of this method
                                        strategy.append((y,node, robotsNeeded, strategy[len(strategy)-1][4], strategy[len(strategy)-1][4] + T.edges[(node,y)].time))
                                        robotCountPerNode[node] += robotsNeeded
                                        robotCountPerNode[y] -= robotsNeeded 
    
                                        remainingFlags = []
                                        for flag in flags:
                                            if flag[0] == node and flag[2]<= max(welcomeBackTime):
                                                robotCountPerNode[node] += flag[1]
                                            else: remainingFlags.append(flag)

                                        flags = remainingFlags


                                    # Save when the robots return from this inner path
                                    welcomeBackTime[counter] = strategy[len(strategy)- 1][4]
                                    counter += 1

                                else: #Only if the for loop is not left with a break
                                    # Continue with the next move when all robots have returned from there inner paths
                                    enteringTime = max(welcomeBackTime)

                        # End of last multiple case
                        #------------------------------------------------------------------------

                        #If we did a last move we are have explored all children hence are finished
                        if exists == True: 
                            break

                        counterSingle = 0
                        counterMultiple = -1

                        #find next best feasible single option
                        (effSingle, robotsNeededSingle, candidateSingle) = combinedLables[counterSingle]
                        while candidateSingle not in unExplored:
                            counterSingle += 1
                            (effSingle, robotsNeededSingle, candidateSingle) = combinedLables[counterSingle]

                        #find next best feasible multiple option if it exists
                        alreadyExplored = True
                        while alreadyExplored == True and counterMultiple + 1 < len(multiples):
                            counterMultiple += 1
                            (candidateMultiple, robotsNeededMultiple, effMultiple) = multiples[counterMultiple]
                            alreadyExplored = False
                            for vertex in candidateMultiple:
                                if vertex not in unExplored:
                                    alreadyExplored = True

                            if not alreadyExplored and set(candidateMultiple) == set(unExplored):
                                alreadyExplored = True


                        if alreadyExplored == True or (effSingle > effMultiple and (candidateSingle not in candidateMultiple or effSingle * 1/2 > effMultiple )) : #This implies that the single move is the better choice or that no multiple move exists
                            #--------------------------------------------------------
                            # A Single Move is the best one, Note that we have to do a case distinction here if we need slide moves & and this opens a decision or not
                            #--------------------------------------------------------
                                counterSingle += 1

                                if robotCountPerNode[node] > robotsNeededSingle: #No slide moves have to be considered
                                    strategy.append((node, candidateSingle, robotsNeededSingle, enteringTime, enteringTime + T.edges[(node,candidateSingle)].time))

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
                                        for index in visitedContaminationIdx:
                                            if maximum < visited[index][1]:
                                                maximum = visited[index][1]

                                        guards.append([node, enemys, enteringTime, max(maximum, enteringTime + T.edges[(node,candidateSingle)].time)])
                                    elif visitedContamination:
                                        maximum = -1
                                        for index in visitedContaminationIdx:
                                            if maximum < visited[index][1]:
                                                maximum = visited[index][1]

                                        flags.append([node, 1,  max(maximum, enteringTime + T.edges[(node,candidateSingle)].time)])
                                    else:
                                        flags.append([node, 1, enteringTime + T.edges[(node,candidateSingle)].time])

                                    #the rest of the nodes can be used how they want
                                    if robotCountPerNode[node] - robotsNeededSingle > 1:
                                        flags.append([node, robotCountPerNode[node] - robotsNeededSingle - 1, enteringTime])
                                    robotCountPerNode[node] = 0
                                    robotCountPerNode[candidateSingle] += robotsNeededSingle

                                    enteringTime += T.edges[(node,candidateSingle)].time


                                    if visited[candidateSingle][0] == 0:
                                        visited[candidateSingle] = (1,enteringTime)
                                        #Convert possible guards into flag
                                        for waechter in guards:
                                            if candidateSingle in waechter[1]:
                                                waechter[1].remove(candidateSingle)
                                                waechter[3] = max(enteringTime, waechter[3])
                                                if len(waechter[1]) == 0:
                                                    #Guard turns into a flag as normal
                                                    # guards[3] saves how long a guard needs at least to stay for slide move prevention
                                                    flags.append([waechter[0], 1, max(enteringTime, waechter[3])])
                                                    guards.remove(waechter)

                                    finish = True
                                    for tuple in visited:
                                        if tuple[0] == 0:
                                            finish = False
                                            break
            


                                    unExplored = list(set(unExplored) - {candidateSingle})
                                    subStrategy, feasible = explorePath(candidateSingle, strategy[len(strategy)- 1][4])

                                    if feasible == False:
                                        # Case Distinction if a decision was opend, meaning did we go early in the biggest path and this made a difference
                                        if twice or currentLablesRobotCost[0][0] != robotsNeededSingle or len(unExplored) == 0:
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
                                            enteringTime = safestadeEnteringTime
                                            unExplored = safestadeUnexplored
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
                                            if flag[0] == node and flag[2]<= enteringTime:
                                                robotCountPerNode[node] += flag[1]
                                            else: remainingFlags.append(flag)

                                        flags = remainingFlags

                                elif robotCountPerNode[node] == robotsNeededSingle and len(list(set(unExplored) - {candidateSingle})) == 0: #Check if "==" stands, this is only feasible if we are at the last move 

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

                                    if contamination or visitedContamination or robotsNeededSingle == 1:
                                    
                                        shortest = sys.maxsize
                                        vertex = 0
                                        counter = 0
                                        bestIdx = 0
                                        found = False

                                        for triplet in flags:
                                            if triplet[2] <= enteringTime:
                                                path = aStar(T.nodes[triplet[0]].pos, T.nodes[node].pos,obstacles, distanceMap, alpha)
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
                                            for index in visitedContaminationIdx:
                                                if maximum < visited[index][1]:
                                                    maximum = visited[index][1]

                                            guards.append([node, enemys, flags[bestIdx][2] + shortest, max(maximum, enteringTime + T.edges[(node,candidateSingle)].time)])
                                        elif visitedContamination:
                                            maximum = -1
                                            for index in visitedContaminationIdx:
                                                if maximum < visited[index][1]:
                                                    maximum = visited[index][1]

                                            flags.append([node, 1, max(maximum, enteringTime + T.edges[(node,candidateSingle)].time)])
                                        else:
                                            flags.append([node, 1, enteringTime + T.edges[(node,candidateSingle)].time])

                                        if flags[bestIdx][1] == 0:
                                            flags.pop(bestIdx)
                                        
                                    if not contamination and not visitedContamination and robotsNeededSingle != 1:
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
                                                waechter[3] = max(enteringTime, waechter[3])
                                                if len(waechter[1]) == 0:
                                                    #Guard turns into a flag as normal
                                                    # guards[3] saves how long a guard needs at least to stay for slide move prevention
                                                    flags.append([waechter[0], 1, max(enteringTime, waechter[3])])
                                                    guards.remove(waechter)

                                        if not contamination and not visitedContamination:
                                            enteringTime += T.edges[(node,candidateSingle)].time

                                    finish = True
                                    for tuple in visited:
                                        if tuple[0] == 0:
                                            finish = False
                                            break
            


                                    unExplored = list(set(unExplored) - {candidateSingle})
                                    subStrategy, feasible = explorePath(candidateSingle, strategy[len(strategy)- 1][4])
                                    if feasible == False:
                                        # Since there is only once child left there can't be a backtrack decision
                                        visited = safestadeVisited
                                        flags = safestadeFlags
                                        guards = safestadeGuards
                                        robotCountPerNode = safestadeRobotCount
                                        
                                        exists = False
                                        return strategy, False

                                            
                                    strategy.extend(subStrategy)
                                    if not finish:
                                        strategy.append((candidateSingle, node, robotsNeededSingle,strategy[len(strategy)- 1][4], strategy[len(strategy)- 1][4] + T.edges[(node, candidateSingle)].time )) 
                                        robotCountPerNode[node] += robotsNeededSingle
                                        robotCountPerNode[candidateSingle] -=  robotsNeededSingle
                                        enteringTime = strategy[len(strategy)- 1][4]

                                else:
                                    if robotCountPerNode[node] == robotsNeededSingle:
                                        robotsNeededSingle += 1
                                        
                                    #Can we retrace robots from flags to perform this move?
                                    if len(list(set(unExplored) - {candidateSingle})) == 0: 
                                        robotsNeededSingle += 1

                                    retraceable = 0

                                    for triplet in flags:
                                        if triplet[2] <= enteringTime and aStar(T.nodes[triplet[0]].pos, T.nodes[node].pos,obstacles, distanceMap, alpha) != None:
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
                                                path = aStar(T.nodes[triplet[0]].pos, T.nodes[node].pos,obstacles, distanceMap, alpha)
                                                if path == None: #is there a vaild path or not?
                                                    continue
                                                dist = len(path) - 1
                                                if triplet[2] +  dist < shortest:
                                                    shortest = dist
                                                    vertex = triplet[0]
                                                    bestIdx = counter
                                                counter += 1

                                        if allreadyMoved + flags[bestIdx][1] > robotsNeededSingle - robotCountPerNode[node]:
                                            old = flags[bestIdx][1] 
                                            flags[bestIdx][1] = old - (robotsNeededSingle - robotCountPerNode[node] - allreadyMoved)
                                            strategy.append((flags[bestIdx][0], node, old - flags[bestIdx][1],   flags[bestIdx][2],   flags[bestIdx][2] + shortest))
                                            movedFlagsTimes.append((flags[bestIdx][2] + shortest))
                                            allreadyMoved = robotsNeededSingle - robotCountPerNode[node]
                                        else: 
                                            allreadyMoved += flags[bestIdx][1]
                                            strategy.append((flags[bestIdx][0], node, flags[bestIdx][1],   flags[bestIdx][2],   flags[bestIdx][2] + shortest))
                                            movedFlagsTimes.append((flags[bestIdx][2] + shortest))
                                            flags.pop(bestIdx)

                                    neededTime = max(movedFlagsTimes)
                                    enteringTime = max(enteringTime, neededTime)

                                    robotCountPerNode[node] = robotsNeededSingle

                                    #Try again for the node with the fixed setup, no going back to safestades is allowed since we moved flags
                                    counterSingle -= 1
                                    unExplored = list(set(unExplored) | {candidateSingle})
                                    continue


                        else: #Take the multiple move
                            #--------------------------------------------------------
                            # A Multiple Move that does not finish all children is the best one        
                            # -> No checking for recontamination through graph edge / Slidemoves are neicessary since we can not leave the node here    
                            #--------------------------------------------------------
                            #Note that no slide move prevention or contamination cheks are neicessary since we need at max availablerobots -1

                            #Check if this is a failed last multiple case
                            if len(list(set(unExplored) - set(candidateMultiple))) == 0:
                                multipleAtOnce
                                continue

                            unExplored = list(set(unExplored) - set(candidateMultiple))

                            

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
                                guards.append([node, enemys, enteringTime, max(maximum, enteringTime + largest)])
                            elif visitedContamination:
                                maximum = -1
                                for index in visitedContaminationIdx:
                                    if maximum < visited[index][1]:
                                        maximum = visited[index][1]
                                flags.append([node, 1, max(maximum, enteringTime + largest)])
                            else:
                                flags.append([node, 1,  enteringTime + largest])

                            if robotCountPerNode[node] - robotsNeededMultiple > 1:
                                flags.append([node,robotCountPerNode[node] - robotsNeededMultiple- 1, enteringTime])

                            robotCountPerNode[node] = 0

                            welcomeBackTime = [enteringTime] * len(candidateMultiple)
                            counter = 0

                            for y in candidateMultiple:
                                for label in combinedLables:
                                    if label[2] == y:
                                        robotsNeeded = label[1]

                                strategy.append((node, y, robotsNeeded, enteringTime, enteringTime + T.edges[(node,y)].time))

                                robotCountPerNode[y] += robotsNeeded

                                if visited[y][0] == 0:
                                    visited[y] = (1,enteringTime + T.edges[(node,y)].time)
                                    for waechter in guards:
                                        if y in waechter[1]:
                                            waechter[1].remove(y)
                                            waechter[3] = max(enteringTime + T.edges[(node,y)].time, waechter[3])
                                            if len(waechter[1]) == 0: 
                                                # Guard turns into a flag as normal
                                                # guards[3] saves how long a guard needs at least to stay for slide move prevention
                                                flags.append([waechter[0], 1, max(enteringTime + T.edges[(node,y)].time, waechter[3])])
                                                guards.remove(waechter)


                                finish = True
                                for tuple in visited:
                                    if tuple[0] == 0:
                                        finish = False
                                        break
                              

                                subStrategy, feasible = explorePath(y, strategy[len(strategy)- 1][4])

                                #Check if during the run not enough robots are available if yes this decision has to be backtracked
                                if feasible == False:
                                    #Note that this strategy might be feasible in a later time, but this is not implemented here
                                    unExplored = safestadeUnexplored
                                    visited = safestadeVisited2
                                    flags = safestadeFlags2
                                    guards = safestadeGuards2
                                    robotCountPerNode = safestadeRobotCount2
                                    strategy = safestadeStrategy
                                    enteringTime = safestadeEnteringTime
                                    multiples.pop(counterMultiple)

                                    break

                                #Remove all flags that represent the spare rebots during the exploration of the subtree
                                strategy.extend(subStrategy)
                                #Note that we always have to backtrack since this can't be the last move
                                strategy.append((y,node,robotsNeeded,strategy[len(strategy)- 1][4],strategy[len(strategy)- 1][4] + T.edges[(node,y)].time))

                                robotCountPerNode[node] += robotsNeeded
                                robotCountPerNode[y] -= robotsNeeded
                                # Save when the robots return from this inner path
                                welcomeBackTime[counter] = strategy[len(strategy)- 1][4]
                                counter += 1
                            else: #Only if the for loop is not left with a break

                                # Continue with the next move when all robots have returned from there inner paths
                                enteringTime = max(welcomeBackTime)

                                remainingFlags = []
                                for flag in flags:
                                    if flag[0] == node and flag[2] <= enteringTime:
                                        robotCountPerNode[node] += flag[1]
                                    else: remainingFlags.append(flag)

                                flags = remainingFlags

                               

            return strategy, True


        strategy, clearance = explorePath(root, 0)

        #Determine the finishtime        
        finishTime = -1
        for tuple in visited:
            if finishTime < tuple[1]:
                finishTime = tuple[1]

        #Cut of strategy moves that go over t_finish, meaning unneicessary backtrack moves
        strategy = [move for move in strategy if move[4] <= finishTime]
            
        #Since through flag movements some nodes can be visited earlier than i implemented, i determine the correct visited times here
        realVisitedTime = [sys.maxsize] * len(T.nodes)
        realVisitedTime[root] = 0
        for move in strategy:
            if realVisitedTime[move[1]] > move[4]:
                realVisitedTime[move[1]] = move[4]

        strategy.insert(0, (None,root,availableRobots))

        return strategy, clearance, realVisitedTime
                

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
    # TODO: Implement method
    # ---------------------------------------------------

    def computeClosingExits(graphStrategy, G):
        print("Hello World!")
        return []


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
    # Computing the cross over of two trees so for the Genetic Algorithim
    # ---------------------------------------------------
    def crossOver(parentA, parentB, G):

        # parent = (Tree, root)
        treeA, rootA = parentA
        treeB, rootB = parentB


        # 1. Choose root from one of the parents
        root = random.choice([rootA, rootB])

        # 2. Union of edges from both parents
        parentEdges = set(treeA.edges.keys()) | set(treeB.edges.keys())

        # 3. Find edges of G that are contained in neither
        #    parent
        possibleRandomEdges = [
            edgeKey
            for edgeKey in G.edges.keys()
            if edgeKey not in parentEdges
        ]

        # 4. Add up to two random new edges
        numberOfRandomEdges = min(2, len(possibleRandomEdges))

        randomEdges = random.sample(
            possibleRandomEdges,
            numberOfRandomEdges
        )

        candidateEdges = parentEdges | set(randomEdges)

        # 5. Build temporary graph containing exactly these
        #    candidate edges
        candidateGraph = Graph()

        for node in G.nodes:
            candidateGraph.add_node2(node)

        for u, v in candidateEdges:

            # Get original edge information from G
            edge = G.edges[(u, v)]

            candidateGraph.add_edge(
                candidateGraph.nodes[u],
                candidateGraph.nodes[v],
                edge.time,
                edge.robotType
            )

        # 6. Generate random spanning tree from candidate graph
        childTree = computeRandomSpanningTree(
            candidateGraph,
            root
        )

        return childTree, root
        

    # ---------------------------------------------------
    # Computing the next Generation of Spanning Trees for the Genetic Algorithim
    # ---------------------------------------------------
    def evolve(oldGen, fitnesses, G):
        newGen = []
        sortedGen = [individual for fitness, individual in sorted(zip(fitnesses, oldGen), key=lambda x: x[0])]
        del sortedGen[3:]
        P = [0.6, 0.25, 0.15]
        for i in range(0,5):
            ParentA = random.choices(sortedGen, weights= P,k=1)[0]
            ParentB = random.choices(sortedGen, weights= P,k=1)[0]
            while ParentB == ParentA:
                ParentB = random.choices(sortedGen, weights= P,k=1)[0]

            newGen.append(crossOver(ParentA, ParentB, G))
        return newGen

    # ------------------------------------------------------------
    # THE REAL GRAPH SEARCH ALGORITHIM USING EVERYTHING FROM ABOVE
    # ------------------------------------------------------------

    #If we have that many robots the optimal solution is obvious
    if len(G.nodes) <= availableRobots:
        root = 0
        strategy = []
        for node in G.nodes:
            if node.idx != root:
                path = aStar(G.nodes[root].pos, node.pos, obstacles,distanceMap, alpha)
                strategy.append((root, node.idx, 1, 0, len(path) - 1))

        strategy.insert(0, (None, root, availableRobots))
        return strategy, G, 0
        
    else:
        startingTime = time.monotonic()
        bestFitness = np.inf
        bestStrategy = None
        bestTree = None
        counter = [0] * startNodes
        currGen = []
        checkedTreesCounter = 0
        #------------------------------------------------------------
        # INITALIZING THE STARTING POPULATION
        #------------------------------------------------------------
        for i in range(0,10):
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
        # ANYTIME ALGORITHIM
        #------------------------------------------------------------

        while time.monotonic() - startingTime < availableTime:
            fitnesses = []
            for indivium in currGen:
                if time.monotonic() - startingTime >= availableTime:
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

            if time.monotonic() - startingTime >= availableTime:
                break      
            
            currGen = evolve(currGen, fitnesses, G) 
            
        if bestFitness == np.inf:
            bestStrategy = computeClosingExits(bestStrategy,G)
            print(availableRobots)
            print("HAALLLO")
            bestTree = G

        return bestStrategy, bestTree, checkedTreesCounter

