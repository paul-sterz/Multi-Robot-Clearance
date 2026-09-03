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
    # COMPUTING B-LABELS and efficiency labels depending on available robots
    # ---------------------------------------------------
    def computeLabelsWithBudget(T: Graph, root, parent, maxRobots: int):

        bLabels = {}
        robotTable = {}
        policyTable = {}

        children = [y.idx for y in T.adj[T.nodes[root]] if y.idx != parent]

        # ---------------- LEAF ----------------
        if len(children) == 0:
            bLabels[(parent, root)] = 1

            edgeTime = T.edges[(parent, root)].time if parent is not None else 1
            prior = T.nodes[root].prior

            robotTable[(parent, root)] = {
                r: (edgeTime, prior / edgeTime)
                for r in range(1, maxRobots + 1)
            }

            policyTable[(parent, root)] = {
                r: [] for r in range(1, maxRobots + 1)
            }

            return bLabels, robotTable, policyTable, prior, edgeTime

        # ---------------- INNER NODE ----------------
        childBLabel = {}
        childPrior = {}
        childTimeTable = {}

        totalPrior = T.nodes[root].prior

        for child in children:
            subB, subTable, subPolicy, subPrior, _ = computeLabelsWithBudget(
                T, child, root, maxRobots
            )

            bLabels.update(subB)
            robotTable.update(subTable)
            policyTable.update(subPolicy)

            childBLabel[child] = subB[(root, child)]
            childPrior[child] = subPrior
            childTimeTable[child] = {
                r: t for r, (t, _eff) in subTable[(root, child)].items()
            }

            totalPrior += subPrior

        # ---------------- B-LABEL ----------------
        childLabelsSorted = sorted(childBLabel.values(), reverse=True)

        p1 = childLabelsSorted[0]
        p2 = childLabelsSorted[1] if len(childLabelsSorted) > 1 else 0

        bLabelRoot = p1 + 1 if p1 == 1 else max(p1, p2 + 1)
        bLabels[(parent, root)] = bLabelRoot

        # ---------------- FEASIBILITY ----------------
        if maxRobots < bLabelRoot:
            robotTable[(parent, root)] = {}
            policyTable[(parent, root)] = {}

            return bLabels, robotTable, policyTable, totalPrior, float('inf')

        edgeTime = T.edges[(parent, root)].time if parent is not None else 1

        # ---------------- BEST POLICY FOR EACH ROBOT BUDGET ----------------
        table = {}
        policies = {}

        for r in range(bLabelRoot, maxRobots + 1):
            bestChildrenTime, bestSchedule = _bestChildrenSchedule(
                children, childBLabel, childTimeTable, childPrior, r
            )

            time_r = edgeTime + bestChildrenTime

            table[r] = (time_r, totalPrior / time_r)
            policies[r] = bestSchedule

        robotTable[(parent, root)] = table
        policyTable[(parent, root)] = policies

        return bLabels, robotTable, policyTable, totalPrior, table[maxRobots][0]


    def _bestChildrenSchedule(children, childBLabel, childTimeTable, childPrior, r):
        """
        Returns:
            bestTime:
                Minimal total clearance time of all child subtrees.

            schedule:
                Ordered list of sequential batches.
                Children inside one batch run in parallel.

                [
                    {
                        "children": [...],
                        "robots": {child: robots, ...},
                        "time": batchTime,
                        "prior": batchPrior,
                        "efficiency": batchPrior / batchTime
                    },
                    ...
                ]
        """

        k = len(children)

        if k == 0:
            return 0.0, []

        fullMask = (1 << k) - 1

        # ---------------- COST OF EVERY POSSIBLE PARALLEL BATCH ----------------
        costOfSubset = [float('inf')] * (1 << k)
        allocationOfSubset = [None] * (1 << k)

        costOfSubset[0] = 0.0
        allocationOfSubset[0] = {}

        for mask in range(1, 1 << k):
            subset = [children[i] for i in range(k) if mask & (1 << i)]

            if sum(childBLabel[c] for c in subset) > r:
                continue

            batchTime, allocation = _parallelBatchCost(
                subset, childTimeTable, r
            )

            costOfSubset[mask] = batchTime
            allocationOfSubset[mask] = allocation

        # ---------------- DP OVER PARTITIONS ----------------
        dp = [float('inf')] * (1 << k)
        choice = [None] * (1 << k)

        dp[0] = 0.0

        for mask in range(1, 1 << k):
            low = mask & (-mask)
            sub = mask

            while sub > 0:
                if sub & low:
                    candidate = costOfSubset[sub] + dp[mask ^ sub]

                    if candidate < dp[mask]:
                        dp[mask] = candidate
                        choice[mask] = sub

                sub = (sub - 1) & mask

        if dp[fullMask] == float('inf'):
            return float('inf'), []

        # ---------------- RECONSTRUCT OPTIMAL PARTITION ----------------
        schedule = []
        mask = fullMask

        while mask != 0:
            chosenMask = choice[mask]

            if chosenMask is None:
                return float('inf'), []

            subset = [
                children[i]
                for i in range(k)
                if chosenMask & (1 << i)
            ]

            batchTime = costOfSubset[chosenMask]
            batchPrior = sum(childPrior[c] for c in subset)
            batchEfficiency = batchPrior / batchTime

            schedule.append({
                "children": subset,
                "robots": allocationOfSubset[chosenMask],
                "time": batchTime,
                "prior": batchPrior,
                "efficiency": batchEfficiency
            })

            mask ^= chosenMask

        # ---------------- ORDER BATCHES BY PRIOR PER TIME ----------------
        schedule.sort(key=lambda batch: batch["efficiency"], reverse=True)

        return dp[fullMask], schedule


    def _parallelBatchCost(subset, childTimeTable, r):
        """
        Returns:
            batchTime:
                Minimal time in which all children of the subset can be
                cleared in parallel.

            allocation:
                child -> required number of robots
        """

        if len(subset) == 0:
            return 0.0, {}

        candidateTimes = set()

        for c in subset:
            candidateTimes.update(childTimeTable[c].values())

        for T in sorted(candidateTimes):
            allocation = {}
            totalRobots = 0
            feasible = True

            for c in subset:
                minRi = None

                for ri in sorted(childTimeTable[c]):
                    if childTimeTable[c][ri] <= T:
                        minRi = ri
                        break

                if minRi is None:
                    feasible = False
                    break

                allocation[c] = minRi
                totalRobots += minRi

            if feasible and totalRobots <= r:
                return T, allocation

        return float('inf'), None
    
    # ---------------------------------------------------
    # CALCULATING A STRATEGY FOR TREES AND TRANSFORMING IT TO THE GRAPH
    # Note: A Strategy is safed in the format [(source node idx,target node idx, amount of Robots, time_Depature, time_Arrival),...]
    # ---------------------------------------------------
                    
    def treeSearch(T : Graph, root, availableRobots, G : Graph):
        # Note: root is the index of the starting node, not the node itself.
        #
        # computeLabelsWithBudget already tells us, for every tree node and
        # every robot budget r, the cheapest way to batch its children and
        # the order (descending prior-per-time) that reaches the best
        # efficiency. explore() walks the tree top-down and, at every node,
        # tries that r-robot plan first. Executing a plan can fail only
        # because of graph-edge recontamination (a node needs a guard/flag
        # that isn't available) -- the BLabel/policy tables don't know about
        # G at all, so this is the one thing that can go wrong at runtime.
        # When it does, we backtrack to before the node's decision and retry
        # the policy for one robot less (which frees that robot as an
        # immediately available flag), all the way down to the node's own
        # BLabel. If even that fails, the last resort is to clear the
        # children strictly one at a time, cheapest (smallest own BLabel)
        # first -- this commits the least robots possible at every step and
        # so maximises the number of idle robots available as flags.

        bLabels, robotTable, policyTable, _, _ = computeLabelsWithBudget(T, root, None, availableRobots)

        n = len(T.nodes)

        # S is the single source of truth for all mutable search state, so
        # that a "safestade" is just a (deep) copy of this dict's contents.
        S = {
            "visited": [(False, 0.0)] * n,  # visited[i] = (wasVisited, firstVisitTime)
            "robots":  [0] * n,             # robots physically present at each node right now
            "flags":   [],                  # idle robots, retraceable anywhere: [node, count, availableFrom]
            "guards":  [],                  # robots pinned down: [node, {enemyNodeIdx, ...}, releaseNotBefore]
        }
        S["visited"][root] = (True, 0.0)
        S["robots"][root] = availableRobots

        def snapshot():
            return (list(S["visited"]), list(S["robots"]), copy.deepcopy(S["flags"]), copy.deepcopy(S["guards"]))

        def restore(snap):
            S["visited"], S["robots"], S["flags"], S["guards"] = snap

        def treeChildren(node, parent):
            return [y.idx for y in T.adj[T.nodes[node]] if y.idx != parent]

        def allFinished():
            return all(v for v, _ in S["visited"]) #True or False 

        def markVisited(node, t):
            # Marks node visited at time t (or improves its recorded time
            # if it is reached earlier than previously known, e.g. via a
            # preemptive flag dispatch). Releases any guard whose enemies
            # have all been dealt with.
            wasVisited, oldT = S["visited"][node]
            if not wasVisited:
                S["visited"][node] = (True, t)
                remaining = []
                for gNode, enemies, releaseNotBefore in S["guards"]:
                    if node in enemies:
                        enemies.discard(node)
                        releaseNotBefore = max(releaseNotBefore, t)
                    if len(enemies) == 0:
                        S["flags"].append([gNode, 1, releaseNotBefore])
                    else:
                        remaining.append([gNode, enemies, releaseNotBefore])
                S["guards"] = remaining
            elif t < oldT:
                S["visited"][node] = (True, t)

        def graphEnemies(node, exclude, atTime):
            # Neighbours of node in the real graph G (outside `exclude`,
            # normally the children about to be departed to) that are not
            # yet safely secured by `atTime`: either never visited, or
            # visited but only at a later point in the overall timeline.
            enemies = set()
            for nb in G.adj[G.nodes[node]]:
                if nb.idx in exclude:
                    continue
                vis, vt = S["visited"][nb.idx]
                #TODO Save vt - atTime so and make a flag instead of a guard with min guard time
                if not vis or vt > atTime:
                    enemies.add(nb.idx)
            return enemies

        def reabsorbFlagsAt(node, atTime):
            remaining = []
            for f in S["flags"]:
                if f[0] == node and f[2] <= atTime:
                    S["robots"][node] += f[1]
                else:
                    remaining.append(f)
            S["flags"] = remaining

        def retraceFlag(toNode, notBefore, count):
            # Greedily retraces `count` robots from the flags that are
            # already available by `notBefore`, always taking the
            # (cumulatively) closest one first. Flags travel the shortest
            # path, not along tree/graph edges. Returns (moves, arrival) or
            # None if there simply aren't enough reachable, available
            # flags.
            need = count
            moves = []
            arrivals = []
            while need > 0:
                best = None
                for i, f in enumerate(S["flags"]):
                    if f[2] > notBefore:
                        continue
                    path = aStar(T.nodes[f[0]].pos, T.nodes[toNode].pos, obstacles, distanceMap, alpha)
                    if path is None:
                        continue
                    dist = len(path) - 1
                    if best is None or dist < best[1]:
                        best = (i, dist)
                if best is None:
                    return None
                i, dist = best
                f = S["flags"][i]
                take = min(f[1], need)
                arrival = f[2] + dist
                if f[0] != toNode:
                    # A flag already sitting at toNode itself needs no
                    # move -- it is simply already there.
                    moves.append((f[0], toNode, take, f[2], arrival))
                arrivals.append(arrival)
                f[1] -= take
                need -= take
                if f[1] == 0:
                    S["flags"].pop(i)
            return moves, max(arrivals)

        def dispatchPreemptive(targetNode, deadline):
            # Instead of parking a guard at the node that would otherwise
            # need one, send the closest idle flag directly to the enemy
            # node itself: once it is touched, it is visited and the
            # contamination it represents is gone, so no guard was needed
            # at all. This only counts as a resolution if the flag actually
            # gets there by `deadline` -- the moment the node whose
            # departure we're securing must have this enemy already dealt
            # with, i.e. no later than when that node is left empty.
            # Returns the move tuple, or None if no flag can make it.
            best = None
            for i, f in enumerate(S["flags"]):
                path = aStar(T.nodes[f[0]].pos, T.nodes[targetNode].pos, obstacles, distanceMap, alpha)
                if path is None:
                    continue
                dist = len(path) - 1
                arrival = f[2] + dist
                if arrival > deadline:
                    continue
                if best is None or arrival < best[2]:
                    best = (i, f[0], arrival, f[2])
            if best is None:
                return None
            i, origin, arrival, departTime = best
            f = S["flags"][i]
            f[1] -= 1
            if f[1] == 0:
                S["flags"].pop(i)
            markVisited(targetNode, arrival)

            # The arriving robot only becomes a free flag if targetNode
            # itself has no other unresolved neighbours (e.g. its own
            # still-unvisited tree-parent) -- otherwise it has to stay as
            # a guard there instead, exactly like any other node that just
            # got visited.
            stillOwnEnemies = graphEnemies(targetNode, set(), arrival)
            if stillOwnEnemies:
                S["guards"].append([targetNode, stillOwnEnemies, arrival])
            else:
                S["flags"].append([targetNode, 1, arrival])

            return (origin, targetNode, 1, departTime, arrival)

        def secureDeparture(fromNode, exclude, count, enteringTime, canInternalSlide, requireSlideMove):
            # Makes it safe for exactly `count` robots to leave `fromNode`
            # (whether to one destination or several at once -- the caller
            # decides that; this only worries about what fromNode itself
            # needs before anyone may leave). requireSlideMove should be
            # True only when the robots are heading into still-uncleared
            # territory (the 2.5D "don't know what happens along the edge"
            # rule only ever applied to that -- a return trip to an already
            # secured node carries no such risk). Returns a dict describing
            # the outcome, or None if it just isn't possible:
            #   moves               -- flag-retrace/preemption moves to splice in
            #   enteringTime        -- possibly delayed departure time
            #   pendingGuardEnemies -- enemy set a stay-behind robot must guard, or None
            #   pendingFlag         -- True if a stay-behind robot is needed purely
            #                          for slide-move safety (no enemies)
            #   internalSlide       -- True if the caller should delay one robot
            #                          within its own departing group instead
            #                          (no stay-behind robot needed at all)
            allDepart = count == S["robots"][fromNode]
            moves = []

            enemies = graphEnemies(fromNode, exclude, enteringTime)
            stillEnemies = set()
            for e in sorted(enemies):
                # Must actually arrive no later than enteringTime: that's
                # the moment fromNode may go empty, so the enemy has to
                # already be dealt with by then, not merely "scheduled".
                move = dispatchPreemptive(e, deadline=enteringTime)
                if move is not None:
                    moves.append(move)
                else:
                    stillEnemies.add(e)

            if not allDepart:
                # A robot stays at fromNode regardless, so its nonzero
                # headcount alone already prevents recontamination; any
                # enemy we couldn't pre-clear above is simply left for a
                # later, fully-departing batch at fromNode to deal with.
                return {"moves": moves, "enteringTime": enteringTime,
                        "pendingGuardEnemies": None, "pendingFlag": False,
                        "internalSlide": False}

            if stillEnemies:
                # A guard is unavoidable, and it doubles as the slide-move
                # preventer (if one is even needed).
                retrace = retraceFlag(fromNode, enteringTime, 1)
                if retrace is None:
                    return None
                rMoves, arrival = retrace
                moves.extend(rMoves)
                if arrival > enteringTime:
                    enteringTime = arrival
                return {"moves": moves, "enteringTime": enteringTime,
                        "pendingGuardEnemies": stillEnemies, "pendingFlag": False,
                        "internalSlide": False}

            if not requireSlideMove:
                # Nothing left to guard against, and we're heading to an
                # already-secured node -- no stay-behind robot needed.
                return {"moves": moves, "enteringTime": enteringTime,
                        "pendingGuardEnemies": None, "pendingFlag": False,
                        "internalSlide": False}

            if canInternalSlide:
                # No contamination left to guard against -- prefer delaying
                # one robot within the departing group (costs time, not an
                # extra robot) over spending a flag purely for the slide
                # move.
                return {"moves": moves, "enteringTime": enteringTime,
                        "pendingGuardEnemies": None, "pendingFlag": False,
                        "internalSlide": True}

            retrace = retraceFlag(fromNode, enteringTime, 1)
            if retrace is None:
                return None
            rMoves, arrival = retrace
            moves.extend(rMoves)
            if arrival > enteringTime:
                enteringTime = arrival
            return {"moves": moves, "enteringTime": enteringTime,
                    "pendingGuardEnemies": None, "pendingFlag": True,
                    "internalSlide": False}

        def departOne(fromNode, toNode, count, enteringTime):
            # Moves `count` robots from fromNode back to toNode (an
            # already-secured node -- this is only ever used for backtrack
            # returns), handling recontamination guarding for fromNode.
            # Returns (moves, arrivalTime), or None if infeasible.
            secured = secureDeparture(fromNode, {toNode}, count, enteringTime, count >= 2, requireSlideMove=False)
            if secured is None:
                return None
            moves = list(secured["moves"])
            t = secured["enteringTime"]
            # Tree edges are only ever stored in the parent->child
            # direction; departOne always returns child->parent.
            edgeTime = T.edges[(toNode, fromNode)].time

            if secured["internalSlide"]:
                moves.append((fromNode, toNode, count - 1, t, t + edgeTime))
                moves.append((fromNode, toNode, 1, t + edgeTime, t + 2 * edgeTime))
                arrival = t + 2 * edgeTime
            else:
                moves.append((fromNode, toNode, count, t, t + edgeTime))
                arrival = t + edgeTime

            S["robots"][toNode] += count
            S["robots"][fromNode] -= count

            if secured["pendingGuardEnemies"] is not None:
                S["guards"].append([fromNode, secured["pendingGuardEnemies"], arrival])
            elif secured["pendingFlag"]:
                S["flags"].append([fromNode, 1, arrival])

            return moves, arrival

        def executeBatch(node, batchChildren, allocation, enteringTime):
            # Departs `allocation[c]` robots from `node` to each child c in
            # batchChildren simultaneously, handles slide-move / graph
            # recontamination guarding for the departure, recurses into
            # each child, and (unless the whole tree is done) brings the
            # batch's robots back. Returns (strategy, endTime), or None if
            # this batch cannot be carried out at all (no available
            # guard/flag when one is required, or a child subtree failed).
            committed = sum(allocation[c] for c in batchChildren)
            strategy = []

            # The schedule assumed `committed` robots would be available at
            # node for this batch, but an earlier sibling batch's own
            # exploration may have permanently diverted some of its robots
            # elsewhere via the flag economy (e.g. to guard a shortcut edge
            # deep in its subtree) instead of bringing them all home. Top up
            # any shortfall by retracing reinforcements before proceeding.
            shortfall = committed - S["robots"][node]
            if shortfall > 0:
                reinforcement = retraceFlag(node, enteringTime, shortfall)
                if reinforcement is None:
                    return None
                moves, arrival = reinforcement
                strategy.extend(moves)
                if arrival > enteringTime:
                    enteringTime = arrival
                S["robots"][node] += shortfall

            canSlide = any(allocation[c] >= 2 for c in batchChildren)
            secured = secureDeparture(node, set(batchChildren), committed, enteringTime, canSlide, requireSlideMove=True)
            if secured is None:
                return None

            strategy.extend(secured["moves"])
            enteringTime = secured["enteringTime"]
            internalSlideChild = None
            if secured["internalSlide"]:
                internalSlideChild = min(
                    (c for c in batchChildren if allocation[c] >= 2),
                    key=lambda c: T.edges[(node, c)].time,
                )
            largest = max((T.edges[(node, c)].time for c in batchChildren), default=0.0)

            visitTimes = {}
            fullTimes = {}
            for c in batchChildren:
                edgeTime = T.edges[(node, c)].time
                if c == internalSlideChild:
                    strategy.append((node, c, allocation[c] - 1, enteringTime, enteringTime + edgeTime))
                    strategy.append((node, c, 1, enteringTime + largest, enteringTime + largest + edgeTime))
                    visitTimes[c] = enteringTime + edgeTime
                    fullTimes[c] = enteringTime + largest + edgeTime
                else:
                    strategy.append((node, c, allocation[c], enteringTime, enteringTime + edgeTime))
                    visitTimes[c] = enteringTime + edgeTime
                    fullTimes[c] = enteringTime + edgeTime
                S["robots"][c] += allocation[c]
            S["robots"][node] -= committed
            maxArrival = max(fullTimes.values())

            # The stay-behind robot's minimum duty is "until everyone
            # arrived" (slide-move safety); markVisited will push this out
            # further still if it is also guarding enemies that only get
            # resolved later.
            if secured["pendingGuardEnemies"] is not None:
                S["guards"].append([node, secured["pendingGuardEnemies"], maxArrival])
            elif secured["pendingFlag"]:
                S["flags"].append([node, 1, maxArrival])

            for c in batchChildren:
                markVisited(c, visitTimes[c])

            finished = allFinished()

            doneTimes = {}
            for c in batchChildren:
                subStrategy, ok = explore(c, node, fullTimes[c])
                if not ok:
                    return None
                strategy.extend(subStrategy)
                # The last-appended move isn't necessarily the last one in
                # time -- a batch sibling processed earlier in this loop
                # can easily finish later than one processed after it.
                doneTimes[c] = max((m[4] for m in subStrategy), default=fullTimes[c])

            if not finished:
                backTimes = []
                for c in batchChildren:
                    # Bring back whatever is actually at c now, not the
                    # original allocation[c]: c's own exploration may have
                    # shed some of those robots as flags that then got
                    # retraced away elsewhere instead of coming home.
                    result = departOne(c, node, S["robots"][c], doneTimes[c])
                    if result is None:
                        return None
                    moves, arrival = result
                    strategy.extend(moves)
                    backTimes.append(arrival)
                endTime = max(backTimes)
                reabsorbFlagsAt(node, endTime)
            else:
                endTime = maxArrival

            return strategy, endTime

        def runSchedule(node, schedule, enteringTime):
            strategy = []
            t = enteringTime
            for batch in schedule:
                result = executeBatch(node, batch["children"], batch["robots"], t)
                if result is None:
                    return None
                batchStrategy, t = result
                strategy.extend(batchStrategy)
            return strategy, t

        def explore(node, parentInTree, enteringTime):
            children = treeChildren(node, parentInTree)
            if not children:
                return [], True

            bLabel = bLabels[(parentInTree, node)]
            r = S["robots"][node]

            if r < bLabel:
                return [], False

            for rTry in range(r, bLabel - 1, -1):
                snap = snapshot()
                if rTry < r:
                    # The robots we chose not to commit are idle from the
                    # start, immediately available to be retraced anywhere.
                    S["flags"].append([node, r - rTry, enteringTime])
                    S["robots"][node] = rTry
                schedule = policyTable[(parentInTree, node)][rTry]
                result = runSchedule(node, schedule, enteringTime)
                if result is not None:
                    strategy, _ = result
                    return strategy, True
                restore(snap)

            # Last resort: clear children strictly one at a time, cheapest
            # (smallest own BLabel) first -- this is the schedule that
            # commits the fewest robots possible at every point in time.
            snap = snapshot()
            spare = r - bLabel
            if spare > 0:
                S["flags"].append([node, spare, enteringTime])
                S["robots"][node] = bLabel
            fallbackSchedule = [
                {"children": [c], "robots": {c: bLabels[(node, c)]}}
                for c in sorted(children, key=lambda c: bLabels[(node, c)])
            ]
            result = runSchedule(node, fallbackSchedule, enteringTime)
            if result is not None:
                strategy, _ = result
                return strategy, True
            restore(snap)

            return [], False

        strategy, ok = explore(root, None, 0.0)

        if not ok:
            return [(None, root, availableRobots)], False, [sys.maxsize] * n

        # Determine the finish time and cut off strategy moves that go past
        # it (unneeded backtrack moves planned before we knew the subtree
        # they were in would turn out to be the last one finished).
        finishTime = max(t for _, t in S["visited"])
        strategy = [move for move in strategy if move[4] <= finishTime]

        # Flag movements can visit nodes earlier than the naive tree order
        # would suggest, so the true first-visit times are re-derived here.
        realVisitedTime = [sys.maxsize] * n
        realVisitedTime[root] = 0
        for move in strategy:
            if realVisitedTime[move[1]] > move[4]:
                realVisitedTime[move[1]] = move[4]

        strategy.insert(0, (None, root, availableRobots))

        return strategy, True, realVisitedTime


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

