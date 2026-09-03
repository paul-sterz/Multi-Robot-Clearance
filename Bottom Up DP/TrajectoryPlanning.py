import numpy as np
from collections import deque
import heapq
from Graph import Graph, Node, Edge


# ---------------------------------------------------------------------------
# NOTE ON THIS REWRITE:
# aStar and computeObstacleDistance used to be nested INSIDE computeTrajectory,
# which meant `from TrajectoryPlanning import aStar` (used in Searcher.py and
# GraphBuilder.py) could not actually work. They are now module-level
# functions, imported the same way everywhere.
# ---------------------------------------------------------------------------


# ---------------------------------------------------
# Computing distance map which saves the manhattan distance to the nearest obstacle
# ---------------------------------------------------
def computeObstacleDistance(obstacles):

    H, W = obstacles.shape

    # Creating the distance array with same dimensions and infinity for every entry
    dist = np.full((H, W), np.inf)

    q = deque()

    for x in range(H):
        for y in range(W):

            if obstacles[x, y] == 1:
                # Setting every obstacle
                dist[x, y] = 0
                q.append((x, y))

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while len(q) > 0:

        x, y = q.popleft()

        for dx, dy in directions:

            nx = x + dx
            ny = y + dy

            if 0 <= nx < H and 0 <= ny < W:  # Stay inside obstacle array

                if dist[nx, ny] > dist[x, y] + 1:
                    # If something changes, compute distances for all neighbours new by adding the changed one to the queue
                    dist[nx, ny] = dist[x, y] + 1
                    q.append((nx, ny))

    return dist


# ---------------------------------------------------
# A-Star algorithm that finds a path from start to goal
# ---------------------------------------------------
def aStar(start, goal, obstacles, distanceMap, alpha):
    # start and goal are points of the form (x,y)

    H, W = obstacles.shape

    # Function that returns the euclidean distance between a and b
    def euclideanDistance(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # heap so that minimum distance is always first entry (seems fitting for dijkstra)
    queue = []
    heapq.heappush(queue, (euclideanDistance(start, goal), start))

    # For path reconstruction at the end
    predecessor = {}

    # Cost function (saved in a dictionary of the form: (x,y) | cost(x,y))
    cost = {}
    cost[start] = 0

    # Keeping track of which cells we already visited since the same cell can be added multiple times in the queue
    visited = np.zeros((H, W))

    while len(queue) > 0:

        _, current = heapq.heappop(queue)

        # Check if we already visited
        if visited[current[0]][current[1]] == 1:
            continue

        # Set cell to visited
        visited[current[0]][current[1]] = 1

        # Check if we are at the goal
        if current == goal:

            # Reconstruct path
            path = [goal]

            while path[-1] != start:
                path.append(predecessor[path[-1]])

            path.reverse()

            return path

        x, y = current

        # check all new possible cells
        for dx, dy in directions:

            nx = x + dx
            ny = y + dy

            # Stay inside map
            if not (0 <= nx < H and 0 <= ny < W):
                continue

            # Check if traversable
            if obstacles[nx, ny] == 1:
                continue

            neighbour = (nx, ny)

            # d(si,si+1) = Manhattan distance (At the moment this is always 1 because of our grid representation)
            stepDistance = abs(dx) + abs(dy)

            df = distanceMap[nx, ny]  # distance to next obstacle

            # Equation for the cost function from the paper
            newCost = cost[current] + alpha * stepDistance / df

            # Did we already set a cost for this neighbour somewhere else and if yes is it cheaper?
            if neighbour not in cost or newCost < cost[neighbour]:

                cost[neighbour] = newCost

                predecessor[neighbour] = current

                heuristic = euclideanDistance(neighbour, goal)

                fScore = newCost + heuristic

                heapq.heappush(queue, (fScore, neighbour))

    # No path found
    return None


# ---------------------------------------------------
# Dijkstra's algorithm computing the single-source shortest path cost from
# start to every reachable free cell. Used for hotspots
# ---------------------------------------------------
def dijkstra(start, obstacles, distanceMap, alpha):
    # start is a point of the form (x,y)
    # Returns (cost, predecessor): cost[p] is the cheapest cost from start to
    # p, predecessor[p] is the previous cell on that cheapest path (used for
    # path reconstruction / hop counting, see pathLength below).

    H, W = obstacles.shape

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    queue = []
    heapq.heappush(queue, (0, start))

    predecessor = {}

    cost = {}
    cost[start] = 0

    visited = np.zeros((H, W))

    while len(queue) > 0:

        currentCost, current = heapq.heappop(queue)

        if visited[current[0]][current[1]] == 1:
            continue

        visited[current[0]][current[1]] = 1

        x, y = current

        for dx, dy in directions:

            nx = x + dx
            ny = y + dy

            if not (0 <= nx < H and 0 <= ny < W):
                continue

            if obstacles[nx, ny] == 1:
                continue

            neighbour = (nx, ny)

            stepDistance = abs(dx) + abs(dy)

            df = distanceMap[nx, ny]

            newCost = currentCost + alpha * stepDistance / df

            if neighbour not in cost or newCost < cost[neighbour]:

                cost[neighbour] = newCost

                predecessor[neighbour] = current

                heapq.heappush(queue, (newCost, neighbour))

    return cost, predecessor


# ---------------------------------------------------
# Number of grid steps (hops) from start to goal along a Dijkstra
# predecessor tree, equivalent to len(path) - 1 but without needing to
# materialize the path itself. Returns None if goal is unreachable from start.
# ---------------------------------------------------
def pathLength(predecessor, start, goal):

    if goal == start:
        return 0

    if goal not in predecessor:
        return None

    steps = 0
    node = goal

    while node != start:
        node = predecessor[node]
        steps += 1

    return steps



# ---------------------------------------------------
# Interpolate a robot's continuous grid position along a path at time t,
# given it departed the source at t_departure and arrives at the target
# at t_arrival. Needed because with parallel moves several robots can be
# "in flight" at the same simulation time t.
# ---------------------------------------------------
def positionAtTime(path, t_departure, t_arrival, t):

    if path is None or len(path) == 0:
        return None

    if t <= t_departure:
        return path[0]

    if t_arrival <= t_departure or t >= t_arrival:
        return path[-1]

    frac = (t - t_departure) / (t_arrival - t_departure)
    posF = frac * (len(path) - 1)
    idx = int(posF)
    rem = posF - idx

    if idx >= len(path) - 1:
        return path[-1]

    x0, y0 = path[idx]
    x1, y1 = path[idx + 1]

    return (x0 + (x1 - x0) * rem, y0 + (y1 - y0) * rem)


# ---------------------------------------------------
# Compute how many robots are stationed at every node at a given time t.
# Works with the new strategy format where each move is
# (source, target, robots, t_departure, t_arrival), except the very first
# entry which is the initial placement (None, root, availableRobots) with
# no timing info (robots are considered present from t=0 onward).
# ---------------------------------------------------
def computeRobotCounts(strategy, numNodes, t):

    counts = [0] * numNodes

    for move in strategy:

        if len(move) == 3:
            # Initial placement: (None, root, availableRobots), always active from t=0
            _, target, robots = move
            counts[target] += robots
            continue

        source, target, robots, t_departure, t_arrival = move

        if t_departure <= t:
            counts[source] -= robots

        if t_arrival <= t:
            counts[target] += robots

    return counts


# ---------------------------------------------------
# THE REAL TRAJECTORY ALGORITHM USING EVERYTHING FROM ABOVE
# ---------------------------------------------------
def computeTrajectory(obstacles, G: Graph, strategy, alpha=1, distanceMap=None):
    # INPUT:
    # obstacles: (H,W) dimensional numpy array that represents the environment, 1=obstacle, 0=free
    # G: a Graph object representing the given Graph
    # strategy: list of moves. First entry is (None, root, availableRobots).
    #           All following entries are (source, target, robots, t_departure, t_arrival)
    # distanceMap: optional precomputed obstacle-distance map (avoids recomputation
    #              if the caller, e.g. graphSearch, already built one)

    # OUTPUT:
    # trajectories: list of dicts, one per movement (the initial placement is skipped
    #               since there is nothing to animate), each with:
    #               source, target, robots, t_departure, t_arrival, path (grid cells)

    if distanceMap is None:
        distanceMap = computeObstacleDistance(obstacles)

    P = [G.nodes[i].pos for i in range(len(G.nodes))]

    trajectories = []

    for move in strategy:

        if len(move) == 3:
            # Initial placement, nothing to move along a path
            continue

        source, target, robots, t_departure, t_arrival = move

        if source is None:
            continue

        path = aStar(P[source], P[target], obstacles, distanceMap, alpha)

        trajectories.append({
            "source": source,
            "target": target,
            "robots": robots,
            "t_departure": t_departure,
            "t_arrival": t_arrival,
            "path": path,
        })

    return trajectories