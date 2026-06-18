import numpy as np
from collections import deque
import heapq
from Graph import Graph, Node, Edge

def computeTrajectory(obstacles, G : Graph, strategy, alpha=1):
    #INPUT:
    #obstacles:(H,W) dimensional numpy array that represents the enviroment, 1=obstacle, 0=free
    #P: a list containing for every vertex a corresponding position (x,y) in the 2D obstacles array
    #strategy:A list containing the best strategy where each entry is in the form (source node,target node, amount of Robots)

    #OUTPUT:
    #trajectorys: A list where each entry represents a trajectory for a step of the strategy in the format (strategy step details, list of cells to visit in the correct order)

    # ---------------------------------------------------
    # Computing distance map which saves the manhattan distance to the neares obstacle
    # ---------------------------------------------------
    def computeObstacleDistance(obstacles):

        H, W = obstacles.shape

        #Creating the distance array with same dimensions and infinty for every entry
        dist = np.full((H,W), np.inf)

        q = deque()

        for x in range(H):
            for y in range(W):

                if obstacles[x,y] == 1:
                    #Setting every obstacle
                    dist[x,y] = 0
                    q.append((x,y))

        directions = [(1,0),(-1,0),(0,1),(0,-1)]

        while len(q) > 0:

            x,y = q.popleft()

            for dx,dy in directions:

                nx = x + dx
                ny = y + dy

                if 0 <= nx < H and 0 <= ny < W: #Stay inside obstackle array

                    if dist[nx,ny] > dist[x,y] + 1:
                        #If something changes, compute distances for all neighbours new by adding the changed one to the queue
                        dist[nx,ny] = dist[x,y] + 1
                        q.append((nx,ny))

        return dist
    
    # ---------------------------------------------------
    # A-Star algorithim that finds a path from start to goal 
    # ---------------------------------------------------
    
    def aStar(start, goal, obstacles, distanceMap, alpha):
        #start and goal are points of the form (x,y)

        H, W = obstacles.shape

        # Function that returns the euclidean distance between a and b
        def euclideanDistance(a, b):
            return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

        directions = [(1,0), (-1,0), (0,1), (0,-1)]

        # heap so that minimum distance is always first entry (seems fitting for dijkstra)
        queue = []
        heapq.heappush(queue, (euclideanDistance(start, goal), start))

        # For path reconstruction at the end
        predecessor = {}

        # Cost function (saved in a dictionary of the from: (x,y) | cost(x,y))
        cost = {}
        cost[start] = 0

        #Keeping track of which cells we already visited since the same cell can be added multiple times in the queue
        visited = np.zeros((H,W))

        while len(queue) > 0:

            _, current = heapq.heappop(queue)

            #Check if we already visited 
            if visited[current[0]][current[1]] == 1:
                continue
            
            #Set cell to visited
            visited[current[0]][current[1]] = 1

            #Check if we are at the goal
            if current == goal:

                # Reconstruct path
                path = [goal]

                while path[-1] != start:
                    path.append(predecessor[path[-1]])

                path.reverse()

                return path

            x, y = current
            
            #check all new possible cells
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

                # d(si,si+1) = Manhattan distance (At the momemnt this is always 1 because of our grid represenation)
                stepDistance = abs(dx) + abs(dy)

                df = distanceMap[nx, ny] #distance to next obstacle

                # Equation for the cost function from the paper
                newCost = cost[current] + alpha * stepDistance / df

                #Did we already set a cost for this neighbour somewhere else and if yes is it cheaper?
                if neighbour not in cost or newCost < cost[neighbour]: 

                    cost[neighbour] = newCost

                    predecessor[neighbour] = current

                    heuristic = euclideanDistance(neighbour, goal)

                    fScore = newCost + heuristic

                    heapq.heappush(queue,(fScore, neighbour))

        # No path found
        return None
            
        
    

    # ---------------------------------------------------
    # THE REAL TRAJECTORY ALGORITHIM USING EVERYTHING FROM ABOVE
    # ---------------------------------------------------
    P = []
    for i in range(len(G.nodes)):
        P.append(G.nodes[i].pos)

    distanceMap = computeObstacleDistance(obstacles)
    trajectorys = []
    
    for (start,goal,amountOfRobots) in strategy:
        if start == None:
            continue
        trajectorys.append((start, goal, amountOfRobots ,aStar(P[start],P[goal], obstacles, distanceMap, alpha)))

    return trajectorys
