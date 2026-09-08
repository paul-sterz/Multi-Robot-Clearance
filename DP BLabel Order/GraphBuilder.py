import random
import numpy as np
from Graph import Graph, Node, Edge
from TrajectoryPlanning import aStar, computeObstacleDistance, dijkstra, pathLength




def graphBuilder(obstacles, hotspots, detectionFnc, startRegion, l, sigma, alpha, epsilon = 0.05):
    # INPUT:
    # obstacles: (H,W) dimensional numpy array that represents the enviroment, 1=obstacle, 0=free
    # hotspots: List of hotspots in format (pos, low/medium/high) beeing ((x,y),0/1/2)
    # detectionFnc: function p -> set of detected cells
    # startRegion: set of tuples of point where the robots can start
    # k: factor of how much eache hotspot category increases probability
    # alpha: Constant needed for a-star algorithim

    # OUTPUT:
    # G: a Graph object representing the created Graph using only the regular edges
    # edges_shady:a list containing all shady edges in the form (i,j)
    # D :a list containing the detection set for every point in P

    

    # ---------------------------------------------------
    # PART 0: BUILD FREE SPACE SET E
    # ---------------------------------------------------

    H, W = obstacles.shape

    E = set() 
    for x in range(H):
        for y in range(W):
            if obstacles[x, y] == 0:
                E.add((x, y))


    # ---------------------------------------------------
    # PART 1: CALCULATE PRIORS FOR EACH CELL
    # ---------------------------------------------------

    obstacleDistance = computeObstacleDistance(obstacles)

    priors = np.zeros((H,W))
    weights = [1, l, l**2]

    # Gaussian mixture
    for hotspot in hotspots:
        _, predecessor = dijkstra(
            hotspot[0],
            obstacles,
        )

        for cell in E:
            dist = pathLength(
                predecessor,
                hotspot[0],
                cell
            )

            if dist is None:
                continue

            priors[cell[0], cell[1]] += (
                weights[hotspot[1]]
                * np.exp(-(dist**2) / (2 * sigma**2))
            )


    # ---------------------------------------------------
    # NORMALIZE + UNIFORM BACKGROUND PRIOR
    # ---------------------------------------------------

    if len(E) > 0:
        val = sum(priors[x, y] for x, y in E)
        uniformPrior = 1.0 / len(E)

        if val > 0:
            # Normalize Gaussian mixture
            for x, y in E:
                priors[x, y] /= val

            # Add background uncertainty
            for x, y in E:
                priors[x, y] = (
                    (1 - epsilon) * priors[x, y]
                    + epsilon * uniformPrior
                )

        else:
            # No hotspots -> uniform distribution
            for x, y in E:
                priors[x, y] = uniformPrior


    # ---------------------------------------------------
    # PART 2: VERTEX GENERATION
    # ---------------------------------------------------

    G = Graph()
    D = []   # detection sets

    covered = set()
    uncovered = E
    k = 0
    #Adding start region
    while len(startRegion & uncovered) > 0:
        root = random.choice(tuple(startRegion & uncovered))
        Di = set(detectionFnc(root, obstacles))
        Di = Di & E 
        D.append(Di)
        p = 0
        for (x,y) in Di:
            p += priors[x,y]
        G.add_node(k,root, p)
        covered = covered | Di 
        uncovered = uncovered - Di
        k += 1

    startNodes = k

    i = startNodes

    while len(uncovered) > 0:

        pi = random.choice(tuple(uncovered))

        if obstacles[pi[0]][pi[1]] == 1:
            continue

        Di = set(detectionFnc(pi, obstacles))
        Di = Di & E  # Only consider intersection with traversable points

        D.append(Di)

        # Prior of node is the sum over the detection set
        p = 0
        for (x,y) in Di:
            p += priors[x,y]

        G.add_node(i,pi, p)
        covered = covered | Di #Joining covered with Di
        uncovered = uncovered - Di #Difference uncovered with Di
        i += 1

    # ---------------------------------------------------
    # PART 3: BOUNDARY COMPUTATION
    # Note: Each cell in the detection set with neighbours outside or a Node outside the detection set with a neighbour inside is contained in the boundary 
    # ---------------------------------------------------

    def compute_boundary(Di):
        boundary = set()

        for (x, y) in E:
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

            if (x, y) in Di:
                for nx, ny in neighbors:
                    if (nx, ny) in E and (nx, ny) not in Di:
                        boundary.add((x, y))
                        break
            elif (x,y) not in Di:
                for nx, ny in neighbors:
                    if (nx, ny) in E and (nx, ny) in Di:
                        boundary.add((x, y))
                        break
        return boundary

    boundaries = [compute_boundary(Di) for Di in D] #Calculating boundaries for every detection set


    # ---------------------------------------------------
    # PART 4: EDGE CONSTRUCTION
    # ---------------------------------------------------

    edges_shady = []

    n = len(D)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            Gij = boundaries[i] & D[j]

            if len(Gij) == 0:
                continue

            is_shady = False

            for k in range(n):
                if k == i or k == j:
                    continue

                Gik = boundaries[i] & D[k]

                # dominance check 
                if Gij < Gik:
                    is_shady = True
                    break

            if is_shady:
                edges_shady.append((i, j))
            else:
                trajectory = aStar(G.nodes[i].pos,G.nodes[j].pos, obstacles, obstacleDistance, alpha)
                G.add_edge(G.nodes[i], G.nodes[j],len(trajectory) - 1,2)

    return G, edges_shady, D, startNodes, priors