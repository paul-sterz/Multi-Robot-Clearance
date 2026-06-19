import random
import numpy as np
from Graph import Graph, Node, Edge




def graphBuilder(obstacles, detectionFnc, root):
    # INPUT:
    # obstacles: (H,W) dimensional numpy array that represents the enviroment, 1=obstacle, 0=free
    # detectionFnc: function p -> set of detected cells
    # root: tuple of point where the robots start

    # OUTPUT:
    # G: a Graph object representing the created Graph using only the regular edges
    # edges_shady:a list containing all shady edges in the form (i,j)
    #D :a list containing the detection set for every point in P

    

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
    # PART 1: VERTEX GENERATION
    # ---------------------------------------------------

    G = Graph()
    D = []   # detection sets

    covered = set()
    uncovered = E
    
    #Adding root
    G.add_node(0,root)
    Di = set(detectionFnc(root, obstacles))
    Di = Di & E 
    D.append(Di)
    covered = covered | Di 
    uncovered = uncovered - Di

    i = 1

    while len(uncovered) > 0:

        pi = random.choice(tuple(uncovered))

        if obstacles[pi[0]][pi[1]] == 1:
            continue

        G.add_node(i,pi)

        Di = set(detectionFnc(pi, obstacles))
        Di = Di & E  # Only consider intersection with traversable points

        D.append(Di)

        covered = covered | Di #Joining covered with Di
        uncovered = uncovered - Di #Difference uncovered with Di
        i += 1

    # ---------------------------------------------------
    # PART 2: BOUNDARY COMPUTATION
    # Note: Each cell in the detection set with neighbours outside the set is contained in the boundary 
    # ---------------------------------------------------

    def compute_boundary(Di):
        boundary = set()

        for (x, y) in Di:
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

            for nx, ny in neighbors:
                if (nx, ny) in E and (nx, ny) not in Di:
                    boundary.add((x, y))
                    break

        return boundary

    boundaries = [compute_boundary(Di) for Di in D] #Calculating boundaries for every detection set

    # ---------------------------------------------------
    # PART 3: EDGE CONSTRUCTION
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
                G.add_edge(G.nodes[i], G.nodes[j])

    return G, edges_shady, D