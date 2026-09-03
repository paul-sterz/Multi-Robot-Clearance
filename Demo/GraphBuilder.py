import random
import numpy as np
from Graph import Graph, Node, Edge
from TrajectoryPlanning import aStar, computeObstacleDistance




def graphBuilder(obstacles, hotspots, detectionFnc, startRegion, l, sigma, alpha):
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
    # PART 1: CALCULATE PRIORS FOR EACH NODE 
    # ---------------------------------------------------
    #TODO Optimize for sped such that a star ist not called for every hotspot cell pair alone but for each hotspot calculates the distances to all cells with dijkstra,...
    obstacleDistance = computeObstacleDistance(obstacles)

    priors = np.zeros((H,W)) 
    weights = [1, l, l**2]

    for cell in E:
        for hotspot in hotspots:
            dist = len(aStar(hotspot[0], cell, obstacles, obstacleDistance , alpha)) -1 
            priors[cell[0], cell[1]]  += weights[hotspot[1]] * np.exp(-(dist)**2/(2*sigma**2))

    val = 0
    for x in range(H):
        for y in range(W):
            val += priors[x,y]
            
    if val != 0:
        for x in range(H):
            for y in range(W):
                priors[x, y] = priors[x,y] / val


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

    #FOR DEMO

    G = Graph()

    # ==================================================
    # NODES
    # ==================================================

    G.add_node(0, (0, 0), 4.075239074726153e-05)
    G.add_node(1, (6, 9), 0.04799394018707494)
    G.add_node(2, (1, 9), 0.137647348077356)
    G.add_node(3, (5, 6), 0.16168491255760328)
    G.add_node(4, (5, 0), 0.0028260906812142667)
    G.add_node(5, (0, 6), 0.14238805578892497)
    G.add_node(6, (8, 1), 0.5161055847724962)
    G.add_node(7, (9, 5), 0.1362577915545183)
    G.add_node(8, (9, 9), 0.09188143176170127)
    G.add_node(9, (3, 3), 0.007535537229425514)


    # ==================================================
    # EDGES
    # ==================================================

    G.add_edge(G.nodes[0], G.nodes[4], 5, 2)
    G.add_edge(G.nodes[0], G.nodes[9], 6, 2)

    G.add_edge(G.nodes[1], G.nodes[2], 5, 2)
    G.add_edge(G.nodes[1], G.nodes[3], 4, 2)

    G.add_edge(G.nodes[2], G.nodes[1], 5, 2)
    G.add_edge(G.nodes[2], G.nodes[3], 7, 2)
    G.add_edge(G.nodes[2], G.nodes[5], 4, 2)

    G.add_edge(G.nodes[3], G.nodes[1], 4, 2)
    G.add_edge(G.nodes[3], G.nodes[2], 7, 2)
    G.add_edge(G.nodes[3], G.nodes[5], 5, 2)
    G.add_edge(G.nodes[3], G.nodes[7], 5, 2)
    G.add_edge(G.nodes[3], G.nodes[9], 5, 2)

    G.add_edge(G.nodes[4], G.nodes[0], 5, 2)
    G.add_edge(G.nodes[4], G.nodes[9], 5, 2)

    G.add_edge(G.nodes[5], G.nodes[2], 4, 2)
    G.add_edge(G.nodes[5], G.nodes[3], 5, 2)

    G.add_edge(G.nodes[6], G.nodes[7], 5, 2)

    G.add_edge(G.nodes[7], G.nodes[3], 5, 2)
    G.add_edge(G.nodes[7], G.nodes[6], 5, 2)
    G.add_edge(G.nodes[7], G.nodes[8], 6, 2)

    G.add_edge(G.nodes[8], G.nodes[7], 6, 2)

    G.add_edge(G.nodes[9], G.nodes[0], 6, 2)
    G.add_edge(G.nodes[9], G.nodes[3], 5, 2)
    G.add_edge(G.nodes[9], G.nodes[4], 5, 2)


    # ==================================================
    # DETECTION SETS
    # ==================================================

    D = [
        {
            (0, 1), (1, 2), (2, 1), (0, 0), (1, 1),
            (0, 3), (2, 0), (3, 0), (0, 2), (2, 2),
            (1, 0)
        },

        {
            (5, 8), (4, 9), (6, 8), (5, 7), (6, 7),
            (3, 9), (4, 8), (6, 6), (5, 9), (6, 9),
            (4, 7)
        },

        {
            (0, 7), (3, 8), (2, 7), (4, 9), (3, 7),
            (1, 8), (0, 9), (2, 9), (1, 7), (3, 9),
            (1, 6), (0, 8), (1, 9), (2, 8)
        },

        {
            (3, 7), (5, 4), (4, 6), (5, 7), (8, 6),
            (6, 5), (6, 8), (4, 5), (5, 6), (4, 8),
            (3, 6), (5, 3), (5, 9), (6, 4), (6, 7),
            (7, 6), (4, 7), (3, 5), (4, 4), (3, 8),
            (5, 5), (5, 8), (2, 6), (6, 6), (7, 5)
        },

        {
            (6, 2), (4, 0), (3, 1), (6, 1), (2, 0),
            (5, 1), (4, 2), (3, 0), (5, 0), (6, 0),
            (5, 3), (3, 2), (4, 1), (5, 2)
        },

        {
            (0, 7), (2, 7), (1, 5), (1, 8), (0, 9),
            (0, 6), (1, 7), (2, 6), (0, 5), (3, 6),
            (1, 6), (0, 8), (2, 5), (2, 8)
        },

        {
            (9, 0), (8, 4), (9, 3), (8, 1), (9, 2),
            (8, 0), (8, 3), (8, 2), (9, 1)
        },

        {
            (6, 5), (8, 7), (9, 6), (9, 5),
            (7, 6), (8, 6), (7, 5), (8, 5)
        },

        {
            (8, 8), (9, 9), (8, 7), (8, 9), (9, 8)
        },

        {
            (4, 3), (3, 1), (5, 4), (5, 1), (2, 2),
            (1, 3), (4, 2), (3, 0), (3, 3), (5, 3),
            (1, 2), (2, 1), (3, 2), (4, 1), (5, 2),
            (1, 1), (0, 3), (2, 3), (6, 3)
        }
    ]


    # ==================================================
    # START NODES
    # ==================================================

    startNodes = 1


    # ==================================================
    # CELL PRIORS
    # ==================================================

    priors = np.array([
        [
            1.7671155949237124e-09,
            1.5674355383940384e-08,
            1.1928263269762003e-07,
            7.800875535245022e-07,
            0.0,
            0.009541205690691768,
            0.008397716602157736,
            0.007459801472548489,
            0.007459724516398689,
            0.00839707000804437
        ],

        [
            6.230215962282265e-07,
            1.1928263269762003e-07,
            7.800875535245022e-07,
            4.39309584686191e-06,
            0.0,
            0.011074534297977586,
            0.011706628459056994,
            0.011820556381039635,
            0.011819995729887033,
            0.011702517798567302
        ],

        [
            3.33986224076797e-06,
            1.5290114609990156e-05,
            4.39309584686191e-06,
            2.135629205634551e-05,
            0.0,
            0.011777550885715136,
            0.011842309247651433,
            0.011824029434226927,
            0.011703155405869702,
            0.011058661939429327
        ],

        [
            1.5290114609990156e-05,
            5.986323352418332e-05,
            0.00020093802528052624,
            8.987977842372149e-05,
            0.0,
            0.008699390127704681,
            0.007563777657659456,
            0.007482114990312889,
            0.008401189655345028,
            0.009538947331986163
        ],

        [
            5.986323352418332e-05,
            0.00020093802528052624,
            0.0005808095444055839,
            0.0014567799847992249,
            0.0032109611696645923,
            0.006339777067381927,
            0.0044501361457207796,
            0.004140937878777557,
            0.005317906916429349,
            0.007272954954643483
        ],

        [
            1.813571244827516e-05,
            7.536975136725582e-05,
            0.0002730751464203588,
            0.000867171362556873,
            0.0024260075225181646,
            0.006003947441843409,
            0.0032761007549483886,
            0.002285928071837241,
            0.003012703178076013,
            0.004826977150447487
        ],

        [
            1.9292266449090115e-05,
            9.027283358810775e-05,
            0.00036173156951854475,
            0.0012423026841392988,
            0.0036601576347356883,
            0.009261828855700153,
            0.004821598899923595,
            0.002147866054246844,
            0.0017949983058553783,
            0.002833121444851832
        ],

        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.017811788265039746,
            0.010390063556401697,
            0.0,
            0.0,
            0.0
        ],

        [
            0.04681667214514512,
            0.0579565431998732,
            0.06437172072867577,
            0.06465546327357942,
            0.0484384846168832,
            0.032682784829786135,
            0.02164927464682496,
            0.017000691243263994,
            0.017129911898427934,
            0.01899188933420306
        ],

        [
            0.047280890300067525,
            0.064257546579723,
            0.0642816923748509,
            0.05804657155369807,
            0.0,
            0.01730634847200142,
            0.010155011685500195,
            0.0,
            0.01899188933420306,
            0.019767049951603232
        ]
    ])


    return G, edges_shady, D, startNodes, priors