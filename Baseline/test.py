import numpy as np
import matplotlib.pyplot as plt

from GraphBuilderV2 import graphBuilder
from Searcher import graphSearch
from TrajectoryPlanning import computeTrajectory


# --------------------------------------------------
# SIMPLE DETECTION FUNCTION
# --------------------------------------------------

def detectionFnc(p, obstacles, radius = 3):

    H, W = obstacles.shape

    px, py = p

    visible = set()

    for x in range(H):
        for y in range(W):

            if obstacles[x,y] == 1:
                continue

            if np.sqrt((x-px)**2 + (y-py)**2) <= radius:
                visible.add((x,y))

    return visible


# --------------------------------------------------
# TEST ENVIRONMENT
# --------------------------------------------------

H = 10
W = 10

obstacles = np.zeros((H,W))

# vertical wall
obstacles[1:4,5] = 1

# horizontal wall
obstacles[4,3:7] = 1

#root
root = (0,0)
rootidx = 0

# --------------------------------------------------
# PART 1: Build Graph
# --------------------------------------------------

V, P, D, regularEdges, shadyEdges = graphBuilder(obstacles,detectionFnc, root)

print("Vertices:", P)
print("Regular edges:", regularEdges)
print("Shady edges:", len(shadyEdges))


# --------------------------------------------------
# PART 2: Compute Strategy
# --------------------------------------------------

strategy = graphSearch(V,regularEdges,rootidx,numOfTrees=50)

print("-------------------------------------------")
print("Strategy:")
for step in strategy:
    print(step)


# --------------------------------------------------
# PART 3: Compute Trajecotrys from Strategy
# --------------------------------------------------

trajectories = computeTrajectory(obstacles,P,strategy,1)

print("-------------------------------------------")
for step in trajectories:
    print(step)


# --------------------------------------------------
# VISUALIZATION 
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10,10))

# Show Map
ax.imshow(obstacles,origin="lower",cmap="Greys")


# Make grid visable

H, W = obstacles.shape

ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
ax.set_yticks(np.arange(-0.5, H, 1), minor=True)

ax.grid(which="minor", color="lightgray", linewidth=0.5)

ax.tick_params(which="minor", bottom=False, left=False)

# Graph Edges (blue)

for u, v in regularEdges:

    p1 = P[u]
    p2 = P[v]

    ax.plot([p1[1], p2[1]],[p1[0], p2[0]],color="blue",linewidth=1.5,alpha=0.7)


# Nodes (Points)

for i, (x, y) in enumerate(P):

    ax.scatter(y,x,s=150, color="red",edgecolors="black",zorder=10)

    ax.text(y,x,str(i),fontsize=10,ha="center",va="center",color="white",zorder=11)


# Trajectories (green)

for start, goal, robots, path in trajectories:

    if path is None:
        continue

    xs = [p[1] for p in path]
    ys = [p[0] for p in path]

    ax.plot(xs, ys, color="lime", linewidth=3, zorder=5)

    #Arrows

    if len(path) >= 2:

        mid = len(path) // 2

        dx = xs[mid] - xs[mid-1]
        dy = ys[mid] - ys[mid-1]

        ax.quiver(
            xs[mid-1],
            ys[mid-1],
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="lime",
            width=0.005,
            zorder=6
        )

# --------------------------------------------------

ax.set_title("Multi-Robot Clearance Test")

plt.tight_layout()
plt.show()