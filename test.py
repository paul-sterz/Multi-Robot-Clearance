import matplotlib.pyplot as plt


def bresenham(x0, y0, x1, y1):
    """Return all grid cells traversed by Bresenham's line algorithm."""

    cells = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy

    while True:
        cells.append((x0, y0))

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x0 += sx

        if e2 < dx:
            err += dx
            y0 += sy

    return cells


# --------------------------------------------------
# Start and target cells
# --------------------------------------------------

x0, y0 = 2, 2
x1, y1 = 11, 7

# Bresenham cells
cells = bresenham(x0, y0, x1, y1)


# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

xmin, xmax = 0, 14
ymin, ymax = 0, 10

# --------------------------------------------------
# Grid
# --------------------------------------------------

ax.set_xticks(range(xmin, xmax + 1))
ax.set_yticks(range(ymin, ymax + 1))
ax.grid(True, linewidth=0.8)


# --------------------------------------------------
# Color all Bresenham cells green
# --------------------------------------------------

for x, y in cells:
    rectangle = plt.Rectangle(
        (x, y),
        1,
        1,
        facecolor="green",
        alpha=0.35,
        edgecolor="black",
        linewidth=0.8
    )
    ax.add_patch(rectangle)


# --------------------------------------------------
# Continuous line between cell centers
# --------------------------------------------------

ax.plot(
    [x0 + 0.5, x1 + 0.5],
    [y0 + 0.5, y1 + 0.5],
    color="black",
    linewidth=2,
    zorder=4
)


# --------------------------------------------------
# Start and target points
# --------------------------------------------------

ax.scatter(
    x0 + 0.5,
    y0 + 0.5,
    color="black",
    s=60,
    zorder=5
)

ax.scatter(
    x1 + 0.5,
    y1 + 0.5,
    color="black",
    s=60,
    zorder=5
)


# --------------------------------------------------
# Formatting
# --------------------------------------------------

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.set_aspect("equal")

ax.set_xlabel("x")
ax.set_ylabel("y")

ax.set_title("Bresenham Line Algorithm")

plt.tight_layout()
plt.show()