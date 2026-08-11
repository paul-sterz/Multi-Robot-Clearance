import numpy as np
import streamlit as st
import plotly.graph_objects as go

from GraphBuilder import graphBuilder
from Searcher import graphSearch
from TrajectoryPlanning import computeTrajectory, computeObstacleDistance, positionAtTime, computeRobotCounts
from Graph import Graph, Node, Edge


# --------------------------------------------------
# SIMPLE DETECTION FUNCTION
# --------------------------------------------------

def detectionFnc(p, obstacles, radius=3):
    H, W = obstacles.shape
    px, py = p

    # Simplified version of the Bresenham Algorithm
    def has_clear_line_of_sight(x0, y0, x1, y1):
        xold = x0
        yold = y0
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy))

        if steps == 0:
            return True

        for i in range(1, steps + 1):
            # Calculating the new position
            x = round(x0 + dx * i / steps)
            y = round(y0 + dy * i / steps)

            # If it's a cross move also check the cells to get there
            if x != xold and y != yold:
                if abs(dy) == abs(dx):
                    if obstacles[x, yold] == 1 or obstacles[xold, y] == 1:
                        return False
                elif steps == abs(dx):
                    if obstacles[x, yold] == 1:
                        return False
                else:
                    if obstacles[xold, y] == 1:
                        return False

            # Check the new position
            if obstacles[x, y] == 1:
                return False

            xold = x
            yold = y

        return True

    visible = set()
    for x in range(H):
        for y in range(W):
            if obstacles[x, y] == 1:
                continue
            if np.sqrt((x - px) ** 2 + (y - py) ** 2) <= radius:
                if has_clear_line_of_sight(px, py, x, y):
                    visible.add((x, y))
    return visible


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Multi-Robot Clearance", layout="wide")
st.title("Multi-Robot Clearance")


# --------------------------------------------------
# SIDEBAR WITH SLIDERS
# Note: Slider format is (min, max, default)
# --------------------------------------------------

with st.sidebar:
    st.markdown("## Parameters")
    H = st.slider("Height", 5, 20, 10)
    W = st.slider("Width", 5, 20, 10)
    detection_radius = st.slider("Detection radius", 1, 8, 3)
    num_trees = st.slider("Trees (graphSearch)", 10, 200, 50, step=10)
    available_robots = st.slider("Available Robots", 1, 50, 10)
    alpha = st.slider("Alpha", 0.1, 5.0, 1.0, step=0.1)

    st.markdown("---")
    run = st.button("▶ Run", use_container_width=True, type="primary")
    if st.button("🗑 Reset grid", use_container_width=True):
        st.session_state.obstacles = np.zeros((H, W))
        st.session_state.priors = np.zeros((H, W))
        st.session_state.start_region = set()
        st.rerun()

    st.markdown("---")
    st.markdown("## Edit Mode")
    edit_mode = st.radio("Click sets:", ["Obstacle", "Prior", "Start Region"], horizontal=False)
    if edit_mode == "Prior":
        prior_value = st.slider("Prior value", 0.0, 1.0, 0.5, step=0.05)
    else:
        prior_value = 0.5  # default, unused


# --------------------------------------------------
# SESSION STATE
# -> Streamlit deletes all variables after each run.
#    st.session_state is retained through all runs
# --------------------------------------------------

def _init_grid(H, W):
    if "obstacles" not in st.session_state or st.session_state.obstacles.shape != (H, W):
        st.session_state.obstacles = np.zeros((H, W))
    if "priors" not in st.session_state or st.session_state.priors.shape != (H, W):
        st.session_state.priors = np.zeros((H, W))
    if "start_region" not in st.session_state:
        st.session_state.start_region = {(0, 0), (0, 1), (1, 0)}  # default

_init_grid(H, W)

for key, default in [
    ("trajectories", None),
    ("current_time", 0),
    ("total_time", 0),
    ("graph_data", None),
    ("strategy", None),
    ("show_spanning_tree", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# --------------------------------------------------
# INTERACTIVE GRID EDITOR  (native Streamlit buttons)
# --------------------------------------------------

st.markdown("### Grid Editor — click a cell to toggle obstacle / set prior")

# CSS to make buttons square and colored
st.markdown("""
<style>
div[data-testid="column"] > div > div > div > button {
    width: 100% !important;
    min-height: 38px !important;
    padding: 2px 0px !important;
    font-size: 11px !important;
    line-height: 1.1 !important;
}
</style>
""", unsafe_allow_html=True)

obstacles_arr = st.session_state.obstacles
priors_arr = st.session_state.priors
start_region = st.session_state.start_region

clicked_r, clicked_c = None, None

for r in range(H - 1, -1, -1):  # highest row first -> (0,0) ends up bottom-left
    cols = st.columns(W)
    for c in range(W):
        with cols[c]:
            # Determine button label and style
            if obstacles_arr[r, c] == 1:
                label = "■"
                btn_type = "primary"
            elif (r, c) in start_region:
                label = "S"
                btn_type = "primary"
            elif priors_arr[r, c] > 0:
                label = f"{priors_arr[r, c]:.2f}"
                btn_type = "secondary"
            else:
                label = " "
                btn_type = "secondary"

            if st.button(label, key=f"cell_{r}_{c}", type=btn_type, use_container_width=True):
                clicked_r, clicked_c = r, c

# Handle click
if clicked_r is not None:
    r, c = clicked_r, clicked_c
    if edit_mode == "Obstacle":
        if st.session_state.obstacles[r, c] == 1:
            st.session_state.obstacles[r, c] = 0
        else:
            st.session_state.obstacles[r, c] = 1
            st.session_state.priors[r, c] = 0.0
            st.session_state.start_region.discard((r, c))
    elif edit_mode == "Start Region":
        if st.session_state.obstacles[r, c] == 0:  # can't place start on obstacle
            if (r, c) in st.session_state.start_region:
                st.session_state.start_region.discard((r, c))
            else:
                st.session_state.start_region.add((r, c))
    else:  # Prior
        if st.session_state.obstacles[r, c] == 0:
            if st.session_state.priors[r, c] == prior_value:
                st.session_state.priors[r, c] = 0.0
            else:
                st.session_state.priors[r, c] = prior_value
    st.rerun()


# --------------------------------------------------
# COMPUTE (if run is pressed)
# --------------------------------------------------

if run:
    obstacles = st.session_state.obstacles.copy()
    priors = st.session_state.priors.copy()
    root = st.session_state.start_region if st.session_state.start_region else {(0, 0)}

    with st.spinner("Building graph..."):
        G, shadyEdges, D, startNodes = graphBuilder(
            obstacles, priors,
            lambda p, obs: detectionFnc(p, obs, detection_radius),
            root,
            alpha,
        )

    with st.spinner("Computing distance map..."):
        distanceMap = computeObstacleDistance(obstacles)

    with st.spinner("Computing strategy..."):
        strategy, T = graphSearch(
            G, num_trees, available_robots, startNodes, obstacles, distanceMap, alpha
        )

    if strategy is None:
        st.error("Keine gültige Strategie gefunden (evtl. zu wenig Roboter für diesen Graphen).")
        st.stop()

    with st.spinner("Planning trajectories..."):
        trajectories = computeTrajectory(obstacles, G, strategy, alpha, distanceMap)

    total_time = max((m["t_arrival"] for m in trajectories), default=0)

    V = list(range(len(G.nodes)))
    P = [G.nodes[i].pos for i in range(len(G.nodes))]

    regularEdges = list(G.edges.keys())
    treeEdges = list(T.edges.keys())

    st.session_state.trajectories = trajectories
    st.session_state.graph_data = (obstacles, V, P, shadyEdges, regularEdges, H, W)
    st.session_state.strategy = (strategy, treeEdges)
    st.session_state.current_time = 0
    st.session_state.total_time = total_time
    st.session_state.show_spanning_tree = False
    st.rerun()


# --------------------------------------------------
# STOP if nothing computed yet
# --------------------------------------------------

if st.session_state.graph_data is None:
    st.info("Parameter wählen, Hindernisse/Priors setzen und **Run** drücken.")
    st.stop()


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

obstacles, V, P, shadyEdges, regularEdges, H, W = st.session_state.graph_data
trajectories = st.session_state.trajectories
current_time = st.session_state.current_time
total_time = st.session_state.total_time
strategy, treeEdges = st.session_state.strategy
show_spanning_tree = st.session_state.show_spanning_tree

# Robot counter per node at the currently selected simulation time
robot_counts = computeRobotCounts(strategy, len(V), current_time)


# --------------------------------------------------
# METRICS ROW
# --------------------------------------------------

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Nodes", len(V))
with m2:
    st.metric("Regular edges", len(regularEdges))
with m3:
    st.metric("Total mission time", total_time)
with m4:
    st.metric("Current time", current_time)
with m5:
    st.metric("Available robots", available_robots if "available_robots" in dir() else "-")


# --------------------------------------------------
# BUTTONS
# One click = one time unit (once the spanning tree is displayed)
# --------------------------------------------------

col1, col2, col3, col4 = st.columns([1, 1, 1, 4])

with col1:
    at_end = show_spanning_tree and current_time >= total_time
    if st.button("Next Step (+1)", disabled=at_end):
        if not show_spanning_tree:
            st.session_state.show_spanning_tree = True
        else:
            st.session_state.current_time = min(current_time + 1, total_time)
        st.rerun()

with col2:
    if st.button("Reset"):
        st.session_state.current_time = 0
        st.session_state.show_spanning_tree = False
        st.rerun()

with col3:
    phase = "Navigation graph" if not show_spanning_tree else f"t = {current_time} / {total_time}"
    st.caption(f"Phase: **{phase}**")

if show_spanning_tree and total_time > 0:
    new_time = st.slider("Jump to time", 0, int(total_time), int(current_time))
    if new_time != current_time:
        st.session_state.current_time = new_time
        st.rerun()


# --------------------------------------------------
# PLOTLY FIGURE
# --------------------------------------------------

COLORS = ["#e76f51", "#2a9d8f", "#e9c46a", "#264653", "#a8dadc", "#f4a261",
          "#9b5de5", "#00bbf9", "#00f5d4", "#f15bb5"]

fig = go.Figure()

fig.update_layout(
    height=600,
    margin=dict(l=10, r=10, t=30, b=10),
    plot_bgcolor="white",
    legend=dict(bordercolor="gray", borderwidth=1),
    xaxis=dict(
        title="Column",
        range=[-0.5, W - 0.5],
        tickmode="array", tickvals=list(range(W)),
        showgrid=True, gridcolor="lightgray",
        zeroline=False, fixedrange=True, constrain="domain",
    ),
    yaxis=dict(
        title="Row",
        range=[-0.5, H - 0.5],
        tickmode="array", tickvals=list(range(H)),
        showgrid=True, gridcolor="lightgray",
        zeroline=False, fixedrange=True,
        scaleanchor="x", scaleratio=1, constrain="domain",
    ),
)


# -- Obstacles ---------------------------------------------------------------

for x in range(H):
    for y in range(W):
        if obstacles[x, y] == 1:
            fig.add_shape(
                type="rect",
                x0=y - 0.5, x1=y + 0.5,
                y0=x - 0.5, y1=x + 0.5,
                fillcolor="#555555", line=dict(width=0), layer="below",
            )

fig.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers",
    marker=dict(symbol="square", size=12, color="#555555"),
    name="Obstacle",
))


# -- Regular edges -----------------------------------------------------------

for idx, (u, v) in enumerate(regularEdges):
    p1, p2 = P[u], P[v]
    fig.add_trace(go.Scatter(
        x=[p1[1], p2[1]], y=[p1[0], p2[0]],
        mode="lines",
        line=dict(color="royalblue", width=1.2),
        opacity=0.5,
        showlegend=(idx == 0),
        name="Regular edge" if idx == 0 else "",
        hoverinfo="skip",
    ))


# -- Shady edges (dashed, different colour) ----------------------------------

for idx, (u, v) in enumerate(shadyEdges):
    p1, p2 = P[u], P[v]
    fig.add_trace(go.Scatter(
        x=[p1[1], p2[1]], y=[p1[0], p2[0]],
        mode="lines",
        line=dict(color="darkorange", width=1.5, dash="dash"),
        opacity=0.6,
        showlegend=(idx == 0),
        name="Shady edge" if idx == 0 else "",
        hoverinfo="skip",
    ))


# -- Spanning tree (only after first "Next Step") ----------------------------

if show_spanning_tree:
    for idx, (u, v) in enumerate(treeEdges):
        p1, p2 = P[u], P[v]
        fig.add_trace(go.Scatter(
            x=[p1[1], p2[1]], y=[p1[0], p2[0]],
            mode="lines",
            line=dict(color="limegreen", width=3.6),
            opacity=0.8,
            showlegend=(idx == 0),
            name="Spanning tree" if idx == 0 else "",
            hoverinfo="skip",
        ))


# -- Nodes -------------------------------------------------------------------

node_x = [P[i][1] for i in V]
node_y = [P[i][0] for i in V]

# Node colour: highlight nodes with robots
node_colors = []
for i in V:
    if robot_counts[i] > 0:
        node_colors.append("gold")
    else:
        node_colors.append("tomato")

fig.add_trace(go.Scatter(
    x=node_x, y=node_y,
    mode="markers+text",
    marker=dict(size=22, color=node_colors, line=dict(color="black", width=1.5)),
    text=[str(i) for i in V],
    textposition="middle center",
    textfont=dict(size=9, color="white"),
    name="Node",
    hovertemplate="Node %{text}<br>pos (%{y}, %{x})<extra></extra>",
))

# Robot count badges (shown above each node that currently has robots)
for i in V:
    if robot_counts[i] > 0:
        fig.add_annotation(
            x=P[i][1] + 0.3, y=P[i][0] - 0.3,
            text=f"🤖×{robot_counts[i]}",
            showarrow=False,
            font=dict(size=10, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=2,
        )


# -- Start region ------------------------------------------------------------
# Highlight all nodes whose position is in the start region

start_region_stored = st.session_state.get("start_region", {(0, 0)})
start_xs = [P[i][1] for i in V if tuple(P[i]) in start_region_stored]
start_ys = [P[i][0] for i in V if tuple(P[i]) in start_region_stored]

if start_xs:
    fig.add_trace(go.Scatter(
        x=start_xs, y=start_ys,
        mode="markers",
        marker=dict(size=26, color="limegreen", symbol="star", line=dict(color="black", width=1.5)),
        name="Start region",
    ))


# -- All movements active AT THE CURRENT TIME t (parallel moves shown together) --
# Every trajectory whose [t_departure, t_arrival] window covers `current_time`
# gets its own path line + an interpolated robot marker. This replaces the old
# "one step = one trajectory" logic, since several robot groups can now be
# in flight simultaneously.

if show_spanning_tree:
    active = [
        m for m in trajectories
        if m["t_departure"] <= current_time <= m["t_arrival"] and m["path"] is not None
    ]

    for idx, m in enumerate(active):
        color = COLORS[idx % len(COLORS)]
        path = m["path"]
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=color, width=3),
            opacity=0.7,
            name=f"{m['source']}→{m['target']} ({m['robots']} robots)",
            hovertemplate=f"{m['source']}→{m['target']}: {m['robots']} robots<extra></extra>",
        ))

        pos = positionAtTime(path, m["t_departure"], m["t_arrival"], current_time)
        if pos is not None:
            fig.add_trace(go.Scatter(
                x=[pos[1]], y=[pos[0]],
                mode="markers",
                marker=dict(size=16, color=color, symbol="diamond", line=dict(color="black", width=1)),
                showlegend=False,
                hovertemplate=f"{m['robots']} robots en route to node {m['target']}<extra></extra>",
            ))


st.plotly_chart(fig, use_container_width=True)