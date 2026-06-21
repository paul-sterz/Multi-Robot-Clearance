import numpy as np
import streamlit as st
import plotly.graph_objects as go

from GraphBuilder import graphBuilder
from Searcher import graphSearch
from TrajectoryPlanning import computeTrajectory
from Graph import Graph, Node, Edge


# --------------------------------------------------
# SIMPLE DETECTION FUNCTION
# -> Just returing all cells within radius ignoring obstacles
# --------------------------------------------------

def detectionFnc(p, obstacles, radius=3):
    H, W = obstacles.shape
    px, py = p
    visible = set()
    for x in range(H):
        for y in range(W):
            if obstacles[x, y] == 1:
                continue
            if np.sqrt((x - px) ** 2 + (y - py) ** 2) <= radius:
                visible.add((x, y))
    return visible


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Multi-Robot Clearance", layout="wide")
st.title("Multi-Robot Clearance")


# --------------------------------------------------
# SIDEBAR WITH SLIDERS
# Note: Sliderformat is (min, max, default)
# --------------------------------------------------

with st.sidebar:
    st.markdown("## Parameters")
    H = st.slider("Height", 5, 20, 10)
    W = st.slider("Width", 5, 20, 10)
    detection_radius = st.slider("Detection radius", 1, 8, 3)
    num_trees = st.slider("Trees (graphSearch)", 10, 200, 50, step=10)
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
    ("step", 0),
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
priors_arr    = st.session_state.priors
start_region  = st.session_state.start_region

clicked_r, clicked_c = None, None

for r in range(H - 1, -1, -1):  # highest row first → (0,0) ends up bottom-left
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
    priors    = st.session_state.priors.copy()
    root      = st.session_state.start_region if st.session_state.start_region else {(0, 0)}

    with st.spinner("Building graph..."):
        G, shadyEdges, D, startNodes = graphBuilder(
            obstacles, priors,
            lambda p, obs: detectionFnc(p, obs, detection_radius),
            root
        )

    with st.spinner("Computing strategy..."):
        strategy, T = graphSearch(G, shadyEdges, num_trees, 99, startNodes)

    with st.spinner("Planning trajectories..."):
        trajectories = computeTrajectory(obstacles, G, strategy, alpha)

    V = list(range(len(G.nodes)))
    P = [G.nodes[i].pos for i in range(len(G.nodes))]

    regularEdges = list(G.edges.keys())
    treeEdges    = list(T.edges.keys())

    st.session_state.trajectories  = trajectories
    st.session_state.graph_data    = (obstacles, V, P, shadyEdges, regularEdges, H, W)
    st.session_state.strategy      = (strategy, treeEdges)
    st.session_state.step          = 0
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
current_step = st.session_state.step
strategy, treeEdges = st.session_state.strategy
show_spanning_tree  = st.session_state.show_spanning_tree

# Robot counter: how many robots are currently on each node
# Replay strategy up to current_step to compute positions
robot_counts = [0] * len(V)
if strategy:
    robot_counts[strategy[0][1]] = strategy[0][2]   # initial placement
for i in range(1, current_step + 1):
    if i >= len(strategy):
        break
    src, dst, n = strategy[i]
    if src is not None:
        robot_counts[src] = max(0, robot_counts[src] - n)
    robot_counts[dst] = robot_counts[dst] + n


# --------------------------------------------------
# METRICS ROW
# --------------------------------------------------

robot_cost = strategy[0][2] if strategy else "–"

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Nodes", len(V))
with m2:
    st.metric("Regular edges", len(regularEdges))
with m3:
    st.metric("Strategy steps", len(strategy))
with m4:
    st.metric("Robots needed", robot_cost)


# --------------------------------------------------
# BUTTONS
# --------------------------------------------------

col1, col2, col3, col4 = st.columns([1, 1, 1, 4])

with col1:
    if st.button("Next Step", disabled=(current_step >= len(trajectories))):
        if not show_spanning_tree:
            st.session_state.show_spanning_tree = True
        else:
            st.session_state.step += 1
        st.rerun()

with col2:
    if st.button("Reset"):
        st.session_state.step = 0
        st.session_state.show_spanning_tree = False
        st.rerun()

with col3:
    phase = "Navigation graph" if not show_spanning_tree else f"Step {current_step} / {len(trajectories)}"
    st.caption(f"Phase: **{phase}**")


# --------------------------------------------------
# PLOTLY FIGURE
# --------------------------------------------------

COLORS = ["#e76f51", "#2a9d8f", "#e9c46a", "#264653", "#a8dadc", "#f4a261"]

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

# Robot count badges (shown above each node that has robots)
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


# -- Current trajectory only (not all previous ones) -------------------------

if show_spanning_tree and current_step > 0 and current_step <= len(trajectories):
    idx = current_step - 1
    start, goal, robots, path = trajectories[idx]
    if path is not None:
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        color = COLORS[idx % len(COLORS)]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=color, width=4),
            name=f"Step {idx+1} ({robots} robots)",
            hovertemplate=f"Step {idx+1}: {start}→{goal}, {robots} robots<extra></extra>",
        ))

        # Direction arrows every few steps along the path
        step_size = max(1, len(path) // 4)
        for seg in range(step_size, len(path), step_size):
            fig.add_annotation(
                ax=path[seg - 1][1], ay=path[seg - 1][0],
                x=path[seg][1],     y=path[seg][0],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=4,
                arrowsize=1.8,
                arrowwidth=2.5,
                arrowcolor=color,
            )

        # Extra arrow at the very end to make destination clear
        if len(path) >= 2:
            fig.add_annotation(
                ax=path[-2][1], ay=path[-2][0],
                x=path[-1][1],  y=path[-1][0],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=4,
                arrowsize=2.0,
                arrowwidth=3.0,
                arrowcolor=color,
            )


st.plotly_chart(fig, use_container_width=True)