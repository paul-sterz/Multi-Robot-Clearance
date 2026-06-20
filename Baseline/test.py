import numpy as np
import streamlit as st
import plotly.graph_objects as go

from GraphBuilderV2 import graphBuilder
from Searcher import graphSearch
from TrajectoryPlanning import computeTrajectory
from Graph import Graph, Node, Edge


# --------------------------------------------------
# DETECTION FUNCTION
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
# SIDEBAR
# Remark: Format for a silder is (minVal maxVal defaultVal)
# --------------------------------------------------

with st.sidebar:
    st.markdown("## Parameters")
    H = st.slider("Height", 5, 20, 10)
    W = st.slider("Width", 5, 20, 10)
    detection_radius = st.slider("Detection radius", 1, 8, 3)
    num_trees = st.slider("Trees (graphSearch)", 10, 200, 50, step=10)
    alpha = st.slider("Alpha", 0.1, 5.0, 1.0, step=0.1)
    run = st.button("▶ Run", use_container_width=True, type="primary")


# --------------------------------------------------
# SESSION STATE
# Streamlit verliert alle Variablen bei jedem Rerun.
# st.session_state ist ein Dictionary das zwischen Reruns erhalten bleibt.
# --------------------------------------------------

if "trajectories" not in st.session_state:
    st.session_state.trajectories = None
if "step" not in st.session_state:
    st.session_state.step = 0
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None
if "strategy" not in st.session_state:
    st.session_state.strategy = None


# --------------------------------------------------
# COMPUTE (if run is pressed)
# --------------------------------------------------

if run:
    obstacles = np.zeros((H, W))
    obstacles[1:4, 5] = 1
    obstacles[4, 3:7] = 1
    root = {(0, 0), (0,1), (1,0)}

    with st.spinner("Building graph..."):
        G, shadyEdges, D, startNodes = graphBuilder(obstacles, lambda p, obs: detectionFnc(p, obs, detection_radius), root)

    with st.spinner("Computing strategy..."):
        strategy = graphSearch(G, shadyEdges,num_trees, startNodes)

    with st.spinner("Planning trajectories..."):
        trajectories = computeTrajectory(obstacles, G, strategy, alpha)

    V = []
    for i in range(len(G.nodes)):
        V.append(i)

    P = []
    for i in range(len(G.nodes)):
        P.append(G.nodes[i].pos)

    regularEdges = G.edges.keys()

    # Alles in session_state speichern
    st.session_state.trajectories = trajectories
    st.session_state.graph_data = (obstacles, V, P, shadyEdges, regularEdges, H, W)
    st.session_state.strategy = strategy
    st.session_state.step = 0  


# --------------------------------------------------
# OTHERWISE
# --------------------------------------------------

if st.session_state.graph_data is None:
    st.info("Parameter waehlen und **Run** druecken.")
    st.stop()



# --------------------------------------------------
# LOADING DATA
# --------------------------------------------------

obstacles, V, P, shadyEdges, regularEdges, H, W = st.session_state.graph_data
trajectories = st.session_state.trajectories
current_step = st.session_state.step  # How many trajectorys are already visible
strategy = st.session_state.strategy



# --------------------------------------------------
# METRICS ROW
# --------------------------------------------------

m1, m2, m3, m4 = st.columns(4)

robot_cost = strategy[0][2] if strategy else "–"

with m1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Nodes</div>
        <div class="metric-value">{len(V)}</div>
    </div>""", unsafe_allow_html=True)

with m2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Regular edges</div>
        <div class="metric-value">{len(regularEdges)}</div>
    </div>""", unsafe_allow_html=True)

with m3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Strategy steps</div>
        <div class="metric-value">{len(strategy)}</div>
    </div>""", unsafe_allow_html=True)

with m4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Robots needed</div>
        <div class="metric-value">{robot_cost}</div>
    </div>""", unsafe_allow_html=True)


# --------------------------------------------------
# BUTTONS
# --------------------------------------------------

col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    if st.button("Next Step", disabled=(current_step >= len(trajectories))):
        st.session_state.step += 1
        st.rerun()

with col2:
    if st.button("Reset"):
        st.session_state.step = 0
        st.rerun()

st.caption(f"Schritt {current_step} / {len(trajectories)}")


# --------------------------------------------------
# PLOTLY FIGURE 
# --------------------------------------------------

fig = go.Figure()

fig.update_layout(
    height=600,
    margin=dict(l=10, r=10, t=30, b=10),
    plot_bgcolor="white",
    legend=dict(bordercolor="gray", borderwidth=1),

    xaxis=dict(
        title="Column",
        range=[-0.5, W - 0.5],
        tickmode="array",
        tickvals=list(range(W)),
        dtick=1,
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False,
        fixedrange=True,
        constrain="domain",
    ),

    yaxis=dict(
        title="Row",
        range=[-0.5, H -0.5],      
        tickmode="array",
        tickvals=list(range(H)),
        dtick=1,
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False,
        fixedrange=True,
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
    ),
)


# -- Obstacles ------------------------------------

for x in range(H):
    for y in range(W):
        if obstacles[x, y] == 1:
            fig.add_shape(
                type="rect",
                x0=y - 0.5, x1=y + 0.5,
                y0=x - 0.5, y1=x + 0.5,
                fillcolor="#555555",
                line=dict(width=0),
                layer="below",
            )

# Dummy-Trace for the legend
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers",
    marker=dict(symbol="square", size=12, color="#555555"),
    name="Obstacle",
))


# -- Edges -------------------------------------------------------------------

for idx, (u, v) in enumerate(regularEdges):
    p1, p2 = P[u], P[v]
    fig.add_trace(go.Scatter(
        x=[p1[1], p2[1]],
        y=[p1[0], p2[0]],
        mode="lines",
        line=dict(color="royalblue", width=1.2),
        opacity=0.5,
        showlegend=(idx == 0),
        name="Edge" if idx == 0 else "",
        hoverinfo="skip",
    ))

for idx, (u, v) in enumerate(shadyEdges):
    p1, p2 = P[u], P[v]
    fig.add_trace(go.Scatter(
        x=[p1[1], p2[1]],
        y=[p1[0], p2[0]],
        mode="lines",
        line=dict(color="red", width=1.2),
        opacity=0.5,
        showlegend=(idx == 0),
        name="Edge" if idx == 0 else "",
        hoverinfo="skip",
    ))



# -- Nodes -------------------------------------------------------------------
# P[i] = (row, col) -> x=col=P[i][1], y=row=P[i][0]

node_x = [P[i][1] for i in V]
node_y = [P[i][0] for i in V]

fig.add_trace(go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers+text",
    marker=dict(
        size=22,
        color="tomato",
        line=dict(color="black", width=1.5),
    ),
    text=[str(i) for i in V],
    textposition="middle center",
    textfont=dict(size=9, color="white"),
    name="Node",
    hovertemplate="Node %{text}<br>pos (%{y}, %{x})<extra></extra>",
))


# -- Root --------------------------------------------------------------------

fig.add_trace(go.Scatter(
    x=[P[0][1]], y=[P[0][0]],
    mode="markers",
    marker=dict(size=26, color="limegreen", symbol="star", line=dict(color="black", width=1.5)),
    name="Root",
))


# -- Trajectorys up to the current step ----------------------------------

COLORS = ["#e76f51", "#2a9d8f", "#e9c46a", "#264653", "#a8dadc", "#f4a261"]

for idx in range(current_step):
    start, goal, robots, path = trajectories[idx]
    if path is None:
        continue

    xs = [p[1] for p in path]
    ys = [p[0] for p in path]
    color = COLORS[idx % len(COLORS)]

    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(color=color, width=4),
        name=f"Schritt {idx+1} ({robots} Robots)",
        hovertemplate=f"Schritt {idx+1}: {start}>{goal}, {robots} Roboter<extra></extra>",
    ))

    # Direction arrow in the middle of the trajectory
    if len(path) >= 2:
        mid = len(path) // 2
        fig.add_annotation(
            ax=path[mid-1][1], ay=path[mid-1][0],
            x=path[mid][1],   y=path[mid][0],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2.5,
            arrowcolor=color,
        )


st.plotly_chart(fig, use_container_width=True)

