
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from GraphBuilder import graphBuilder
from Searcher import graphSearch
from TrajectoryPlanning import (
    computeTrajectory,
    computeObstacleDistance,
    positionAtTime,
    computeRobotCounts,
)
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


st.set_page_config(
    page_title="Multi-Robot Clearance",
    layout="wide",
)

st.title("Multi-Robot Clearance")


# --------------------------------------------------
# SIDEBAR WITH SLIDERS
# --------------------------------------------------

with st.sidebar:

    st.markdown("## Parameters")

    H = st.slider("Height", 5, 20, 10)
    W = st.slider("Width", 5, 20, 10)

    detection_radius = st.slider(
        "Detection radius",
        1,
        8,
        3,
    )

    num_trees = st.slider(
        "Trees (graphSearch)",
        10,
        200,
        50,
        step=10,
    )

    available_robots = st.slider(
        "Available Robots",
        1,
        50,
        10,
    )

    alpha = st.slider(
        "Alpha",
        0.1,
        5.0,
        1.0,
        step=0.1,
    )

    st.markdown("---")

    run = st.button(
        "▶ Run",
        use_container_width=True,
        type="primary",
    )

    if st.button(
        "🗑 Reset grid",
        use_container_width=True,
    ):
        st.session_state.obstacles = np.zeros((H, W))
        st.session_state.priors = np.zeros((H, W))
        st.session_state.start_region = set()
        st.rerun()

    st.markdown("---")

    st.markdown("## Edit Mode")

    edit_mode = st.radio(
        "Click sets:",
        ["Obstacle", "Prior", "Start Region"],
        horizontal=False,
    )

    if edit_mode == "Prior":
        prior_value = st.slider(
            "Prior value",
            0.0,
            1.0,
            0.5,
            step=0.05,
        )
    else:
        prior_value = 0.5

    st.markdown("---")

    st.markdown("## Layers")

    show_regular = st.checkbox(
        "Regular edges",
        value=False,
    )

    show_shady = st.checkbox(
        "Shady edges",
        value=False,
    )

    show_tree = st.checkbox(
        "Spanning tree",
        value=True,
    )

    view_mode = st.radio(
        "View",
        ["Timestep", "Full strategy overview"],
    )


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------


def _init_grid(H, W):

    if (
        "obstacles" not in st.session_state
        or st.session_state.obstacles.shape != (H, W)
    ):
        st.session_state.obstacles = np.zeros((H, W))

    if (
        "priors" not in st.session_state
        or st.session_state.priors.shape != (H, W)
    ):
        st.session_state.priors = np.zeros((H, W))

    if "start_region" not in st.session_state:
        st.session_state.start_region = {
            (0, 0),
            (0, 1),
            (1, 0),
        }


_init_grid(H, W)


for key, default in [
    ("trajectories", None),
    ("current_time", 0),
    ("total_time", 0),
    ("graph_data", None),
    ("strategy", None),

    # Visualization phases:
    #
    # 0 = nothing displayed yet
    # 1 = complete navigation graph
    # 2 = spanning tree highlighted
    #
    ("visualization_phase", 0),
]:

    if key not in st.session_state:
        st.session_state[key] = default


# --------------------------------------------------
# INTERACTIVE GRID EDITOR
# --------------------------------------------------


st.markdown(
    "### Grid Editor — click a cell to toggle obstacle / set prior"
)


# CSS to make buttons square and colored
st.markdown(
    """
<style>
div[data-testid="column"] > div > div > div > button {
    width: 100% !important;
    min-height: 38px !important;
    padding: 2px 0px !important;
    font-size: 11px !important;
    line-height: 1.1 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


obstacles_arr = st.session_state.obstacles
priors_arr = st.session_state.priors
start_region = st.session_state.start_region


clicked_r, clicked_c = None, None


for r in range(H - 1, -1, -1):

    cols = st.columns(W)

    for c in range(W):

        with cols[c]:

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

            if st.button(
                label,
                key=f"cell_{r}_{c}",
                type=btn_type,
                use_container_width=True,
            ):
                clicked_r, clicked_c = r, c


# --------------------------------------------------
# HANDLE GRID CLICK
# --------------------------------------------------


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

        if st.session_state.obstacles[r, c] == 0:

            if (r, c) in st.session_state.start_region:
                st.session_state.start_region.discard((r, c))

            else:
                st.session_state.start_region.add((r, c))

    else:

        if st.session_state.obstacles[r, c] == 0:

            if st.session_state.priors[r, c] == prior_value:
                st.session_state.priors[r, c] = 0.0

            else:
                st.session_state.priors[r, c] = prior_value

    st.rerun()


# --------------------------------------------------
# COMPUTE
# --------------------------------------------------


if run:

    obstacles = st.session_state.obstacles.copy()
    priors = st.session_state.priors.copy()

    root = (
        st.session_state.start_region
        if st.session_state.start_region
        else {(0, 0)}
    )

    with st.spinner("Building graph..."):

        G, shadyEdges, D, startNodes = graphBuilder(
            obstacles,
            priors,
            lambda p, obs: detectionFnc(
                p,
                obs,
                detection_radius,
            ),
            root,
            alpha,
        )

    with st.spinner("Computing distance map..."):

        distanceMap = computeObstacleDistance(obstacles)

    with st.spinner("Computing strategy..."):

        strategy, T = graphSearch(
            G,
            num_trees,
            available_robots,
            startNodes,
            obstacles,
            distanceMap,
            alpha,
        )

    if strategy is None:

        st.error(
            "Keine gültige Strategie gefunden "
            "(evtl. zu wenig Roboter für diesen Graphen)."
        )

        st.stop()

    with st.spinner("Planning trajectories..."):

        trajectories = computeTrajectory(
            obstacles,
            G,
            strategy,
            alpha,
            distanceMap,
        )

    total_time = max(
        (
            m["t_arrival"]
            for m in trajectories
        ),
        default=0,
    )

    V = list(range(len(G.nodes)))

    P = [
        G.nodes[i].pos
        for i in range(len(G.nodes))
    ]

    regularEdges = list(G.edges.keys())

    treeEdges = list(T.edges.keys())

    # Save everything
    st.session_state.trajectories = trajectories

    st.session_state.graph_data = (
        obstacles,
        V,
        P,
        shadyEdges,
        regularEdges,
        H,
        W,
    )

    st.session_state.strategy = (
        strategy,
        treeEdges,
    )

    # Start visualisation at phase 0.
    #
    # First click:
    #     phase 0 -> 1 : show complete graph
    #
    # Second click:
    #     phase 1 -> 2 : highlight tree
    #
    # Following clicks:
    #     increase simulation time

    st.session_state.current_time = 0
    st.session_state.total_time = total_time
    st.session_state.visualization_phase = 0

    st.rerun()


# --------------------------------------------------
# STOP IF NOTHING COMPUTED YET
# --------------------------------------------------


if st.session_state.graph_data is None:

    st.info(
        "Parameter wählen, Hindernisse/Priors setzen "
        "und **Run** drücken."
    )

    st.stop()


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------


(
    obstacles,
    V,
    P,
    shadyEdges,
    regularEdges,
    H,
    W,
) = st.session_state.graph_data


trajectories = st.session_state.trajectories

current_time = st.session_state.current_time

total_time = st.session_state.total_time

strategy, treeEdges = st.session_state.strategy

visualization_phase = st.session_state.visualization_phase


# --------------------------------------------------
# ROBOT COUNTS
# --------------------------------------------------


robot_counts = computeRobotCounts(
    strategy,
    len(V),
    current_time,
)


# --------------------------------------------------
# METRICS ROW
# --------------------------------------------------


m1, m2, m3, m4, m5 = st.columns(5)


with m1:
    st.metric(
        "Nodes",
        len(V),
    )


with m2:
    st.metric(
        "Regular edges",
        len(regularEdges),
    )


with m3:
    st.metric(
        "Total mission time",
        total_time,
    )


with m4:
    st.metric(
        "Current time",
        current_time,
    )


with m5:
    st.metric(
        "Available robots",
        available_robots
        if "available_robots" in dir()
        else "-",
    )


# --------------------------------------------------
# VISUALIZATION CONTROLS
#
# Phase 0:
#     Ready
#
# Phase 1:
#     Full graph
#
# Phase 2:
#     Spanning tree + simulation
# --------------------------------------------------


col1, col2, col3, col4 = st.columns(
    [1, 1, 2, 4]
)


with col1:

    at_end = (
        visualization_phase >= 2
        and current_time >= total_time
    )

    if visualization_phase == 0:
        button_text = "Show Graph"

    elif visualization_phase == 1:
        button_text = "Show Spanning Tree"

    else:
        button_text = "Next Step (+1)"

    if st.button(
        button_text,
        disabled=at_end,
        use_container_width=True,
    ):

        if visualization_phase < 2:

            st.session_state.visualization_phase += 1

        else:

            st.session_state.current_time = min(
                current_time + 1,
                total_time,
            )

        st.rerun()


with col2:

    if st.button(
        "Reset",
        use_container_width=True,
    ):

        st.session_state.current_time = 0
        st.session_state.visualization_phase = 0

        st.rerun()


with col3:

    if visualization_phase == 0:

        phase = "Ready"

    elif visualization_phase == 1:

        phase = "Full graph"

    else:

        phase = (
            f"Spanning tree → "
            f"t = {current_time} / {total_time}"
        )

    st.caption(
        f"Phase: **{phase}**"
    )


# --------------------------------------------------
# TIME SLIDER
# --------------------------------------------------


if (
    visualization_phase >= 2
    and total_time > 0
    and view_mode == "Timestep"
):

    new_time = st.slider(
        "Jump to time",
        0,
        int(total_time),
        int(current_time),
    )

    if new_time != current_time:

        st.session_state.current_time = new_time

        st.rerun()


# --------------------------------------------------
# PLOTLY FIGURE
# --------------------------------------------------


BASE_GRAY = "#9a9a9a"
OBSTACLE_GRAY = "#4a4a4a"

TREE_TEAL = "#2a9d8f"
ROBOT_GOLD = "#e9a72b"
MOTION_ORANGE = "#e76f51"

START_GREEN = "#2a9d8f"
TEXT_DARK = "#264653"


fig = go.Figure()


fig.update_layout(
    height=650,

    margin=dict(
        l=10,
        r=10,
        t=30,
        b=10,
    ),

    plot_bgcolor="white",

    legend=dict(
        bordercolor="lightgray",
        borderwidth=1,
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),

    xaxis=dict(
        title="Column",
        range=[-0.5, W - 0.5],
        tickmode="array",
        tickvals=list(range(W)),
        showgrid=True,
        gridcolor="#eeeeee",
        zeroline=False,
        fixedrange=True,
        constrain="domain",
    ),

    yaxis=dict(
        title="Row",
        range=[-0.5, H - 0.5],
        tickmode="array",
        tickvals=list(range(H)),
        showgrid=True,
        gridcolor="#eeeeee",
        zeroline=False,
        fixedrange=True,
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
    ),
)


# --------------------------------------------------
# OBSTACLES
# --------------------------------------------------


for x in range(H):

    for y in range(W):

        if obstacles[x, y] == 1:

            fig.add_shape(
                type="rect",

                x0=y - 0.5,
                x1=y + 0.5,

                y0=x - 0.5,
                y1=x + 0.5,

                fillcolor=OBSTACLE_GRAY,

                line=dict(width=0),

                layer="below",
            )


fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],

        mode="markers",

        marker=dict(
            symbol="square",
            size=12,
            color=OBSTACLE_GRAY,
        ),

        name="Obstacle",
    )
)


# --------------------------------------------------
# COMPLETE NAVIGATION GRAPH
#
# Phase >= 1:
#     The entire graph is visible.
#
# This is deliberately drawn before the spanning
# tree so that the tree can be placed on top of it.
# --------------------------------------------------


if visualization_phase >= 1:

    for idx, (u, v) in enumerate(regularEdges):

        p1 = P[u]
        p2 = P[v]

        fig.add_trace(
            go.Scatter(
                x=[
                    p1[1],
                    p2[1],
                ],

                y=[
                    p1[0],
                    p2[0],
                ],

                mode="lines",

                line=dict(
                    color=BASE_GRAY,
                    width=1.5,
                ),

                opacity=0.55,

                showlegend=(idx == 0),

                name=(
                    "Navigation graph"
                    if idx == 0
                    else ""
                ),

                hovertemplate=(
                    f"Edge {u} → {v}"
                    "<extra></extra>"
                ),
            )
        )


# --------------------------------------------------
# SHADY EDGES
# --------------------------------------------------


if visualization_phase >= 1 and show_shady:

    for idx, (u, v) in enumerate(shadyEdges):

        p1 = P[u]
        p2 = P[v]

        fig.add_trace(
            go.Scatter(
                x=[
                    p1[1],
                    p2[1],
                ],

                y=[
                    p1[0],
                    p2[0],
                ],

                mode="lines",

                line=dict(
                    color="#f4a261",
                    width=1.2,
                    dash="dash",
                ),

                opacity=0.5,

                showlegend=(idx == 0),

                name=(
                    "Shady edge"
                    if idx == 0
                    else ""
                ),

                hoverinfo="skip",
            )
        )


# --------------------------------------------------
# OPTIONAL REGULAR EDGE HIGHLIGHT
#
# Kept as a layer option. The complete graph is
# already shown in phase 1 even if this checkbox
# is disabled.
# --------------------------------------------------


if (
    visualization_phase >= 1
    and show_regular
):

    for idx, (u, v) in enumerate(regularEdges):

        p1 = P[u]
        p2 = P[v]

        fig.add_trace(
            go.Scatter(
                x=[
                    p1[1],
                    p2[1],
                ],

                y=[
                    p1[0],
                    p2[0],
                ],

                mode="lines",

                line=dict(
                    color=BASE_GRAY,
                    width=2.2,
                ),

                opacity=0.8,

                showlegend=(idx == 0),

                name=(
                    "Regular edge"
                    if idx == 0
                    else ""
                ),

                hoverinfo="skip",
            )
        )


# --------------------------------------------------
# SPANNING TREE HIGHLIGHT
#
# Phase >= 2:
#     Draw tree edges on top of the complete graph.
# --------------------------------------------------


if (
    visualization_phase >= 2
    and show_tree
):

    for idx, (u, v) in enumerate(treeEdges):

        p1 = P[u]
        p2 = P[v]

        fig.add_trace(
            go.Scatter(
                x=[
                    p1[1],
                    p2[1],
                ],

                y=[
                    p1[0],
                    p2[0],
                ],

                mode="lines",

                line=dict(
                    color=TREE_TEAL,
                    width=4,
                ),

                opacity=0.95,

                showlegend=(idx == 0),

                name=(
                    "Spanning tree"
                    if idx == 0
                    else ""
                ),

                hovertemplate=(
                    f"Tree edge {u} → {v}"
                    "<extra></extra>"
                ),
            )
        )


# --------------------------------------------------
# NODES
# --------------------------------------------------


node_x = [
    P[i][1]
    for i in V
]

node_y = [
    P[i][0]
    for i in V
]


highlight_robots = (
    visualization_phase >= 2
    and view_mode == "Timestep"
)


node_colors = []
node_sizes = []
node_text = []
node_text_colors = []


for i in V:

    has_robots = (
        highlight_robots
        and robot_counts[i] > 0
    )

    node_colors.append(
        ROBOT_GOLD
        if has_robots
        else "#c9c9c9"
    )

    node_sizes.append(
        24
        if has_robots
        else 16
    )

    node_text.append(
        str(i)
    )

    node_text_colors.append(
        "black"
        if has_robots
        else "#555555"
    )


fig.add_trace(
    go.Scatter(
        x=node_x,
        y=node_y,

        mode="markers+text",

        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(
                color="black",
                width=1,
            ),
        ),

        text=node_text,

        textposition="middle center",

        textfont=dict(
            size=7,
            color=node_text_colors,
        ),

        name="Node",

        customdata=[
            i
            for i in V
        ],

        hovertemplate=(
            "Node %{customdata}"
            "<br>pos (%{y}, %{x})"
            "<extra></extra>"
        ),
    )
)


# --------------------------------------------------
# ROBOT COUNT BADGES
# --------------------------------------------------


if highlight_robots:

    for i in V:

        if robot_counts[i] > 0:

            fig.add_annotation(
                x=P[i][1] + 0.32,
                y=P[i][0] - 0.32,

                text=(
                    f"🤖×{robot_counts[i]}"
                ),

                showarrow=False,

                font=dict(
                    size=10,
                    color="black",
                ),

                bgcolor="white",

                bordercolor="black",

                borderwidth=1,

                borderpad=2,
            )


# --------------------------------------------------
# START REGION
# --------------------------------------------------


start_region_stored = st.session_state.get(
    "start_region",
    {(0, 0)},
)


start_xs = [
    P[i][1]
    for i in V
    if tuple(P[i]) in start_region_stored
]


start_ys = [
    P[i][0]
    for i in V
    if tuple(P[i]) in start_region_stored
]


if start_xs:

    fig.add_trace(
        go.Scatter(
            x=start_xs,
            y=start_ys,

            mode="markers",

            marker=dict(
                size=24,
                color=START_GREEN,
                symbol="star",
                line=dict(
                    color="black",
                    width=1.2,
                ),
            ),

            name="Start region",
        )
    )


# --------------------------------------------------
# FULL STRATEGY OVERVIEW
# --------------------------------------------------


if (
    visualization_phase >= 2
    and view_mode == "Full strategy overview"
):

    for m in trajectories:

        path = m.get("path")

        if not path or len(path) < 1:
            continue

        xs = [
            p[1]
            for p in path
        ]

        ys = [
            p[0]
            for p in path
        ]

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,

                mode="lines",

                line=dict(
                    color=MOTION_ORANGE,
                    width=1.6,
                ),

                opacity=0.4,

                showlegend=False,

                hoverinfo="skip",
            )
        )

        if len(xs) > 1:

            fig.add_annotation(
                x=xs[-1],
                y=ys[-1],

                ax=xs[-2],
                ay=ys[-2],

                xref="x",
                yref="y",
                axref="x",
                ayref="y",

                showarrow=True,

                arrowhead=2,

                arrowsize=1,

                arrowwidth=1.3,

                arrowcolor=MOTION_ORANGE,

                opacity=0.55,
            )

        mid = path[
            len(path) // 2
        ]

        fig.add_annotation(
            x=mid[1],
            y=mid[0],

            text=(
                f"{m['robots']}🤖 "
                f"t{m['t_departure']}–"
                f"{m['t_arrival']}"
            ),

            showarrow=False,

            font=dict(
                size=8,
                color=TEXT_DARK,
            ),

            bgcolor="white",

            opacity=0.9,

            borderpad=1,
        )


    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],

            mode="lines",

            line=dict(
                color=MOTION_ORANGE,
                width=2,
            ),

            name="Robot movement",
        )
    )


# --------------------------------------------------
# TIMESTEP MODE
#
# Only show movements that are active at the
# currently selected time.
# --------------------------------------------------


if (
    visualization_phase >= 2
    and view_mode == "Timestep"
):

    active = [
        m
        for m in trajectories
        if (
            m["t_departure"]
            <= current_time
            <= m["t_arrival"]
            and m.get("path") is not None
        )
    ]


    for idx, m in enumerate(active):

        path = m["path"]

        xs = [
            p[1]
            for p in path
        ]

        ys = [
            p[0]
            for p in path
        ]


        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,

                mode="lines",

                line=dict(
                    color=MOTION_ORANGE,
                    width=3,
                ),

                opacity=0.8,

                showlegend=(idx == 0),

                name=(
                    "Robot movement"
                    if idx == 0
                    else ""
                ),

                hovertemplate=(
                    f"{m['source']}→"
                    f"{m['target']}: "
                    f"{m['robots']} robots"
                    "<extra></extra>"
                ),
            )
        )


        pos = positionAtTime(
            path,
            m["t_departure"],
            m["t_arrival"],
            current_time,
        )


        if pos is not None:

            fig.add_trace(
                go.Scatter(
                    x=[pos[1]],
                    y=[pos[0]],

                    mode="markers",

                    marker=dict(
                        size=16,
                        color=MOTION_ORANGE,
                        symbol="diamond",
                        line=dict(
                            color="black",
                            width=1,
                        ),
                    ),

                    showlegend=False,

                    hovertemplate=(
                        f"{m['robots']} robots "
                        f"en route to node "
                        f"{m['target']}"
                        "<extra></extra>"
                    ),
                )
            )


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------


st.plotly_chart(
    fig,
    use_container_width=True,
)
