import time

import numpy as np
import streamlit as st
import plotly.graph_objects as go


from GraphBuilderV2 import graphBuilder
from Searcher import graphSearch
from TrajectoryPlanning import (
    computeTrajectory,
    computeObstacleDistance,
    positionAtTime,
    computeRobotCounts,
)
from Graph import Graph, Node, Edge


# ==================================================
# SIMPLE DETECTION FUNCTION
# ==================================================


def detectionFnc(p, obstacles, radius=3):

    H, W = obstacles.shape
    px, py = p

    # --------------------------------------------------
    # Simplified Bresenham line-of-sight test
    # --------------------------------------------------

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

            # If it is a diagonal/cross move,
            # also check the cells crossed on the way.
            if x != xold and y != yold:

                if abs(dy) == abs(dx):

                    if (
                        obstacles[x, yold] == 1
                        or obstacles[xold, y] == 1
                    ):
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

            if (
                np.sqrt(
                    (x - px) ** 2
                    + (y - py) ** 2
                )
                <= radius
            ):

                if has_clear_line_of_sight(
                    px,
                    py,
                    x,
                    y,
                ):

                    visible.add((x, y))

    return visible


# ==================================================
# PAGE CONFIG
# ==================================================


st.set_page_config(
    page_title="Multi-Robot Clearance — Baseline Modified",
    layout="wide",
)

st.title("Multi-Robot Clearance — Baseline Modified")


# ==================================================
# SIDEBAR
# ==================================================


with st.sidebar:

    st.markdown("## Parameters")

    H = st.slider(
        "Height",
        5,
        20,
        10,
    )

    W = st.slider(
        "Width",
        5,
        20,
        10,
    )

    detection_radius = st.slider(
        "Detection radius",
        1,
        8,
        3,
    )

    stopping_criterion = st.radio(
        "Stop after",
        [
            "Computation time",
            "Number of spanning trees",
        ],
        horizontal=True,
    )

    if stopping_criterion == "Computation time":

        computation_time = st.slider(
            "Computation Time (s)",
            1,
            60,
            3,
            step=1,
        )

        max_trees = None

    else:

        max_trees = st.slider(
            "Number of spanning trees",
            1,
            500,
            20,
        )

        computation_time = None

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

        st.session_state.obstacles = np.zeros(
            (H, W)
        )

        st.session_state.start_region = set()

        st.session_state.playing = False

        st.rerun()

    st.markdown("---")

    st.markdown("## Edit Mode")

    edit_mode = st.radio(
        "Click sets:",
        [
            "Obstacle",
            "Start Region",
        ],
        horizontal=False,
    )

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

    show_cleared = st.checkbox(
        "Cleared area",
        value=False,
    )

    view_mode = st.radio(
        "View",
        [
            "Timestep",
            "Full strategy overview",
        ],
    )


# ==================================================
# SESSION STATE
# ==================================================


def _init_grid(H, W):

    if (
        "obstacles" not in st.session_state
        or st.session_state.obstacles.shape
        != (H, W)
    ):

        st.session_state.obstacles = np.zeros(
            (H, W)
        )

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

    ("checked_tree_counter", 0),

    ("robots_needed", 0),

    # Detection sets per node (node_idx -> set of (row, col) cells)
    # and the time at which each node was first reached ("visited").
    ("detection_sets", None),

    ("node_visited_time", None),

    # Whether the timestep animation is currently auto-advancing.
    ("playing", False),

    # Visualization phases:
    #
    # 0 = nothing displayed yet
    # 1 = complete navigation graph
    # 2 = spanning tree + simulation

    ("visualization_phase", 0),
]:

    if key not in st.session_state:

        st.session_state[key] = default


# ==================================================
# INTERACTIVE GRID EDITOR
# ==================================================


st.markdown(
    "### Grid Editor — click a cell to toggle obstacle / set start region"
)


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


# ==================================================
# HANDLE GRID CLICK
# ==================================================


if clicked_r is not None:

    r, c = clicked_r, clicked_c

    if edit_mode == "Obstacle":

        if st.session_state.obstacles[r, c] == 1:

            st.session_state.obstacles[r, c] = 0

        else:

            st.session_state.obstacles[r, c] = 1

            st.session_state.start_region.discard(
                (r, c)
            )

    else:  # Start Region

        if st.session_state.obstacles[r, c] == 0:

            if (
                r,
                c,
            ) in st.session_state.start_region:

                st.session_state.start_region.discard(
                    (r, c)
                )

            else:

                st.session_state.start_region.add(
                    (r, c)
                )

    st.rerun()


# ==================================================
# COMPUTE
# ==================================================


if run:

    obstacles = (
        st.session_state.obstacles.copy()
    )

    root = (
        st.session_state.start_region
        if st.session_state.start_region
        else {(0, 0)}
    )

    with st.spinner("Building graph..."):

        G, shadyEdges, D, startNodes = (
            graphBuilder(
                obstacles,
                lambda p, obs: detectionFnc(
                    p,
                    obs,
                    detection_radius,
                ),
                root,
            )
        )

    with st.spinner(
        "Computing distance map..."
    ):

        distanceMap = (
            computeObstacleDistance(
                obstacles
            )
        )

    with st.spinner(
        "Computing strategy..."
    ):

        strategy, T, checkedTreeCounter, robotsNeeded = graphSearch(
            G,
            computation_time,
            startNodes,
            obstacles,
            distanceMap,
            alpha,
            maxTrees=max_trees,
        )

    if strategy is None:

        st.error(
            "Keine gültige Strategie gefunden "
            "(evtl. zu wenig Zeit/Bäume für diesen Graphen)."
        )

        st.stop()

    with st.spinner(
        "Planning trajectories..."
    ):

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

    V = list(
        range(len(G.nodes))
    )

    P = [
        G.nodes[i].pos
        for i in range(len(G.nodes))
    ]

    regularEdges = list(
        G.edges.keys()
    )

    treeEdges = list(
        T.edges.keys()
    )

    # --------------------------------------------------
    # Determine, for every node, the earliest time at
    # which it was reached ("visited"). Start-region
    # nodes count as visited from t = 0. Every other
    # node counts as visited as soon as a move targeting
    # it has arrived (t_arrival). Detection sets are not
    # disjoint, so several nodes can share cells - this
    # is only used to decide *when* a node's detection
    # set is added to the cleared area, not to dedupe
    # the cells themselves.
    # --------------------------------------------------

    # graphBuilder keeps adding start-region nodes until the
    # whole start region is *covered* - this is purely about
    # graph coverage and does NOT mean every one of these
    # nodes actually has robots on it at t = 0. Only the
    # node that is never the target of a move (i.e. the
    # true root of the spanning tree) starts out occupied.
    # Every other node - including start-region nodes that
    # merely help cover the area - only becomes "visited"
    # once a trajectory actually arrives there.

    targeted_nodes = {
        m["target"] for m in trajectories
    }

    node_visited_time = {}

    # startNodes is the *count* of start-region nodes
    # (they are created first, so they occupy indices
    # 0 .. startNodes - 1).
    for i in range(startNodes):

        if i not in targeted_nodes:

            node_visited_time[i] = 0

    for m in trajectories:

        tgt = m["target"]
        t_arr = m["t_arrival"]

        if (
            tgt not in node_visited_time
            or t_arr < node_visited_time[tgt]
        ):

            node_visited_time[tgt] = t_arr

    # Save everything
    st.session_state.trajectories = (
        trajectories
    )

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

    st.session_state.checked_tree_counter = (
        checkedTreeCounter
    )

    st.session_state.robots_needed = (
        robotsNeeded
    )

    st.session_state.detection_sets = D

    st.session_state.node_visited_time = (
        node_visited_time
    )

    st.session_state.current_time = 0

    st.session_state.total_time = (
        total_time
    )

    st.session_state.visualization_phase = 0

    st.session_state.playing = False

    st.rerun()


# ==================================================
# STOP IF NOTHING COMPUTED
# ==================================================


if st.session_state.graph_data is None:

    st.info(
        "Parameter wählen, Hindernisse/Startregion setzen "
        "und **Run** drücken."
    )

    st.stop()


# ==================================================
# LOAD DATA
# ==================================================


(
    obstacles,
    V,
    P,
    shadyEdges,
    regularEdges,
    H,
    W,
) = st.session_state.graph_data


trajectories = (
    st.session_state.trajectories
)

current_time = (
    st.session_state.current_time
)

total_time = (
    st.session_state.total_time
)

strategy, treeEdges = (
    st.session_state.strategy
)

checkedTreeCounter = (
    st.session_state.checked_tree_counter
)

robotsNeeded = (
    st.session_state.robots_needed
)

detection_sets = (
    st.session_state.detection_sets
)

node_visited_time = (
    st.session_state.node_visited_time
)

visualization_phase = (
    st.session_state.visualization_phase
)


# ==================================================
# ROBOT COUNTS
#
# These counts describe robots that are currently
# associated with graph nodes.
#
# Moving robots are handled separately below using
# positionAtTime().
# ==================================================


robot_counts = computeRobotCounts(
    strategy,
    len(V),
    current_time,
)


# ==================================================
# METRICS
# ==================================================


m1, m2, m3, m4, m5, m6 = st.columns(6)


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
        "Robots needed",
        robotsNeeded,
    )

with m6:

    st.metric(
        "Trees Searched",
        checkedTreeCounter
    )


# ==================================================
# VISUALIZATION CONTROLS
# ==================================================


col1, col2, col3, col4, col5 = st.columns(
    [1, 1, 1, 2, 3]
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
        disabled=at_end or st.session_state.playing,
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

    # Play / Pause is only meaningful once the spanning
    # tree / simulation is being shown in Timestep view.
    play_disabled = not (
        visualization_phase >= 2
        and view_mode == "Timestep"
    )

    play_label = (
        "⏸ Pause"
        if st.session_state.playing
        else "▶ Play"
    )

    if st.button(
        play_label,
        use_container_width=True,
        disabled=play_disabled,
    ):

        st.session_state.playing = (
            not st.session_state.playing
        )

        # If we start playing right at the end, jump back
        # to the start so the animation is visible again.
        if (
            st.session_state.playing
            and current_time >= total_time
        ):

            st.session_state.current_time = 0

        st.rerun()


with col3:

    if st.button(
        "Reset",
        use_container_width=True,
    ):

        st.session_state.current_time = 0

        st.session_state.visualization_phase = 0

        st.session_state.playing = False

        st.rerun()


with col4:

    if visualization_phase == 0:

        phase = "Ready"

    elif visualization_phase == 1:

        phase = "Full graph"

    else:

        phase = (
            f"Spanning tree → "
            f"t = {current_time} / {total_time}"
        )

    if st.session_state.playing:

        phase += "  ▶ playing..."

    st.caption(
        f"Phase: **{phase}**"
    )


# ==================================================
# TIME SLIDER
# ==================================================


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
        disabled=st.session_state.playing,
    )

    if (
        not st.session_state.playing
        and new_time != current_time
    ):

        st.session_state.current_time = (
            new_time
        )

        st.rerun()


# ==================================================
# PLOTLY FIGURE
# ==================================================


BASE_GRAY = "#9a9a9a"
OBSTACLE_GRAY = "#4a4a4a"

TREE_TEAL = "#2a9d8f"
ROBOT_GOLD = "#e9a72b"
MOTION_ORANGE = "#e76f51"

START_GREEN = "#2a9d8f"
TEXT_DARK = "#264653"

CLEARED_GREEN_FILL = "rgba(56, 176, 0, 0.28)"
CLEARED_GREEN_SOLID = "#38b000"


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
        range=[
            -0.5,
            W - 0.5,
        ],
        tickmode="array",
        tickvals=list(
            range(W)
        ),
        showgrid=True,
        gridcolor="#eeeeee",
        zeroline=False,
        fixedrange=True,
        constrain="domain",
    ),

    yaxis=dict(
        title="Row",
        range=[
            -0.5,
            H - 0.5,
        ],
        tickmode="array",
        tickvals=list(
            range(H)
        ),
        showgrid=True,
        gridcolor="#eeeeee",
        zeroline=False,
        fixedrange=True,
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
    ),
)


# ==================================================
# OBSTACLES
# ==================================================


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


# ==================================================
# CLEARED AREA
#
# For every node that has already been visited (i.e.
# node_visited_time[node] <= t_ref), all cells in that
# node's detection set are shaded green. Detection sets
# are not disjoint - multiple nodes can contribute the
# same cell - so we simply union them into a set.
# ==================================================


if (
    show_cleared
    and visualization_phase >= 2
    and detection_sets is not None
    and node_visited_time is not None
):

    if view_mode == "Full strategy overview":

        t_ref = total_time

    else:

        t_ref = current_time

    cleared_cells = set()

    # detection_sets (D) is a list indexed like G.nodes:
    # detection_sets[node_idx] -> set of (row, col) cells.
    for node_idx, t_visit in node_visited_time.items():

        if (
            t_visit <= t_ref
            and node_idx < len(detection_sets)
        ):

            cleared_cells |= detection_sets[node_idx]

    for (x, y) in cleared_cells:

        if obstacles[x, y] == 1:
            continue

        fig.add_shape(

            type="rect",

            x0=y - 0.5,
            x1=y + 0.5,

            y0=x - 0.5,
            y1=x + 0.5,

            fillcolor=CLEARED_GREEN_FILL,

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
                color=CLEARED_GREEN_SOLID,
            ),

            name="Cleared area",
        )
    )


# ==================================================
# COMPLETE NAVIGATION GRAPH
# ==================================================


if visualization_phase >= 1:

    for idx, (u, v) in enumerate(
        regularEdges
    ):

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


# ==================================================
# SHADY EDGES
# ==================================================


if (
    visualization_phase >= 1
    and show_shady
):

    for idx, (u, v) in enumerate(
        shadyEdges
    ):

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


# ==================================================
# OPTIONAL REGULAR EDGE HIGHLIGHT
# ==================================================


if (
    visualization_phase >= 1
    and show_regular
):

    for idx, (u, v) in enumerate(
        regularEdges
    ):

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


# ==================================================
# SPANNING TREE
# ==================================================


if (
    visualization_phase >= 2
    and show_tree
):

    for idx, (u, v) in enumerate(
        treeEdges
    ):

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


# ==================================================
# NODES
# ==================================================


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


# ==================================================
# NODE ROBOT COUNT BADGES
#
# These labels are only used for robots that are
# currently considered stationary by
# computeRobotCounts().
#
# Moving robots receive their own label below.
# ==================================================


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


# ==================================================
# START REGION
# ==================================================


start_region_stored = (
    st.session_state.get(
        "start_region",
        {(0, 0)},
    )
)


start_xs = [
    P[i][1]
    for i in V
    if tuple(P[i])
    in start_region_stored
]

start_ys = [
    P[i][0]
    for i in V
    if tuple(P[i])
    in start_region_stored
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


# ==================================================
# FULL STRATEGY OVERVIEW
# ==================================================


if (
    visualization_phase >= 2
    and view_mode
    == "Full strategy overview"
):

    for m in trajectories:

        path = m.get("path")

        if (
            not path
            or len(path) < 1
        ):
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


# ==================================================
# TIMESTEP MODE
#
# Time convention:
#
#   t < departure
#       robot is at source
#
#   departure <= t < arrival
#       robot is moving
#
#   t >= arrival
#       robot is at target
#
# IMPORTANT:
# We deliberately use
#
#   t_departure <= current_time < t_arrival
#
# instead of
#
#   t_departure <= current_time <= t_arrival
#
# This prevents a robot from being displayed both
# as moving and as already arrived at the target.
# ==================================================


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
            < m["t_arrival"]
            and m.get("path") is not None
        )
    ]


    # ==================================================
    # MOVING ROBOTS
    # ==================================================

    active_positions = []


    for idx, m in enumerate(active):

        path = m["path"]


        # --------------------------------------------------
        # Current robot position
        # --------------------------------------------------

        pos = positionAtTime(

            path,

            m["t_departure"],

            m["t_arrival"],

            current_time,
        )


        if pos is None:
            continue


        # --------------------------------------------------
        # Draw only the already travelled part of the path.
        #
        # We use the current position as the last point.
        # This prevents the visualization from showing the
        # future part of the trajectory as if it had already
        # been travelled.
        # --------------------------------------------------

        progress = (
            current_time
            - m["t_departure"]
        ) / (
            m["t_arrival"]
            - m["t_departure"]
        )


        progress = max(
            0.0,
            min(1.0, progress),
        )


        number_of_path_points = len(
            path
        )


        if number_of_path_points <= 1:

            travelled_path = [
                pos
            ]

        else:

            # Determine approximately how far along
            # the discrete path we currently are.
            current_index = int(
                progress
                * (
                    number_of_path_points
                    - 1
                )
            )

            current_index = max(
                0,
                min(
                    current_index,
                    number_of_path_points - 1,
                ),
            )

            travelled_path = (
                path[
                    : current_index + 1
                ]
            )

            # Replace the last path point by the
            # actual interpolated position.
            if travelled_path:

                travelled_path = (
                    travelled_path[:-1]
                    + [pos]
                )


        xs = [
            p[1]
            for p in travelled_path
        ]

        ys = [
            p[0]
            for p in travelled_path
        ]


        # --------------------------------------------------
        # Draw travelled path
        # --------------------------------------------------

        if len(xs) >= 1:

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

                    showlegend=(
                        idx == 0
                    ),

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


        # --------------------------------------------------
        # Draw current robot position
        # --------------------------------------------------

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


        # Save the position for grouping labels.
        active_positions.append(
            (
                pos,
                m["robots"],
            )
        )


    # ==================================================
    # GROUP MOVING ROBOTS
    #
    # Multiple movement groups can temporarily occupy
    # the same position. We merge those labels.
    # ==================================================

    position_groups = {}


    for pos, robots in active_positions:

        key = (
            round(pos[0], 3),
            round(pos[1], 3),
        )


        if key not in position_groups:

            position_groups[key] = {
                "pos": pos,
                "robots": 0,
            }


        position_groups[key][
            "robots"
        ] += robots


    # ==================================================
    # DRAW MOVING ROBOT LABELS
    # ==================================================

    for group in position_groups.values():

        pos = group["pos"]

        fig.add_annotation(

            x=pos[1] + 0.32,

            y=pos[0] - 0.32,

            text=(
                f"🤖×{group['robots']}"
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


# ==================================================
# DISPLAY
# ==================================================


st.plotly_chart(
    fig,
    use_container_width=True,
)


# ==================================================
# AUTOPLAY LOOP
#
# When "playing" is active, we render the current
# frame (above), wait one second, advance the time
# by one step and trigger a rerun. This repeats until
# total_time is reached or the user hits Pause.
# ==================================================


if st.session_state.playing:

    if (
        visualization_phase >= 2
        and current_time < total_time
    ):

        time.sleep(0.5)

        st.session_state.current_time = min(
            current_time + 1,
            total_time,
        )

        st.rerun()

    else:

        st.session_state.playing = False

        st.rerun()
