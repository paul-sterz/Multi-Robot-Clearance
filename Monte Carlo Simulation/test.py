import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


from approachTest import (
    buildEnvironment,
    detectionFnc,
    graphBuilder,
    approachTest,
    APPROACH_NAMES,
    START_REGION,
    GRAPH_PRIOR_L,
    GRAPH_PRIOR_SIGMA,
    GRAPH_ALPHA,
)


# ==================================================
# PAGE CONFIG
# ==================================================


st.set_page_config(
    page_title="Multi-Robot Clearance",
    layout="wide",
)

st.title("Multi-Robot Clearance — Monte Carlo Comparison")


# ==================================================
# ENVIRONMENT SIZE PRESETS
#
# Selecting a size only changes the detection radius
# used when building the navigation graph.
# ==================================================


ENVIRONMENT_SIZES = {
    "Small": 3,
    "Medium": 2,
    "Large": 1,
}


# ==================================================
# SESSION STATE
# ==================================================


for key, default in [

    ("environment_size", "Medium"),

    ("num_of_runs", 100),

    ("stopping_criterion", "Computation time"),

    ("computation_time", 3),

    ("max_trees", 100),

    ("available_robots", 6),

    ("simulation_results", None),

]:

    if key not in st.session_state:

        st.session_state[key] = default


detection_radius = ENVIRONMENT_SIZES[
    st.session_state.environment_size
]


# ==================================================
# LOAD ENVIRONMENT & COMPUTE PRIORS
#
# cellpriors is computed by graphBuilder() itself (PART 1 of
# GraphBuilder.py), exactly like in the approaches' test.py files - it is
# not re-derived here. This is only used for the preview plot before a
# simulation has been run; the plot below the "Simulate Strategys" button
# uses the graph actually built (and used for every run) inside
# approachTest() instead.
# ==================================================


obstacles, hotspots = buildEnvironment()

H, W = obstacles.shape


@st.cache_data(show_spinner="Computing priors...")
def computeCellPriors(detection_radius):

    _, _, _, _, cellpriors = graphBuilder(
        obstacles,
        hotspots,
        lambda p, obs: detectionFnc(p, obs, detection_radius),
        START_REGION,
        GRAPH_PRIOR_L,
        GRAPH_PRIOR_SIGMA,
        GRAPH_ALPHA,
    )

    return cellpriors


cellpriors = computeCellPriors(detection_radius)


# ==================================================
# PLOTLY FIGURE HELPER
#
# Builds the obstacles + prior-heatmap + start-region figure shared by the
# preview plot (no graph, shown immediately) and the post-simulation plot
# (with the navigation graph overlaid, shown once results exist).
# ==================================================


OBSTACLE_GRAY = "#4a4a4a"

START_GREEN = "#2a9d8f"

HEATMAP_RED_SOLID = "#d62828"

BASE_GRAY = "#9a9a9a"

NODE_GRAY = "#c9c9c9"


def buildEnvironmentFigure(obstacles, cellpriors, H, W, G=None, regularEdges=None):

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
    # PRIOR HEATMAP
    #
    # Shades every free cell red, with opacity scaling with its
    # (normalized) prior value. Higher prior -> redder.
    # --------------------------------------------------

    max_prior = float(np.max(cellpriors))

    if max_prior > 0:

        for x in range(H):

            for y in range(W):

                if obstacles[x, y] == 1:
                    continue

                value = cellpriors[x, y]

                if value <= 0:
                    continue

                norm = min(
                    1.0,
                    value / max_prior,
                )

                fig.add_shape(

                    type="rect",

                    x0=y - 0.5,
                    x1=y + 0.5,

                    y0=x - 0.5,
                    y1=x + 0.5,

                    fillcolor=(
                        f"rgba(214, 40, 40, "
                        f"{0.08 + 0.72 * norm:.3f})"
                    ),

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
                color=HEATMAP_RED_SOLID,
            ),

            name="Prior heatmap",
        )
    )

    # --------------------------------------------------
    # NAVIGATION GRAPH (only once a graph has actually been built)
    # --------------------------------------------------

    if G is not None and regularEdges is not None:

        for idx, (u, v) in enumerate(regularEdges):

            p1 = G.nodes[u].pos
            p2 = G.nodes[v].pos

            fig.add_trace(
                go.Scatter(

                    x=[p1[1], p2[1]],

                    y=[p1[0], p2[0]],

                    mode="lines",

                    line=dict(
                        color=BASE_GRAY,
                        width=1.5,
                    ),

                    opacity=0.55,

                    showlegend=(idx == 0),

                    name="Navigation graph" if idx == 0 else "",

                    hovertemplate=f"Edge {u} → {v}<extra></extra>",
                )
            )

        node_x = [node.pos[1] for node in G.nodes]
        node_y = [node.pos[0] for node in G.nodes]
        node_text = [str(node.idx) for node in G.nodes]

        fig.add_trace(
            go.Scatter(

                x=node_x,
                y=node_y,

                mode="markers+text",

                marker=dict(
                    size=16,
                    color=NODE_GRAY,
                    line=dict(
                        color="black",
                        width=1,
                    ),
                ),

                text=node_text,

                textposition="middle center",

                textfont=dict(
                    size=7,
                    color="#555555",
                ),

                name="Node",

                hovertemplate="Node %{text}<extra></extra>",
            )
        )

    # --------------------------------------------------
    # START REGION
    # --------------------------------------------------

    start_xs = [y for (x, y) in START_REGION]
    start_ys = [x for (x, y) in START_REGION]

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

    return fig


st.plotly_chart(
    buildEnvironmentFigure(obstacles, cellpriors, H, W),
    use_container_width=True,
)


# ==================================================
# ENVIRONMENT SIZE SELECTION
# ==================================================


st.markdown("### Environment Size")

size_cols = st.columns(3)

for col, size_name in zip(size_cols, ENVIRONMENT_SIZES):

    with col:

        is_selected = (
            st.session_state.environment_size == size_name
        )

        if st.button(
            size_name,
            key=f"size_{size_name}",
            type="primary" if is_selected else "secondary",
            use_container_width=True,
        ):

            st.session_state.environment_size = size_name

            st.rerun()

st.caption(
    f"Selected: **{st.session_state.environment_size}** "
    f"(detection radius = {detection_radius})"
)


# ==================================================
# RUN PARAMETERS
# ==================================================


st.markdown("### Run Parameters")

num_of_runs = st.slider(
    "Number of runs per graph",
    1,
    1000,
    st.session_state.num_of_runs,
)

st.session_state.num_of_runs = num_of_runs

stopping_criterion = st.radio(
    "Stop after",
    [
        "Computation time",
        "Number of spanning trees",
    ],
    horizontal=True,
    index=[
        "Computation time",
        "Number of spanning trees",
    ].index(st.session_state.stopping_criterion),
)

st.session_state.stopping_criterion = stopping_criterion

if stopping_criterion == "Computation time":

    computation_time = st.slider(
        "Computation Time (s)",
        1,
        60,
        st.session_state.computation_time,
    )

    st.session_state.computation_time = computation_time

    max_trees = None

else:

    max_trees = st.slider(
        "Number of spanning trees",
        100,
        10000,
        st.session_state.max_trees,
        step=100,
    )

    st.session_state.max_trees = max_trees

    computation_time = None

available_robots = st.slider(
    "Available robots",
    1,
    50,
    st.session_state.available_robots,
)

st.session_state.available_robots = available_robots


# ==================================================
# SIMULATE
# ==================================================


st.markdown("---")

if st.button(
    "Simulate Strategys",
    type="primary",
    use_container_width=True,
):

    with st.spinner(
        f"Running {num_of_runs} runs × {len(APPROACH_NAMES)} approaches..."
    ):

        (
            simG,
            simEdgesShady,
            simD,
            simStartNodes,
            simPriors,
            resultsCellPrior,
            resultsNodePrior,
            resultsTrees,
            resultsTime,
        ) = approachTest(
            detecRad=detection_radius,
            numOfRuns=num_of_runs,
            availableRobots=available_robots,
            availableTime=computation_time,
            maxTrees=max_trees,
        )

    st.session_state.simulation_results = {
        "G": simG,
        "regularEdges": list(simG.edges.keys()),
        "priors": simPriors,
        "resultsCellPrior": resultsCellPrior,
        "resultsNodePrior": resultsNodePrior,
        "resultsTrees": resultsTrees,
        "resultsTime": resultsTime,
        "stopping_criterion": stopping_criterion,
        "computation_time": computation_time,
        "max_trees": max_trees,
        "available_robots": available_robots,
    }


#------------------------------------------------------------------
# PLOTS OF THE RESULTS
#------------------------------------------------------------------


results = st.session_state.simulation_results

if results is not None:

    st.markdown("### Environment with Navigation Graph")

    st.plotly_chart(
        buildEnvironmentFigure(
            obstacles,
            results["priors"],
            H,
            W,
            G=results["G"],
            regularEdges=results["regularEdges"],
        ),
        use_container_width=True,
    )

    # --------------------------------------------------
    # STATS TABLES
    #
    # Runs without clearance (objective value == inf) are excluded from
    # min/max/mean/variance - they only increase the "No clearance
    # possible" counter.
    # --------------------------------------------------

    # Whichever of {computation time, spanning trees} was the stopping
    # criterion is fixed (prescribed) and shown as-is; the other one was
    # left free to vary run-by-run, so it is averaged over all runs
    # instead (per approach, since approaches don't check trees / use
    # time at the same rate).
    timeIsPrescribed = results["stopping_criterion"] == "Computation time"

    numGraphNodes = len(results["G"].nodes)
    numGraphEdges = len(results["G"].edges)

    def buildStatsTable(resultsArray):

        rows = []

        for approachIdx, approachName in enumerate(APPROACH_NAMES):

            column = resultsArray[:, approachIdx]

            finiteMask = np.isfinite(column)
            finiteValues = column[finiteMask]

            noClearanceCount = int(np.sum(~finiteMask))

            if timeIsPrescribed:
                usedTime = f"{results['computation_time']} s"
                spanningTrees = float(np.mean(results["resultsTrees"][:, approachIdx]))
            else:
                usedTime = float(np.mean(results["resultsTime"][:, approachIdx]))
                spanningTrees = f"{results['max_trees']} trees"

            rows.append({
                "Approach Name": approachName,
                "Nodes": numGraphNodes,
                "Edges": numGraphEdges,
                "Spanning Trees": spanningTrees,
                "Used Time": usedTime,
                "Available Robots": results["available_robots"],
                "Min": float(np.min(finiteValues)) if finiteValues.size > 0 else None,
                "Max": float(np.max(finiteValues)) if finiteValues.size > 0 else None,
                "Mean": float(np.mean(finiteValues)) if finiteValues.size > 0 else None,
                "Variance": float(np.var(finiteValues)) if finiteValues.size > 0 else None,
                "No Clearance Count": noClearanceCount,
            })

        return pd.DataFrame(rows)

    st.markdown("### Cell Prior Objective")

    st.dataframe(
        buildStatsTable(results["resultsCellPrior"]).round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Node Prior Objective")

    st.dataframe(
        buildStatsTable(results["resultsNodePrior"]).round(4),
        use_container_width=True,
        hide_index=True,
    )
