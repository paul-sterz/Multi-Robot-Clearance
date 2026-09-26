import importlib.util
import os

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

from baselineTest import baselineTest, BASELINE_METHOD_NAMES, COMPARED_APPROACH_NAMES

from treeTest import runSpanningTreeEvolution, runSpanningTreeEvolutionRepeated

from closingExitsTest import runClosingExitsComparison

# 3DTest.py can't be `import`ed by that name (a module name can't start with
# a digit) - loaded the same way approachTest.py loads each approach's
# Searcher.py.
_spec = importlib.util.spec_from_file_location(
    "threeDTest", os.path.join(os.path.dirname(os.path.abspath(__file__)), "3DTest.py")
)
threeDTest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(threeDTest)


# ==================================================
# 3D SCENE SCREENSHOTS
#
# There is no in-app 3D preview (see the "no preview plot" caption below) -
# drop a manually-taken screenshot in scene_screenshots/<scene name>.png
# (e.g. scene_screenshots/christ-church.png) and it is picked up here
# automatically. Take one from the standalone 3D viewer
# (python -m strategy_service.server, http://localhost:8000): pick the
# scene, orbit/zoom to a good angle, then a normal OS screenshot
# (Cmd+Shift+4 on macOS) of the view - no in-app export needed.
# ==================================================

SCENE_SCREENSHOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_screenshots")


def renderSceneScreenshot(sceneName: str):

    for ext in (".png", ".jpg", ".jpeg"):
        path = os.path.join(SCENE_SCREENSHOTS_DIR, sceneName + ext)
        if os.path.isfile(path):
            st.image(path, caption=sceneName, use_container_width=True)
            return

    st.caption(
        f"No screenshot yet for **{sceneName}** - drop one at "
        f"`scene_screenshots/{sceneName}.png` to show it here."
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
# MODE SELECTION
# ==================================================


app_mode = st.radio(
    "Mode",
    [
        "Approach Test",
        "Baseline Test",
        "Spanning Tree Evolution",
        "Closing Exits Test",
    ],
    horizontal=True,
    key="app_mode",
)

# ==================================================
# ENVIRONMENT SOURCE: 2D synthetic grid (as before) or one of the six real
# clearing-scenes 3D scenes (christ-church, keble-college, ...), loaded via
# 3DTest.py. Every slider below this point is unchanged either way - only
# the graph/priors these three modes run their comparisons on differs.
# ==================================================

env_source = st.radio(
    "Environment",
    ["2D (synthetic grid)", "3D (real scene)"],
    horizontal=True,
    key="env_source",
)
is3D = env_source.startswith("3D")

if is3D:
    scene_name = st.selectbox(
        "Scene",
        threeDTest.SCENE_NAMES,
        key="scene_name",
    )
    st.caption(
        "The 4 approaches / 4 baseline methods run on this scene's real "
        "graph and detection sets, exactly as they would on the 2D grid - "
        "hotspots are a small fixed layout (see 3DTest.py's "
        "default3DHotspots()), not interactively placed, so runs stay "
        "comparable across the many repeats these tests do."
    )

# GraphBuilder's uncertainty floor and FHPE_SA's probabilistic search budget -
# shared across all three modes and both environment sources below, exactly
# like env_source itself.
epsilon = st.slider(
    "Epsilon (GraphBuilder uncertainty floor)",
    0.0,
    0.5,
    0.05,
    step=0.01,
    key="epsilon",
)

prob_budget = st.slider(
    "FHPE+SA probabilistic search budget",
    10,
    1000,
    200,
    step=10,
    key="prob_budget",
)

st.markdown("---")


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

    ("baseline_num_of_runs", 100),

    ("baseline_stopping_criterion", "Computation time"),

    ("baseline_computation_time", 3),

    ("baseline_max_trees", 100),

    ("baseline_available_robots", 6),

    ("baseline_simulation_results", None),

    ("evo_approach", APPROACH_NAMES[0]),

    ("evo_stopping_criterion", "Computation time"),

    ("evo_computation_time", 3),

    ("evo_max_trees", 100),

    ("evo_available_robots", 6),

    ("evo_population_size", 10),

    ("evolution_results", None),

    ("evo_repeated_num_of_runs", 100),

    ("evo_repeated_results", None),

    ("ce_num_of_runs", 100),

    ("ce_stopping_criterion", "Computation time"),

    ("ce_computation_time", 3),

    ("ce_max_trees", 100),

    ("ce_available_robots", 6),

    ("ce_horizon", 5),

    ("ce_hurt_probability", 0.02),

    ("ce_simulation_results", None),

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
def computeCellPriors(detection_radius, epsilon):

    _, _, _, _, cellpriors = graphBuilder(
        obstacles,
        hotspots,
        lambda p, obs: detectionFnc(p, obs, detection_radius),
        START_REGION,
        GRAPH_PRIOR_L,
        GRAPH_PRIOR_SIGMA,
        GRAPH_ALPHA,
        epsilon,
    )

    return cellpriors


cellpriors = computeCellPriors(detection_radius, epsilon)


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


if not is3D:

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

else:

    _sceneStats = threeDTest.loadScene3D(scene_name)
    st.caption(
        f"**{scene_name}** — {_sceneStats.n_vertices} vertices, "
        f"{_sceneStats.n_cells:,} cells."
    )
    renderSceneScreenshot(scene_name)


if app_mode == "Approach Test":

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

            if is3D:
                simResult = threeDTest.approachTest3D(
                    sceneName=scene_name,
                    numOfRuns=num_of_runs,
                    availableRobots=available_robots,
                    availableTime=computation_time,
                    maxTrees=max_trees,
                    epsilon=epsilon,
                    probBudget=prob_budget,
                )
                simG = simResult["G"]
                simPriors = simResult["priors"]
                resultsCellPrior = simResult["resultsCellPrior"]
                resultsNodePrior = simResult["resultsNodePrior"]
                resultsTrees = simResult["resultsTrees"]
                resultsTime = simResult["resultsTime"]
            else:
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
                    epsilon=epsilon,
                    probBudget=prob_budget,
                )

        st.session_state.simulation_results = {
            "is3D": is3D,
            "sceneName": scene_name if is3D else None,
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

        st.markdown("### Environment")

        if results.get("is3D"):
            st.caption(f"Scene: **{results['sceneName']}**")
            renderSceneScreenshot(results["sceneName"])
        else:
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


elif app_mode == "Baseline Test":

    # ==================================================
    # RUN PARAMETERS
    # ==================================================


    st.markdown("### Run Parameters")

    baseline_num_of_runs = st.slider(
        "Number of runs per graph",
        1,
        1000,
        st.session_state.baseline_num_of_runs,
        key="baseline_num_of_runs_slider",
    )

    st.session_state.baseline_num_of_runs = baseline_num_of_runs

    baseline_stopping_criterion = st.radio(
        "Stop after",
        [
            "Computation time",
            "Number of spanning trees",
        ],
        horizontal=True,
        index=[
            "Computation time",
            "Number of spanning trees",
        ].index(st.session_state.baseline_stopping_criterion),
        key="baseline_stopping_radio",
    )

    st.session_state.baseline_stopping_criterion = baseline_stopping_criterion

    if baseline_stopping_criterion == "Computation time":

        baseline_computation_time = st.slider(
            "Computation Time (s)",
            1,
            60,
            st.session_state.baseline_computation_time,
            key="baseline_computation_time_slider",
        )

        st.session_state.baseline_computation_time = baseline_computation_time

        baseline_max_trees = None

    else:

        baseline_max_trees = st.slider(
            "Number of spanning trees",
            100,
            10000,
            st.session_state.baseline_max_trees,
            step=100,
            key="baseline_max_trees_slider",
        )

        st.session_state.baseline_max_trees = baseline_max_trees

        baseline_computation_time = None

    baselineMethodsLabel = ", ".join(BASELINE_METHOD_NAMES)
    comparedApproachesLabel = ", ".join(COMPARED_APPROACH_NAMES)

    baseline_available_robots = st.slider(
        "Available robots",
        1,
        50,
        st.session_state.baseline_available_robots,
        key="baseline_available_robots_slider",
    )

    st.session_state.baseline_available_robots = baseline_available_robots

    st.caption(
        f"{baselineMethodsLabel}, {comparedApproachesLabel} all run with "
        f"the exact same fixed robot budget ({baseline_available_robots}). "
        f"{baselineMethodsLabel} treat it as a hard cap - a run finds no "
        "clearance if no random spanning tree fits within it."
    )


    # ==================================================
    # SIMULATE
    # ==================================================


    st.markdown("---")

    if st.button(
        "Simulate Strategys",
        type="primary",
        use_container_width=True,
        key="baseline_simulate_button",
    ):

        with st.spinner(
            f"Running {baseline_num_of_runs} runs × "
            f"({baselineMethodsLabel}, {comparedApproachesLabel})..."
        ):

            if is3D:
                baselineResults = threeDTest.baselineTest3D(
                    sceneName=scene_name,
                    numOfRuns=baseline_num_of_runs,
                    availableRobots=baseline_available_robots,
                    availableTime=baseline_computation_time,
                    maxTrees=baseline_max_trees,
                    epsilon=epsilon,
                    probBudget=prob_budget,
                )
            else:
                baselineResults = baselineTest(
                    detecRad=detection_radius,
                    numOfRuns=baseline_num_of_runs,
                    availableRobots=baseline_available_robots,
                    availableTime=baseline_computation_time,
                    maxTrees=baseline_max_trees,
                    epsilon=epsilon,
                    probBudget=prob_budget,
                )

        st.session_state.baseline_simulation_results = {
            "is3D": is3D,
            "sceneName": scene_name if is3D else None,
            "G": baselineResults["G"],
            "regularEdges": list(baselineResults["G"].edges.keys()),
            "priors": baselineResults["priors"],
            "methodNames": baselineResults["methodNames"],
            "numGraphNodes": baselineResults["numGraphNodes"],
            "numGraphEdges": baselineResults["numGraphEdges"],
            "resultsCellPrior": baselineResults["resultsCellPrior"],
            "resultsRobots": baselineResults["resultsRobots"],
            "resultsTrees": baselineResults["resultsTrees"],
            "resultsTime": baselineResults["resultsTime"],
            "stopping_criterion": baseline_stopping_criterion,
            "computation_time": baseline_computation_time,
            "max_trees": baseline_max_trees,
        }


    #------------------------------------------------------------------
    # PLOTS OF THE RESULTS
    #------------------------------------------------------------------


    baselineResultsState = st.session_state.baseline_simulation_results

    if baselineResultsState is not None:

        st.markdown("### Environment")

        st.caption(
            f"{baselineMethodsLabel}, {comparedApproachesLabel} all run on "
            "this exact same navigation graph, built once and reused for "
            "every run."
            + (f" Scene: **{baselineResultsState['sceneName']}**." if baselineResultsState.get("is3D") else "")
        )

        if baselineResultsState.get("is3D"):
            renderSceneScreenshot(baselineResultsState["sceneName"])
        else:
            st.plotly_chart(
                buildEnvironmentFigure(
                    obstacles,
                    baselineResultsState["priors"],
                    H,
                    W,
                    G=baselineResultsState["G"],
                    regularEdges=baselineResultsState["regularEdges"],
                ),
                use_container_width=True,
            )

        # --------------------------------------------------
        # STATS TABLE
        #
        # Runs without clearance (objective value == inf) are excluded from
        # min/max/mean/variance - they only increase the "No clearance
        # possible" counter. All 4 methods share the same graph AND the same
        # fixed Available Robots budget, like in Approach Test.
        # --------------------------------------------------

        baselineTimeIsPrescribed = (
            baselineResultsState["stopping_criterion"] == "Computation time"
        )

        baselineNumGraphNodes = baselineResultsState["numGraphNodes"]
        baselineNumGraphEdges = baselineResultsState["numGraphEdges"]

        def buildBaselineStatsTable(resultsArray):

            rows = []

            for methodIdx, methodName in enumerate(baselineResultsState["methodNames"]):

                column = resultsArray[:, methodIdx]

                finiteMask = np.isfinite(column)
                finiteValues = column[finiteMask]

                noClearanceCount = int(np.sum(~finiteMask))

                if baselineTimeIsPrescribed:
                    usedTime = f"{baselineResultsState['computation_time']} s"
                    spanningTrees = float(
                        np.mean(baselineResultsState["resultsTrees"][:, methodIdx])
                    )
                else:
                    usedTime = float(
                        np.mean(baselineResultsState["resultsTime"][:, methodIdx])
                    )
                    spanningTrees = f"{baselineResultsState['max_trees']} trees"

                rows.append({
                    "Approach Name": methodName,
                    "Nodes": baselineNumGraphNodes,
                    "Edges": baselineNumGraphEdges,
                    "Spanning Trees": spanningTrees,
                    "Used Time": usedTime,
                    "Available Robots": float(
                        np.mean(baselineResultsState["resultsRobots"][:, methodIdx])
                    ),
                    "Min": float(np.min(finiteValues)) if finiteValues.size > 0 else None,
                    "Max": float(np.max(finiteValues)) if finiteValues.size > 0 else None,
                    "Mean": float(np.mean(finiteValues)) if finiteValues.size > 0 else None,
                    "Variance": float(np.var(finiteValues)) if finiteValues.size > 0 else None,
                    "No Clearance Count": noClearanceCount,
                })

            return pd.DataFrame(rows)

        st.markdown("### Cell Prior Objective")

        st.dataframe(
            buildBaselineStatsTable(baselineResultsState["resultsCellPrior"]).round(4),
            use_container_width=True,
            hide_index=True,
        )


elif app_mode == "Spanning Tree Evolution":

    # ==================================================
    # SPANNING TREE EVOLUTION
    #
    # Runs a single approach once (no averaging over many runs) and tracks,
    # tree by tree, the objective value of the tree just checked and the
    # best objective value found so far - so the search's convergence can
    # be plotted directly.
    # ==================================================


    st.markdown("### Run Parameters")

    evo_approach = st.selectbox(
        "Approach",
        APPROACH_NAMES,
        index=APPROACH_NAMES.index(st.session_state.evo_approach),
    )

    st.session_state.evo_approach = evo_approach

    evo_stopping_criterion = st.radio(
        "Stop after",
        [
            "Computation time",
            "Number of spanning trees",
        ],
        horizontal=True,
        index=[
            "Computation time",
            "Number of spanning trees",
        ].index(st.session_state.evo_stopping_criterion),
        key="evo_stopping_radio",
    )

    st.session_state.evo_stopping_criterion = evo_stopping_criterion

    if evo_stopping_criterion == "Computation time":

        evo_computation_time = st.slider(
            "Computation Time (s)",
            1,
            60,
            st.session_state.evo_computation_time,
            key="evo_computation_time_slider",
        )

        st.session_state.evo_computation_time = evo_computation_time

        evo_max_trees = None

    else:

        evo_max_trees = st.slider(
            "Number of spanning trees",
            100,
            10000,
            st.session_state.evo_max_trees,
            step=100,
            key="evo_max_trees_slider",
        )

        st.session_state.evo_max_trees = evo_max_trees

        evo_computation_time = None

    evo_available_robots = st.slider(
        "Available robots",
        1,
        50,
        st.session_state.evo_available_robots,
        key="evo_available_robots_slider",
    )

    st.session_state.evo_available_robots = evo_available_robots

    evo_population_size = st.slider(
        "Initial generation size",
        2,
        50,
        st.session_state.evo_population_size,
        key="evo_population_size_slider",
    )

    st.session_state.evo_population_size = evo_population_size


    # ==================================================
    # RUN EVOLUTION
    # ==================================================


    st.markdown("---")

    if st.button(
        "Run Spanning Tree Evolution",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            f"Running {evo_approach} "
            f"(evolutionary search + random spanning tree baseline)..."
        ):

            if is3D:
                evolution = threeDTest.runSpanningTreeEvolution3D(
                    sceneName=scene_name,
                    approachName=evo_approach,
                    availableRobots=evo_available_robots,
                    availableTime=evo_computation_time,
                    maxTrees=evo_max_trees,
                    populationSize=evo_population_size,
                    epsilon=epsilon,
                    probBudget=prob_budget,
                )
            else:
                evolution = runSpanningTreeEvolution(
                    approachName=evo_approach,
                    detecRad=detection_radius,
                    availableRobots=evo_available_robots,
                    availableTime=evo_computation_time,
                    maxTrees=evo_max_trees,
                    populationSize=evo_population_size,
                    epsilon=epsilon,
                    probBudget=prob_budget,
                )

        st.session_state.evolution_results = {
            "is3D": is3D,
            "sceneName": scene_name if is3D else None,
            "approach": evo_approach,
            "G": evolution["G"],
            "regularEdges": list(evolution["G"].edges.keys()),
            "priors": evolution["priors"],
            "evolutionary": evolution["evolutionary"],
            "random": evolution["random"],
        }


    #------------------------------------------------------------------
    # PLOTS OF THE RESULTS
    #------------------------------------------------------------------


    evoResults = st.session_state.evolution_results

    if evoResults is not None:

        st.markdown("### Environment")

        if evoResults.get("is3D"):
            st.caption(f"Scene: **{evoResults['sceneName']}**")
            renderSceneScreenshot(evoResults["sceneName"])
        else:
            st.plotly_chart(
                buildEnvironmentFigure(
                    obstacles,
                    evoResults["priors"],
                    H,
                    W,
                    G=evoResults["G"],
                    regularEdges=evoResults["regularEdges"],
                ),
                use_container_width=True,
            )

        def renderRunCaption(runResult, methodLabel):

            if np.isfinite(runResult["bestFitness"]):
                st.caption(
                    f"**{evoResults['approach']} — {methodLabel}** — "
                    f"{runResult['checkedTrees']} spanning trees checked, "
                    f"best objective found: {runResult['bestFitness']:.4f}"
                )
            else:
                st.caption(
                    f"**{evoResults['approach']} — {methodLabel}** — "
                    f"{runResult['checkedTrees']} spanning trees checked, "
                    f"no clearance found"
                )

        def renderHistoryPlots(history, methodLabel, colorBest, colorCurrent):

            treeIndices = [entry[0] for entry in history]
            currentFitness = [entry[1] for entry in history]
            bestFitness = [entry[2] for entry in history]

            # inf (no clearance found by that tree) can't be plotted - leave
            # a gap in the line instead.
            currentFitnessPlot = [v if np.isfinite(v) else None for v in currentFitness]
            bestFitnessPlot = [v if np.isfinite(v) else None for v in bestFitness]

            st.markdown(f"##### {methodLabel} — Best Objective Value Found So Far")

            bestFig = go.Figure()

            bestFig.add_trace(
                go.Scatter(
                    x=treeIndices,
                    y=bestFitnessPlot,
                    mode="lines",
                    line=dict(color=colorBest, width=2),
                    name="Best objective so far",
                )
            )

            bestFig.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor="white",
                xaxis=dict(
                    title=dict(text="Number of Spanning Trees", font=dict(size=20)),
                    tickfont=dict(size=14),
                    gridcolor="#eeeeee",
                ),
                yaxis=dict(
                    title=dict(text="Best Objective Value", font=dict(size=20)),
                    tickfont=dict(size=14),
                    gridcolor="#eeeeee",
                ),
            )

            st.plotly_chart(bestFig, use_container_width=True)

            st.markdown(f"##### {methodLabel} — Objective Value per Checked Spanning Tree")

            currentFig = go.Figure()

            currentFig.add_trace(
                go.Scatter(
                    x=treeIndices,
                    y=currentFitnessPlot,
                    mode="markers",
                    marker=dict(color=colorCurrent, size=5),
                    name="Objective value of checked tree",
                )
            )

            currentFig.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor="white",
                xaxis=dict(
                    title=dict(text="Number of Spanning Trees", font=dict(size=20)),
                    tickfont=dict(size=14),
                    gridcolor="#eeeeee",
                ),
                yaxis=dict(
                    title=dict(text="Objective Value", font=dict(size=20)),
                    tickfont=dict(size=14),
                    gridcolor="#eeeeee",
                ),
            )

            st.plotly_chart(currentFig, use_container_width=True)

        st.markdown("### Evolutionary Search")

        renderRunCaption(evoResults["evolutionary"], "Evolutionary Search")

        renderHistoryPlots(
            evoResults["evolutionary"]["history"],
            "Evolutionary Search",
            colorBest="#2a9d8f",
            colorCurrent="#d62828",
        )

        st.markdown("### Random Spanning Tree Generation")

        renderRunCaption(evoResults["random"], "Random Spanning Tree Generation")

        renderHistoryPlots(
            evoResults["random"]["history"],
            "Random Spanning Tree Generation",
            colorBest="#264653",
            colorCurrent="#e76f51",
        )

        # --------------------------------------------------
        # COMPARISON TABLE: EVOLUTIONARY SEARCH vs. RANDOM SPANNING TREES
        #
        # Best / mean / variance over all checked trees' objective values
        # (trees without clearance, i.e. objective value == inf, are
        # excluded - same convention as the Approach Test stats tables).
        # --------------------------------------------------

        st.markdown("### Evolutionary Search vs. Random Spanning Tree Generation")

        def buildEvolutionComparisonRow(runResult, methodLabel):

            fitnessValues = np.array(
                [entry[1] for entry in runResult["history"]],
                dtype=float,
            )

            finiteMask = np.isfinite(fitnessValues)
            finiteValues = fitnessValues[finiteMask]

            return {
                "Method": methodLabel,
                "Spanning Trees Checked": runResult["checkedTrees"],
                "Best": float(np.min(finiteValues)) if finiteValues.size > 0 else None,
                "Mean": float(np.mean(finiteValues)) if finiteValues.size > 0 else None,
                "Variance": float(np.var(finiteValues)) if finiteValues.size > 0 else None,
                "No Clearance Count": int(np.sum(~finiteMask)),
            }

        comparisonTable = pd.DataFrame([
            buildEvolutionComparisonRow(evoResults["evolutionary"], "Evolutionary Search"),
            buildEvolutionComparisonRow(evoResults["random"], "Random Spanning Tree Generation"),
        ])

        st.dataframe(
            comparisonTable.round(4),
            use_container_width=True,
            hide_index=True,
        )

    # ==================================================
    # REPEATED (100 RUNS) COMPARISON
    #
    # Unlike the single run above (which plots how one search converges
    # tree by tree), this repeats both searchModes ("evolutionary" and
    # "random") several times each - always on the exact same graph (built
    # once) and with the exact same run parameters (approach, stopping
    # criterion, available robots, population size) chosen above - so the
    # methods are compared under identical conditions across many repeats,
    # like the Approach Test / Baseline Test tables.
    # ==================================================

    st.markdown("---")
    st.markdown("### 100-Run Comparison: Evolution vs. Random Spanning Trees")

    evo_repeated_num_of_runs = st.slider(
        "Number of runs per method",
        2,
        500,
        st.session_state.evo_repeated_num_of_runs,
        key="evo_repeated_num_of_runs_slider",
    )

    st.session_state.evo_repeated_num_of_runs = evo_repeated_num_of_runs

    st.caption(
        f"Both methods run {evo_repeated_num_of_runs}× each, on the exact "
        "same graph and with the same run parameters chosen above "
        "(approach, stopping criterion, available robots, initial "
        "generation size)."
    )

    if st.button(
        "Run 100x Comparison",
        use_container_width=True,
        key="evo_repeated_run_button",
    ):

        with st.spinner(
            f"Running {evo_repeated_num_of_runs} runs × 2 methods "
            f"({evo_approach})..."
        ):

            if is3D:
                repeated = threeDTest.runSpanningTreeEvolution3DRepeated(
                    sceneName=scene_name,
                    approachName=evo_approach,
                    availableRobots=evo_available_robots,
                    availableTime=evo_computation_time,
                    maxTrees=evo_max_trees,
                    populationSize=evo_population_size,
                    numOfRuns=evo_repeated_num_of_runs,
                    epsilon=epsilon,
                    probBudget=prob_budget,
                )
            else:
                repeated = runSpanningTreeEvolutionRepeated(
                    approachName=evo_approach,
                    detecRad=detection_radius,
                    availableRobots=evo_available_robots,
                    availableTime=evo_computation_time,
                    maxTrees=evo_max_trees,
                    populationSize=evo_population_size,
                    numOfRuns=evo_repeated_num_of_runs,
                    epsilon=epsilon,
                    probBudget=prob_budget,
                )

        st.session_state.evo_repeated_results = {
            "is3D": is3D,
            "sceneName": scene_name if is3D else None,
            "approach": evo_approach,
            "numOfRuns": evo_repeated_num_of_runs,
            "evolutionary": repeated["evolutionary"],
            "random": repeated["random"],
        }

    repeatedResults = st.session_state.evo_repeated_results

    if repeatedResults is not None:

        def buildRepeatedComparisonRow(runResult, methodLabel):

            bestFitness = runResult["bestFitness"]
            checkedTrees = runResult["checkedTrees"]

            finiteMask = np.isfinite(bestFitness)
            finiteValues = bestFitness[finiteMask]

            return {
                "Method": methodLabel,
                "Runs": repeatedResults["numOfRuns"],
                "Spanning Trees Checked (avg)": float(np.mean(checkedTrees)),
                "Best": float(np.min(finiteValues)) if finiteValues.size > 0 else None,
                "Mean": float(np.mean(finiteValues)) if finiteValues.size > 0 else None,
                "Variance": float(np.var(finiteValues)) if finiteValues.size > 0 else None,
                "No Clearance Count": int(np.sum(~finiteMask)),
            }

        repeatedComparisonTable = pd.DataFrame([
            buildRepeatedComparisonRow(repeatedResults["evolutionary"], "Evolutionary Search"),
            buildRepeatedComparisonRow(repeatedResults["random"], "Random Spanning Tree Generation"),
        ])

        st.caption(
            f"**{repeatedResults['approach']}** — "
            f"{repeatedResults['numOfRuns']} runs per method, all on the "
            "same graph."
            + (f" Scene: **{repeatedResults['sceneName']}**." if repeatedResults.get("is3D") else "")
        )

        st.dataframe(
            repeatedComparisonTable.round(4),
            use_container_width=True,
            hide_index=True,
        )


elif app_mode == "Closing Exits Test":

    # ==================================================
    # CLOSING EXITS TEST
    #
    # Compares Greedy BLabel Order's graphSearch() (a full clearance if one
    # is found, otherwise its computeClosingExits() fallback: local
    # clearances + guards + a final FHPE_SA patrol) against a plain FHPE_SA
    # baseline that skips local clearing entirely and starts from the exact
    # same root. Both are scored by simulating a randomly moving, possibly
    # "hurt"-and-frozen target - see closingExitsTest.py's module docstring
    # for the exact model. Always runs on the 2D synthetic grid - the
    # comparison itself has nothing scene-specific about it.
    # ==================================================


    if is3D:
        st.info(
            "The Closing Exits test always runs on the 2D synthetic grid - "
            "the 3D scene selection above is ignored here."
        )


    st.markdown("### Run Parameters")

    ce_num_of_runs = st.slider(
        "Number of runs per graph",
        1,
        1000,
        st.session_state.ce_num_of_runs,
    )

    st.session_state.ce_num_of_runs = ce_num_of_runs

    ce_stopping_criterion = st.radio(
        "Stop after",
        [
            "Computation time",
            "Number of spanning trees",
        ],
        horizontal=True,
        index=[
            "Computation time",
            "Number of spanning trees",
        ].index(st.session_state.ce_stopping_criterion),
    )

    st.session_state.ce_stopping_criterion = ce_stopping_criterion

    if ce_stopping_criterion == "Computation time":

        ce_computation_time = st.slider(
            "Computation Time (s)",
            1,
            60,
            st.session_state.ce_computation_time,
        )

        st.session_state.ce_computation_time = ce_computation_time

        ce_max_trees = None

    else:

        ce_max_trees = st.slider(
            "Number of spanning trees",
            100,
            10000,
            st.session_state.ce_max_trees,
            step=100,
        )

        st.session_state.ce_max_trees = ce_max_trees

        ce_computation_time = None

    ce_available_robots = st.slider(
        "Available robots",
        1,
        50,
        st.session_state.ce_available_robots,
    )

    st.session_state.ce_available_robots = ce_available_robots

    ce_horizon = st.slider(
        "FHPE_SA horizon (max edges per planned path)",
        1,
        30,
        st.session_state.ce_horizon,
    )

    st.session_state.ce_horizon = ce_horizon

    # Reuses the SAME global "FHPE+SA probabilistic search budget" slider
    # every other mode already shows near the top of the page (prob_budget)
    # instead of a second, redundant one here.
    ce_prob_budget = prob_budget

    ce_hurt_probability = st.slider(
        "Target hurt probability q (per step)",
        0.0,
        1.0,
        st.session_state.ce_hurt_probability,
        step=0.01,
    )

    st.session_state.ce_hurt_probability = ce_hurt_probability


    # ==================================================
    # SIMULATE
    # ==================================================


    st.markdown("---")

    if st.button(
        "Simulate Target Search",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            f"Running {ce_num_of_runs} target-search simulations..."
        ):

            ceResult = runClosingExitsComparison(
                detecRad=detection_radius,
                numOfRuns=ce_num_of_runs,
                availableRobots=ce_available_robots,
                availableTime=ce_computation_time,
                maxTrees=ce_max_trees,
                horizon=ce_horizon,
                probBudget=ce_prob_budget,
                hurtProbability=ce_hurt_probability,
            )

        st.session_state.ce_simulation_results = {
            **ceResult,
            "available_robots": ce_available_robots,
            "horizon": ce_horizon,
            "prob_budget": ce_prob_budget,
            "hurt_probability": ce_hurt_probability,
        }


    #------------------------------------------------------------------
    # RESULTS
    #------------------------------------------------------------------


    ceResults = st.session_state.ce_simulation_results

    if ceResults is not None:

        st.markdown("### Environment")

        st.plotly_chart(
            buildEnvironmentFigure(
                obstacles,
                ceResults["priors"],
                H,
                W,
                G=ceResults["G"],
                regularEdges=list(ceResults["G"].edges.keys()),
            ),
            use_container_width=True,
        )

        st.markdown("### Target-Finding Comparison")

        st.caption(
            "Mean / P90 find time are computed over runs that actually "
            "found the target (timed-out runs are excluded there, and "
            "reported separately via the timeout rate). Clearance mass is "
            "the mean fraction of graph nodes that ended up permanently "
            "cleared (1.0 on a full clearance) - always 0 for plain "
            "FHPE_SA, since it never clears anything."
        )

        summaryTable = pd.DataFrame([
            {
                "Strategy": name,
                "Nodes": ceResults["numGraphNodes"],
                "Edges": ceResults["numGraphEdges"],
                "Runs": len(ceResults["findTimesClosingExits"]),
                "Available Robots": ceResults["available_robots"],
                "Horizon (edges)": ceResults["horizon"],
                "Prob Budget": ceResults["prob_budget"],
                "Hurt Probability q": ceResults["hurt_probability"],
                "Mean Find Time": stats["mean_find_time"],
                "Variance Find Time": stats["variance_find_time"],
                "P90 Find Time": stats["p90_find_time"],
                "Timeout Rate": stats["timeout_rate"],
                "Clearance Mass": stats["clearance_mass"],
            }
            for name, stats in ceResults["summary"].items()
        ])

        st.dataframe(
            summaryTable.round(4),
            use_container_width=True,
            hide_index=True,
        )
