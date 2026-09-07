import numpy as np
import streamlit as st
import plotly.graph_objects as go


from approachTest import (
    buildEnvironment,
    detectionFnc,
    graphBuilder,
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

    ("available_robots", 6),

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
# not re-derived here.
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
# PLOTLY FIGURE — ENVIRONMENT ONLY (NO GRAPH)
# ==================================================


OBSTACLE_GRAY = "#4a4a4a"

START_GREEN = "#2a9d8f"

HEATMAP_RED_SOLID = "#d62828"


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
# PRIOR HEATMAP
#
# Shades every free cell red, with opacity scaling with
# its (normalized) prior value. Higher prior -> redder.
# ==================================================


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


# ==================================================
# START REGION
# ==================================================


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


st.plotly_chart(
    fig,
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

    pass


#------------------------------------------------------------------
# PLOTS OF THE RESULTS
#------------------------------------------------------------------
