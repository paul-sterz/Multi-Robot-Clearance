# 🤖 Guaranteed Multi-Robot Clearance with Probabilistic Priors

**Bachelor Thesis Project · TU Darmstadt**

This project investigates how multiple robots can **systematically search and clear an environment while taking probabilistic information about the target's location into account**.

The goal is to combine the guarantees of **graph-based search and clearance algorithms** with the efficiency of **probabilistic search strategies**. The resulting approach aims to coordinate multiple robots in complex environments while minimizing the expected time required to find a hidden target.

---
## 🎥 Visualization

<video src="./visualization.mp4" controls width="800"></video>
---

## 🛠️ Technologies & Methods

The project combines concepts from mathematics, computer science, robotics, and optimization.

### Programming & Tools

* **Python** — algorithm implementation and simulation
* **Three.js (WebGL)** — interactive 3D visualization of real-world clearing 
* **Git** — version control
* **LaTeX** — scientific documentation

### Mathematical & Algorithmic Foundations

* **Graph Theory** — representation of environments as graphs
* **Guaranteed Search with Spanning Trees** — guaranteed exploration and clearance on trees
* **Probability & Statistical Modeling** — representation of prior information about the target location
* **Optimization of the Exploration Order** — optimization of the order in which regions are explored
* **Evolutionary Algorithms for Spanning Tree Generation** — generation and optimization of promising spanning trees
* **Raycasting & Bresenham's Algorithm** — visibility and sensor detection
* **Multi-Robot Coordination** — coordination of multiple robots during exploration

---

## 🔄 The Process

The project builds upon the **Graph Search and Spanning Tree (GSST)** framework for guaranteed graph-based clearance and extends it with **probabilistic target priors and multi-robot exploration optimization**.

The overall process can be summarized as:

```text
Environment
     ↓
Graph Representation
     ↓
Probabilistic Prior Model
     ↓
Spanning Tree Generation
     ↓
Exploration Order Optimization
     ↓
Exploration Strategy
     ↓
Clearance Feasibility Check
     ↓
Evaluation
```

### 1. Environment Representation

The environment is represented as a graph in which nodes correspond to spatial cells and edges represent possible movements between neighboring cells.

For 3D environments, the graph is constructed from the underlying spatial representation while taking obstacles into account. Sensor visibility is modeled using raycasting to determine which cells can be detected from each robot position.

### 2. Probabilistic Prior

Instead of assuming that the target is equally likely to be located anywhere in the environment, the search incorporates a **probabilistic prior** describing where the target is expected to be found.

The prior is modeled using a **mixture of Gaussian distributions**, allowing spatial information to be represented by multiple regions of increased probability.

The prior influences the exploration objective: regions with a higher probability of containing the target should ideally be explored earlier.

### 3. Spanning Tree Generation

The graph is transformed into spanning trees that define possible systematic exploration strategies.

Different spanning trees result in different exploration paths and therefore different clearance and search times. Several strategies for generating spanning trees are investigated, including an **evolutionary approach** designed to identify promising trees efficiently.

### 4. Exploration Order Optimization

For multiple robots, determining **which parts of the environment should be explored first** introduces an additional optimization problem.

The exploration order is optimized with respect to the probabilistic prior and the time at which individual regions are first explored.

### 5. Exploration Strategy

The optimized exploration order defines a sequence of tasks that must be assigned to the available robots.

This results in a **multi-level assignment problem**, which is approximated by repeatedly solving **Linear Bottleneck Assignment Problems (LBAPs)**. This provides an efficient way of distributing exploration tasks among multiple robots while accounting for their current positions and travel times.

### 6. Clearance Feasibility

Guaranteed clearance is not always feasible with the available number of robots. If the required clearance strategy cannot be executed with the given robot team, the algorithm switches to a **probabilistic clearance approach**.

This allows the robots to continue searching even when guaranteed clearance cannot be maintained, while prioritizing regions according to the underlying probability distribution.

### 7. Evaluation

The developed algorithms are evaluated through simulation experiments in both **custom-built and real-world environments**.

The experiments investigate the behavior and performance of the different approaches with respect to factors such as the number of robots, environment structure, probabilistic priors, and exploration strategy.



---

## 🧠 What I Learned

This project provided practical experience at the intersection of mathematical theory, algorithm development, and simulation.

### Algorithm Development

* Translating mathematical ideas into working algorithms
* Designing and modifying graph-search algorithms
* Working with complex interacting algorithmic components
* Handling edge cases and infeasible clearance situations

### Optimization

* Formulating exploration strategies as optimization problems
* Designing objective functions based on probabilistic information
* Working with evolutionary algorithms
* Comparing different search and optimization strategies

### Robotics & Simulation

* Modeling robot motion in structured environments
* Coordinating multiple robots
* Representing obstacles and sensor visibility
* Extending algorithms from 2D to 3D environments

### Research

* Designing experiments to evaluate algorithmic approaches
* Interpreting simulation results
* Identifying limitations of theoretical assumptions
* Iteratively improving an algorithm based on experimental observations

---

## 🚀 How It Can Be Improved

Despite the promising results, several limitations remain and provide interesting directions for future work.

### More Realistic Target and Sensor Models

The current framework assumes **perfect sensing**, a **static spatial prior**, and a target with **unbounded speed**. Future work could incorporate temporally changing and correlated priors to model target movement and recency effects, as well as probabilistic sensor models to account for imperfect detection.

### Continuous Robot Trajectories

The current GSST framework only considers detection at graph nodes. Robots cannot detect parts of the environment while moving along edges.

Extending the framework to **continuous robot trajectories** could therefore lead to more efficient strategies by allowing robots to detect their surroundings while travelling. This would require a more detailed treatment of trajectories, visibility, and potential recontamination during movement.

### Non-Monotone Search

GSST is restricted to **monotone search strategies**, which means that previously cleared regions cannot become contaminated again. As a result, some environments may require more robots than are actually necessary for clearance.

Extending the approach to **non-monotone strategies** could enable additional environments to be cleared with fewer robots. However, this would substantially increase the complexity of strategy generation and execution.

### Improved Local Clearance Decisions

The current local clearance strategy uses a heuristic to determine when performing local clearance is beneficial.

This decision could be improved by developing a more principled analytical criterion based on the **additional guaranteed probability mass** obtained through clearance relative to the additional time required. Alternatively, a **learned decision model** could predict when local clearance is advantageous based on properties of the environment and its prior distribution.

### More Accurate Objective Functions

The current objective is based on node probabilities and does not fully account for **overlapping detection sets**. Incorporating the actual probability mass covered by each detection set could provide a more accurate estimate of expected search performance.

Overall, extending the framework beyond the modelling assumptions and structural constraints of GSST could lead to more realistic, efficient, and flexible multi-robot search strategies.

---

## ▶️ Running the Project

#### Requirements

- Python 3.10+ (developed and tested with Python 3.14)
- Install the required dependencies:

```bash
pip install -r strategy_service/requirements.txt -r clearing-scenes/requirements.txt
```

#### Start the Server

From the repository root, run:

```bash
python -m strategy_service.server
```

Once the server has started, open [**http://localhost:8000**](http://localhost:8000) in your browser.

#### Usage

1. **Select a scene** — Choose one of the available real-world environments, such as Christ Church, Blenheim Palace, or the Bodleian Library.
2. **Define the search problem** — Place hotspots and starting vertices directly on the 3D surface.
3. **Configure the strategy** — Select an approach (DP, DP B-Label Order, Greedy, or Greedy B-Label Order) and specify the available robot and time budget.
4. **Run the algorithm** — Click **Run** to compute the clearance strategy.
5. **Visualize the result** — The computed strategy is automatically played back in the 3D environment, allowing the multi-robot exploration to be observed in real time.


