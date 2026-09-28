# 🤖 Guaranteed Multi-Robot Clearance with Probabilistic Priors

**Bachelor Thesis Project · TU Darmstadt**

This project investigates how multiple robots can **systematically search and clear an environment while taking probabilistic information about the target's location into account**.

The goal is to combine the guarantees of **graph-based search and clearance algorithms** with the efficiency of **probabilistic search strategies**. The resulting approach aims to coordinate multiple robots in complex environments while minimizing the expected time required to find a hidden target.

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

One of the objectives considered in the project is the expected search time:

$$
\mathbb{E}[T]
=
\sum_{v \in V} p(v)\,t_{\mathrm{first}}(v),
$$

where \(p(v)\) denotes the prior probability of the target being located at node \(v\), and \(t_{\mathrm{first}}(v)\) denotes the time at which node \(v\) is first cleared.

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

There are several directions in which the approach could be extended.

### More Efficient Search

The current approach evaluates a potentially large number of candidate exploration strategies. More sophisticated search and optimization methods could reduce the computational effort.

### Dynamic Priors

The current framework primarily considers static prior information. A future system could update the probability distribution dynamically as robots collect new information.

### Larger Robot Teams

The optimization of exploration orders becomes increasingly difficult as the number of robots grows. More scalable approaches could be investigated for larger teams.

### Real-World Deployment

The current evaluation is simulation-based. An important next step would be to transfer the approach to physical robots and investigate the effects of sensor noise, localization errors, communication delays, and imperfect motion.

### Learning-Based Approaches

Machine learning could be investigated as an additional mechanism for predicting promising exploration strategies or learning search policies from previously solved environments.

---

## ▶️ Running the Project

### Requirements

* MATLAB [version]
* [Additional dependencies]
* [Required toolboxes]

### Installation

Clone the repository:

```bash
git clone https://github.com/paul-sterz/multi-robot-clearance.git
cd multi-robot-clearance
```

### Running an Experiment

[Describe the main entry point here.]

For example:

```matlab
[main script / function]
```

The repository contains implementations of the different approaches in the following directories:

```text
Baseline/
Modified Baseline/
My Approach/
Monte Carlo Simulation/
Sydney-Decomposition Approach/
```

Further information about the individual experiments can be found in the corresponding directories.

---

## 🎥 Visualization

The following video shows the developed multi-robot clearance algorithm in action.

<!-- Replace with embedded video / GIF / link -->

[▶️ Watch the visualization](VIDEO_LINK)

The visualization demonstrates the coordinated exploration of the environment and the resulting clearance strategy.

---

## 📄 License

[Add license if applicable.]
