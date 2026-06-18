from collections import defaultdict
from dataclasses import dataclass

#------------------------------------------------
# MODELLING EDGES OF A DIRECTED GRAPH
#------------------------------------------------
@dataclass
class Edge:
    u: int
    v: int
    travel_time: float
    robot_type: int

#------------------------------------------------
# MODELLING A DIRECTED GRAPH
#------------------------------------------------

class Graph:
    def __init__(self):
        self.nodes = {}
        self.adj = defaultdict(list)   # Saving all adjacent nodes for each node
        self.edges = []                # Saving all edges

    #Adding a new node
    def add_node(self, id, prio):
        self.nodes[id] = prio

    #Adding an edge between two existing nodes
    def add_edge(self, u, v, travel_time, robot_type):
        e = Edge(u, v, travel_time, robot_type)
        self.adj[u].append(e)
        self.edges.append(e)

    #Returning all edges to neighbours of a node
    def neighbors(self, u):
        return self.adj[u]
