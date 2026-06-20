from collections import defaultdict
from dataclasses import dataclass


#------------------------------------------------
# MODELLING Nodes OF A DIRECTED GRAPH
#------------------------------------------------
@dataclass(frozen=True)
class Node:
    idx: int
    pos: tuple[int,int]
    prior: float

#------------------------------------------------
# MODELLING EDGES OF A DIRECTED GRAPH
#------------------------------------------------
@dataclass(frozen=True)
class Edge:
    u: Node
    v: Node
    time: float
    robotType: int

#------------------------------------------------
# MODELLING A DIRECTED GRAPH
#------------------------------------------------

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = {}                     # Saving all edges in the form {(u,v) : e}
        self.adj = defaultdict(list)        # Saving all adjacent nodes for each node
                     
    #Adding a new node
    def add_node(self, idx, pos, prior):
        n = Node(idx,pos,prior)
        self.nodes.append(n)

    def add_node2(self, n):
        self.nodes.append(n)

    #Adding an edge between two existing nodes
    def add_edge(self, u : Node, v: Node, time, robotType=2):
        e = Edge(u, v, time, robotType)
        self.adj[u].append(v)
        self.edges[(u.idx, v.idx)] = e

    #Returning all edges to neighbours of a node
    def neighbors(self, u):
        return list(self.adj[u])
