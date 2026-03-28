# parses the .txt file to read info regarding: 
# Reads: Nodes, Edges, Origin and Destination
# produces: nodes (dict id -> (x,y)), directed graph with edge costs
#   origin (int), destinations (set of ints)
# 

import re
import sys
from typing import Dict, Tuple, Set
from W3_Search import Graph, GraphProblem # imports existing classes 

import numpy as np
# Regular expressions to parse node and edge lines.
# NODE_RE matches lines like: 1: (4,1)  (node id : (x,y))
NODE_RE = re.compile(r'^\s*(\d+)\s*:\s*\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)\s*$')
# EDGE_RE matches lines like: (2,1): 4  (edge from 2 to 1 with cost 4)
EDGE_RE = re.compile(r'^\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*:\s*([-\d\.]+)\s*$')


def parse_problem_file(path: str):
    """
    Parse the problem file and return a tuple:
      (nodes, graph, origin, destinations)

    - nodes: Dict[int, (x, y)]  -- coordinates stored as floats
    - graph: Graph (directed)   -- edges added with their costs
    - origin: int               -- starting node id
    - destinations: Set[int]    -- set of goal node ids

    Raises ValueError on malformed lines or missing required sections.
    """
    nodes: Dict[int, Tuple[float, float]] = {}
    graph = Graph(graph_dict={}, directed=True)  # directed graph for one-way edges
    origin = None
    destinations: Set[int] = set()

    section = None  # current section being parsed

    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            # Trim whitespace and skip empty lines
            line = raw.strip()
            if not line:
                continue

            # Detect section headers (case-insensitive)
            low = line.lower()
            if low.startswith('nodes:'):
                section = 'nodes'
                continue
            elif low.startswith('edges:'):
                section = 'edges'
                continue
            elif low.startswith('origin:'):
                section = 'origin'
                # Origin may be on the same line after the colon, e.g. "Origin: 2"
                rest = line[len('origin:'):].strip()
                if rest:
                    origin = int(rest)
                continue
            elif low.startswith('destinations:'):
                section = 'destinations'
                # Destinations may be inline after the header, separated by semicolons
                rest = line[len('destinations:'):].strip()
                if rest:
                    for token in rest.split(';'):
                        token = token.strip()
                        if token:
                            destinations.add(int(token))
                continue

            # Parse content lines depending on the active section
            if section == 'nodes':
                # Expect lines like: 1: (4,1)
                m = NODE_RE.match(line)
                if not m:
                    raise ValueError(f"Invalid node line: {line}")
                nid = int(m.group(1))
                x = float(m.group(2))
                y = float(m.group(3))
                nodes[nid] = (x, y)

            elif section == 'edges':
                # Expect lines like: (2,1): 4
                m = EDGE_RE.match(line)
                if not m:
                    raise ValueError(f"Invalid edge line: {line}")
                a = int(m.group(1))
                b = int(m.group(2))
                cost = float(m.group(3))
                # Use connect1 to add a directed edge from a -> b with given cost
                graph.connect1(a, b, cost)

            elif section == 'origin':
                # If origin wasn't provided on the header line, parse it here
                if origin is None:
                    origin = int(line)

            elif section == 'destinations':
                # Destinations lines may contain multiple ids separated by semicolons
                for token in line.split(';'):
                    token = token.strip()
                    if token:
                        destinations.add(int(token))

            else:
                # Ignore lines outside recognized sections
                continue

    # Validate required fields
    if origin is None:
        raise ValueError("Origin not found in file.")
    if not destinations:
        raise ValueError("Destinations not found in file.")

    return nodes, graph, origin, destinations


class RouteFindingProblem(GraphProblem):
    """
    Subclass of GraphProblem tailored for the assignment:
    - Accepts multiple destinations (goals).
    - Ensures deterministic neighbor ordering (ascending numeric order)
      to satisfy the assignment tie-breaking rules.
    - Keeps node coordinates for heuristic calculations (used later by GBFS/A*).
    """

    def __init__(self,
                 initial: int,
                 goals: Set[int],
                 graph: Graph,
                 nodes_coords: Dict[int, Tuple[float, float]]):
        # GraphProblem constructor accepts a single goal or a list of goals.
        # We pass a list so the default goal_test (which checks membership)
        # would work, but we override goal_test below for clarity.
        super().__init__(initial, list(goals), graph)
        # Keep a set of destinations for fast membership checks
        self.destinations = set(goals)
        # Store coordinates for heuristics (node id -> (x,y))
        self.nodes_coords = nodes_coords

    def actions(self, A):
        """
        Return the list of neighbor node ids reachable from node A.
        The list is sorted in ascending numeric order to enforce the
        assignment's tie-breaking rule (expand smaller node ids first).
        """
        neighbors = list(self.graph.get(A).keys())
        try:
            neighbors_sorted = sorted(neighbors, key=int)
        except Exception:
            neighbors_sorted = sorted(neighbors)
        return neighbors_sorted

    def goal_test(self, state):
        """Return True if the given state (node id) is one of the destinations."""
        return state in self.destinations

    # path_cost is inherited from GraphProblem and uses the graph's edge costs.


# -------------------------
# Hard-coded default path
# -------------------------
# If you prefer a hard-coded default file path when running parser.py without
# arguments, set DEFAULT_PROBLEM_PATH below to the absolute path of your file.
# Change this string to match your local file location if needed.
DEFAULT_PROBLEM_PATH = r"C:\Users\arifa\OneDrive\Desktop\intro2ai\intro2ai tutorialweek03\PathFinder-test.txt"
# !!! ^ Local directory on Ari's computer, if running locally, you will need to change this one. 

# If run as a script, perform a quick parse test and print a summary.
if __name__ == "__main__":
    # Use the command-line argument if provided; otherwise fall back to the hard-coded path.
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
    else:
        filepath = DEFAULT_PROBLEM_PATH

    try:
        nodes, graph, origin, destinations = parse_problem_file(filepath)
    except Exception as e:
        print("Error parsing file:", e)
        sys.exit(2)

    # Print a concise summary of the parsed problem
    print("Parsed problem file:", filepath)
    print(" Origin:", origin)
    print(" Destinations:", sorted(destinations))
    print(" Nodes (count):", len(nodes))
    print(" Sample nodes (id -> coords):")
    for nid in sorted(nodes)[:10]:
        print(f"  {nid}: {nodes[nid]}")
    print(" Sample edges (first 20):")
    count = 0
    for a in sorted(graph.graph_dict.keys()):
        for b, cost in graph.get(a).items():
            print(f"  ({a} -> {b}) cost={cost}")
            count += 1
            if count >= 20:
                break
        if count >= 20:
            break
