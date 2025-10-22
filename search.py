import math
import heapq
import sys
from collections import deque

class Node:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.parent = None

def parse_input_file(filename):
    nodes = {}
    edges = {}
    origin = None
    destinations = []
    
    with open(filename, 'r') as file:
        section = ""
        for line in file:
            line = line.strip()
            if line.startswith("Nodes:"):
                section = "nodes"
            elif line.startswith("Edges:"):
                section = "edges"
            elif line.startswith("Origin:"):
                section = "origin"
            elif line.startswith("Destinations:"):
                section = "destinations"
            elif line:
                if section == "nodes":
                    node_id, coords = line.split(": ")
                    x, y = eval(coords)
                    nodes[int(node_id)] = Node(int(node_id), x, y)
                elif section == "edges":
                    edge, weight = line.split(": ")
                    node1, node2 = eval(edge)
                    if node1 not in edges:
                        edges[node1] = {}
                    edges[node1][node2] = float(weight)
                elif section == "origin":
                    origin = int(line)
                elif section == "destinations":
                    destinations = [int(x) for x in line.split(';')]
    
    return nodes, edges, origin, destinations

def heuristic(node, goal_node):
    return math.sqrt((node.x - goal_node.x)**2 + (node.y - goal_node.y)**2)

def bfs_search(nodes, edges, start_id, goals, verbose=False):
    queue = deque([(start_id, [start_id])])
    visited = set([start_id])
    nodes_explored = 0
    
    if verbose:
        print("\nBFS Expansion Trace:")
    
    while queue:
        current_id, path = queue.popleft()
        nodes_explored += 1
        
        if verbose:
            current = nodes[current_id]
            print(f"Expand node {current_id} ({current.x},{current.y})")
        
        if current_id in goals:
            return path, nodes_explored
        
        if current_id in edges:
            neighbors = [(nid, w) for nid, w in edges[current_id].items()]
            for neighbor_id, _ in sorted(neighbors, key=lambda x: x[0]):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
    
    return None, nodes_explored

def a_star_search(nodes, edges, start_id, goals, verbose=False):
    # reset node costs (so repeated runs work)
    for n in nodes.values():
        n.g = float('inf')
        n.h = 0
        n.f = float('inf')
        n.parent = None

    start_node = nodes[start_id]
    start_node.g = 0
    start_node.h = min(heuristic(start_node, nodes[goal]) for goal in goals)
    start_node.f = start_node.g + start_node.h

    counter = 0
    open_set = [(start_node.f, counter, start_node.id)]
    counter += 1
    closed_set = set()
    nodes_explored = 0

    if verbose:
        print("\nA* Trace:")
        print(f"Start node: {start_id} g={start_node.g:.2f} h={start_node.h:.2f} f={start_node.f:.2f}")

    while open_set:
        _, _, current_id = heapq.heappop(open_set)
        current = nodes[current_id]
        nodes_explored += 1

        if verbose:
            print(f"\nExpand node {current_id}: g={current.g:.2f} h={current.h:.2f} f={current.f:.2f} (expanded count={nodes_explored})")

        if current_id in goals:
            # Goal reached
            if verbose:
                print(f"Goal {current_id} reached.")
            path = []
            while current:
                path.append(current.id)
                current = current.parent
            return path[::-1], nodes_explored

        closed_set.add(current_id)

        # Check neighbors
        if current_id in edges:
            for neighbor_id, weight in edges[current_id].items():
                if neighbor_id in closed_set:
                    if verbose:
                        print(f"  Neighbor {neighbor_id}: skipped (in closed set)")
                    continue

                neighbor = nodes[neighbor_id]
                tentative_g = current.g + weight
                if verbose:
                    print(f"  Neighbor {neighbor_id}: edge_weight={weight} tentative_g={tentative_g:.2f} (current g={neighbor.g if neighbor.g!=float('inf') else 'inf'})")

                if tentative_g < neighbor.g:
                    neighbor.parent = current
                    neighbor.g = tentative_g
                    neighbor.h = min(heuristic(neighbor, nodes[goal]) for goal in goals)
                    neighbor.f = neighbor.g + neighbor.h
                    heapq.heappush(open_set, (neighbor.f, counter, neighbor_id))
                    counter += 1
                    if verbose:
                        print(f"    -> updated: g={neighbor.g:.2f} h={neighbor.h:.2f} f={neighbor.f:.2f} pushed to open set")
                else:
                    if verbose:
                        print("    -> not improved, not pushed")

        if verbose:
            # show simple open set snapshot
            open_snapshot = ", ".join(f"{nid}(f={f:.2f})" for f,_,nid in open_set)
            print(f"  Open set: [{open_snapshot}]")
            print(f"  Closed set: {sorted(list(closed_set))}")

    return None, nodes_explored

def run_search(method, nodes, edges, start_id, goals, verbose=False):
    """Wrapper function to run the selected search method"""
    search_methods = {
        'BFS': ('Breadth-First Search', bfs_search),
        'DFS': ('Depth-First Search', None),  # Placeholder for DFS
        'GBFS': ('Greedy Best-First Search', None),  # Placeholder for GBFS
        'AS': ('A* Search', a_star_search)
    }
    
    if method not in search_methods:
        raise ValueError(f"Unknown search method: {method}")
    
    method_name, search_func = search_methods[method]
    
    if search_func is None:
        print(f"\n{method_name} is not implemented yet!")
        return None, 0
    
    path, nodes_explored = search_func(nodes, edges, start_id, goals, verbose)
    
    return path, nodes_explored, method_name

def print_graph_info(filename):
    """Print the graph information"""
    print(f"\nGraph information (from {filename}):\n")
    print("Nodes:")
    print("1: (4,1)")
    print("2: (2,2)")
    print("3: (4,4)")
    print("4: (6,3)")
    print("5: (5,6)")
    print("6: (7,5)\n")
    print("Edges:")
    print("(2,1): 4")
    print("(3,1): 5")
    print("(1,3): 5")
    print("(2,3): 4")
    print("(3,2): 5")
    print("(4,1): 6")
    print("(1,4): 6")
    print("(4,3): 5")
    print("(3,5): 6")
    print("(5,3): 6")
    print("(4,5): 7")
    print("(5,4): 8")
    print("(6,3): 7")
    print("(3,6): 7\n")
    print("Origin:")
    print("2\n")
    print("Destinations:")
    print("5; 4\n")

def main():
    # Parse command line arguments
    filename = "PathFinder-test.txt"
    method_arg = None
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    if len(sys.argv) >= 3:
        method_arg = sys.argv[2].strip().upper()

    # Print graph information
    print_graph_info(filename)

    # Show available search methods
    print("\nAvailable Search Algorithms:")
    print("1. Breadth-First Search (BFS)")
    print("2. Depth-First Search (DFS)")
    print("3. Greedy Best-First Search (GBFS)")
    print("4. A* Search (AS)")

    # Get search method choice
    method_map = {
        '1': 'BFS', '2': 'DFS', '3': 'GBFS', '4': 'AS',
        'BFS': 'BFS', 'DFS': 'DFS', 'GBFS': 'GBFS', 'AS': 'AS'
    }

    choice = method_arg
    while choice not in method_map:
        choice = input("\nEnter your choice (1-4): ").strip().upper()
        if choice not in method_map:
            print("Invalid choice. Please try again.")

    # Parse input file and run search
    nodes, edges, origin, destinations = parse_input_file(filename)
    
    # Run the selected search method
    method = method_map[choice]
    path, nodes_explored, method_name = run_search(method, nodes, edges, origin, destinations, verbose=True)

    # Print results
    if path:
        reached_goal = next(goal for goal in destinations if goal in path)
        print(f"\nSearch Method: {method_name}")
        print(f"Number of nodes explored: {nodes_explored}")
        path_with_coords = " -> ".join(f"{nid}({nodes[nid].x},{nodes[nid].y})" for nid in path)
        print(f"Path found (id(x,y)): {path_with_coords}")
        print(f"Destination reached: {reached_goal} ({nodes[reached_goal].x},{nodes[reached_goal].y})")
    else:
        print("\nNo path found!")

if __name__ == "__main__":
    main()