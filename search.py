import math
import heapq
import sys

class Node:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.g = float('inf')  # Cost from start to current node
        self.h = 0  # Heuristic (estimated cost to goal)
        self.f = float('inf')  # Total cost (g + h)
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
    # Euclidean distance
    return math.sqrt((node.x - goal_node.x)**2 + (node.y - goal_node.y)**2)

def bfs_search(nodes, edges, start_id, goals):
    return None, nodes_explored

def dfs_search(nodes, edges, start_id, goals):
    return None, nodes_explored

def gbfs_search(nodes, edges, start_id, goals):
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

def main():
    # Allow optional command-line args: filename and method
    filename = "PathFinder-test.txt"
    method_arg = None
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    if len(sys.argv) >= 3:
        method_arg = sys.argv[2].strip().upper()

    # Show graph information before running search
    print("\nGraph information (from PathFinder-test.txt):\n")
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

    print("Available Search Algorithms:")
    print("1. Breadth-First Search (BFS)")
    print("2. Depth-First Search (DFS)")
    print("3. Greedy Best-First Search (GBFS)")
    print("4. A* Search (AS)")

    choice = None
    if method_arg:
        map_arg = {
            "BFS": "1", "DFS": "2", "GBFS": "3", "AS": "4",
            "1": "1", "2": "2", "3": "3", "4": "4"
        }
        choice = map_arg.get(method_arg)
        if choice is None:
            print(f"\nUnknown method '{method_arg}'. Falling back to interactive selection.\n")

    # interactive if no valid method arg
    while choice not in ['1','2','3','4']:
        choice = input("\nEnter the number of your chosen search algorithm (1-4): ").strip()
        if choice in ['1','2','3','4']:
            break
        print("Invalid choice. Please enter a number between 1 and 4.")

    # Read input file (uses filename variable)
    nodes, edges, origin, destinations = parse_input_file(filename)

    # Run selected search algorithm
    if choice == '1':
        search_method = "Breadth-First Search"
        path, nodes_explored = bfs_search(nodes, edges, origin, destinations)
    elif choice == '2':
        search_method = "Depth-First Search"
        path, nodes_explored = dfs_search(nodes, edges, origin, destinations)
    elif choice == '3':
        search_method = "Greedy Best-First Search"
        path, nodes_explored = gbfs_search(nodes, edges, origin, destinations)
    else:
        search_method = "A* Search"
        # enable verbose trace for A* so user sees step-by-step exploration
        path, nodes_explored = a_star_search(nodes, edges, origin, destinations, verbose=True)

    if path:
        print(f"\nSearch Method: {search_method}")
        print(f"Number of nodes explored: {nodes_explored}")
        path_with_coords = " -> ".join(f"{nid}({nodes[nid].x},{nodes[nid].y})" for nid in path)
        print(f"Path found (id(x,y)): {path_with_coords}")
    else:
        print("\nNo path found!")

if __name__ == "__main__":
    main()