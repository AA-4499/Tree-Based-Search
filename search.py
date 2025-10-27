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

def run_search(method, nodes, edges, start_id, goals, verbose=False):
    """Wrapper function to run the selected search method"""
    search_methods = {
        'BFS': ('Breadth-First Search', bfs_search),
        'DFS': ('Depth-First Search', dfs_search),
        'GBFS': ('Greedy Best-First Search', gbfs_search),
        'AS': ('A* Search', a_star_search),
        'CUS1': ('Custom Uninformed (Cus1)', cus1_search),
        'CUS2': ('Weighted A* Search (w=2.0) (Cus2)', cus2_search)
    }
    
    if method not in search_methods:
        raise ValueError(f"Unknown search method: {method}")
    
    method_name, search_func = search_methods[method]
    
    if search_func is None:
        print(f"\n{method_name} is not implemented yet!")
        return None, 0, method_name, None, None
    
    res = search_func(nodes, edges, start_id, goals, verbose)
    # normalize return values: support legacy (path, nodes_explored) and new (path, nodes_explored, cost, reached_goal)
    if isinstance(res, tuple):
        if len(res) == 2:
            path, nodes_explored = res
            cost = None
            reached_goal = None
        elif len(res) == 4:
            path, nodes_explored, cost, reached_goal = res
        else:
            # fallback
            path = res[0] if len(res) > 0 else None
            nodes_explored = res[1] if len(res) > 1 else 0
            cost = None
            reached_goal = None
    else:
        path = res
        nodes_explored = 0
        cost = None
        reached_goal = None
    
    return path, nodes_explored, method_name, cost, reached_goal

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

# -----------------------
# DFS
# -----------------------
def dfs_search(nodes, edges, start_id, goals, verbose=False):
    """
    Exhaustive DFS that finds the minimum-cost path to any goal.
    Tie-breaker: if equal cost, prefer the path ending at the smaller goal id.
    Returns (path, nodes_explored, cost, reached_goal) where path is list of node ids
    or None if no path.
    """
    best_cost = float('inf')
    best_path = None
    best_goal = None
    nodes_created = 0

    def _dfs(node_id, path, cost, visited):
        nonlocal best_cost, best_path, best_goal, nodes_created
        nodes_created += 1

        if node_id in goals:
            # record if better or tie-breaker by smaller goal id
            if cost < best_cost or (cost == best_cost and (best_goal is None or node_id < best_goal)):
                best_cost = cost
                best_path = path[:]
                best_goal = node_id
            return

        # prune branches that already exceed best known cost
        if cost >= best_cost:
            return

        # neighbors sorted ascending so expansion order is ascending by node id
        for nbr_id, weight in sorted(edges.get(node_id, {}).items(), key=lambda x: x[0]):
            if nbr_id not in visited:
                visited.add(nbr_id)
                _dfs(nbr_id, path + [nbr_id], cost + weight, visited)
                visited.remove(nbr_id)

    # start DFS
    visited = {start_id}
    _dfs(start_id, [start_id], 0.0, visited)

    if best_path is None:
        return None, nodes_created, None, None
    return best_path, nodes_created, best_cost, best_goal

def gbfs_search(nodes, edges, start_id, goals, verbose=False):
    """
    Greedy Best-First Search that expands nodes purely by the heuristic distance
    to the nearest goal. Returns (path, nodes_explored) or (None, nodes_explored)
    if no goal can be reached.
    """
    for node in nodes.values():
        node.parent = None

    def best_heuristic(node_id):
        current = nodes[node_id]
        return min(heuristic(current, nodes[goal]) for goal in goals)

    counter = 0
    open_set = []
    start_h = best_heuristic(start_id)
    heapq.heappush(open_set, (start_h, counter, start_id))
    counter += 1

    explored = set()
    parent = {start_id: None}
    nodes_explored = 0

    if verbose:
        print("\nGBFS Trace:")
        print(f"Start node: {start_id} h={start_h:.2f}")

    while open_set:
        current_h, _, current_id = heapq.heappop(open_set)
        if current_id in explored:
            continue

        explored.add(current_id)
        nodes_explored += 1

        if verbose:
            print(f"\nExpand node {current_id}: h={current_h:.2f} (expanded count={nodes_explored})")

        if current_id in goals:
            path = []
            walker = current_id
            while walker is not None:
                path.append(walker)
                walker = parent[walker]
            return path[::-1], nodes_explored

        for neighbor_id, _ in sorted(edges.get(current_id, {}).items(), key=lambda x: x[0]):
            if neighbor_id in explored:
                if verbose:
                    print(f"  Neighbor {neighbor_id}: skipped (in closed set)")
                continue

            if neighbor_id not in parent:
                parent[neighbor_id] = current_id

            neighbor_h = best_heuristic(neighbor_id)
            # Greedy strategy: always push the neighbor with the lowest heuristic distance first.
            heapq.heappush(open_set, (neighbor_h, counter, neighbor_id))
            counter += 1

            if verbose:
                print(f"  Neighbor {neighbor_id}: h={neighbor_h:.2f} pushed to open set")

        if verbose:
            open_snapshot = ", ".join(f"{nid}(h={hval:.2f})" for hval, _, nid in open_set)
            print(f"  Open set: [{open_snapshot}]")

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

# -----------------------
# Custom searches: Cus1 (uninformed first-path), Cus2 (Weighted A*)
# -----------------------
def cus1_search(nodes, edges, start_id, goals, verbose=False):
    """
    Uninformed search that returns the first path found to any goal (DFS-style, not guaranteed optimal).
    Returns (path, nodes_explored, None, reached_goal) or (None, nodes_explored, None, None).
    """
    stack = [(start_id, [start_id])]
    nodes_explored = 0

    if verbose:
        print("\nCus1 (Uninformed) Trace:")

    while stack:
        current_id, path = stack.pop()  # LIFO -> DFS-like
        nodes_explored += 1

        if verbose:
            cur = nodes[current_id]
            print(f"Expand node {current_id} ({cur.x},{cur.y})")

        if current_id in goals:
            return path, nodes_explored, None, current_id

        # push neighbors in reverse-sorted order so smallest id is expanded first when popped
        for neighbor_id, _ in sorted(edges.get(current_id, {}).items(), key=lambda x: x[0], reverse=True):
            if neighbor_id not in path:  # avoid cycles by checking path
                stack.append((neighbor_id, path + [neighbor_id]))

    return None, nodes_explored, None, None

def cus2_search(nodes, edges, start_id, goals, verbose=False):
    """
    Weighted A* Search (w=2.0): An informed search that uses f = g + w*h where w > 1.
    This emphasizes the heuristic more than standard A*, potentially finding solutions
    faster but not guaranteeing optimal paths. Useful when speed is more important than optimality.
    Returns (path, nodes_explored, cost, reached_goal) or (None, nodes_explored, None, None).
    """
    WEIGHT = 10.0  # Weight factor for the heuristic (w > 1 for faster, suboptimal search)
    
    # reset node fields
    for n in nodes.values():
        n.g = float('inf')
        n.h = 0
        n.f = float('inf')
        n.parent = None

    start = nodes[start_id]
    start.g = 0
    start.h = min(heuristic(start, nodes[g]) for g in goals)
    start.f = start.g + WEIGHT * start.h  # Weighted f-value

    counter = 0
    open_set = [(start.f, counter, start_id)]
    counter += 1
    closed = set()
    nodes_explored = 0

    if verbose:
        print(f"\nCus2 (Weighted A* with w={WEIGHT}) Trace:")
        print(f"Start node: {start_id} g={start.g:.2f} h={start.h:.2f} f={start.f:.2f} (f = g + {WEIGHT}*h)")

    while open_set:
        _, _, current_id = heapq.heappop(open_set)
        current = nodes[current_id]

        if current_id in closed:
            continue

        nodes_explored += 1

        if verbose:
            print(f"\nExpand node {current_id}: g={current.g:.2f} h={current.h:.2f} f={current.f:.2f} (expanded count={nodes_explored})")

        if current_id in goals:
            # reconstruct path
            path = []
            walker = current
            while walker:
                path.append(walker.id)
                walker = walker.parent
            return path[::-1], nodes_explored, current.g, current_id

        closed.add(current_id)

        for neighbor_id, weight in sorted(edges.get(current_id, {}).items(), key=lambda x: x[0]):
            if neighbor_id in closed:
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
                neighbor.h = min(heuristic(neighbor, nodes[g]) for g in goals)
                neighbor.f = neighbor.g + WEIGHT * neighbor.h  # Weighted f-value
                heapq.heappush(open_set, (neighbor.f, counter, neighbor_id))
                counter += 1
                if verbose:
                    print(f"    -> updated: g={neighbor.g:.2f} h={neighbor.h:.2f} f={neighbor.f:.2f} (f = g + {WEIGHT}*h) pushed to open set")
            else:
                if verbose:
                    print("    -> not improved, not pushed")

        if verbose:
            open_snapshot = ", ".join(f"{nid}(f={f:.2f})" for f,_,nid in open_set)
            print(f"  Open set: [{open_snapshot}]")
            print(f"  Closed set: {sorted(list(closed))}")

    return None, nodes_explored, None, None

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
    print("5. Custom Uninformed (Cus1)")
    print("6. Weighted A* Search w=2.0 (Cus2)")

    # Get search method choice
    method_map = {
        '1': 'BFS', '2': 'DFS', '3': 'GBFS', '4': 'AS', '5': 'CUS1', '6': 'CUS2',
        'BFS': 'BFS', 'DFS': 'DFS', 'GBFS': 'GBFS', 'AS': 'AS', 'CUS1': 'CUS1', 'CUS2': 'CUS2'
    }

    choice = method_arg
    while choice not in method_map:
        choice = input("\nEnter your choice (1-6): ").strip().upper()
        if choice not in method_map:
            print("Invalid choice. Please try again.")

    # Parse input file and run search
    nodes, edges, origin, destinations = parse_input_file(filename)
    
    # Run the selected search method
    method = method_map[choice]
    path, nodes_explored, method_name, found_cost, reached_goal = run_search(method, nodes, edges, origin, destinations, verbose=True)

    # Print results
    if path:
        # prefer returned reached_goal if available, otherwise infer from path
        goal_to_show = reached_goal if reached_goal is not None else next((g for g in destinations if g in path), None)
        print(f"\nSearch Method: {method_name}")
        print(f"Number of nodes explored: {nodes_explored}")
        path_with_coords = " -> ".join(f"{nid}({nodes[nid].x},{nodes[nid].y})" for nid in path)
        print(f"Path found (id(x,y)): {path_with_coords}")
        if goal_to_show is not None:
            print(f"Destination reached: {goal_to_show} ({nodes[goal_to_show].x},{nodes[goal_to_show].y})")
        if found_cost is not None:
            print(f"Total Cost: {found_cost}")
    else:
        print("\nNo path found!")

if __name__ == "__main__":
    main()