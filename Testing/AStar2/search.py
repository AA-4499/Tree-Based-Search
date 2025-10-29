from flask import Flask, render_template, jsonify, request
import math
import heapq
from collections import deque
import sys
import re

app = Flask(__name__)
DEFAULT_GRAPH_FILE = 'PathFinder-test.txt'

class Node:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.parent = None

def parse_graph_data(data):
    """Parse graph data from the frontend"""
    nodes = {}
    edges = {}
    
    for node_data in data['nodes']:
        node_id = node_data['id']
        nodes[node_id] = Node(node_id, node_data['x'], node_data['y'])
    
    for edge_data in data['edges']:
        from_id = edge_data['from']
        to_id = edge_data['to']
        weight = edge_data['weight']
        
        if from_id not in edges:
            edges[from_id] = {}
        edges[from_id][to_id] = weight
    
    return nodes, edges

def parse_text_file(filepath):
    """Parses graph data from the project's specific text file format for CLI use."""
    nodes = {}
    edges = {}
    start_id = None
    goals = []
    
    # Use 'r' for read mode
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file '{filepath}' not found.")
    
    # 1. Parse Nodes
    nodes_match = re.search(r'Nodes:\s*([\s\S]*?)Edges:', content, re.IGNORECASE)
    if nodes_match:
        nodes_content = nodes_match.group(1).strip()
        node_lines = [line.strip() for line in nodes_content.split('\n') if line.strip()]
        for line in node_lines:
            # e.g., 1: (4,1)
            try:
                node_id_str, coords_str = line.split(':', 1)
                node_id = int(node_id_str.strip())
                # Finds coordinates (x, y)
                x, y = map(int, re.findall(r'\((\d+),(\d+)\)', coords_str)[0])
                nodes[node_id] = Node(node_id, x, y)
            except (ValueError, IndexError):
                # Ignore lines that don't match the expected format
                continue
            
    # 2. Parse Edges
    edges_match = re.search(r'Edges:\s*([\s\S]*?)Origin:', content, re.IGNORECASE)
    if edges_match:
        edges_content = edges_match.group(1).strip()
        edge_lines = [line.strip() for line in edges_content.split('\n') if line.strip()]
        for line in edge_lines:
            # e.g., (2,1): 4
            match = re.search(r'\((\d+),(\d+)\):\s*(\d+)', line)
            if match:
                from_id, to_id, weight = map(int, match.groups())
                if from_id not in edges:
                    edges[from_id] = {}
                edges[from_id][to_id] = weight
                
    # 3. Parse Origin
    origin_match = re.search(r'Origin:\s*(\d+)', content, re.IGNORECASE)
    if origin_match:
        start_id = int(origin_match.group(1))
        
    # 4. Parse Destinations
    # Match everything after 'Destinations:' until the end of the file.
    dest_match = re.search(r'Destinations:\s*([\s\S]*)', content, re.IGNORECASE)
    if dest_match:
        goals_content = dest_match.group(1).strip()
        # Robustly extract all digits (node IDs) separated by non-digit characters (like ';', newline, or source tags)
        goals = [int(g.strip()) for g in re.split(r'\D+', goals_content) if g.strip().isdigit()]

    if not nodes or not edges or start_id is None or not goals:
        # Check if the missing component is the file itself being almost empty
        if not content.strip():
             raise ValueError("Input file is empty.")
        raise ValueError("Could not parse all required graph components (Nodes, Edges, Origin, Destinations) or some components were empty/missing.")

    return nodes, edges, start_id, goals

def graph_to_frontend(nodes, edges):
    """Convert parsed nodes/edges into the structure expected by the frontend."""
    node_list = [
        {'id': node.id, 'x': node.x, 'y': node.y}
        for node in nodes.values()
    ]
    node_list.sort(key=lambda item: item['id'])

    edge_list = []
    for from_id in sorted(edges.keys()):
        for to_id, weight in sorted(edges[from_id].items(), key=lambda item: item[0]):
            edge_list.append({'from': from_id, 'to': to_id, 'weight': weight})

    return {'nodes': node_list, 'edges': edge_list}

def load_default_graph():
    """Load the default graph data for the web UI from the configured text file."""
    graph_payload = {'nodes': [], 'edges': []}
    start_id = None
    goals = []

    try:
        nodes, edges, start_id, goals = parse_text_file(DEFAULT_GRAPH_FILE)
        graph_payload = graph_to_frontend(nodes, edges)
    except (FileNotFoundError, ValueError) as exc:
        app.logger.warning("Failed to load default graph '%s': %s", DEFAULT_GRAPH_FILE, exc)

    if start_id is None and graph_payload['nodes']:
        start_id = graph_payload['nodes'][0]['id']
    if goals is None:
        goals = []

    return graph_payload, start_id, goals

def get_algorithm_name(algorithm):
    """Get the full name of the algorithm for display"""
    names = {
        'BFS': 'Breadth-First Search',
        'DFS': 'Depth-First Search',
        'GBFS': 'Greedy Best-First Search',
        'AS': 'A* Search',
        'CUS1': 'Iterative Deepening Search (CUS1-Uninformed)',
        'CUS2': 'Weighted A* Search (CUS2-Informed)',
    }
    return names.get(algorithm, algorithm)

def run_cli(file_path, algorithm):
    """Loads data, executes the search algorithm, and prints the result to CLI."""
    try:
        nodes, edges, start_id, goals = parse_text_file(file_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error parsing file '{file_path}': {e}")
        return

    # Dictionary mapping algorithm names to their functions
    search_algorithms = {
        'BFS': bfs_search_steps,
        'DFS': dfs_search_steps,
        'GBFS': gbfs_search_steps,
        'AS': astar_search_steps,
        'CUS1': cus1_search_steps,
        'CUS2': cus2_search_steps,
    }

    if algorithm not in search_algorithms:
        print(f"Error: Unknown algorithm '{algorithm}'. Available: {', '.join(search_algorithms.keys())}")
        return

    # Execute the search
    search_func = search_algorithms[algorithm]
    result = search_func(nodes, edges, start_id, goals)

    # Print header
    print(f"{file_path} {algorithm}")
    print("=" * 53)
    
    if result['success']:
        goal_node = result['path'][-1]
        print(f"Goal Reached: Node {goal_node}")
        
        # Print nodes created if available
        if 'nodes_created' in result:
            print(f"Total Nodes Created: {result['nodes_created']}")
        
        # Print search method name
        print(f"Search Method: {get_algorithm_name(algorithm)}")
        
        # Build and print path
        path_str = "->".join(str(x) for x in result['path'])
        print(f"Shortest Path: {path_str}")
        
        # Print depth level if available (mainly for CUS1)
        if 'depth_reached' in result:
            print(f"Depth Level Reached: {result['depth_reached']}")
    else:
        print("Goal not reached.")
        
        # Print nodes created if available
        if 'nodes_created' in result:
            print(f"Total Nodes Created: {result['nodes_created']}")
        
        # Print search method name
        print(f"Search Method: {get_algorithm_name(algorithm)}")
        print("Path: None")
    
    print("=" * 53)


def heuristic(node, goal_node):
    return math.sqrt((node.x - goal_node.x)**2 + (node.y - goal_node.y)**2)

def bfs_search_steps(nodes, edges, start_id, goals):
    queue = deque([(start_id, [start_id], 0)])
    visited = set([start_id])
    steps = []
    nodes_created = 1  # Count the start node
    
    steps.append({'type': 'start', 'node': start_id, 'message': f'Starting BFS from node {start_id}'})
    
    while queue:
        current_id, path, cost = queue.popleft()
        steps.append({'type': 'expand', 'node': current_id, 'path': path, 'cost': cost, 'message': f'Expanding node {current_id}'})
        
        if current_id in goals:
            steps.append({'type': 'goal', 'node': current_id, 'path': path, 'cost': cost, 'message': f'Goal {current_id} reached!'})
            return {'success': True, 'path': path, 'steps': steps, 'cost': cost, 'nodes_created': nodes_created}
        
        if current_id in edges:
            for neighbor_id, weight in sorted(edges[current_id].items(), key=lambda x: x[0]):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    nodes_created += 1
                    new_cost = cost + weight
                    queue.append((neighbor_id, path + [neighbor_id], new_cost))
                    steps.append({'type': 'discover', 'node': neighbor_id, 'parent': current_id, 'message': f'Discovered node {neighbor_id}'})
    
    return {'success': False, 'path': None, 'steps': steps, 'nodes_created': nodes_created}

def dfs_search_steps(nodes, edges, start_id, goals):
    steps = []
    result_path = None
    result_cost = None
    nodes_created = 0
    
    steps.append({'type': 'start', 'node': start_id, 'message': f'Starting DFS from node {start_id}'})
    
    def _dfs(node_id, path, cost, path_set):
        nonlocal result_path, result_cost, nodes_created
        nodes_created += 1
        steps.append({'type': 'expand', 'node': node_id, 'path': path, 'cost': cost, 'message': f'Exploring node {node_id} (cost: {cost:.1f})'})
        
        if node_id in goals:
            result_path = path[:]
            result_cost = cost
            steps.append({'type': 'goal', 'node': node_id, 'path': path, 'cost': cost, 'message': f'Goal {node_id} found with cost {cost:.1f}'})
            return True
        
        for nbr_id, weight in sorted(edges.get(node_id, {}).items(), key=lambda x: x[0]):
            if nbr_id not in path_set:
                path_set.add(nbr_id)
                steps.append({'type': 'discover', 'node': nbr_id, 'parent': node_id, 'message': f'Discovered node {nbr_id}'})
                if _dfs(nbr_id, path + [nbr_id], cost + weight, path_set):
                    return True
                path_set.remove(nbr_id)
        
        return False
    
    _dfs(start_id, [start_id], 0.0, {start_id})
    return {'success': result_path is not None, 'path': result_path, 'steps': steps, 'cost': result_cost, 'nodes_created': nodes_created}

def gbfs_search_steps(nodes, edges, start_id, goals):
    def best_heuristic(node_id):
        return min(heuristic(nodes[node_id], nodes[goal]) for goal in goals)
    
    open_set = [(best_heuristic(start_id), 0, start_id)]
    parent = {start_id: None}
    cost_so_far = {start_id: 0}
    explored = set()
    steps = []
    counter = 1
    nodes_created = 1
    
    steps.append({'type': 'start', 'node': start_id, 'h': best_heuristic(start_id), 'message': f'Starting GBFS from node {start_id}'})
    
    while open_set:
        current_h, _, current_id = heapq.heappop(open_set)
        if current_id in explored:
            continue
        
        explored.add(current_id)
        path = []
        temp = current_id
        while temp is not None:
            path.append(temp)
            temp = parent[temp]
        path = path[::-1]
        
        steps.append({'type': 'expand', 'node': current_id, 'path': path, 'h': current_h, 'cost': cost_so_far[current_id], 'message': f'Expanding node {current_id} (h={current_h:.1f})'})
        
        if current_id in goals:
            steps.append({'type': 'goal', 'node': current_id, 'path': path, 'cost': cost_so_far[current_id], 'message': f'Goal {current_id} reached!'})
            return {'success': True, 'path': path, 'steps': steps, 'cost': cost_so_far[current_id], 'nodes_created': nodes_created}
        
        for neighbor_id, weight in sorted(edges.get(current_id, {}).items(), key=lambda x: x[0]):
            if neighbor_id not in explored:
                if neighbor_id not in parent:
                    parent[neighbor_id] = current_id
                    cost_so_far[neighbor_id] = cost_so_far[current_id] + weight
                    nodes_created += 1
                neighbor_h = best_heuristic(neighbor_id)
                heapq.heappush(open_set, (neighbor_h, counter, neighbor_id))
                counter += 1
                steps.append({'type': 'discover', 'node': neighbor_id, 'h': neighbor_h, 'message': f'Discovered node {neighbor_id} (h={neighbor_h:.1f})'})
    
    return {'success': False, 'path': None, 'steps': steps, 'nodes_created': nodes_created}

def astar_search_steps(nodes, edges, start_id, goals):
    for n in nodes.values():
        n.g = float('inf')
        n.parent = None
    
    start = nodes[start_id]
    start.g = 0
    start.h = min(heuristic(start, nodes[goal]) for goal in goals)
    start.f = start.g + start.h
    
    open_set = [(start.f, 0, start_id)]
    closed = set()
    steps = []
    counter = 1
    nodes_created = 1
    
    steps.append({'type': 'start', 'node': start_id, 'g': 0, 'h': start.h, 'f': start.f, 'message': f'Starting A* from node {start_id}'})
    
    while open_set:
        _, _, current_id = heapq.heappop(open_set)

        if current_id in closed:
            continue

        current = nodes[current_id]
        
        path = []
        temp = current
        while temp:
            path.append(temp.id)
            temp = temp.parent
        path = path[::-1]
        
        steps.append({'type': 'expand', 'node': current_id, 'path': path, 'g': current.g, 'h': current.h, 'f': current.f, 'cost': current.g, 'message': f'Expanding node {current_id} (f={current.f:.1f})'})
        
        if current_id in goals:
            steps.append({'type': 'goal', 'node': current_id, 'path': path, 'cost': current.g, 'message': f'Goal {current_id} reached with cost {current.g:.1f}!'})
            return {'success': True, 'path': path, 'steps': steps, 'cost': current.g, 'nodes_created': nodes_created}
        
        closed.add(current_id)
        
        for neighbor_id, weight in edges.get(current_id, {}).items():
            if neighbor_id in closed:
                continue
            
            neighbor = nodes[neighbor_id]
            tentative_g = current.g + weight
            
            if tentative_g < neighbor.g:
                if neighbor.g == float('inf'):  # First time visiting
                    nodes_created += 1
                neighbor.parent = current
                neighbor.g = tentative_g
                neighbor.h = min(heuristic(neighbor, nodes[goal]) for goal in goals)
                neighbor.f = neighbor.g + neighbor.h
                heapq.heappush(open_set, (neighbor.f, counter, neighbor_id))
                counter += 1
                steps.append({'type': 'discover', 'node': neighbor_id, 'g': neighbor.g, 'h': neighbor.h, 'f': neighbor.f, 'message': f'Updated node {neighbor_id} (f={neighbor.f:.1f})'})
    
    return {'success': False, 'path': None, 'steps': steps, 'nodes_created': nodes_created}

def cus1_search_steps(nodes, edges, start_id, goals):
    """CUS1: Iterative Deepening Search (IDS) - Uninformed tree-based search"""
    steps = []
    destinations_set = set(goals)
    total_nodes_created = 0
    
    # Determine max depth limit
    all_node_ids = list(nodes.keys())
    max_depth_limit = max(all_node_ids + [start_id] + goals) + 20
    
    steps.append({'type': 'start', 'node': start_id, 'message': f'Starting CUS1 (Iterative Deepening Search) from node {start_id}'})
    
    def depth_limited_search(node_id, current_path, limit, depth):
        """Performs depth-limited search (tree-based, no visited set)"""
        nonlocal total_nodes_created
        nodes_created = 1
        total_nodes_created += 1
        
        # Add step for node expansion
        if depth == 0:
            steps.append({'type': 'expand', 'node': node_id, 'path': current_path, 'depth': depth, 
                         'message': f'Starting depth-limited search at depth {limit} from node {node_id}'})
        else:
            steps.append({'type': 'expand', 'node': node_id, 'path': current_path, 'depth': depth,
                         'message': f'Exploring node {node_id} at depth {depth}/{limit}'})
        
        # Goal test
        if node_id in destinations_set:
            return True, current_path, nodes_created, False
        
        # Depth limit reached
        if limit == 0:
            return False, None, nodes_created, True
        
        # Expand children in ascending order (tree search - allows revisiting)
        children = []
        if node_id in edges:
            children = sorted(edges[node_id].keys())
        
        for child_id in children:
            steps.append({'type': 'discover', 'node': child_id, 'parent': node_id, 
                         'message': f'Discovered node {child_id} from {node_id}'})
            
            found, path, child_created, cutoff = depth_limited_search(
                child_id, current_path + [child_id], limit - 1, depth + 1
            )
            nodes_created += child_created
            
            if found:
                return True, path, nodes_created, False
        
        # Exhausted all children without finding goal
        return False, None, nodes_created, False
    
    # Iterative deepening loop
    for depth_limit in range(0, max_depth_limit + 1):
        steps.append({'type': 'iteration', 'depth': depth_limit, 
                     'message': f'Starting iteration with depth limit {depth_limit}'})
        
        found, path, nodes_created, cutoff = depth_limited_search(start_id, [start_id], depth_limit, 0)
        
        if found:
            # Calculate cost
            cost = 0
            for i in range(len(path) - 1):
                if path[i] in edges and path[i+1] in edges[path[i]]:
                    cost += edges[path[i]][path[i+1]]
            
            steps.append({'type': 'goal', 'node': path[-1], 'path': path, 'cost': cost, 
                         'depth': depth_limit, 'nodes_created': total_nodes_created,
                         'message': f'Goal {path[-1]} reached at depth {len(path)-1}!'})
            
            return {
                'success': True, 
                'path': path, 
                'steps': steps, 
                'cost': cost,
                'nodes_created': total_nodes_created,
                'depth_reached': len(path) - 1
            }
    
    steps.append({'type': 'failure', 'message': 'Goal not reached within depth limit'})
    return {
        'success': False, 
        'path': None, 
        'steps': steps,
        'nodes_created': total_nodes_created
    }

def cus2_search_steps(nodes, edges, start_id, goals):
    """CUS2: Weighted A* Search (w=10.0) - emphasizes heuristic for faster search"""
    WEIGHT = 10.0
    
    for n in nodes.values():
        n.g = float('inf')
        n.parent = None
    
    start = nodes[start_id]
    start.g = 0
    start.h = min(heuristic(start, nodes[goal]) for goal in goals)
    start.f = start.g + WEIGHT * start.h
    
    open_set = [(start.f, 0, start_id)]
    closed = set()
    steps = []
    counter = 1
    nodes_created = 1
    
    steps.append({'type': 'start', 'node': start_id, 'g': 0, 'h': start.h, 'f': start.f, 'message': f'Starting CUS2 (Weighted A* w={WEIGHT}) from node {start_id}'})
    
    while open_set:
        _, _, current_id = heapq.heappop(open_set)
        current = nodes[current_id]
        
        if current_id in closed:
            continue
        
        path = []
        temp = current
        while temp:
            path.append(temp.id)
            temp = temp.parent
        path = path[::-1]
        
        steps.append({'type': 'expand', 'node': current_id, 'path': path, 'g': current.g, 'h': current.h, 'f': current.f, 'cost': current.g, 'message': f'Expanding node {current_id} (f={current.f:.1f})'})
        
        if current_id in goals:
            steps.append({'type': 'goal', 'node': current_id, 'path': path, 'cost': current.g, 'message': f'Goal {current_id} reached with cost {current.g:.1f}!'})
            return {'success': True, 'path': path, 'steps': steps, 'cost': current.g, 'nodes_created': nodes_created}
        
        closed.add(current_id)
        
        for neighbor_id, weight in sorted(edges.get(current_id, {}).items(), key=lambda x: x[0]):
            if neighbor_id in closed:
                continue
            
            neighbor = nodes[neighbor_id]
            tentative_g = current.g + weight
            
            if tentative_g < neighbor.g:
                if neighbor.g == float('inf'):  # First time visiting
                    nodes_created += 1
                neighbor.parent = current
                neighbor.g = tentative_g
                neighbor.h = min(heuristic(neighbor, nodes[goal]) for goal in goals)
                neighbor.f = neighbor.g + WEIGHT * neighbor.h
                heapq.heappush(open_set, (neighbor.f, counter, neighbor_id))
                counter += 1
                steps.append({'type': 'discover', 'node': neighbor_id, 'g': neighbor.g, 'h': neighbor.h, 'f': neighbor.f, 'message': f'Updated node {neighbor_id} (f={neighbor.f:.1f})'})
    
    return {'success': False, 'path': None, 'steps': steps, 'nodes_created': nodes_created}

@app.route('/')
def index():
    graph_data, start_id, goals = load_default_graph()
    return render_template(
        'index.html',
        graph_data=graph_data,
        start_id=start_id,
        goals=goals
    )

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    algorithm = data['algorithm']
    graph_data = data['graph']
    start_id = data['start']
    goals = data['goals']
    
    nodes, edges = parse_graph_data(graph_data)
    
    if algorithm == 'BFS':
        result = bfs_search_steps(nodes, edges, start_id, goals)
    elif algorithm == 'DFS':
        result = dfs_search_steps(nodes, edges, start_id, goals)
    elif algorithm == 'GBFS':
        result = gbfs_search_steps(nodes, edges, start_id, goals)
    elif algorithm == 'AS':
        result = astar_search_steps(nodes, edges, start_id, goals)
    elif algorithm == 'CUS1':
        result = cus1_search_steps(nodes, edges, start_id, goals)
    elif algorithm == 'CUS2':
        result = cus2_search_steps(nodes, edges, start_id, goals)
    else:
        return jsonify({'error': 'Unknown algorithm'}), 400
    
    return jsonify(result)

@app.route('/reload', methods=['POST'])
def reload_graph():
    """Reload the graph data from the file"""
    try:
        graph_data, start_id, goals = load_default_graph()
        return jsonify({
            'success': True,
            'graph_data': graph_data,
            'start_id': start_id,
            'goals': goals
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
if __name__ == '__main__':
    # Logic to switch between CLI and Web GUI mode
    if len(sys.argv) > 1:
        # CLI mode: Arguments are present.
        # Format: python search.py <file> [<method>]
        file_path = sys.argv[1]
        
        # Default to 'AS' (A*) if only the file is provided (Option 1 in .bat)
        # Use the provided method if available (Option 3 in .bat)
        algorithm = sys.argv[2].upper().replace('*', 'S') if len(sys.argv) == 3 else 'BFS'
        
        run_cli(file_path, algorithm)
    else:
        # Web GUI mode: No command-line arguments, run the Flask app.
        app.run(debug=True, port=5000)