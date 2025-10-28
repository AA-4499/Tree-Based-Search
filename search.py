from flask import Flask, render_template, jsonify, request
import math
import heapq
from collections import deque

app = Flask(__name__)

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

def heuristic(node, goal_node):
    return math.sqrt((node.x - goal_node.x)**2 + (node.y - goal_node.y)**2)

def bfs_search_steps(nodes, edges, start_id, goals):
    queue = deque([(start_id, [start_id], 0)])
    visited = set([start_id])
    steps = []
    
    steps.append({'type': 'start', 'node': start_id, 'message': f'Starting BFS from node {start_id}'})
    
    while queue:
        current_id, path, cost = queue.popleft()
        steps.append({'type': 'expand', 'node': current_id, 'path': path, 'cost': cost, 'message': f'Expanding node {current_id}'})
        
        if current_id in goals:
            steps.append({'type': 'goal', 'node': current_id, 'path': path, 'cost': cost, 'message': f'Goal {current_id} reached!'})
            return {'success': True, 'path': path, 'steps': steps, 'cost': cost}
        
        if current_id in edges:
            for neighbor_id, weight in sorted(edges[current_id].items(), key=lambda x: x[0]):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    new_cost = cost + weight
                    queue.append((neighbor_id, path + [neighbor_id], new_cost))
                    steps.append({'type': 'discover', 'node': neighbor_id, 'parent': current_id, 'message': f'Discovered node {neighbor_id}'})
    
    return {'success': False, 'path': None, 'steps': steps}

def dfs_search_steps(nodes, edges, start_id, goals):
    best_cost = float('inf')
    best_path = None
    steps = []
    
    steps.append({'type': 'start', 'node': start_id, 'message': f'Starting DFS from node {start_id}'})
    
    def _dfs(node_id, path, cost, visited):
        nonlocal best_cost, best_path
        steps.append({'type': 'expand', 'node': node_id, 'path': path, 'cost': cost, 'message': f'Exploring node {node_id} (cost: {cost:.1f})'})
        
        if node_id in goals:
            if cost < best_cost:
                best_cost = cost
                best_path = path[:]
                steps.append({'type': 'goal', 'node': node_id, 'path': path, 'cost': cost, 'message': f'Goal {node_id} found with cost {cost:.1f}'})
            return
        
        if cost >= best_cost:
            return
        
        for nbr_id, weight in sorted(edges.get(node_id, {}).items(), key=lambda x: x[0]):
            if nbr_id not in visited:
                visited.add(nbr_id)
                _dfs(nbr_id, path + [nbr_id], cost + weight, visited)
                visited.remove(nbr_id)
    
    _dfs(start_id, [start_id], 0.0, {start_id})
    return {'success': best_path is not None, 'path': best_path, 'steps': steps, 'cost': best_cost if best_path else None}

def gbfs_search_steps(nodes, edges, start_id, goals):
    def best_heuristic(node_id):
        return min(heuristic(nodes[node_id], nodes[goal]) for goal in goals)
    
    open_set = [(best_heuristic(start_id), 0, start_id)]
    parent = {start_id: None}
    cost_so_far = {start_id: 0}
    explored = set()
    steps = []
    counter = 1
    
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
            return {'success': True, 'path': path, 'steps': steps, 'cost': cost_so_far[current_id]}
        
        for neighbor_id, weight in sorted(edges.get(current_id, {}).items(), key=lambda x: x[0]):
            if neighbor_id not in explored:
                if neighbor_id not in parent:
                    parent[neighbor_id] = current_id
                    cost_so_far[neighbor_id] = cost_so_far[current_id] + weight
                neighbor_h = best_heuristic(neighbor_id)
                heapq.heappush(open_set, (neighbor_h, counter, neighbor_id))
                counter += 1
                steps.append({'type': 'discover', 'node': neighbor_id, 'h': neighbor_h, 'message': f'Discovered node {neighbor_id} (h={neighbor_h:.1f})'})
    
    return {'success': False, 'path': None, 'steps': steps}

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
    
    steps.append({'type': 'start', 'node': start_id, 'g': 0, 'h': start.h, 'f': start.f, 'message': f'Starting A* from node {start_id}'})
    
    while open_set:
        _, _, current_id = heapq.heappop(open_set)
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
            return {'success': True, 'path': path, 'steps': steps, 'cost': current.g}
        
        closed.add(current_id)
        
        for neighbor_id, weight in edges.get(current_id, {}).items():
            if neighbor_id in closed:
                continue
            
            neighbor = nodes[neighbor_id]
            tentative_g = current.g + weight
            
            if tentative_g < neighbor.g:
                neighbor.parent = current
                neighbor.g = tentative_g
                neighbor.h = min(heuristic(neighbor, nodes[goal]) for goal in goals)
                neighbor.f = neighbor.g + neighbor.h
                heapq.heappush(open_set, (neighbor.f, counter, neighbor_id))
                counter += 1
                steps.append({'type': 'discover', 'node': neighbor_id, 'g': neighbor.g, 'h': neighbor.h, 'f': neighbor.f, 'message': f'Updated node {neighbor_id} (f={neighbor.f:.1f})'})
    
    return {'success': False, 'path': None, 'steps': steps}

def cus1_search_steps(nodes, edges, start_id, goals):
    """CUS1: Uninformed search - returns first path found (DFS-style)"""
    stack = [(start_id, [start_id], 0)]
    steps = []
    
    steps.append({'type': 'start', 'node': start_id, 'message': f'Starting CUS1 (Uninformed DFS) from node {start_id}'})
    
    while stack:
        current_id, path, cost = stack.pop()
        steps.append({'type': 'expand', 'node': current_id, 'path': path, 'cost': cost, 'message': f'Exploring node {current_id}'})
        
        if current_id in goals:
            steps.append({'type': 'goal', 'node': current_id, 'path': path, 'cost': cost, 'message': f'Goal {current_id} reached!'})
            return {'success': True, 'path': path, 'steps': steps, 'cost': cost}
        
        for neighbor_id, weight in sorted(edges.get(current_id, {}).items(), key=lambda x: x[0], reverse=True):
            if neighbor_id not in path:
                new_cost = cost + weight
                stack.append((neighbor_id, path + [neighbor_id], new_cost))
                steps.append({'type': 'discover', 'node': neighbor_id, 'message': f'Discovered node {neighbor_id}'})
    
    return {'success': False, 'path': None, 'steps': steps}

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
            return {'success': True, 'path': path, 'steps': steps, 'cost': current.g}
        
        closed.add(current_id)
        
        for neighbor_id, weight in sorted(edges.get(current_id, {}).items(), key=lambda x: x[0]):
            if neighbor_id in closed:
                continue
            
            neighbor = nodes[neighbor_id]
            tentative_g = current.g + weight
            
            if tentative_g < neighbor.g:
                neighbor.parent = current
                neighbor.g = tentative_g
                neighbor.h = min(heuristic(neighbor, nodes[goal]) for goal in goals)
                neighbor.f = neighbor.g + WEIGHT * neighbor.h
                heapq.heappush(open_set, (neighbor.f, counter, neighbor_id))
                counter += 1
                steps.append({'type': 'discover', 'node': neighbor_id, 'g': neighbor.g, 'h': neighbor.h, 'f': neighbor.f, 'message': f'Updated node {neighbor_id} (f={neighbor.f:.1f})'})
    
    return {'success': False, 'path': None, 'steps': steps}

@app.route('/')
def index():
    return render_template('index.html')

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

if __name__ == '__main__':
    app.run(debug=True, port=5000)