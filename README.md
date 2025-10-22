Tree-Based-Search: Route Finding Algorithms

This repository contains the implementation of various **tree-based search algorithms** for solving the **Route Finding Problem**, developed as **Part A of Assignment 2** for the course **Introduction to Artificial Intelligence**.

The project is written in **Python**.

---

🗺️ The Route Finding Problem

The core problem involves an agent tasked with finding **optimal paths** (i.e., those with the **lowest total cost**) from a single **Origin node ($O$)** to one or more **Destination nodes ($D$)** on a given 2D graph.

Problem Specification

The graph structure is loaded from a simple text file.

File Format Structure:

* The file starts with the keyword "Nodes:", followed by node specifications and their (x, y) coordinates, e.g., `1: (4,1)`.
* Next is "Edges:", followed by directed edges and their costs, e.g., `(2,1): 4`.
* Next is "Origin:", specifying the starting node, e.g., `2`.
* Finally, "Destinations:", listing the target nodes separated by semi-colons, e.g., `5; 4`. A solution is found upon reaching *any* of these destinations.

---

🔍 Implemented Search Algorithms

The program implements both uninformed and informed search strategies, as well as two custom methods.

Uninformed Methods:

* **DFS** (Depth-First Search): Select one option, try it, go back when there are no more options.
* **BFS** (Breadth-First Search): Expand all options one level at a time.

Informed Methods:

* **GBFS** (Greedy Best-First Search): Uses only the **heuristic cost** (cost to reach the goal from the current node) to evaluate the node.
* **A\* (A Star)**: Uses both the **path cost** (cost to reach the current node from the origin) and the **heuristic cost** to evaluate the node.

Custom Methods:

* **CUS1** (Custom Strategy 1): An **uninformed** method to find *a* path to reach the goal.
* **CUS2** (Custom Strategy 2): An **informed** method to find a **shortest path** (with least moves/minimum depth) to reach the goal.

---

🚀 Getting Started

Prerequisites

* Python 3.x

Execution

The program is run from the terminal using the following command format:

`python3 search.py <input-file.txt> <method>`

* `<input-file.txt>`: The path to the problem specification file (e.g., `PathFinder-test.txt`).
* `<method>`: The abbreviation of the desired search algorithm (`DFS`, `BFS`, `GBFS`, `A*`, `CUS1`, `CUS2`).

Example:

`Tree-Based-Search>python3 search.py PathFinder-test.txt A*`

A secondary way to run the code (if no arguments are supplied) is:

`python3 search.py`

---

👥 Group Members

* Andy Wei Jian CHU (104401283)
* Desmond Khai Tie OH (104389075)
* Nicholina Sheron Avanthi Pinto JAYANTHI LEKAMGE (104395470)
* Nur Najiehah BINTI MOHD SHYAFUDDIN LOH (104402312)