# =============================================================================
# STANFORD CS221 ASSIGNMENTS - COMPREHENSIVE DOCUMENTATION
# =============================================================================
# Team: Group C
# Course: Artificial Intelligence
# Instructor: Dr. Nakibule Mary
# Date: November 2025
# =============================================================================

## PROJECT OVERVIEW
This notebook contains implementations for all 7 Stanford CS221 programming
assignments, integrated into a unified interactive GUI launcher.

## ASSIGNMENT BREAKDOWN

### Assignment 1: Foundations
**Purpose**: Implement fundamental algorithms and data structures
**Key Functions**:
- find_alphabetically_first_word(): Lexicographic string comparison
- euclidean_distance(): 2D distance calculation using Pythagorean theorem
- mutate_sentences(): DFS-based sentence generation
- sparse_vector_dot_product(): Efficient sparse vector operations
- increment_sparse_vector(): In-place vector updates for ML algorithms
- find_nonsingleton_words(): Word frequency analysis

**Algorithm Complexity**:
- Most functions: O(n) where n is input size
- mutate_sentences(): O(2^n) worst case due to exponential combinations

### Assignment 2: Sentiment Analysis
**Purpose**: Build a text classifier to predict sentiment (positive/negative)
**Approach**: Feature extraction + linear classification
**Key Components**:
- extractWordFeatures(): Bag-of-words representation
- learnPredictor(): Stochastic gradient descent (SGD) training
- Loss function: Hinge loss for robustness

**Training Process**:
1. Extract word features from each review
2. Initialize weight vector to zeros
3. For each training example:
   - Compute prediction score
   - If misclassified, update weights
4. Iterate for multiple epochs

### Assignment 3: Pacman
**Purpose**: Implement AI agents for Pacman game
**Search Algorithms**:
- Depth-First Search (DFS): Memory efficient, not optimal
- Breadth-First Search (BFS): Optimal for unweighted graphs
- Uniform Cost Search (UCS): Optimal for weighted graphs
- A* Search: Uses heuristics for faster pathfinding

**Heuristics Implemented**:
- Manhattan distance: |x1-x2| + |y1-y2|
- Euclidean distance: sqrt((x1-x2)² + (y1-y2)²)

### Assignment 4: Mountaincar (Reinforcement Learning)
**Purpose**: Train RL agents to drive a car up a steep mountain
**Algorithms Implemented**:

1. **Model-Based Monte Carlo**:
   - Learns transition probabilities from samples
   - Uses value iteration for policy extraction
   - Time Complexity: O(|S|² |A|) per iteration

2. **Tabular Q-Learning**:
   - Model-free temporal difference learning
   - Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
   - Exploration: ε-greedy policy

3. **Function Approximation Q-Learning**:
   - Uses feature representation for generalization
   - Linear function approximation: Q(s,a) = w·φ(s,a)
   - Weight updates via gradient descent

4. **Constrained Q-Learning**:
   - Adds action constraints (velocity limits)
   - Filters invalid actions before selection

**Visualization**: Training plots show learning curves and performance metrics

### Assignment 5: Scheduling (Constraint Satisfaction)
**Purpose**: Solve course scheduling problems using CSP algorithms
**Problem Formulation**:
- Variables: Course assignments
- Domains: Possible quarters for each course
- Constraints: Prerequisites, unit limits, request conflicts

**Algorithms**:
1. **Backtracking Search**:
   - Systematic exploration of assignment space
   - Prunes invalid branches early

2. **Arc Consistency (AC-3)**:
   - Domain reduction before search
   - Removes values that can never be part of solution

**Optimizations**:
- Most Constrained Variable (MCV) heuristic
- Least Constraining Value (LCV) ordering
- Forward checking to detect failures early

### Assignment 6: Route Planning
**Purpose**: Find optimal paths on Stanford campus map
**Map Representation**:
- Nodes: Physical locations (buildings, intersections)
- Edges: Walkable paths with distance costs
- Tags: Location labels (landmarks, amenities)

**Search Algorithms**:
1. **Uniform Cost Search (UCS)**:
   - Dijkstra's algorithm
   - Guarantees optimal path
   - Time: O((V+E)log V) with priority queue

2. **A* Search**:
   - UCS + heuristic guidance
   - Heuristic: Straight-line distance to goal
   - Much faster than UCS in practice

**Waypoints Problem**:
- Must visit specific locations in any order
- State space: (current_location, remaining_waypoints)
- NP-hard problem - exponential in waypoints

**Visualization**: Interactive Plotly map showing:
- Full campus graph
- Calculated optimal path (blue line)
- Start point (red marker)
- End point (green marker)
- Waypoints and landmarks (purple markers)

### Assignment 7: Car Tracking (Probabilistic Reasoning)
**Purpose**: Track hidden car positions using noisy sensor readings
**Inference Methods**:

1. **Exact Inference**:
   - Maintains full probability distribution
   - Updates via Bayes' rule
   - Prediction: P(X_{t+1}) = Σ P(X_{t+1}|X_t)·P(X_t)
   - Update: P(X_t|e_t) ∝ P(e_t|X_t)·P(X_t)

2. **Particle Filter**:
   - Approximate inference using samples
   - Steps:
     a. Prediction: Move particles according to transition model
     b. Weighting: Compute likelihood of observation
     c. Resampling: Sample particles proportional to weights
   - Advantages: Handles continuous spaces, multimodal distributions

**Multiple Car Tracking**:
- Junction tree algorithm for exact inference
- Particle filter with joint state representation

## GUI IMPLEMENTATION DETAILS

### Architecture
The GUI uses Pygame for rendering and event handling. Each assignment has
its own menu with options for:
- Interactive testing of individual functions
- Running automated grader tests
- Visualizing results

### Menu Structure
```
Main Menu
├── Foundations (interactive function tests)
├── Sentiment (train/test classifier)
├── Pacman (search algorithms + game)
├── Mountaincar (RL training + visualization)
├── Scheduling (CSP solver)
├── Route Planning (pathfinding + map)
└── Car Tracking (particle filters + animation)
```

### Terminal Integration
- Graders and complex demos run in separate PowerShell windows
- Uses subprocess.Popen with CREATE_NEW_CONSOLE flag
- Keeps terminal open with "Press Enter to close" prompt

### Visualization Features
1. **Mountaincar**: Auto-opens 3 most recent training plot PNG files
2. **Route Planning**: Generates interactive HTML map, opens in browser
3. **Car Tracking**: Real-time animation of particle distributions

## TECHNICAL IMPLEMENTATION NOTES

### Python Version Compatibility
- Developed on Python 3.13
- Some assignments expect Python 3.12 (grader warnings expected)
- All core functionality works correctly

### Dependencies
- Standard library: collections, math, typing, sys, os, subprocess
- External: pygame (GUI), plotly (route visualization), numpy (RL)

### File Structure
```
AI_Assignments.ipynb          # Main notebook with all code
├── foundations/              # Assignment 1 files
│   ├── submission.py
│   └── grader.py
├── sentiment/                # Assignment 2 files
├── pacman/                   # Assignment 3 files
├── mountaincar/              # Assignment 4 files
├── scheduling/               # Assignment 5 files
├── route/                    # Assignment 6 files
└── car/                      # Assignment 7 files
```

### Universal Design
- No hardcoded paths - uses os.path.join()
- Works on any Windows system with required packages
- Suitable for submission to instructor

## ALGORITHM COMPLEXITY ANALYSIS

### Time Complexity Summary
| Assignment | Algorithm | Time Complexity | Space Complexity |
|-----------|-----------|----------------|------------------|
| Foundations | String ops | O(n) | O(n) |
| Sentiment | SGD training | O(T·N·F) | O(F) |
| Pacman | A* search | O(b^d) | O(b^d) |
| Mountaincar | Q-learning | O(episodes·steps) | O(|S||A|) |
| Scheduling | Backtracking | O(d^n) | O(n) |
| Route | A* | O((V+E)log V) | O(V) |
| Car | Particle filter | O(K) | O(K) |

Where:
- N: training examples, F: features, T: epochs
- b: branching factor, d: solution depth
- V: vertices, E: edges, K: particles

## TESTING AND VALIDATION

### Grader Integration
Each assignment includes automated graders that:
- Test correctness of implemented functions
- Verify output format and edge cases
- Provide detailed feedback on failures

### Manual Testing via GUI
Interactive menus allow testing individual functions with custom inputs,
providing immediate visual feedback for debugging.

## SUBMISSION CHECKLIST
✅ All 7 assignments implemented
✅ Single Python notebook
✅ Team members listed
✅ Assignments numbered 1-7
✅ Each assignment titled and described
✅ Comprehensive inline documentation
✅ Code is well-commented
✅ Functions have docstrings
✅ GUI provides professional interface
✅ Universal paths (no hardcoding)
✅ Ready for instructor evaluation

## CONCLUSION
This notebook demonstrates proficiency in:
- Classical AI algorithms (search, CSP)
- Machine learning (classification, RL)
- Probabilistic reasoning (Bayesian inference)
- Software engineering (modular design, GUI development)

The unified GUI launcher provides an innovative way to test and demonstrate
all assignments in one cohesive interface, going beyond the basic requirements.
