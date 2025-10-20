import heapq
import numpy as np
from collections import defaultdict


class Node:
    def __init__(self, position, g_cost=0, h_cost=0, parent=None):
        self.position = np.array(position)
        self.g_cost = g_cost  # Cost from start to this node
        self.h_cost = h_cost  # Heuristic cost from this node to goal
        self.f_cost = g_cost + h_cost  # Total cost
        self.parent = parent
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost


class AStar:
    name = "AStar"
    
    def __init__(self, environment, step_size=0.5, max_iter=100000, goal_tolerance=0.2, heuristic_weight=1.0, exploration_mode=False):
        self.env = environment
        self.max_iter = max_iter  # Maximum iterations to prevent infinite loops
        self.heuristic_weight = heuristic_weight  # Weight for heuristic (1.0 = standard A*)
        self.exploration_mode = exploration_mode  # If True, use original map for more permissive planning
        
        # Set step size and goal tolerance based on mode
        if exploration_mode:
            self.step_size = min(step_size, environment.resolution * 5)  # Slightly larger steps
            self.goal_tolerance = min(goal_tolerance, environment.resolution * 3)  # 3 pixels max for exploration
            self.grid_step_size = 2  # Use 2-pixel steps for smoother paths
            print(f"🎯 A* exploration mode: step={self.step_size:.3f}m, tolerance={self.goal_tolerance:.3f}m, grid_step=2")
        else:
            self.step_size = step_size
            self.goal_tolerance = goal_tolerance
            # Calculate grid step size based on world step size and resolution
            self.grid_step_size = max(1, int(self.step_size / environment.resolution))
        
        # Use environment's dilated map by default, original map for exploration
        if exploration_mode:
            self.occupancy_grid = environment.occupancy_grid  # Use original map for exploration
            print("🔍 A* initialized in exploration mode (using original map)")
        else:
            self.occupancy_grid = environment.occupancy_grid_dilated  # Use dilated map for safety
        
        self.origin = environment.origin
        self.resolution = environment.resolution
        self.goal = None
        
        # Calculate grid step size based on world step size and resolution (already set above in exploration mode)

    def set_params(self, step_size=0.5, max_iter=100000, goal_tolerance=0.5, heuristic_weight=1.0):
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_tolerance = goal_tolerance
        self.heuristic_weight = heuristic_weight
        # Update grid step size when parameters change
        self.grid_step_size = max(1, int(self.step_size / self.resolution))

    def reinitialize(self, environment=None, goal=None):
        """
        Reinitializes the A* controller, resetting its internal state and optionally updating the environment or goal.
        
        Args:
            environment (optional): New environment to use for the controller.
            goal (optional): New goal to set for the controller.
        """
        if environment is not None:
            self.env = environment
            # Use environment's pre-computed dilated map for consistency
            self.occupancy_grid = environment.occupancy_grid_dilated
            self.origin = environment.origin
            self.resolution = environment.resolution
            # Recalculate grid step size for new environment
            self.grid_step_size = max(1, int(self.step_size / self.resolution))

        if goal is not None:
            self.goal = goal
        else:
            self.goal = getattr(self.env, 'goal', None)  # Reset to the current environment's goal

    def plan_path(self, start):
        """
        Plan a path from start to goal using A* algorithm.
        
        Args:
            start: [x, y] starting position in world coordinates
            
        Returns:
            list: Path as list of [x, y] waypoints, empty list if no path found
            
        Raises:
            ValueError: If goal is not set or start position is invalid
        """
        if self.goal is None:
            raise ValueError("Goal must be set before planning path. Call set_goal() first.")
        
        # Validate start and goal positions
        self.validate_position(start, "Start")
        self.validate_position(self.goal, "Goal")
        
        # Check if start is in dilated obstacle region but not in original obstacle
        start_grid = self.world_to_grid(start)
        start_in_dilated = not self.is_free_space(start_grid)
        start_in_original = self.env.occupancy_grid[start_grid[1], start_grid[0]] != 0
        
        # If agent is trapped in dilated region, temporarily use original map for planning
        if start_in_dilated and not start_in_original:
            print(f" Agent at {start} is in dilated obstacle region - using original map for planning")
            original_occupancy = self.occupancy_grid
            self.occupancy_grid = self.env.occupancy_grid  # Temporarily use original map
        
        start_grid = self.world_to_grid(start)
        goal_grid = self.world_to_grid(self.goal)
        
        print(f" Planning path from {start} to {self.goal}")
        
        # Initialize A* data structures
        open_set = []
        closed_set = set()
        came_from = {}
        
        # Create start node
        start_node = Node(start_grid, 0, self.heuristic(start_grid, goal_grid))
        heapq.heappush(open_set, start_node)
        
        # Track g_scores for efficient updates
        g_scores = defaultdict(lambda: float('inf'))
        g_scores[tuple(start_grid)] = 0
        
        iterations = 0
        
        while open_set and iterations < self.max_iter:
            iterations += 1
            
            # Get node with lowest f_cost
            current = heapq.heappop(open_set)
            current_pos = tuple(current.position)
            
            # Skip if we've already processed this position with a better cost
            if current_pos in closed_set:
                continue
                
            closed_set.add(current_pos)
            
            # Check if we reached the goal
            if np.linalg.norm(current.position - goal_grid) * self.resolution <= self.goal_tolerance:
                path = self.reconstruct_path(current)
                print(f"✅ Path found with {len(path)} waypoints after {iterations} iterations")
                self.env.path = path  # Store the path in environment
                
                # Restore original dilated map if we temporarily changed it
                if start_in_dilated and not start_in_original:
                    self.occupancy_grid = original_occupancy
                
                return path
            
            # Explore neighbors
            for neighbor_pos in self.get_neighbors(current.position):
                neighbor_tuple = tuple(neighbor_pos)
                
                # Skip if already processed or not free space
                if neighbor_tuple in closed_set or not self.is_free_space(neighbor_pos):
                    continue
                
                # Calculate tentative g_score using world distance
                current_world = self.grid_to_world(current.position)
                neighbor_world = self.grid_to_world(neighbor_pos)
                tentative_g = current.g_cost + np.linalg.norm(neighbor_world - current_world)
                
                # Skip if we found a worse path to this neighbor
                if tentative_g >= g_scores[neighbor_tuple]:
                    continue
                
                # This is the best path to this neighbor so far
                g_scores[neighbor_tuple] = tentative_g
                h_cost = self.heuristic(neighbor_pos, goal_grid)
                neighbor_node = Node(neighbor_pos, tentative_g, h_cost, current)
                
                heapq.heappush(open_set, neighbor_node)
        
        # Restore original dilated map if we temporarily changed it
        if start_in_dilated and not start_in_original:
            self.occupancy_grid = original_occupancy
        
        # No path found
        print(f"❌ No path found after {iterations} iterations from {start} to {self.goal}")
        print(f"   Start grid: {self.world_to_grid(start)}, Goal grid: {self.world_to_grid(self.goal)}")
        print(f"   Goal tolerance: {self.goal_tolerance}m, Step size: {self.step_size}m")
        return []

    def get_neighbors(self, position):
        """Get valid neighboring grid positions using grid_step_size."""
        neighbors = []
        x, y = int(position[0]), int(position[1])
        
        # 8-connected neighbors with larger step size
        for dx in [-self.grid_step_size, 0, self.grid_step_size]:
            for dy in [-self.grid_step_size, 0, self.grid_step_size]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                
                # Check bounds
                if (0 <= nx < self.occupancy_grid.shape[1] and 
                    0 <= ny < self.occupancy_grid.shape[0]):
                    neighbors.append(np.array([nx, ny]))
        
        return neighbors

    def heuristic(self, pos1, pos2):
        """Euclidean distance heuristic."""
        return self.heuristic_weight * np.linalg.norm(pos1 - pos2) * self.resolution

    def reconstruct_path(self, node):
        """Reconstruct path from goal node back to start."""
        path = []
        current = node
        
        while current is not None:
            path.append(self.grid_to_world(current.position))
            current = current.parent
        
        return path[::-1]  # Reverse to get start-to-goal path

    def is_free_space(self, position):
        """Checks if a grid cell is free (not an obstacle) and within bounds."""
        x, y = int(position[0]), int(position[1])
        if 0 <= x < self.occupancy_grid.shape[1] and 0 <= y < self.occupancy_grid.shape[0]:
            return self.occupancy_grid[y, x] == 0
        return False

    def validate_position(self, position, position_name="Position"):
        """
        Validate that a world position is within bounds and in free space.
        
        Args:
            position: [x, y] position in world coordinates
            position_name: Name for error messages (e.g., "Goal", "Start")
            
        Returns:
            bool: True if position is valid
            
        Raises:
            ValueError: If position is invalid with detailed error message
        """
        if position is None:
            raise ValueError(f"{position_name} cannot be None")
            
        grid_pos = self.world_to_grid(position)
        
        # Check bounds
        if (grid_pos[0] < 0 or grid_pos[0] >= self.occupancy_grid.shape[1] or
            grid_pos[1] < 0 or grid_pos[1] >= self.occupancy_grid.shape[0]):
            raise ValueError(f"{position_name} {position} is outside map bounds. "
                           f"Grid position: {grid_pos}, Map size: {self.occupancy_grid.shape}")
        
        # For start positions, be more lenient - only check against original obstacles
        if position_name.lower() == "start":
            original_grid = self.env.occupancy_grid
            if original_grid[grid_pos[1], grid_pos[0]] != 0:
                raise ValueError(f"{position_name} {position} is in actual obstacle! "
                               f"Grid position: {grid_pos}, "
                               f"Original occupancy value: {original_grid[grid_pos[1], grid_pos[0]]}")
            
            # If start is in dilated region but not original obstacle, issue warning but allow it
            if not self.is_free_space(grid_pos):
                print(f"  Warning: Start position {position} is in dilated obstacle region but proceeding anyway")
        else:
            # For goals, be more permissive for exploration - check original map first
            original_grid = self.env.occupancy_grid
            if original_grid[grid_pos[1], grid_pos[0]] != 0:
                raise ValueError(f"{position_name} {position} is in actual obstacle! "
                               f"Grid position: {grid_pos}, "
                               f"Original occupancy value: {original_grid[grid_pos[1], grid_pos[0]]}")
            
            # If goal is in dilated region but not original obstacle, allow it with warning
            if not self.is_free_space(grid_pos):
                print(f"  Warning: {position_name} {position} is in dilated obstacle region - allowing for exploration")
        
        return True

    def world_to_grid(self, position):
        """Converts world coordinates to grid coordinates."""
        grid_x = int((position[0] - self.origin['x']) / self.resolution)
        grid_y = int((position[1] - self.origin['y']) / self.resolution)
        return np.array([grid_x, grid_y])

    def grid_to_world(self, grid_position):
        """Converts grid coordinates back to world coordinates."""
        x = grid_position[0] * self.resolution + self.origin['x']
        y = grid_position[1] * self.resolution + self.origin['y']
        return np.array([x, y])

    def set_goal(self, goal):
        """
        Set the goal position with validation to check if it's in free space.
        
        Args:
            goal: [x, y] position in world coordinates
            
        Raises:
            ValueError: If the goal position is occupied/invalid
        """
        self.validate_position(goal, "Goal")
        print(f" Goal {goal} is valid and in free space")
        self.goal = goal
