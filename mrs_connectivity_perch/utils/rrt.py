import random

import numpy as np

np.random.seed(14)
random.seed(14)

class Node:
    def __init__(self, position):
        self.position = np.array(position)
        self.parent = None

class RRT:
    name = "RRT"

    def __init__(self, environment, step_size=0.5, max_iter=100000, goal_tolerance=0.2, goal_bias=0.5):
        self.env = environment
        self.step_size = step_size  # Distance to move toward sampled points
        self.max_iter = max_iter  # Maximum iterations to attempt
        self.goal_tolerance = goal_tolerance  # Distance to consider as reaching the goal
        self.goal_bias = goal_bias  # Probability of sampling the goal directly
        self.tree = []  # Tree to store nodes
        
        # Use environment's pre-computed dilated map for consistency
        self.occupancy_grid = environment.occupancy_grid_dilated
        self.origin = environment.origin
        self.resolution = environment.resolution
        self.goal = None

    def set_params(self, step_size=0.5, max_iter=100000, goal_tolerance=0.5, goal_bias=0.1):
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_tolerance = goal_tolerance
        self.goal_bias = goal_bias

    def reinitialize(self, environment=None, goal=None):
        """
        Reinitializes the RRT controller, resetting its internal state and optionally updating the environment or goal.
        
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

        if goal is not None:
            self.goal = goal
        else:
            self.goal = self.env.goal  # Reset to the current environment's goal

        # Clear the tree for a fresh start
        self.tree = []

    def plan_path(self, start):
        """
        Plan a path from start to goal using RRT algorithm.
        
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
        
        start_node = Node(self.world_to_grid(start))
        goal_node = Node(self.world_to_grid(self.goal))
        self.tree = [start_node]  # Initialize tree with the start node

        print(f" Planning path from {start} to {self.goal}")
        
        for i in range(self.max_iter):
            # Sample a random point in the environment, with goal biasing
            rand_point = self.sample_point()
            nearest_node = self.nearest_node(rand_point)
            new_node = self.steer(nearest_node, rand_point)

            if new_node and self.is_free_space(new_node.position):
                # Add new node to the tree
                new_node.parent = nearest_node
                self.tree.append(new_node)

                # Render the tree incrementally
                # self.env.render()  # Call the render function to update the display

                # Check if the new node is within goal tolerance
                if np.linalg.norm(new_node.position - goal_node.position) * self.resolution <= self.goal_tolerance:
                    path = self.build_path(new_node)
                    print(f"✅ Path found with {len(path)} waypoints after {i+1} iterations")
                    self.env.path = path  # Store the path in environment
                    
                    # Restore original dilated map if we temporarily changed it
                    if start_in_dilated and not start_in_original:
                        self.occupancy_grid = original_occupancy
                    
                    return path

        # Restore original dilated map if we temporarily changed it
        if start_in_dilated and not start_in_original:
            self.occupancy_grid = original_occupancy
        
        # Return an empty path if the goal was not reached within max iterations
        print(f"❌ No path found after {self.max_iter} iterations from {start} to {self.goal}")
        return []

    def sample_point(self):
        """Randomly samples a point within the grid bounds, with a probability of sampling the goal."""
        if random.random() < self.goal_bias:
            # Bias towards the goal
            return self.world_to_grid(self.goal)
        else:
            # Random point in the grid
            x_max, y_max = self.occupancy_grid.shape[1], self.occupancy_grid.shape[0]
            return np.array([random.randint(0, x_max - 1), random.randint(0, y_max - 1)])

    def nearest_node(self, point):
        """Finds the nearest node in the tree to the given point."""
        nearest = self.tree[0]
        min_dist = np.linalg.norm(nearest.position - point)
        for node in self.tree:
            dist = np.linalg.norm(node.position - point)
            if dist < min_dist:
                nearest = node
                min_dist = dist
        return nearest

    def steer(self, from_node, to_point):
        """Attempts to move from `from_node` towards `to_point` by `step_size`."""
        direction = to_point - from_node.position
        distance = np.linalg.norm(direction)
        if distance == 0:
            return None

        direction = direction / distance  # Normalize direction
        new_position = from_node.position + direction * min(self.step_size / self.resolution, distance)
        new_node = Node(new_position.astype(int))

        # Check if the path from `from_node` to `new_node` is free of obstacles
        if self.is_path_free(from_node.position, new_node.position):
            return new_node
        return None

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
                print(f" Warning: Start position {position} is in dilated obstacle region but proceeding anyway")
        else:
            # For goals and other positions, use dilated map for safety
            if not self.is_free_space(grid_pos):
                raise ValueError(f"{position_name} {position} is occupied! "
                               f"Grid position: {grid_pos}, "
                               f"Occupancy value: {self.occupancy_grid[grid_pos[1], grid_pos[0]]}")
        
        return True

    def is_path_free(self, from_position, to_position):
        """Checks if the straight line path between two points is free of obstacles."""
        from_position = np.array(from_position)
        to_position = np.array(to_position)
        num_points = int(np.linalg.norm(to_position - from_position))

        for i in range(num_points):
            intermediate_position = from_position + (to_position - from_position) * (i / num_points)
            if not self.is_free_space(intermediate_position):
                return False
        return True

    def world_to_grid(self, position):
        """Converts world coordinates to grid coordinates."""
        grid_x = int((position[0] - self.origin['x']) / self.resolution)
        grid_y = int((position[1] - self.origin['y']) / self.resolution)
        return np.array([grid_x, grid_y])

    def build_path(self, node):
        """Builds the path from the start node to the goal node."""
        path = []
        while node is not None:
            path.append(self.grid_to_world(node.position))
            node = node.parent
        return path[::-1]  # Reverse the path

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
        print(f"✅ Goal {goal} is valid and in free space")
        self.goal = goal
