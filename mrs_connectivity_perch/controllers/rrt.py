import random

import numpy as np


class Node:
    def __init__(self, position):
        self.position = np.array(position)
        self.parent = None


class RRT:
    name = "RRT"

    def __init__(self, environment, step_size=0.5, max_iter=100000, goal_tolerance=0.5, goal_bias=0.1):
        self.env = environment
        self.step_size = step_size  # Distance to move toward sampled points
        self.max_iter = max_iter  # Maximum iterations to attempt
        self.goal_tolerance = goal_tolerance  # Distance to consider as reaching the goal
        self.goal_bias = goal_bias  # Probability of sampling the goal directly
        self.tree = []  # Tree to store nodes
        self.occupancy_grid = environment.occupancy_grid
        self.origin = environment.origin
        self.resolution = environment.resolution
        self.goal = environment.goal

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
            self.occupancy_grid = environment.occupancy_grid
            self.origin = environment.origin
            self.resolution = environment.resolution

        if goal is not None:
            self.goal = goal
        else:
            self.goal = self.env.goal  # Reset to the current environment's goal

        # Clear the tree for a fresh start
        self.tree = []

    def plan_path(self, start):
        start_node = Node(self.world_to_grid(start))
        goal_node = Node(self.world_to_grid(self.goal))
        self.tree = [start_node]  # Initialize tree with the start node

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
                self.env.render()  # Call the render function to update the display

                # Check if the new node is within goal tolerance
                if np.linalg.norm(new_node.position - goal_node.position) * self.resolution <= self.goal_tolerance:
                    self.env.path = self.build_path(new_node)  # Store the path in environment
                    return self.env.path

        # Return an empty path if the goal was not reached within max iterations
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
        self.goal = goal
