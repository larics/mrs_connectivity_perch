import math

import numpy as np


class Bug0:
    name = "Bug0"

    def __init__(self, environment):
        """
        Initializes the Bug0 controller.
        
        Args:
            environment: The environment instance where the agent operates.
        """
        self.env = environment
        self.goal = None
        self.vis_radius = environment.vis_radius
        self.path_following = False  # Whether the agent is currently following an obstacle
        self.goal = environment.goal
        self.goal_tolerance = 0.2

    def set_goal(self, goal):
        self.goal = np.array(goal)

    def compute_direction_to_goal(self, current_position):
        """
        Computes the normalized direction vector from the current position to the goal.
        
        Args:
            current_position: The agent's current position as a numpy array (x, y).
        
        Returns:
            A normalized direction vector as a numpy array.
        """
        direction = self.goal - current_position
        distance = np.linalg.norm(direction)
        if distance > 0:
            return direction / distance  # Normalize
        return np.array([0, 0])

    def decide_action(self):
        """
        Decides the next action based on the Bug0 algorithm.
        
        Returns:
            A numpy array representing the velocity vector to move the agent.
        """

        if np.linalg.norm(self.env.state[:2] - self.goal) <= self.goal_tolerance:
            return np.array([0, 0])

        current_position = self.env.state[:2]
        direction_to_goal = self.compute_direction_to_goal(current_position)

        # Get the local observation (distances to obstacles in all directions)
        n_angles = 10
        local_obs = self.env.get_local_observation(n_angles=n_angles)

        # Check if there's a clear path toward the goal
        goal_angle = math.atan2(direction_to_goal[1], direction_to_goal[0])
        goal_angle_index = int((goal_angle % (2 * np.pi)) * n_angles / (2 * np.pi))

        n = len(local_obs)
        if local_obs[goal_angle_index % n] >= self.vis_radius and \
                local_obs[(goal_angle_index + 1) % n] >= self.vis_radius and \
                local_obs[(goal_angle_index - 1) % n] >= self.vis_radius:
            # Path is clear, move directly toward the goal
            return direction_to_goal * self.env.robot_speed

        # If the path to the goal is blocked, follow the obstacle (path-following mode)
        for i in range(len(local_obs) - 1, 0, -1):
            if local_obs[i] >= self.vis_radius:
                # Find the first clear direction to move
                angle = math.radians(i * 360 / n_angles)
                return np.array([math.cos(angle), math.sin(angle)]) * self.env.robot_speed

        # If no clear path is found (should not happen with typical Bug 0 logic), stop
        return np.array([0, 0])
