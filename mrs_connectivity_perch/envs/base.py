import os

from math import sin, cos
from pathlib import Path

import cv2
from gymnasium import Env
import numpy as np
import yaml

from mrs_connectivity_perch.controllers.frontier_explore import FrontierDetector
from mrs_connectivity_perch.utils.mission_metrics import MissionMetrics

class BaseEnv(Env):
    def __init__(self, config):
        super().__init__()

        self.dt = config.get('dt')  # Returns float

        self.occupancy_grid = None  # Original map - to be loaded via config
        self.occupancy_grid_dilated = None  # Dilated map for obstacle avoidance
        self.origin = None
        self.resolution = None

        self.fig = None

        # exploration map variables
        self.exploration_map = None
        self.frontier_detector = None

        # swarm variables
        self.swarm = None
        
        # mission metrics tracking (optional, non-invasive)
        self.mission_metrics = None
        self._enable_metrics = config.get('enable_metrics', False)

        # map setup
        self.map_name = config.get('map_name')
        self.dilation_distance = config.get('obstacle_dilation_distance', 0.05)  # Default 10cm dilation
        
        PACKAGE_ROOT = Path(__file__).resolve().parents[1]
        self.maps_folder = PACKAGE_ROOT / "maps" / self.map_name
        self.image_path = self.maps_folder / "map.bmp"
        self.yaml_path = self.maps_folder / "data.yaml"
        
        # Load both original and dilated maps
        self.occupancy_grid = self.load_occupancy_grid(str(self.image_path))
        self.origin, self.resolution, self.sector_size = self.load_yaml_config(str(self.yaml_path))
        self.occupancy_grid_dilated = self._create_dilated_map(self.occupancy_grid, self.dilation_distance)

        # Initialize sector grid
        self.sector_grid = self._create_sector_grid()

        self.exploration_map = np.full_like(self.occupancy_grid, -1)
        self.frontier_detector = FrontierDetector(self.exploration_map, self.resolution,
                                                  [self.origin['x'], self.origin['y']], robot_size=0.5)

    def _create_dilated_map(self, occupancy_grid, dilation_distance_meters):
        """
        Create a dilated version of the occupancy grid for obstacle avoidance.
        
        Args:
            occupancy_grid: Binary grid where 1=occupied, 0=free
            dilation_distance_meters: Distance to dilate obstacles in meters
        
        Returns:
            Dilated occupancy grid
        """
        
        if dilation_distance_meters <= 0:
            return occupancy_grid.copy()
        
        # Convert distance to pixels
        dilation_pixels = max(1, int(dilation_distance_meters / self.resolution))
        
        # Create circular kernel for more natural dilation
        kernel_size = dilation_pixels * 2 + 1
        y, x = np.ogrid[-dilation_pixels:dilation_pixels+1, -dilation_pixels:dilation_pixels+1]
        kernel = (x**2 + y**2 <= dilation_pixels**2).astype(np.uint8)
        
        # Convert to uint8 for OpenCV
        occupancy_uint8 = (occupancy_grid * 255).astype(np.uint8)
        
        # Dilate obstacles (white areas in the image)
        dilated_uint8 = cv2.dilate(occupancy_uint8, kernel, iterations=1)
        
        # Convert back to binary
        dilated_grid = (dilated_uint8 / 255.0).astype(np.float64)
        
        print(f"Created dilated map with {dilation_distance_meters}m ({dilation_pixels} pixels) dilation")
        print(f"Original obstacles: {np.sum(occupancy_grid)} pixels")
        print(f"Dilated obstacles: {np.sum(dilated_grid)} pixels")
        
        return dilated_grid

    def _create_sector_grid(self):
        """
        Create a chessboard-like grid of sectors based on map bounds and sector size.
        
        Returns:
            dict: Contains sector information including grid lines and sector bounds
        """
        # Calculate map bounds in world coordinates
        map_width = self.occupancy_grid.shape[1] * self.resolution
        map_height = self.occupancy_grid.shape[0] * self.resolution
        
        map_x_min = self.origin['x']
        map_y_min = self.origin['y']
        map_x_max = map_x_min + map_width
        map_y_max = map_y_min + map_height
        
        # Calculate number of sectors in each dimension (ceiling to handle non-divisible cases)
        sectors_x = int(np.ceil(map_width / self.sector_size))
        sectors_y = int(np.ceil(map_height / self.sector_size))
        
        # Generate grid lines
        x_lines = []
        y_lines = []
        
        # Vertical lines (x-coordinates)
        for i in range(sectors_x + 1):
            x = map_x_min + i * self.sector_size
            if x <= map_x_max:  # Only add lines within map bounds
                x_lines.append(x)
        
        # Horizontal lines (y-coordinates)
        for i in range(sectors_y + 1):
            y = map_y_min + i * self.sector_size
            if y <= map_y_max:  # Only add lines within map bounds
                y_lines.append(y)
        
        # Create sector information with chessboard naming
        sectors = []
        for i in range(sectors_x):
            for j in range(sectors_y):
                sector_x_min = map_x_min + i * self.sector_size
                sector_y_min = map_y_min + j * self.sector_size
                sector_x_max = min(sector_x_min + self.sector_size, map_x_max)
                sector_y_max = min(sector_y_min + self.sector_size, map_y_max)
                
                # Generate chessboard-style name (like a1, b2, c3, etc.)
                # Column letter: a, b, c, ... (using i for column index)
                col_letter = chr(ord('a') + i)
                # Row number: 1, 2, 3, ... (using j+1 for row index, +1 to start from 1)
                row_number = j + 1
                sector_name = f"{col_letter}{row_number}"
                
                sectors.append({
                    'id': i * sectors_y + j,
                    'name': sector_name,
                    'row': j,
                    'col': i,
                    'x_min': sector_x_min,
                    'y_min': sector_y_min,
                    'x_max': sector_x_max,
                    'y_max': sector_y_max,
                    'center': [(sector_x_min + sector_x_max) / 2, (sector_y_min + sector_y_max) / 2],
                    'size': self.sector_size
                })
        
        print(f"Created sector grid: {sectors_x}x{sectors_y} sectors of size {self.sector_size}m")
        print(f"Map bounds: ({map_x_min:.2f}, {map_y_min:.2f}) to ({map_x_max:.2f}, {map_y_max:.2f})")
        
        return {
            'sectors': sectors,
            'x_lines': x_lines,
            'y_lines': y_lines,
            'sectors_x': sectors_x,
            'sectors_y': sectors_y,
            'sector_size': self.sector_size
        }

    def get_sector_from_position(self, position):
        """
        Get the sector ID for a given world position.
        
        Args:
            position: [x, y] position in world coordinates
            
        Returns:
            int or None: Sector ID if position is within map bounds, None otherwise
        """
        x, y = position[:2]
        
        # Calculate sector indices
        sector_col = int((x - self.origin['x']) // self.sector_size)
        sector_row = int((y - self.origin['y']) // self.sector_size)
        
        # Check if within bounds
        if (0 <= sector_col < self.sector_grid['sectors_x'] and 
            0 <= sector_row < self.sector_grid['sectors_y']):
            return sector_col * self.sector_grid['sectors_y'] + sector_row
        
        return None

    def load_yaml_config(self, yaml_path):
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        origin = {
            'x': float(config.get('origin', {}).get('x', 0.0)),
            'y': float(config.get('origin', {}).get('y', 0.0))
        }
        resolution = float(config.get('resolution', 0.1))
        sector_size = float(config.get('sector_size', 1.0))  # Default 1.0 meter sectors
        return origin, resolution, sector_size

    def load_occupancy_grid(self, image_path):

        # Debugging: Check if the file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File does not exist: {image_path}")

        # Attempt to read the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(
                f"Failed to load image from {image_path}. Ensure the file exists and is a valid BMP image.")

        # Convert the image to a binary grid
        _, binary_grid = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
        return 1.0 - binary_grid

    def position_to_grid(self, position):
        grid_x = int((position[0] - self.origin['x']) / self.resolution)
        grid_y = int((position[1] - self.origin['y']) / self.resolution)
        return grid_x, grid_y

    def grid_to_position(self, grid_coord):
        grid_x, grid_y = grid_coord  # (col, row)
        x = self.origin['x'] + grid_x * self.resolution
        y = self.origin['y'] + grid_y * self.resolution
        return [x, y]


    def is_free_space(self, position, use_dilated=False):
        """
        Check if a position is in free space.
        
        Args:
            position: [x, y] position in world coordinates
            use_dilated: If True, use dilated map for obstacle avoidance
        
        Returns:
            bool: True if position is free, False if occupied
        """
        grid_x, grid_y = self.position_to_grid(position)
        
        # Choose which map to use
        map_to_check = self.occupancy_grid_dilated if use_dilated else self.occupancy_grid
        
        if 0 <= grid_x < map_to_check.shape[1] and 0 <= grid_y < map_to_check.shape[0]:
            return map_to_check[grid_y, grid_x] == 0
        return False  # Out of bounds should be considered occupied/not free

    # TODO: Refactor this function
    def is_line_of_sight_free(self, position1, position2, use_dilated=False):
        """
        Checks if the line of sight between two positions is obstacle-free.

        Args:
            position1 (np.ndarray): [x, y] of the first position.
            position2 (np.ndarray): [x, y] of the second position.
            use_dilated: If True, use dilated map for obstacle avoidance

        Returns:
            bool: True if line of sight is free, False otherwise.
        """

        def position_to_grid(position):
            grid_x = int((position[0] - self.origin['x']) / self.resolution)
            grid_y = int((position[1] - self.origin['y']) / self.resolution)
            return grid_x, grid_y

        start = position_to_grid(position1)
        end = position_to_grid(position2)
        
        # Choose which map to use
        map_to_check = self.occupancy_grid_dilated if use_dilated else self.occupancy_grid

        # Bresenham's line algorithm
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            # Check bounds first
            if not (0 <= x0 < map_to_check.shape[1] and 0 <= y0 < map_to_check.shape[0]):
                return False
                
            # Check if the current grid cell is an obstacle
            if map_to_check[y0, x0] == 1:
                return False

            if (x0, y0) == (x1, y1):
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # for i, agent in enumerate(self.swarm.agents):
        #     if agent.path is None:
        #         agent.state[:2] = self.init_positions[i]
        #         agent.state[2:] = 0
        return self.swarm.get_all_states(), {}

    def step(self, actions=None):
        if actions is not None:
            self.swarm.step(actions, self.is_free_space)

        rewards = np.zeros(self.num_agents)
        terminated = False
        truncated = False
        obs = self.swarm.get_states()
        return obs, rewards, terminated, truncated, {}

    def get_dummy_action(self):
        """
        Creates a dummy action as defined in the swarm class. Usually a translation.

        Returns:
            numpy.ndarray: An n x k array representing actions for all states.
        """
        return self.swarm.get_dummy_action()

    def render(self, mode='human'):
        pass

    # TODO: return all the velocities from the controller for learning purposes
    def controller(self):
        self.swarm.run_all_controllers()
        
        # Update mission metrics if enabled
        if self.mission_metrics is not None:
            self.mission_metrics.update_step(self.swarm)

    def update_exploration_map(self, state, local_obs, n_angles, sensor_radius):
        """
        Updates the exploration map given

        Args:
            state (numpy.ndarray): position of the agent
            local_obs (list): lidar values
            n_angles (int): resolution of lidar
            sensor_radius (float): max range of the lidar sensor
        """
        current_x, current_y = state
        for i in range(n_angles):
            angle = i * (2 * np.pi / n_angles)
            found_obstacle = False
            if local_obs[i] < sensor_radius:
                for r in np.linspace(0, local_obs[i], int(local_obs[i] / self.resolution)):
                    x = current_x + r * cos(angle)
                    y = current_y + r * sin(angle)
                    grid_x, grid_y = self.position_to_grid((x, y))
                    if 0 <= grid_x < self.exploration_map.shape[1] and 0 <= grid_y < self.exploration_map.shape[0]:
                        if not self.is_free_space((x, y)):
                            self.exploration_map[grid_y, grid_x] = 1
                            found_obstacle = True
                            break
                        else:
                            self.exploration_map[grid_y, grid_x] = 0

    # TODO: include error handling if no goal is present
    def get_frontier_goal(self, state):
        """
        Runs the frontier exploration algorithm to find the closest frontier to the agent's state.

        Args:
            state (numpy.ndarray): The position of the agent.

        Returns:
            numpy.ndarray or None: The frontier goal selected from the exploration map,
        """
        if not isinstance(state, np.ndarray) or state.shape[0] < 2:
            raise ValueError(f"Invalid state provided: {state}. State must be a numpy array with at least 2 elements.")

        try:
            self.frontier_detector.set_map(self.exploration_map)
            frontier_map = self.frontier_detector.detect_frontiers()
            candidate_points, labelled_frontiers = self.frontier_detector.label_frontiers(frontier_map)
            ordered_points = self.frontier_detector.nearest_frontier(candidate_points, state)
        except Exception as e:
            raise RuntimeError(f"Error during frontier detection: {e}")


        if len(ordered_points) > 0:
            goal = np.array(ordered_points[0])
        else:
            print("NO GOAL FOUND")
            goal = None

        return goal

    def enable_mission_metrics(self, log_directory: str = "logs", mission_name: str = None, config_name: str = None, trial_number: int = 1):
        """
        Enable mission metrics tracking (optional, non-invasive).
        
        Args:
            log_directory: Directory to save metric logs
            mission_name: Name for this mission
            config_name: Name of the config file used
            trial_number: Trial number for this configuration
        """
        try:
            self.mission_metrics = MissionMetrics(log_directory, mission_name, config_name, trial_number)
            self._enable_metrics = True
            print("📊 Mission metrics enabled")
        except ImportError:
            print("📊 Mission metrics not available")
    
    def get_mission_metrics(self):
        """Get current mission metrics summary."""
        if self.mission_metrics is not None:
            return self.mission_metrics.get_current_metrics()
        return None
    
    def save_mission_metrics(self, filename: str = None):
        """Save mission metrics to file."""
        if self.mission_metrics is not None:
            return self.mission_metrics.save_metrics(self.swarm, filename)
        return None
