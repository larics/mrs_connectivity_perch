import numpy as np
from gymnasium import spaces

from mrs_connectivity_perch.envs.base import BaseEnv
from mrs_connectivity_perch.utils.swarm import Swarm
from mrs_connectivity_perch.utils.vis import SwarmRenderer

np.random.seed(12)


class PerchV0(BaseEnv):
    def __init__(self, config):
        super().__init__(config)

        self.start = True

        self.agent_config = config.get('agent_config')
        
        # Cache for combined obstacle map
        self._cached_combined_map = None
        self._cached_danger_regions_hash = None

        self.vis_radius = config.get('vis_radius')

        self.total_agents = sum(inner_dict["num_agents"] for inner_dict in self.agent_config.values())
        self.old_total_agents = self.total_agents

        self.render_type = config.get('vis_params')['render_type']
        self.show_old_path = config.get('vis_params')['show_old_path']

        self.swarm = Swarm(env=self,
                           config=config,
                           map_resolution=self.resolution,
                           map_handlers={'update_exploration_map': self.update_exploration_map,
                                         'is_free_space': self.is_free_space,
                                         'is_line_of_sight_free': self.is_line_of_sight_free,
                                         'is_line_of_sight_free_obstacle_avoidance': self.is_line_of_sight_free_for_obstacle_avoidance,
                                         'get_frontier_goal': self.get_frontier_goal})

        self.render_func = SwarmRenderer(render_type=self.render_type,
                                         env=self,
                                         swarm=self.swarm, occupancy_grid=self.occupancy_grid,
                                         origin=self.origin, resolution=self.resolution,
                                         vis_radius=self.vis_radius,
                                         plot_limits=config.get('vis_params')['plot_limits'] if
                                         config.get('vis_params')['plot_limits'] != 'None' else None)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.total_agents, 4), dtype=np.float64)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.total_agents, 2), dtype=np.float32)

    def render(self, mode='human'):
        # if self.old_total_agents > self.swarm.total_agents:
        #     if self.swarm.total_agents % 2 == 0: 
        self.render_func.render()
        #         self.old_total_agents = self.swarm.total_agents
        # elif self.start:
        #     self.render_func.render()
        #     self.start = False

    def get_combined_obstacle_map(self):
        """
        Get a cached combined obstacle map that includes both regular obstacles and danger regions.
        The map is only regenerated when danger regions change.
        
        Returns:
            numpy.ndarray: Combined obstacle map (1=obstacle, 0=free)
        """
        # Create a simple hash of current danger regions for change detection
        current_hash = None
        if hasattr(self.swarm, 'danger_regions') and self.swarm.danger_regions:
            # Create hash based on danger region coordinates
            hash_data = []
            for region in self.swarm.danger_regions:
                hash_data.extend([region['x_min'], region['y_min'], region['x_max'], region['y_max']])
            current_hash = hash(tuple(hash_data))
        else:
            current_hash = hash(tuple())  # Empty hash for no danger regions
        
        # Check if we need to regenerate the map
        if (self._cached_combined_map is None or 
            self._cached_danger_regions_hash != current_hash):
            
            # Regenerate combined map
            combined_map = self.occupancy_grid_dilated.copy()
            
            # Add danger regions to the map
            if hasattr(self.swarm, 'danger_regions') and self.swarm.danger_regions:
                for region in self.swarm.danger_regions:
                    # Convert world coordinates to grid coordinates
                    x_min_grid = int((region['x_min'] - self.origin['x']) / self.resolution)
                    y_min_grid = int((region['y_min'] - self.origin['y']) / self.resolution)
                    x_max_grid = int((region['x_max'] - self.origin['x']) / self.resolution)
                    y_max_grid = int((region['y_max'] - self.origin['y']) / self.resolution)
                    
                    # Ensure coordinates are within map bounds
                    x_min_grid = max(0, x_min_grid)
                    y_min_grid = max(0, y_min_grid)
                    x_max_grid = min(combined_map.shape[1] - 1, x_max_grid)
                    y_max_grid = min(combined_map.shape[0] - 1, y_max_grid)
                    
                    # Mark danger region as obstacle in the combined map
                    combined_map[y_min_grid:y_max_grid+1, x_min_grid:x_max_grid+1] = 1
            
            # Cache the result
            self._cached_combined_map = combined_map
            self._cached_danger_regions_hash = current_hash
        
        return self._cached_combined_map

    def is_line_of_sight_free_for_obstacle_avoidance(self, position1, position2):
        """
        Check if line of sight is free using the combined obstacle map (regular obstacles + danger regions).
        This is used specifically for obstacle avoidance, not for communication.
        
        Args:
            position1 (np.ndarray): [x, y] of the first position.
            position2 (np.ndarray): [x, y] of the second position.

        Returns:
            bool: True if line of sight is free, False otherwise.
        """
        # Get the cached combined obstacle map
        combined_map = self.get_combined_obstacle_map()
        
        # Use the same line-of-sight algorithm as the base class but with combined map
        def position_to_grid(position):
            grid_x = int((position[0] - self.origin['x']) / self.resolution)
            grid_y = int((position[1] - self.origin['y']) / self.resolution)
            return grid_x, grid_y

        start = position_to_grid(position1)
        end = position_to_grid(position2)

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
            if not (0 <= x0 < combined_map.shape[1] and 0 <= y0 < combined_map.shape[0]):
                return False
                
            # Check if the current grid cell is an obstacle (including danger regions)
            if combined_map[y0, x0] == 1:
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
