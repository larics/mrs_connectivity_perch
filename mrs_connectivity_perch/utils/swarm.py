import math
import os

import numpy as np
from scipy.linalg import eig
from scipy.spatial.distance import euclidean

import numpy as np
from numpy.linalg import inv
from scipy.linalg import eig
from collections import deque

# from mrs_connectivity_perch.utils.add_agents import AddAgent
from mrs_connectivity_perch.utils.agent_base import Agent
from mrs_connectivity_perch.utils.robot import Robot
from mrs_connectivity_perch.utils.perching_pos import PerchingPos
from mrs_connectivity_perch.utils.uav import UAV
# # from mrs_connectivity_perch.utils.lattice_generation import gen_lattice
from mrs_connectivity_perch.utils.rrt import RRT
from mrs_connectivity_perch.utils.astar import AStar
from mrs_connectivity_perch.utils.utils import get_curve
from mrs_connectivity_perch.utils.data_logger import SwarmDataLogger
# from mrs_connectivity_perch.utils.relay_astar import RelayPlanner
from mrs_connectivity_perch.utils.base import Base
from mrs_connectivity_perch.utils.controller import PerchController
from mrs_connectivity_perch.utils.connectivity_controller import ConnectivityController

from mrs_connectivity_perch.utils.rrt import RRT

# The comments and docstrings in this file help the LLM understand the purpose and functionality of the Swarm class.

"""
The Swarm class manages a collection of heterogeneous agents in a multi-agent system.
This class handles the initialization, simulation, and control of agents in a swarm, 
including path planning, battery management, neighbor updates, and connectivity checks. 
It supports dynamic addition and removal of agents based on battery levels or user-defined 
criteria.
"""

np.random.seed(124)


class Swarm:
    def __init__(self, env, config, map_resolution, map_handlers, is_initialize_swarm=True):
        """
        Initialize the Swarm with environment, configuration, and map handlers.
        
        Args:
            env: The simulation environment
            config: Configuration dictionary containing swarm parameters
            map_resolution: Resolution of the map grid
            map_handlers: Dictionary of map-related functions (line of sight, free space, etc.)
            is_initialize_swarm: Whether to initialize agents during construction (default: True)
        """
        self.env = env
        self.action_dim = 2
        self.vis_radius = config.get('vis_radius')
        self.dt = config.get('dt')
        self.agent_config = config.get('agent_config', {})

        # Controller parameters
        self.controller_params = config.get('controller_params', {}).copy()
        delta = self.controller_params.get('delta', 0.1)
        repel_threshold = self.controller_params.get('repelThreshold', 0.5)
        self.controller_params['sigma'] = math.sqrt(-self.vis_radius / (2 * math.log(delta)))
        self.controller_params['range'] = self.vis_radius
        self.controller_params['repelThreshold'] = self.vis_radius * repel_threshold

        self.total_agents = sum(
            agent_conf.get("num_agents", 1) for agent_conf in self.agent_config.values()
        )

        # Map handlers
        self.is_line_of_sight_free_fn = map_handlers.get('is_line_of_sight_free')
        self.is_free_space_fn = map_handlers.get('is_free_space')
        self.update_exploration_map_fn = map_handlers.get('update_exploration_map')
        self.get_frontier_goal_fn = map_handlers.get('get_frontier_goal')
        # Obstacle avoidance function that considers danger regions
        self.is_line_of_sight_free_obstacle_avoidance_fn = map_handlers.get('is_line_of_sight_free_obstacle_avoidance')

        self.agents = []
        self.is_setup_init = False
        self.n_relays = None

        # Initialize data logger
        self.data_logger = SwarmDataLogger(config)

        self._initialize_agents(map_resolution) if is_initialize_swarm else None

        # Agent Additions
        # if self.agents:
        #     self.add_agent_params = config.get('add_agent_params', {})
        #     self.add_agent = AddAgent(self.add_agent_params, self.agents, self.vis_radius)
        #     self.add_agent_already_added = []

        comm_radius_grid = self.vis_radius / self.env.resolution
        # self.relay_planner = RelayPlanner(grid=env.occupancy_grid, comm_radius=comm_radius_grid * 0.7)
        self.old_relay_positions_n = 0
        self.old_relay_positions = None
        
        # Danger regions - list of rectangular regions that act as obstacles
        self.danger_regions = []
        
        # Sector information - store sector grid from environment
        self.sectors = {}  # Dictionary mapping sector names to sector info
        self.sector_grid = None  # Reference to the full sector grid
        
        # Pause/resume control
        self.is_paused = False
        
        # Dynamic Critical Areas Management
        self.critical_areas_config = config.get('critical_areas', {})
        self.critical_areas_enabled = self.critical_areas_config.get('enabled', False)
        self.critical_areas_definitions = self.critical_areas_config.get('areas', [])
        self.current_timestep = 0  # Track current timestep for critical area management
        self.active_critical_areas = set()  # Track currently active critical area IDs (subset of danger_regions)

        # Initialize sector information after environment setup
        self._initialize_sector_information()


    def _initialize_sector_information(self):
        """
        Initialize sector information from the environment's sector grid.
        Stores sector data in the swarm for easy access by name and position.
        """
        if hasattr(self.env, 'sector_grid') and self.env.sector_grid:
            self.sector_grid = self.env.sector_grid
            
            # Create a dictionary mapping sector names to sector information
            for sector in self.sector_grid['sectors']:
                sector_name = sector['name']
                self.sectors[sector_name] = {
                    'name': sector_name,
                    'center': sector['center'],
                    'size': sector['size'],
                    'bounds': {
                        'x_min': sector['x_min'],
                        'y_min': sector['y_min'],
                        'x_max': sector['x_max'],
                        'y_max': sector['y_max']
                    },
                    'row': sector['row'],
                    'col': sector['col'],
                    'id': sector['id']
                }
            
            print(f"Initialized {len(self.sectors)} sectors in swarm")
            print(f"Sector names: {sorted(self.sectors.keys())}")
        else:
            print("No sector grid found in environment")

    def _initialize_agents(self, map_resolution):
        """
        Initialize all agents according to the agent configuration.
        
        Args:
            map_resolution: Resolution of the map grid for agent initialization
        """
        for agent_type, agent_conf in self.agent_config.items():
            num_agents = agent_conf.get('num_agents', 1)
            controller_type = agent_conf.get('controller_type', 'do_not_move')
            obstacle_avoidance = agent_conf.get('obstacle_avoidance', 0)
            speed = agent_conf.get('speed', 0.0)

            # Initial Positions
            init_position = agent_conf.get('init_position', None)
            if not init_position or init_position == 'None':
                init_formation = agent_conf.get('init_formation', {'shape': 'point'})
                print(f"Using initialization formation: {init_formation}")
                if init_formation['shape'].lower() != 'lattice':
                    init_position = get_curve(init_formation, num_agents)
                else:
                    init_position = gen_lattice(
                        num_agents, self.vis_radius,
                        np.array([0.0, 0.0]), np.array([3.5, 0.0])
                    )
            if len(init_position) < num_agents:
                raise ValueError(f"Not enough initial positions for {num_agents} agents.")

            # Path or Goal Planning
            path_planner = None
            paths = []
            goals = None
            if controller_type in ['explore', 'go_to_goal', 'random_walk']:
                # Get path planner type from config (default to A* for better performance)
                planner_type = agent_conf.get('path_planner', 'astar').lower()
                if planner_type == 'rrt':
                    path_planner = RRT(self.env)
                else:  # Default to A*
                    # Only enable exploration mode for explore controller (not random_walk)
                    exploration_mode = (controller_type == 'explore')
                    path_planner = AStar(self.env, exploration_mode=exploration_mode)
            if controller_type == 'go_to_goal':
                goals = agent_conf.get('goals', [[0.0, 0.0]] * num_agents)
            if controller_type == 'path_tracker':
                paths_dict = agent_conf.get('paths', {})
                for i in range(num_agents):
                    path_def = paths_dict.get(i, None)
                    paths.append(get_curve(path_def, speed=speed, dt=self.dt) if path_def else [])

            # Battery
            init_battery = agent_conf.get('init_battery', 1.0)
            if init_battery == 'autofill':
                init_battery = np.linspace(0.25, 1.0, num_agents)
            elif isinstance(init_battery, (int, float)):
                init_battery = [init_battery] * num_agents
            elif isinstance(init_battery, list) and len(init_battery) != num_agents:
                raise ValueError("init_battery list length must match num_agents.")

            battery_decay_rate = agent_conf.get('battery_decay_rate', 0.0)
            battery_threshold = agent_conf.get('battery_threshold', 0.0)
            decay_select = agent_conf.get('decay_select', None)
            battery_decays = [
                battery_decay_rate if decay_select and i in decay_select else 0.0
                for i in range(num_agents)
            ]

            # Agent Creation
            for i in range(num_agents):
                agent_class = Agent
                if agent_type == 'robot':
                    agent_class = Robot
                elif agent_type == 'perching_pos':
                    agent_class = PerchingPos
                elif agent_type == 'UAV':
                    agent_class = UAV
                elif agent_type == 'base':  
                    agent_class = Base

                self.agents.append(agent_class(
                    agent_type=agent_type,
                    agent_id=len(self.agents),
                    init_position=init_position[i],
                    dt=self.dt,
                    vis_radius=self.vis_radius,
                    map_resolution=map_resolution,
                    config=agent_conf,
                    controller_params=self.controller_params,
                    path_planner=path_planner,
                    path=paths[i] if i < len(paths) else [],
                    goal=goals[i] if goals else None,
                    init_battery=init_battery[i] if isinstance(init_battery, (list, np.ndarray)) else init_battery,
                    battery_decay_rate=battery_decays[i],
                    battery_threshold=battery_threshold,
                    show_old_path=getattr(self.env, 'show_old_path', False)
                ))

        self._initialize_topology()

        # After creating all agents, set the line-of-sight function for UAVs
        for agent in self.agents:
            # if isinstance(agent, UAV):
            agent.set_swarm_los_function(self.is_line_of_sight_free_fn)

    def _initialize_topology(self):
        """
        Initialize the topology of the swarm by categorizing agents and setting up relay connections.
        Sets up robots, perching positions, UAVs, and establishes relay paths between robots.
        """ 
        self.robots = [agent for agent in self.agents if isinstance(agent, Robot) or isinstance(agent, Base)]
        self.perching_pos = [agent for agent in self.agents if isinstance(agent, PerchingPos)]
        self.uavs = [agent for agent in self.agents if isinstance(agent, UAV)]

        relay_perches = []  # <-- Always define relay_perches before use
        # If there are at least two robots, find the shortest path using perching_pos as relays
        if len(self.robots) >= 2 and len(self.perching_pos) > 0:
            # Build a graph: nodes are robots and perching_pos, edges exist if within vis_radius
            nodes = self.robots + self.perching_pos
            node_ids = {id(agent): agent for agent in nodes}
            edges = {id(agent): [] for agent in nodes}
            for i, agent_i in enumerate(nodes):
                for j, agent_j in enumerate(nodes):
                    if i == j:
                        continue
                    dist = np.linalg.norm(agent_i.state[:2] - agent_j.state[:2])
                    if dist <= self.vis_radius:
                        edges[id(agent_i)].append(id(agent_j))
            # BFS for shortest path from robot[0] to robot[1] using only perching_pos as relays
            start = id(self.robots[0])
            goal = id(self.robots[1])
            queue = deque([(start, [start])])
            visited = set()
            best_path = None
            max_relay_count = -1
            while queue:
                current, path = queue.popleft()
                if current == goal:
                    relay_count = sum(1 for nid in path if isinstance(node_ids[nid], PerchingPos))
                    if relay_count > max_relay_count:
                        max_relay_count = relay_count
                        best_path = path
                    continue
                for neighbor in edges[current]:
                    if neighbor not in path:
                        queue.append((neighbor, path + [neighbor]))
            # Mark perching_pos on the best path as relays
            if best_path is not None:
                for nid in best_path:
                    agent = node_ids[nid]
                    if isinstance(agent, PerchingPos):
                        agent.is_relay = True
                        relay_perches.append(agent)

        print(f"relay perches {[relay.state[:2] for relay in relay_perches]}")

        # Place UAVs only on relay perching positions
        for i, uav in enumerate(self.uavs):
            if i < len(relay_perches):
                perch = relay_perches[i]
                uav.state[:2] = perch.state[:2]
                uav.mode = "perch"
                
                # Since they start perched, mark as occupied (not reserved)
                perch.occupied_by = uav
                
                uav.assigned_perching_agent = perch

                uav.assigned_perching_agent.battery = uav.battery

                uav.controller_mode = 'perched'

                print(f"UAV {uav.agent_id} placed on perching position {perch.agent_id}")

        # --- Print all agents with id, type, and controller in a neat way ---
        print("\n========== Swarm Agent Summary ==========")
        for agent in self.agents:
            ctrl = getattr(agent, 'controller_type', None)
            print(f"ID: {getattr(agent, 'agent_id', 'N/A'):>2} | Type: {getattr(agent, 'type', 'N/A'):>12} | Controller: {str(ctrl):>20} | Position: {getattr(agent, 'state', 'N/A')}")
        print("=========================================\n")
    

    def compute_adjacency_matrix(self, consider_obstacles=True, allowed_edge_types=None):
        """
        Compute adjacency matrix with specified allowed edge types.
        
        Args:
            consider_obstacles: Whether to check line of sight for obstacle avoidance
            allowed_edge_types: Set of tuples specifying allowed connections, e.g.:
                               {('UAV', 'UAV'), ('UAV', 'robot'), ('robot', 'base')}
                               If None, all connections are allowed except perching_pos
        
        Returns:
            Adjacency matrix for filtered agents
        """
        # Default behavior: exclude perching_pos if no allowed_edge_types specified
        if allowed_edge_types is None:
            allowed_edge_types = {
            ('UAV', 'UAV'), ('UAV', 'robot'), ('UAV', 'base'), ('UAV', 'perching_pos'),
            ('robot', 'robot'), ('robot', 'base'), ('robot', 'perching_pos'),
            ('base', 'base'), ('base', 'perching_pos')}
        
        # Filter agents based on allowed edge types
        allowed_agent_types = set()
        for edge_type in allowed_edge_types:
            allowed_agent_types.update(edge_type)
        
        filtered_agents = [agent for agent in self.agents if agent.type in allowed_agent_types]
        
        if len(filtered_agents) == 0:
            return np.array([])
        
        positions = np.array([agent.state[:2] for agent in filtered_agents])
        edge_obstacle = np.array([getattr(agent, 'is_obstacle_avoidance', False) for agent in filtered_agents])
        num_agents = len(filtered_agents)
        adjacency_matrix = np.zeros((num_agents, num_agents), dtype=int)
        
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                # Check if this edge type is allowed
                agent_i_type = filtered_agents[i].type
                agent_j_type = filtered_agents[j].type
                edge_type = (agent_i_type, agent_j_type)
                edge_type_reverse = (agent_j_type, agent_i_type)
                
                if edge_type not in allowed_edge_types and edge_type_reverse not in allowed_edge_types:
                    continue  # Skip this edge - not allowed
                
                distance = euclidean(positions[i], positions[j])
                if distance <= self.vis_radius:
                    if consider_obstacles and edge_obstacle[i] and edge_obstacle[j]:
                        if self.is_line_of_sight_free_fn(positions[i], positions[j]):
                            adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
                    elif not consider_obstacles:
                        adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
                    else:
                        # One or both agents don't have obstacle avoidance enabled
                        adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1

        return adjacency_matrix
  
    #================ CONTROLLER EXECUTION ================

    def algebraic_connectivity(self, A):
        """
        Calculate the algebraic connectivity (Fiedler value) of the adjacency matrix.
        
        Args:
            A: Adjacency matrix
            
        Returns:
            float: Fiedler value (second smallest eigenvalue of the Laplacian matrix)
        """
        D = self.degree(A)
        if np.all(np.diag(D) != 0):
            L = D - A
            eValues, _ = eig(L)
            eValues = np.sort(eValues.real)
            ac = eValues[1]
        else:
            ac = 0
        return ac
    
    def degree(self, A):
        """
        Compute the degree matrix of adjacency matrix A.
        
        Args:
            A: Adjacency matrix
            
        Returns:
            numpy.ndarray: Diagonal matrix where diagonal elements are the sum of each row in A
        """
        return np.diag(np.sum(A, axis=1))

    def run_all_controllers(self, remove_low_battery=True, battery_removal_threshold=0.05, auto_log_data=True):
        """
        Run controllers for all agents and optionally remove agents with low battery.
        Also handles dynamic critical area management based on timesteps.
        
        Args:
            remove_low_battery: Whether to remove agents with low battery (default: True)
            battery_removal_threshold: Battery level threshold for removal (default: 0.05)
            auto_log_data: Whether to automatically log timestep data (default: True)
        """
        # === DYNAMIC CRITICAL AREA MANAGEMENT ===
        self._manage_critical_areas()
        
        # Check if system is paused
        if self.is_paused:
            return  # Skip all controller execution when paused
        
        self.update_all_neighbors(consider_obstacles=True)
        
        # Run controllers for all agents
        for agent in self.agents:
            agent.run_controller(self)
        
        # Automatically log data if enabled
        if auto_log_data and self.data_logger.enabled:
            self.log_timestep_data()
        
        # Remove agents with low battery if enabled
        if remove_low_battery:
            agents_to_remove = []
            for agent in self.agents:
                if agent.type == 'UAV' and hasattr(agent, 'battery') and agent.battery <= battery_removal_threshold:
                    agents_to_remove.append(agent)
            
            # Remove the agents
            for agent in agents_to_remove:
                # print(f"Removing agent {agent.agent_id} due to low battery: {agent.battery:.3f}")
                
                # If it's a UAV with an assigned perching position, free it
                if (hasattr(agent, 'assigned_perching_agent') and 
                    agent.assigned_perching_agent is not None):
                    agent.assigned_perching_agent.occupied_by = None
                    agent.assigned_perching_agent.occupied_by_agent = None
                    agent.assigned_perching_agent.reserved_by = None
                    # print(f"Freed perching position {agent.assigned_perching_agent.agent_id}")
                
                self.remove_agent(agent)
        
        # Increment timestep counter for critical area management
        self.current_timestep += 1

    def degree_matrix(self, adjacency):
        """
        Compute the degree matrix from an adjacency matrix.
        
        Args:
            adjacency: Adjacency matrix
            
        Returns:
            numpy.ndarray: Degree matrix (diagonal matrix with node degrees)
        """
        return np.diag(np.sum(adjacency, axis=1))

    #================ SWARM MANAGEMENT ================

    def are_all_agents_active(self):
        """
        Check if all agents in the swarm are in active mode.
        
        Returns:
            bool: True if all agents are active, False otherwise
        """
        return all(agent.mode == 'active' for agent in self.agents)

    def remove_agent(self, agent):
        """
        Remove an agent from the swarm.
        
        Args:
            agent: Agent object to be removed from the swarm
        """
        self.agents.remove(agent)
        self.total_agents -= 1
        # print(f"Agent {agent.agent_id} removed due to low battery.")

    def update_all_neighbors(self, consider_obstacles=True):
        """
        Update neighbor lists for all agents based on the current adjacency matrix.
        
        Args:
            consider_obstacles: Whether to consider obstacles when determining neighbors (default: True)
        """
        adjacency_matrix = self.compute_adjacency_matrix(consider_obstacles)
        for i, agent in enumerate(self.agents):
            neighbors = [self.agents[j] for j in range(len(self.agents)) if adjacency_matrix[i, j] == 1]
            agent.set_neighbors(neighbors)
            
            # If this agent is a UAV, set discoverability to True for robot/perching_pos neighbors
            if agent.type == 'UAV':
                for neighbor in neighbors:
                    if neighbor.type in ['robot', 'perching_pos']:
                        neighbor.discoverability = True

    # TODO: Sometimes when we say LLM to go to, it tries to set a path instead of planning it so making it private
    def _set_agent_path(self, idx, path):
        """
        Set the path for an agent at the specified index.
        
        Args:
            idx: Index of the agent in the agents list
            path: Path to assign to the agent
        """
        self.agents[idx].set_path(path)

    
    # TODO: Not useful but needed by reset
    def get_all_states(self):
        """
        Get the state vectors for all agents in the swarm.
        
        Returns:
            numpy.ndarray: Array of state vectors for all agents
        """
        return np.array([agent.state for agent in self.agents])

    def step(self, actions, is_free_space_fn):
        pass
    
    
    # ================ LLM HELPER FUNCTIONS ================

    def _get_existing_agent_of_type(self, agent_type):
        """Get an existing agent of the specified type for parameter copying."""
        if agent_type == 'UAV':
            return self.uavs[0] if self.uavs else None
        elif agent_type == 'robot':
            # Find first actual robot (not base)
            for robot in self.robots:
                if getattr(robot, 'type', None) == 'robot':
                    return robot
            return None
        elif agent_type == 'base':
            # Find first actual base
            for robot in self.robots:
                if getattr(robot, 'type', None) == 'base':
                    return robot
            return None
        elif agent_type == 'perching_pos':
            return self.perching_pos[0] if self.perching_pos else None
        return None
    
    def _get_default_controller_type(self, agent_type, existing_agent):
        """Get default controller type based on agent type and existing agents."""
        if existing_agent:
            return getattr(existing_agent, 'controller_type', None)
        
        # Default controllers by type
        defaults = {
            'UAV': 'connectivity_controller',
            'robot': 'do_not_move',
            'base': 'do_not_move',
            'perching_pos': 'do_not_move'
        }
        return defaults.get(agent_type, 'do_not_move')
    
    def _get_battery_params(self, existing_agent, init_battery, battery_decay_rate, battery_threshold):
        """Get battery parameters with auto-fill from existing agent."""
        if init_battery is None:
            init_battery = getattr(existing_agent, 'battery', 1.0) if existing_agent else 1.0
        
        if battery_decay_rate is None:
            battery_decay_rate = getattr(existing_agent, 'battery_decay_rate', 0.0) if existing_agent else 0.0
        
        if battery_threshold is None:
            battery_threshold = getattr(existing_agent, 'battery_threshold', 0.0) if existing_agent else 0.0
        
        return {
            'init_battery': init_battery,
            'battery_decay_rate': battery_decay_rate,
            'battery_threshold': battery_threshold
        }
    
    def _get_default_config(self, agent_type, existing_agent, controller_type):
        """Get default configuration based on agent type."""
        if existing_agent:
            # Extract config-like parameters from existing agent
            base_config = {
                'controller_type': controller_type,
                'speed': getattr(existing_agent, 'speed', self._get_default_speed(agent_type, existing_agent)),
                'obstacle_avoidance': getattr(existing_agent, 'is_obstacle_avoidance', self._get_default_obstacle_avoidance(agent_type)),
                'sensor_radius': getattr(existing_agent, 'sensor_radius', self.vis_radius)
            }
            
            # Add UAV-specific parameters
            if agent_type == 'UAV':
                base_config.update({
                    'projection_velocity_factor': getattr(existing_agent, 'projection_velocity_factor', 14.0),
                    'battery_switch_threshold': getattr(existing_agent, 'battery_switch_threshold', 0.0)
                })
        else:
            # Default configs by type
            base_config = {
                'controller_type': controller_type,
                'speed': self._get_default_speed(agent_type, None),
                'obstacle_avoidance': self._get_default_obstacle_avoidance(agent_type),
                'sensor_radius': self.vis_radius
            }
            
            if agent_type == 'UAV':
                base_config.update({
                    'projection_velocity_factor': 14.0,
                    'battery_switch_threshold': 0.0
                })
        
        return base_config
    
    def _get_default_speed(self, agent_type, agent=None):
        """Get default speed based on agent type from swarm's agent_config."""
        
        # Get speed from the swarm's agent_config for this agent type
        if hasattr(self, 'agent_config') and agent_type in self.agent_config:
            agent_conf_speed = self.agent_config[agent_type].get('speed')
            if agent_conf_speed is not None:
                return agent_conf_speed
        
        # Fallback to type-based defaults if no config available
        speeds = {
            'UAV': 0.05,
            'robot': 0.02,
            'base': 0.0,
            'perching_pos': 0.0
        }
        return speeds.get(agent_type, 0.0)
    
    def _get_default_obstacle_avoidance(self, agent_type):
        """Get default obstacle avoidance setting based on agent type."""
        obstacle_avoidance = {
            'UAV': 1,
            'robot': 1,
            'base': 0,
            'perching_pos': 0
        }
        return obstacle_avoidance.get(agent_type, 0)
    
    def _setup_path_planner_and_goal(self, controller_type, planner_type='rrt'):
        """
        Set up path planner and goal based on controller type.
        
        Args:
            controller_type: Type of controller ('explore', 'go_to_goal', 'random_walk', etc.)
            planner_type: Type of path planner ('rrt' or 'astar')
        
        Returns:
            tuple: (path_planner, goal)
        """
        path_planner = None
        goal = None
        
        if controller_type in ['explore', 'go_to_goal', 'random_walk']:
            if planner_type.lower() == 'astar':
                # Enable exploration mode for explore controller
                exploration_mode = (controller_type == 'explore')
                path_planner = AStar(self.env, exploration_mode=exploration_mode)
            else:  # Default to RRT
                path_planner = RRT(self.env)
        
        if controller_type == 'go_to_goal':
            goal = [0.0, 0.0]  # Default goal, can be changed later
        
        return path_planner, goal
    
    def _create_agent_instance(self, agent_type, position, config, battery_params, path_planner, goal, is_relay):
        """Create the actual agent instance based on type."""
        
        common_args = {
            'agent_type': agent_type,
            'agent_id': len(self.agents),
            'init_position': np.array(position),
            'dt': self.dt,
            'vis_radius': self.vis_radius,
            'map_resolution': self.env.resolution,
            'config': config,
            'controller_params': self.controller_params,
            'path_planner': path_planner,
            'path': [],
            'goal': goal,
            'init_battery': battery_params['init_battery'],
            'battery_decay_rate': battery_params['battery_decay_rate'],
            'battery_threshold': battery_params['battery_threshold'],
            'show_old_path': getattr(self.env, 'show_old_path', False)
        }
        
        if agent_type == 'UAV':
            return UAV(**common_args)
        elif agent_type == 'robot':
            return Robot(**common_args)
        elif agent_type == 'base':
            return Base(**common_args)
        elif agent_type == 'perching_pos':
            new_perch = PerchingPos(**common_args)
            if is_relay is not None:
                new_perch.is_relay = is_relay
            return new_perch
    
    def _add_agent_to_lists(self, new_agent, agent_type):
        """Add agent to appropriate lists and update counters."""
        self.agents.append(new_agent)
        self.total_agents += 1
        
        if agent_type == 'UAV':
            self.uavs.append(new_agent)
        elif agent_type in ['robot', 'base']:
            self.robots.append(new_agent)
        elif agent_type == 'perching_pos':
            self.perching_pos.append(new_agent)
    
    def _get_agent_by_id(self, agent_id):
        """
        Get an agent by its ID.
        
        Args:
            agent_id: ID of the agent to find
        
        Returns:
            Agent object if found, None otherwise
        """
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None
    
    def _create_path_planner(self, planner_type='rrt', exploration_mode=False, **params):
        """
        Create a path planner instance.
        
        Args:
            planner_type: Type of planner ('rrt' or 'astar')
            exploration_mode: For A*, use original map instead of dilated for exploration
            **params: Additional parameters for the planner
            
        Returns:
            Path planner instance
        """
        if planner_type.lower() == 'astar':
            planner = AStar(self.env, exploration_mode=exploration_mode)
            # Set A* specific parameters
            if params:
                planner.set_params(
                    step_size=params.get('step_size', 1.0),
                    max_iter=params.get('max_iter', 10000),
                    goal_tolerance=params.get('goal_tolerance', 0.2),
                    heuristic_weight=params.get('heuristic_weight', 1.0)
                )
        else:  # Default to RRT
            planner = RRT(self.env)
            # Set RRT specific parameters
            if params:
                planner.set_params(
                    step_size=params.get('step_size', 0.5),
                    max_iter=params.get('max_iter', 100000),
                    goal_tolerance=params.get('goal_tolerance', 0.2),
                    goal_bias=params.get('goal_bias', 0.5)
                )
        
        return planner


    #================ PAUSE/RESUME ================

    def pause_system(self):
        """
        Pause all agent controllers. No agents will move or execute controllers until resumed.
        This is useful for debugging, manual intervention, or emergency stops.
        
        Returns:
            dict: Status message confirming the pause
        """
        self.is_paused = True
        return {
            "message": "System paused. All agent controllers stopped.",
            "paused": True,
            "total_agents": len(self.agents),
            "timestamp": "paused"
        }
    
    def resume_system(self):
        """
        Resume all agent controllers after a pause.
        
        Returns:
            dict: Status message confirming the resume
        """
        self.is_paused = False
        return {
            "message": "System resumed. All agent controllers active.",
            "paused": False,
            "total_agents": len(self.agents),
            "timestamp": "resumed"
        }
    
    def get_system_status(self):
        """
        Get the current system pause/resume status.
        
        Returns:
            dict: Current system status including pause state
        """
        return {
            "paused": self.is_paused,
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents if getattr(a, 'mode', 'active') == 'active']),
            "message": "System is paused" if self.is_paused else "System is running"
        }


    #================ ACTIONS ================
 

    def change_agent_controller(self, agent_id, new_controller_type, **controller_params):
        """
        Change the controller type for a specific agent.
        
        Args:
            agent_id (int): ID of the agent to modify
            new_controller_type (str): New controller type to set
            **controller_params: Additional parameters specific to the controller type
                For 'go_to_goal': goal=[x, y] (required), speed=float (optional, default: 0.01)
                For 'path_tracker': path=[[x1,y1], [x2,y2], ...] (required)
                For 'random_walk': random_walk_min_distance=float (optional, default: 1.0), 
                                  random_walk_max_distance=float (optional, default: 3.0)
                For 'explore': No additional params needed
                For 'connectivity_controller': No additional params needed
                For 'dummy': No additional params needed
                For 'do_not_move': No additional params needed
        
        Returns:
            dict: Result of the controller change operation
        """
        # Find the agent by ID
        agent = self._get_agent_by_id(agent_id)
        if agent is None:
            return {"error": f"Agent with ID {agent_id} not found"}
        
        # Validate controller type
        valid_controllers = ['connectivity_controller', 'go_to_goal', 
            'explore', 'path_tracker', 'random_walk', 'dummy', 'do_not_move']
        
        if new_controller_type not in valid_controllers:
            return {"error": f"Invalid controller type '{new_controller_type}'. "
                        f"Valid types: {valid_controllers}"}
        
        # Store old controller info for response
        old_controller_type = agent.controller_type
        
        try:
            # Set the new controller type
            agent.controller_type = new_controller_type
            
            # Handle controller-specific setup
            if new_controller_type == 'connectivity_controller':
                agent.controller = PerchController(params=self.controller_params)
                if hasattr(agent, 'mode'):
                    agent.mode = 'active'  # Set mode for UAVs that have perch/active modes
                
                # Clear any previous controller state
                if hasattr(agent, 'goal'):
                    agent.goal = None
                if hasattr(agent, 'path'):
                    agent.path = []
                
                # Reset speed to default for agent type
                if hasattr(agent, 'speed'):
                    default_speed = self._get_default_speed(agent.type, agent)
                    agent.speed = default_speed
                    print(f"Agent {agent_id} speed reset to {default_speed}")
                
                # Reset perch-specific state if needed
                if hasattr(agent, 'controller_mode'):
                    agent.controller_mode = 'connectivity'
                    
                print(f"Agent {agent_id} switched to connectivity_controller, mode set to 'active', path/goal cleared")
                
            elif new_controller_type == 'go_to_goal':
                # Requires goal parameter
                # Handle both direct keyword args and nested controller_params dict
                if 'controller_params' in controller_params:
                    # Case: function called with nested controller_params
                    nested_params = controller_params['controller_params']
                    goal = nested_params.get('goal')
                    # Don't use the speed parameter - let controller use controller_params['speed']
                else:
                    # Case: function called with direct keyword args
                    goal = controller_params.get('goal')
                    # Don't use the speed parameter - let controller use controller_params['speed']
                    
                if goal is None:
                    return {"error": "Controller 'go_to_goal' requires 'goal' parameter: goal=[x, y]"}
                
                # Create path planner (using RRT)
                path_planner = RRT(self.env)
                
                agent.controller = path_planner
                agent.goal = goal  # Set the goal on the agent
                # Don't modify agent.speed - leave it as is for other purposes
                agent.controller.set_goal(goal)
                # Plan initial path
                planned_path = agent.controller.plan_path(agent.state[:2])
                agent.set_path(planned_path)
                
            elif new_controller_type == 'explore':
                # Create path planner for exploration
                path_planner = RRT(self.env)
                agent.controller = path_planner
                # Clear existing path to force new exploration goal
                agent.path = None
                
            elif new_controller_type == 'path_tracker':
                # TODO: Add interpolation between waypoints
                # Requires path parameter
                # Handle both direct keyword args and nested controller_params dict
                if 'controller_params' in controller_params:
                    # Case: function called with nested controller_params
                    nested_params = controller_params['controller_params']
                    path = nested_params.get('path')
                else:
                    # Case: function called with direct keyword args
                    path = controller_params.get('path')
                    
                if path is None:
                    return {"error": "Controller 'path_tracker' requires 'path' parameter: path=[[x1,y1], [x2,y2], ...]"}
                
                agent.controller = None  # Path tracker doesn't use a controller object
                agent.set_path(path)
                
            elif new_controller_type == 'random_walk':
                # Create path planner for random walk (similar to explore)
                path_planner = RRT(self.env)
                agent.controller = path_planner
                
                # Set random walk parameters from controller_params or use defaults
                if 'controller_params' in controller_params:
                    nested_params = controller_params['controller_params']
                    min_distance = nested_params.get('random_walk_min_distance', 1.0)
                    max_distance = nested_params.get('random_walk_max_distance', 3.0)
                else:
                    min_distance = controller_params.get('random_walk_min_distance', 1.0)
                    max_distance = controller_params.get('random_walk_max_distance', 3.0)
                
                # Set parameters on the agent
                agent.random_walk_min_distance = min_distance
                agent.random_walk_max_distance = max_distance
                
                # Clear existing path to force new random goal sampling
                agent.path = None
                
                print(f"Agent {agent_id} switched to random_walk, min_dist={min_distance}, max_dist={max_distance}")
                
            elif new_controller_type in ['dummy', 'do_not_move']:
                # These controllers don't need special setup
                agent.controller = None
                
                # Clear any previous controller state
                if hasattr(agent, 'goal'):
                    agent.goal = None
                if hasattr(agent, 'path'):
                    agent.path = []
                
                # Reset speed to default for agent type (though these controllers typically don't move)
                if hasattr(agent, 'speed'):
                    default_speed = self._get_default_speed(agent.type, agent)
                    agent.speed = default_speed
                    print(f"Agent {agent_id} speed reset to {default_speed}")
                    
                print(f"Agent {agent_id} switched to {new_controller_type}, path/goal cleared")
                
            else:
                # This shouldn't happen due to validation above
                return {"error": f"Unhandled controller type: {new_controller_type}"}
            
            return {
                "message": f"Successfully changed agent {agent_id} controller from '{old_controller_type}' to '{new_controller_type}'",
                "agent_id": agent_id,
                "old_controller": old_controller_type,
                "new_controller": new_controller_type,
                "agent_type": agent.type,
                "controller_params": controller_params
            }
            
        except Exception as e:
            # Restore old controller type if something went wrong
            agent.controller_type = old_controller_type
            return {
                "error": f"Failed to change controller for agent {agent_id}: {str(e)}",
                "agent_id": agent_id,
                "attempted_controller": new_controller_type
            }

    def change_agent_path_planner(self, agent_id, planner_type='rrt', **params):
        """
        Change the path planner for a specific agent.
        
        Args:
            agent_id: ID of the agent to update
            planner_type: Type of planner ('rrt' or 'astar')
            **params: Additional parameters for the planner
            
        Returns:
            dict: Status message
        """
        agent = self._get_agent_by_id(agent_id)
        if agent is None:
            return {"error": f"Agent {agent_id} not found"}
        
        # Check if agent has a controller that uses path planning
        if not hasattr(agent, 'controller') or agent.controller is None:
            return {"error": f"Agent {agent_id} does not have a path planning controller"}
        
        if agent.controller_type not in ['explore', 'go_to_goal', 'random_walk']:
            return {"error": f"Agent {agent_id} controller type '{agent.controller_type}' does not use path planning"}
        
        # Create new path planner
        new_planner = self._create_path_planner(planner_type, **params)
        
        # If the agent had a goal set, transfer it to the new planner
        if hasattr(agent.controller, 'goal') and agent.controller.goal is not None:
            new_planner.set_goal(agent.controller.goal)
        
        # Replace the controller
        agent.controller = new_planner
        
        return {
            "message": f"Changed agent {agent_id} path planner to {planner_type.upper()}",
            "agent_id": agent_id,
            "planner_type": planner_type,
            "parameters": params
        }

    def assign_uav_to_perching_position(self, uav_id, perching_id):
        """
        Assign a specific UAV to a specific perching position. This creates the assignment relationship but does NOT move the UAV.
        To make the UAV actually go to the perching position, you must also change its controller to 'go_to_goal' with the perching coordinates.
        Once assigned and moving, the UAV will automatically transition to 'occupied' status when it gets close to the perching spot.
        
        Args:
            uav_id (int): ID of the UAV to assign
            perching_id (int): ID of the perching position to assign to
            
        Returns:
            dict: Result of the assignment operation
        """
        # Find the UAV
        uav = self._get_agent_by_id(uav_id)
        if uav is None:
            return {"error": f"UAV with ID {uav_id} not found"}
        
        if uav.type != 'UAV':
            return {"error": f"Agent {uav_id} is not a UAV (type: {uav.type})"}
        
        # Find the perching position
        perching_pos = self._get_agent_by_id(perching_id)
        if perching_pos is None:
            return {"error": f"Perching position with ID {perching_id} not found"}
        
        if perching_pos.type != 'perching_pos':
            return {"error": f"Agent {perching_id} is not a perching position (type: {perching_pos.type})"}
        
        # Check if perching position is available or just reserved
        if hasattr(perching_pos, 'occupied_by') and perching_pos.occupied_by is not None and perching_pos.occupied_by != uav_id:
            return {"error": f"Perching position {perching_id} is occupied by UAV {perching_pos.occupied_by}"}
        
        # Make the assignment
        uav.assigned_perching_id = perching_id
        uav.assigned_perching_agent = perching_pos
        perching_pos.reserved_by = uav_id
                
        return {"message": f"UAV {uav_id} assigned to perching position {perching_id}"}

    def detach_uav_from_perching_position(self, uav_id):
        """
        Detach a UAV from its assigned perching position, freeing up the spot.
        
        Args:
            uav_id (int): ID of the UAV to detach
            
        Returns:
            dict: Result of the detachment operation
        """
        # Find the UAV
        uav = self._get_agent_by_id(uav_id)
        if uav is None:
            return {"error": f"UAV with ID {uav_id} not found"}
        
        if uav.type != 'UAV':
            return {"error": f"Agent {uav_id} is not a UAV (type: {uav.type})"}
        
        # Check if UAV has an assigned perching position
        if not hasattr(uav, 'assigned_perching_agent') or uav.assigned_perching_agent is None:
            return {"error": f"UAV {uav_id} is not assigned to any perching position"}
        
        # Get the assigned perching position
        perching_pos = uav.assigned_perching_agent
        perching_id = perching_pos.agent_id
        
        # Clear the assignment on both sides
        uav.assigned_perching_id = None
        uav.assigned_perching_agent = None
        
        # Free up the perching position
        perching_pos.reserved_by = None
        perching_pos.occupied_by = None
        perching_pos.occupied_by_agent = None
        perching_pos.battery = 1.0  # Restore battery to full
        
        return {"message": f"UAV {uav_id} detached from perching position {perching_id}"}

    # TODO: It confuses LLM when we say move agent, it tries to remove and add a new
    # def add_agent(self, agent_type, position, config=None, controller_type=None, 
        #           init_battery=None, battery_decay_rate=None, battery_threshold=None,
        #           assign_to_perch=False, is_relay=None):
        # """
        # Unified function to add any type of agent to the swarm.
        
        # Args:
        #     agent_type (str): Type of agent ('UAV', 'robot', 'base', 'perching_pos')
        #     position: Initial position [x, y] for the agent
        #     config: Optional config dictionary. If None, copies from existing agent of same type
        #     controller_type: Controller type. If None, uses type-specific default
        #     init_battery: Initial battery level. If None, copies from existing agent
        #     battery_decay_rate: Battery decay rate. If None, copies from existing agent
        #     battery_threshold: Battery threshold. If None, copies from existing agent
        #     assign_to_perch: Whether to assign UAV to available perching position (UAV only)
        #     is_relay: Whether perching position is a relay (perching_pos only)
        
        # Returns:
        #     The newly created agent
        # """
        # # Validate agent type
        # valid_types = ['UAV', 'robot', 'base', 'perching_pos']
        # if agent_type not in valid_types:
        #     raise ValueError(f"Invalid agent_type '{agent_type}'. Must be one of: {valid_types}")
        
        # # Get existing agents of the same type for auto-filling parameters
        # existing_agent = self._get_existing_agent_of_type(agent_type)
        
        # # Auto-fill controller_type if not specified
        # if controller_type is None:
        #     controller_type = self._get_default_controller_type(agent_type, existing_agent)
        
        # # Auto-fill battery parameters if not specified
        # battery_params = self._get_battery_params(existing_agent, init_battery, battery_decay_rate, battery_threshold)
        
        # # Get or create config
        # if config is None:
        #     config = self._get_default_config(agent_type, existing_agent, controller_type)
        # else:
        #     config = config.copy()
        #     config['controller_type'] = controller_type
        
        # # Create path planner and goal if needed
        # path_planner, goal = self._setup_path_planner_and_goal(controller_type)
        
        # # Create the agent based on type
        # new_agent = self._create_agent_instance(
        #     agent_type, position, config, battery_params, 
        #     path_planner, goal, is_relay
        # )
        
        # # Add to appropriate lists and update topology
        # self._add_agent_to_lists(new_agent, agent_type)
        
        # # Handle special cases
        # if agent_type == 'UAV':
        #     new_agent.set_swarm_los_function(self.is_line_of_sight_free_fn)
        #     if assign_to_perch:
        #         self._assign_uav_to_perch(new_agent)
        
        # print(f"Added new {agent_type} with ID {new_agent.agent_id} at position {position}")
        # return new_agent

     # TODO: It confuses LLM when we say move agent, it tries to remove and add a new
    # def remove_agent(self, agent_id):
    #     """
    #     Remove an agent from the swarm by ID.
        
    #     Args:
    #         agent_id: ID of the agent to remove
        
    #     Returns:
    #         bool: True if agent was found and removed, False otherwise
    #     """
    #     agent_to_remove = None
    #     for agent in self.agents:
    #         if agent.agent_id == agent_id:
    #             agent_to_remove = agent
    #             break
        
    #     if agent_to_remove is None:
    #         # print(f"Agent with ID {agent_id} not found")
    #         return False
        
    #     # Remove from main agents list
    #     self.agents.remove(agent_to_remove)
        
    #     # Remove from type-specific lists
    #     if isinstance(agent_to_remove, UAV):
    #         self.uavs.remove(agent_to_remove)
    #         # Release perching position if occupied
    #         if hasattr(agent_to_remove, 'assigned_perching_agent') and agent_to_remove.assigned_perching_agent:
    #             perch = agent_to_remove.assigned_perching_agent
    #             perch.occupied_by = None
    #             perch.occupied_by_agent = None
    #     elif isinstance(agent_to_remove, Robot):
    #         self.robots.remove(agent_to_remove)
    #     elif isinstance(agent_to_remove, PerchingPos):
    #         self.perching_pos.remove(agent_to_remove)
        
    #     self.total_agents -= 1
    #     print(f"Removed agent {agent_id} (type: {agent_to_remove.type})")
    #     return True
   

    # ================ LLM INFORMATION GATHERING FUNCTIONS ================
    
    # def get_agent_status(self, agent_id):
    #     """
    #     Get detailed status information about a specific agent.
        
    #     Args:
    #         agent_id: ID of the agent to query
            
    #     Returns:
    #         dict: Detailed agent status including controller, mode, position, etc.
    #     """
    #     agent = self._get_agent_by_id(agent_id)
    #     if agent is None:
    #         return {"error": f"Agent with ID {agent_id} not found"}
            
    #     status = {
    #         "agent_id": agent_id,
    #         "type": agent.type,
    #         "position": [float(agent.state[0]), float(agent.state[1])],
    #         "controller_type": getattr(agent, 'controller_type', 'unknown'),
    #         "controller_object": type(agent.controller).__name__ if hasattr(agent, 'controller') and agent.controller else 'None',
    #     }
        
    #     # Add UAV-specific status
    #     if hasattr(agent, 'mode'):
    #         status["mode"] = agent.mode
    #     if hasattr(agent, 'controller_mode'):
    #         status["controller_mode"] = agent.controller_mode
    #     if hasattr(agent, 'battery'):
    #         status["battery"] = float(agent.battery)
    #     if hasattr(agent, 'goal'):
    #         status["goal"] = agent.goal
    #     if hasattr(agent, 'path'):
    #         status["path_length"] = len(agent.path) if agent.path else 0
    #     if hasattr(agent, 'assigned_perching_id'):
    #         status["assigned_perching_id"] = agent.assigned_perching_id
    #     if hasattr(agent, 'assigned_perching_agent'):
    #         status["has_assigned_perch"] = agent.assigned_perching_agent is not None
            
    #     return status
            
            
    def get_agent_details(self):
        """
        Get detailed information about specific agents or all agents.
        
        Args:
            None
        
        Returns:
            list: List of agent details with ID, type, position, battery, and status
        """

        agents_to_report = self.agents
        # agents_to_report = self.agents if agent_type is None else [a for a in self.agents if a.type == agent_type]
        
        details = []
        for agent in agents_to_report:
            agent_info = {
                "id": agent.agent_id,
                "type": agent.type,
                "position": [float(agent.state[0]), float(agent.state[1])],
                "controller_type": getattr(agent, 'controller_type', 'unknown')
            }
            
            # Add UAV-specific information
            if agent.type == 'UAV':
                if hasattr(agent, 'battery'):
                    agent_info["battery"] = float(agent.battery)
                # if hasattr(agent, 'mode'):
                #     agent_info["mode"] = agent.mode
                if hasattr(agent, 'assigned_perching_id'):
                    agent_info["assigned_perching_id"] = agent.assigned_perching_id
            
            # Add perching position-specific information
            elif agent.type == 'perching_pos':
                occupied_by = getattr(agent, 'occupied_by', None)
                reserved_by = getattr(agent, 'reserved_by', None)
                agent_info["is_occupied"] = occupied_by is not None
                agent_info["occupied_by"] = occupied_by
                agent_info["reserved_by"] = reserved_by
            
            # Add speed for robots and bases
            elif agent.type in ['robot', 'base']:
                if hasattr(agent, 'speed'):
                    agent_info["speed"] = float(agent.speed)
                
            # Add goal information if available (for any agent type)
            if hasattr(agent, 'goal') and agent.goal is not None:
                agent_info["goal"] = [float(agent.goal[0]), float(agent.goal[1])]
                
            details.append(agent_info)
            
        return details
    
    def get_agents_near_position(self, target_position, radius=1.0, agent_type=None):
        """
        Find agents within a specified radius of a target position.
        Perfect for spatial queries like "agents near the robot".
        
        Args:
            target_position (list): [x, y] coordinates of target position
            radius (float): Search radius
            agent_type (str, optional): Filter by agent type
            
        Returns:
            list: Agents within the specified radius with distances
        """
        if len(target_position) < 2:
            return {"error": "Target position must have at least x, y coordinates"}
            
        target_x, target_y = float(target_position[0]), float(target_position[1])
        nearby_agents = []
        
        for agent in self.agents:
            if agent_type is not None and agent.type != agent_type:
                continue
                
            distance = euclidean([target_x, target_y], agent.state[:2])
            
            if distance <= radius:
                nearby_agents.append({
                    "id": agent.agent_id,
                    "type": agent.type,
                    "position": [float(agent.state[0]), float(agent.state[1])],
                    "distance_from_target": float(distance)
                })
        
        # Sort by distance
        nearby_agents.sort(key=lambda x: x["distance_from_target"])
        
        return {
            "target_position": [target_x, target_y],
            "search_radius": radius,
            "agents_found": len(nearby_agents),
            "agents": nearby_agents
        }
    
    def get_path_planner_info(self, agent_id):
        """
        Get information about an agent's current path planner.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            dict: Path planner information
        """
        agent = self._get_agent_by_id(agent_id)
        if agent is None:
            return {"error": f"Agent {agent_id} not found"}
        
        if not hasattr(agent, 'controller') or agent.controller is None:
            return {"error": f"Agent {agent_id} does not have a controller"}
        
        if agent.controller_type not in ['explore', 'go_to_goal', 'random_walk']:
            return {"message": f"Agent {agent_id} does not use path planning"}
        
        planner = agent.controller
        planner_type = planner.name if hasattr(planner, 'name') else "Unknown"
        
        info = {
            "agent_id": agent_id,
            "planner_type": planner_type,
            "controller_type": agent.controller_type,
            "has_goal": hasattr(planner, 'goal') and planner.goal is not None
        }
        
        if hasattr(planner, 'goal') and planner.goal is not None:
            info["current_goal"] = planner.goal.tolist() if hasattr(planner.goal, 'tolist') else planner.goal
        
        return info

    def get_all_agent_paths(self):
        """
        Get paths for all agents in the swarm.
        
        Returns:
            list: List of paths for each agent, None if agent has no path
        """
        return [getattr(agent, 'path', None) for agent in self.agents]

    def is_perching_spot_occupied(self, perching_id):
        """
        Check if a perching spot is occupied and get basic details.
        
        Args:
            perching_id (int): ID of the perching position to check
            
        Returns:
            dict: Occupancy status and basic details
        """
        perching_pos = self._get_agent_by_id(perching_id)
        if perching_pos is None:
            return {"error": f"Perching position with ID {perching_id} not found"}
        
        if perching_pos.type != 'perching_pos':
            return {"error": f"Agent {perching_id} is not a perching position"}
        
        occupied_by = getattr(perching_pos, 'occupied_by', None)
        reserved_by = getattr(perching_pos, 'reserved_by', None)
        
        return {
            "perching_id": perching_id,
            "position": [float(perching_pos.state[0]), float(perching_pos.state[1])],
            "is_occupied": occupied_by is not None,
            "is_reserved": reserved_by is not None,
            "occupied_by": occupied_by,
            "reserved_by": reserved_by,
            "is_available": occupied_by is None and reserved_by is None
        }


    

    # ================ DANGER REGION MANAGEMENT ================
    
    def add_danger_region(self, x_min, y_min, x_max, y_max, region_id=None):
        """
        Add a rectangular danger region that acts as an obstacle for agents.
        
        Args:
            x_min (float): Minimum x coordinate of the rectangle
            y_min (float): Minimum y coordinate of the rectangle  
            x_max (float): Maximum x coordinate of the rectangle
            y_max (float): Maximum y coordinate of the rectangle
            region_id (str, optional): Unique identifier for the region
            
        Returns:
            dict: Information about the created danger region
        """
        if x_min >= x_max or y_min >= y_max:
            return {"error": "Invalid rectangle bounds: min values must be less than max values"}
            
        if region_id is None:
            region_id = f"danger_region_{len(self.danger_regions)}"
            
        # Check if region_id already exists
        if any(region['id'] == region_id for region in self.danger_regions):
            return {"error": f"Danger region with id '{region_id}' already exists"}
            
        danger_region = {
            'id': region_id,
            'x_min': float(x_min),
            'y_min': float(y_min),
            'x_max': float(x_max),
            'y_max': float(y_max),
            'width': float(x_max - x_min),
            'height': float(y_max - y_min),
            'center': [float((x_min + x_max) / 2), float((y_min + y_max) / 2)]
        }
        
        self.danger_regions.append(danger_region)
        

        
        return {
            "message": f"Danger region '{region_id}' added successfully",
            "region": danger_region,
            "total_danger_regions": len(self.danger_regions)
        }
    
    def remove_danger_region(self, region_id):
        """
        Remove a danger region by its ID.
        
        Args:
            region_id (str): ID of the region to remove
            
        Returns:
            dict: Result of the removal operation
        """
        initial_count = len(self.danger_regions)
        self.danger_regions = [region for region in self.danger_regions if region['id'] != region_id]
        
        if len(self.danger_regions) == initial_count:
            return {"error": f"Danger region with id '{region_id}' not found"}
            

        
        return {
            "message": f"Danger region '{region_id}' removed successfully",
            "remaining_regions": len(self.danger_regions)
        }
    
    def get_danger_regions(self): 
        """
        Get information about all current danger regions.
        
        Returns:
            dict: Information about all danger regions
        """
        return {
            "total_regions": len(self.danger_regions),
            "regions": self.danger_regions.copy()
        }
    
    def is_point_in_danger_region(self, position):
        """
        Check if a point is inside any danger region.
        
        Args:
            position (list or np.array): [x, y] coordinates to check
            
        Returns:
            dict: Information about whether point is in danger and which regions
        """
        x, y = float(position[0]), float(position[1])
        regions_containing_point = []
        
        for region in self.danger_regions:
            if (region['x_min'] <= x <= region['x_max'] and 
                region['y_min'] <= y <= region['y_max']):
                regions_containing_point.append(region['id'])
        
        return {
            "position": [x, y],
            "in_danger": len(regions_containing_point) > 0,
            "danger_regions": regions_containing_point,
            "total_regions_containing_point": len(regions_containing_point)
        }

    # ================ DATA LOGGING CONVENIENCE METHODS ================
    
    def get_data_logger_stats(self):
        """Get statistics about the data logger."""
        return self.data_logger.get_stats()
    
    def is_data_logging_enabled(self):
        """Check if data logging is enabled."""
        return self.data_logger.enabled

    def log_timestep_data(self):
        """Log data for the current timestep using the data logger."""
        # Compute fiedler value WITHOUT perching positions
        allowed_edge_types = {('UAV', 'UAV'), ('UAV', 'robot'), ('UAV', 'base'), 
                             ('robot', 'robot'), ('robot', 'base'), ('base', 'base')}
        A = self.compute_adjacency_matrix(allowed_edge_types=allowed_edge_types)
        
        fiedler_value = self.algebraic_connectivity(A)
        
        # Log using the data logger
        self.data_logger.log_timestep_data(self.agents, fiedler_value)
        self.data_logger.log_fiedler_data(fiedler_value, len(self.agents), A)
        
        return fiedler_value

    def save_all_data(self):
        """Save all collected data using the data logger."""
        return self.data_logger.save_all_data()

    def save_fiedler_value(self):
        """
        Compute and save the Fiedler value (algebraic connectivity) using the data logger.
        
        Note:
            Uses the data logger to save Fiedler values and adjacency matrices for analysis
        """
        A = self.compute_adjacency_matrix()
        fiedler_value = self.algebraic_connectivity(A)
        
        # Use the data logger to save the data
        self.data_logger.log_fiedler_data(fiedler_value, self.total_agents, A)
        
        return fiedler_value

    def configure_data_logging(self, **config_updates):
        """
        Update data logging configuration at runtime.
        
        Args:
            **config_updates: Configuration parameters to update
                enabled: bool - Enable/disable logging
                log_timestep_data: bool - Enable/disable timestep data logging
                log_fiedler_values: bool - Enable/disable Fiedler value logging
                save_adjacency_matrices: bool - Enable/disable adjacency matrix saving
        
        Returns:
            dict: Updated configuration status
        """
        if 'enabled' in config_updates:
            self.data_logger.enabled = config_updates['enabled']
            if self.data_logger.enabled:
                self.data_logger._create_directories()
        
        if 'log_timestep_data' in config_updates:
            self.data_logger.timestep_logging_enabled = config_updates['log_timestep_data']
        
        if 'log_fiedler_values' in config_updates:
            self.data_logger.fiedler_logging_enabled = config_updates['log_fiedler_values']
        
        if 'save_adjacency_matrices' in config_updates:
            self.data_logger.save_adjacency_matrices = config_updates['save_adjacency_matrices']
        
        return {
            "message": "Data logging configuration updated",
            "current_config": self.data_logger.get_stats()
        }       
    
    
    #=============== DYNAMIC CRITICAL AREAS MANAGEMENT ================
    
    def _manage_critical_areas(self):
        """
        Manage dynamic critical areas based on current timestep.
        Uses existing danger_regions system - just adds/removes at specific timesteps.
        """
        if not self.critical_areas_enabled or not self.critical_areas_definitions:
            return
        
        # Check each critical area definition
        for area_def in self.critical_areas_definitions:
            area_id = area_def['id']
            add_timestep = area_def.get('add_timestep', 0)
            remove_timestep = area_def.get('remove_timestep', float('inf'))
            
            # Check if we should add this area
            if (self.current_timestep == add_timestep and 
                area_id not in self.active_critical_areas):
                bounds = area_def['bounds']
                description = area_def.get('description', f'Critical area {area_id}')
                
                result = self.add_danger_region(
                    x_min=bounds['x_min'],
                    y_min=bounds['y_min'],
                    x_max=bounds['x_max'],
                    y_max=bounds['y_max'],
                    region_id=area_id
                )
                
                if "error" not in result:
                    self.active_critical_areas.add(area_id)
                    print(f"CRITICAL AREA ADDED at timestep {self.current_timestep}: {area_id} - {description}")
                    print(f"   Bounds: ({bounds['x_min']:.2f}, {bounds['y_min']:.2f}) to ({bounds['x_max']:.2f}, {bounds['y_max']:.2f})")
                else:
                    print(f"Failed to add critical area {area_id}: {result['error']}")
            
            # Check if we should remove this area
            elif (self.current_timestep == remove_timestep and 
                  area_id in self.active_critical_areas):
                result = self.remove_danger_region(area_id)
                
                if "error" not in result:
                    self.active_critical_areas.discard(area_id)
                    print(f" CRITICAL AREA REMOVED at timestep {self.current_timestep}: {area_id}")
                else:
                    print(f" Failed to remove critical area {area_id}: {result['error']}")
    
    
    def get_critical_areas_status(self):
        """
        Get status of all critical areas and upcoming changes.
        Uses existing danger_regions system for current status.
        
        Returns:
            dict: Status information about critical areas
        """
        if not self.critical_areas_enabled:
            return {"enabled": False, "message": "Critical areas system disabled"}
        
        status = {
            "enabled": True,
            "current_timestep": self.current_timestep,
            "active_critical_areas": list(self.active_critical_areas),
            "total_defined": len(self.critical_areas_definitions),
            "total_danger_regions": len(self.danger_regions),  # Use existing danger_regions
            "upcoming_changes": []
        }
        
        # Check for upcoming changes in next 10 timesteps
        for area_def in self.critical_areas_definitions:
            area_id = area_def['id']
            add_timestep = area_def.get('add_timestep', 0)
            remove_timestep = area_def.get('remove_timestep', float('inf'))
            
            # Check for upcoming additions
            if (add_timestep > self.current_timestep and 
                add_timestep <= self.current_timestep + 10):
                status["upcoming_changes"].append({
                    "type": "add",
                    "area_id": area_id,
                    "timestep": add_timestep,
                    "in_steps": add_timestep - self.current_timestep
                })
            
            # Check for upcoming removals
            if (remove_timestep > self.current_timestep and 
                remove_timestep <= self.current_timestep + 10):
                status["upcoming_changes"].append({
                    "type": "remove", 
                    "area_id": area_id,
                    "timestep": remove_timestep,
                    "in_steps": remove_timestep - self.current_timestep
                })
        
        return status
    
    def reset_critical_areas_timestep(self, new_timestep=0):
        """
        Reset the timestep counter for critical areas (useful for testing).
        
        Args:
            new_timestep: New timestep value (default: 0)
        """
        old_timestep = self.current_timestep
        self.current_timestep = new_timestep
        print(f"Critical areas timestep reset: {old_timestep} → {new_timestep}")


    #=============== SECTOR GETTERS ================

    def get_sector_info(self):
        """
        Get information about all sectors.
        
        Returns:
            dict: All sectors information with name, center, bounds, and agents in each sector
        """
        if not hasattr(self, 'sectors') or not self.sectors:
            return {"error": "No sectors available"}
        
        # Return all sectors
        result = {}
        for name, data in self.sectors.items():
            result[name] = {
                'name': name,
                'center': data['center'],
                'bounds': data['bounds'],
                'agents': self._get_agents_in_sector_by_name(name)
            }
        return result

    def _get_sector_name_from_position(self, position):
        """
        Helper method to get sector name from position.
        
        Args:
            position: [x, y] position in world coordinates
            
        Returns:
            str or None: Sector name if position is within map bounds, None otherwise
        """
        if hasattr(self.env, 'get_sector_from_position'):
            sector_id = self.env.get_sector_from_position(position)
            if sector_id is not None:
                # Find the sector with this ID
                for sector_name, sector_info in self.sectors.items():
                    if sector_info['id'] == sector_id:
                        return sector_name
        return None
    
    def _get_agents_in_sector_by_name(self, sector_name):
        """
        Helper method to get agents in a specific sector by name.
        Only includes agents with discoverability = True.
        
        Args:
            sector_name: Name of the sector (e.g., 'a1', 'b2')
            
        Returns:
            list: List of agent IDs and types in the specified sector (only discoverable agents)
        """
        agents_in_sector = []
        for agent in self.agents:
            # Only include discoverable agents
            if not getattr(agent, 'discoverability', True):
                continue
                
            agent_sector = self._get_sector_name_from_position(agent.state[:2])
            if agent_sector == sector_name:
                agents_in_sector.append({
                    'id': agent.agent_id,
                    'type': agent.type,
                    'position': [float(agent.state[0]), float(agent.state[1])]
                })
        return agents_in_sector

    
        
