import math
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any
import numpy as np
from mrs_connectivity_perch.utils.connectivity_controller import ConnectivityController
from mrs_connectivity_perch.utils.controller import PerchController
from collections import defaultdict, deque
import numpy as np

class Agent:
    """
    Base class for all agent types in the swarm.
    """
    def __init__(
        self,
        agent_type: str,
        agent_id: int,
        init_position: np.ndarray,
        dt: float,
        vis_radius: float,
        map_resolution: float,
        config: dict,
        controller_params: dict,
        path_planner=None,
        path=None,
        goal=None,
        init_battery: float = 1.0,
        battery_decay_rate: float = 0.0,
        battery_threshold: float = 0.0,
        show_old_path: int = 0,
    ):
        self.type = agent_type
        self.agent_id = agent_id
        self.dt = dt
        self.vis_radius = vis_radius
        self.map_resolution = map_resolution
        self.mode = 'active'
        self.uav_type = "None"
        # Config
        self.is_obstacle_avoidance = config.get('obstacle_avoidance', 0)
        self.speed = config.get('speed', 0.0)
        self.sensor_radius = config.get('sensor_radius', 0.0)
        self.obstacle_radius = config.get('obs_radius', 0)
        # State
        self.state = np.zeros(4) # [x, y, vx, vy]
        self.state[:2] = init_position
        self.path = None
        self.path_idx = 0
        self.path_len = 0
        self.neighbors = []
        # Battery
        self.battery = init_battery
        self.battery_decay_rate = battery_decay_rate
        self.battery_threshold = battery_threshold
        # Debugging
        self.n_agents = None
        self.prev_n_agents = None
        self.problem = False
        # Path history
        self.old_path_len = show_old_path
        self.old_path = []
        
        self.discoverability = False

        # Store line-of-sight function reference (will be set by swarm)
        self.swarm_los_fn = None

        # Controller
        self.controller_type = config['controller_type']
        self._setup_controller(controller_params, path_planner, path, goal)
        self._validate_state()

    def set_swarm_los_function(self, los_fn):
        """Set the swarm's line-of-sight function for connectivity checks."""
        self.swarm_los_fn = los_fn

    def _setup_controller(self, controller_params, path_planner, path, goal):
        if self.controller_type == 'connectivity_controller':
            self.controller = PerchController(params=controller_params)
        elif self.controller_type == 'go_to_goal':
            self.controller = path_planner
            if self.controller is not None:
                self.controller.set_goal(goal)
                self.set_path(self.controller.plan_path(self.state[:2]))
        elif self.controller_type == 'explore':
            self.controller = path_planner
        elif self.controller_type == 'random_walk':
            self.controller = path_planner
            # Initialize random walk parameters
            self.random_walk_min_distance = controller_params.get('random_walk_min_distance', 1.0)
            self.random_walk_max_distance = controller_params.get('random_walk_max_distance', 3.0)
        elif self.controller_type == 'path_tracker':
            self.set_path(path)
        else:
            self.controller = None

    def _validate_state(self):
        assert self.state.shape == (4,), "State must be a 4D vector."
        if self.path is not None:
            assert isinstance(self.path, (list, np.ndarray)), "Path must be a list or ndarray."

    def set_path(self, path):
        self.path = path
        if path is not None and len(path) > 0:
            self.state[:2] = path[0]
            self.path_len = len(self.path)
            self.path_idx = 0

    def run_controller(self, swarm: Any):
        # Store previous position for velocity calculation
        prev_pos = self.state[:2].copy()
        
        controller_map = {
            'connectivity_controller': self._run_connectivity_controller,
            'go_to_goal': self._run_go_to_goal,
            'explore': self._run_explore,
            'path_tracker': self._run_path_tracker,
            'random_walk': self._run_random_walk,
            'dummy': self._run_dummy,
            'do_not_move': lambda swarm: None,
        }
        controller_map.get(self.controller_type, self._run_unknown_controller)(swarm)
        self._perching_logic()
        
        # Update velocity based on position change (new_pos - old_pos)
        self.state[2:4] = (self.state[:2] - prev_pos) / self.dt

    def _run_connectivity_controller(self, swarm):
        A, id_to_index = self.compute_adjacency()
        v = self.controller(self.agent_id, self.get_pos(), self.neighbors, A, id_to_index, self)
        proposed_position = self.state[:2] + self.speed * v * self.dt
        self.state[:2] = proposed_position
        self.update_path_history(deepcopy(self.state[:2]))

    def _run_connectivity_controller(self, swarm):
        A, id_to_index = self.compute_adjacency()
        v = self.controller(self.agent_id, self.get_pos(), self.neighbors, A, id_to_index)
        print(f"agent {self.agent_id}, force {v}")
        proposed_position = self.state[:2] + self.speed * v * self.dt
        if self.is_obstacle_avoidance:
            # Use obstacle avoidance function that includes danger regions if available
            obstacle_avoidance_fn = getattr(swarm, 'is_line_of_sight_free_obstacle_avoidance_fn', None)
            if obstacle_avoidance_fn is not None:
                # Use the combined obstacle+danger region function
                free_path_fn = obstacle_avoidance_fn
            else:
                # Fall back to regular line of sight function
                free_path_fn = swarm.is_line_of_sight_free_fn
            
            adjusted_position = self.obstacle_avoidance(proposed_position=proposed_position,
                                                        is_free_path_fn=free_path_fn)
            self.state[:2] = adjusted_position
        else:
            self.state[:2] = proposed_position
        self.update_path_history(deepcopy(self.state[:2]))

    def _run_go_to_goal(self, swarm):
        if self.path is not None and self.path_idx < self.path_len:
            displacement = self.path[self.path_idx] - self.state[:2]
            if np.linalg.norm(displacement) > self.speed * self.dt:
                if np.linalg.norm(displacement) != 0:
                    self.state[:2] += (displacement / np.linalg.norm(displacement)) * self.speed * self.dt
            else:
                self.state[:2] = self.path[self.path_idx]
                self.path_idx += 1

    def _run_explore(self, swarm):
        local_obs, n_angles = self.get_local_observation(swarm.is_free_space_fn)
        swarm.update_exploration_map_fn(self.state[:2], local_obs, n_angles, self.sensor_radius)
        if self.path is None:
            goal = swarm.get_frontier_goal_fn(self.state[:2])
            print("GOAL: ", goal)
            if goal is not None:
                try:
                    self.controller.set_goal(goal)
                    planned_path = self.controller.plan_path(self.state[:2])
                    if planned_path and len(planned_path) > 0:
                        self.set_path(planned_path)
                        print(f"✅ Path planned successfully with {len(planned_path)} waypoints")
                    else:
                        # Goal is unreachable - exploration complete for reachable areas
                        print("🏁 Frontier goal unreachable - exploration complete for accessible areas")
                        return
                except Exception as e:
                    print(f"❌ Path planning failed: {e}")
                    print(f"   Start: {self.state[:2]}, Goal: {goal}")
                    print(f"   Controller type: {type(self.controller).__name__}")
                    # Clear the goal and continue
                    return
            else:
                # No frontiers available - exploration complete, switch to stationary
                print("🏁 Exploration complete - no more frontiers available")
                # Stay in current position (do nothing, like do_not_move controller)
                return
        if self.path is not None and self.path_idx < self.path_len:
            displacement = self.path[self.path_idx] - self.state[:2]
            # Use dynamic threshold based on speed and dt for better exploration
            movement_threshold = max(0.05, self.speed * self.dt * 0.5)  # At least 5cm or half of max movement
            if np.linalg.norm(displacement) > movement_threshold:
                if np.linalg.norm(displacement) != 0:
                    self.state[:2] += (displacement / np.linalg.norm(displacement)) * self.speed * self.dt
            else:
                self.state[:2] = self.path[self.path_idx]
                self.path_idx += 1
        elif self.path is not None and self.path_idx >= self.path_len:
            print("path reached")
            self.path = None

    def _run_path_tracker(self, swarm):
        if self.path is not None and self.path_len > 0:
            if self.path_idx < self.path_len:
                displacement = self.path[self.path_idx] - self.state[:2]
                if np.linalg.norm(displacement) > self.speed * self.dt:
                    if np.linalg.norm(displacement) != 0:
                        self.state[:2] += (displacement / np.linalg.norm(displacement)) * self.speed * self.dt
                else:
                    self.state[:2] = self.path[self.path_idx]
                    self.path_idx += 1
            # else:
            #     # When path ends, reverse direction and go back
            #     # Reverse the path and reset index
            #     self.path = self.path[::-1]  # Reverse the path array
            #     self.path_idx = 1  # Start from index 1 (skip current position)
                
            #     # Move to next position in reversed path
            #     if self.path_idx < self.path_len:
            #         next_pos = self.path[self.path_idx]
            #         velocity = (next_pos - self.state[:2]) / self.dt
            #         self.state[2:4] = velocity  # Set velocity
            #         self.state[:2] = next_pos
            #         self.path_idx += 1

    def _run_random_walk(self, swarm):
        """
        Random walk controller: Sample random points in free space and navigate to them.
        When goal is reached, sample a new random point and repeat.
        """
        # Check if we need a new goal (no path or reached current goal)
        if self.path is None or self.path_idx >= self.path_len:
            # Sample a new random goal
            new_goal = self._sample_random_goal(swarm)
            if new_goal is not None and self.controller is not None:
                self.controller.set_goal(new_goal)
                new_path = self.controller.plan_path(self.state[:2])
                self.set_path(new_path)
        
        # Follow current path (same logic as go_to_goal)
        if self.path is not None and self.path_idx < self.path_len:
            displacement = self.path[self.path_idx] - self.state[:2]
            if np.linalg.norm(displacement) > self.speed * self.dt:
                if np.linalg.norm(displacement) != 0:
                    self.state[:2] += (displacement / np.linalg.norm(displacement)) * self.speed * self.dt
            else:
                self.state[:2] = self.path[self.path_idx]
                self.path_idx += 1
    
    def _sample_random_goal(self, swarm, max_attempts=20):
        """
        Sample a random goal in free space at least min_distance away from current position.
        Uses dilated obstacles map for safety - ensures robot can physically reach the goal.
        """
        current_pos = self.state[:2]
        min_dist = getattr(self, 'random_walk_min_distance', 1.0)
        max_dist = getattr(self, 'random_walk_max_distance', 3.0)
        
        for attempt in range(max_attempts):
            # Sample random direction and distance
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(min_dist, max_dist)
            
            # Calculate candidate position
            candidate_goal = current_pos + distance * np.array([np.cos(angle), np.sin(angle)])
            
            # Check if position is in free space using DILATED map for safety
            if swarm.is_free_space_fn(candidate_goal, use_dilated=True):
                print(f"✅ Sampled safe goal {candidate_goal} (attempt {attempt+1})")
                return candidate_goal
            else:
                print(f"❌ Goal {candidate_goal} is not in safe free space (attempt {attempt+1})")
        
        print(f"⚠️  Failed to find safe goal after {max_attempts} attempts, using closer fallback")
        # Fallback: try a much closer goal with dilated map
        for _ in range(10):
            angle = np.random.uniform(0, 2 * np.pi)
            distance = min_dist * 0.5  # Half the minimum distance as fallback
            candidate_goal = current_pos + distance * np.array([np.cos(angle), np.sin(angle)])
            
            if swarm.is_free_space_fn(candidate_goal, use_dilated=True):
                print(f"🔄 Using safe fallback goal {candidate_goal}")
                return candidate_goal
        
        return None

    def _run_dummy(self, swarm):
        self.state[0] += 1 * self.speed

    def _run_unknown_controller(self, swarm):
        raise NotImplementedError(f"Unknown controller type: {self.controller_type}")

    def _update_battery(self):
        if self.battery_decay_rate is not None and self.battery > 0:
            self.battery = max(0, self.battery - self.battery_decay_rate)

    def _perching_logic(self):
        # No-op in base class; override in PerchingPos
        pass

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def get_data(self):
        return {"id": self.agent_id, "position": self.state[:2]}

    def get_id(self):
        return self.agent_id

    def get_neighbors(self):
        return self.neighbors

    def get_neighbors_pos(self):
        positions = []
        ids = []
        for agent in self.neighbors:
            ids.append(agent.get_id())
            positions.append(agent.get_pos())
        paired = list(zip(ids, positions))
        sorted_paired = sorted(paired)
        sorted_positions = [pos for _, pos in sorted_paired]
        return np.array(sorted_positions)

    def get_pos(self):
        return self.state[:2]

    def set_pos(self, pos):
        self.state[:2] = pos

    def is_battery_critical(self):
        if self.battery_decay_rate is not None:
            if self.battery <= self.battery_threshold:
                return True
        return False

    def update_path_history(self, element):
        if len(self.old_path) >= self.old_path_len:
            self.old_path.pop(0)
        if len(self.old_path) > 0:
            distance = np.linalg.norm(element - self.old_path[-1])
            if distance > 0.1:
                self.old_path.append(element)
        elif len(self.old_path) == 0:
            self.old_path.append(element)

    def compute_adjacency(self, allowed_edge_types=None):
        """
        Computes the adjacency matrix representing the communication graph of the swarm by using BFS to traverse the swarm's agents and build an adjacency matrix.
        Optionally, only include edges of types in allowed_edge_types (set of (type1, type2) tuples).
        Returns:
            tuple:
                - adjacency_matrix (numpy.ndarray): A binary matrix where a value of 1
                  indicates a direct connection between agents.
                - id_to_index (dict): A mapping from agent IDs to their respective
                  indices in the adjacency matrix.
        """
        # DEBUG: For UAV 3, print detailed information
        # if getattr(self, 'agent_id', None) == 3:
        #     print(f"\n=== DEBUG UAV 3 compute_adjacency ===")
        #     print(f"UAV 3 position: {self.get_pos()}")
        #     print(f"UAV 3 total neighbors: {len(self.neighbors)}")
        #     print(f"Allowed edge types: {allowed_edge_types}")
        #     for i, neighbor in enumerate(self.neighbors):
        #         dist = np.linalg.norm(np.array(self.get_pos()) - np.array(neighbor.get_pos()))
        #         print(f"  Neighbor {i}: type={neighbor.type}, id={neighbor.agent_id}, pos={neighbor.get_pos()}, dist={dist:.3f}")
        
        adjacency = defaultdict(set)
        queue = deque([self])
        visited = set()
        while queue:
            current_agent = queue.popleft()
            current_info = current_agent.get_data()
            current_id = current_info["id"]
            if current_id in visited:
                continue
            visited.add(current_id)
            for neighbor in current_agent.neighbors:
                # Only add edge if allowed_edge_types is None or the edge type is allowed
                if allowed_edge_types is not None:
                    edge = (getattr(current_agent, 'type', None), getattr(neighbor, 'type', None))
                    edge_rev = (getattr(neighbor, 'type', None), getattr(current_agent, 'type', None))
                    
                    # DEBUG: For UAV 3, print edge filtering
                    # if getattr(current_agent, 'agent_id', None) == 3 and neighbor.type == 'perching_pos':
                    #     print(f"  DEBUG UAV 3: Checking edge {edge} and {edge_rev}")
                    #     print(f"    Edge in allowed: {edge in allowed_edge_types}")
                    #     print(f"    Edge_rev in allowed: {edge_rev in allowed_edge_types}")
                    #     print(f"    Will include: {edge in allowed_edge_types or edge_rev in allowed_edge_types}")
                    
                    if edge not in allowed_edge_types and edge_rev not in allowed_edge_types:
                        continue
                neighbor_info = neighbor.get_data()
                neighbor_id = neighbor_info["id"]
                adjacency[current_id].add(neighbor_id)
                adjacency[neighbor_id].add(current_id)
                if neighbor_id not in visited:
                    queue.append(neighbor)
        
        # DEBUG: For UAV 3, print final adjacency
        # if getattr(self, 'agent_id', None) == 3:
        #     print(f"  DEBUG UAV 3: Final adjacency connections: {dict(adjacency)}")
        
        all_ids = sorted(adjacency.keys())
        id_to_index = {agent_id: index for index, agent_id in enumerate(all_ids)}
        size = len(all_ids)
        matrix = [[0] * size for _ in range(size)]
        for agent_id, neighbors in adjacency.items():
            for neighbor_id in neighbors:
                i, j = id_to_index[agent_id], id_to_index[neighbor_id]
                matrix[i][j] = 1
        
        return np.array(matrix), id_to_index

    def get_local_observation(self, is_free_space_fn, n_angles=360):
        """
        Simulates a local observation using a simulated LIDAR-like sensor. It calculates the distances to obstacles in the agent's surroundings
        by casting rays in `n_angles` directions up to the agent's sensor radius.

        Args:
            is_free_space_fn (callable): A function that checks if a given position in space is free.
            n_angles (int, optional): Number of angles to sample in a 360-degree field of view. Defaults to 360.

        Returns:
            tuple:
                - local_obs (numpy.ndarray): Array of distances to obstacles for each sampled angle.
                - n_angles (int): Number of angles sampled (same as input).
        """
        local_obs = np.full(n_angles, self.sensor_radius)
        current_x, current_y = self.state[:2]
        for i in range(n_angles):
            angle = i * (2 * np.pi / n_angles)
            for r in np.linspace(0, self.sensor_radius, int(self.sensor_radius / self.map_resolution)):
                x = current_x + r * math.cos(angle)
                y = current_y + r * math.sin(angle)
                if not is_free_space_fn((x, y)):
                    local_obs[i] = r
                    break
        return local_obs, n_angles

    def obstacle_avoidance(self, proposed_position, is_free_path_fn, num_samples=4):
        """
        Adjust the proposed position to avoid collisions using free path sampling by steer to avoid methodology.

        Parameters:
        proposed_position (np.ndarray): The next position proposed by the controller.
        obstacle_radius (float): The radius within which the agent checks for obstacles.
        is_free_path_fn (function): Function to check if the path between two points is free of obstacles.
        num_samples (int): Number of directions to sample around the agent.

        Returns:
        np.ndarray: Adjusted position to avoid collisions.
        """
        current_position = self.state[:2]
        direction_to_target = proposed_position - current_position
        magnitude = np.linalg.norm(direction_to_target)
        if magnitude == 0:
            return proposed_position
        direction_to_target /= magnitude
        check_point = current_position + direction_to_target * self.obstacle_radius
        if is_free_path_fn(current_position, check_point):
            return proposed_position
        best_direction = None
        max_clear_distance = 0
        base_angles = np.linspace(0, np.pi, num_samples // 2, endpoint=False)
        angles = np.empty((num_samples,))
        angles[0::2] = base_angles
        angles[1::2] = -base_angles
        for angle in angles:
            candidate_direction = np.array([
                np.cos(angle) * self.obstacle_radius,
                np.sin(angle) * self.obstacle_radius
            ])
            candidate_position = current_position + candidate_direction
            if is_free_path_fn(current_position, candidate_position):
                clear_distance = np.linalg.norm(candidate_direction)
                if clear_distance > max_clear_distance:
                    max_clear_distance = clear_distance
                    best_direction = candidate_direction
        if best_direction is not None:
            best_direction /= np.linalg.norm(best_direction)
            best_direction *= magnitude
            adjusted_position = current_position + best_direction
            return adjusted_position
        return current_position

    def will_become_disconnected_next(self, dt=None, check_neighbors=True, check_line_of_sight=True, consider_neighbor_types={'UAV', 'robot', 'base'}, min_neighbors=2):
        """
        Predict if the UAV will become disconnected from all active UAV neighbors
        in the next timestep using the projection velocity factor.
        Additionally, ensure that none of its active UAV neighbors will become isolated either.
        Returns True if this UAV or any of its relevant neighbors will be disconnected from active UAVs.
        
        Args:
            dt: Time step
            check_neighbors: Whether to recursively check neighbors
            next_control: Control input for next step
            next_pos: Alternative way to specify next position
            check_line_of_sight: Whether to check line-of-sight for connectivity
        """

        if dt is None:
            dt = self.dt if hasattr(self, 'dt') else 1.0

        # Project forward in time (projection_velocity_factor should be time in seconds)
        projection_time = self.projection_velocity_factor * self.dt  # Convert to actual time
        my_next_pos = self.state[:2] + self.state[2:] * projection_time
        # if self.type == 'robot':

        connected_neighbors = []

        for neighbor in self.neighbors:
            if neighbor.type in consider_neighbor_types:
                if hasattr(neighbor, 'state') and len(neighbor.state) >= 4:
                    n_pos = neighbor.state[:2]
                    n_vel = np.array(neighbor.state[2:], dtype=float)
                    projection_time = self.projection_velocity_factor * self.dt  # Convert to actual time
                    n_next_pos = n_pos + n_vel * projection_time
                else:
                    n_next_pos = neighbor.get_pos()  # fallback: no velocity info

                # Check distance
                dist = np.linalg.norm(my_next_pos - n_next_pos)

                if dist <= self.vis_radius:
                    # Additional line-of-sight check if enabled
                    if check_line_of_sight:
                        if self.swarm_los_fn(my_next_pos, n_next_pos, use_dilated=False):
                            connected_neighbors.append(neighbor)
                        # If line-of-sight is blocked, don't add to connected list
                    else:
                        # No line-of-sight check, just distance-based
                        connected_neighbors.append(neighbor)

        # Print connected neighbors info
        connected_ids = [n.agent_id for n in connected_neighbors]
        connected_types = [n.type for n in connected_neighbors]
        # print(f"  Connected: {len(connected_neighbors)} neighbors - IDs: {connected_ids}, types: {connected_types}")

        # If fewer than 2 connected neighbors (any type), will become disconnected
        if len(connected_neighbors) < min_neighbors:
            return True

        if check_neighbors:
            for neighbor in self.neighbors:
                if neighbor.type != 'base' and neighbor.type != 'perching_pos':
                    if neighbor.will_become_disconnected_next(dt=dt, check_neighbors=False, 
                                                            check_line_of_sight=check_line_of_sight,
                                                            min_neighbors=1):
                        return True

        return False