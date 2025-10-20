from mrs_connectivity_perch.utils.agent_base import Agent
from mrs_connectivity_perch.utils.perching_pos import PerchingPos

import numpy as np

class UAV(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.assigned_perching_agent = None

        self.controller_mode = 'connectivity_controller'  # Default mode

        self.max_vel = 0.05
        
        config = kwargs.get('config', {})

        self.controller_mode = config.get('initial_mode', 'connectivity_controller')

        self.discoverability = True

        self.perch_dist_sensitivity = config.get('perch_dist_sensitivity', 0.03)
        self.unperch_dist_sensitivity = config.get('unperch_dist_sensitivity', 0.15)

        # Connectivity prediction parameters
        self.projection_velocity_factor = config.get('projection_velocity_factor', 6.0)
        
        self.battery_switch_threshold = config.get('battery_switch_threshold', 0.5)

        self.battery_switch_differential = config.get('battery_switch_differential', 0.5)


    def run_controller(self, swarm):
        prev_pos = self.state[:2].copy()

        if self.controller_type == 'connectivity_controller':
            self._run_perch_controller(swarm)

            # Update velocity after movement
            self.state[2:] = (self.state[:2] - prev_pos) / self.dt      
        
        else:
            self.controller_mode = 'connectivity'  # Default for other controllers
            super().run_controller(swarm)
            if self.assigned_perching_agent is not None:
                self._check_detach()
            if self.controller_type == 'do_not_move':
                self._update_battery()
        

    def _run_perch_controller(self, swarm):

        if self.controller_mode == 'connectivity_controller':
            self._connectivity_controller_mode(swarm)

        elif self.controller_mode == 'perch_seek':
            self._perch_seek_mode()

        elif self.controller_mode == 'perched':
            self._perched_mode()

    #________________________  Main controller modes  _________________________

    def _connectivity_controller_mode(self, swarm):

        self.run_connectivity_controller(swarm) 
        
        self._update_battery()  

        # Check if battery is critical
        if self.battery <= self.battery_switch_threshold:

            # Search perching position if not already assigned
            if not self.assigned_perching_agent:
                assigned_platform = self.search_perch_platform() 

                if assigned_platform is not None:
                    self.transition_to_mode('perch_seek', context=assigned_platform)


    def _perch_seek_mode(self):

        # Check if the assigned perching position is still a neighbor
        # if self.assigned_perching_agent not in self.neighbors:
        #     self.transition_to_mode('connectivity_controller')

        self.run_perch_seek_controller()

        self._update_battery()

        # TODO: Is flying neighbor critical?
        

        # Check if we are near the platform to perch
        distance = np.linalg.norm(np.array(self.get_pos()) - np.array(self.assigned_perching_agent.get_pos()))
        if distance < self.perch_dist_sensitivity:  # Close enough to perch
            self.transition_to_mode('perched')


    def _perched_mode(self):

        # check if a switching neighbor is nearby (highest priority)
        if self.assigned_perching_agent.reserved_by is not None:
            distance = np.linalg.norm(np.array(self.get_pos()) - np.array(self.assigned_perching_agent.reserved_by.get_pos()))
            if distance < self.unperch_dist_sensitivity:
                self.transition_to_mode('connectivity_controller')
        
        
        # check if network is about to disconnect
        if self.will_become_disconnected_next(check_neighbors=True):
            self.transition_to_mode('connectivity_controller')


        # check if any neighbor is critical battery and about to be removed
        if self.has_low_battery_neighbor(battery_threshold=0.075):
            self.transition_to_mode('connectivity_controller')

    #________________________  Mode transition handlers  _______________________

    def transition_to_mode(self, new_mode, context=None):
        """
        Handle transitions between controller modes.
        connectivity_to_perchseek: context = assigned_perching_agent
        perchseek_to_perched: context = None
        perched_to_connectivity: context = None
        """

        if new_mode == self.controller_mode:
            return
            
        # connectivity_to_perchseek
        if self.controller_mode == 'connectivity_controller' and new_mode == 'perch_seek':
            context.reserved_by = self
            self.assigned_perching_agent = context
            self.assigned_perching_agent.battery = self.battery

        # perchseek_to_perched
        elif self.controller_mode == 'perch_seek' and new_mode == 'perched':
            # clear reservation
            self.assigned_perching_agent.reserved_by = None
            # mark occupation
            self.assigned_perching_agent.occupied_by = self
            # update battery
            self.assigned_perching_agent.battery = self.battery

        # perched_to_connectivity
        elif self.controller_mode == 'perched' and new_mode == 'connectivity_controller':
            # Clear assignment and occupation
            self.assigned_perching_agent.battery = 1.0
            self.assigned_perching_agent.occupied_by = None
            self.assigned_perching_agent = None


        # Update controller mode
        self.controller_mode = new_mode

    #________________________  Controller implementations  _______________________

    def run_connectivity_controller(self, swarm):
        # Compute adjacency matrix and connectivity metrics
        allowed = {('UAV', 'UAV'), ('robot', 'UAV'), ('base', 'UAV'), ('UAV', 'perching_pos')}
        A, id_to_index = self.compute_adjacency(allowed_edge_types=allowed)
        fiedler_value = self.controller.algebraic_connectivity(A)
        fiedler_vector = self.controller.compute_eig_vector(A)
        
        # Compute connectivity control vector
        control_vector = self._compute_connectivity_forces(id_to_index, fiedler_value, fiedler_vector)
        
        # Apply scaling and velocity limits (optimized: combined operations)
        control_vector *= self.controller.params['gainConnectivity']
        control_vector = np.clip(control_vector, -self.max_vel, self.max_vel)
        
        # Calculate proposed position
        proposed_position = self.state[:2] + self.controller.params['speed'] * control_vector * self.dt
        
        # Obstacle avoidance if enabled (optimized: early return if not enabled)
        if getattr(self, 'is_obstacle_avoidance', 0):
            # Use danger region aware obstacle avoidance if available
            obstacle_fn = getattr(swarm, 'is_line_of_sight_free_obstacle_avoidance_fn', None)
            if obstacle_fn is not None:
                free_path_fn = obstacle_fn
            else:
                free_path_fn = lambda pos1, pos2: swarm.is_line_of_sight_free_fn(pos1, pos2, use_dilated=True)
            
            proposed_position = self.obstacle_avoidance(
                proposed_position=proposed_position,
                is_free_path_fn=free_path_fn
            )
        
        # Update position and path history
        self.state[:2] = proposed_position
        self.update_path_history(np.copy(self.state[:2]))        


    def run_perch_seek_controller(self):

        allowed = {('UAV', 'UAV'), ('robot', 'UAV'), ('base', 'UAV'), ('UAV', 'perching_pos')}
        A, id_to_index = self.compute_adjacency(allowed_edge_types=allowed)

        control_vector = self._compute_perch_seek_forces()

        control_vector *= self.controller.params['gainConnectivity']
        control_vector = np.clip(control_vector, -self.max_vel, self.max_vel)
        proposed_position = self.state[:2] + self.controller.params['speed'] * control_vector * self.dt

        # Update position and path history
        self.state[:2] = proposed_position
        self.update_path_history(np.copy(self.state[:2]))   


    #________________________  utils  _____________________

    def _compute_connectivity_forces(self, id_to_index, fiedler_value, fiedler_vector):
        """
        Compute connectivity-based control forces for maintaining network connectivity.
        
        Args:
            id_to_index: Dictionary mapping agent IDs to adjacency matrix indices
            fiedler_value: Algebraic connectivity (second smallest eigenvalue of Laplacian)
            fiedler_vector: Eigenvector corresponding to fiedler_value
            
        Returns:
            np.array: 2D control vector representing connectivity forces
        """
        control = np.zeros(2)
        for agent in self.neighbors:
            neighbor_position = agent.get_pos()
            neighbor_id = id_to_index[agent.get_id()]
            dx, dy = self.state[:2] - neighbor_position
            dist = np.linalg.norm([dx, dy])
            if fiedler_value > self.controller.params['epsilon']:
                k_conn = (-(1 / self.controller.params['sigma'] ** 2) *
                        self.controller.csch(fiedler_value - self.controller.params['epsilon']) ** 2) * \
                        (fiedler_vector[id_to_index[self.agent_id]] - fiedler_vector[neighbor_id]) ** 2
            else:
                k_conn = -(1 / self.controller.params['sigma'] ** 2) * 100 * \
                        (fiedler_vector[id_to_index[self.agent_id]] - fiedler_vector[neighbor_id]) ** 2

            # Double the force if the neighbor is a robot
            if getattr(agent, 'type', None) == 'robot':
                k_conn *= 2.0
                repulsion_strength = self.controller.params['gainRepel']
            elif getattr(agent, 'type', None) == 'perching_pos':
                k_conn *= 0.0
                repulsion_strength = 0.0
            else:
                repulsion_strength = self.controller.params['gainRepel']


            control[0] += k_conn * dx
            control[1] += k_conn * dy
            # Repulsion
            if 0 < dist < self.controller.params['repelThreshold']:
                unit = np.array([dx, dy]) / dist
                control += unit * repulsion_strength
        return control

    def _compute_perch_seek_forces(self):  # Add swarm parameter

        neighbor_position = self.assigned_perching_agent.get_pos()
        dx, dy = neighbor_position[0] - self.state[0], neighbor_position[1] - self.state[1]
        dist = np.linalg.norm([dx, dy])
        
        # Continue moving towards perch if not close enough
        control = np.array([dx / (dist + 1e-6), dy / (dist + 1e-6)])
        control = np.clip(control, -self.max_vel, self.max_vel)
        
        return control

    def search_perch_platform(self):
        # print(f"UAV {self.agent_id} trying to find a perching position...")
        
        # Check if any neighbor is below critical battery level
        if self.has_low_battery_neighbor(battery_threshold=0.1):
            # print(f"UAV {self.agent_id} has low battery neighbor, can't seek perching position.")
            return None

        allowed = {('UAV', 'UAV'), ('robot', 'UAV'), ('base', 'UAV'), ('UAV', 'perching_pos')}
        A, id_to_index = self.compute_adjacency(allowed_edge_types=allowed)

        pos = self.get_pos()
        best_battery_diff = self.battery_switch_differential

        # Build priority list of all viable perching positions
        perch_candidates = []
        
        for i, agent in enumerate(self.neighbors):
            if agent.type == 'perching_pos' and agent.reserved_by is None: 
                battery_diff = agent.battery - self.battery
                if battery_diff > best_battery_diff:
                    perch_pos = agent.get_pos()
                    dist = np.linalg.norm(np.array(pos) - np.array(perch_pos))
                    
                    # Calculate priority score (lower is better). Higher battery difference, Closer distance
                    priority_score = dist / (battery_diff + 1e-6)
                    
                    perch_candidates.append({
                        'agent': agent,
                        'index': i,
                        'distance': dist,
                        'battery_diff': battery_diff,
                        'priority_score': priority_score
                    })
        
        # Sort by priority
        perch_candidates.sort(key=lambda x: x['priority_score'])
        
        # print(f"UAV {self.agent_id} found {len(perch_candidates)} viable perching candidates")
        
        # Try each candidate in priority order
        for rank, candidate in enumerate(perch_candidates):
            agent = candidate['agent']
            
            # if candidate is occupied, just go there as we know it will not break connectivity
            if agent.occupied_by is not None:
                return agent

            # If candidate is not occupied, check if going there breaks connectivity
            else:
                if self.is_network_connected_without_agent(A, id_to_index[self.agent_id]):
                    return agent
                else:
                    # print(f"UAV {self.agent_id} skipping perch {agent.agent_id} - would break connectivity")
                    pass

        # No suitable perch found
        # print(f"UAV {self.agent_id} could not find any suitable perching position")
        return None
    
    # TODO: Update/refactor
    def has_low_battery_neighbor(self, battery_threshold=0.1):

        """
        Check if any neighbor is a UAV that is unperched (active mode) and has battery below threshold.
        
        Args:
            battery_threshold (float): Battery level below which UAV is considered low battery
            
        Returns:
            tuple: (bool, UAV_agent or None) - True if such neighbor exists, and the UAV agent
        """
        for neighbor in self.neighbors:
            # Check if neighbor is a UAV
            if getattr(neighbor, 'type', None) == 'UAV':
                if neighbor.controller_mode != 'perched':
                    # Check if neighbor has low battery
                    neighbor_battery = getattr(neighbor, 'battery', 1.0)
                    if neighbor_battery < battery_threshold:
                        return True
        
        return False
    
    def is_network_connected_without_agent(self, adj_matrix, agent_idx):
        # Remove the agent by deleting its row and column
        adj_mod = np.delete(adj_matrix, agent_idx, axis=0)
        adj_mod = np.delete(adj_mod, agent_idx, axis=1)

        D = np.diag(np.sum(adj_mod, axis=1))
        L = D - adj_mod

        eigvals = np.linalg.eigvalsh(L)
        eigvals = np.sort(eigvals)
        fiedler_value = eigvals[1] if len(eigvals) > 1 else 0

        return fiedler_value > 0.01
    

    # If we are close enough, attempt to perch
        # else:
        #     # Simulate being at the perch position
        #     temp_pos = self.state[:2].copy()
        #     self.state[:2] = neighbor_position  # Temporarily move to perch position
            
        #     # Check connectivity from perch position (use lower projection factor for final check)
        #     # will_disconnect = self.will_become_disconnected_next(next_control=np.zeros(2))
        #     will_disconnect = False

        #     # Restore original position
        #     self.state[:2] = temp_pos
            
        #     # if will_disconnect:
        #     #     print(f"UAV {self.agent_id} aborting perch - would disconnect network from perch position!")
                
        #     #     # Free the reservation
        #     #     if assigned_perch.reserved_by == self.agent_id:
        #     #         assigned_perch.reserved_by = None
        #     #         print(f"UAV {self.agent_id} freed reservation on perching position {assigned_id}")
                
        #     #     # Clear assignment and revert to connectivity mode
        #     #     self.assigned_perching_id = None
        #     #     self.assigned_perching_agent = None
        #     #     self.controller_mode = 'connectivity'
                
        #     #     return np.zeros(2)
            
        #     # Safe to perch
        #     self.state[:2] = neighbor_position
        #     self.mode = 'perch'
        #     self.controller_mode = 'perched'

        #     assigned_perch.occupied_by = self

        #     # Clear reservation since we're now occupied
        #     assigned_perch.reserved_by = None
            
        #     #set perching agent battery as UAV battery
        #     assigned_perch.battery = self.battery
            
        #     self.assigned_perching_agent = assigned_perch

        #     print(f"UAV {self.agent_id} successfully perched at position {assigned_id}")
