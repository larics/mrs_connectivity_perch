import numpy as np
from numpy.linalg import inv
from scipy.linalg import eig

"""
The original code structure is adapted from the work of Sabbatini, 2013.
The code is adapted from part of MATLAB codebase at https://github.com/Cinaraghedini/adaptive-topology
Battery awareness is included.
See code documentation and associated work for details.
"""


class PerchController:

    def __init__(self, params):
        """
        Initialize the KConnectivityController with the given parameters.

        Parameters:
        params (dict): Configuration parameters for the controller.
        """
        self.params = params
        self.fiedler_value = None
        self.fiedler_vector = None
        self.critical_battery_level = self.params['critical_battery_level']

        self.assigned_perching_id = None

    def __call__(self, agent_idx, agent_position, neighbors, A, id_to_index, curr_agent):
        """
        Computes the control input for an agent based on connectivity and repulsion from neighbors.

        Args:
            agent_idx (int): Index of the agent for which to compute the control input.
            agent_position (ndarray): Position of this agent (shape: (2,)).
            neighbors (list): List of neighboring Agent objects.
            A (ndarray): Adjacency matrix.
            id_to_index (dict): Maps agent IDs to indices in A.

        Returns:
            ndarray: Control input vector (2D).
        """

        if A.size == 0:
            return np.array([0, 0])

        # Compute Fiedler vector and value
        fiedler_value = self.algebraic_connectivity(A)
        fiedler_vector = self.compute_eig_vector(A)

        control_vector = np.zeros(2)

        if curr_agent.battery < 0.41 and not self.assigned_perching_id:
            self.find_free_perching(curr_agent, neighbors, id_to_index)

        for agent in neighbors:
            neighbor_position = agent.get_pos()
            neighbor_id = id_to_index[agent.get_id()]
            dx, dy = agent_position - neighbor_position
            dist = np.linalg.norm([dx, dy])


            if curr_agent.mode == 'perch' and self.will_become_disconnected_next(curr_agent):
                self.mode = 'active'
                self.occupied_by = None
                self.occupied_by_agent = None


            if agent.type == 'perching_pos' and curr_agent.mode != 'perch' and agent.agent_id == self.assigned_perching_id:
            
                if dist>0.12:
                    if curr_agent.battery >= 0.4:
                        k_conn = (-(1 / self.params['sigma'] ** 2) * \
                                self.csch(fiedler_value - self.params['epsilon']) ** 2) * \
                                    (50*self.csch(max(curr_agent.battery, 1e-3))+1) * self.perch_gain(curr_agent.battery, agent.battery) * \
                                    (fiedler_vector[id_to_index[agent_idx]] - fiedler_vector[neighbor_id]) ** 2
                    else:
                        if agent.agent_id == self.assigned_perching_id:
                            k_conn = -(1 / self.params['sigma'] ** 2) * self.perch_gain(curr_agent.battery, agent.battery)

                    # print(f"{k_conn, dx, dy}")
                    # print("------------")

                    control_vector[0] += k_conn * dx
                    control_vector[1] += k_conn * dy
                else:
                    curr_agent.mode = 'perch'
                    agent.occupied_by = curr_agent.agent_id
                    agent.occupied_by_agent = curr_agent

                # print(f"Battery: {curr_agent.battery}, Perch Gain: ", k_conn)

            else:
                if curr_agent.mode != 'perch':
                    if fiedler_value > self.params['epsilon']:
                        k_conn = (-(1 / self.params['sigma'] ** 2) *
                                self.csch(fiedler_value - self.params['epsilon']) ** 2) * \
                                (fiedler_vector[id_to_index[agent_idx]] - fiedler_vector[neighbor_id]) ** 2
                    else:
                        k_conn = -(1 / self.params['sigma'] ** 2) * 100 * \
                                (fiedler_vector[id_to_index[agent_idx]] - fiedler_vector[neighbor_id]) ** 2

                    # print(f"{k_conn, dx, dy}")
                    # print("------------")

                    control_vector[0] += k_conn * dx
                    control_vector[1] += k_conn * dy

        
                    # --- Repulsion control (moved inside loop) ---
                    if 0 < dist < self.params['repelThreshold']:
                        repulsion_strength = self.params['gainRepel']
                        unit = np.array([dx, dy]) / dist
                        control_vector += unit * repulsion_strength

                    # print(f"Battery: {curr_agent.battery}, Connectivity Gain: ", k_conn)



        # Final scaling
        control_vector *= self.params['gainConnectivity']

        return np.clip(control_vector, -0.1, 0.1)


    def will_become_disconnected_next(self, agent, safety_factor=6.0, dt=None):
        """
        Predict if the UAV will become disconnected from all neighbors in the next timestep,
        given their current velocities. If so, return True (should unperch), else False.
        """
        if dt is None:
            dt = agent.dt
        my_next_pos = agent.state[:2]  # Perching agent does not move
        for neighbor in agent.neighbors:
            # Predict neighbor's next position
            if hasattr(neighbor, 'state') and len(neighbor.state) >= 4:
                n_pos = neighbor.state[:2]
                n_vel = neighbor.state[2:]
                n_next_pos = n_pos + n_vel * dt * safety_factor
            else:
                n_next_pos = neighbor.get_pos()  # fallback: no velocity info
            dist = np.linalg.norm(my_next_pos - n_next_pos)
            if dist <= agent.vis_radius:
                return False 
        return True  

    def find_free_perching(self, curr_agent, neighbors, id_to_index):
        pos = curr_agent.get_pos()
        min_dist = float('inf')
        closest_id = None

        for i, agent in enumerate(neighbors):
            if agent.type == 'perching_pos':
                if agent.occupied_by is None and agent.reserved_by is None:
                    perch_pos = agent.get_pos()
                    dist = np.linalg.norm(np.array(pos) - np.array(perch_pos))
                    if dist < min_dist:
                        min_dist = dist
                        closest_id = i
                else:
                    if agent.occupied_by_agent.battery >= 0.8:
                        perch_pos = agent.get_pos()
                        dist = np.linalg.norm(np.array(pos) - np.array(perch_pos))
                        if dist < min_dist:
                            min_dist = dist
                            closest_id = i
        
        if closest_id is not None:
            self.assigned_perching_id = neighbors[closest_id].agent_id
            neighbors[closest_id].occupied_by = curr_agent.agent_id
            neighbors[closest_id].occupied_by_agent = curr_agent
            neighbors[closest_id].reserved_by = curr_agent.agent_id

            print(f"agent {curr_agent.agent_id} assigned {self.assigned_perching_id}")


    def csch(self, x):
        """Computes the hyperbolic cosecant of x."""
        return 1.0 / np.sinh(x)

    def degree(self, A):
        """Compute degree matrix from adjacency matrix A."""
        # Ensure A is 2D
        A = np.atleast_2d(A)
        
        # Handle edge case where A is a single element
        if A.shape == (1, 1):
            return np.array([[A[0, 0]]])
        
        return np.diag(np.sum(A, axis=1))

    def algebraic_connectivity(self, A):
        """Compute the algebraic connectivity (Fiedler value) of the graph."""
        # Ensure A is 2D
        A = np.atleast_2d(A)
        
        # Handle edge cases
        if A.shape[0] <= 1:
            return 0.0  # Single node or empty graph has zero connectivity
        
        # Compute Laplacian matrix
        D = self.degree(A)
        L = D - A
        
        # Handle 2x2 case separately (common edge case)
        if L.shape[0] == 2:
            # For 2x2 Laplacian, second eigenvalue is sum of diagonal minus trace
            eigenvals = np.linalg.eigvals(L)
            eigenvals = np.sort(np.real(eigenvals))
            return eigenvals[1] if len(eigenvals) > 1 else 0.0
        
        # General case: compute eigenvalues and return second smallest
        try:
            eigenvals = np.linalg.eigvals(L)
            eigenvals = np.sort(np.real(eigenvals))
            
            # The algebraic connectivity is the second smallest eigenvalue
            # (the smallest should be approximately 0)
            if len(eigenvals) > 1:
                return eigenvals[1]
            else:
                return 0.0
                
        except np.linalg.LinAlgError:
            # If eigenvalue computation fails, return 0
            return 0.0

    def compute_eig_vector(self, A):
        """
        Computes the eigenvector corresponding to the second-smallest eigenvalue (Fiedler value) of the Laplacian matrix.

        Parameters:
        A (ndarray): Adjacency matrix.

        Returns:
        ndarray: Eigenvector corresponding to the second-smallest eigenvalue.
        """
        # Ensure A is 2D
        A = np.atleast_2d(A)
        
        # Handle edge cases
        if A.shape[0] <= 1:
            # For single node or empty graph, return a zero vector
            return np.zeros(A.shape[0])
        
        # Ensure A is square
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Expected square adjacency matrix, got shape {A.shape}")
        
        try:
            D = self.degree(A)
            L = D - A
            
            # Handle normalization if required
            if self.params['normalized']:
                # Check if D is invertible (no isolated nodes)
                D_diag = np.diag(D)
                if np.any(D_diag <= 0):
                    # If there are isolated nodes, skip normalization
                    pass
                else:
                    D_inv_sqrt = np.diag(1.0 / np.sqrt(D_diag))
                    L = D_inv_sqrt @ L @ D_inv_sqrt
            
            # Compute eigenvalues and eigenvectors
            eValues, eVectors = eig(L)
            
            # Sort by real part of eigenvalues
            Y = np.argsort(eValues.real)
            
            # Return the eigenvector corresponding to the second smallest eigenvalue
            if len(Y) > 1:
                v = eVectors[:, Y[1]]
                return v.real
            else:
                # Only one eigenvalue - return zero vector
                return np.zeros(A.shape[0])
                
        except (np.linalg.LinAlgError, ValueError) as e:
            # If eigenvalue computation fails, return zero vector
            print(f"Warning: Eigenvector computation failed: {e}")
            return np.zeros(A.shape[0])

    def clip(self, velocities):
        """ Clips velocities to ensure they do not exceed the maximum allowed velocity. """
        magnitudes = np.linalg.norm(velocities, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale_factors = np.where(magnitudes > self.params['v_max'], self.params['v_max'] / magnitudes, 1)
        scaled_velocities = velocities * scale_factors[:, np.newaxis]
        return scaled_velocities

    # def battery_gain(self, b):
    #     """
    #     Computes the battery gain based on the current battery level.

    #     Parameters:
    #     b (float): Current battery level.

    #     Returns:
    #     float: Battery gain.
    #     """
    #     return np.exp((self.critical_battery_level - b) / self.params['tau']) + 1
    
    def perch_gain(self, b_i, b_j,f=1.3):
        exponent = (b_j - f*b_i) / self.params['tau']
        gain = (1 - b_i) * (np.exp(exponent) / (1 + np.exp(exponent)))
        return gain

