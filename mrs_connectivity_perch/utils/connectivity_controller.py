import numpy as np
from numpy.linalg import inv
from scipy.linalg import eig

"""
The original code structure is adapted from the work of Sabbatini, 2013.
The code is adapted from part of MATLAB codebase at https://github.com/Cinaraghedini/adaptive-topology
Battery awareness is included.
See code documentation and associated work for details.
"""


class ConnectivityController:

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

    def __call__(self, agent_idx, agent_position, neighbors, A, id_to_index):
        """
        Computes the control input for an agent based on its position, neighbors, and adjacency matrix.

        Args:
            agent_idx (int): Index of the agent for which to compute the control input.
            agent_position (ndarray): Current position of the agent (shape: (2,)).
            neighbors (list): List of neighboring agents.
            A (ndarray): Adjacency matrix of the multi-agent network.
            id_to_index (dict): Mapping from agent IDs to indices in the adjacency matrix.

        Returns:
            ndarray: Control input vector for the agent (shape: (2,)).
        """

        if A.size == 0:
            return np.array([0, 0])

        # Compute Fiedler value and vector
        fiedler_value = self.algebraic_connectivity(A)  # Update if adjacency is required
        fiedler_vector = self.compute_eig_vector(A)  # Update if adjacency is required

        # Initialize control input
        control_vector = np.zeros(2)

        # Compute position differences and control contribution
        for agent in neighbors:

            neighbor_position = agent.get_pos()
            neighbor_id = id_to_index[agent.get_id()]

            dx = agent_position[0] - neighbor_position[0]
            dy = agent_position[1] - neighbor_position[1]

            # print(A.shape)
            # print(agent.agent_id)
            # print(fiedler_vector.shape)

            # Compute the interaction gain
            if fiedler_value > self.params['epsilon']:
                k = (-(1 / (self.params['sigma'] ** 2)) *
                     (self.csch(fiedler_value - self.params['epsilon']) ** 2)) * (
                        ((fiedler_vector[id_to_index[agent_idx]] - fiedler_vector[neighbor_id]) ** 2))
            else:
                k = -(1 / (self.params['sigma'] ** 2)) * 100 * (
                    ((fiedler_vector[id_to_index[agent_idx]] - fiedler_vector[neighbor_id]) ** 2))

            # Accumulate control contributions
            if agent.type == 'robot' or agent.type == 2:
                control_vector[0] += k * dx * 2
                control_vector[1] += k * dy * 2
            else:
                control_vector[0] += k * dx 
                control_vector[1] += k * dy

        # Scale control input
        control_vector = control_vector * self.params['gainConnectivity'] \
                         + self.calculate_repulsion_forces(agent_position, neighbors)

        return np.clip(control_vector, -0.1, 0.1)

    def calculate_repulsion_forces(self, agent_position, neighbor_positions):
        """
        Calculates the repulsion forces acting on an agent due to nearby agents.

        Args:
            agent_position (ndarray): Position of the agent (shape: (2,)).
            neighbors (list): List of neighboring agents.

        Returns:
            ndarray: Repulsion vector (shape: (2,)).
        """

        threshold = self.params['repelThreshold']
        repulsion_strength = self.params['gainRepel']
        repulsion_vector = np.zeros(2)

        for agent in neighbor_positions:
            neighbor_position = agent.get_pos()
            difference_vector = agent_position - neighbor_position
            distance = np.linalg.norm(difference_vector)
            if 0 < distance < threshold:  # Avoid division by zero
                unit_vector = difference_vector / distance
                repulsion_vector += unit_vector * repulsion_strength

        return repulsion_vector

    def csch(self, x):
        """Computes the hyperbolic cosecant of x."""
        return 1.0 / np.sinh(x)

    def degree(self, A):
        """Compute the degree matrix of adjacency matrix A."""
        return np.diag(np.sum(A, axis=1))

    def algebraic_connectivity(self, A):
        """ Calculates the algebraic connectivity (Fiedler value) of the adjacency matrix. """

        D = self.degree(A)
        if np.all(np.diag(D) != 0):
            L = D - A
            if self.params['normalized']:
                D_inv_sqrt = inv(np.sqrt(D))
                L = D_inv_sqrt @ L @ D_inv_sqrt
            eValues, _ = eig(L)
            eValues = np.sort(eValues.real)
            ac = eValues[1]
        else:
            ac = 0
        return ac

    def compute_eig_vector(self, A):
        """
        Computes the eigenvector corresponding to the second-smallest eigenvalue (Fiedler value) of the Laplacian matrix.

        Parameters:
        A (ndarray): Adjacency matrix.

        Returns:
        ndarray: Eigenvector corresponding to the second-smallest eigenvalue.
        """
        D = self.degree(A)
        L = D - A
        if self.params['normalized']:
            D_inv_sqrt = inv(np.sqrt(D))
            L = D_inv_sqrt @ L @ D_inv_sqrt
        eValues, eVectors = eig(L)
        Y = np.argsort(eValues.real)
        v = eVectors[:, Y[1]]
        return v.real

    def clip(self, velocities):
        """ Clips velocities to ensure they do not exceed the maximum allowed velocity. """
        magnitudes = np.linalg.norm(velocities, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale_factors = np.where(magnitudes > self.params['v_max'], self.params['v_max'] / magnitudes, 1)
        scaled_velocities = velocities * scale_factors[:, np.newaxis]
        return scaled_velocities

    def battery_gain(self, b):
        """
        Computes the battery gain based on the current battery level.

        Parameters:
        b (float): Current battery level.

        Returns:
        float: Battery gain.
        """
        return np.exp((self.critical_battery_level - b) / self.params['tau']) + 1
