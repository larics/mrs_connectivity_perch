import os
import numpy as np
from typing import List, Dict, Any, Optional
from scipy.linalg import eig


class SwarmDataLogger:
    """
    A separate class to handle all data logging and saving functionality for the swarm.
    This keeps the main Swarm class cleaner and allows for flexible logging configuration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data logger with configuration options.
        
        Args:
            config: Configuration dictionary containing logging settings
        """
        # Extract logging configuration
        logging_config = config.get('data_logging', {})
        
        # Core logging flags
        self.enabled = logging_config.get('enabled', False)
        self.timestep_logging_enabled = logging_config.get('log_timestep_data', True)
        self.fiedler_logging_enabled = logging_config.get('log_fiedler_values', False)
        self.save_adjacency_matrices = logging_config.get('save_adjacency_matrices', False)
        
        # Output directories
        self.timestep_data_dir = logging_config.get('timestep_data_dir', "simulation_data")
        self.fiedler_data_dir = logging_config.get('fiedler_data_dir', "fiedler_data")
        self.adjacency_data_dir = logging_config.get('adjacency_data_dir', "adjacency_matrices")
        
        # File naming
        self.timestep_filename = logging_config.get('timestep_filename', "simulation_data.npy")
        self.fiedler_filename = logging_config.get('fiedler_filename', "fiedler_list.npy")
        self.n_agents_filename = logging_config.get('n_agents_filename', "n_agents_list.npy")
        
        # Data filters
        self.log_only_uavs = logging_config.get('log_only_uavs', True)
        self.log_agent_types = logging_config.get('log_agent_types', ['UAV'])  # Which agent types to log
        
        # Internal storage
        self.simulation_data = []
        self.fiedler_list = []
        self.n_agents_list = []
        self.current_timestep = 0
        
        # Create directories if logging is enabled
        if self.enabled:
            self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories for data storage."""
        directories = []
        
        if self.timestep_logging_enabled:
            directories.append(self.timestep_data_dir)
        
        if self.fiedler_logging_enabled:
            directories.append(self.fiedler_data_dir)
        
        if self.save_adjacency_matrices:
            directories.append(self.adjacency_data_dir)
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def log_timestep_data(self, agents: List, fiedler_value: float):
        """
        Log data for the current timestep.
        
        Args:
            agents: List of all agents in the swarm
            fiedler_value: Current Fiedler value of the swarm
        """
        if not self.enabled or not self.timestep_logging_enabled:
            return
        
        # Filter agents based on configuration
        if self.log_only_uavs:
            filtered_agents = [agent for agent in agents if agent.type == 'UAV']
        else:
            filtered_agents = [agent for agent in agents if agent.type in self.log_agent_types]
        
        if not filtered_agents:
            return
        
        # Create structured array for this timestep
        n_agents = len(filtered_agents)
        timestep_data = np.zeros(n_agents, dtype=[
            ('timestep', 'i4'),
            ('agent_id', 'i4'), 
            ('battery', 'f4'),
            ('fiedler_value', 'f4'),
            ('velocity_x', 'f4'),
            ('velocity_y', 'f4'),
            ('velocity_magnitude', 'f4'),
            ('mode', 'U10'),  # String field for mode ('active', 'perch')
            ('position_x', 'f4'),
            ('position_y', 'f4'),
            ('controller_type', 'U20'),
            ('agent_type', 'U15')
        ])
        
        for i, agent in enumerate(filtered_agents):
            # Get velocity from state
            velocity = agent.state[2:4] if len(agent.state) >= 4 else [0.0, 0.0]
            velocity_magnitude = np.linalg.norm(velocity)
            
            timestep_data[i] = (
                self.current_timestep,
                agent.agent_id,
                getattr(agent, 'battery', 1.0),
                fiedler_value,
                velocity[0],
                velocity[1],
                velocity_magnitude,
                getattr(agent, 'mode', 'unknown'),
                agent.state[0],
                agent.state[1],
                getattr(agent, 'controller_type', 'unknown'),
                agent.type
            )
        
        self.simulation_data.append(timestep_data)
        self.current_timestep += 1
    
    def log_fiedler_data(self, fiedler_value: float, n_agents: int, adjacency_matrix: Optional[np.ndarray] = None):
        """
        Log Fiedler value and related connectivity data.
        
        Args:
            fiedler_value: Current Fiedler value
            n_agents: Number of agents in the swarm
            adjacency_matrix: Optional adjacency matrix to save
        """
        if not self.enabled or not self.fiedler_logging_enabled:
            return
        
        self.fiedler_list.append(fiedler_value)
        self.n_agents_list.append(n_agents)
        
        # Save adjacency matrix if enabled and provided
        if self.save_adjacency_matrices and adjacency_matrix is not None:
            filename = f"A_{int(n_agents)}_t_{self.current_timestep}.npy"
            filepath = os.path.join(self.adjacency_data_dir, filename)
            np.save(filepath, adjacency_matrix)
    
    def save_all_timestep_data(self):
        """Save all collected timestep data to file."""
        if not self.enabled or not self.timestep_logging_enabled or not self.simulation_data:
            return
        
        # Concatenate all timesteps
        all_data = np.concatenate(self.simulation_data)
        
        filepath = os.path.join(self.timestep_data_dir, self.timestep_filename)
        np.save(filepath, all_data)
        print(f" Saved simulation data to {filepath}")
        return filepath
    
    def save_all_fiedler_data(self):
        """Save all Fiedler value data to files."""
        if not self.enabled or not self.fiedler_logging_enabled:
            return
        
        if self.fiedler_list:
            fiedler_path = os.path.join(self.fiedler_data_dir, self.fiedler_filename)
            np.save(fiedler_path, np.array(self.fiedler_list))
            print(f" Saved Fiedler data to {fiedler_path}")
        
        if self.n_agents_list:
            n_agents_path = os.path.join(self.fiedler_data_dir, self.n_agents_filename)
            np.save(n_agents_path, np.array(self.n_agents_list))
            print(f" Saved agent count data to {n_agents_path}")
    
    def save_all_data(self):
        """Save all collected data to files."""
        if not self.enabled:
            return
        
        saved_files = []
        
        # Save timestep data
        timestep_file = self.save_all_timestep_data()
        if timestep_file:
            saved_files.append(timestep_file)
        
        # Save Fiedler data
        self.save_all_fiedler_data()
        
        return saved_files
    
    def clear_data(self):
        """Clear all stored data (useful for resetting between runs)."""
        self.simulation_data.clear()
        self.fiedler_list.clear()
        self.n_agents_list.clear()
        self.current_timestep = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the logged data."""
        return {
            "enabled": self.enabled,
            "timesteps_logged": len(self.simulation_data),
            "fiedler_values_logged": len(self.fiedler_list),
            "current_timestep": self.current_timestep,
            "log_timestep_data": self.timestep_logging_enabled,
            "log_fiedler_values": self.fiedler_logging_enabled,
            "save_adjacency_matrices": self.save_adjacency_matrices,
            "output_dirs": {
                "timestep_data": self.timestep_data_dir,
                "fiedler_data": self.fiedler_data_dir,
                "adjacency_data": self.adjacency_data_dir
            }
        }
