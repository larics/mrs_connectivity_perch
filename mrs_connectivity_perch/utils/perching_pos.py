import numpy as np
from mrs_connectivity_perch.utils.agent_base import Agent

class PerchingPos(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.occupied_by = None
        self.occupied_by_agent = None
        self.reserved_by = None
        
        # Make perching positions visible in visualization
        self.discoverability = True

    def _perching_logic(self):
        if self.occupied_by is not None and self.occupied_by_agent is not None:
            self.battery = self.occupied_by_agent.battery

    def run_controller(self, swarm):
        """Check if reserved UAV is close enough and transition to occupied status."""
        # Check if we have a reserved UAV
        if self.reserved_by is not None:
            # Find the reserved UAV in the swarm
            reserved_uav = None
            for agent in swarm.agents:
                if agent.agent_id == self.reserved_by and agent.type == 'UAV':
                    reserved_uav = agent
                    break
            
            if reserved_uav is not None:
                # Check distance between perching position and reserved UAV
                distance = np.linalg.norm(np.array(self.state[:2]) - np.array(reserved_uav.state[:2]))
                
                if distance <= 0.15:
                    # Transition from reserved to occupied
                    self.reserved_by = None
                    self.occupied_by = reserved_uav.agent_id
                    self.occupied_by_agent = reserved_uav
                    
                    # Set UAV's assigned perching agent reference
                    reserved_uav.assigned_perching_agent = self
        
