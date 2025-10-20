from logging import config
from mrs_connectivity_perch.utils.agent_base import Agent
import numpy as np

class Robot(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = kwargs.get('config', {})
        self.projection_velocity_factor = config.get('projection_velocity_factor', 6.0)
