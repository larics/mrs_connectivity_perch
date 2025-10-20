from mrs_connectivity_perch.utils.agent_base import Agent

class Base(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = 'base'
        self.discoverability = True