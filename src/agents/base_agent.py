class BaseAgent:
    def __init__(self, agent_id, start_pos, label):
        self.agent_id = agent_id
        self.position = start_pos
        self.label = label
        
        self.history = []

    def get_position(self):
        return self.position

    def set_position(self, new_pos):
        self.position = new_pos

    def step(self, environment_dynamics):
        raise NotImplementedError("Każdy agent musi mieć zaimplementowaną metodę step()")

    def reset(self, start_pos):
        self.position = start_pos
        self.history = []
        
    def __str__(self):
        return f"Agent({self.agent_id}) @ {self.position}"