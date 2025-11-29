import config

class EnvironmentDynamics:
    def __init__(self, grid_map):
        self.grid_map = grid_map

    def get_next_state(self, current_pos, action_idx):
        x, y = current_pos
        
        if action_idx == 0:
            dy, dx = -1, 0
        elif action_idx == 1:
            dy, dx = 1, 0
        elif action_idx == 2:
            dy, dx = 0, -1
        elif action_idx == 3:
            dy, dx = 0, 1
        else:
            dy, dx = 0, 0

        new_x = x + dx
        new_y = y + dy
        
        return new_x, new_y

    def step(self, current_pos, action_idx):
        target_x, target_y = self.get_next_state(current_pos, action_idx)
         
        if self.grid_map.is_obstacle(target_x, target_y):
            next_state = current_pos
            reward = config.REWARD_COLLISION
            done = False
            
        elif self.grid_map.grid[target_y][target_x] == self.grid_map.GOAL:
            next_state = (target_x, target_y)
            reward = config.REWARD_GOAL
            done = True
            
        else:
            next_state = (target_x, target_y)
            reward = config.REWARD_STEP
            done = False
            
        return next_state, reward, done