import numpy as np

class GridMap:
    EMPTY = 0
    OBSTACLE = 1
    GOAL = 2
    START = 3

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        
        self.start_pos = (0, 0)
        
    def generate_simple_map(self):
        self.grid.fill(self.EMPTY)
        
        for y in range(2, 8):
            self.grid[y][4] = self.OBSTACLE
            
        self.grid[2][2] = self.OBSTACLE
        self.grid[2][3] = self.OBSTACLE
        
        self.grid[0][0] = self.START
        
        target_x, target_y = self.width - 1, self.height - 1
        self.grid[target_y][target_x] = self.GOAL
        
        return self.grid

    def is_obstacle(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True 
            
        return self.grid[y][x] == self.OBSTACLE

    def get_state_space_size(self):
        return self.width * self.height

    def reset(self):
        pass
        
    def render_console(self, agent_pos=None):
        print("-" * (self.width * 2 + 2))
        for y in range(self.height):
            row_str = "|"
            for x in range(self.width):
                if agent_pos and agent_pos == (x, y):
                    row_str += "A "
                elif self.grid[y][x] == self.OBSTACLE:
                    row_str += "# "
                elif self.grid[y][x] == self.GOAL:
                    row_str += "G "
                else:
                    row_str += ". " 
            print(row_str + "|")
        print("-" * (self.width * 2 + 2))