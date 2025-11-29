# Grid
GRID_WIDTH = 10
GRID_HEIGHT = 10

# Possible Agent moves
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
NUM_ACTIONS = len(ACTIONS)

# Params of Q-learningu aka Reinforcement Learning

# Formula: Q(s,a) <- Q(s,a) + alpha * (...) 
ALPHA = 0.1

GAMMA = 0.9

EPSILON_START = 1.0       
EPSILON_MIN = 0.01        
EPSILON_DECAY = 0.995 # 0.005

REWARD_GOAL = 100        
REWARD_COLLISION = -10    
REWARD_STEP = -1          