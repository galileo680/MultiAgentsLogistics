import pytest
import numpy as np
import os

from src.environment.grid import GridMap
from src.environment.dynamics import EnvironmentDynamics
from src.learning.q_learning import QLearningTable
from src.learning.policy import Policy
import config

class TestGridAndDynamics:
    @pytest.fixture
    def setup_grid(self):
        #mapa 10x10 przeszkoda 2,2
        grid = GridMap(10, 10)
        grid.grid[2][2] = GridMap.OBSTACLE
        return grid

    def test_bounds_checking(self, setup_grid):
        #wyjście poza granice mapy
        assert setup_grid.is_obstacle(-1, 0) == True
        assert setup_grid.is_obstacle(0, -1) == True
        assert setup_grid.is_obstacle(10, 0) == True
        assert setup_grid.is_obstacle(0, 0) == False

    def test_obstacle_collision(self, setup_grid):
        #wejscie w przeszkodę
        dynamics = EnvironmentDynamics(setup_grid)
        start_pos = (2, 1)
        next_pos, reward, done = dynamics.step(start_pos, 1) 
        
        assert next_pos == start_pos
        assert reward < 0 

    def test_real_map_walls(self):
        grid = GridMap(10, 10)
        grid.generate_simple_map() 
        
        dynamics = EnvironmentDynamics(grid)

        start_pos = (3, 3)
        next_pos, reward, done = dynamics.step(start_pos, 3) 

        assert next_pos == start_pos 
        assert grid.is_obstacle(4, 3) == True

        def test_successful_move_reward(self, setup_grid):
            #nagrody
            dynamics = EnvironmentDynamics(setup_grid)
            start_pos = (0, 0)
            
            next_pos, reward, done = dynamics.step(start_pos, 1) 
            
            assert next_pos == (0, 1)
            assert reward == config.REWARD_MOVE

class TestQLearning:
    def test_q_table_update(self):
        #tab q
        q = QLearningTable(actions=[0, 1, 2, 3], learning_rate=0.2, reward_decay=0.5)
        state = (0, 0)
        action = 1
        reward = 100
        next_state = (0, 1)
        
        q.q_table[state][action] = 20.0
        initial_q = 20.0
        
        q.q_table[next_state][0] = 10.0
        
        expected_q = 37.0

        q.learn(state, action, reward, next_state, done=False)
        
        new_q = q.get_q_values(state)[action]
        
        assert new_q == pytest.approx(expected_q) 

    def test_save_and_load(self, tmp_path):
        #zapis i odczyt info
        save_file = tmp_path / "test_brain.npy"
        
        q1 = QLearningTable(actions=[0, 1, 2, 3])
        test_state = (5, 5)
        q1.q_table[test_state][0] = 123.456
        
        q1.save_table(str(save_file))
        
        q2 = QLearningTable(actions=[0, 1, 2, 3])
        q2.load_table(str(save_file))
        
        loaded_value = q2.get_q_values(test_state)[0]
        assert loaded_value == 123.456

class TestPolicy:
    def test_epsilon_greedy_exploitation(self):
        #epsilon=0 best move
        q_values = np.array([1.0, 0.5, 100.0, 0.2])
        n_actions = len(q_values)
        
        action = Policy.epsilon_greedy(q_values, epsilon=0.0, n_actions=n_actions)
        
        assert action == 2

    def test_epsilon_greedy_tie_breaking(self):
        #remis
        q_values = np.array([50.0, 10.0, 50.0, 5.0])
        n_actions = len(q_values)
        
        chosen_actions = set()
        for _ in range(20):
            action = Policy.epsilon_greedy(q_values, epsilon=0.0, n_actions=n_actions)
            chosen_actions.add(action)
            
        assert 1 not in chosen_actions
        assert 3 not in chosen_actions
        assert 0 in chosen_actions or 2 in chosen_actions