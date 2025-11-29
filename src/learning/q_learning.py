import numpy as np
from collections import defaultdict

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9):
        self.actions = actions
        self.lr = learning_rate     # Alpha
        self.gamma = reward_decay   # Gamma
        
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))

    def get_q_values(self, state):
        state_key = tuple(state)
        return self.q_table[state_key]

    def learn(self, state, action_idx, reward, next_state, done):
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        
        q_predict = self.q_table[state_key][action_idx]
        
        if done:
            q_target = reward
        else:
            next_q_values = self.q_table[next_state_key]
            max_next_q = np.max(next_q_values)
            
            q_target = reward + self.gamma * max_next_q

        self.q_table[state_key][action_idx] += self.lr * (q_target - q_predict)

    def save_table(self, filename="q_table.npy"):
        np.save(filename, dict(self.q_table))

    def load_table(self, filename="q_table.npy"):
        try:
            data = np.load(filename, allow_pickle=True).item()
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), data)
            print(f"Wczytano tabelę Q: {len(self.q_table)} stanów.")
        except FileNotFoundError:
            print("Nie znaleziono pliku zapisu, startuję od zera.")