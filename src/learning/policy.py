import numpy as np

class Policy:
    @staticmethod
    def epsilon_greedy(q_values, epsilon, n_actions):
        if np.random.rand() < epsilon:
            return np.random.randint(0, n_actions)
        
        else:
            max_value = np.max(q_values)
            best_actions = np.flatnonzero(q_values == max_value)
            return np.random.choice(best_actions)

    @staticmethod
    def boltzmann(q_values, tau):
        exp_values = np.exp(q_values / tau)
        probs = exp_values / np.sum(exp_values)
        return np.random.choice(len(q_values), p=probs)