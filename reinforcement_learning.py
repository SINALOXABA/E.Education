# reinforcement_learning.py
import numpy as np
import random


# Simulating learning paths and rewards
def q_learning():
    learning_paths = [1, 2, 3]
    rewards = {
        (1, 'good'): 10,
        (1, 'average'): 5,
        (1, 'poor'): 2,
        (2, 'good'): 15,
        (2, 'average'): 8,
        (2, 'poor'): 3,
        (3, 'good'): 20,
        (3, 'average'): 10,
        (3, 'poor'): 5
    }

    # Q-learning parameters
    Q = np.zeros((len(learning_paths), len(rewards)))
    alpha, gamma, epsilon = 0.1, 0.6, 0.1

    # Simulating Q-learning
    for i in range(1000):
        current_path = random.choice(learning_paths)
        performance = random.choice(['good', 'average', 'poor'])
        reward = rewards[(current_path, performance)]
        future_rewards = np.max(Q[learning_paths.index(current_path), :])

        Q[learning_paths.index(current_path), rewards.keys().index((current_path, performance))] = \
            (1 - alpha) * Q[learning_paths.index(current_path), rewards.keys().index((current_path, performance))] + \
            alpha * (reward + gamma * future_rewards)

    return Q


if __name__ == "__main__":
    q_table = q_learning()
    print("Q-table after training:")
    print(q_table)
