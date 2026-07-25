from maze_env import MazeEnv
from q_learning import QLearningAgent
from utils import astar
from utils import plot_steps, plot_rewards

"""define the maze"""

maze = [
        "S..#...",
        ".#.#.#.",
        ".#...#.",
        ".####.#",
        ".....#.",
        ".###...",
        "....#.G",
    ]

"""build the environment"""
env = MazeEnv(maze, max_steps=200)

"""compute the a* baseline first"""
_, optimal = astar(env)
print("A* optimal:", optimal)

"""train the agent"""
agent = QLearningAgent()
rewards, steps = agent.train(env, episodes=1000)

"""get the final greedy result"""
final_greedy = agent.greedy_path(env)

"""print a short summary"""
print("first 5 episodes (steps):", steps[:5])
print("Final greedy path length:", final_greedy[1])

"""save the plots"""
plot_steps(steps, optimal=optimal, title="Steps to goal per episode (Q-learning)", filename="results/q_learning_steps.png", window=50)
plot_rewards(rewards, title="Total reward per episode (Q-learning)", filename="results/q_learning_rewards.png", window=50)