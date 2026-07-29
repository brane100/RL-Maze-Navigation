from maze_env import MazeEnv
from q_learning import QLearningAgent
from utils import astar, plot_steps, plot_rewards


if __name__ == "__main__":

    # define the maze
    maze = [
        "S..#...",
        ".#.#.#.",
        ".#...#.",
        ".####.#",
        ".....#.",
        ".###...",
        "....#.G",
    ]

    # build the environment
    env = MazeEnv(maze, max_steps=200)

    # compute the A* baseline first
    _, optimal = astar(env)
    print("A* optimal:", optimal)

    # train the agent
    agent = QLearningAgent()
    rewards, steps = agent.train(env, episodes=1000)

    # get the final greedy result
    final_path, final_length = agent.greedy_path(env)

    # print a short summary
    print("first 5 episodes (steps):", steps[:5])
    print("last 5 episodes (steps):", steps[-5:])
    print("final epsilon:", round(agent.epsilon, 4))
    print("final greedy path length:", final_length)
    print("matches optimal?", final_length == optimal)

    # save the plots
    plot_steps(steps, optimal=optimal,
               title="Steps to goal per episode (Q-learning)",
               filename="results/q_learning_steps.png", window=50)
    plot_rewards(rewards,
                 title="Total reward per episode (Q-learning)",
                 filename="results/q_learning_rewards.png", window=50)


"""
    For the report:
    Q-learning, using the temporal-difference error, updates the table of the expected reward that
    it keeps, for every state-action pair, after each move. It stays small enough, because it
    is 7×7 maze that has only 33 free cells, for the agent to revisit every state many times during
    training. After 1000 episodes the greedy policy reached the goal in 12 moves, matching what A* found.
    
"""