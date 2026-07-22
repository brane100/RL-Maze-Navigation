import random
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maze_env import load_maze

env = load_maze("mazes/test.txt")
env.reset()
env.render()

total = 0.0
for _ in range(100):
    action = random.randint(0, 3)
    state, reward, done = env.step(action)
    total += reward
    if done:
        break

env.render()
print("steps:", env.steps)
print("total reward:", round(total, 3))
print("reached goal:", env.agent_pos == env.goal)