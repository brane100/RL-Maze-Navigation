# Maze Navigation Using Reinforcement Learning

## Overview

A maze navigation agent that reaches a certain goal within the maze (maze exit or a desired cell). It can navigate in four directions and cannot move through walls. A small amount of reward is given for each step and reaching the goal pays out, meaning the agent is encouraged to find the shortest paths rather than any path.

## Learning Without Prior Knowledge

The agent has no previous knowledge of the maze. It learns from the transitions it experiences—each one captured as a tuple of (state, action, reward, next_state).

**State:** The agent's current observation, which includes its position and the goal position within the maze grid.

**Action:** One of the four possible movements: up, down, left, or right.

**Reward:** A scalar value received after taking an action. Reaching the goal yields a positive reward. Each step incurs a cost of −0.1. Colliding with walls carries a penalty of −1.0.

**Next State:** The resulting state after the action is executed, representing the agent's updated position and observation of the environment.

## Validation Method

A shortest-path solver is used at the beginning to verify the maze is solvable and at the end to check how close the learned route came to optimal. Importantly, it is never used during training.