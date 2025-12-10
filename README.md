# Robotics2025
# Robot Navigation in Uneven Terrain

## Project Overview
A robot navigation system for rough terrain using Monte Carlo Localization (MCL) and Reinforcement Learning (RL). The robot learns to navigate to a goal while avoiding obstacles and maintaining cargo.

## Team Members
- **Basel** - Reinforcement Learning Framework
- **Damil** - Q-Learning Algorithm
- **Nasser** - Odometry System
- **Mohammed** - Monte Carlo Localization

## Software and Libraries Used

### Pre-Built Tools
- **Webots R2023b** - Robot simulation platform
- **NumPy** - Array operations and mathematical computations
- **Python Math Library** - Trigonometric functions

### What We Implemented

**Basel - RL Framework:**
- RobotState class for state representation
- Integration layer connecting localization to Q-learning
- State transition prediction functions

**Damil - Q-Learning:**
- Q-table implementation and updates
- Action selection with exploration
- Reward function design

**Nasser - Odometry:**
- Wheel encoder processing
- Pose estimation from motion
- Particle generation for MCL

**Mohammed - Localization:**
- Sensor likelihood models
- Particle weight updates and resampling
- Pose estimation from particles

## Repository Structure
```
controllers/
├── mainController/
│   ├── mainController.py          # Main control loop
│   ├── rl_integration.py          # RL integration [Basel]
│   └── rl_framework.py            # State transitions [Basel]
└── lib/
    ├── reinforcement_learning.py  # Q-learning [Damil]
    ├── odometry.py                # Odometry [Nasser]
    └── localisation.py            # MCL [Mohammed]
```

## How to Run
1. Install Webots R2023b and Python 3.14.0
2. Install NumPy: `pip install numpy`
3. Open `worlds/UnevenTerrainSimulation.wbt` in Webots
4. Press Play

## System Overview
- **State Space:** 50×50 grid, 4 directions, cargo status = 10,000 states
- **Action Space:** Forward, Left, Right, Stop = 4 actions
- **Q-Table:** 40,000 entries stored as NumPy array

## GitHub Repository
https://github.com/Drmil7129/Robotics2025
