# gym-minigrid

The goal of this repository is to train an autonomous autonomous agent to solve multi-stage logic puzzles within the Gym MiniGrid framework using Deep Reinforcement Learning. The project utilizes the Proximal Policy Optimization (PPO) algorithm and explores the critical impact of reward shaping and custom Convolutional Neural Network (CNN) architectures in helping the agent overcome local minima,

## Features
* **Custom MiniGrid Environments:** Three distinct map layouts designed to test specific agent capabilities (hazard avoidance, sequential logic, and spatial generalization).
* **Custom Entity Rendering:** Implementation of a `RimWorldEnemy` using procedural geometric primitives.
* **Optimized Vision System:** A lightweight CNN feature extractor optimized for grid-based environments.
* **Visualization Tools:** Utilities to generate side-by-side video comparisons of baseline vs. trained agents.

## Key Results
Summary of important outcomes from some of the experiments. 

| Experiment | Success Rate | Average Return | Notes |
| :--- | :---: | :---: | :--- |
| Map 1 | (fill in) | (fill in) | Mostly Hazard Avoidance |
| Map 2 | (fill in) | (fill in) | Sequential Logic and Inventory Management |
| Map 3 | (fill in) | (fill in) | Exploration and Spatial Generalization |

## Setup (virtual environment)

Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone https://github.com/lazoulios/gym-minigrid
cd gym-minigrid

# Create and activate the virtual environment
python -m venv venv
source venv/bin/activate  # Use venv\Scripts\activate.bat on Windows CMD

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

## Requirements
- Install packages from `requirements.txt`

## Running training
Use the provided training script to train agents. Will train an agent on each map independently.
```bash
python app/src/train.py 
```

## Evaluation and visualization
- Use `app/utils/video_overlay.py` to overlay metrics on saved videos.
- Logs and tensorboard runs are stored under `app/data/runs/`. Use 
`tensorboard --logdir app/data/runs/` to view

## Project structure

```
gym-minigrid/
├── app/
│   ├── src/                # environment maps and training scripts
│   │   ├── first_map.py
│   │   ├── second_map.py
│   │   ├── third_map.py
│   │   ├── enemy.py
│   │   └── train.py
|   ├── utils/              # small helpers (geometry, overlays)
│   └── data/               # models, media, run logs, tensorboard
├── tex/                    # latex report code and media
├── requirements.txt
└── README.md
```
