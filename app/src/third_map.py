import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, "../../"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.world_object import Goal, Wall, Lava, Door, Key
from minigrid.wrappers import ImgObsWrapper
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
import time

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

class Third(MiniGridEnv):
    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps=500, 
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps, 
            agent_view_size=7,
            **kwargs,
        )

    def reset(self, **kwargs):
        print("Resetting environment...")
        obs, info = super().reset(**kwargs)
        self.visited_cells = set()
        self.visited_cells.add(tuple(self.agent_pos))
        self.rewarded_for_key = False 
        self.rewarded_for_door = False
        
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        current_cell = self.grid.get(*self.agent_pos)
        if current_cell is not None:
            if current_cell.type == 'goal':
                reward += 5.0  
                terminated = True
                print(f'\n{Colors.YELLOW}Agent reached Goal. +5.0 Reward{Colors.RESET}')
            

        if not terminated and not truncated:
            reward -= 0.005

        current_pos_tuple = tuple(self.agent_pos)
        if current_pos_tuple not in self.visited_cells:
            reward += 0.005
            self.visited_cells.add(current_pos_tuple)

        return obs, reward, terminated, truncated, info

    @staticmethod
    def _gen_mission():
        return "labyrinth mission"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.put_obj(Wall(), 2, 1)
        self.put_obj(Wall(), 2, 2)
        self.put_obj(Wall(), 2, 4)
        self.put_obj(Wall(), 2, 5)
        self.put_obj(Wall(), 1, 5)
        self.put_obj(Wall(), 1, 4)
        self.put_obj(Wall(), 4, 2)
        self.put_obj(Wall(), 4, 3)
        self.put_obj(Wall(), 4, 4)
        self.put_obj(Wall(), 4, 6)
        self.put_obj(Wall(), 5, 2)
        self.put_obj(Wall(), 5, 3)
        self.put_obj(Wall(), 5, 6)
        self.put_obj(Wall(), 6, 6)
        self.put_obj(Wall(), 6, 5)

        self.put_obj(Goal(), 1, 6)

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()


if __name__ == "__main__":
    env = Third(size=8, render_mode="human")
    env = ImgObsWrapper(env)

    obs, info = env.reset()

    try:
        for _ in range(500):
            env.render()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            time.sleep(0.05)

            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()
