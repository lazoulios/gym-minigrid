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
from minigrid.core.mission import MissionSpace
from app.src.enemy import RimWorldEnemy
import time

class Second(MiniGridEnv):
    def __init__(
        self,
        size=16,
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
            agent_view_size=11,
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
            if current_cell.type == 'lava' or isinstance(current_cell, RimWorldEnemy):
                reward -= 10.0 
                terminated = True 
                print('Agent died. -10.0 Reward')

        if not terminated and not truncated:
            reward -= 0.005

        if self.carrying is not None and self.carrying.type == 'key' and not self.rewarded_for_key:
            reward += 0.5
            self.rewarded_for_key = True
            print('Key picked up. +0.5 Reward')

        if action == self.actions.toggle:
            front_cell = self.grid.get(*self.front_pos)
            if front_cell is not None and front_cell.type == 'door' and front_cell.is_open and not self.rewarded_for_door:
                reward += 0.5
                self.rewarded_for_door = True
                print('Door opened. +0.5 Reward')

        current_pos_tuple = tuple(self.agent_pos)
        if current_pos_tuple not in self.visited_cells:
            reward += 0.02 
            self.visited_cells.add(current_pos_tuple)

        return obs, reward, terminated, truncated, info

    @staticmethod
    def _gen_mission():
        return "grand mission"


    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(Goal(), width - 8, height - 8)

        for i in range(3, height-3):
            self.grid.set(3, i, Wall())
        for i in range(3, height-3):
            self.grid.set(12, i, Wall())
        for i in range(3, height-3):
            self.grid.set(i, 3, Wall())
        for i in range(3, height-3):
            self.grid.set(i, 12, Wall())
        for i in range(3, height-3):
            self.grid.set(9, i, Wall())

        self.grid.set(12, 7, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(2, 11, Key(COLOR_NAMES[0]))
    
        self.grid.set(3, 7, Door(COLOR_NAMES[4], is_locked=True))
        self.grid.set(10, 5, Key(COLOR_NAMES[4]))

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

if __name__ == "__main__":
    env = Second(size=16,render_mode="human")
    env = ImgObsWrapper(env)

    obs, info = env.reset()
    #print(obs.shape)

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

