import sys
import os
import time

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

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

class Second(MiniGridEnv):
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
        print(f"\n{Colors.BLUE}Resetting Map 2...{Colors.RESET}")
        obs, info = super().reset(**kwargs)
        
        self.key1_pos = (2, 11)   
        self.door1_pos = (12, 7)  
        self.key2_pos = (10, 5)   
        self.door2_pos = (3, 7)   
        self.goal_pos = (self.width - 8, self.height - 8)
        
        self.current_phase = 1

        self.rewarded_for_key1 = False 
        self.rewarded_for_door1 = False
        self.rewarded_for_key2 = False 
        self.rewarded_for_door2 = False
        
        self.visited_cells = set()
        self.visited_cells.add(tuple(self.agent_pos))
        
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.done:
            terminated = False 

        current_cell = self.grid.get(*self.agent_pos)
        if current_cell is not None:
            if current_cell.type == 'goal':
                reward += 5.0  
                terminated = True
                print(f'\n{Colors.YELLOW}Agent reached Goal. +5.0 Reward{Colors.RESET}')
            
        if not terminated and not truncated:
            reward -= 0.005 

        front_cell = self.grid.get(*self.front_pos)
        if front_cell is not None:
            if front_cell.type == 'key':
                if self.current_phase == 1 and front_cell.color == COLOR_NAMES[0]:
                    reward += 0.005
                elif self.current_phase == 3 and front_cell.color == COLOR_NAMES[4]:
                    reward += 0.005
            elif front_cell.type == 'door' and not front_cell.is_open:
                if self.current_phase == 2 and front_cell.color == COLOR_NAMES[0]:
                    reward += 0.005
                elif self.current_phase == 4 and front_cell.color == COLOR_NAMES[4]:
                    reward += 0.005

        if action == self.actions.drop and self.carrying is not None and self.carrying.type == 'key':
            if self.current_phase in [2, 4]:    
                reward -= 2.0
                terminated = True
                print(f"\n{Colors.RED}Agent dropped the key. -2.0 Reward{Colors.RESET}")

        if self.carrying and self.carrying.type == 'key' and self.carrying.color == COLOR_NAMES[0]:
            if self.current_phase == 1:
                self.current_phase = 2 
                reward += 0.5
                self.rewarded_for_key1 = True
                print(f'\n{Colors.GREEN}Key 1 picked up. +0.5 Reward{Colors.RESET}')

        if action == self.actions.toggle and front_cell and front_cell.type == 'door' and front_cell.color == COLOR_NAMES[0] and front_cell.is_open:
            if self.current_phase == 2:
                self.current_phase = 3
                reward += 1.0
                self.rewarded_for_door1 = True
                print(f'\n{Colors.GREEN}Door 1 opened. +1.0 Reward{Colors.RESET}')

        if self.carrying and self.carrying.type == 'key' and self.carrying.color == COLOR_NAMES[4]:
            if self.current_phase == 3:
                self.current_phase = 4 
                reward += 0.5
                self.rewarded_for_key2 = True
                print(f'\n{Colors.GREEN}Key 2 picked up. +0.5 Reward{Colors.RESET}')

        if action == self.actions.toggle and front_cell and front_cell.type == 'door' and front_cell.color == COLOR_NAMES[4] and front_cell.is_open:
            if self.current_phase == 4:
                self.current_phase = 5
                reward += 1.0
                self.rewarded_for_door2 = True
                print(f'\n{Colors.GREEN}Door 2 opened. +1.0 Reward{Colors.RESET}')

        print(f"\rPhase: {self.current_phase} | Reward: {reward:.3f}    ", end="", flush=True)

        return obs, reward, terminated, truncated, info

    @staticmethod
    def _gen_mission():
        return "grand mission"


    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(Goal(), 6, 1)

        for i in range(1, height):
            self.grid.set(3, i, Wall())
        for i in range(3, height):
            self.grid.set(i, 3, Wall())

        self.grid.set(3, 5, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(1, 6, Key(COLOR_NAMES[0]))
    
        self.grid.set(3, 2, Door(COLOR_NAMES[4], is_locked=True))
        self.grid.set(6, 5, Key(COLOR_NAMES[4]))

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

if __name__ == "__main__":
    env = Second(size=8,render_mode="human")
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