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

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

class First(MiniGridEnv):
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
        print(f"\n{Colors.BLUE}Resetting environment...{Colors.RESET}")
        obs, info = super().reset(**kwargs)
        
        self.key_pos = (3, 14)
        self.door_pos = (5, 7)
        self.goal_pos = (self.width - 2, self.height - 2)
        
        self.current_phase = 1

        self.prev_distance = abs(self.agent_pos[0] - self.key_pos[0]) + abs(self.agent_pos[1] - self.key_pos[1])
        
        self.rewarded_for_key = False 
        self.rewarded_for_door = False

        self.min_phase3_distance = float('inf')
        
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.done:
            terminated = False 

        current_cell = self.grid.get(*self.agent_pos)
        if current_cell is not None:
            if current_cell.type == 'lava' or isinstance(current_cell, RimWorldEnemy):
                reward -= 0.5  
                terminated = True 
                print(f'\n{Colors.RED}Agent died. -0.5 Reward{Colors.RESET}')
            elif current_cell.type == 'goal':
                reward += 5.0  
                terminated = True
                print(f'\n{Colors.YELLOW}Agent reached Goal. +5.0 Reward{Colors.RESET}')

        if action == self.actions.drop and self.carrying is not None and self.carrying.type == 'key' and not self.rewarded_for_door and self.current_phase == 2:
            reward -= 2.0
            terminated = True
            print(f"\n{Colors.RED}Agent dropped the item! -2.0 Reward{Colors.RESET}")

        if not terminated and not truncated:
            reward -= 0.005

        front_cell = self.grid.get(*self.front_pos)
        if self.current_phase == 2 and front_cell is not None and front_cell.type == 'door' and not front_cell.is_open:
            reward += 0.005

        if self.carrying is not None and self.carrying.type == 'key' and not self.rewarded_for_key:
            reward += 0.5
            self.rewarded_for_key = True
            self.current_phase = 2 
            self.prev_distance = abs(self.agent_pos[0] - self.door_pos[0]) + abs(self.agent_pos[1] - self.door_pos[1])
            print(f'\n{Colors.GREEN}Key picked up. +0.5 Reward{Colors.RESET}')

        if action == self.actions.toggle:
            if front_cell is not None and front_cell.type == 'door' and front_cell.is_open and not self.rewarded_for_door:
                reward += 1.0
                self.rewarded_for_door = True
                self.current_phase = 3
                print(f'\n{Colors.GREEN}Door opened. +1.0 Reward{Colors.RESET}')
                self.min_phase3_distance = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])

        if self.current_phase == 1:
            target = self.key_pos
        elif self.current_phase == 2:
            target = self.door_pos
        else:
            target = self.goal_pos

        current_distance = abs(self.agent_pos[0] - target[0]) + abs(self.agent_pos[1] - target[1])
        
        if self.current_phase in [1, 2]:
            if current_distance < self.prev_distance:
                reward += 0.1
            elif current_distance > self.prev_distance:
                reward -= 0.1 

        elif self.current_phase == 3:
            if current_distance < self.min_phase3_distance:
                reward += 0.05
                self.min_phase3_distance = current_distance
            
        self.prev_distance = current_distance

        print(f"\rPhase: {self.current_phase} | Target Distance: {current_distance:<2}    ", end="", flush=True)

        return obs, reward, terminated, truncated, info

    @staticmethod
    def _gen_mission():
        return "grand mission"


    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(Goal(), width - 2, height - 2)

        for i in range(0, height):
            self.grid.set(5, i, Wall())

        self.grid.set(5, 7, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(3, 14, Key(COLOR_NAMES[0]))

        self.grid.set(12, 12, Lava())
        self.grid.set(12, 11, Lava())
        self.grid.set(11, 12, Lava())
        self.grid.set(11, 11, Lava())
        self.grid.set(10, 12, Lava())
        self.grid.set(10, 13, Lava())
        self.grid.set(11, 13, Lava())

        enemy = RimWorldEnemy()
        self.grid.set(10, 6, enemy)
        self.grid.set(11, 7, enemy)

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

if __name__ == "__main__":
    env = First(size=16,render_mode="human")
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

