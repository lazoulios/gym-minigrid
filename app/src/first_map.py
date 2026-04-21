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

class First(MiniGridEnv):
    def __init__(
        self,
        size=16,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=256,
            **kwargs,
        )

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

