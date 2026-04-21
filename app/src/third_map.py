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
import time


class Third(MiniGridEnv):
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
        return "labyrinth mission"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        for x in range(1, width - 1):
            if x not in (2, 4, 7, 9, 12,14):
                self.grid.set(x, 2, Wall())
            if x not in (1, 3, 6, 8, 11,14):
                self.grid.set(x, 5, Wall())
            if x not in (2, 5, 7, 10, 13):
                self.grid.set(x, 8, Wall())
            if x not in (1, 4, 6, 9, 12):
                self.grid.set(x, 11, Wall())
        self.grid.set(9, 7, Wall())
        self.grid.set(9, 4, Wall())
        self.grid.set(2, 8, Wall())
        self.grid.set(11, 13, Wall())
        self.grid.set(11, 12, Wall())
        self.grid.set(9, 11, Wall())
        self.grid.set(6, 5, Wall())
        self.grid.set(13, 8, Wall())

        for y in range(1, height - 1):
            if y not in (3, 6, 9):
                self.grid.set(3, y, Wall())
            if y not in (2, 5, 10):
                self.grid.set(6, y, Wall())
            if y not in (1, 4, 7, 11):
                self.grid.set(9, y, Wall())
            if y not in (2,10,12, 7,8):
                self.grid.set(13, y, Wall())

        self.put_obj(Goal(), width - 2, height - 2)

        self.grid.set(7, 2, Door(COLOR_NAMES[2], is_locked=True))
        self.grid.set(1, 13, Key(COLOR_NAMES[2]))

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()


if __name__ == "__main__":
    env = Third(size=16, render_mode="human")
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
