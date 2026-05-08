import sys
import os

import torch 

torch.backends.nnpack.set_flags(False)

import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from minigrid.wrappers import ImgObsWrapper

from app.src.first_map import First
from app.src.second_map import Second
from app.src.third_map import Third

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

MAPS = {
    "1": {"class": First, "name": "first_map"},
    "2": {"class": Second, "name": "second_map"},
    "3": {"class": Third, "name": "third_map"}
}

if __name__ == "__main__":
    for i in range(1, 4):
        choice = str(i) 
        EnvClass = MAPS[choice]["class"]
        map_name = MAPS[choice]["name"]

        print(f"\nLoading {map_name}...")
        
        env = EnvClass()
        env = ImgObsWrapper(env)

        policy_kwargs = dict(
            features_extractor_class=MinigridFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )

        model = PPO(
            "CnnPolicy", 
            env, 
            policy_kwargs=policy_kwargs,
            verbose=1, 
            learning_rate=0.0003,
            tensorboard_log=f"app/data/runs/tensorboard_500k_{map_name}/"
        )

        print(f"\nTraining {map_name}")
        model.learn(total_timesteps=500000)

        save_path = f"app/data/model/ppo_agent_500k_{map_name}"
        model.save(save_path)
        
        print(f"\nFinished. Saved as '{save_path}.zip'")
        env.close()