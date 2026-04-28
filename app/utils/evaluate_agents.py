import sys
import os
import numpy as np
from stable_baselines3 import PPO

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from minigrid.wrappers import ImgObsWrapper
from app.src.train import MinigridFeaturesExtractor
from app.src.first_map import First
from app.src.second_map import Second
from app.src.third_map import Third

def evaluate_agent(env_class, model_path, env_size, num_episodes=100):
    if not os.path.exists(model_path + ".zip"):
        print(f"model_path '{model_path}.zip' does not exist.")
        return

    env = env_class(size=env_size, render_mode=None)
    env = ImgObsWrapper(env) 
    
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128), 
    )

    custom_objects = {
        "policy_kwargs": policy_kwargs
    }
    
    model = PPO.load(model_path, custom_objects=custom_objects)

    successes = 0
    total_rewards = []

    print(f"\nEvaluating {env_class.__name__} Map with model '{model_path}' for {num_episodes} episodes...")

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            episode_reward += reward

            if terminated or truncated:
                done = True
                if reward > 0: 
                    successes += 1
        
        total_rewards.append(episode_reward)

    avg_return = np.mean(total_rewards)
    success_rate = (successes / num_episodes) * 100

    print("-" * 40)
    print(f"Results for {env_class.__name__} Map:")
    print(f"  Success Rate:   {success_rate:.1f}%")
    print(f"  Average Return: {avg_return:.3f}")
    print("-" * 40 + "\n")

    env.close()

if __name__ == "__main__":
    
    model_first = "app/data/model/ppo_agent_100k_first_map"
    model_second = "app/data/model/ppo_agent_100k_second_map"
    model_third = "app/data/model/ppo_agent_100k_third_map"
    
    evaluate_agent(First, model_first, env_size=16, num_episodes=100)
    
    evaluate_agent(Second, model_second, env_size=8, num_episodes=100)

    evaluate_agent(Third, model_first, env_size=8, num_episodes=100)