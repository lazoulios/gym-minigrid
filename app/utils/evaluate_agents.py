import sys
import os
import numpy as np
import pandas as pd
import csv
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

def evaluate_agent(env_class, model_path, num_episodes=100):
    if not os.path.exists(model_path + ".zip"):
        print(f"model_path '{model_path}.zip' does not exist.")
        return

    env = env_class(render_mode=None,max_steps=200)
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
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward

            if terminated or truncated:
                done = True
                if info.get('is_success', False) == True: 
                    successes += 1
        
        total_rewards.append(episode_reward)

    avg_return = np.mean(total_rewards)
    success_rate = (successes / num_episodes) * 100

    print("-" * 40)
    print(f"Results for {env_class.__name__} Map:")
    print(f"  Success Rate:   {success_rate:.1f}%")
    print(f"  Average Return: {avg_return:.3f}")
    print("-" * 40 + "\n")
    df = pd.DataFrame({
        "map": [env_class.__name__],
        "success_rate": [success_rate],
        "average_return": [avg_return]
    })
    csv_path = f"app/data/evaluation/{env_class.__name__.lower()}_evaluation.csv"
    #append to csv if it exists, otherwise create new
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:        
        df.to_csv(csv_path, index=False)
    print(f"Saved evaluation results to '{csv_path}'")

    env.close()

if __name__ == "__main__":
    #loop through the three maps and evaluate the corresponding models for 100k, 300k, and 500k timesteps
    for i in range(1, 4):
        choice = str(i) 
        EnvClass = {"1": First, "2": Second, "3": Third}[choice]
        map_name = {"1": "first_map", "2": "second_map", "3": "third_map"}[choice]

        for model_type in ["100k", "300k", "500k"]:
            model_path = f"app/data/model/ppo_agent_{model_type}_{map_name}"
            evaluate_agent(EnvClass, model_path, num_episodes=500)
            