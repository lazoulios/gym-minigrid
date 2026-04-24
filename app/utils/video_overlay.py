import sys
import os
import cv2
import numpy as np
from stable_baselines3 import PPO
from minigrid.wrappers import ImgObsWrapper

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.src.first_map import First
from app.src.second_map import Second
from app.src.third_map import Third

def collect_frames(model_path, env, seed, max_steps=100):
    if model_path:
        model = PPO.load(model_path)
    else:
        model = None

    frames = []
    obs, _ = env.reset(seed=seed)
    
    frames.append(env.render())

    for _ in range(max_steps):
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
            
        obs, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        
        if terminated or truncated:
            break
            
    return frames

if __name__ == "__main__":
    for i in range(1, 4):
        choice = str(i) 
        EnvClass = {"1": First, "2": Second, "3": Third}[choice]
        map_name = {"1": "first_map", "2": "second_map", "3": "third_map"}[choice]

        print(f"\nLoading {map_name}...")

        env = EnvClass(size=16, render_mode="rgb_array", max_steps=500)
        env = ImgObsWrapper(env)

        test_seed = 42 

        print("Καταγραφή κινήσεων του 'Αρχάριου' Agent (Τυχαίες κινήσεις)...")
        frames_bad = collect_frames(None, env, test_seed)

        print("Καταγραφή κινήσεων του 'Εκπαιδευμένου' Agent...")
        model_path = f"app/data/model/ppo_agent_50_{map_name}"
        frames_good = collect_frames(model_path, env, test_seed)

        env.close()

        print("Επεξεργασία και Overlay των βίντεο...")
        max_length = max(len(frames_bad), len(frames_good))
        
        height, width, layers = frames_good[0].shape

        save_path = f"app/data/media/evolution_overlay_{map_name}.mp4"
        video_name = 'evolution_overlay.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(save_path, fourcc, 10, (width, height))

        for i in range(max_length):
            frame1 = frames_bad[i] if i < len(frames_bad) else frames_bad[-1]
            frame2 = frames_good[i] if i < len(frames_good) else frames_good[-1]

            # Επειδή το Gymnasium επιστρέφει RGB και το OpenCV θέλει BGR, τα γυρνάμε
            frame1_bgr = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
            frame2_bgr = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)

            # OVERLAY: 40% διαφάνεια στον κακό agent, 60% στον καλό
            blended_frame = cv2.addWeighted(frame1_bgr, 0.4, frame2_bgr, 0.6, 0)
            
            video.write(blended_frame)

        cv2.destroyAllWindows()
        video.release()
        print(f"Έτοιμο! Το βίντεο αποθηκεύτηκε ως '{video_name}'")