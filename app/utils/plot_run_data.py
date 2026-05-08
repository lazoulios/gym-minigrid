import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_learning_curves(csv_dict, title, ylabel, save_name="training_curve.png", window_size=20):
    plt.figure(figsize=(10, 6))
    
    sns.set_theme(style="darkgrid")

    colors = sns.color_palette("husl", len(csv_dict))

    for (label, path), color in zip(csv_dict.items(), colors):
        if not os.path.exists(path):
            print(f"Το αρχείο δεν βρέθηκε: {path}")
            continue
            
        # Φορτώνουμε το CSV
        df = pd.read_csv(path)
        
        steps = df['Step']
        values = df['Value']

        smoothed_values = values.rolling(window=window_size, min_periods=1).mean()

        plt.plot(steps, values, color=color, alpha=0.2)
        
        plt.plot(steps, smoothed_values, color=color, label=f"{label}", linewidth=2.5)

    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    plt.savefig(save_name, dpi=300)
    print(f"Το γράφημα αποθηκεύτηκε ως: {save_name}")
    

if __name__ == "__main__":
    csv_files = {
        "100k Timesteps": "app/data/csv/ep_len/tensorboard_100k_first_map_PPO_1.csv",
        "300k Timesteps": "app/data/csv/ep_len/tensorboard_300k_first_map_PPO_1.csv",
        "500k Timesteps": "app/data/csv/ep_len/tensorboard_500k_first_map_PPO_1.csv"
    }

    plot_learning_curves(
        csv_dict=csv_files,
        title="Average Episode Length: Map 1 (First)",
        ylabel="Average Episode Length",
        save_name="map1_convergence.png",
        window_size=15  
    )