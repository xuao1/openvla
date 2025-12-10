import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
from tqdm import tqdm

DATA_DIR = '/root/' 
DATASET_NAME = 'aloha2openvla_multi_rgb'

def compute_stats():
    ds = tfds.load(DATASET_NAME, data_dir=DATA_DIR, split='train')
    
    # 用于收集所有的 action 和 state
    all_actions = []
    all_states = []
    
    print("Iterating through dataset to collect stats...")
    for episode in tqdm(ds):
        # 遍历 episode 中的每一步
        for step in episode['steps']:
            all_actions.append(step['action'].numpy())
            all_states.append(step['observation']['state'].numpy())
            
    all_actions = np.array(all_actions)
    all_states = np.array(all_states)
    
    print(f"Total steps collected: {len(all_actions)}")
    
    # 计算均值和标准差
    action_mean = np.mean(all_actions, axis=0)
    action_std = np.std(all_actions, axis=0)
    
    state_mean = np.mean(all_states, axis=0)
    state_std = np.std(all_states, axis=0)
    
    # 打印可以直接复制到 configs.py 的格式
    print("\n=== Copy these into prismatic/vla/datasets/rlds/oxe/configs.py ===")
    print(f"Action Mean: {list(action_mean)}")
    print(f"Action Std:  {list(action_std)}")
    print("-" * 20)
    print(f"Proprio (State) Mean: {list(state_mean)}")
    print(f"Proprio (State) Std:  {list(state_std)}")

if __name__ == "__main__":
    compute_stats()