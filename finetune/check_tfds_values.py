import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

# 指向你的数据根目录
DATA_DIR = '/home/aox/model/aloha_rlds/' 
DATASET_NAME = 'aloha2openvla_multi_rgb'

print(f"Checking dataset: {DATASET_NAME} in {DATA_DIR}")

ds = tfds.load(DATASET_NAME, data_dir=DATA_DIR, split='train')

for episode in ds.take(1): # 只看第1个episode
    print("\n=== Checking First Episode ===")
    steps = list(episode['steps'].as_numpy_iterator())
    
    # 打印前5帧的状态数据
    print("First 5 frames of 'state':")
    for i in range(5):
        state_val = steps[i]['observation']['state']
        print(f"Frame {i}: {state_val}")
        
    # 检查是否全为0
    all_states = np.array([s['observation']['state'] for s in steps])
    print(f"\nMax value in state: {np.max(all_states)}")
    print(f"Min value in state: {np.min(all_states)}")
    
    if np.max(np.abs(all_states)) == 0:
        print("\n❌ 严重错误: TFDS 中的 state 数据全是 0！")
        print("请检查你的 aloha2openvla_multi_rgb.py 转换脚本。")
    else:
        print("\n✅ 数据正常: state 中包含非零值。")