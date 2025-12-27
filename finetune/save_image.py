import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os
from PIL import Image  # 需要安装: pip install pillow

# 指向你生成数据的目录
DATA_DIR = '/data/aox/model/'
DATASET_NAME = 'aloha2openvla_multi_rgb_lift'
OUTPUT_DIR = './saved_frames' # 图片保存的根目录

def save_dataset_images():
    # 0. 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # 1. 加载数据集
    print(f"Loading dataset from {DATA_DIR}...")
    ds = tfds.load(DATASET_NAME, data_dir=DATA_DIR, split='train')

    # 2. 遍历 Episode (这里限制只处理前 2 个 Episode，防止数据量太大)
    # enumerate 用于获取 episode 的序号
    for episode_idx, episode in enumerate(ds.take(2)):
        print(f"Processing Episode {episode_idx}...")
        
        # 将整个 episode 的步骤转换为列表
        steps = list(episode['steps'].as_numpy_iterator())
        
        # 为当前 Episode 创建一个单独的子文件夹
        episode_save_dir = os.path.join(OUTPUT_DIR, f"episode_{episode_idx}")
        os.makedirs(episode_save_dir, exist_ok=True)
        
        # 3. 逐帧保存图片
        for step_idx, step in enumerate(steps):
            # 获取 Primary Image (对应原代码中的 'image')
            # image_array 的形状通常是 (Height, Width, 3)
            image_array = step['observation']['image']
            
            # --- 核心修改部分 ---
            # 使用 PIL 将 numpy 数组转换为图片对象
            img = Image.fromarray(image_array)
            
            # 构造文件名，例如: episode_0/frame_000.png
            file_name = f"frame_{step_idx:04d}.png"
            save_path = os.path.join(episode_save_dir, file_name)
            
            # 保存图片
            img.save(save_path)
            
            # (可选) 每隔几帧打印一下进度，避免刷屏
            if step_idx % 10 == 0:
                print(f"  Saved {save_path}")

        print(f"Episode {episode_idx} finished. Saved {len(steps)} images to {episode_save_dir}\n")

if __name__ == "__main__":
    # 确保安装了 Pillow: pip install pillow
    save_dataset_images()