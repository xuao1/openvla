import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 指向你生成数据的目录 (即 tfds build 命令中 --data_dir 指定的目录)
# 也是包含 "aloha2openvla_multi_rgb" 文件夹的上级目录
DATA_DIR = '/root/' 
DATASET_NAME = 'aloha2openvla_multi_rgb'

def visualize_dataset():
    # 1. 加载数据集
    # split='train' 加载训练集
    print(f"Loading dataset from {DATA_DIR}...")
    ds = tfds.load(DATASET_NAME, data_dir=DATA_DIR, split='train')

    # 2. 取出一个 Episode 进行检查
    # convert_to_numpy=True 方便后续处理
    for episode in ds.take(2): 
        steps = list(episode['steps'].as_numpy_iterator())
        print(f"Episode length: {len(steps)} steps")
        
        # 3. 检查元数据
        print(f"File path: {episode['episode_metadata']['file_path'].numpy().decode('utf-8')}")
        
        # 4. 随机抽取几帧进行可视化 (比如开头、中间、结尾)
        indices_to_show = [0, len(steps)//2, len(steps)-1]
        
        plt.figure(figsize=(15, 10))
        
        for plot_idx, step_idx in enumerate(indices_to_show):
            step = steps[step_idx]
            
            # --- 打印数值信息 ---
            print(f"\n--- Step {step_idx} ---")
            print(f"Language: {step['language_instruction'].decode('utf-8')}")
            print(f"Is First: {step['is_first']}, Is Last: {step['is_last']}")
            print(f"Action (Right Arm): {np.round(step['action'], 3)}")
            print(f"State  (Right Arm): {np.round(step['observation']['state'], 3)}")
            
            # --- 绘制图像 ---
            # 主相机
            plt.subplot(3, 2, plot_idx * 2 + 1)
            plt.imshow(step['observation']['image'])
            plt.title(f"Step {step_idx}: High Cam")
            plt.axis('off')
            
            # 手腕相机
            plt.subplot(3, 2, plot_idx * 2 + 2)
            plt.imshow(step['observation']['wrist_image'])
            plt.title(f"Step {step_idx}: Wrist Cam")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('dataset_verification.png')
        print("\nVisualization saved to 'dataset_verification.png'. Check this image!")

if __name__ == "__main__":
    visualize_dataset()