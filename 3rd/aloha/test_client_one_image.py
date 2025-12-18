import numpy as np
import json_numpy
json_numpy.patch()
import requests
from PIL import Image  # 导入 PIL 库

def simple_test():
    # --- 替换部分开始 ---
    # 加载本地图片
    img_path = "debug_img_image_save.png"
    try:
        # 打开图片并确保是 RGB 模式
        pil_img = Image.open(img_path).convert("RGB")
        # 转换为 NumPy 数组，类型为 uint8
        image = np.array(pil_img)
        print(f"成功加载图片: {img_path}, 形状为: {image.shape}")
    except FileNotFoundError:
        print(f"错误：找不到文件 {img_path}，请确保文件在当前目录下。")
        return
    instruction = "flip the object upright"
    
    try:
        response = requests.post(
            "http://127.0.1.1:8000/act",
            json={"debug_use_saved_pt": False, "image": image, "instruction": instruction, "unnorm_key": "aloha2openvla_multi_rgb_flip_upright"}
        )
        
        if response.status_code == 200:
            action = response.json()
            print("Success! Action received:")
            print(f"Action type: {type(action)}")
            if isinstance(action, dict) and "action" in action:
                action_data = action["action"]
                print(f"Action shape: {np.array(action_data).shape}")
            else:
                print(f"Action: {action}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    simple_test()