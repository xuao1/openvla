"""
simple_test.py

Minimal test for OpenVLA server - single request version.
"""

import numpy as np
import json_numpy
json_numpy.patch()
import requests

def simple_test():
    # Create a simple test image
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    instruction = "pick up the blue block"
    
    try:
        response = requests.post(
            "http://127.0.1.1:8000/act",
            json={"image": image, "instruction": instruction, "unnorm_key": "bridge_orig"}
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