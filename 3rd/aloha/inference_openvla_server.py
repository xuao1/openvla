"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import (
    AutoModelForVision2Seq, 
    AutoProcessor, 
    AutoConfig, 
    AutoImageProcessor
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class OpenVLAServer:
    def __init__(self, openvla_path: Union[str, Path], attn_implementation: Optional[str] = "flash_attention_2") -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.openvla_path, self.attn_implementation = openvla_path, attn_implementation
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        # # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        # if os.path.isdir(self.openvla_path):
        #     with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
        #         self.vla.norm_stats = json.load(f)

    def predict_action(self, payload: Dict[str, Any]) -> str:
        # >>> [DEBUG START] <<<
        if payload.get("debug_use_saved_pt", False):

            # Parse payload components
            instruction = payload["instruction"]

            raw_images = []
            if "image_primary" in payload and "image_wrist" in payload:
                raw_images = [payload["image_primary"], payload["image_wrist"]]
            elif "image" in payload:
                raw_images = [payload["image"]]
            else:
                raise ValueError("Payload must contain either 'image' or both 'image_primary' and 'image_wrist'.")
            
            pil_images = []
            for img in raw_images:
                if not isinstance(img, np.ndarray):
                    img = np.asarray(img, dtype=np.uint8)
                pil_images.append(Image.fromarray(img).convert("RGB"))

            unnorm_key = payload.get("unnorm_key", "aloha2openvla_multi_rgb_flip_upright")
            prompt = get_openvla_prompt(instruction, self.openvla_path)

            if len(pil_images) > 1:
                pixel_values_list = []
                inputs_text = None

                for img in pil_images:
                    single_inputs = self.processor(prompt, img).to(self.device, dtype=torch.bfloat16)
                    pixel_values_list.append(single_inputs["pixel_values"])
                    inputs_text = single_inputs
                
                combined_pixel_values = torch.cat(pixel_values_list, dim=0)
                combined_pixel_values = combined_pixel_values.unsqueeze(0)
                inputs = {
                    "input_ids": inputs_text["input_ids"],
                    "attention_mask": inputs_text["attention_mask"],
                    "pixel_values": combined_pixel_values
                }
            else:
                inputs = self.processor(prompt, pil_images[0]).to(self.device, dtype=torch.bfloat16)

            
            print("\n🚨 DEBUG MODE: Loading inputs from 'first_train_step_inputs.pt'...")
            saved_batch = torch.load("/home/aox/openvla/first_train_step_inputs.pt")

            # inputs = {
            #     "input_ids": saved_batch["input_ids"][0:1].to(self.device),
            #     "attention_mask": saved_batch["attention_mask"][0:1].to(self.device),
            # }

            raw_pixels = saved_batch["pixel_values"]
            
            if isinstance(raw_pixels, dict):
                # 检查是否同时存在 siglip 和 dino
                if "siglip" in raw_pixels and "dino" in raw_pixels:
                    print("ℹ️ Found both 'siglip' and 'dino' images. Concatenating for Fused Backbone...")
                    
                    # 1. 取出两个 Tensor, 切片 [0:1] 保留 Batch 维度
                    siglip_img = raw_pixels["siglip"][0:1].to(self.device, dtype=torch.bfloat16)
                    dino_img = raw_pixels["dino"][0:1].to(self.device, dtype=torch.bfloat16)
                    
                    # 2. 拼接成 [1, 6, 224, 224]
                    # 注意：OpenVLA 的 Prismatic 实现通常期望顺序是 cat([siglip, dino]) 还是 cat([dino, siglip])?
                    # 查看报错代码 modeling_prismatic.py line 124: img, img_fused = torch.split(pixel_values, [3, 3])
                    # 通常第一个分块给 SigLIP (img), 第二个给 DINO (img_fused) 或者反过来。
                    # 大多数 Prismatic 代码库的默认行为是：inputs_dict['siglip'] 对应第一个 3 通道。
                    inputs["pixel_values"] = torch.cat([dino_img, siglip_img], dim=1)
                    
                elif "pixel_values" in raw_pixels:
                    # 如果字典里直接就有 pixel_values (可能已经是合体的)
                    inputs["pixel_values"] = raw_pixels["pixel_values"][0:1].to(self.device, dtype=torch.bfloat16)
                else:
                    raise ValueError(f"Cannot construct 6-channel input. Available keys: {list(raw_pixels.keys())}")

                # 处理 mask (如果有)
                # if "pixel_attention_mask" in raw_pixels:
                #     inputs["pixel_attention_mask"] = raw_pixels["pixel_attention_mask"][0:1].to(self.device, dtype=torch.bfloat16)
            else:
                # 只是 Tensor 的情况
                inputs["pixel_values"] = raw_pixels[0:1].to(self.device, dtype=torch.bfloat16)

            print(f"✅ Final pixel_values shape: {inputs['pixel_values'].shape}")

            # === [DEBUG PRINT START] 打印数值以进行对比 ===
            print("inputs_ids:", inputs["input_ids"])
            print("attention_mask:", inputs["attention_mask"])
            print("dino_img stats - shape, range:", inputs["pixel_values"][:, :3, :, :].shape,
                  inputs["pixel_values"][:, :3, :, :].min().item(), inputs["pixel_values"][:, :3, :, :].max().item())
            print("siglip_img stats - shape, range:", inputs["pixel_values"][:, 3:, :, :].shape,
                  inputs["pixel_values"][:, 3:, :, :].min().item(), inputs["pixel_values"][:, 3:, :, :].max().item())
            # === [DEBUG PRINT END] ===

            if "labels" in saved_batch:
                gt_ids = saved_batch["labels"][0]
                valid_ids = gt_ids[gt_ids != -100]
                if len(valid_ids) > 0 and valid_ids[-1] == 2: 
                     valid_ids = valid_ids[:-1]
                print(f"🎯 Ground Truth Action Tokens: {valid_ids.tolist()}")
                
            unnorm_key = payload.get("unnorm_key", "aloha2openvla_multi_rgb_flip_upright")
            
            print("🚀 Running inference on saved training input...")
            action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            
            # === [FIX] 格式化输出 ===
            print(f"🤖 Predicted Action (Raw): {action}")
            
            # 这里的 action 是 numpy array，FastAPI 需要 string
            # 我们可以把它转成列表再转 JSON 字符串，或者直接转 string
            import json
            return json.dumps(action.tolist())
        # >>> [DEBUG END] <<<

        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            instruction = payload["instruction"]

            raw_images = []
            if "image_primary" in payload and "image_wrist" in payload:
                raw_images = [payload["image_primary"], payload["image_wrist"]]
            elif "image" in payload:
                raw_images = [payload["image"]]
            else:
                raise ValueError("Payload must contain either 'image' or both 'image_primary' and 'image_wrist'.")
            
            pil_images = []
            for img in raw_images:
                if not isinstance(img, np.ndarray):
                    img = np.asarray(img, dtype=np.uint8)
                pil_images.append(Image.fromarray(img).convert("RGB"))

            unnorm_key = payload.get("unnorm_key", "aloha2openvla_multi_rgb_flip_upright")
            prompt = get_openvla_prompt(instruction, self.openvla_path)

            if len(pil_images) > 1:
                pixel_values_list = []
                inputs_text = None

                for img in pil_images:
                    single_inputs = self.processor(prompt, img).to(self.device, dtype=torch.bfloat16)
                    pixel_values_list.append(single_inputs["pixel_values"])
                    inputs_text = single_inputs
                
                combined_pixel_values = torch.cat(pixel_values_list, dim=0)
                combined_pixel_values = combined_pixel_values.unsqueeze(0)
                inputs = {
                    "input_ids": inputs_text["input_ids"],
                    "attention_mask": inputs_text["attention_mask"],
                    "pixel_values": combined_pixel_values
                }
            else:
                inputs = self.processor(prompt, pil_images[0]).to(self.device, dtype=torch.bfloat16)

            # print("Inputs keys:", inputs.keys())
            # print("Pixel values shape:", inputs["pixel_values"].shape)
            # print("inputs input_ids shape:", inputs["input_ids"].shape)
            # print("inputs attention_mask shape:", inputs["attention_mask"].shape)
            # print("Inputs attention_mask: ", inputs["attention_mask"])

            # Run VLA Inference
            action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"

    def run(self, host: str = "127.0.1.1", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off
    openvla_path: Union[str, Path] = "/data/aox/openvla_hf/aloha_7b_flip_upright_one_image_full"              # HF Hub Path (or path to local run directory)

    # Server Configuration
    host: str = "127.0.1.1"                                                         # Host IP Address
    port: int = 8000                                                                # Host Port

    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenVLAServer(cfg.openvla_path)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
