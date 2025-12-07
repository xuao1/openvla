"""
将ALOHA格式的HDF5数据转换为模型所需的格式。
"""

import dataclasses
import json
import logging
from pathlib import Path
import shutil
from typing import Literal, Optional, Dict

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro
import cv2
import torch
from tqdm import tqdm


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 60
    image_writer_threads: int = 30
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def find_metadata_json(hdf5_dir: Path) -> Path:
    """在同级目录下查找metadata.json文件"""
    metadata_path = hdf5_dir / "metadata.json"
    if metadata_path.exists():
        return metadata_path
    raise ValueError(f"No metadata.json found in {hdf5_dir}")


def load_language_instruction_from_json(json_path: Path) -> str:
    """从JSON文件加载语言指令"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get('task_description_english', '')


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    """自动检测相机列表"""
    with h5py.File(hdf5_files[0], "r") as ep:
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]


def has_velocity(hdf5_files: list[Path]) -> bool:
    """检查是否包含速度数据"""
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    """检查是否包含力矩数据"""
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def has_depth(hdf5_files: list[Path]) -> bool:
    """检查是否包含深度图像"""
    with h5py.File(hdf5_files[0], "r") as ep:
        return any(f"/observations/images_depth/{cam}" in ep for cam in get_cameras(hdf5_files))


def has_robot_base(hdf5_files: list[Path]) -> bool:
    """检查是否包含机器人基座动作"""
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/base_action" in ep


def create_empty_dataset(
    repo_id: str,
    robot_type: str = "aloha",
    mode: Literal["video", "image"] = "video",
    *,
    has_effort: bool = False,
    has_velocity: bool = False,
    has_depth: bool = False,
    has_robot_base: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    """创建空的数据集结构"""
    # 基础关节
    base_motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]

    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]
    
    # 如果包含机器人基座，添加基座动作
    if has_robot_base:
        motors = base_motors + ["base_x", "base_y"]
    else:
        motors = base_motors

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        }
    }

    # 添加速度特征
    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        }

    # 添加力矩特征
    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        }

    # 添加图像特征
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }
        
        # 如果包含深度图像，添加深度特征
        if has_depth:
            features[f"observation.images_depth.{cam}"] = {
                "dtype": "float32",
                "shape": (480, 640),
                "names": ["height", "width"],
            }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=25,# 后续加为超参数，默认为25
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    language_instruction: str,
    has_robot_base: bool = False,
    has_depth: bool = False,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    """填充数据集"""
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm(episodes, desc="Processing episodes"):
        ep_path = hdf5_files[ep_idx]
        logging.info(f"Processing episode {ep_idx}: {ep_path}")

        with h5py.File(ep_path, "r") as ep:
            # 加载基本数据
            state = torch.from_numpy(ep["/observations/qpos"][:])
            action = torch.from_numpy(ep["/action"][:])
            
            # 加载图像数据
            imgs_per_cam = {}
            for camera in ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]:
                if f"/observations/images/{camera}" in ep:
                    img_data = ep[f"/observations/images/{camera}"][:]
                    decoded_video = []
                    for encoded in img_data:
                        img = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        decoded_video.append(img)
                    imgs_per_cam[camera] = np.array(decoded_video)

            # 加载可选数据
            velocity = torch.from_numpy(ep["/observations/qvel"][:]) if "/observations/qvel" in ep else None
            effort = torch.from_numpy(ep["/observations/effort"][:]) if "/observations/effort" in ep else None

            # 添加每一帧数据
            num_frames = state.shape[0]
            for i in range(num_frames):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "task": language_instruction
                }

                # 添加图像数据
                for camera, img_array in imgs_per_cam.items():
                    frame[f"observation.images.{camera}"] = img_array[i]

                # 添加可选数据
                if velocity is not None:
                    frame["observation.velocity"] = velocity[i]
                if effort is not None:
                    frame["observation.effort"] = effort[i]

                dataset.add_frame(frame)

            # 保存episode，使用与Libero相同的方式
            dataset.save_episode()

    return dataset


@dataclasses.dataclass
class Args:
    hdf5_dir: list[str]  # 多个HDF5文件目录
    repo_id: str   # 输出数据集ID
    push_to_hub: bool = False  # 是否推送到Hugging Face Hub
    is_mobile: bool = False    # 是否是移动机器人
    mode: Literal["video", "image"] = "image"  # 存储模式
    has_depth: bool = False    # 是否包含深度图像
    has_robot_base: bool = False  # 是否包含机器人基座动作


def main(args: Args):
    dataset = None

    for dir_path in args.hdf5_dir:
        raw_dir = Path(dir_path)
        if not raw_dir.exists():
            logging.warning(f"Directory not found: {raw_dir}")
            continue

        try:
            metadata_path = find_metadata_json(raw_dir)
            language_instruction = load_language_instruction_from_json(metadata_path)
            logging.info(f"[{raw_dir.name}] Loaded task prompt: {language_instruction}")
            print(language_instruction)
        except Exception as e:
            logging.warning(f"Metadata error in {raw_dir}: {e}")
            language_instruction = ""

        hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
        if not hdf5_files:
            logging.warning(f"No episode files found in {raw_dir}")
            continue

        # 检查数据特征（只用第一个目录判断一次）
        if dataset is None:
            has_effort_data = has_effort(hdf5_files)
            has_velocity_data = has_velocity(hdf5_files)
            has_depth_data = args.has_depth
            has_robot_base_data = args.has_robot_base

            dataset = create_empty_dataset(
                args.repo_id,
                robot_type="mobile_aloha" if args.is_mobile else "aloha",
                mode=args.mode,
                has_effort=has_effort_data,
                has_velocity=has_velocity_data,
                has_depth=has_depth_data,
                has_robot_base=has_robot_base_data,
            )

        # 添加数据（使用 prompt）
        dataset = populate_dataset(
            dataset,
            hdf5_files=hdf5_files,
            language_instruction=language_instruction,
            has_robot_base=args.has_robot_base,
            has_depth=args.has_depth,
        )

    # if dataset is not None:
    #     dataset.consolidate()
    #     if args.push_to_hub:
    #         dataset.push_to_hub()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    print(args)
    main(args) 