"""
OpenVLA 专用：将 ALOHA HDF5 转换为 RLDS 格式，仅保留右臂数据，并调整图像大小。
"""

import h5py
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import glob
import json
import os
import cv2

# 定义数据集的基本信息
TARGET_WIDTH = 256
TARGET_HEIGHT = 256
_DESCRIPTION = """
Custom dataset for OpenVLA fine-tuning (Right Arm Only).
Converted from ALOHA HDF5 format with image resizing.
"""

_CITATION = ""
_HOMEPAGE = ""

class Aloha2openvlaMultiRgb(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for aloha2openvla_multi_rgb."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release for OpenVLA.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """指定数据集的 Feature 结构 (RLDS 标准)"""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        # 主相机 (cam_high)
                        'image': tfds.features.Image(
                            shape=(TARGET_HEIGHT, TARGET_WIDTH, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        # 手腕相机 (cam_right_wrist) - OpenVLA 可选，但建议保留
                        'wrist_image': tfds.features.Image(
                            shape=(TARGET_HEIGHT, TARGET_WIDTH, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation.',
                        ),
                        # 机器人状态：仅右臂 7 维 (6关节 + 1夹爪)
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot state (Right Arm qpos)',
                        ),
                    }),
                    # 动作：仅右臂 7 维
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action (Right Arm)',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if any, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if any, default to 0.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step if the episode finished successfully.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='K-LITE embedding (Optional, fill with zeros if using Prismatic pipeline)',
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(doc='Path to the original HDF5 file.'),
                }),
            }),
            supervised_keys=None,  # Set to `None` to disable
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """定义数据分割，这里直接读取本地路径"""
        
        # TODO: 修改这里为你的 HDF5 文件夹绝对路径
        INPUT_DIR = '/home/aox/model/xuao/flip_upright'  # 示例路径
        
        return {
            'train': self._generate_examples(path=INPUT_DIR),
            # 如果有验证集文件夹，可以加一行 'val': ...
        }

    def _generate_examples(self, path):
        """生成数据样本的核心逻辑"""
        
        # 获取所有 .hdf5 文件
        hdf5_files = sorted(glob.glob(os.path.join(path, '*.hdf5')))
        
        # 尝试读取 metadata.json 中的通用指令
        default_instruction = "do something"
        metadata_path = os.path.join(path, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                    default_instruction = meta.get('task_description_english', default_instruction)
            except:
                pass

        for file_path in hdf5_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    # 读取原始数据
                    # 假设 shape 是 (T, 14) -> [左臂7, 右臂7]
                    qpos_all = f['/observations/qpos'][:]
                    action_all = f['/action'][:]
                    
                    # === 核心修改：只切片右臂数据 ===
                    # 根据你的代码：0-6是左臂，7-13是右臂
                    qpos_right = qpos_all[:, 7:14]
                    action_right = action_all[:, 7:14]
                    
                    # 读取图像 (OpenVLA 只需要 RGB)
                    # 注意：如果你的 HDF5 存的是压缩的 bytes，需要解码；如果存的是 raw array，直接读取
                    # 这里假设和你的脚本一样，是压缩的 jpg bytes，需要解码
                    
                    # 读取主相机 (cam_high)
                    img_high_raw = f['/observations/images/cam_high'][:]
                    
                    # 读取腕部相机 (cam_right_wrist)
                    if 'cam_right_wrist' in f['/observations/images']:
                        img_wrist_raw = f['/observations/images/cam_right_wrist'][:]
                    else:
                        # 如果没有右腕相机，用全黑填充或者复制主相机(不推荐)
                        img_wrist_raw = [None] * len(img_high_raw)

                    # 隔帧采样 (Downsampling)，每隔2帧取1帧
                    img_high_raw = img_high_raw[::2]
                    img_wrist_raw = img_wrist_raw[::2]
                    qpos_right = qpos_right[::2]
                    action_right = action_right[::2]

                    num_steps = len(qpos_right)
                    episode = []
                    
                    for i in range(num_steps):
                        # 解码图像 (OpenVLA 标准输入是 RGB)
                        # tfds.features.Image 会自动处理 numpy array，
                        # 但如果为了节省 TFDS 生成时的空间，最好保持 jpg 格式不解码，让 tfds 自己去 encode。
                        # 这里为了保险，先解码成 numpy array (H, W, 3)
                        
                        # 1. 解码 High Cam
                        img_high = cv2.imdecode(np.frombuffer(img_high_raw[i], np.uint8), cv2.IMREAD_COLOR)
                        # Resize 到 (256, 256)
                        img_high = cv2.resize(img_high, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
                        # img_high = cv2.cvtColor(img_high, cv2.COLOR_BGR2RGB) # 转 RGB
                        
                        # 2. 解码 Wrist Cam
                        if img_wrist_raw[i] is not None:
                            img_wrist = cv2.imdecode(np.frombuffer(img_wrist_raw[i], np.uint8), cv2.IMREAD_COLOR)
                            img_wrist = cv2.resize(img_wrist, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
                            # img_wrist = cv2.cvtColor(img_wrist, cv2.COLOR_BGR2RGB)
                        else:
                            img_wrist = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)

                        episode.append({
                            'observation': {
                                'image': img_high,
                                'wrist_image': img_wrist,
                                'state': qpos_right[i].astype(np.float32),
                            },
                            'action': action_right[i].astype(np.float32),
                            'discount': 1.0,
                            'reward': float(i == (num_steps - 1)),
                            'is_first': i == 0,
                            'is_last': i == (num_steps - 1),
                            'is_terminal': i == (num_steps - 1),
                            'language_instruction': default_instruction,
                            'language_embedding': np.zeros(512, dtype=np.float32), # 占位符
                        })

                    # Yield 整个 episode
                    # key 需要是唯一的字符串
                    yield os.path.basename(file_path), {
                        'steps': episode,
                        'episode_metadata': {
                            'file_path': file_path
                        }
                    }
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
