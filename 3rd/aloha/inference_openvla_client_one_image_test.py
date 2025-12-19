#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import torch
import numpy as np
import os
import pickle
import argparse
from einops import rearrange


import collections
from collections import deque

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import time
import threading
import math
import threading

import sys
sys.path.append("./")

from openpi_client import websocket_client_policy, image_tools

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
from PIL import Image as PILImage
import cv2
import requests
import json
import json_numpy

json_numpy.patch()

task_config = {'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']}

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None

class OpenVLA:
    def __init__(self, server_url: str):
        self.server_url = server_url

    def predict_action(self, payload: Dict[str, Any]) -> str:
        # Parse payload components
        instruction = payload["instruction"]
        image = payload["image"]
        unnorm_key = payload.get("unnorm_key", "aloha2openvla_multi_rgb_flip_upright")

        # Run VLA Inference
        response = requests.post(
            f"{self.server_url}/act",
            json={"image": image, "instruction": instruction, "unnorm_key": unnorm_key},
        )

        if response.status_code == 200:
            action = response.json()
            return action
        else:
            print(f"Request failed with status code: {response.status_code}")
            return None

def adapt_gripper(x, x_min, x_max, c):
    c = torch.tensor(c)
    z = (x - x_min) / (x_max - x_min)
    d = 1 - torch.exp(-c)
    if d == 0:
        z_w = torch.ones_like(z)
    else:
        z_w = (1 - torch.exp(-c * z)) / d
    y = z_w * (x_max - x_min) + x_min
    return y 


def interpolate_action(args, prev_action, cur_action):
    # print(f"✅ prev_action: {prev_action}")
    # print(f"✅ cur_action: {cur_action}")
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    
    # 对夹爪关节进行特殊处理
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    
    # # 获取夹爪关节的索引（第7个关节）
    # gripper_indices = [6, 13]  # 左右夹爪的索引
    
    # # 对夹爪关节直接设置为目标位置，不进行插值
    # for idx in gripper_indices:
    #     new_actions[:, idx] = cur_action[idx]
    
    return new_actions[1:]

# def get_image(observation, camera_names):
#     curr_images = []
#     for cam_name in camera_names:
#         curr_image = rearrange(observation['images'][cam_name], 'h w c -> c h w')
    
#         curr_images.append(curr_image)
#     curr_image = np.stack(curr_images, axis=0)
#     curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
#     return curr_image


def get_depth_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_images.append(observation['images_depth'][cam_name])
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def inference_process(args, ros_operator, t, openvla_model):
    global inference_lock
    global inference_actions
    global inference_timestep
    print_flag = True
    rate = rospy.Rate(args.publish_rate)
    
    # --- [新增 1] 初始化本地图片读取配置 ---
    # 图片所在的文件夹
    dataset_img_dir = '../../finetune/saved_frames/episode_0'
    
    # 获取文件夹内所有 png 图片，并按文件名排序 (确保 frame_0000 -> frame_0001 顺序)
    if os.path.exists(dataset_img_dir):
        img_files = sorted([f for f in os.listdir(dataset_img_dir) if f.endswith('.png')])
    else:
        img_files = []
        print(f"Warning: Directory {dataset_img_dir} not found!")

    img_idx = 0 # 当前读取的图片索引
    total_imgs = len(img_files)
    print(f"Start loading images from local disk. Total frames: {total_imgs}")
    # -----------------------------------

    while True and not rospy.is_shutdown():
        # 依然调用 ros_operator 以保持时钟同步和获取其他数据(如 robot_base, arm_state)
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        
        # 解包数据
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = result
        
        # --- [新增 2] 从文件读取图片并覆盖 img_front ---
        if total_imgs > 0:
            # 构造当前帧的文件路径
            current_file = img_files[img_idx]
            img_path = os.path.join(dataset_img_dir, current_file)
            
            # 使用 opencv 读取图片
            # 注意：cv2.imread 默认读入格式为 BGR
            img_from_disk = cv2.imread(img_path)
            
            if img_from_disk is not None:
                # 转换 BGR -> RGB (因为模型通常需要 RGB，且之前的 PIL 保存也是 RGB)
                img_front = cv2.cvtColor(img_from_disk, cv2.COLOR_BGR2RGB)
                
                # (可选) 打印当前使用的是哪一张图
                # print(f"Using local image: {current_file}")
                
                # 更新索引，准备下一帧
                img_idx += 1
                # 如果读完了，可以选择循环播放或者停留在最后一帧
                if img_idx >= total_imgs:
                    img_idx = 0 # 循环播放
                    print("Episode finished, looping back to start.")
            else:
                print(f"Failed to read {img_path}")
        # ----------------------------------------------
        
        # 对图像进行resize处理
        # 这里的 img_front 已经被替换成了本地文件读取的图片
        img_front = cv2.resize(img_front, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA) 
        
        img_front = img_front.astype(np.uint8)
        vis_image = PILImage.fromarray(img_front)
        vis_image.save(f"./debug_inference/frame_{t:05d}.png")
        # img_left = np.transpose(img_left, (2, 0, 1)).astype(np.uint8)
        # img_right = np.transpose(img_right, (2, 0, 1)).astype(np.uint8)
        # print(f"✅ after img_front shape: {img_front.shape}")
        # # 准备观察数据
        # observation = {
        #     "images": {
        #         "cam_high": img_front,
        #         "cam_left_wrist": img_left,
        #         "cam_right_wrist": img_right
        #     },
        #     "state": np.concatenate(
        #         (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0
        #     ).tolist(),
        #     "prompt": args.task_instruction,
        # }
        
        # if args.use_robot_base:
        #     observation["state"] = np.concatenate(
        #         (observation["state"], [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z])
        #     ).tolist()

        payload = {
            "image": img_front.tolist(),
            "instruction": args.task_instruction,
        }
            
        # 从服务器获取动作
        start_time = time.time()
        # action_chunk_ori = client.infer(observation)["actions"]
        action_chunk_ori = openvla_model.predict_action(payload)
        action_chunk = action_chunk_ori.copy()
        # print("right_gripper", action_chunk[:, 13])
        # print("left_gripper", action_chunk[:, 6])
        # action_chunk[:, 6] -= 0.004
        # action_chunk[:, 13] -= 0.004
        # action_chunk[:, 6] = np.where(action_chunk[:, 6] > 0.02, action_chunk[:, 6], 0.001)
        # action_chunk[:, 13] = np.where(action_chunk[:, 13] > 0.02, action_chunk[:,13], 0.0001)
        # print("A_right_gripper", action_chunk[:, 13])
        # print("A_left_gripper", action_chunk[:, 6])
        # print(action_chunk.shape)
        end_time = time.time()
        print("model cost time: ", end_time -start_time)
        # left_gripper [0.001, 0.0988]
        # right_gripper [0.0001, 0.0992]


        inference_lock.acquire()
        inference_actions = action_chunk
        inference_timestep = t
        inference_lock.release()
        break

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def model_inference(args, ros_operator, save_episode=True):
    global inference_lock
    global inference_actions
    global inference_timestep
    global inference_thread
    set_seed(1000)

    # create openvla model
    openvla_model = OpenVLA(server_url=f"http://{args.server_host}:{args.server_port}")

    # 使用参数中的max_publish_step
    max_publish_step = args.max_publish_step

    # 发布基础的姿态
    left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
    right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
    left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
    # right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]
    right1 = [0.0377837, 1.4326408, -1.1984551, 0.5086321, 0.89058596, -0.1972742, 0.0328]

    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("Enter any key to continue :")
    ros_operator.puppet_arm_publish_continuous(left1, right1)
    action = None
    chunk_size = args.action_chunk_size
    # pre_action = np.array(
    #     [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258] + 
    #     [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]
    # )
    pre_action = np.array(left1 + right1)
    # 推理
    with torch.inference_mode():
        while True and not rospy.is_shutdown():
            # 每个回合的步数
            t = 0
            rate = rospy.Rate(args.publish_rate)
            action_buffer = np.zeros([1, 7])
            start_time = time.time()
            while t < max_publish_step and not rospy.is_shutdown():
                
                inference_thread = threading.Thread(target=inference_process,
                                                args=(args, ros_operator, t, openvla_model))
                inference_thread.start()
                inference_thread.join()
                inference_lock.acquire()
                if inference_actions is not None:
                    inference_thread = None
                    action_buffer = inference_actions
                    inference_actions = None
                inference_lock.release()
                    # print(f"✅ action_buffer: {action_buffer}")

                action = action_buffer
                # Interpolate the original action sequence
                # print(f"args.use_actions_interpolation: {args.use_actions_interpolation}")
                if args.use_actions_interpolation:
                    # print(f"Time {t}, pre {pre_action}, act {action}")
                    interp_actions = interpolate_action(args, pre_action, action)                
                else:
                    interp_actions = action[np.newaxis, :]
                # Execute the interpolated actions one by one
                # print(f"✅ interp_actions: {interp_actions}")
                for act in interp_actions:
                    left_action = left1
                    right_action = act[:7]
                    # for debug here
                    print("left_action", left_action)
                    print("right_action", right_action)
                    input("Enter any key to continue :")
                    # print("r_gripper", right_action[6])
                    # if right_action[6]>0.077:
                    #     print("before", act[13])
                    #     right_action[6] = 0.1
                    #     print("after", right_action[6])
                    # right_action[-1] -= delta
                    # print(f"args.disable_puppet_arm: {args.disable_puppet_arm}")
                    if not args.disable_puppet_arm:
                        ros_operator.puppet_arm_publish(left_action, right_action)  # puppet_arm_publish_continuous_thread
                
                    if args.use_robot_base:
                        vel_action = act[14:16]
                        ros_operator.robot_base_publish(vel_action)
                    rate.sleep()
                    # print(f"doing action: {act}")
                t += 1
                
                print("Published Step", t, "used time is: ", time.time() - start_time)
                pre_action = action.copy()



class RosOperator:
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def puppet_arm_publish(self, left, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = left_arm
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            step += 1
            print("puppet_arm_publish_continuous:", step)
            rate.sleep()

    def puppet_arm_publish_linear(self, left, right):
        num_step = 100
        rate = rospy.Rate(200)

        left_arm = None
        right_arm = None

        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break

        traj_left_list = np.linspace(left_arm, left, num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)

        for i in range(len(traj_left_list)):
            traj_left = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1] = left[-1]
            traj_right[-1] = right[-1]
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = traj_left
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = traj_right
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            rate.sleep()

    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.puppet_arm_publish_continuous, args=(left, right))
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def ctrl_callback(self, msg):
        self.ctrl_state_lock.acquire()
        self.ctrl_state = msg.data
        self.ctrl_state_lock.release()

    def get_ctrl_state(self):
        self.ctrl_state_lock.acquire()
        state = self.ctrl_state
        self.ctrl_state_lock.release()
        return state

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)


def get_arguments():
    parser = argparse.ArgumentParser()
  
  
    parser.add_argument('--temporal_agg', action='store', type=bool, help='temporal_agg', default=True, required=False)



    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/puppet/pos_joint_cmd_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/puppet/pos_joint_cmd_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, help='publish_rate',
                        default=50, required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=0, required=False)
    # parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
    #                     default=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05], required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                        default=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], required=False)

    parser.add_argument('--use_actions_interpolation', action='store', type=bool, help='use_actions_interpolation',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)
    parser.add_argument('--disable_puppet_arm', action='store_true',
                        help='Whether to disable the puppet arm. This is useful for safely debugging',default=False)

    parser.add_argument('--img_size', action='store', type=int, help='img_size', default=224, required=False)
    parser.add_argument('--server_host', action='store', type=str, help='server_host', default='127.0.1.1', required=False)
    parser.add_argument('--server_port', action='store', type=int, help='server_port', default=8000, required=False)
    parser.add_argument('--action_chunk_size', action='store', type=int, help='action_chunk_size', default=16, required=False)
    parser.add_argument('--task_instruction', 
                       type=str,
                       default="Searching through a cluttered box for the 'Innovation Dragon' doll, remove the obstructing items and place them on either side of the table, until the doll is retrieved and handed forward.",
                       help='任务指令描述') #Grasp objects on the table and put them into the box.
    # Use the right arm to cut a small slice from the sausage. If the sausage is placed horizontally, cut from the left end; if it's placed vertically, cut from the bottom. If the sliced piece sticks to the knife, use the left arm to wipe it off. Then return the knife to the knife holder.
    # task61 Use the right arm to hold the knife and the left arm to press the knife tip, then cut a small slice of sausage; if the slice sticks to the knife, use the left arm to wipe it off, then put the knife back on the knife rack.
    # task100 Using the right gripper, pick up the match, and with the left gripper, pick up the matchbox. Then, strike the match to ignite it, place the lit match into the water cup, and finally, return the matchbox to where it started.
    # task105 Searching through a cluttered box for the 'Innovation Dragon' doll, remove the obstructing items and place them on either side of the table, until the doll is retrieved and handed forward.
    parser.add_argument('--max_publish_step', 
                       type=int,
                       default=50000,
                       help='最大发布步数')
    parser.add_argument('--openvla_path', 
                       type=str,
                       default="models/openvla_v01_b16_ft_robotics",
                       help='OpenVLA模型路径')
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    model_inference(args, ros_operator, save_episode=True)


if __name__ == '__main__':
    main()

# python inference_pi_remote_v2.py --task_instruction "Grasp objects on the table and put them into the box."


# uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_aloha_task49_merge --policy.dir=checkpoints/task49_merge/29999

# uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_aloha_absjoint --policy.dir=checkpoints/task61_merge_v3/29999

# uv run scripts/serve_policy.py policy:checkpoint --policy.config=pick_cut --policy.dir=checkpoints/pick_cut/39999
# uv run scripts/serve_policy.py policy:checkpoint --policy.config=sii_demo2 --policy.dir=checkpoints/new_aloha_task105_0908_4_7_8_9_10_99999/99999

# uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_aloha --policy.dir=.cache/openpi-assets/checkpoints/pi0_base


