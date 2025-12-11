#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8

import torch
import numpy as np
import os
import pickle
import argparse
from einops import rearrange
import pandas as pd  # 新增: 用于读取CSV

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
import sys
import cv2

# 假设不需要 import openpi_client 了，因为不走网络
# from openpi_client import websocket_client_policy, image_tools

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
from PIL import Image as PILImage

# 保持原有的一些配置
task_config = {'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']}

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None

class CSVPolicy:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        print(f"Loading actions from CSV: {self.csv_path}")
        try:
            # 读取 CSV 文件
            # 假设 CSV 没有 header，或者是标准的 Aloha 格式
            # 如果 CSV 有 header，pandas 会自动处理
            # 这里的假设是 CSV 的每一行包含了 14 个关节数据 (左7 + 右7)
            # 或者 16 个数据 (左7 + 右7 + base 2)
            self.df = pd.read_csv(self.csv_path)
            
            # 如果你的 CSV 包含除了动作以外的列（比如时间戳、图像路径等），
            # 你需要在这里进行筛选。
            # 例如：如果只是纯数据的 CSV，可以直接用 .values
            self.actions = self.df.values
            
            # 如果你的 CSV 包含 header 且列名包含 'action' 或具体关节名，建议在这里筛选列
            # 示例：假设 CSV 是纯动作数据，不需要筛选
            
            self.total_steps = len(self.actions)
            print(f"Loaded {self.total_steps} steps.")
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            sys.exit(1)

    def get_action(self, step_index: int) -> np.ndarray:
        """
        根据当前步数获取动作
        """
        if step_index < self.total_steps:
            # 获取第 step_index 行的数据
            action = self.actions[step_index]
            
            # 确保数据是 float 类型
            action = action.astype(np.float32)
            
            # 因为原来的代码 OpenVLA 返回的是 [chunk_size, action_dim]
            # 这里我们模拟一个 chunk_size=1 的返回，或者直接返回一行
            # 原代码逻辑处理了 batch 维度，所以我们增加一个维度变为 [1, 14]
            if len(action.shape) == 1:
                action = action[np.newaxis, :]
                
            return action
        else:
            print("End of CSV actions.")
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
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]

def inference_process(args, ros_operator, t, policy_model):
    global inference_lock
    global inference_actions
    global inference_timestep
    
    # 即使是从 CSV 读取，为了保持与真实机器人同步，
    # 我们依然等待新的帧到达（保持原来的控制频率节奏）
    # 如果只是单纯回放不需要看图，可以注释掉下面等待帧的逻辑
    
    rate = rospy.Rate(args.publish_rate)
    print_flag = True
    
    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("waiting for syn frame...")
                print_flag = False
            rate.sleep()
            continue
        
        # 虽然我们要从 CSV 读动作，不需要处理图像，
        # 但原来的代码在这里获取了 robot_base 等信息，保留结构以免破坏逻辑
        # (img_front, img_left, img_right, ..., robot_base) = result
        
        # 从 CSV 获取动作
        start_time = time.time()
        
        # 核心修改：从 policy_model (CSVPolicy) 获取动作
        action_chunk = policy_model.get_action(t)
        
        if action_chunk is None:
            print("No more actions in CSV or error.")
            # 可以选择在这里退出或者发送停止指令
            break

        end_time = time.time()
        # print("csv read time: ", end_time - start_time)

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

    # Modified: Initialize CSV Policy instead of OpenVLA
    csv_policy = CSVPolicy(csv_path=args.csv_path)

    # 使用参数中的max_publish_step，如果CSV更短，则受限于CSV长度
    max_publish_step = min(args.max_publish_step, csv_policy.total_steps)

    # 发布基础的姿态 (保持不变，用于复位初始状态)
    left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
    right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
    left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
    right1 = [0.0377837, 1.4326408, -1.1984551, 0.5086321, 0.89058596, -0.1972742, 0.0328]

    print("Moving to initial position...")
    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("Enter any key to continue (Move to Start Position):")
    ros_operator.puppet_arm_publish_continuous(left1, right1)
    
    input("Enter any key to start playing CSV actions:")

    action = None
    # 这里的 pre_action 初始化为 left1 + right1，防止第一帧插值跳跃太大
    # 注意：如果 CSV 的第一帧和这个位置差别很大，可能会有剧烈运动
    pre_action = np.array(left1 + right1)
    
    with torch.inference_mode():
        t = 0
        rate = rospy.Rate(args.publish_rate)
        action_buffer = np.zeros([1, 14]) # 假设14维
        start_time = time.time()
        
        while t < max_publish_step and not rospy.is_shutdown():
            
            # 启动线程获取 CSV 里的这一步动作
            # (虽然读CSV很快不需要线程，但为了改动最小，保持原结构)
            inference_thread = threading.Thread(target=inference_process,
                                            args=(args, ros_operator, t, csv_policy))
            inference_thread.start()
            inference_thread.join()
            
            inference_lock.acquire()
            if inference_actions is not None:
                inference_thread = None
                action_buffer = inference_actions
                inference_actions = None
            else:
                # 如果没拿到动作（比如CSV读完了）
                inference_lock.release()
                print("Finished or Error.")
                break
            inference_lock.release()

            action = action_buffer
            
            # 处理插值
            if args.use_actions_interpolation:
                # 假设 action 是 [1, 14]
                cur_act_flat = action[0] if len(action.shape) > 1 else action
                interp_actions = interpolate_action(args, pre_action, cur_act_flat)                
            else:
                interp_actions = action if len(action.shape) > 1 else action[np.newaxis, :]

            # 执行动作
            for act in interp_actions:
                # 根据 CSV 的结构，这里需要注意：
                # 假设 CSV 的前7列是左手，后7列是右手（或者是反过来的，根据你的数据源）
                # 下面的代码假设：act[:7] 是左手，act[7:14] 是右手
                # 原来的代码有点特殊：left_action 居然是固定的 left1，只动右手？
                # 原代码: left_action = left1
                # 原代码: right_action = act[:7] (看起来原模型只输出了右手的7维？)
                
                # --- 修改逻辑开始 ---
                # 如果你的 CSV 包含双臂数据 (14维)：
                if act.shape[0] >= 14:
                    left_action = act[:7]
                    right_action = act[7:14]
                else:
                    # 如果 CSV 只有7维，假设是右手，左手保持不动
                    left_action = left1
                    right_action = act[:7]
                # --- 修改逻辑结束 ---

                # print("left_action", left_action)
                # print("right_action", right_action)
                
                if not args.disable_puppet_arm:
                    ros_operator.puppet_arm_publish(left_action, right_action)
                
                if args.use_robot_base and act.shape[0] >= 16:
                    vel_action = act[14:16]
                    ros_operator.robot_base_publish(vel_action)
                
                rate.sleep()
            
            t += 1
            # print("Published Step", t, "used time is: ", time.time() - start_time)
            
            # 更新 pre_action 用于下一次插值
            # 如果是双臂，pre_action 应该是 14维
            if action.shape[1] >= 14:
                 pre_action = action[0][:14]
            else:
                 # 兼容单臂情况
                 pre_action = np.concatenate([left1, action[0][:7]])

class RosOperator:
    # ... (保持原有的 RosOperator 类代码完全不变) ...
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
            # print("puppet_arm_publish_continuous:", step)
            rate.sleep()

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
        
        # 简化版：这里不需要真正去解压图像，因为我们只用来做同步
        # 为了提高性能，直接 return True 和占位符，或者保持原样
        # 这里保持原样以确保逻辑一致性
        return (None, None, None, None, None, None, None, None, None) 

        # ... (如果需要实际图像处理，请取消注释下面的代码) ...
        # while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
        #     self.img_left_deque.popleft()
        # img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')
        # ... (以此类推)

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

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        # 订阅逻辑保持不变
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        
        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)


def get_arguments():
    parser = argparse.ArgumentParser()
    
    # 新增参数：CSV文件路径
    parser.add_argument('--csv_path', action='store', type=str, required=True, 
                        help='Path to the .csv file containing actions')

    parser.add_argument('--temporal_agg', action='store', type=bool, help='temporal_agg', default=True, required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, default='/camera_r/color/image_raw', required=False)
    
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, default='/puppet/joint_right', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, default=50, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, default=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], required=False)

    parser.add_argument('--use_actions_interpolation', action='store', type=bool, default=False, required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, default=False, required=False)
    parser.add_argument('--disable_puppet_arm', action='store_true', default=False)
    parser.add_argument('--img_size', action='store', type=int, default=224, required=False)
    parser.add_argument('--max_publish_step', type=int, default=50000)

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    model_inference(args, ros_operator, save_episode=True)


if __name__ == '__main__':
    main()