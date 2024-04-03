import argparse
import copy
import os
import sys
import time
from typing import Optional

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

import numpy as np
import sapien.core as sapien
import transforms3d
from gym.utils import seeding
from sapien.utils import Viewer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from hand_teleop.env.rl_env.base import BaseRLEnv
from hand_teleop.env.sim_env.relocate_env import LabRelocateEnv, RelocateEnv
from hand_teleop.real_world import lab


class RelocateRLEnv(RelocateEnv, BaseRLEnv):
    def __init__(
        self,
        use_gui=False,
        frame_skip=5,
        robot_name="adroit_hand_free",
        constant_object_state=False,
        rotation_reward_weight=0,
        object_category="YCB",
        object_name="tomato_soup_can",
        object_scale=1.0,
        randomness_scale=1,
        friction=1,
        object_pose_noise=0.01,
        **renderer_kwargs,
    ):
        super().__init__(
            use_gui,
            frame_skip,
            object_category,
            object_name,
            object_scale,
            randomness_scale,
            friction,
            **renderer_kwargs,
        )
        self.setup(robot_name)
        self.constant_object_state = constant_object_state
        self.rotation_reward_weight = rotation_reward_weight
        self.object_pose_noise = object_pose_noise

        # Parse link name
        self.palm_link_name = self.robot_info.palm_name
        # self.palm_link = [link for link in self.robot.get_links() if link.get_name() == self.palm_link_name][0]
        for link in self.robot.get_links():
            # print(link.get_name())
            name = link.get_name()
            if name == self.palm_link_name:
                self.palm_link = link
            elif name == "link_3.0_tip":  # shizhi
                self.link3tip = link
            elif name == "link_7.0_tip":  # zhongzhi
                self.link7tip = link
            elif name == "link_11.0_tip":  # wumingzhi
                self.link11tip = link
            elif name == "link_15.0_tip":  # muzhi
                self.link15tip = link

        # Object init pose

        self.object_episode_init_pose = sapien.Pose()

    def get_oracle_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        object_pose = (
            self.object_episode_init_pose if self.constant_object_state else self.manipulated_object.get_pose()
        )
        object_pose_vec = np.concatenate([object_pose.p, object_pose.q])
        palm_pose = self.palm_link.get_pose()
        target_in_object = self.target_pose.p - object_pose.p
        target_in_palm = self.target_pose.p - palm_pose.p
        object_in_palm = object_pose.p - palm_pose.p
        palm_v = self.palm_link.get_velocity()
        palm_w = self.palm_link.get_angular_velocity()
        theta = np.arccos(np.clip(np.power(np.sum(object_pose.q * self.target_pose.q), 2) * 2 - 1, -1 + 1e-8, 1 - 1e-8))
        return np.concatenate(
            [
                robot_qpos_vec,
                object_pose_vec,
                palm_v,
                palm_w,
                object_in_palm,
                target_in_palm,
                target_in_object,
                self.target_pose.q,
                np.array([theta]),
            ]
        )

    def get_robot_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        palm_pose = self.palm_link.get_pose()
        return np.concatenate([robot_qpos_vec, palm_pose.p, self.target_pose.p, self.target_pose.q])

    def check_collision(self):
        is_contact = self.check_contact(self.robot_collision_links, [self.manipulated_object], impulse_threshold=1e-4)
        return is_contact

    def get_reward(self, action):
        object_pose = self.manipulated_object.get_pose()
        palm_pose = self.palm_link.get_pose()
        is_contact = self.check_contact(self.robot_collision_links, [self.manipulated_object])
        return is_contact
        reward = -0.1 * min(np.linalg.norm(palm_pose.p - object_pose.p), 0.5)
        if is_contact:
            reward += 0.1
            lift = min(object_pose.p[2], self.target_pose.p[2]) - self.object_height
            lift = max(lift, 0)
            reward += 5 * lift
            if lift > 0.015:
                reward += 2
                obj_target_distance = min(np.linalg.norm(object_pose.p - self.target_pose.p), 0.5)
                reward += -1 * min(np.linalg.norm(palm_pose.p - self.target_pose.p), 0.5)
                reward += -3 * obj_target_distance  # make object go to target

                if obj_target_distance < 0.1:
                    reward += (0.1 - obj_target_distance) * 20
                    theta = np.arccos(
                        np.clip(np.power(np.sum(object_pose.q * self.target_pose.q), 2) * 2 - 1, -1 + 1e-8, 1 - 1e-8)
                    )
                    reward += max((np.pi / 2 - theta) * self.rotation_reward_weight, 0)
                    if theta < np.pi / 4 and self.rotation_reward_weight >= 1e-6:
                        reward += (np.pi / 4 - theta) * 6 * self.rotation_reward_weight

        return reward

    def set_target_tips(self, target_tip_pos):
        self.target_tip_pos = target_tip_pos

    def get_robot_tips(self):
        tip_pos = []
        for each in [self.link3tip, self.link7tip, self.link11tip, self.link15tip]:
            tip_pos.append(each.get_pose().p)
        tip_pos = np.array(tip_pos)
        return tip_pos

    def set_target_object_pose(self, object_pose):
        self.target_object_pose = object_pose

    def get_mpc_reward(self):
        object_pose = self.manipulated_object.get_pose()
        target_object_pose = self.target_object_pose
        # palm_pose = self.link11tip.get_pose()
        tips_pos = self.get_robot_tips()
        reward = -0.1 * np.linalg.norm(self.target_tip_pos.flatten() - tips_pos.flatten())
        reward = reward - np.linalg.norm(target_object_pose.p - object_pose.p)
        reward = reward - np.linalg.norm(target_object_pose.p - object_pose.p)
        return reward

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        # super().reset(seed=seed)
        if not self.is_robot_free:
            qpos = np.zeros(self.robot.dof)
            xarm_qpos = self.robot_info.arm_init_qpos
            qpos[: self.arm_dof] = xarm_qpos
            self.robot.set_qpos(qpos)
            self.robot.set_drive_target(qpos)
            init_pos = np.array(lab.ROBOT2BASE.p) + self.robot_info.root_offset
            init_pose = sapien.Pose(init_pos, transforms3d.euler.euler2quat(0, 0, 0))
        else:
            init_pose = sapien.Pose(np.array([-0.4, 0, 0.2]), transforms3d.euler.euler2quat(0, np.pi / 2, 0))
        self.robot.set_pose(init_pose)
        self.reset_internal()
        self.object_episode_init_pose = self.manipulated_object.get_pose()
        random_quat = transforms3d.euler.euler2quat(*(self.np_random.randn(3) * self.object_pose_noise * 10))
        random_pos = self.np_random.randn(3) * self.object_pose_noise
        self.object_episode_init_pose = self.object_episode_init_pose * sapien.Pose(random_pos, random_quat)
        return self.get_observation()

    # @cached_property
    def obs_dim(self):
        if not self.use_visual_obs:
            return self.robot.dof + 7 + 6 + 9 + 4 + 1
        else:
            return len(self.get_robot_state())

    def is_done(self):
        return False
        return self.current_step >= self.horizon

    # @cached_property
    def horizon(self):
        return 250


def main_env():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="output/2024_03_22_16_56_36/sample/result_filter_sapien/010_potted_meat_can.npy",
    )
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()

    data_path = args.data_path
    vis = args.vis
    # ============== some hardcoded parameters ==============
    loop_for_selection = False
    height = 0.04
    pre_step_num = 10
    post_step_num = 15
    # ======================================================

    limit = np.array(
        [
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.57, 1.57],
            [-1.57, 1.57],
            [-1.57, 1.57],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
        ]
    )

    object_name = data_path.split("/")[-1].split(".")[0][4:]
    file_prefix = data_path.split("/")[-1].split(".")[0]
    print("Loading object:", object_name)

    init_tls = {
        "potted_meat_can": [0.0014802, -0.0478232, 0.0505776],
        "tomato_soup_can": [0.0014802, -0.0478232, 0.0601341],
        "mustard_bottle": [0.0014483, -0.0479091, 0.084818],
        "bleach_cleanser": [0.0014483, -0.0479091, 0.112906],
        "mug": [0.0014802, -0.0478232, 0.0384848],
        "banana": [0.0014802, -0.0478232, 0.0177182],
        "bowl": [0.0014802, -0.0478232, 0.0241855],
        "ball": [0.0014802, -0.0478232, 0.05],
        "power_drill": [0.0014802, -0.0478232, 0.0255277],
        "large_clamp": [0.0014802, -0.0478232, 0.0204929],
    }
    retargets_data = np.load(data_path)
    print("retargets_data.shape", retargets_data.shape)
    env = RelocateRLEnv(
        use_gui=vis, robot_name="allegro_hand_free", object_name=object_name, frame_skip=5, use_visual_obs=False
    )
    base_env = env
    np.random.seed(0)
    env.seed(0)
    env.reset()

    q_limits = env.robot.get_qlimits()
    q_limits[:6] = limit[:6]

    if vis:
        viewer = Viewer(base_env.renderer)
        viewer.set_scene(base_env.scene)
        viewer.toggle_pause(True)
        base_env.viewer = viewer

    prev_success_id = []
    success_id = list(range(len(retargets_data)))
    pre_trajectories, post_trajectories, object_trajectories = {}, {}, {}

    init_tl = init_tls[object_name]
    full_length_per_traj = pre_step_num + post_step_num + len(retargets_data[0])

    begin = time.time()
    while len(prev_success_id) != len(success_id):
        print("len(prev_success_id):", len(prev_success_id))
        print("prev_success_id:", prev_success_id)
        prev_success_id = copy.deepcopy(success_id)
        success_id = []
        for jj in prev_success_id:
            np.random.seed(0)
            env.seed(0)
            env.reset()

            retargets = retargets_data[jj]
            env.manipulated_object.set_pose(sapien.Pose(init_tl, [1, 0, 0, 0]))

            for i in range(pre_step_num):
                action_for_rl = retargets[0].copy()
                action_for_rl = (action_for_rl - env.robot.get_qpos()) / 0.05
                action_for_rl = (action_for_rl - q_limits[:, 0]) * 2 / (q_limits[:, 1] - q_limits[:, 0]) - 1
                obs, reward, done, info = env.step(action_for_rl)
                # print(env.robot.get_qpos(), retargets[0])

            pre_trajectory, post_trajectory, object_trajectory = [], [], []
            for t in range(len(retargets)):
                # print("env.manipulated_object:", env.manipulated_object.get_pose())
                action_for_rl = retargets[t].copy()
                action_for_rl[:6] = (action_for_rl[:6] - env.robot.get_qpos()[:6]) / 0.03
                action_for_rl = (action_for_rl - q_limits[:, 0]) * 2 / (q_limits[:, 1] - q_limits[:, 0]) - 1
                obs, reward, done, info = env.step(action_for_rl)
                pre_trajectory.append(retargets[t].copy())
                post_trajectory.append(env.robot.get_qpos())

                if vis:
                    for _ in range(10):
                        env.render()

            for i in range(post_step_num):
                action_for_rl = retargets[-1].copy()
                action_for_rl[0] = action_for_rl[0] - 0.01 * (i + 1)
                action_for_rl[:6] = (action_for_rl[:6] - env.robot.get_qpos()[:6]) / 0.03
                action_for_rl = (action_for_rl - q_limits[:, 0]) * 2 / (q_limits[:, 1] - q_limits[:, 0]) - 1
                obs, reward, done, info = env.step(action_for_rl)
                if vis:
                    for _ in range(10):
                        env.render()

            pre_trajectories[str(jj)] = np.asarray(pre_trajectory)
            post_trajectories[str(jj)] = np.asarray(post_trajectory)

            if (reward is True) and ((env.manipulated_object.get_pose().p[2]) > init_tl[2] + height):
                print(jj, "success")
                success_id.append(jj)

        if not loop_for_selection:
            break

    print("success num", len(success_id))
    print("successid", success_id)
    os.makedirs("output/ours", exist_ok=True)
    np.savez_compressed(f"output/ours/{file_prefix}_post_trajectories.npz", **post_trajectories)
    np.savez_compressed(f"output/ours/{file_prefix}_pre_trajectories.npz", **pre_trajectories)

    total_cost = full_length_per_traj * len(retargets_data)
    total_time = time.time() - begin
    print(f"method ours obj_name {object_name}")
    print(
        f"traj_num {len(retargets_data)} success_num {len(success_id)} success_rate {len(success_id) / len(retargets_data):.8f}"
    )
    print(
        f"total_cost {total_cost} cost_per_success {total_cost / (len(success_id) + 1e-6):.6f} "
        f"cost_per_success_log_10 {np.log10(total_cost / (len(success_id) + 1e-6)):.6f} "
        f"cost_per_traj {total_cost / len(retargets_data):.6f}"
    )
    print(
        f"total_time {total_time} time_per_success {total_time / (len(success_id) + 1e-6):.6f} time_per_traj {total_time / len(retargets_data):.6f}"
    )


if __name__ == "__main__":
    main_env()
