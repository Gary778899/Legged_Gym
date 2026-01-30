
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import numpy as np

class X2Robot(LeggedRobot):
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        self.gait_period = self.cfg.rewards.gait_period
        self.gait_offset = self.cfg.rewards.gait_offset
        self.stance_threshold = self.cfg.rewards.stance_threshold
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.leg_phase = torch.zeros(self.num_envs, self.feet_num, device=self.device)
        

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        self.phase = (self.episode_length_buf * self.dt) % self.gait_period / self.gait_period
        self.phase_left = self.phase
        self.phase_right = (self.phase + self.gait_offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    
    def _reward_feet_swing_height(self):
        # (num_envs, num_feet) bool
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * (~contact)
        return torch.sum(pos_error, dim=1)
    
    def _reward_alive(self):
        # Reward for staying alive
        return torch.ones(self.num_envs, device=self.device)
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        # Penalize hip deviation from default position
        # hip roll joints are at indices 1 , 2, 7 and 8
        idx = torch.tensor([1,2,7,8], device=self.device)
        err = self.dof_pos[:, idx] - self.default_dof_pos[:, idx]
        return torch.sum(err * err, dim=1)


    def _reward_waist_pos(self):
        # Penalize waist deviation from default position
        # waist joints are at indices 12, 13, 14 (after the 12 leg joints)
        idx = torch.tensor([12,13,14], device=self.device)
        err = self.dof_pos[:, idx] - self.default_dof_pos[:, idx]
        return torch.sum(err * err, dim=1)

    def _reward_foot_near(self, threshold: float = 0.2):
        """
        惩罚双脚距离过近，防止交叉和碰撞，确保稳定的支撑基面
        
        【设计原理】
        1. 稳定性要求：支撑基面（两脚之间的多边形区域）需要足够大
        2. 防碰撞：避免双脚在摆动过程中相互碰撞
        3. 自然步态：保持合理的步宽（类似人类行走的步宽约15-20cm）
        
        【惩罚机制】
        - 当双脚距离 < threshold 时，施加惩罚
        - 距离越近，惩罚越大（线性递增）
        - 距离 >= threshold 时，惩罚为0
        
        【阈值建议】
        - X2髋宽约30-40cm
        - 推荐阈值：0.15-0.25m
        - 平地训练：0.2m（默认）
        - 窄道/精确控制：0.15m
        
        Args:
            threshold: 最小安全距离（米），默认0.2m
            
        Returns:
            torch.Tensor: 每个环境的惩罚值，形状为 (num_envs,)
                         范围: [0, threshold]，0表示无惩罚
        """
        # 获取两只脚的位置 (num_envs, 2, 3) -> 每只脚的世界坐标(x,y,z)
        # self.feet_pos 已在 _init_foot() 中初始化
        left_foot_pos = self.feet_pos[:, 0, :]   # (num_envs, 3)
        right_foot_pos = self.feet_pos[:, 1, :]  # (num_envs, 3)
        
        # 计算两脚之间的XY距离
        distance = torch.norm((left_foot_pos - right_foot_pos)[:, :2], dim=-1)  # (num_envs,)
        
        # 惩罚计算：当距离 < threshold 时，惩罚 = threshold - distance
        # 使用 clamp(min=0) 确保距离 >= threshold 时惩罚为0
        # 例如：threshold=0.2, distance=0.1 -> penalty=0.1
        #      threshold=0.2, distance=0.25 -> penalty=0
        reward = (threshold - distance).clamp(min=0)
        
        return reward

    """
    Gait rewards.
    """

    def _reward_feet_gait(self):
        """
        奖励脚部接触模式与预期步态节奏的一致性（步态同步奖励）
        
        【参数调优指南】
        period (步态周期):
        - 0.5-0.7s: 快速步态，适合高速运动
        - 0.8-1.0s: 标准步态，平衡速度和稳定性（推荐）
        - 1.0-1.5s: 慢速步态，适合复杂地形
        
        offset (相位偏移):
        - [0.0, 0.5]: 对侧步态（走步/trot），最常用
        - [0.0, 0.0]: 同步步态（跳跃），双足同时动作
        - [0.0, 0.25, 0.5, 0.75]: 四足步态
        
        stance_threshold (支撑相比例):
        - 0.5: 支撑和摆动时间相等（50%-50%）
        - 0.55-0.6: 更多支撑时间，更稳定（推荐）
        - 0.4-0.45: 更多摆动时间，更快速
        
        Returns:
            torch.Tensor: 每个环境的奖励值，形状为 (num_envs,)
                         范围: [0, num_feet]，满分为脚的数量
        """
        stance_threshold = self.stance_threshold
        command_threshold = self.cfg.rewards.command_threshold
        is_stance_expected = self.leg_phase < stance_threshold
        is_contact_actual  = self.contact_forces[:, self.feet_indices, 2] > 1.0
        is_correct = ~(is_stance_expected ^ is_contact_actual)
        reward = torch.sum(is_correct.float(), dim=1)

        if command_threshold > 0:
            is_moving = torch.norm(self.commands[:, :3], dim=1) > command_threshold
            reward = reward * is_moving.float()
        return reward


    def _reward_joint_mirror(self, mirror_pairs: list = None):
        """
        惩罚左右关节位置不对称，鼓励对称运动模式
        【参数调优】
        - 权重 -0.5 到 -2.0: 适度鼓励对称
        - 权重 -2.0 到 -5.0: 强制对称（可能限制灵活性）
        - 权重 0: 允许完全非对称（不推荐）
        
        Args:
            mirror_pairs: 镜像关节对的索引列表，格式为 [[left_idx, right_idx], ...]
                         如果为None，则使用X2的默认配置
            
        Returns:
            torch.Tensor: 每个环境的惩罚值，形状为 (num_envs,)
                         值越小越对称，0表示完美对称
        """
        # 默认的X2镜像关节对（左腿vs右腿）
        if mirror_pairs is None:
            mirror_pairs = [
                [0, 6],   # hip_yaw
                [1, 7],   # hip_roll
                [2, 8],   # hip_pitch
                [3, 9],   # knee
                [4, 10],  # ankle_pitch
                [5, 11],  # ankle_roll
            ]
        
        # 累加所有镜像对的位置差异平方
        reward = torch.zeros(self.num_envs, device=self.device)
        
        for left_idx, right_idx in mirror_pairs:
            # 计算左右关节位置差的平方
            # dof_pos 形状: (num_envs, num_dof)
            pos_diff = self.dof_pos[:, left_idx] - self.dof_pos[:, right_idx]
            reward += torch.square(pos_diff)
        
        # 归一化：除以镜像对数量，使得惩罚值不依赖于关节数
        reward = reward / len(mirror_pairs)
        
        return reward