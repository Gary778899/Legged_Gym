
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym.torch_utils import quat_rotate_inverse
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
        self._step_last_contacts = torch.zeros(
            (self.num_envs, self.feet_num), dtype=torch.bool, device=self.device, requires_grad=False
        )
        

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
        # Penalize waist pitch/roll deviation from default position.
        # NOTE: yaw is excluded to avoid fighting turning behaviors.
        # waist joints are at indices 12, 13, 14 (after the 12 leg joints)
        idx = torch.tensor([13, 14], device=self.device)  # pitch, roll
        err = self.dof_pos[:, idx] - self.default_dof_pos[:, idx]
        weights = torch.tensor([2.0, 1.0], device=self.device)  # pitch > roll
        err = err * weights
        return torch.sum(err * err, dim=1)

    def _reward_foot_near(self, threshold: float = 0.2):
        """
        Penalizes the robot when its feet are too close together in the horizontal plane.
        
        Args:
            threshold: Minimum safe distance (meters), default is 0.2m
            
        Returns:
            torch.Tensor: Penalty value for each environment, shape (num_envs,)
                         Range: [0, threshold], 0 means no penalty
        """
        # Acquire left and right foot positions
        left_foot_pos = self.feet_pos[:, 0, :]   # (num_envs, 3)
        right_foot_pos = self.feet_pos[:, 1, :]  # (num_envs, 3)
        
        # Calculate horizontal (XY plane) distance between feet
        distance = torch.norm((left_foot_pos - right_foot_pos)[:, :2], dim=-1)  # (num_envs,)
        
        # Penalty calculation: when distance < threshold, penalty = threshold - distance
        # Use clamp(min=0) to ensure penalty is 0 when distance >= threshold
        reward = (threshold - distance).clamp(min=0)
        
        return reward

    def _reward_feet_width(self):
        """Penalize deviation from a target horizontal distance between the feet.

        This is a simple way to tune stance width (too wide vs too narrow).
        """
        if self.feet_num < 2:
            return torch.zeros(self.num_envs, device=self.device)

        left_foot_pos = self.feet_pos[:, 0, :]
        right_foot_pos = self.feet_pos[:, 1, :]
        # Use lateral (Y-axis) separation only so we don't penalize normal fore-aft stepping.
        distance_y = torch.abs(left_foot_pos[:, 1] - right_foot_pos[:, 1])
        target = float(self.cfg.rewards.feet_width_target)
        penalty = torch.square(distance_y - target)
        # Only shape width when actually walking; avoid fighting stand-still behavior.
        penalty *= torch.norm(self.commands[:, :2], dim=1) > float(self.cfg.rewards.command_threshold)
        return penalty

    def _reward_step_length(self):
        """Reward landing the swing foot ahead of the stance foot.

        This discourages the 'shuffle then bring feet parallel' behavior by explicitly
        encouraging a forward placement at touchdown that matches the gait phase.
        """
        if self.feet_num < 2:
            return torch.zeros(self.num_envs, device=self.device)

        # Contact detection with simple filtering
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0  # (num_envs, 2)
        contact_filt = torch.logical_or(contact, self._step_last_contacts)
        first_contact = (~self._step_last_contacts) & contact_filt
        self._step_last_contacts = contact

        # Feet positions in base frame
        rel = self.feet_pos - self.root_states[:, :3].unsqueeze(1)  # (num_envs, 2, 3)
        rel_flat = rel.reshape(-1, 3)
        quat_rep = self.base_quat.repeat_interleave(self.feet_num, dim=0)
        rel_body = quat_rotate_inverse(quat_rep, rel_flat).reshape(self.num_envs, self.feet_num, 3)
        feet_x = rel_body[:, :, 0]

        delta_x = feet_x[:, 0] - feet_x[:, 1]  # left - right

        stance_threshold = float(self.stance_threshold)
        is_stance_expected = self.leg_phase < stance_threshold  # (num_envs, 2)
        left_swing_expected = ~is_stance_expected[:, 0]
        right_swing_expected = ~is_stance_expected[:, 1]

        cmd_ok = torch.norm(self.commands[:, :2], dim=1) > float(self.cfg.rewards.command_threshold)

        target = float(self.cfg.rewards.step_length_target)
        sigma = float(self.cfg.rewards.step_length_sigma)

        # When left lands, want left ahead of right by +target (delta_x ~ +target)
        left_mask = first_contact[:, 0] & left_swing_expected & cmd_ok
        left_err = delta_x - target
        left_rew = torch.exp(-torch.square(left_err) / sigma) * left_mask.float()

        # When right lands, want right ahead of left by +target (delta_x ~ -target)
        right_mask = first_contact[:, 1] & right_swing_expected & cmd_ok
        right_err = (-delta_x) - target
        right_rew = torch.exp(-torch.square(right_err) / sigma) * right_mask.float()

        return left_rew + right_rew

    """
    Gait rewards.
    """

    def _reward_feet_gait(self):
        """
        Rewards the consistency of foot contact patterns with the expected gait rhythm (gait synchronization reward).
        
        Tuning suggestions:
        period :
        - 0.5-0.7s: Fast gait, suitable for high-speed movement
        - 0.8-1.0s: Standard gait, balanced speed and stability (recommended)
        - 1.0-1.5s: Slow gait, suitable for complex terrain
        
        offset (phase offset):
        - [0.0, 0.5]: Alternating gait (walk/trot), most common
        
        stance_threshold (stance phase ratio):
        - 0.5: Equal stance and swing time (50%-50%)
        - 0.55-0.6: More stance time, more stable (recommended)
        - 0.4-0.45: More swing time, faster
        
        Returns:
            torch.Tensor: Reward value for each environment, shape (num_envs,)
                         Range: [0, num_feet], full score equals number of feet
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
        Penalizes asymmetry in left and right joint positions, encouraging symmetrical movement patterns.
        Tuning suggestions:
        - Weight -0.5 to -2.0: Moderately encourage symmetry
        - Weight -2.0 to -5.0: Enforce symmetry (may limit flexibility)
        - Weight 0: Allow complete asymmetry (not recommended)
        
        Args:
            mirror_pairs: List of index pairs for mirrored joints, format [[left_idx, right_idx], ...]
                         If None, the default configuration for X2 is used
            
        Returns:
            torch.Tensor: Penalty value for each environment, shape (num_envs,)
                         Lower values indicate more symmetry, 0 means perfect symmetry
        """
        # Default X2 mirrored joint pairs (left leg vs right leg)
        if mirror_pairs is None:
            mirror_pairs = [
                [0, 6],   # hip_yaw
                [1, 7],   # hip_roll
                [2, 8],   # hip_pitch
                [3, 9],   # knee
                [4, 10],  # ankle_pitch
                [5, 11],  # ankle_roll
            ]
        
        # Accumulate squared differences of all mirrored joint pairs
        reward = torch.zeros(self.num_envs, device=self.device)
        
        for left_idx, right_idx in mirror_pairs:
            # Calculate squared differences of left and right joint positions
            # dof_pos shape: (num_envs, num_dof)
            pos_diff = self.dof_pos[:, left_idx] - self.dof_pos[:, right_idx]
            reward += torch.square(pos_diff)
        
        # Normalize by the number of mirrored pairs to make penalty independent of joint count
        reward = reward / len(mirror_pairs)
        
        return reward