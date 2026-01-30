from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class X2FlatCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'waist_yaw_joint': 0.,
           'waist_pitch_joint': 0.,
           'waist_roll_joint': 0.
        }
    
    class env(LeggedRobotCfg.env):
        num_observations = 56
        num_privileged_obs = 59
        num_actions = 15


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     'waist_yaw': 80,     
                     'waist_pitch': 120,  
                     'waist_roll': 120,   
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'waist_yaw': 3,      
                     'waist_pitch': 5,    
                     'waist_roll': 5,     
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/x2/urdf/x2_ultra_simple_collision.urdf'
        name = "x2"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        gait_period = 0.8
        gait_offset = 0.5
        stance_threshold = 0.55
        command_threshold = 0.1
        feet_width_target = 0.28
        step_length_target = 0.16
        step_length_sigma = 0.05
        
        class scales( LeggedRobotCfg.rewards.scales ):
            # Main objective: track velocity commands
            tracking_lin_vel = 2.5          # Increased: main positive reward
            tracking_ang_vel = 1.0          # Increased: encourage turning
            
            # Survival and stability
            alive = 1.0                     # Increased: encourage standing
            orientation = -2.0              # Stronger: discourage overall tilt (roll/pitch)
            base_pitch = -2.0               # New: directly penalize lean forward/back
            base_roll = -1.2                # Stronger: discourage left/right sway
            base_height = -1.0              # Significantly reduced: from -10 to -1
            
            # Gait quality
            feet_air_time = 1.2             # Encourage longer swing/steps
            feet_gait = 0.5                   # Reduced: encourage contact but not too strong
            feet_swing_height = -5.0        # Stronger: enforce swing clearance target
            contact_no_vel = -0.1           # Reduced: allow learning process
            
            # Stance width
            foot_near = 0.0                  # Disable: this term repels feet when too close (tends to widen stance)
            feet_width = -0.2                # Penalize deviation from rewards.feet_width_target
            joint_mirror = -0.1              # Reduced: allow asymmetry

            # Step quality
            step_length = 0.4                # Reward forward foot placement at touchdown
            
            # Joint and action smoothness
            hip_pos = -0.3                  # Reduced: allow more hip movement
            waist_pos = -0.8                # Stronger: keep torso straighter (pitch/roll only)
            dof_pos_limits = -1.0           # Reduced: from -5 to -1
            dof_vel = -5e-4                 # Reduced: decrease velocity penalty
            dof_acc = -1e-7                 # Reduced: decrease acceleration penalty
            action_rate = -0.05             # Stronger smoothing to reduce jittery stepping
            
            # Avoid collisions
            lin_vel_z = -1.0                # Reduced: allow some vertical movement
            ang_vel_xy = -0.08              # Stronger: reduce torso wobble (roll/pitch angular velocity)
            collision = -0.5                # Enabled: penalize non-foot collisions

            # Energy
            torques = -1.0e-5               # Helps reduce violent upper-body motion

class X2FlatCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 1.0            # Increase exploration
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 5e-4            # Increase learning rate
        
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCritic"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'x2'

  
