from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class X2RoughCfg( LeggedRobotCfg ):
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
                     'waist_yaw': 80,     # Lower: more compliant
                     'waist_pitch': 80,   # Lower: more compliant
                     'waist_roll': 80,    # Lower: more compliant
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'waist_yaw': 3,      # Higher: increase damping
                     'waist_pitch': 3,    # Higher: increase damping
                     'waist_roll': 3,     # Higher: increase damping
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
        
        class scales( LeggedRobotCfg.rewards.scales ):
            # Main objective: track velocity commands
            tracking_lin_vel = 2.0          # Increased: main positive reward
            tracking_ang_vel = 1.0          # Increased: encourage turning
            
            # Survival and stability
            alive = 0.5                     # Increased: encourage standing
            orientation = -0.5              # Reduced: allow more freedom
            base_height = -1.0              # Significantly reduced: from -10 to -1
            
            # Gait quality
            feet_air_time = 1.0             # Enabled: encourage foot lifting
            contact = 0.5                   # Reduced: encourage contact but not too strong
            feet_swing_height = -5.0        # Significantly reduced: from -20 to -5
            contact_no_vel = -0.1           # Reduced: allow learning process
            symmetry = 0.3                  # NEW: encourage symmetric left-right gait
            
            # Joint and action smoothness
            hip_pos = -0.3                  # Reduced: allow more hip movement
            waist_pos = -0.1                # Reduced: allow more waist freedom
            dof_pos_limits = -1.0           # Reduced: from -5 to -1
            dof_vel = -5e-4                 # Reduced: decrease velocity penalty
            dof_acc = -1e-7                 # Reduced: decrease acceleration penalty
            action_rate = -0.005            # Reduced: allow faster action changes
            
            # Avoid collisions
            lin_vel_z = -1.0                # Reduced: allow some vertical movement
            ang_vel_xy = -0.02              # Reduced: decrease pitch/roll penalty
            collision = -0.5                # Enabled: penalize non-foot collisions

class X2RoughCfgPPO( LeggedRobotCfgPPO ):
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

  
