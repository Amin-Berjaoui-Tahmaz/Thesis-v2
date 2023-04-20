#pip install numpy==1.22.4

import argparse

from maple.launchers.launcher_util import run_experiment
from maple.launchers.robosuite_launcher import experiment
import maple.util.hyperparameter as hyp
import collections

# Making a dictionary of all necessary parameters for the environment and framework
base_variant = dict(

    # ckpt_path = "wipe/04-18-wipe-position/04-18-wipe_position_2023_04_18_17_28_40_0000--s-70458",
    # ckpt_epoch = 25,

    # Parameters for setting up and training neural networks
    layer_size=256,
    replay_buffer_size=int(1E6),
    rollout_fn_kwargs=dict(
        terminals_all_false=True,
    ),
    algorithm_kwargs=dict(
        num_epochs=10000,
        num_expl_steps_per_train_loop=3000,
        num_eval_steps_per_epoch=3000,
        num_trains_per_train_loop=1000,
        min_num_steps_before_training=30000,
        max_path_length=150, # maximum number of atomic actions allowed to be taken
        batch_size=1024,
        eval_epoch_freq=10,
    ),
    # algorithm_kwargs=dict(
    #     num_epochs=1000,
    #     num_expl_steps_per_train_loop=300,
    #     num_eval_steps_per_epoch=300,
    #     num_trains_per_train_loop=100,
    #     min_num_steps_before_training=3000,
    #     max_path_length=15,
    #     batch_size=1024, 
    #     eval_epoch_freq=10,
    # ),
    trainer_kwargs=dict(
        discount=0.99,
        soft_target_tau=1e-3,
        target_update_period=1,
        policy_lr=3e-5,
        qf_lr=3e-5,
        reward_scale=1,
        use_automatic_entropy_tuning=True,
    ),
    ll_sac_variant=dict(
        high_init_ent=True,
    ),
    pamdp_variant=dict(
        one_hot_s=True,
        high_init_ent=True,
        one_hot_factor=0.50,
    ),


    # Skill parameters
    env_variant=dict(
        
        # robot state (used in observations in robosuite_launcher.py); ROBABLY NEEDS CHANGING TO EXTRACT FORCES AND MAYBE TORQUES (ADDED 'robot0_eef_force')
        robot_keys=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel'],

        # object state (used in observations in robosuite_launcher.py)
        obj_keys=['object-state'],

        # default controller type if not specified in skill; PROBABLY NEEDS CHANGING TO IMPEDANCE CONTROLLER
        controller_type='OSC_POSITION_YAW',
        controller_config_update=dict(
            position_limits=[
                [-0.30, -0.30, 0.75],       # these limits might be specified based on the physical constraints of the robot (?)
                [0.15, 0.30, 1.15]
            ],
        ),

        # environment initialization parameters
        env_kwargs=dict(
            ignore_done=True,
            reward_shaping=True,
            hard_reset=False,
            control_freq=10,
            camera_heights=512,
            camera_widths=512,
            table_offset=[-0.075, 0, 0.8],
            reward_scale=5.0,

            # specifying the available set of skills
            skill_config=dict(
                skills=['atomic', 'open', 'reach', 'grasp', 'push'],
                aff_penalty_fac=15.0,

                base_config=dict(
                    global_xyz_bounds=[
                        [-0.30, -0.30, 0.80],           # This might be also specified based on the physical constraints of the robot (?)
                        [0.15, 0.30, 0.95]
                    ],
                    lift_height=0.95, # robot lifts end-effector to a pre-specified height; used for tabletop environments
                    binary_gripper=True, # gripper either open or closed; PROBABLY NEEDS CHANGING TO ACCOMODATE IMPEDANCE

                    aff_threshold=0.06, # affordance threshold representing how close the end-effector is to the graspable/pushable/reaching object keypoint
                    aff_type='dense', # dense reward signal for affordance
                    aff_tanh_scaling=10.0,
                ),
                atomic_config=dict(
                    use_ori_params=True,
                ),

                # Reaching: The robot moves its end-effector to a target
                # location (x, y, z), specified by the input parameters.
                reach_config=dict(
                    use_gripper_params=False,
                    local_xyz_scale=[0.0, 0.0, 0.06],
                    use_ori_params=False, # uses only (x,y,z)
                    max_ac_calls=15, # at max 15 atomic actions
                ),

                #The robot moves its end-effector to a pre-grasp location (x, y, z) at a yaw angle ψ 
                # and closes its gripper.
                grasp_config=dict(
                    global_xyz_bounds=[
                        [-0.30, -0.30, 0.80],
                        [0.15, 0.30, 0.85]
                    ],
                    aff_threshold=0.03,

                    local_xyz_scale=[0.0, 0.0, 0.0],
                    use_ori_params=True, # uses (x,y,z) and yaw angle ψ
                    max_ac_calls=20,  # at max 20 atomic actions
                    num_reach_steps=2,
                    num_grasp_steps=3,
                ),

                # The robot reaches a starting location (x, y, z) at a yaw angle ψ 
                # and then moves its end-effector by a displacement (δx, δy, δz).
                # The input parameters are 7D.
                push_config=dict(
                    global_xyz_bounds=[
                        [-0.30, -0.30, 0.80],
                        [0.15, 0.30, 0.85]
                    ],
                    delta_xyz_scale=[0.25, 0.25, 0.05], # delta_xyz_scale means that its only allowed to displace a delta from the current position

                    max_ac_calls=20,  # at max 20 atomic actions
                    use_ori_params=True, # uses (x,y,z) and yaw angle ψ

                    aff_threshold=[0.12, 0.12, 0.04],
                ),
            ),
        ),
    ),
    save_video=True,
    save_video_period=10, #was 100
    dump_video_kwargs=dict(
        rows=1,
        columns=6,
        pad_length=5,
        pad_color=0,
    ),
)


# Modifies the base parameters to perform better in the specific environment; PROBABLY NEEDS CHANGING TO ACCOMODATE IMPEDANCE PARAMETERS
env_params = dict(
    lift={
        'env_variant.env_type': ['Lift'],
    },
    door={
        'env_variant.env_type': ['Door'],
        'env_variant.controller_type': ['OSC_POSITION'],
        'env_variant.controller_config_update.position_limits': [[[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]]],
        'env_variant.env_kwargs.skill_config.base_config.global_xyz_bounds': [
            [[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]]],
        'env_variant.env_kwargs.skill_config.grasp_config.global_xyz_bounds': [
            [[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]]],
        'env_variant.env_kwargs.skill_config.push_config.global_xyz_bounds': [
            [[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]]],
        'env_variant.env_kwargs.skill_config.base_config.lift_height': [1.15],
        'env_variant.env_kwargs.skill_config.grasp_config.aff_threshold': [0.06],
        'env_variant.env_kwargs.skill_config.skills': [['atomic', 'grasp', 'reach_osc', 'push', 'open']],
    },
    pnp={
        'env_variant.env_type': ['PickPlaceCan'],
        'env_variant.env_kwargs.bin1_pos': [[0.0, -0.25, 0.8]],
        'env_variant.env_kwargs.bin2_pos': [[0.0, 0.28, 0.8]],
        'env_variant.controller_config_update.position_limits': [[[-0.15, -0.50, 0.75], [0.15, 0.50, 1.15]]],
        'env_variant.env_kwargs.skill_config.base_config.global_xyz_bounds': [
            [[-0.15, -0.50, 0.82], [0.15, 0.50, 1.02]]],
        'env_variant.env_kwargs.skill_config.grasp_config.global_xyz_bounds': [
            [[-0.15, -0.50, 0.82], [0.15, 0.50, 0.88]]],
        'env_variant.env_kwargs.skill_config.push_config.global_xyz_bounds': [
            [[-0.15, -0.50, 0.82], [0.15, 0.50, 0.88]]],
        'env_variant.env_kwargs.skill_config.base_config.lift_height': [1.0],
        'env_variant.env_kwargs.skill_config.reach_config.aff_threshold': [[0.15, 0.25, 0.06]],
    },
    wipe={
        'env_variant.env_type': ['Wipe'],
        'env_variant.obj_keys': [['robot0_contact-obs', 'object-state']],
        'algorithm_kwargs.max_path_length': [300],
        'env_variant.controller_type': ['OSC_POSITION'],
        'env_variant.controller_config_update.position_limits': [[[-0.10, -0.30, 0.75], [0.20, 0.30, 1.00]]],
        'env_variant.env_kwargs.table_offset':[[0.05, 0, 0.8]],
        'env_variant.env_kwargs.skill_config.base_config.global_xyz_bounds': [
            [[-0.10, -0.30, 0.75], [0.20, 0.30, 1.00]]],
        'env_variant.env_kwargs.skill_config.grasp_config.global_xyz_bounds': [
            [[-0.10, -0.30, 0.80], [0.20, 0.30, 0.85]]],
        'env_variant.env_kwargs.skill_config.push_config.global_xyz_bounds': [
            [[-0.10, -0.30, 0.80], [0.20, 0.30, 0.85]]],
        'env_variant.env_kwargs.skill_config.base_config.aff_threshold': [[0.15, 0.25, 0.03]],
        'env_variant.env_kwargs.skill_config.reach_config.aff_threshold': [[0.15, 0.25, 0.03]],
        'env_variant.env_kwargs.skill_config.grasp_config.aff_threshold': [[0.15, 0.25, 0.03]],
        'env_variant.env_kwargs.skill_config.push_config.aff_threshold': [[0.15, 0.25, 0.03]],
        'env_variant.env_kwargs.skill_config.skills': [['atomic', 'reach', 'push']],
    },

    wipe_mod={
        'env_variant.env_type': ['Wipe'],
        'env_variant.obj_keys': [['robot0_contact-obs', 'object-state']],
        'algorithm_kwargs.max_path_length': [300],

#        
        'env_variant.robot_keys': [['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel','robot0_eef_force']], # changed above so commented here
#        'env_variant.controller_type': ['OSC_POSE'], ###IMPORTANT NOTE: wipe has POSITION, so we need to make orientation fixed at 0
        'env_variant.controller_type': ['OSC_POSITION'],
#        'env_variant.controller_config_update.kp': [500],
#        'env_variant.controller_config_update.impedance': [100],
#        'env_variant.controller_config_update.kp': [10],
        'env_variant.controller_config_update.impedance_mode': ['fixed'],
#
        'env_variant.controller_config_update.position_limits': [[[-0.10, -0.30, 0.75], [0.20, 0.30, 1.00]]],
        'env_variant.env_kwargs.table_offset':[[0.05, 0, 0.8]], # I think this is just the initial offset from the table
        'env_variant.env_kwargs.skill_config.base_config.global_xyz_bounds': [
            [[-0.10, -0.30, 0.75], [0.20, 0.30, 1.00]]],
        'env_variant.env_kwargs.skill_config.grasp_config.global_xyz_bounds': [
            [[-0.10, -0.30, 0.80], [0.20, 0.30, 0.85]]],
        'env_variant.env_kwargs.skill_config.push_config.global_xyz_bounds': [
            [[-0.10, -0.30, 0.80], [0.20, 0.30, 0.85]]],
        'env_variant.env_kwargs.skill_config.base_config.aff_threshold': [[0.15, 0.25, 0.03]],
        'env_variant.env_kwargs.skill_config.reach_config.aff_threshold': [[0.15, 0.25, 0.03]],
        'env_variant.env_kwargs.skill_config.grasp_config.aff_threshold': [[0.15, 0.25, 0.03]],
        'env_variant.env_kwargs.skill_config.push_config.aff_threshold': [[0.15, 0.25, 0.03]],
        'env_variant.env_kwargs.skill_config.skills': [['atomic', 'reach', 'push']],
    },

    stack={
        'env_variant.env_type': ['Stack'],
    },
    nut={
        'env_variant.env_type': ['NutAssemblyRound'],
        'env_variant.env_kwargs.skill_config.grasp_config.aff_threshold': [0.06],
    },
    cleanup={
        'env_variant.env_type': ['Cleanup'],
    },
    peg_ins={
        'env_variant.env_type': ['PegInHole'],
        'env_variant.controller_config_update.position_limits': [[[-0.30, -0.30, 0.75], [0.15, 0.30, 1.00]]],
        'env_variant.env_kwargs.skill_config.reach_config.aff_threshold': [0.06],
        'pamdp_variant.one_hot_factor': [0.375],
    },
)

def process_variant(variant):
    if args.debug:
        variant['algorithm_kwargs']['num_epochs'] = 3
        variant['algorithm_kwargs']['batch_size'] = 64
        steps = 50
        variant['algorithm_kwargs']['max_path_length'] = steps
        variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = steps
        variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = steps
        variant['algorithm_kwargs']['min_num_steps_before_training'] = steps
        variant['algorithm_kwargs']['num_trains_per_train_loop'] = 50
        variant['replay_buffer_size'] = int(1E3)
        variant['dump_video_kwargs']['columns'] = 3

    if args.no_video:
        variant['save_video'] = False

    variant['exp_label'] = args.label

    return variant

def deep_update(source, overrides):
    """
    Copied from https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    for key, value in overrides.items():
        if isinstance(value, collections.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source

if __name__ == "__main__":
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--label', type=str, default='test')
    parser.add_argument('--no_video', action='store_true')
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--first_variant', action='store_true')
    parser.add_argument('--snapshot_gap', type=int, default=25)

    args = parser.parse_args()

    search_space = env_params[args.env]

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=base_variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant = process_variant(variant)

        run_experiment(
            experiment,
            exp_folder=args.env,
            exp_prefix=args.label,
            variant=variant,
            snapshot_mode='gap_and_last',
            snapshot_gap=args.snapshot_gap,
            exp_id=exp_id,
            use_gpu=(not args.no_gpu),
            gpu_id=args.gpu_id,
            mode='local',
            num_exps_per_instance=1,
#            verbose=True,
        )

        if args.first_variant:
            exit()


# TODO:
# CHECK OUT ROBOMIMIC, it has human demonstration dataset
# Rerun normal wiping but remove the force from observations, also experiment if u can update the observations from the env_params
# Just change debug mode parameters: Run wipe for 1 epoch and see output (maybe even force it to execute reach for specific (x,y,z))
# Change controller and see what parameters are needed
# Update parameters based on 1 epoch output so that it now completely works with impedance controller and uses impedance parameters
# Downgrade pandas to match numpy version
###########################################################################





# Done:
# Understand what the videos in data mean - DONE
### In data folder, _expl (in vis_expl: visualize exploring) means exploration (during training) while _eval means evaluation (during evaluation)
###############
# Figure out viskit to plot results and visualize wtf is going on - DONE
###############
# How is the primitve behavior captured by the neural network?
### The neural network simply takes in the observation and outputs parameters, which are in turn used as input to these primitives. 
### So, the output of the neural network should be expanded to also output impedance. But this also means we have to modify the 
### primtives to take impedance as input or is this already done by the controller since the effective role of the primitives is
### to specify the new position to go to by specifying the reference frame and affordances
### Also, technically the primitives are closed-loop controllers, so the action parameters must be the desired position and controller params (Are they 
# separate controllers or do they all just belong to the same controller??)
###############
# Figure out how to use checkpoints - DONE
###############