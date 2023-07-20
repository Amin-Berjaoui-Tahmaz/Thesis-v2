import numpy as np
import robosuite.utils.transform_utils as trans

class BaseSkill:
    def __init__(
            self,
            skill_type,

            ### common settings ###
            global_xyz_bounds=np.array([
                [-0.30, -0.30, 0.80],
                [0.15, 0.30, 0.90]
            ]),
            delta_xyz_scale=np.array([0.15, 0.15, 0.05]),
            local_xyz_scale=np.array([0.05, 0.05, 0.05]),
            lift_height=0.95,
            reach_threshold=0.02,
            aff_threshold=0.08,
            aff_type=None,

            binary_gripper=True,

            aff_tanh_scaling=10.0,

            **config
    ):
        self._skill_type = skill_type
        self._config = dict(
            global_xyz_bounds=global_xyz_bounds,
            delta_xyz_scale=delta_xyz_scale,
            local_xyz_scale=local_xyz_scale,
            lift_height=lift_height,
            reach_threshold=reach_threshold,
            aff_threshold=aff_threshold,
            aff_type=aff_type,
            binary_gripper=binary_gripper,
            aff_tanh_scaling=aff_tanh_scaling,
            **config
        )

        for k in ['global_xyz_bounds', 'delta_xyz_scale', 'local_xyz_scale']:
            assert self._config[k] is not None
            self._config[k] = np.array(self._config[k])

        assert self._config['aff_type'] in [None, 'sparse', 'dense']

    def get_param_dim(self, base_param_dim):
        assert NotImplementedError

    def update_state(self, info):
        pass

    def reset(self, params, config_update, info):
        self._params = params
        self._state = None
        self._config.update(config_update)
        self._aff_reward, self._aff_success = \
            self._compute_aff_reward_and_success(info)

    def get_pos_ac(self, info):
        raise NotImplementedError

    def get_ori_ac(self, info):
        params = self._params
        rc_dim = self._config['robot_controller_dim']
        if rc_dim==3 or rc_dim==4: # representing OSC_POSITION and OSC_POSITION_YAW
            ori = params[3:rc_dim].copy()
            impedance_ori = 999
        elif rc_dim==6 or rc_dim==7: # representing osc.py with 3 translational stiffness values
            ori = params[6:rc_dim].copy()
            impedance_ori = 999
        else:
            ori = params[9:rc_dim].copy()
            impedance_ori = params[3:6].copy()
        return ori, impedance_ori

    def get_gripper_ac(self, info):
        params = self._params
        rc_dim = self._config['robot_controller_dim']
        rg_dim = self._config['robot_gripper_dim']
        gripper_action = params[rc_dim: rc_dim + rg_dim].copy()

        if self._config['binary_gripper']:
            if np.abs(gripper_action) < 0.10:
                gripper_action[:] = 0
            elif gripper_action < 0:
                gripper_action[:] = -1
            else:
                gripper_action[:] = 1

        return gripper_action

    def get_max_ac_calls(self):
        return self._config['max_ac_calls']

    def get_aff_reward(self):
        return self._aff_reward

    def get_aff_success(self):
        return self._aff_success

    def _get_unnormalized_pos(self, pos, bounds):
        pos = np.clip(pos, -1, 1)
        pos = (pos + 1) / 2
        low, high = bounds[0], bounds[1]
        return low + (high - low) * pos

    def _reached_goal_ori(self, info):
        ori, _, _ = self.get_ori_ac(info)
        ori = ori.copy()

        if len(ori) == 0 or (not self._config['use_ori_params']):
            return True
        robot_controller = self._config['robot_controller']
        goal_ori = robot_controller.get_global_euler_from_ori_ac(ori)
        cur_ori = trans.mat2euler(robot_controller.ee_ori_mat, axes="rxyz")
        ee_ori_diff = np.minimum(
            (goal_ori - cur_ori) % (2 * np.pi),
            (cur_ori - goal_ori) % (2 * np.pi)
        )
        if ee_ori_diff[-1] <= 0.20:
            return True
        else:
            return False

    def _compute_aff_reward_and_success(self, info):
        if self._config['aff_type'] is None:         # if 'aff_type' is None, reward is 1 (meaning the affordance info is not being used)
            return 1.0, True

        # Retreive the affordance centers and reaching positions
        aff_centers = self._get_aff_centers(info)
        reach_pos , _ = self._get_reach_pos(info)

        if aff_centers is None:        # If affordance centers is None (such as the case of Atomic and GripperSkill), then return 1.0 reward and success is True
            return 1.0, True
        # SIDE NOTE: I think the reason atomic is spammed early on is because you get a reward for just using it

        if len(aff_centers) == 0:         # If there are no affordance centers (object keypoints), then 0 reward and success is False
#            if info['f_excess_max_force'] and self._config['use_force_aff']: # amin
#                return 0.2, False
#            else:
            return 0.0, False

        th = self._config['aff_threshold'] # retreive pre-defined affordance threshold
        within_th = (np.abs(aff_centers - reach_pos) <= th) # check if the executed skill is within range of the threshold 
        aff_success = np.any(np.all(within_th, axis=1)) # return that it is successful if it is in range

        if self._config['aff_type'] == 'dense': # If the affordance type is 'dense', then the affordance reward increases the closer it is to the keypoint IF IT IS SOMEWHERE OUTSIDE THE THRESHOLD AREA
            if aff_success:
                aff_reward = 1.0
            else:
                dist = np.clip(np.abs(aff_centers - reach_pos) - th, 0, None)
                min_dist = np.min(np.sum(dist, axis=1))
                aff_reward = 1.0 - np.tanh(self._config['aff_tanh_scaling'] * min_dist)
        else:
            aff_reward = float(aff_success)

        # I added this part:
        aff_reward_pos = aff_reward
        if self._config['use_force_aff']:
            force_weight = 0.2
            pos_weight = 1 - force_weight            

            stiffness_magnitude = np.linalg.norm(np.array(info['stiffness_force']))
            scaled_stiffness_magnitude = stiffness_magnitude / np.sqrt(3)
            stiffness_aff = 1 - scaled_stiffness_magnitude
            aff_reward = pos_weight*aff_reward_pos + force_weight*stiffness_aff

        return aff_reward, aff_success

    def _get_aff_centers(self, info):
        raise NotImplementedError

    def _get_reach_pos(self, info):
        raise NotImplementedError

class AtomicSkill(BaseSkill):
    def __init__(
            self,
            skill_type,
            use_ori_params=True,
            use_gripper_params=True,
            **config
    ):
        super().__init__(
            skill_type,
            use_ori_params=use_ori_params,
            use_gripper_params=use_gripper_params,
            max_ac_calls=1,
            **config
        )

    def get_param_dim(self, base_param_dim):
        return base_param_dim

    def get_pos_ac(self, info):
        params = self._params
#        print('atomic params',params)
        is_delta = True
        rc_dim = self._config['robot_controller_dim']
        if rc_dim==3 or rc_dim==4: # representing OSC_POSITION and OSC_POSITION_YAW
            pos = params[:3].copy()
            impedance_pos = 999
        elif rc_dim==6 or rc_dim==7:
            pos = params[3:6].copy()
            impedance_pos = params[:3].copy()
        else:
            pos = params[6:9].copy()
            impedance_pos = params[:3].copy()

        # Discretizes impedance to 3 values: -1, 0, 1
        if self._config['discrete_impedance']:
            impedance_pos = np.round(impedance_pos)

        return pos, is_delta, impedance_pos

    def get_ori_ac(self, info):
        ori, impedance_ori = super().get_ori_ac(info)
        if not self._config['use_ori_params']:
            ori[:] = 0.0
        is_delta = True
        return ori, is_delta, impedance_ori

    def get_gripper_ac(self, info):
        gripper_action = super().get_gripper_ac(info)
        if not self._config['use_gripper_params']:
            gripper_action[:] = 1

        return gripper_action

    def is_success(self, info):
        return True

    def _get_aff_centers(self, info):
        return None

    def _get_reach_pos(self, info):
        return info['cur_ee_pos'], 999

class GripperSkill(BaseSkill):
    def __init__(
            self,
            skill_type,
            max_ac_calls=4,
            **config
    ):
        super().__init__(
            skill_type,
            max_ac_calls=max_ac_calls,
            **config
        )
        self._num_steps_steps = 0

    def get_param_dim(self, base_param_dim):
        return 0

    def reset(self, *args, **kwargs):
        super().reset(*args, *kwargs)
        self._num_steps_steps = 0

    def update_state(self, info):
        self._num_steps_steps += 1

    def get_pos_ac(self, info):
        pos = np.zeros(3)
        is_delta = True
        params = self._params.copy()
        impedance_pos = [-1,-1,-1] # might cause problems, probably not
        return pos, is_delta, impedance_pos

    def get_ori_ac(self, info):
        rc_dim = self._config['robot_controller_dim']

        if rc_dim==3 or rc_dim==6 or rc_dim==9: # representing OSC_POSITION with fixed/variable_kp_mod/variable_kp
            ori = np.zeros(0)
        elif rc_dim==4 or rc_dim==7 or rc_dim==10:  #representing OSC_POSITION_YAW with fixed/variable_kp_mod/variable_kp
            ori = np.zeros(1)

        ori[:] = 0.0
        is_delta = True
        impedance_ori = [0,0,0]
        return ori, is_delta, impedance_ori

    def get_gripper_ac(self, info):
        rg_dim = self._config['robot_gripper_dim']
        gripper_action = np.zeros(rg_dim)
        if self._skill_type in ['close', 'close_pos']:
            gripper_action[:] = 1
        elif self._skill_type in ['open', 'open_pos']:
            gripper_action[:] = -1
        else:
            raise ValueError

        return gripper_action

    def is_success(self, info):
        return (self._num_steps_steps == self._config['max_ac_calls'])

    def _get_aff_centers(self, info):
        return None

    def _get_reach_pos(self, info):
        return info['cur_ee_pos'], 999

class ReachOSCSkill(BaseSkill):
    def __init__(
            self,
            skill_type,
            use_gripper_params=True,
            use_ori_params=False,
            max_ac_calls=15,
            use_delta=False, # only applicable for skill_type=r1
            **config
    ):
        super().__init__(
            skill_type,
            use_gripper_params=use_gripper_params,
            use_ori_params=use_ori_params,
            max_ac_calls=max_ac_calls,
            use_delta=use_delta,
            **config
        )
        self._start_pos = None
        assert (not use_delta)

    def get_param_dim(self, base_param_dim):
        return base_param_dim

    def reset(self, params, config_update, info):
        self._start_pos = info['cur_ee_pos']
        super().reset(params, config_update, info)

    def get_pos_ac(self, info):
        goal_pos, impedance_pos = self._get_reach_pos(info)
        is_delta = False
        return goal_pos, is_delta, impedance_pos

    def get_ori_ac(self, info):
        use_ori_params = self._config['use_ori_params']
        assert use_ori_params is False

        ori, impedance_ori = super().get_ori_ac(info)
        if use_ori_params:
            raise NotImplementedError
        else:
            ori[:] = 0.0
        is_delta = True
        return ori, is_delta, impedance_ori

    def get_gripper_ac(self, info):
        gripper_action = super().get_gripper_ac(info)
        if not self._config['use_gripper_params']:
            gripper_action[:] = 0

        return gripper_action

    def _get_reach_pos(self, info):
        params = self._params
        rc_dim = self._config['robot_controller_dim']

        if self._config['use_delta']:
            if rc_dim==3 or rc_dim==4: # representing OSC_POSITION and OSC_POSITION_YAW
                delta_pos = params[:3].copy()
                delta_pos = np.clip(delta_pos, -1, 1)
                delta_pos *= self._config['delta_xyz_scale']
                pos = self._start_pos + delta_pos
                impedance_pos = 999
                return pos, impedance_pos

            elif rc_dim==6 or rc_dim==7:
                delta_pos = params[3:6].copy()
                delta_pos = np.clip(delta_pos, -1, 1)
                delta_pos *= self._config['delta_xyz_scale']
                pos = self._start_pos + delta_pos
                impedance_pos = params[:3].copy()

            else:
                delta_pos = params[6:9].copy()
                delta_pos = np.clip(delta_pos, -1, 1)
                delta_pos *= self._config['delta_xyz_scale']
                pos = self._start_pos + delta_pos
                impedance_pos = params[:3].copy()

            # Discretizes impedance to 3 values: -1, 0, 1
            if self._config['discrete_impedance']:
                impedance_pos = np.round(impedance_pos)

            return pos, impedance_pos

        else:
            if rc_dim==3 or rc_dim==4: # representing OSC_POSITION and OSC_POSITION_YAW
                pos = self._get_unnormalized_pos(
                    params[:3], self._config['global_xyz_bounds'])
                impedance_pos = 999
                return pos, impedance_pos
            if rc_dim==6 or rc_dim==7:
                pos = self._get_unnormalized_pos(
                    params[3:6], self._config['global_xyz_bounds'])
                impedance_pos = params[:3].copy()
            else:
                pos = self._get_unnormalized_pos(
                    params[6:9], self._config['global_xyz_bounds'])
                impedance_pos = params[:3].copy()

            # Discretizes impedance to 3 values: -1, 0, 1
            if self._config['discrete_impedance']:
                impedance_pos = np.round(impedance_pos)

            return pos, impedance_pos

    def is_success(self, info):
        pos, delta, impedance_pos = self.get_pos_ac(info)
        cur_pos = info['cur_ee_pos']
        th = self._config['reach_threshold']
        return (np.linalg.norm(pos - cur_pos) <= th)

    def _get_aff_centers(self, info):
        aff_centers = info.get('reach_pos', [])
        if aff_centers is None:
            return None
        return np.array(aff_centers, copy=True)

class ReachSkill(BaseSkill):

    STATES = ['INIT', 'LIFTED', 'HOVERING', 'REACHED']

    def __init__(
            self,
            skill_type,
            use_gripper_params=True,
            use_ori_params=False,
            max_ac_calls=15,
            **config
    ):
        super().__init__(
            skill_type,
            use_gripper_params=use_gripper_params,
            use_ori_params=use_ori_params,
            max_ac_calls=max_ac_calls,
            **config,
        )

    def get_param_dim(self, base_param_dim):
        return base_param_dim

    def update_state(self, info):
        cur_pos = info['cur_ee_pos']
        goal_pos, _ = self._get_reach_pos(info)

        th = self._config['reach_threshold']
        reached_lift = (cur_pos[2] >= self._config['lift_height'] - th)
        reached_xy = (np.linalg.norm(cur_pos[0:2] - goal_pos[0:2]) < th)
        reached_xyz = (np.linalg.norm(cur_pos - goal_pos) < th)
        reached_ori = self._reached_goal_ori(info)

        if reached_xyz and reached_ori:
            self._state = 'REACHED'
        else:
            if reached_xy and reached_ori:
                self._state = 'HOVERING'
            else:
                if reached_lift:
                    self._state = 'LIFTED'
                else:
                    self._state = 'INIT'

        assert self._state in ReachSkill.STATES

    def get_pos_ac(self, info):
        cur_pos = info['cur_ee_pos']
        goal_pos, impedance_pos = self._get_reach_pos(info)

        is_delta = False
        if self._state == 'INIT':
            pos = cur_pos.copy()
            pos[2] = self._config['lift_height']
        elif self._state == 'LIFTED':
            pos = goal_pos.copy()
            pos[2] = self._config['lift_height']
        elif self._state == 'HOVERING':
            pos = goal_pos.copy()
        elif self._state == 'REACHED':
            pos = goal_pos.copy()
        else:
            raise NotImplementedError

        return pos, is_delta, impedance_pos

    def get_ori_ac(self, info):
        ori, impedance_ori = super().get_ori_ac(info)
        if self._config['use_ori_params']:
            if self._state == 'INIT':
                ori[:] = 0.0
                is_delta = True
            else:
                is_delta = False
        else:
            ori[:] = 0.0
            is_delta = True
        return ori, is_delta, impedance_ori

    def get_gripper_ac(self, info):
        gripper_action = super().get_gripper_ac(info)
        if not self._config['use_gripper_params']:
            gripper_action[:] = 0

        return gripper_action

    def _get_reach_pos(self, info):
        params = self._params
        rc_dim = self._config['robot_controller_dim']
        if rc_dim==3 or rc_dim==4: # representing OSC_POSITION and OSC_POSITION_YAW
            pos = self._get_unnormalized_pos(
                params[:3], self._config['global_xyz_bounds'])
            impedance_pos = 999
            return pos, impedance_pos
        elif rc_dim==6 or rc_dim==7:
            pos = self._get_unnormalized_pos(
                params[3:6], self._config['global_xyz_bounds'])
            impedance_pos = params[:3]
        else:
            pos = self._get_unnormalized_pos(
                params[6:9], self._config['global_xyz_bounds'])
            impedance_pos = params[:3]

        # Discretizes impedance to 3 values: -1, 0, 1
        if self._config['discrete_impedance']:
            impedance_pos = np.round(impedance_pos)

        return pos, impedance_pos

    def is_success(self, info):
        return self._state == 'REACHED'

    def _get_aff_centers(self, info):
        aff_centers = info.get('reach_pos', [])
        if aff_centers is None:
            return None
        return np.array(aff_centers, copy=True)

class GraspSkill(BaseSkill):

    STATES = ['INIT', 'LIFTED', 'HOVERING', 'REACHED', 'GRASPED']

    def __init__(
            self,
            skill_type,
            use_ori_params=True,
            max_ac_calls=15,
            num_reach_steps=1,
            num_grasp_steps=1,
            **config
    ):
        super().__init__(
            skill_type,
            use_ori_params=use_ori_params,
            max_ac_calls=max_ac_calls,
            num_reach_steps=num_reach_steps,
            num_grasp_steps=num_grasp_steps,
            **config
        )
        self._num_reach_steps = 0
        self._num_grasp_steps = 0

    def get_param_dim(self, base_param_dim):
        return base_param_dim

    def reset(self, *args, **kwargs):
        super().reset(*args, *kwargs)
        self._num_reach_steps = 0
        self._num_grasp_steps = 0

    def update_state(self, info):
        cur_pos = info['cur_ee_pos']
        goal_pos, _ = self._get_reach_pos(info)

        th = self._config['reach_threshold']
        reached_lift = (cur_pos[2] >= self._config['lift_height'] - th)
        reached_xy = (np.linalg.norm(cur_pos[0:2] - goal_pos[0:2]) < th)
        reached_xyz = (np.linalg.norm(cur_pos - goal_pos) < th)
        reached_ori = self._reached_goal_ori(info)

        if self._state == 'GRASPED' or \
                (self._state == 'REACHED' and (self._num_reach_steps >= self._config['num_reach_steps'])):
            self._state = 'GRASPED'
            self._num_grasp_steps += 1
        elif self._state == 'REACHED' or (reached_xyz and reached_ori):
            self._state = 'REACHED'
            self._num_reach_steps += 1
        elif reached_xy and reached_ori:
            self._state = 'HOVERING'
        elif reached_lift:
            self._state = 'LIFTED'
        else:
            self._state = 'INIT'

        assert self._state in GraspSkill.STATES

    def get_pos_ac(self, info):
        cur_pos = info['cur_ee_pos']
        goal_pos, impedance_pos = self._get_reach_pos(info)

        is_delta = False
        if self._state == 'INIT':
            pos = cur_pos.copy()
            pos[2] = self._config['lift_height']
        elif self._state == 'LIFTED':
            pos = goal_pos.copy()
            pos[2] = self._config['lift_height']
        elif self._state == 'HOVERING':
            pos = goal_pos.copy()
        elif self._state == 'REACHED':
            pos = goal_pos.copy()
        elif self._state == 'GRASPED':
            pos = goal_pos.copy()
        else:
            raise NotImplementedError

        return pos, is_delta, impedance_pos

    def get_ori_ac(self, info):
        ori, impedance_ori = super().get_ori_ac(info)
        if self._config['use_ori_params']:
            if self._state == 'INIT':
                ori[:] = 0.0
                is_delta = True
            else:
                is_delta = False
        else:
            ori[:] = 0.0
            is_delta = True
        return ori, is_delta, impedance_ori

    def get_gripper_ac(self, info):
        gripper_action = super().get_gripper_ac(info)
        if self._state in ['GRASPED', 'REACHED']:
            gripper_action[:] = 1
        else:
            gripper_action[:] = -1
        return gripper_action

    def _get_reach_pos(self, info):
        params = self._params
        rc_dim = self._config['robot_controller_dim']
        if rc_dim==3 or rc_dim==4: # representing OSC_POSITION and OSC_POSITION_YAW
            pos = self._get_unnormalized_pos(
                params[:3], self._config['global_xyz_bounds'])
            impedance_pos = 999
        elif rc_dim==6 or rc_dim==7:
            pos = self._get_unnormalized_pos(
                params[3:6], self._config['global_xyz_bounds'])
            impedance_pos = params[:3].copy()
        else:
            pos = self._get_unnormalized_pos(
                params[6:9], self._config['global_xyz_bounds'])
            impedance_pos = params[:3].copy()

        # Discretizes impedance to 3 values: -1, 0, 1
        if self._config['discrete_impedance']:
            impedance_pos = np.round(impedance_pos)

        return pos, impedance_pos

    def is_success(self, info):
        return self._num_grasp_steps >= self._config['num_grasp_steps']

    def _get_aff_centers(self, info):
        aff_centers = info.get('grasp_pos', [])
        if aff_centers is None:
            return None
        return np.array(aff_centers, copy=True)

class PushSkill(BaseSkill):

    STATES = ['INIT', 'LIFTED', 'HOVERING', 'REACHED', 'PUSHED']

    def __init__(
            self,
            skill_type,
            max_ac_calls=20,
            use_ori_params=True,
            **config
    ):
        super().__init__(
            skill_type,
            max_ac_calls=max_ac_calls,
            use_ori_params=use_ori_params,
            **config
        )

    def get_param_dim(self, base_param_dim):
        return base_param_dim + 3 # Push has 3 extra parameters since it executes a reach and a delta position

    def update_state(self, info):
        cur_pos = info['cur_ee_pos']
        src_pos, _ = self._get_reach_pos(info)
        target_pos, _ = self._get_push_pos(info)

        th = self._config['reach_threshold']
        reached_lift = (cur_pos[2] >= self._config['lift_height'] - th)
        reached_src_xy = (np.linalg.norm(cur_pos[0:2] - src_pos[0:2]) < th)
        reached_src_xyz = (np.linalg.norm(cur_pos - src_pos) < th)
        reached_target_xyz = (np.linalg.norm(cur_pos - target_pos) < th)
        reached_ori = self._reached_goal_ori(info)

        if self._state in ['REACHED', 'PUSHED'] and reached_target_xyz:
            self._state = 'PUSHED'
        else:
            if self._state == 'REACHED' or (reached_src_xyz and reached_ori):
                self._state = 'REACHED'
            else:
                if reached_src_xy and reached_ori:
                    self._state = 'HOVERING'
                else:
                    if reached_lift:
                        self._state = 'LIFTED'
                    else:
                        self._state = 'INIT'

        assert self._state in PushSkill.STATES

    def get_pos_ac(self, info):
        cur_pos = info['cur_ee_pos']
        src_pos, impedance_pos_reach = self._get_reach_pos(info)
        target_pos, impedance_pos_push = self._get_push_pos(info)

        is_delta = False
        if self._state == 'INIT':
            pos = cur_pos.copy()
            pos[2] = self._config['lift_height']
            impedance_pos = impedance_pos_reach
        elif self._state == 'LIFTED':
            pos = src_pos.copy()
            pos[2] = self._config['lift_height']
            impedance_pos = impedance_pos_reach
        elif self._state == 'HOVERING':
            pos = src_pos.copy()
            impedance_pos = impedance_pos_reach
        elif self._state == 'REACHED':
            pos = target_pos.copy()
            impedance_pos = impedance_pos_push
        elif self._state == 'PUSHED':
            pos = target_pos.copy()
            impedance_pos = impedance_pos_push
        else:
            raise NotImplementedError

        return pos, is_delta, impedance_pos

    def get_ori_ac(self, info):
        ori, impedance_ori = super().get_ori_ac(info)
        use_ori_params = self._config['use_ori_params']
        if use_ori_params:
            if self._state == 'INIT':
                ori[:] = 0.0
                is_delta = True
            else:
                is_delta = False
        else:
            ori[:] = 0.0
            is_delta = True
        return ori, is_delta, impedance_ori

    def get_gripper_ac(self, info):
        gripper_action = super().get_gripper_ac(info)
        gripper_action[:] = 1
        return gripper_action

    def _get_reach_pos(self, info):
        params = self._params
        rc_dim = self._config['robot_controller_dim']
        if rc_dim==3 or rc_dim==4: # representing OSC_POSITION and OSC_POSITION_YAW
            pos = self._get_unnormalized_pos(
                params[:3], self._config['global_xyz_bounds'])
            pos = pos.copy()
            impedance_pos = 999
            return pos, impedance_pos
        elif rc_dim==6 or rc_dim==7:
            pos = self._get_unnormalized_pos(
                params[3:6], self._config['global_xyz_bounds'])
            pos = pos.copy()
            impedance_pos = params[:3].copy()
        else:
            pos = self._get_unnormalized_pos(
                params[6:9], self._config['global_xyz_bounds'])
            pos = pos.copy()
            impedance_pos = params[:3].copy()

        # Discretizes impedance to 3 values: -1, 0, 1
        if self._config['discrete_impedance']:
            impedance_pos = np.round(impedance_pos)

        return pos, impedance_pos

    def _get_push_pos(self, info):
        params = self._params

        src_pos, impedance_pos = self._get_reach_pos(info)
        pos = src_pos.copy()

#        rc_dim = self._config['robot_controller_dim']
#        if rc_dim==3 or rc_dim==9: # representing OSC_POSITION
        delta_pos = params[-3:].copy() # INVESTIGATE MAYBE? #if wiping gets worse, then just make this [-4:-1]
#        elif rc_dim==4 or rc_dim==10: # representing OSC_POSITION_YAW
#            delta_pos = params[-4:-1].copy()
        delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._config['delta_xyz_scale']
        pos += delta_pos

        return pos, impedance_pos

    def is_success(self, info):
        return self._state == 'PUSHED'

    def _get_aff_centers(self, info):
        aff_centers = info.get('push_pos', [])
        if aff_centers is None:
            return None
        return np.array(aff_centers, copy=True)