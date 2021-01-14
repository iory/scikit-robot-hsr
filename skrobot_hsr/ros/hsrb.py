import math
from numbers import Number
import sys

import actionlib
from actionlib_msgs.msg import GoalStatus
import control_msgs.msg
import dynamic_reconfigure.client
import rospy
import std_msgs.msg
import std_srvs.srv
import tmc_control_msgs.msg
import tmc_suction.msg
import trajectory_msgs.msg
import tmc_planning_msgs.srv
from tmc_manipulation_msgs.msg import ArmManipulationErrorCodes
from tmc_manipulation_msgs.msg import BaseMovementType
import geometry_msgs.msg
from hsrb_interface import exceptions
from hsrb_interface import trajectory

from skrobot.interfaces.ros.move_base import ROSRobotMoveBaseInterface
from skrobot.interfaces.ros import tf_utils


_GRIPPER_APPLY_FORCE_DELICATE_THRESHOLD = 0.8
_HAND_MOMENT_ARM_LENGTH = 0.07
_PALM_TO_PROXIMAL_Y = 0.0245
_PROXIMAL_TO_DISTAL_Z = 0.07
_DISTAL_JOINT_ANGLE_OFFSET = 0.087
_DISTAL_TO_TIP_Y = 0.01865
_DISTAL_TO_TIP_Z = 0.04289

_DISTANCE_CONTROL_PGAIN = 0.5
_DISTANCE_CONTROL_IGAIN = 1.0
_DISTANCE_CONTROL_RATE = 10.0
_DISTANCE_CONTROL_TIME_FROM_START = 0.2
_DISTANCE_CONTROL_STALL_THRESHOLD = 0.003
_DISTANCE_CONTROL_STALL_TIMEOUT = 1.0

_HAND_MOTOR_JOINT_MAX = 1.2
_HAND_MOTOR_JOINT_MIN = -0.5

_DISTANCE_MAX = (_PALM_TO_PROXIMAL_Y -
                 (_DISTAL_TO_TIP_Y *
                  math.cos(_DISTAL_JOINT_ANGLE_OFFSET) +
                  _DISTAL_TO_TIP_Z *
                  math.sin(_DISTAL_JOINT_ANGLE_OFFSET)) +
                 _PROXIMAL_TO_DISTAL_Z *
                 math.sin(_HAND_MOTOR_JOINT_MAX)) * 2
_DISTANCE_MIN = (_PALM_TO_PROXIMAL_Y -
                 (_DISTAL_TO_TIP_Y *
                  math.cos(_DISTAL_JOINT_ANGLE_OFFSET) +
                  _DISTAL_TO_TIP_Z *
                  math.sin(_DISTAL_JOINT_ANGLE_OFFSET)) +
                 _PROXIMAL_TO_DISTAL_Z *
                 math.sin(_HAND_MOTOR_JOINT_MIN)) * 2


class HSRBROSRobotInterface(ROSRobotMoveBaseInterface):

    def __init__(self, *args, **kwargs):
        enable_suction = kwargs.pop('enable_suction', True)
        enable_gripper = kwargs.pop('enable_gripper', True)
        kwargs.setdefault('namespace', '/hsrb')
        ns = kwargs['namespace']
        kwargs['use_tf2'] = True
        kwargs['move_base_action_name'] = '/move_base/move'
        kwargs['odom_topic'] = '{}/odom'.format(ns)
        kwargs['base_controller_joint_names'] = ["odom_x", "odom_y", "odom_t"]
        kwargs['base_controller_action_name'] = \
            '{}/omni_base_controller/follow_joint_trajectory'.format(ns)
        super(HSRBROSRobotInterface, self).__init__(*args, **kwargs)

        # gripper
        self.enable_gripper = enable_gripper
        if enable_gripper is True:
            self.grasp_client = actionlib.SimpleActionClient(
                '{}/gripper_controller/grasp'.format(ns),
                tmc_control_msgs.msg.GripperApplyEffortAction
            )
            self.apply_force_client = actionlib.SimpleActionClient(
                '{}/gripper_controller/apply_force'.format(ns),
                tmc_control_msgs.msg.GripperApplyEffortAction
            )
            self.gripper_follow_joint_trajectory_client = \
                actionlib.SimpleActionClient(
                    '{}/gripper_controller/follow_joint_trajectory'.format(ns),
                    control_msgs.msg.FollowJointTrajectoryAction)

        # suction
        self.enable_suction = enable_suction
        if enable_suction is True:
            suction_action_name = '{}/suction_control'.format(ns)
            self.suction_control_client = actionlib.SimpleActionClient(
                suction_action_name,
                tmc_suction.msg.SuctionControlAction)
            try:
                if not self.suction_control_client.wait_for_server(
                        rospy.Duration(10.0)):
                    raise Exception(suction_action_name + ' does not exist')
            except Exception as e:
                rospy.logerr(e)
                sys.exit(1)
            self._suction_command_pub = rospy.Publisher(
                '{}/command_suction'.format(ns),
                std_msgs.msg.Bool, queue_size=1)
            self._pressor_sensor_msg = None
            self._pressor_sensor_sub = rospy.Subscriber(
                '{}/pressure_sensor'.format(ns),
                std_msgs.msg.Bool,
                callback=self._pressor_sensor_callback,
                queue_size=1)

        self.led_pub = rospy.Publisher('{}/command_status_led_rgb'.format(ns),
                                       std_msgs.msg.ColorRGBA,
                                       queue_size=1)
        self.display_image_pub = rospy.Publisher(
            '/robot_mount_wui/video_mode',
            std_msgs.msg.String,
            queue_size=1)

        self._linear_weight = 3.0
        self._angular_weight = 1.0
        self._joint_weights = {}
        self._use_base_timeopt = True

        self._position_control_clients = []
        arm_config = "/hsrb/arm_trajectory_controller"
        self._position_control_clients.append(
            trajectory.TrajectoryController(arm_config))
        head_config = "/hsrb/head_trajectory_controller"
        self._position_control_clients.append(
            trajectory.TrajectoryController(head_config))
        hand_config = "/hsrb/gripper_controller"
        self._position_control_clients.append(
            trajectory.TrajectoryController(hand_config))
        base_config = "/hsrb/omni_base_controller"
        self._base_client = trajectory.TrajectoryController(
            base_config, "/base_coordinates")
        self._position_control_clients.append(self._base_client)

        self._end_effector_frames = [
            "hand_palm_link", "hand_l_finger_vacuum_frame"]
        self._end_effector_frame = "hand_palm_link"
        self._looking_hand_constraint = True

    @property
    def end_effector_frame(self):
        """Get or set the target end effector frame of motion planning.

        This attribute affects behaviors of following methods:
        * get_end_effector_pose
        * move_end_effector_pose
        * move_end_effector_by_line
        """
        return self._end_effector_frame

    @end_effector_frame.setter
    def end_effector_frame(self, value):
        if value in set(self._end_effector_frames):
            self._end_effector_frame = value
        else:
            msg = "`ref_frame_id` must be one of end-effector frames({0})"
            raise ValueError(msg.format(self._end_effector_frames))

    @property
    def linear_weight(self):
        return self._linear_weight

    @linear_weight.setter
    def linear_weight(self, value):
        f_value = max(min(float(value), 100.0), 0.0)
        self._linear_weight = f_value

    @property
    def angular_weight(self):
        return self._angular_weight

    @angular_weight.setter
    def angular_weight(self, value):
        f_value = max(min(float(value), 100.0), 0.0)
        self._angular_weight = f_value

    @property
    def joint_weights(self):
        return self._joint_weights

    @joint_weights.setter
    def joint_weights(self, value):
        if not isinstance(value, dict):
            raise ValueError("value should be dictionary")
        for key, weight in value.iteritems():
            if key not in ["wrist_flex_joint",
                           "wrist_roll_joint",
                           "arm_roll_joint",
                           "arm_flex_joint",
                           "arm_lift_joint",
                           "hand_motor_joint",
                           "head_pan_joint",
                           "head_tilt_joint"]:
                raise ValueError(key + " is not in motion planning joints")
            if float(weight) <= 0.0:
                raise ValueError("weight should be positive")
        self._joint_weights = {key: float(weight)
                               for key, weight in value.iteritems()}

    @property
    def use_base_timeopt(self):
        """If true, time-optimal filter is applied to a base trajectory.

        Returns
        -------
        self._use_base_timeopt : bool
            flag of time-optical filter.
        """
        return self._use_base_timeopt

    @use_base_timeopt.setter
    def use_base_timeopt(self, value):
        self._use_base_timeopt = value

    @property
    def rarm_controller(self):
        return dict(
            controller_type='rarm_controller',
            controller_action='arm_trajectory_controller/follow_joint_trajectory',  # NOQA
            controller_state='arm_trajectory_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=['arm_lift_joint',
                         'arm_flex_joint',
                         'arm_roll_joint',
                         'wrist_flex_joint',
                         'wrist_roll_joint']
        )

    @property
    def head_controller(self):
        return dict(
            controller_type='head_controller',
            controller_action='head_trajectory_controller/follow_joint_trajectory',  # NOQA
            controller_state='head_trajectory_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=['head_tilt_joint',
                         'head_pan_joint'])

    def default_controller(self):
        return [self.rarm_controller,
                self.head_controller]

    def _send_suction_goal(self, timeout, switch, wait=True):
        if self.enable_suction is False:
            rospy.logwarn('sunction is not enabled.')

        if not isinstance(switch, bool):
            raise TypeError('suction_on.data should be bool, get {}'.format(
                type(switch)))
        if isinstance(timeout, Number):
            timeout = rospy.Duration(timeout)
        if not isinstance(timeout, rospy.rostime.Duration):
            raise TypeError('timeout should be rospy.rostime.Duration, get {}'.
                            format(type(timeout)))

        # Send a goal to start suction
        suction_on_goal = tmc_suction.msg.SuctionControlGoal()
        suction_on_goal.timeout = timeout
        suction_on_goal.suction_on.data = switch
        if switch is True:
            rospy.loginfo('Suction ON will start. timeout: {}[sec]'
                          .format(timeout.to_sec()))
        else:
            rospy.loginfo('Suction OFF will start.'
                          'timeout: {}[sec]'.format(timeout.to_sec()))
        try:
            self.suction_control_client.send_goal(suction_on_goal)
            if wait is True:
                self.suction_control_client.wait_for_result()
        except KeyboardInterrupt:
            # self.suction_control_client.cancel_goal()
            # TODO(cancel goal cound not work well.)
            if switch is True:
                self._suction_command_pub.publish(not switch)
        return self.suction_control_client

    @property
    def pressure_sensor(self):
        """Get a sensor value (On/Off) of a suction-nozzle sensor.

        Returns
        -------
        self._pressor_sensor_msg.data : bool
            True if ON.
        """
        if self.enable_suction:
            return self._pressor_sensor_msg.data
        else:
            return False

    def _pressor_sensor_callback(self, msg):
        self._pressor_sensor_msg = msg

    def start_suction(self, timeout=10.0, wait=True):
        return self._send_suction_goal(timeout, True, wait=wait)

    def stop_suction(self, timeout=10.0, wait=True):
        return self._send_suction_goal(timeout, False, wait=wait)

    def is_suctioning(self):
        if self.enable_suction is False:
            rospy.logwarn('sunction is not enabled.')
            return False
        act = self.start_suction(timeout=0.0, wait=True)
        return act.get_state() == GoalStatus.SUCCEEDED

    def gripper_distance(self, distance=None, control_time=3.0):
        """Gripper distance function.

        Parameters
        ----------
        distance : None or float
            If this value is `None`, return gripper distance.
            If `float` value is specified, set gripper distance.
            Distance between gripper finger tips [m]
        """
        if distance is None:
            return self.get_gripper_distance()
        self.set_gripper_distance(distance, control_time)

    def get_gripper_distance(self):
        joint_state = self._joint_state_msg
        hand_motor_pos = joint_state.position[
            joint_state.name.index('hand_motor_joint')]
        hand_left_position = joint_state.position[
            joint_state.name.index(
                'hand_l_spring_proximal_joint')] + hand_motor_pos
        hand_right_position = joint_state.position[
            joint_state.name.index(
                'hand_r_spring_proximal_joint')] + hand_motor_pos
        return ((math.sin(hand_left_position) +
                 math.sin(hand_right_position)) *
                _PROXIMAL_TO_DISTAL_Z +
                2 * (_PALM_TO_PROXIMAL_Y -
                     (_DISTAL_TO_TIP_Y *
                      math.cos(_DISTAL_JOINT_ANGLE_OFFSET) +
                      _DISTAL_TO_TIP_Z *
                      math.sin(_DISTAL_JOINT_ANGLE_OFFSET))))

    def set_gripper_distance(self, distance, control_time=3.0):
        """Command set gripper finger tip distance.

        Parameters
        ----------
        distance : float
            Distance between gripper finger tips [m]
        """
        if distance > _DISTANCE_MAX:
            open_angle = _HAND_MOTOR_JOINT_MAX
            self.gripper_command(open_angle)
        elif distance < _DISTANCE_MIN:
            open_angle = _HAND_MOTOR_JOINT_MIN
            self.gripper_command(open_angle)
        else:
            _DISTANCE_CONTROL_RATE = 10.0
            goal = control_msgs.msg.FollowJointTrajectoryGoal()
            goal.trajectory.joint_names = ['hand_motor_joint']
            goal.trajectory.points = [
                trajectory_msgs.msg.JointTrajectoryPoint(
                    time_from_start=rospy.Duration(
                        1.0 / _DISTANCE_CONTROL_RATE))
            ]

            start_time = rospy.Time().now()
            elapsed_time = rospy.Duration(0.0)
            ierror = 0.0
            theta_ref = math.asin(((distance / 2 -
                                    (_PALM_TO_PROXIMAL_Y -
                                     (_DISTAL_TO_TIP_Y *
                                      math.cos(_DISTAL_JOINT_ANGLE_OFFSET) +
                                      _DISTAL_TO_TIP_Z *
                                      math.sin(_DISTAL_JOINT_ANGLE_OFFSET)))) /
                                   _PROXIMAL_TO_DISTAL_Z))
            rate = rospy.Rate(_DISTANCE_CONTROL_RATE)
            last_movement_time = rospy.Time.now()
            while elapsed_time.to_sec() < control_time:
                try:
                    error = distance - self.get_gripper_distance()
                    if abs(error) > _DISTANCE_CONTROL_STALL_THRESHOLD:
                        last_movement_time = rospy.Time.now()
                    if((rospy.Time.now() - last_movement_time).to_sec() >
                       _DISTANCE_CONTROL_STALL_TIMEOUT):
                        break
                    ierror += error
                    open_angle = (theta_ref +
                                  _DISTANCE_CONTROL_PGAIN * error +
                                  _DISTANCE_CONTROL_IGAIN * ierror)
                    goal.trajectory.points = [
                        trajectory_msgs.msg.JointTrajectoryPoint(
                            positions=[open_angle],
                            time_from_start=rospy.Duration(
                                _DISTANCE_CONTROL_TIME_FROM_START))
                    ]
                    self.gripper_follow_joint_trajectory_client.send_goal(goal)
                    elapsed_time = rospy.Time().now() - start_time
                except KeyboardInterrupt:
                    self.gripper_follow_joint_trajectory_client.cancel_goal()
                    return
                rate.sleep()

    def apply_force(self, effort, delicate=False, timeout=20.0):
        """Command a gripper to execute applying force.

        Parameters
        ----------
        effort : float
            Force applied to grasping [N]
            'effort' should be positive number
        delicate : bool
            Force control is on when delicate is ``True``
            The range force control works well
            is 0.2 [N] < effort < 0.6 [N]

        Returns
        -------
            None
        """
        if effort < 0.0:
            raise ValueError("negative effort is set. "
                             'effort should be greather than 0. '
                             'get {}'.format(effort))
        goal = tmc_control_msgs.msg.GripperApplyEffortGoal()
        goal.effort = - effort * _HAND_MOMENT_ARM_LENGTH
        client = self.grasp_client
        if delicate:
            if effort < _GRIPPER_APPLY_FORCE_DELICATE_THRESHOLD:
                goal.effort = effort
                client = self.apply_force_client
            else:
                rospy.logwarn(
                    "Since effort is high, force control become invalid. "
                    "delicate effort should be smaller than {}".format(
                        _GRIPPER_APPLY_FORCE_DELICATE_THRESHOLD))

        client.send_goal(goal)
        try:
            timeout = rospy.Duration(timeout)
            if client.wait_for_result(timeout):
                client.get_result()
                state = client.get_state()
                if state != actionlib.GoalStatus.SUCCEEDED:
                    raise RuntimeError()("Failed to apply force")
            else:
                client.cancel_goal()
                raise RuntimeError()("Apply force timed out")
        except KeyboardInterrupt:
            client.cancel_goal()

    def gripper_command(self, open_angle, motion_time=1.0,
                        timeout=10.0):
        """Command open a gripper

        Parameters
        ----------
        open_angle : float
            How much angle to open[rad]
        motion_time : float
            Time to execute command[s]

        Returns
        -------
            None
        """
        goal = control_msgs.msg.FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ['hand_motor_joint']
        goal.trajectory.points = [
            trajectory_msgs.msg.JointTrajectoryPoint(
                positions=[open_angle],
                time_from_start=rospy.Duration(motion_time))
        ]

        self.gripper_follow_joint_trajectory_client.send_goal(goal)
        timeout = rospy.Duration(timeout)
        try:
            if self.gripper_follow_joint_trajectory_client.wait_for_result(
                    timeout):
                s = self.gripper_follow_joint_trajectory_client.get_state()
                if s != actionlib.GoalStatus.SUCCEEDED:
                    raise RuntimeError("Failed to follow commanded trajectory")
            else:
                self.gripper_follow_joint_trajectory_client.cancel_goal()
                raise RuntimeError("Timed out")
        except KeyboardInterrupt:
            self.gripper_follow_joint_trajectory_client.cancel_goal()

    def switch_head_rgbd_map_merger(self, switch):
        if not isinstance(switch, bool):
            raise TypeError('value switch should be bool, get {}'.format(
                type(switch)))
        client = dynamic_reconfigure.client.Client(
            '/tmc_map_merger/inputs/head_rgbd_sensor', timeout=10.0)
        res = client.update_configuration({"enable": switch})
        return res['enable'] == switch

    def enable_head_rgbd_map_merger(self):
        return self.switch_head_rgbd_map_merger(True)

    def disable_head_rgbd_map_merger(self):
        return self.switch_head_rgbd_map_merger(False)

    def switch_base_scan_map_merger(self, switch):
        if not isinstance(switch, bool):
            raise TypeError('value switch should be bool, get {}'.format(
                type(switch)))
        client = dynamic_reconfigure.client.Client(
            '/tmc_map_merger/inputs/base_scan', timeout=10.0)
        res = client.update_configuration({"enable": switch})
        return res['enable'] == switch

    def enable_base_scan_map_merger(self):
        return self.switch_base_scan_map_merger(True)

    def disable_base_scan_map_merger(self):
        return self.switch_base_scan_map_merger(False)

    def start_viewpoint_controller(self):
        return rospy.ServiceProxy(
            '/viewpoint_controller/start',
            std_srvs.srv.Empty)()

    def stop_viewpoint_controller(self):
        return rospy.ServiceProxy(
            '/viewpoint_controller/stop',
            std_srvs.srv.Empty)()

    def change_led(self,
                   r=0.0,
                   g=0.0,
                   b=0.0,
                   duration=1.0):
        '''Change LED color on HSR's back

        Parameters
        ----------
        r : float
            0.0-1.0 value (ratio divided by 255.0)
        g : float
            0.0-1.0 value (ratio divided by 255.0)
        b : float
            0.0-1.0 value (ratio divided by 255.0)
        duration: float
            duration for lightning led
        '''
        color = std_msgs.msg.ColorRGBA(
            r=max(0.0, min(r, 1.0)),
            g=max(0.0, min(g, 1.0)),
            b=max(0.0, min(b, 1.0)),
            a=1.0)
        rospy.loginfo(
            "LED changes: r: {} b: {} g: {}".
            format(r, g, b))
        start = rospy.Time.now()
        rate = rospy.Rate(50.0)
        while not rospy.is_shutdown() and \
                (rospy.Time.now() - start).to_sec() < duration:
            self.led_pub.publish(color)
            rate.sleep()

    def display_image(self, image_topic_name):
        """Display image on HSR

        Parameters
        ----------
        image_topic_name : string
            topic name of image

        Examples
        --------
        >>> ri = HSRBRobotInterface()
        >>> ri.display_image('/hsrb/head_rgbd_sensor/rgb/image_rect_color')
        """
        self.display_image_pub.publish(
            std_msgs.msg.String(data=image_topic_name))

    def move_end_effector_pose(self, pose, ref_frame_id=None):
        """Move an end effector to a given pose.

        Parameters
        ----------
        pose : skrobot.coordinates.Coordinates or list[skrobot.coordinates.Coordinates]
            The target pose(s) of the end effector frame.
        ref_frame_id : str or None
            A base frame of an end effector.
            The default is the robot frame(```base_footprint``).
        """
        # Default is the robot frame (the base frame)
        if ref_frame_id is None:
            ref_frame_id = 'base_footprint'

        if isinstance(pose, list):
            ref_to_hand_poses = pose
        else:
            ref_to_hand_poses = [pose]

        odom_to_ref_pose = self._lookup_odom_to_ref(ref_frame_id)
        odom_to_ref_transform = tf_utils.geometry_pose_to_coords(odom_to_ref_pose)
        odom_to_hand_poses = []
        for ref_to_hand in ref_to_hand_poses:
            odom_to_hand = odom_to_ref_transform.copy_worldcoords().transform(ref_to_hand)
            odom_to_hand_poses.append(tf_utils.coords_to_geometry_pose(odom_to_hand))

        req = self._generate_planning_request(
            tmc_planning_msgs.srv.PlanWithHandGoalsRequest)
        req.origin_to_hand_goals = odom_to_hand_poses
        req.ref_frame_id = self._end_effector_frame

        service_name = '/plan_with_hand_goals'
        plan_service = rospy.ServiceProxy(
            service_name, tmc_planning_msgs.srv.PlanWithHandGoals)
        res = plan_service.call(req)
        if res.error_code.val != ArmManipulationErrorCodes.SUCCESS:
            msg = "Fail to plan move_endpoint"
            raise exceptions.MotionPlanningError(msg, res.error_code)
        res.base_solution.header.frame_id = 'odom'
        constrained_traj = self._constrain_trajectories(res.solution,
                                                        res.base_solution)
        self._execute_trajectory(constrained_traj)

    def _lookup_odom_to_ref(self, ref_frame_id, timeout=5.0):
        """Resolve current reference frame transformation from ``odom``.

        Parameters
        ----------
        ref_frame_id : str
            reference frame id
        timeout : float
            lookup transform's timeout.

        Returns
        -------
        pose : geometry_msgs.msg.Pose
            A transform from robot ``odom`` to ``ref_frame_id``.
        """
        odom_to_ref_ros = self.tf_listener.lookup_transform(
            'odom',
            ref_frame_id,
            rospy.Time.now(),
            rospy.Duration(timeout)).transform
        pose = geometry_msgs.msg.Pose()
        pose.position.x = odom_to_ref_ros.translation.x
        pose.position.y = odom_to_ref_ros.translation.y
        pose.position.z = odom_to_ref_ros.translation.z
        pose.orientation.x = odom_to_ref_ros.rotation.x
        pose.orientation.y = odom_to_ref_ros.rotation.y
        pose.orientation.z = odom_to_ref_ros.rotation.z
        pose.orientation.w = odom_to_ref_ros.rotation.w
        return pose

    def _generate_planning_request(self, request_type, max_iteration=10000,
                                   planning_goal_deviation=0.3,
                                   planning_goal_generation=0.3,
                                   planning_timeout=10.0):
        """Generate a planning request and assign common parameters to it.

        Parameters
        ----------
        request_type : tmc_planning_msgs.srv.PlanWithHandGoalsRequest
            A type of "planning service request".
        max_iteration : int
            Max number of iteration of moition planning.
        planning_goal_deviation : float
            Goal deviation in motion planning
        planning_goal_generation : float
            Goal generation probability in moition planning.
        planning_timeout : float
            Timeout for motion planning [sec]

        Retruns
        -------
        request : tmc_planning_msgs.srv.PlanWithXXX
            An instance with common parameters.
        """
        request = request_type()
        request.origin_to_basejoint = self._lookup_odom_to_ref(
            'base_footprint')
        request.initial_joint_state = self._joint_state_msg
        request.timeout = rospy.Duration(planning_timeout)
        request.max_iteration = max_iteration
        # TODO(iory) add collision world
        # if self._collision_world is not None:
        #     snapshot = self._collision_world.snapshot(
        #         settings.get_frame('odom'))
        #     request.environment_before_planning = snapshot

        if request_type is tmc_planning_msgs.srv.PlanWithJointGoalsRequest:
            request.base_movement_type.val = BaseMovementType.NONE
            return request
        else:
            use_joints = set([b'wrist_flex_joint',
                              b'wrist_roll_joint',
                              b'arm_roll_joint',
                              b'arm_flex_joint',
                              b'arm_lift_joint'])
            if self._looking_hand_constraint:
                use_joints.update(["head_pan_joint", "head_tilt_joint"])
                request.extra_goal_constraints.append(
                    'hsrb_planner_plugins/LookHand')
            request.use_joints = use_joints
            request.base_movement_type.val = BaseMovementType.PLANAR
            request.uniform_bound_sampling = False
            request.deviation_for_bound_sampling = planning_goal_deviation
            request.probability_goal_generate = planning_goal_generation
            request.weighted_joints = ['_linear_base', '_rotational_base']
            request.weighted_joints.extend(self._joint_weights.keys())
            request.weight = [self._linear_weight, self._angular_weight]
            request.weight.extend(self._joint_weights.values())
            return request

    def _constrain_trajectories(self, joint_trajectory, base_trajectory=None,
                                tf_timeout=5.0):
        """Apply constraints to given trajectories.

        Parameters:
            joint_trajectory (trajectory_msgs.msg.JointTrajectory):
                A upper body trajectory
            base_trajectory (trajectory_msgs.msg.JointTrajectory):
                A base trajectory
        Returns:
            trajectory_msgs.msg.JointTrajectory:
                A constrained trajectory
        Raises:
            TrajectoryFilterError:
                Failed to execute trajectory-filtering
        """
        if base_trajectory:
            odom_base_trajectory = trajectory.transform_base_trajectory(
                base_trajectory, self.tf_listener.tf_listener, tf_timeout,
                self._base_client.joint_names)
            merged_traj = trajectory.merge(joint_trajectory,
                                           odom_base_trajectory)
        else:
            merged_traj = joint_trajectory

        filtered_merged_traj = None
        if self._use_base_timeopt:
            start_state = self._joint_state_msg
            # use traj first point for odom
            if base_trajectory:
                start_state.name += self._base_client.joint_names
                start_state.position += \
                    tuple(odom_base_trajectory.points[0].positions)
            filtered_merged_traj = trajectory.hsr_timeopt_filter(
                merged_traj, start_state)
        if filtered_merged_traj is None:
            filtered_merged_traj = trajectory.constraint_filter(merged_traj)
        return filtered_merged_traj

    def _execute_trajectory(self, joint_traj):
        """Execute a trajectory with given action clients.

        Action clients that actually execute trajectories are selected
        automatically.

        Parameters:
            joint_traj (trajectory_msgs.msg.JointTrajectory):
                A trajectory to be executed
        Returns:
            None
        """
        clients = []
        # TODO(iory) Enable impedance controller
        # if self._impedance_client.config is not None:
        #     clients.append(self._impedance_client)
        # else:
        for client in self._position_control_clients:
            for joint in joint_traj.joint_names:
                if joint in client.joint_names:
                    clients.append(client)
                    break
        joint_states = self._joint_state_msg

        for client in clients:
            traj = trajectory.extract(joint_traj, client.joint_names,
                                      joint_states)
            client.submit(traj)

        trajectory.wait_controllers(clients)

    def clear_costmap(self):
        pass
