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

from skrobot.interfaces.ros.move_base import ROSRobotMoveBaseInterface


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
        kwargs['use_tf2'] = True
        kwargs['namespace'] = 'hsrb'
        kwargs['move_base_action_name'] = 'move_base/move'
        kwargs['odom_topic'] = 'hsrb/odom'
        kwargs['base_controller_joint_names'] = ["odom_x", "odom_y", "odom_t"]
        kwargs['base_controller_action_name'] = \
            'hsrb/omni_base_controller/follow_joint_trajectory'
        super(HSRBROSRobotInterface, self).__init__(*args, **kwargs)

        # gripper
        self.enable_gripper = enable_gripper
        if enable_gripper is True:
            self.grasp_client = actionlib.SimpleActionClient(
                '/hsrb/gripper_controller/grasp',
                tmc_control_msgs.msg.GripperApplyEffortAction
            )
            self.apply_force_client = actionlib.SimpleActionClient(
                '/hsrb/gripper_controller/apply_force',
                tmc_control_msgs.msg.GripperApplyEffortAction
            )
            self.gripper_follow_joint_trajectory_client = \
                actionlib.SimpleActionClient(
                    '/hsrb/gripper_controller/follow_joint_trajectory',
                    control_msgs.msg.FollowJointTrajectoryAction)

        # suction
        self.enable_suction = enable_suction
        if enable_suction is True:
            suction_action_name = '/hsrb/suction_control'
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
                '/hsrb/command_suction',
                std_msgs.msg.Bool, queue_size=1)
            self._pressor_sensor_msg = None
            self._pressor_sensor_sub = rospy.Subscriber(
                '/hsrb/pressure_sensor',
                std_msgs.msg.Bool,
                callback=self._pressor_sensor_callback,
                queue_size=1)

        self.led_pub = rospy.Publisher('/hsrb/command_status_led_rgb',
                                       std_msgs.msg.ColorRGBA,
                                       queue_size=1)
        self.display_image_pub = rospy.Publisher(
            '/robot_mount_wui/video_mode',
            std_msgs.msg.String,
            queue_size=1)

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
