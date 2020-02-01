from numbers import Number
import math
import sys

import actionlib
from actionlib_msgs.msg import GoalStatus
import control_msgs.msg
import dynamic_reconfigure.client
import rospy
import std_msgs.msg
import std_srvs.srv
import tmc_suction.msg


from skrobot.interfaces.ros.move_base import ROSRobotMoveBaseInterface


_PALM_TO_PROXIMAL_Y = 0.0245
_PROXIMAL_TO_DISTAL_Z = 0.07
_DISTAL_JOINT_ANGLE_OFFSET = 0.087
_DISTAL_TO_TIP_Y = 0.01865
_DISTAL_TO_TIP_Z = 0.04289


class HSRBROSRobotInterface(ROSRobotMoveBaseInterface):

    def __init__(self, *args, **kwargs):
        enable_suction = kwargs.pop('enable_suction', True)
        kwargs['use_tf2'] = True
        kwargs['namespace'] = 'hsrb'
        kwargs['move_base_action_name'] = 'move_base/move'
        kwargs['odom_topic'] = 'hsrb/odom'
        kwargs['base_controller_joint_names'] = ["odom_x", "odom_y", "odom_t"]
        kwargs['base_controller_action_name'] = \
            'hsrb/omni_base_controller/follow_joint_trajectory'
        super(HSRBROSRobotInterface, self).__init__(*args, **kwargs)

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
