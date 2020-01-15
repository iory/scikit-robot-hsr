import control_msgs.msg

from skrobot.interfaces.ros.base import ROSRobotInterfaceBase


class HSRBROSRobotInterface(ROSRobotInterfaceBase):

    def __init__(self, *args, **kwargs):
        kwargs['namespace'] = 'hsrb'
        super(HSRBROSRobotInterface, self).__init__(*args, **kwargs)

    @property
    def rarm_controller(self):
        return dict(
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
            controller_action='head_trajectory_controller/follow_joint_trajectory',  # NOQA
            controller_state='head_trajectory_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=['head_tilt_joint',
                         'head_pan_joint'])

    def default_controller(self):
        return [self.rarm_controller,
                self.head_controller]
