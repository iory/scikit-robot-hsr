from io import BytesIO as StringIO
import os

from cached_property import cached_property
import numpy as np
import skrobot
from skrobot.coordinates import CascadedCoords
from skrobot.model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF


class HSRB(RobotModelFromURDF):

    """HSR-b Robot Model

    """

    def __init__(self, *args, **kwargs):

        if not ('urdf_file' in kwargs or 'urdf' in kwargs):
            kwargs['urdf'] = self._urdf()
        super(HSRB, self).__init__(*args, **kwargs)
        self.name = 'hsrb'

        self.joint_list = [
            self.base_roll_joint,
            self.torso_lift_joint,
            self.arm_lift_joint,
            self.arm_flex_joint,
            self.arm_roll_joint,
            self.wrist_flex_joint,
            self.wrist_roll_joint,
            self.head_pan_joint,
            self.head_tilt_joint]

        self.rarm_end_coords = CascadedCoords(
            parent=self.hand_palm_link,
            name='rarm_end_coords')
        self.rarm_end_coords.rotate(
            np.pi,
            [7.0710700e-01, 0.0, 7.07107004e-01])
        self.end_coords = self.rarm_end_coords

    def _urdf(self):
        import rospkg
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('hsrb_description')
        self.resolve_filepath = os.path.join(
            package_path, 'robots', 'hsrb.urdf')
        with open(self.resolve_filepath, 'rb') as f:
            urdf_text = f.read()
        base_transmission_name = 'base_transmission'
        r_index = urdf_text.index(base_transmission_name)
        l_index = urdf_text.index(base_transmission_name, r_index + 1)
        new_urdf_text = urdf_text[:r_index] + \
            'base_r_drive_wheel_joint_transmission' + \
            urdf_text[r_index + len(base_transmission_name):l_index] + \
            'base_l_drive_wheel_joint_transmission' + \
            urdf_text[l_index + len(base_transmission_name):]
        return new_urdf_text

    def load_urdf(self, urdf):
        f = StringIO()
        f.write(urdf)
        f.seek(0)
        f.name = self.resolve_filepath
        self.load_urdf_file(file_obj=f)

    def reset_pose(self):
        return self.angle_vector(
            [0.0, 0.0, 0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, 0.0, 0.0])

    def reset_manip_pose(self):
        return self.angle_vector(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @cached_property
    def rarm(self):
        rarm_links = [self.torso_lift_link,
                      self.arm_lift_link,
                      self.arm_flex_link,
                      self.arm_roll_link,
                      self.wrist_flex_link,
                      self.wrist_ft_sensor_mount_link]

        rarm_joints = []
        for link in rarm_links:
            if hasattr(link, 'joint'):
                rarm_joints.append(link.joint)
        r = RobotModel(link_list=rarm_links,
                       joint_list=rarm_joints)
        r.end_coords = self.rarm_end_coords
        r.inverse_kinematics = lambda *args, **kwargs: self.inverse_kinematics(
            link_list=r.link_list,
            *args, **kwargs)
        return r

    @property
    def interlocking_joint_pairs(self):
        return [(self.torso_lift_joint, self.arm_lift_joint)]

    def calc_jacobian_for_interlocking_joints(
            self, link_list, interlocking_joint_pairs=None):
        if interlocking_joint_pairs is None:
            interlocking_joint_pairs = self.interlocking_joint_pairs
        union_link_list = self.calc_union_link_list(link_list)
        joint_list = list(filter(lambda j: j is not None,
                                 [l.joint for l in union_link_list]))
        pairs = list(
            filter(lambda pair:
                   not ((pair[0] not in joint_list) and
                        (pair[1] not in joint_list)),
                   interlocking_joint_pairs))
        jacobi = np.zeros((len(pairs),
                           self.calc_target_joint_dimension(union_link_list)),
                          'f')
        for i, pair in enumerate(pairs):
            index = sum(
                [j.joint_dof for j in joint_list[:joint_list.index(
                    pair[0])]])
            jacobi[i][index] = 2.0
            index = sum(
                [j.joint_dof for j in joint_list[:joint_list.index(
                    pair[1])]])
            jacobi[i][index] = -1.0
        return jacobi

    def set_weighted_angles_for_interlocking_joints(
            self,
            interlocking_joint_pairs=None):
        if interlocking_joint_pairs is None:
            interlocking_joint_pairs = self.interlocking_joint_pairs
        for pair_a, pair_b in interlocking_joint_pairs:
            unit_angle = (pair_a.joint_angle() + pair_b.joint_angle()) / 3.0
            pair_a.joint_angle(1.0 * unit_angle)
            pair_b.joint_angle(2.0 * unit_angle)

    def inverse_kinematics(
            self,
            target_coords,
            move_target=None,
            link_list=None,
            use_base=False,
            **kwargs):
        move_joints_hook = kwargs.pop('move_joints_hook', [])
        move_joints_hook.append(
                lambda:
                self.set_weighted_angles_for_interlocking_joints())
        if move_target is None:
            move_target = self.end_coords
        if link_list is None:
            link_list = self.link_list
        if use_base:
            rlink = self.root_link
            vlink = skrobot.model.Link(name='virtual_link')
            vjoint = skrobot.model.OmniWheelJoint(
                child_link=self,
                parent_link=vlink,
                min_angle=np.array([-5.0, -5.0, -100]),
                max_angle=np.array([5.0, 5.0, 100])
            )
            vlink.add_joint(vjoint)
            rlink.add_parent_link(vlink)
            vlink.add_child_links(rlink)
            link_list = [vlink] + link_list
            ret = super(HSRB, self).inverse_kinematics(
                target_coords,
                link_list=link_list,
                move_target=move_target,
                additional_jacobi=[
                    lambda ll:
                    self.calc_jacobian_for_interlocking_joints(ll)],
                additional_vel=[
                    lambda ll:
                    self.calc_vel_for_interlocking_joints(ll)],
                move_joints_hook=move_joints_hook,
                **kwargs)
            return ret
        else:
            return super(HSRB, self).inverse_kinematics(
                target_coords,
                link_list=link_list,
                move_target=move_target,
                additional_jacobi=[
                    lambda ll:
                    self.calc_jacobian_for_interlocking_joints(ll)],
                additional_vel=[
                    lambda ll:
                    self.calc_vel_for_interlocking_joints(ll)],
                move_joints_hook=move_joints_hook,
                **kwargs)
