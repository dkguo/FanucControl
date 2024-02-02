import pickle
import socket
import numpy as np
import struct
import time

import trimesh
from scipy.spatial.transform import Rotation as R
import numpy as np
# import cv2
# import rospy
import os, time
from datetime import datetime
import pybullet as p
import numpy as np
import math as m
from scipy.spatial.transform import Rotation as sciR


# ------------------------------------------------
class RobotController(object):
    def __init__(self, robot_id, udp_ip_in, udp_port_in, udp_ip_out,
                 udp_port_out_arm, udp_port_out_gripper, simulation_only=False):

        if not simulation_only:
            self.robot_id = robot_id  # 1: right arm / 2: left arm
            self.udp_ip_in = udp_ip_in
            self.udp_port_in = udp_port_in
            self.udp_ip_out = udp_ip_out
            self.udp_port_out_arm = udp_port_out_arm
            self.udp_port_out_gripper = udp_port_out_gripper

            self.s_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.s_in.bind((self.udp_ip_in, self.udp_port_in))
            self.s_in.setblocking(0)  # socket non-blocking mode

            self.unpacker = struct.Struct("6d 6d 6d 6d 6d 6d")
            self.s_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.joint_pos, self.joint_vel, self.ati_forcetorque = None, None, None
        self.gripper_close = False

        self.pybullet = RobotPybullet(use_gui=True)

    # ----------------------------------
    def isConnected(self):
        self.receiveState()
        if self.joint_pos is None:
            return False
        else:
            return True

            # ----------------------------------

    def receiveState(self):
        # receive all the data in the buffer until the most recent data
        # only keep the most recent data
        data = None
        while True:
            try:
                data, _ = self.s_in.recvfrom(1024)
            except socket.error as e:
                break

        # if receive new data, update the robot state
        if data:
            unpacked_data = np.array(self.unpacker.unpack(data))
            if self.robot_id == 1:
                self.joint_pos, self.joint_vel, self.ati_forcetorque = unpacked_data[0:6], unpacked_data[
                                                                                           6:12], unpacked_data[12:18]
            elif self.robot_id == 2:
                self.joint_pos, self.joint_vel, self.ati_forcetorque = unpacked_data[18:24], unpacked_data[
                                                                                             24:30], unpacked_data[
                                                                                                     30:36]
            else:
                return ValueError("Invalid robot_id, must be 1 (right arm) or 2 (left arm).")

            self.joint_pos = np.deg2rad(self.joint_pos)
            self.joint_vel = np.deg2rad(self.joint_vel)

    # ----------------------------------
    def getGripperStatus(self):
        if self.gripper_close:
            return 1
        else:
            return 0

    # ----------------------------------
    def getJointPos(self):
        self.receiveState()
        return self.joint_pos.copy()

    # ----------------------------------
    def getTcpPose(self):
        self.receiveState()
        pos, quat = self.pybullet.getTcpPose(self.joint_pos)
        return np.array(pos), np.array(quat)

    # ----------------------------------
    def getForceTorqueEE(self):
        self.receiveState()
        return np.array(self.ati_forcetorque)

    # ----------------------------------
    def moveToJointPos(self, target_joint_pos, block=True):
        if not simulation_only:
            target = np.rad2deg(target_joint_pos).astype('d').tobytes()
            self.s_out.sendto(target, (self.udp_ip_out, self.udp_port_out_arm))

            if block:
                self.receiveState()
                # print("target_joint_pos: ", target_joint_pos)
                # print("self.joint_pos: ", self.joint_pos)

                while np.linalg.norm(target_joint_pos - self.joint_pos) > 5e-3:
                    # print("target_joint_pos: ", target_joint_pos)
                    # print("self.joint_pos: ", self.joint_pos)
                    # print(f"Error: {np.linalg.norm(target_joint_pos - self.joint_pos)}")
                    time.sleep(1e-4)
                    self.receiveState()
                # print("Reach the target joint position.")
                    
            self.pybullet.setJointPos(self.getJointPos())
        else:
            self.pybullet.setJointPos(target_joint_pos)

    # ----------------------------------
    def moveToTcpPose(self, target_tcp_pos, target_tcp_quat, block=True):
        target_joint_pos = self.pybullet.inverseKinematics(target_tcp_pos, target_tcp_quat, rest_joint_pos=None)
        if not simulation_only:
            self.moveToJointPos(target_joint_pos, block)
        self.pybullet.setJointPos(target_joint_pos)

    # ----------------------------------
    def gripperMove(self, block=True):
        if not simulation_only:
            one = np.array([1])
            zero = np.array([0])
            self.s_out.sendto(one, (self.udp_ip_out, self.udp_port_out_gripper))
            time.sleep(0.05)
            self.s_out.sendto(zero, (self.udp_ip_out, self.udp_port_out_gripper))

            time.sleep(0.3)

            self.s_out.sendto(one, (self.udp_ip_out, self.udp_port_out_gripper))
            time.sleep(0.05)
            self.s_out.sendto(zero, (self.udp_ip_out, self.udp_port_out_gripper))

            if block:
                time.sleep(0.01)

            # mimic the record of current gripper status
            self.gripper_close = not self.gripper_close


    # ----------------------------------
    def keyboardCommand(self, trans_step_size=0.005, rot_step_size=np.deg2rad(1)):
        """
            return: offset: [x, y, z, rx, ry, rz]
        """
        offset = np.zeros(6)
        keys = p.getKeyboardEvents()
        stop = False
        move_gripper = False

        for k, v in keys.items():
            if (k == p.B3G_RIGHT_ARROW):
                offset[1] = 1.0
            elif (k == p.B3G_LEFT_ARROW):
                offset[1] = -1.0
            elif (k == p.B3G_UP_ARROW):
                offset[0] = -1.0
            elif (k == p.B3G_DOWN_ARROW):
                offset[0] = 1.0
            elif (k == p.B3G_SHIFT):
                offset[2] = 1.0
            # elif (k == p.B3G_CONTROL):
            #     offset[2] = -1.0
            elif (k == p.B3G_ALT):
                move_gripper = True

            elif (k == ord('x')):
                offset[4] = 1.0
            elif (k == ord('s')):
                offset[4] = -1.0
            elif (k == ord('z')):
                offset[3] = 1.0
            elif (k == ord('c')):
                offset[3] = -1.0
            elif (k == ord('a')):
                offset[5] = -1.0
            elif (k == ord('d')):
                offset[5] = 1.0

            elif (k == ord('q')):
                stop = True

            if k == 46:
                trans_step_size *= 3.0
                rot_step_size *= 3.0

        offset[:3] *= trans_step_size
        offset[3:] *= rot_step_size

        # print("Keyboard command: ", offset, move_gripper, stop)
        # print("Press 'q' to exit and save data.")

        return offset, move_gripper, stop


# ------------------------------------------------
class RobotPybullet():
    initial_positions = {
        'joint_1': 0.0, 'joint_2': np.deg2rad(-33), 'joint_3': np.deg2rad(-33),
        'joint_4': 0.0, 'joint_5': np.deg2rad(-90), 'joint_6': 0.0,
        'gripper_finger_joint1': 0.025, 'gripper_finger_joint2': 0.025
    }

    # ----------------------------------
    def __init__(self, use_gui=True):
        self._physics_client_id = p.connect(p.GUI) if use_gui else p.connect(p.DIRECT)
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
        self.robot_id = p.loadURDF(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fanuc/lrmate_model.urdf'),
                                   basePosition=[0.0, 0.0, 0.0], useFixedBase=True, flags=flags,
                                   physicsClientId=self._physics_client_id)
        self.end_eff_idx = 8

        # reset joints to home position
        self._joint_name_to_ids = {}
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self._physics_client_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self._physics_client_id)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]
            if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
                assert joint_name in self.initial_positions.keys()
                self._joint_name_to_ids[joint_name] = i

        self.debug_gui()

    # ----------------------------------
    def get_joint_angle(self, eef_pos, eef_quat):
        jointPoses = p.calculateInverseKinematics(self.robot_id, self.end_eff_idx, eef_pos, eef_quat,
                                                  lowerLimits=[-3.14 / 2, -3.14, -3.14, -3.14, -3.14, -3.14, 0.0, 0.0],
                                                  upperLimits=[3.14 / 2, 3.14, 3.14, 3.14, 3.14, 3.14, 0.025, 0.025],
                                                  jointRanges=[6.28] * 6 + [0.025, 0.025],
                                                  restPoses=[0.0, 0.0, 0.0, 0.0, np.deg2rad(-90), 0.0, 0.0, 0.0],
                                                  maxNumIterations=700,
                                                  residualThreshold=.001,
                                                  physicsClientId=self._physics_client_id)
        return jointPoses

        # ----------------------------------

    def inverseKinematics(self, eef_pos, eef_quat, rest_joint_pos=None):
        if rest_joint_pos is None:  # initial value for the iterative IK solver
            rest_joint_pos = [0.0, 0.0, 0.0, 0.0, np.deg2rad(-90), 0.0, 0.0, 0.0]
        else:
            rest_joint_pos = rest_joint_pos.copy()
            rest_joint_pos.extend([0.0, 0.0])

        res_joint_pos = p.calculateInverseKinematics(self.robot_id, self.end_eff_idx, eef_pos, eef_quat,
                                                     lowerLimits=[-3.14 / 2, -3.14, -3.14, -3.14, -3.14, -3.14, 0.0,
                                                                  0.0],
                                                     upperLimits=[3.14 / 2, 3.14, 3.14, 3.14, 3.14, 3.14, 0.025, 0.025],
                                                     jointRanges=[6.28] * 6 + [0.025, 0.025],
                                                     restPoses=rest_joint_pos,
                                                     maxNumIterations=10000,
                                                     residualThreshold=.0001,
                                                     physicsClientId=self._physics_client_id)
        return res_joint_pos[0:6]

    # ----------------------------------        
    def jacobianMatrix(self, joint_pos):
        """
            input: 6-dimensional joint positions
        """
        joint_pos = np.array(joint_pos).tolist()
        joint_pos.extend([0, 0])

        linear_jaco, angu_jaco = p.calculateJacobian(self.robot_id, self.end_eff_idx,
                                                     localPosition=[0, 0, 0],
                                                     objPositions=joint_pos,
                                                     objVelocities=[0] * 8,
                                                     objAccelerations=[0] * 8)

        jacobian_matrix = np.concatenate([linear_jaco, angu_jaco], axis=0)[:, 0:6]

        return jacobian_matrix

    # ----------------------------------
    def setJointPos(self, joints):
        for joint_id, target in zip(list(self._joint_name_to_ids.values())[0:6], joints[0:6]):
            p.resetJointState(self.robot_id, joint_id, target, physicsClientId=self._physics_client_id)

    # ----------------------------------
    def getTcpPose(self, joints=None):
        if joints is not None:
            self.setJointPos(joints)
        state = p.getLinkState(self.robot_id, self.end_eff_idx, computeLinkVelocity=1,
                               computeForwardKinematics=1, physicsClientId=self._physics_client_id)
        pos, quat = state[0], state[1]
        return pos, quat

    # ----------------------------------
    def debug_gui(self):
        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=-1, physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=-1, physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=-1, physicsClientId=self._physics_client_id)

        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self.end_eff_idx, physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self.end_eff_idx, physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self.end_eff_idx, physicsClientId=self._physics_client_id)
    
    def get_current_joint_positions(self):
        joint_positions = []
        for joint_id in list(self._joint_name_to_ids.values())[0:6]:
            joint_positions.append(p.getJointState(self.robot_id, joint_id, physicsClientId=self._physics_client_id)[0])
        return joint_positions


def calculate_orientation_quaternion(finger1_pos, finger2_pos):
    # Convert positions to numpy arrays
    p1 = np.array(finger1_pos)
    p2 = np.array(finger2_pos)

    # Calculate the direction vector from finger1 to finger2
    direction_vector = p2 - p1
    direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)

    # Assuming the reference axis is the x-axis
    x_axis = np.array([0, 1, 0])

    # Calculate the rotation required to align the x-axis with the direction vector
    # rotation = R.from_rotvec(np.cross(x_axis, direction_vector_normalized) *
    #                          np.arccos(np.dot(x_axis, direction_vector_normalized)))
    #
    # rotate_y = R.from_euler('y', np.pi)
    # rotation = rotation * rotate_y

    rot = p.getQuaternionFromAxisAngle(np.cross(x_axis, direction_vector_normalized), np.arccos(np.dot(x_axis, direction_vector_normalized)))
    rot_y = p.getQuaternionFromAxisAngle([0, 1, 0], np.pi)
    rot = p.multiplyTransforms([0, 0, 0], rot, [0, 0, 0], rot_y)
    quaternion = rot[1]

    # Get the quaternion
    # quaternion = rotation.as_quat()  # returns (x, y, z, w)

    return quaternion
    

# ------------------------------------------------
if __name__ == '__main__':
    simulation_only = False

    # right arm
    robot = RobotController(robot_id=1, udp_ip_in="192.168.1.200", udp_port_in=57831,
                            udp_ip_out="192.168.1.100", udp_port_out_arm=3826,
                            udp_port_out_gripper=3828, simulation_only=simulation_only)

    # # left arm
    # robot = RobotController(robot_id=2, udp_ip_in="192.168.1.200", udp_port_in=57831,
    #                      udp_ip_out="192.168.1.100", udp_port_out_arm=3827, 
    #                      udp_port_out_gripper=3828)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # add object
    objects_path = './demo_objects'
    object_name = 'plastic_hammer'
    # object_position = [0.4, 0., 0.07]   # pill_bottle
    # object_position = [0.4, 0., 0.085]  # banana
    object_position = [0.4, 0., 0.095]  # hammer with finger tip


    language_level = 'simple'
    i, j = 0, 0

    lowerLimits = [-3.14 / 2, -3.14, -3.14, -3.14, -3.14, -3.14, 0.0, 0.0]
    upperLimits = [3.14 / 2, 3.14, 3.14, 3.14, 3.14, 3.14, 0.025, 0.025]

    mesh = trimesh.load(f'{objects_path}/{object_name}/{object_name}.obj')
    vertices = mesh.vertices
    indices = mesh.faces.reshape(-1)
    objId = p.createCollisionShape(p.GEOM_MESH, vertices=vertices, indices=indices)
    obj = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=objId, basePosition=object_position,
                            baseOrientation=[0, 0, 0, 1])

    grasp_positions = pickle.load(open(f'{objects_path}/{object_name}/{language_level}_grasps.pkl', 'rb'))
    print(grasp_positions.shape)

    
    # move to initial position
    if not simulation_only:
        print('robot tcp pose', robot.getTcpPose())
        print('moving to initial position')
        initial_joint_pos = np.array([0.0, 0.0, 0.0, 0.0, -np.pi / 2, 0.0])
        robot.moveToJointPos(initial_joint_pos, block=True)
        eef_pos, eef_quat = robot.getTcpPose()
        last_move_gripper = False

    # print('Start......')
    # while True:

    #     # robot.pybullet.setJointPos(robot.getJointPos())

    #     # manual control
    #     offset, move_gripper, stop = robot.keyboardCommand(trans_step_size=0.005)
    #     if stop:
    #         exit()

    #     if np.any(offset != 0):
    #         next_eef_pos = eef_pos + offset[0:3]
    #         next_eef_quat = (sciR.from_euler('xyz', offset[3:6]) * sciR.from_quat(eef_quat)).as_quat()

    #         # target_joint_pos = robot.pybullet.inverseKinematics(next_eef_pos, next_eef_quat, rest_joint_pos=None)
    #         # # robot.pybullet.setJointPos(target_joint_pos)
    #         # print('moving to next pose')
    #         # print(next_eef_pos, next_eef_quat)

    #         robot.moveToTcpPose(next_eef_pos, next_eef_quat, block=False)
    #         eef_pos, eef_quat = next_eef_pos, next_eef_quat
    #     if move_gripper and not last_move_gripper:
    #         print('moving fingers')
    #         robot.gripperMove()
    #     last_move_gripper = move_gripper
    #     time.sleep(0.1)

    while True:
        i, j = input('Enter i, j: ').split()
        i, j = int(i), int(j)
        print(i, j)
        if i < 0 or j < 0:
            break
        p1 = grasp_positions[i, j, :3] + object_position
        p2 = grasp_positions[i, j, 3:] + object_position
        # p1 = np.array([0.25, -0.05, 0.25])
        # p2 = np.array([0.35, 0.05, 0.2])
        ball1 = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, 1], visualFramePosition=p1)
        ball2 = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 0, 1], visualFramePosition=p2)
        ball1 = p.createMultiBody(baseMass=0, baseVisualShapeIndex=ball1, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1])
        ball2 = p.createMultiBody(baseMass=0, baseVisualShapeIndex=ball2, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1])

        eef_pos, eef_quat = robot.pybullet.getTcpPose()
        target_grasp_pos = (p1 + p2) / 2
        target_ee_orn = calculate_orientation_quaternion(p1, p2)
        # print(target_ee_pos, target_ee_orn)
        # target_ee_orn = p.getQuaternionFromEuler([0, 0, 0])

        # ik_rest_poses = np.random.uniform(lowerLimits, upperLimits)
        target_ee_pos = target_grasp_pos + np.array([0, 0, 0.1])
        ik_rest_poses = robot.pybullet.get_current_joint_positions()
        target_joint_pos = robot.pybullet.inverseKinematics(target_ee_pos, target_ee_orn, ik_rest_poses)
        robot.pybullet.setJointPos(target_joint_pos)
        if input('Press enter to move down') == 'q':
            break
        robot.moveToJointPos(target_joint_pos, block=True)

        input('Press enter to move down')
        target_ee_pos = target_grasp_pos
        ik_rest_poses = robot.pybullet.get_current_joint_positions()
        target_joint_pos = robot.pybullet.inverseKinematics(target_ee_pos, target_ee_orn, ik_rest_poses)
        robot.moveToJointPos(target_joint_pos, block=True)

        # close gripper
        while True:
            command = input('Press enter to close gripper, others to continue')
            if command == '':
                robot.gripperMove()
                time.sleep(1)
            else:
                break

        # move up
        input('Press enter to move up')
        target_ee_pos = target_grasp_pos + np.array([0, 0, 0.2])
        target_joint_pos = robot.pybullet.inverseKinematics(target_ee_pos, target_ee_orn, ik_rest_poses)
        robot.moveToJointPos(target_joint_pos, block=True)


        # open gripper
        while True:
            command = input('Press enter to open gripper, others to continue')
            if command == '':
                robot.gripperMove()
                time.sleep(1)
            else:
                break

        p.removeBody(ball1)
        p.removeBody(ball2)

        print('moving to initial position...')
        if not simulation_only:
            robot.moveToJointPos(initial_joint_pos, block=True)
            eef_pos, eef_quat = robot.getTcpPose()
            last_move_gripper = False
        else:
            robot.pybullet.setJointPos(initial_joint_pos)

    
    # print(robot.pybullet.getTcpPose())



    # if stop:
    #     break
    # else:
    #     if np.any(offset != 0):
    #         next_eef_pos = eef_pos + offset[0:3]
    #         next_eef_quat = (sciR.from_euler('xyz', offset[3:6]) * sciR.from_quat(eef_quat)).as_quat()
    #         robot.moveToTcpPose(next_eef_pos, next_eef_quat, block=False)
    #         eef_pos, eef_quat = next_eef_pos, next_eef_quat
    #     if move_gripper and not last_move_gripper:
    #         robot.gripperMove()
    #     last_move_gripper = move_gripper

    # print("ee_forcetorque: ", robot.getForceTorqueEE()[0:3])
    # pos, quat = robot.getTcpPose()
    # print("ee_pos: ", pos)

    # print("isConnected: ", robot.isConnected())

    # time.sleep(0.1)

    # robot.send(target_joints_robot_2=np.asarray([0, 0, 0, 0, -90, 0]))
    # log_save_path = os.path.join(robot.save_folder, 'log.npy')
    # np.save(log_save_path, robot.logs)
