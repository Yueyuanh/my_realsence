import os
import cv2
import csv
from math import *
import numpy as np

# p[321.501 242.543]  f[606.25 605.65]
class Calibration:
    def __init__(self):
        self.K = np.array([[863.43926, 0, 645.61236],
                           [0, 858.73954, 406.81203],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float64)
        self.distortion = np.array([[0, 0, 0.0, 0.0, 0]])
        self.target_x_number = 10
        self.target_y_number = 7
        self.target_cell_size = 20

    def angle2rotation(self, x, y, z):
        Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
        Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
        Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        return R

    def gripper2base(self, x, y, z, tx, ty, tz):
        thetaX = x / 180 * pi
        thetaY = y / 180 * pi
        thetaZ = z / 180 * pi
        R_gripper2base = self.angle2rotation(thetaX, thetaY, thetaZ)
        T_gripper2base = np.array([[tx], [ty], [tz]])
        Matrix_gripper2base = np.column_stack([R_gripper2base, T_gripper2base])
        Matrix_gripper2base = np.row_stack((Matrix_gripper2base, np.array([0, 0, 0, 1])))
        R_gripper2base = Matrix_gripper2base[:3, :3]
        T_gripper2base = Matrix_gripper2base[:3, 3].reshape((3, 1))
        return R_gripper2base, T_gripper2base

    def target2camera(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
        ret, corners = cv2.findChessboardCorners(gray, (self.target_x_number, self.target_y_number), None)
        corner_points = np.zeros((2, corners.shape[0]), dtype=np.float64)
        for i in range(corners.shape[0]):
            corner_points[:, i] = corners[i, 0, :]
        object_points = np.zeros((3, self.target_x_number * self.target_y_number), dtype=np.float64)
        count = 0
        for i in range(self.target_y_number):
            for j in range(self.target_x_number):
                object_points[:2, count] = np.array(
                    [(self.target_x_number - j - 1) * self.target_cell_size,
                     (self.target_y_number - i - 1) * self.target_cell_size])
                count += 1
        retval, rvec, tvec = cv2.solvePnP(object_points.T, corner_points.T, self.K, distCoeffs=self.distortion)
        Matrix_target2camera = np.column_stack(((cv2.Rodrigues(rvec))[0], tvec))
        Matrix_target2camera = np.row_stack((Matrix_target2camera, np.array([0, 0, 0, 1])))
        R_target2camera = Matrix_target2camera[:3, :3]
        T_target2camera = Matrix_target2camera[:3, 3].reshape((3, 1))
        return R_target2camera, T_target2camera



    def process(self, img_path, pose_path):
        image_list = []
        for root, dirs, files in os.walk(img_path):
            if files:
                for file in files:
                    image_name = os.path.join(root, file)
                    image_list.append(image_name)
        
        R_target2camera_list = []
        T_target2camera_list = []
        for img_path in image_list:
            img = cv2.imread(img_path)
            R_target2camera, T_target2camera = self.target2camera(img)
            R_target2camera_list.append(R_target2camera)
            T_target2camera_list.append(T_target2camera)

        R_gripper2base_list = []
        T_gripper2base_list = []

        # 打开CSV文件并跳过第一行（表头）
        with open(pose_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # 跳过表头
            for row in csvreader:
                tx, ty, tz = float(row[0]), float(row[1]), float(row[2])
                x, y, z = float(row[3]), float(row[4]), float(row[5])
                R_gripper2base, T_gripper2base = self.gripper2base(x, y, z, tx, ty, tz)
                R_gripper2base_list.append(R_gripper2base)
                T_gripper2base_list.append(T_gripper2base)

        R_camera2base, T_camera2base = cv2.calibrateHandEye(R_gripper2base_list, T_gripper2base_list,
                                                            R_target2camera_list, T_target2camera_list)
        return R_camera2base, T_camera2base, R_gripper2base_list, T_gripper2base_list, R_target2camera_list, T_target2camera_list


    def check_result(self, R_cb, T_cb, R_gb, T_gb, R_tc, T_tc):
        for i in range(len(R_gb)):
            RT_gripper2base = np.column_stack((R_gb[i], T_gb[i]))
            RT_gripper2base = np.row_stack((RT_gripper2base, np.array([0, 0, 0, 1])))

            RT_camera_to_gripper = np.column_stack((R_cb, T_cb))
            RT_camera_to_gripper = np.row_stack((RT_camera_to_gripper, np.array([0, 0, 0, 1])))
            print(RT_camera_to_gripper)  # 这个就是手眼矩阵

            RT_target_to_camera = np.column_stack((R_tc[i], T_tc[i]))
            RT_target_to_camera = np.row_stack((RT_target_to_camera, np.array([0, 0, 0, 1])))
            RT_target_to_base = RT_gripper2base @ RT_camera_to_gripper @ RT_target_to_camera
            
            print("第{}次验证结果为:".format(i))
            print(RT_target_to_base)
            print('')


if __name__ == "__main__":
    image_path = r"./imgs"
    pose_path = r"./pose.csv"  # CSV 文件路径
    calibrator = Calibration()
    R_cb, T_cb, R_gb, T_gb, R_tc, T_tc = calibrator.process(image_path, pose_path)
    calibrator.check_result(R_cb, T_cb, R_gb, T_gb, R_tc, T_tc)
