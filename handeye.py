import os
import cv2
import csv
from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# p[321.501 242.543]  f[606.25 605.65]
# 可视化函数
def plot_transform(ax, T, label='', length=50):
    """
    绘制齐次矩阵对应的坐标系
    
    参数:
    - ax: matplotlib 3D 轴对象
    - T: 4x4 齐次变换矩阵
    - label: 坐标系标签
    - length: 坐标轴长度
    """
    origin = T[:3, 3]
    
    # 取旋转矩阵的列向量作为坐标轴的方向
    x_axis = T[:3, 0] * length
    y_axis = T[:3, 1] * length
    z_axis = T[:3, 2] * length
    
    # 绘制坐标轴
    ax.quiver(*origin, *x_axis, color='r', label=f'{label} x-axis' if label else '')
    ax.quiver(*origin, *y_axis, color='g', label=f'{label} y-axis' if label else '')
    ax.quiver(*origin, *z_axis, color='b', label=f'{label} z-axis' if label else '')
    
    # 标注原点
    ax.text(*origin, label, size=10, zorder=1, color='k')
class Calibration:
    def __init__(self):

        # 相机内参矩阵
        self.K = np.array([[896.10779, 0, 613.50215],
                           [0, 903.23698, 412.64156],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float64)
        # 畸变矩阵
        self.distortion = np.array([[0.071810, -0.087668, 0.014144, -0.006956, 0.000000]])
        # 棋盘格信息（-1)
        self.target_x_number = 10
        self.target_y_number = 7
        self.target_cell_size = 20 #mm

        # 角度转旋转矩阵
    def angle2rotation(self, x, y, z):
        Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
        Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
        Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])

        # @ 矩阵相乘 三个绕单独轴旋转的合成一个旋转矩阵
        R = Rz @ Ry @ Rx
        return R

        # 加爪转基坐标
    def gripper2base(self, x, y, z, tx, ty, tz):

        thetaX = x 
        thetaY = y 
        thetaZ = z 

        # angle->rotation
        R_gripper2base = self.angle2rotation(thetaX, thetaY, thetaZ)
        T_gripper2base = np.array([[tx], [ty], [tz]])
        # 转齐次矩阵
        Matrix_gripper2base = np.column_stack([R_gripper2base, T_gripper2base])
        Matrix_gripper2base = np.row_stack((Matrix_gripper2base, np.array([0, 0, 0, 1])))

        R_gripper2base = Matrix_gripper2base[:3, :3]
        T_gripper2base = Matrix_gripper2base[:3, 3].reshape((3, 1))
        return R_gripper2base, T_gripper2base

        # 目标->相机坐标系
    def target2camera(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 识别棋盘格
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
        # print(R_target2camera)
        print(T_target2camera)
        # 可视化

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

        R_camera2base, T_camera2base = cv2.calibrateHandEye(R_gripper2base_list, T_gripper2base_list,   #夹爪到base
                                                            R_target2camera_list, T_target2camera_list) #标定板到相机
        return R_camera2base, T_camera2base, R_gripper2base_list, T_gripper2base_list, R_target2camera_list, T_target2camera_list


    def check_result(self, R_cb, T_cb, R_gb, T_gb, R_tc, T_tc):

        #取消numpy的科学计数法显示
        np.set_printoptions(suppress=True, precision=4)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置轴范围
        ax.set_xlim([-1500, 1500])
        ax.set_ylim([-1000, 1000])
        ax.set_zlim([-1000, 1000])

        # 设置标签
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        # 绘制世界坐标系
        plot_transform(ax, np.eye(4), label='World', length=50)

        for i in range(len(R_gb)):
            RT_gripper2base = np.column_stack((R_gb[i], T_gb[i]))
            RT_gripper2base = np.row_stack((RT_gripper2base, np.array([0, 0, 0, 1])))

            RT_camera_to_gripper = np.column_stack((R_cb, T_cb))
            RT_camera_to_gripper = np.row_stack((RT_camera_to_gripper, np.array([0, 0, 0, 1])))

            
            # 验证结果
            RT_target_to_camera = np.column_stack((R_tc[i], T_tc[i]))
            RT_target_to_camera = np.row_stack((RT_target_to_camera, np.array([0, 0, 0, 1])))
            RT_target_to_base = RT_gripper2base @ RT_camera_to_gripper @ RT_target_to_camera

            
            print("第{}次反求标定板结果为:".format(i+1))

            
            # print(RT_gripper2base)
  

            print(RT_target_to_base)
            print('')
            print('')


            if(i+1==len(R_gb)):
                            print("标定结果为")
                            print(RT_camera_to_gripper)  # 这个就是手眼矩阵

            # 可视化 RT_target_to_base
            plot_transform(ax, RT_target_to_base, label=f'Target {i+1}', length=50)
            
        # 显示图例和图像
        ax.legend()
        plt.show()


if __name__ == "__main__":
    image_path = r"./imgs"
    pose_path = r"./pose.csv"  # CSV 文件路径
    calibrator = Calibration()
    R_cb, T_cb, R_gb, T_gb, R_tc, T_tc = calibrator.process(image_path, pose_path)
    calibrator.check_result(R_cb, T_cb, R_gb, T_gb, R_tc, T_tc)
