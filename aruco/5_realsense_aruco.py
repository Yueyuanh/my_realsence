import pyrealsense2 as rs
import cv2.aruco as aruco
import cv2
import numpy as np
import time
import os


# 创建 RealSense 管道
pipeline = rs.pipeline()

# 创建配置对象
config = rs.config()

# 启用 RGB 和深度流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 开始管道
pipeline.start(config)

#相机参数，至少为1
camera_matrix = np.array([[863.439262, 0, 645.612360], 
                          [0, 858.739541, 406.812029], 
                          [0, 0, 1]], dtype=float)

# 如果没有畸变，也可以使用全零数组
dist_coeffs = np.array([0.048015,-0.083847,0.011333,0.007398,0.000000], dtype=float)


#设置预定义的字典
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)



# 当前机械臂末端姿态
p_current = np.array([[0.25], [0], [0.2]])  # 当前位置
roll = 0
pitch = 3.1415/2
yaw = 0

# 计算旋转矩阵 R_current
def rotation_matrix(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    return R_z @ R_y @ R_x

def trans(R_aruco,t_aruco):
    R_current = rotation_matrix(roll, pitch, yaw)

    # 构建当前齐次变换矩阵 T_current
    T_current = np.eye(4)
    T_current[0:3, 0:3] = R_current
    T_current[0:3, 3] = p_current.flatten()

    # 手眼标定矩阵 T_cam（单位为毫米）
    T_cam = np.array([[ 0.01818215,  0.02369534,  0.99955387,  0.05921279],
                      [-0.99943896, -0.02769392,  0.01883657,  0.03229603],
                      [ 0.0281279 , -0.99933557,  0.02317851,  0.05390061],
                      [ 0.        ,  0.        ,  0.        ,  1.        ]])


    # Aruco标记的旋转矩阵 R 和平移向量 t（替换为实际值）
    # R_aruco = np.array([[...], [...], [...]])  # Aruco的旋转矩阵
    # t_aruco = np.array([[...], [...], [...]])   # Aruco的平移向量

    # 构建Aruco标记的齐次变换矩阵 T_aruco
    T_aruco = np.eye(4)
    T_aruco[0:3, 0:3] = R_aruco
    T_aruco[0:3, 3] = t_aruco.flatten()

    # 计算目标末端姿态 T_target
    T_target = np.dot(T_aruco, T_cam)

    # 提取目标位姿
    target_position = T_target[0:3, 3]
    target_rotation = T_target[0:3, 0:3]

    print("目标位置:", target_position)
    print("目标旋转矩阵:", target_rotation)

    return target_position
trail=[
            [0.25,  0, 0.2,  0,  3.1415/2  , 0,   0,  5,  1], #base_fream
            [0.25,  0, 0.2,  0,  3.1415/2   , 0,   0,  5,  1], #base_fream
]
def get_aruco_pose(delay):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        exit()

        # 将图像转换为 numpy 数组
    frame = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())



    #灰度化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #使用默认值初始化检测器参数
    parameters =  aruco.DetectorParameters()


    #使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

    #画出标志位置
    aruco.drawDetectedMarkers(frame, corners,ids)

    # print(corners)


    if ids is not None:
        # 估计姿态 (假设每个标记边长为0.05米)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.027, camera_matrix, dist_coeffs)
        # 旋转矩阵 平移矩阵 计算内部的标记角点


        # 绘制检测到的标记和位姿
        for i in range(len(ids)):
            frame=cv2.aruco.drawDetectedMarkers(frame, corners)  # 绘制标记的边框
            frame=cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.05)  # 绘制坐标轴

            # 输出位姿信息
            print(f"Marker ID: {ids[i][0]}")
            print(f"Rotation Vector (rvec): {rvecs[i].flatten()}")
            print(f"Translation Vector (tvec): {tvecs[i].flatten()}")

            target_pose=trans(rvecs[i],tvecs[i])

        tv=tvecs[0].flatten()

        tv = [float('{:.4f}'.format(i)) for i in tv]
        text=str(target_pose)
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
        print(tv[0],tv[1],tv[2])

        trail[1][0]=0.25-tv[1]
        trail[1][1]=0+tv[0]+0.045
        trail[1][2]=0.2-(tv[2]-0.1)

        print(trail[1])


    # 显示 RGB 图像
    cv2.imshow('RGB Image', frame)

    # 按 'q' 键退出
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        exit()



while True:
    get_aruco_pose(5)