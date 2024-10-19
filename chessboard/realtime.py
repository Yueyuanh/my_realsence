import cv2
import numpy as np

# 定义棋盘格的尺寸，(列数, 行数) - 内角点的数量
chessboard_size = (10, 7)  # 棋盘格为 11x8
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 准备棋盘格的 3D 世界坐标
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# 实时捕获并检测棋盘格
while True:
    ret, frame = cap.read()  # 读取帧
    if not ret:
        print("Error: Failed to capture image")
        break

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # 角点细化
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 在图像上绘制角点
        cv2.drawChessboardCorners(frame, chessboard_size, corners_refined, ret)

    # 显示实时视频流
    cv2.imshow('Real-time Chessboard Detection', frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
