import cv2
import numpy as np
import os

# 定义棋盘格的尺寸，(列数, 行数) - 内角点的数量
chessboard_size = (10, 7)  # 如果棋盘格是 9x6 的内部格子数
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 准备棋盘格的 3D 世界坐标
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

def detect_chessboard(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load image {image_path}")
        return False

    # 转为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # 角点细化
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 在图像上绘制角点
        img_with_corners = cv2.drawChessboardCorners(img, chessboard_size, corners_refined, ret)

        # 显示图像
        cv2.imshow('Chessboard Corners', img_with_corners)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存带有角点的图像
        output_path = os.path.join('output', os.path.basename(image_path))
        os.makedirs('output', exist_ok=True)
        cv2.imwrite(output_path, img_with_corners)
        print(f"Chessboard corners detected and image saved to {output_path}")
        return True
    else:
        print("Chessboard not found in the image.")
        return False

if __name__ == "__main__":
    image_path = "./imgs/4.jpg"  # 修改为你的图像路径
    detect_chessboard(image_path)
