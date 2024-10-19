import numpy as np
from scipy.spatial.transform import Rotation as R

def extract_euler_angles(matrix):
    """
    从4x4的齐次变换矩阵中提取旋转矩阵并将其转换为欧拉角（XYZ顺序）。
    
    参数:
        matrix: 4x4 齐次变换矩阵 (numpy array)

    返回:
        欧拉角 (弧度): 以 [x, y, z] 形式返回的欧拉角
    """
    # 提取旋转矩阵 (3x3)
    rotation_matrix = matrix[:3, :3]
    
    # 使用 scipy 将旋转矩阵转换为欧拉角
    r = R.from_matrix(rotation_matrix)
    
    # 将旋转矩阵转换为 XYZ 顺序的欧拉角（单位为弧度）
    euler_angles = r.as_euler('xyz', degrees=True)
    
    return euler_angles

# 定义一个 4x4 的齐次变换矩阵
transformation_matrix = np.array([[ 3.68630191e-01, -8.95539914e-01,  2.49238930e-01, -3.96404766e+02],
                                  [ 9.11969331e-01,  4.00343487e-01,  8.96494921e-02,  6.83051843e+02],
                                  [-1.80065881e-01,  1.94250751e-01,  9.64283633e-01, -1.01863877e+03],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

# transformation_matrix = np.array([[ 5.88395980e-01,  8.05692718e-01,  6.81866187e-02,  2.59525845e+03],
#                                   [-8.07548304e-01,  5.81313075e-01,  9.97037856e-02,  5.41629361e+02],
#                                   [ 4.06928410e-02, -1.13729295e-01,  9.92678065e+01, -2.60408298e+03],
#                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

# 调用函数，提取并转换欧拉角
euler_angles = extract_euler_angles(transformation_matrix)

print("欧拉角 (角度):", euler_angles)
