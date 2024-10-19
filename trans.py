import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Input quaternion and translation components
qw = 0.5034050915790155
qx = -0.5056425525372688
qy = 0.4824275639372173
qz = -0.5081068474145183
x = 0.05921279046297043
y = 0.03229602987131769
z = 0.053900613920715906

# Construct the rotation matrix from quaternion
rotation_matrix = np.array([
    [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
    [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
    [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
])

# Construct the translation vector
translation_vector = np.array([x, y, z])

# Combine rotation and translation into a 4x4 transformation matrix
transformation_matrix = np.eye(4)
transformation_matrix[:3, :3] = rotation_matrix
transformation_matrix[:3, 3] = translation_vector

# Create a figure and axis for 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the axes using the transformation matrix
origin = [0, 0, 0]
x_axis = transformation_matrix[:3, 0]  # First column
y_axis = transformation_matrix[:3, 1]  # Second column
z_axis = transformation_matrix[:3, 2]  # Third column

# Translation vector (last column)
translation = transformation_matrix[:3, 3]

# Plot the coordinate axes
ax.quiver(*origin, *x_axis, color='r', label='X-axis')  # X-axis in red
ax.quiver(*origin, *y_axis, color='g', label='Y-axis')  # Y-axis in green
ax.quiver(*origin, *z_axis, color='b', label='Z-axis')  # Z-axis in blue

# Plot the translation point
ax.scatter(*translation, color='k', label='Translation Point')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Transformation Visualization')

# Set equal aspect ratio by setting axis limits
max_range = np.array([x_axis, y_axis, z_axis]).max()
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])

# Add legend
ax.legend()

# Show plot
plt.show()
