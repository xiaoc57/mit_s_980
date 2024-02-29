import json


import open3d as o3d
import numpy as np

# 创建一个线段，代表一个坐标轴，输入参数为轴的起点和终点
def create_axis(start, end, color):
    points = [start, end]
    lines = [[0, 1]]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

# 定义坐标轴的起点和终点来表示你想要的坐标系
# 例如，为了绘制一个左手坐标系，我们可以这样定义X, Y, Z轴
origin = np.array([0, 0, 0])
x_axis = np.array([1, 0, 0])  # +X
y_axis = np.array([0, 1, 0])  # +Y
z_axis = np.array([0, 0, 1]) # -Z，注意在左手系中Z轴与右手系相反

# 创建坐标轴
x_line1 = create_axis(origin, x_axis, [1, 0, 0])  # 红色代表X轴
y_line1 = create_axis(origin, y_axis, [0, 1, 0])  # 绿色代表Y轴
z_line1 = create_axis(origin, z_axis, [0, 0, 1])  # 蓝色代表Z轴

# 可视化
# o3d.visualization.draw_geometries([])

with open("D:\\document\\mit_s_980\\homework\\6S980-hw1\\abf149\\metadata.json", 'r') as f:
    metadata = json.load(f)
    
ex = metadata["extrinsics"]

e =  ex[0]
    
a = np.array(e)
a = np.linalg.inv(a)
origin = np.array([0, 0, 0, 1])
x_axis = np.array([1, 0, 0, 1])  # +X
y_axis = np.array([0, -1, 0, 1])  # +Y
z_axis = np.array([0, 0, -1, 1]) # -Z，注意在左手系中Z轴与右手系相反

# 创建坐标轴
x_line = create_axis((a @ origin)[:3], (a @ x_axis)[:3], [1, 0, 0])  # 红色代表X轴
y_line = create_axis((a @ origin)[:3], (a @ y_axis)[:3], [0, 1, 0])  # 绿色代表Y轴
z_line = create_axis((a @ origin)[:3], (a @ z_axis)[:3], [0, 0, 1])  # 蓝色代表Z轴

# 创建一个球体，代表原点
def create_origin_point(radius=0.05, color=[0, 0, 0]):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.paint_uniform_color(color)  # 黑色代表原点
    return mesh

# 可视化
o3d.visualization.draw_geometries([create_origin_point(), x_line, y_line, z_line,x_line1, y_line1, z_line1])