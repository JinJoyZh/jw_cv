import geopy.distance
import numpy as np
import math
from osgeo import gdal


# 定义一个函数计算旋转矩阵
# roll: 翻滚角 pitch: 俯仰角 yaw：航向角
def rotation_matrix(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]])
    
    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])
    
    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]])
    
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

#已知航拍摄像头离地高度、焦距、图像分辨率、翻滚角、俯仰角、航向角，求航拍摄像头镜头中心与航拍图像中某一像素对应的连线与大地的夹角
# 定义一个函数计算角度
def calculate_angle_with_ground(elevation, focal_length, resolution, roll, pitch, yaw, pixel_x, pixel_y):
    # 转换角度到弧度
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    # 生成旋转矩阵
    R = rotation_matrix(roll, pitch, yaw)

    # 相机中心点坐标
    center_x, center_y = resolution[0] / 2, resolution[1] / 2

    # 像素点在相机坐标系中的位置
    x = (pixel_x - center_x) * elevation / focal_length
    y = (pixel_y - center_y) * elevation / focal_length
    z = elevation

    # 像素点的世界坐标
    world_point = np.dot(R, np.array([x, y, z]))

    # 计算与地面的夹角
    angle = math.degrees(math.atan2(world_point[2], math.sqrt(world_point[0]**2 + world_point[1]**2)))
    return angle

#已知航拍摄像头离地高度、焦距、图像分辨率、翻滚角、俯仰角、航向角，求航拍摄像头镜头中心与航拍图像中某一像素对应的连线与地图正东方方向的逆时针夹角。用python实现。
# 定义函数计算角度
def calculate_angle_with_north(elevation, focal_length, resolution, roll, pitch, yaw, pixel_x, pixel_y):
    # 角度转弧度
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    # 生成旋转矩阵
    R = rotation_matrix(roll, pitch, yaw)

    # 计算相机中心点坐标
    center_x, center_y = resolution[0] / 2, resolution[1] / 2

    # 像素点在相机坐标系中的位置
    x = (pixel_x - center_x) * elevation / focal_length
    y = (pixel_y - center_y) * elevation / focal_length
    z = elevation

    # 像素点的世界坐标
    world_point = np.dot(R, np.array([x, y, z]))

    # 计算与地图正东方方向的逆时针夹角
    angle = math.degrees(math.atan2(world_point[1], world_point[0]))
    if angle < 0:
        angle += 360  # 确保角度为正
    return angle


# 使用geocode获取该地点的海拔信息
def calculate_camera_elevation(camera_latitude, camera_longitude, camera_altitude, tif_path='alg/position_calculation/DEM/GDEM_10km.tif'):
    gdal.UseExceptions()
    ds = gdal.Open(tif_path)
    band = ds.GetRasterBand(1)
    elevation = band.ReadAsArray()
    x0, dx, dxdy, y0, dydx, dy = ds.GetGeoTransform()
    new_ncols, new_nrows = int((y0-camera_latitude)/dx), int((camera_longitude-x0)/dx)
    ground_altitude = elevation[new_ncols][new_nrows]
    elevation = camera_altitude - ground_altitude
    return elevation

def calculate_target_coordinates(focal_length, resolution, roll, pitch, yaw, pixel_x, pixel_y, camera_latitude, camera_longitude, camera_altitude):
    elevation = calculate_camera_elevation(camera_latitude, camera_longitude, camera_altitude)
    angle_with_ground = calculate_angle_with_ground(elevation, focal_length, resolution, roll, pitch, yaw, pixel_x, pixel_y)
    angle_with_ground = math.radians(angle_with_ground)
    bearing = calculate_angle_with_north(elevation, focal_length, resolution, roll, pitch, yaw, pixel_x, pixel_y)
    distance = elevation / math.tan(angle_with_ground)
    distance = int(distance)
    camera_pos = (camera_latitude, camera_longitude)
    #把int型距离转化为geopy.distance的格式，单位千米
    distance = geopy.distance.geodesic(distance)
    #计算下目标点坐标
    target = distance.destination(camera_pos, bearing)
    return target.latitude, target.longitude


if __name__ == '__main__':

    focal_length = 35                   # 焦距
    resolution = (4000, 3000)           # 图像分辨率
    roll, pitch, yaw = 20, 45, 10       # 翻滚角、俯仰角、航向角
    pixel_x, pixel_y = 2000, 1500       # 像素位置
    camera_latitude = 39.9042           # 相机纬度
    camera_longitude = 116.4074         # 相机经度
    camera_altitude = 400               # 相机海拔

    target_pos = calculate_target_coordinates(focal_length, resolution, roll, pitch, yaw, pixel_x, pixel_y, camera_latitude, camera_longitude, camera_altitude)
    print(f"目标位置：{target_pos}")







    



