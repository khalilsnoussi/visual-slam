import cv2 as cv
import numpy as np
import open3d as o3d









scan = np.fromfile("./data/dataset/velodyne/00/velodyne/000000.bin", dtype=np.float32)
scan = scan.reshape((-1, 4))
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(scan[:, :3])
o3d.visualization.draw_geometries([point_cloud])





