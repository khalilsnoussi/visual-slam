import numpy as np

poses_file_path = './ground_truth/dataset/poses/00.txt'

# Read the poses from the file
with open(poses_file_path, 'r') as file:
    poses_data = file.readlines()

# Convert the pose strings to 4x4 matrices
poses_matrices = [np.array(list(map(float, pose.strip().split()))).reshape(3, 4) for pose in poses_data]

print(poses_matrices[0])