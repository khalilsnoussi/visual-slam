import numpy as np
import cv2 as cv
from slam import Frame, SLAM
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D








#creating a list of Frame objects
frame = Frame('./data/dataset/sequences/00', img_id="000000")
frame1 = Frame('./data/dataset/sequences/00', img_id="000001")
#print(frame.width, frame.height, frame.channels)
#frame.show()
slam = SLAM(frames=[frame, frame1])
R,t = slam.feature_extractor()



#slam.draw_keypoints()



