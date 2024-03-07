import numpy as np
import cv2 as cv
from slam import Frame, SLAM
import os

myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])


slam = SLAM()


default_path = './data/dataset/sequences/00/image_2/'
frames_id = os.listdir(default_path)
for i in range(len(frames_id)-1):
    path  = default_path + frames_id[i]
    path2 = default_path + frames_id[i+1]
    frame1 = Frame(path)
    frame2 = Frame(path2)
    if True:
        slam.feature_extractor(frame1=frame1, frame2=frame2)
        slam.draw_keypoints()
        #print(R)
    ##frame.show()


#creating a list of Frame objects
#slam= SLAM()
#slam.feature_extractor()


#print(frame.width, frame.height, frame.channels)
#frame.show()
#slam = SLAM(frames=[frame, frame1])
#R,t = slam.feature_extractor()
#print(R)


#slam.draw_keypoints()



