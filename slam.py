import numpy as np
import cv2 as cv
import os
from constants import K

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




def recoverPose(E, pts1, pts2, K):
    """_summary_

    Args:
        E (np.ndarray): Essential matrix
        pts1 (np.ndarray): 2D points from the first image
        pts2 (np.ndarray): 2D points from the second image
        K (np.ndarray): Camera matrix

    Returns:
        np.ndarray: Rotation matrix
        np.ndarray: Translation vector
    """
    _, R, t, mask = cv.recoverPose(E, pts1, pts2, K)
    return R, t 




class Frame():
    def __init__(self, path : str, depth = False) -> None:
        self.path = path
        if depth:
            self.img = cv.imread(self.path, cv.IMREAD_UNCHANGED)
            
        else:
            self.img = cv.imread(self.path, cv.IMREAD_COLOR)
        
        #calculating useful informations about the frame
        if self.img.any() != None:
            self.height, self.width, self.channels = self.img.shape
        
        

    #abreging the use of boilerplate code
    def show(self) -> None:
        if self.img.any() != None:
            cv.imshow("window",self.img)
            cv.waitKey(0)
            cv.destroyAllWindows()
    

class SLAM():
    
    def __init__(self) -> None:
        """_summary_

        Args:
            sequence (str): name of the sequence from kitti that you want to use SLAM on. ie : 00, 01 or 02 etc.
        """
        self.K = K  #camera matrix
        self.frame1 = None
        self.frame2 = None   

    def feature_extractor(self ,frame1 : Frame, frame2 : Frame) -> None:
        self.frame1 = frame1
        self.frame2 = frame2
        orb = cv.ORB_create(nfeatures=3000)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        
        
        self.keypoints1, descriptors1 = orb.detectAndCompute(frame1.img, None)
        self.keypoints2, descriptors2 = orb.detectAndCompute(frame2.img, None)
        self.matches = bf.match(descriptors1, descriptors2)
        
        
            # Extract matched keypoints
        self.pts1 = np.float32([self.keypoints1[m.queryIdx].pt for m in self.matches])
        self.pts2 = np.float32([self.keypoints2[m.trainIdx].pt for m in self.matches])
        
        self.matched_kp = [self.keypoints2[m.trainIdx] for m in self.matches]
        # Find the essential matrix using RANSAC
        E, mask = cv.findEssentialMat(self.pts1, self.pts2, focal=self.K[0,0], pp=(self.K[0,2],self.K[1,2]), method=cv.RANSAC, prob=0.999, threshold=1.0)
             #returns R,t as np.ndarray, it can be computed by svd decomposition of E matrix
        R,t  = recoverPose(E, self.pts1, self.pts2, self.K[:3,:3])
        return R,t
    
    def draw_keypoints(self):
        if self.matches is not None:
            #img = cv.drawKeypoints(self.frame2.img, self.matched_kp, None, color=(0,255,0), flags=0)
            img = cv.line(self.frame2.img, np.array([self.pts1], dtype=np.int32), False,color=(0,0,255), thickness=1, lineType=16)
            
            cv.imshow("window",img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        
    


