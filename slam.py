import numpy as np
import cv2 as cv
import os
from constants import K
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform
#from helpers import EssentialMatrixTransform






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


RANSAC_RESIDUAL_THRES = 0.02
RANSAC_MAX_TRIALS = 100



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
        orb = cv.ORB_create()
        pts_frame1 = cv.goodFeaturesToTrack(np.mean(frame1.img, axis = 2).astype(np.uint8), maxCorners=1000, qualityLevel=0.01, minDistance=7)
        pts_frame2 = cv.goodFeaturesToTrack(np.mean(frame2.img, axis = 2).astype(np.uint8), maxCorners=1000, qualityLevel=0.01, minDistance=7)
        
        kps1 = [cv.KeyPoint(x = pt[0][0], y = pt[0][1], size = 20) for pt in pts_frame1]
        kps2 = [cv.KeyPoint(x = pt[0][0], y = pt[0][1], size = 20) for pt in pts_frame2]
        kps1, desc1 = orb.compute(frame1.img, kps1)
        kps2, desc2 = orb.compute(frame2.img, kps2)
        
        bf = cv.BFMatcher(cv.NORM_HAMMING)
        self.matches = bf.knnMatch(desc1, desc2, k = 2)
        ret = []
        idx1, idx2 = [], []
        idx1s, idx2s = set(), set()
        
        for m,n in self.matches:
            if m.distance < 0.75*n.distance:
                p1 = kps1[m.queryIdx].pt
                p2 = kps2[m.trainIdx].pt
                
                if m.distance < 32:
                    if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
                        idx1.append(m.queryIdx)
                        idx2.append(m.trainIdx)
                        idx1s.add(m.queryIdx)
                        idx2s.add(m.trainIdx)
                        ret.append((p1, p2))
                
        
                       
         # no duplicates
        assert(len(set(idx1)) == len(idx1))
        assert(len(set(idx2)) == len(idx2))
        
       

        
        ret = np.array(ret)
        assert len(ret) >= 8
        idx1 = np.array(idx1)
        idx2 = np.array(idx2)
        self.ret = ret
        
        
        
       
            
        E, inliers = cv.findEssentialMat(ret[:,0], ret[:,1], focal=self.K[0,0], pp=(self.K[0,2],self.K[1,2]), method=cv.RANSAC, prob=0.999, threshold=1.0)
        #returns R,t as np.ndarray, it can be computed by svd decomposition of E matrix
        R,t  = recoverPose(E, ret[:,0], ret[:,1], self.K[:3,:3])
        self.inliers = inliers
        
        
    def draw_keypoints(self):
        inlier_pts1 = self.ret[:,0][self.inliers.ravel() == 1] 
        inlier_pts2 = self.ret[:,1][self.inliers.ravel() == 1]
        
        
        
        for i in range(len(inlier_pts2)):
            
            #cv.line(self.frame2.img, (int(inlier_pts1[i][0]), int(inlier_pts1[i][1])), (int(inlier_pts2[i][0]), int(inlier_pts2[i][1])), (0, 255, 0), 2)
            print(len(inlier_pts2))
            cv.polylines(self.frame2.img, np.array([np.int32([inlier_pts1[i], inlier_pts2[i]])]), False, (0, 255, 0), 1)
            cv.imshow('matches', self.frame2.img)
            cv.waitKey(1)
        
    


