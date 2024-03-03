import numpy as np
import cv2 as cv
import os 





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
    
    def __init__(self, sequence : str  = "00") -> None:
        """_summary_

        Args:
            sequence (str): name of the sequence from kitti that you want to use SLAM on. ie : 00, 01 or 02 etc.
        """
        self.default_path = "./data/dataset/sequences/{i}/".format(i=sequence)
        
        self.frames_ids = os.listdir(self.default_path + "/image_2/")
        self.frames = []
        
        for frame_id in self.frames_ids:
            frame = Frame(self.default_path + "/image_2/" + frame_id)
            self.frames.append(frame)
            
        with open(self.default_path+"/calib.txt", 'r') as f:
            calib_data= f.readlines()
            P2_line = calib_data[2].strip().split(' ')
            P2_matrix = [float(value) for value in P2_line[1:]]  # Skip the first element (contains 'P2:')
            self.K = np.array(P2_matrix).reshape(3, 4)
    
    
    
    def feature_extractor(self) -> None:
        orb = cv.ORB_create()
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        for i in range(1,len(self.frames)):
            frame1 = self.frames[i-1]
            frame2 = self.frames[i]
            keypoints1, descriptors1 = orb.detectAndCompute(frame1.img, None)
            keypoints2, descriptors2 = orb.detectAndCompute(frame2.img, None)
            matches = bf.match(descriptors1, descriptors2)
            # Extract matched keypoints
            pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])   
            # Find the essential matrix using RANSAC
            E, mask = cv.findEssentialMat(pts1, pts2, focal=self.K[0,0], pp=(self.K[0,2],self.K[1,2]), method=cv.RANSAC, prob=0.999, threshold=1.0)
             #returns R,t as np.ndarray, it can be computed by svd decomposition of E matrix
            R,t  = recoverPose(E, pts1, pts2, self.K[:3,:3])
            print(R)
    
    def draw_keypoints(self):
        if self.matches is not None:
            img = cv.drawMatches(self.frame1.img, self.frame1.keypoints, self.frame2.img, self.frame2.keypoints, self.matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv.imshow("window",img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        
    


