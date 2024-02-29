import numpy as np
import cv2 as cv




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
    def __init__(self, path : str,img_id : int, depth = False) -> None:
        self.path = path
        self.img_id = img_id 
        if depth:
            self.img = cv.imread(self.path+ "/image_2/{id}.png".format(id=self.img_id), cv.IMREAD_UNCHANGED)
        else:
            self.img = cv.imread(self.path+ "/image_2/{id}.png".format(id=self.img_id))
        
        #calculating useful informations about the frame
        if self.img.any() != None:
            self.height, self.width, self.channels = self.img.shape
        
        with open(self.path+"/calib.txt", 'r') as f:
            calib_data= f.readlines()
            P2_line = calib_data[2].strip().split(' ')
            P2_matrix = [float(value) for value in P2_line[1:]]  # Skip the first element (contains 'P2:')
            self.K = np.array(P2_matrix).reshape(3, 4)

    #abreging the use of boilerplate code
    def show(self) -> None:
        if self.img.any() != None:
            cv.imshow("window",self.img)
            cv.waitKey(0)
            cv.destroyAllWindows()
    

class SLAM():
    
    def __init__(self, frames : list) -> None:
        """_summary_

        Args:
            frames (list): a list of Frames objects.
        """
        self.frames = frames
        self.frame1 = self.frames[0] #Frame object
        self.frame2 = self.frames[1] #Frame object
    
    
    
    def feature_extractor(self) -> np.ndarray:
        orb = cv.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(self.frame1.img, None)
        keypoints2, descriptors2 = orb.detectAndCompute(self.frame2.img, None)
        self.frame1.keypoints = keypoints1
        self.frame2.keypoints = keypoints2
        
        #brute force matching
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        self.matches = sorted(matches, key = lambda x:x.distance)
        
        
        # Extract matched keypoints
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
        
        
        # Find the essential matrix using RANSAC
        E, mask = cv.findEssentialMat(pts1, pts2, focal=self.frame1.K[0,0], pp=(self.frame1.K[0,2],self.frame1.K[1,2]), method=cv.RANSAC, prob=0.999, threshold=1.0)
        return recoverPose(E, pts1, pts2, self.frame1.K[:3,:3])
    
    def draw_keypoints(self):
        if self.matches is not None:
            img = cv.drawMatches(self.frame1.img, self.frame1.keypoints, self.frame2.img, self.frame2.keypoints, self.matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv.imshow("window",img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        
    


