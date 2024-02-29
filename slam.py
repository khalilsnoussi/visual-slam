import numpy as np
import cv2 as cv




class Frame():
    def __init__(self, path : str, depth = False) -> None:
        self.path = path
        if depth:
            self.img = cv.imread(self.path, cv.IMREAD_UNCHANGED)
        else:
            self.img = cv.imread(self.path)
        
        #calculating useful informations about the frame
        if self.img != None:
            self.height, self.width, self.channels = self.img.shape

    #abreging the use of boilerplate code
    def show(self) -> None:
        if self.img != None:
            cv.imshow(self.img)
            cv.waitKey(0)
            cv.destroyAllWindows()
    

class SLAM():
    
    def __init__(self, frames : list) -> None:
        """_summary_

        Args:
            frames (list): a list of Frames objects.
        """
        self.frames = self.frames
        self.frame1 = self.frames[0] #Frame object
        self.frame2 = self.frames[1] #Frame object
    
    
    
    def feature_extractor(self):
        orb = cv.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(self.frame1.img, None)
        keypoints2, descriptors2 = orb.detectAndCompute(self.frame2.img, None)
        pass
    
    def show(self):
        pass
    


