import numpy as np
import cv2
class KPCalc:
    def __init__(self):
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 1
        params.maxThreshold = 20
        params.minArea = 5
        params.maxArea = 500
        params.filterByInertia = False
        params.filterByConvexity = False
        params.blobColor = 255
        detector = cv2.SimpleBlobDetector_create(params)
        detector.empty()
       
        self.kp = []
        self.params = params
        self.detector = detector
    def calcBlobs(self,im):
        self.kp = self.detector.detect(im)

        blank = np.zeros((1,1))
        
        blobs = cv2.drawKeypoints(im, self.kp, blank, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        return blobs

class GenericVideo:
    vid = None
    cap = None
    frame = None
    def process(self):
        self.cap = cv2.VideoCapture(self.vid)
        while(1):
            ret,frame = self.cap.read()
            if not ret:
                break
            self.frame = frame
            self.doAdvanced()
            cv2.imshow("frame",self.frame)
            k = cv2.waitKey(30) &0xff
            if k == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows()
    def doAdvanced(self):
        pass


class MOG1(GenericVideo):
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    kpc = KPCalc()
    def doAdvanced(self):
        fgmask = self.fgbg.apply(self.frame)
        blob_im = self.kpc.calcBlobs(fgmask)
        cv2.imshow("blob image",blob_im)
