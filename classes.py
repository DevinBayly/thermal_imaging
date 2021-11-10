import numpy as np
import glob
import pandas as pd
import cv2
import time
import json
##
##class GenericVideo:
##    vid = None
##    cap = None
##    frame = None
##    def process(self):
##        self.cap = cv2.VideoCapture(self.vid)
##        while(1):
##            ret,frame = self.cap.read()
##            if not ret:
##                break
##            self.frame = frame
##            self.doAdvanced()
##            cv2.imshow("frame",self.frame)
##            k = cv2.waitKey(30) &0xff
##            if k == 27:
##                break
##        self.cap.release()
##        cv2.destroyAllWindows()
##    def doAdvanced(self):
##        pass
##
##
##class MOG1(GenericVideo):
##    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
##    kpc = KPCalc()
##    def doAdvanced(self):
##        fgmask = self.fgbg.apply(self.frame)
##        blob_im = self.kpc.calcBlobs(fgmask)
##        cv2.imshow("blob image",blob_im)
##
##
class BaseVideo:
  def __init__(self,bgm_type,blob_type,tracker_type):
    self.bgm = bgm_type
    self.blob= blob_type
    self.tracker= tracker_type

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
        
        #blobs = cv2.drawKeypoints(im, self.kp, blank, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        return self.kp

class GenericVideo:
    def __init__ (self,vid):
      self.vid = vid
      self.cap = None
      self.frame = None
      self.output_name = f"{'_'.join(self.vid.split('.')[:-1])}_{time.time()}.json"
    def process(self):
      self.cap = cv2.VideoCapture(self.vid)
      while(1):
          ret,frame = self.cap.read()
          if not ret:
              break
          self.frame = frame
          self.doSteps()
          #cv2.imshow("frame",self.frame)
          k = cv2.waitKey(30) &0xff
          if k == 27:
              break
      self.cap.release()
      cv2.destroyAllWindows()
      ## handle exporting
      self.getAllData()
      self.export()
    def doSteps(self):
      self.doBGSeg()
      self.doBlobAnalysis()
      self.doTrack()
    def doBGSeg(self):
      pass
    def doBlobAnalysis(self):
      pass
    def doTrack(self):
      pass
    def getAllData(self):
      pass
    ## this would work for a spreadsheet but maybe not the easiest format for testing outputs
    def export(self):
      with open(self.output_name,"wb") as phile:
        print(self.out_binary)
        phile.write(self.out_binary)

class SimplestPass(GenericVideo):
    def __init__(self,vid):
      super().__init__(vid)
      self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
      self.kpc = KPCalc()
      self.tracker = TrackSetCentroids()
    def doBlobAnalysis(self):
      #calculate the newest detection points
      self.kpc.calcBlobs(self.fgmask)
      # make blobs into list of detections

    def doBGSeg(self):
      self.fgmask = self.fgbg.apply(self.frame)
      #cv2.imshow("blob image",blob_im)
    def doTrack(self):
      #pass in the new detections which are on the self.kpc this stands for keypoints calc I think
      #make them into the correct tracked objects
      detections = []
      for kp in self.kpc.kp:
        centroid_object = CentroidObject(np.array([kp.pt[0],kp.pt[1]]))
      self.tracker.compute(detections)
    def getAllData(self):
      ## 
      self.serialized = json.dumps(self.tracker.tracks)
      self.out_binary = self.serialized.encode("utf-8")






class TrackSetBase:
  def __init__(self):
    self.tracks =[]
  def compute(self,new_detections):
    #TB overridden
    pass
#not implemented  
class TrackSetBB(TrackSetBase):
  def __init__(self):
    super().__init__()
  def compute(self,new_detections):
    pass

class TrackSetCentroids(TrackSetBase):
  def __init__(self):
    super().__init__()
  def compute(self,new_detections):
    ## go over list of tracks
    for t in self.tracks:
      for d in new_detections:
        ##compare to the very last of the elements we are tracking
        if d.compare(t[-1]):
          t.append(d.copy())

    ## for each compare to the point in the new detections

## this object is on frame's detection, multiple of these make up a set to track
class TrackObjectBase:
  def __init__(self,center):
    self.center = center
    self.count=0
  def compare(self,other,threshold):
    pass
  def copy(self):
    pass


class CentroidObject(TrackObjectBase):
  def __init__(self,center):
    super().__init__(center)
  def compare(self,other,threshold):
    # I guess threshold might be many different things to different objects 
    ## numpy method for tackling this is
    dist = np.linalg.norm(self.center - other.center)
    result = dist < threshold
    if result:
      self.count +=1
    return result
  def copy(self):
    copy = CentroidObject(self.center)
    copy.count = self.count
    return copy

# testing classes
videos = glob.glob("mine*/*mp4")
vid = videos[0]
simple_process = SimplestPass(vid)
simple_process.process()
