import numpy as np
import cv2
import glob
import pandas as pd
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
##            #cv2.imshow("frame",self.frame)
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
##        #cv2.imshow("blob image",blob_im)
##
##

class BaseVideo:
  def __init__(self,bgm_type,blob_type,tracker_type):
    self.bgm = bgm_type
    self.blob= blob_type
    self.tracker= tracker_type

class KPCalc:
    def __init__(self):
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 1
        params.maxThreshold = 20
        params.minArea = 1
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
        #print(self.kp)
        blank = np.zeros((1,1))
        
        blobs = cv2.drawKeypoints(im, self.kp, blank, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        #cv2.imshow("blobs",blobs)
        return self.kp

class GenericVideo:
    def __init__ (self,vid):
      self.vid = vid
      self.cap = None
      self.frame = None
      overly_verbose_name = f"{'_'.join(self.vid.split('.')[:-1])}_{time.time()}"
      ## stripping out the non essential characters
      self.output_name = ''.join([c if c.isalnum() else '_' for c in overly_verbose_name]) +".json"  
    def process(self):
      self.cap = cv2.VideoCapture(self.vid)
      while(1):
          ret,frame = self.cap.read()
          if not ret:
              break
          self.frame = frame
          self.doSteps()
          ##cv2.imshow("frame",self.frame)
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
        #print(self.out_binary)
        phile.write(self.out_binary)

class SimplestPass(GenericVideo):
    def __init__(self,vid,centroid_threshold):
      super().__init__(vid)
      self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
      self.kpc = KPCalc()
      self.tracker = TrackSetCentroids(centroid_threshold)
      self.frame_count = 0
    def doBlobAnalysis(self):
      #calculate the newest detection points
      self.kpc.calcBlobs(self.fgmask)
      if len(self.kpc.kp) > 0:
        cv2.imwrite(f"images/{self.frame_count}.png",self.fgmask)
      # make blobs into list of detections

    def doBGSeg(self):
      self.frame_count+=1 
      self.fgmask = self.fgbg.apply(self.frame)
      ##cv2.imshow("blob image",self.fgmask)
    def doTrack(self):
      #pass in the new detections which are on the self.kpc this stands for keypoints calc I think
      #make them into the correct tracked objects
      detections = []
      for kp in self.kpc.kp:
        centroid_object = CentroidObject(np.array([kp.pt[0],kp.pt[1]]))
        #print(centroid_object)
        detections.append(centroid_object)
      self.tracker.compute(detections)
    def getAllData(self):
      ## 
      output = []
      for track in self.tracker.tracks:
        trackList = []
        for d in track:
          trackList.append({"x":d.center[0],"y":d.center[1]})
        output.append(trackList)
      self.serialized = json.dumps(output)
      self.out_binary = self.serialized.encode("utf-8")


class ObjectEncoder(json.JSONEncoder):
  def default(self,o):
    return o.__dict__


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
  def __init__(self,threshold):
    super().__init__()
    self.threshold = threshold
  def compute(self,new_detections):
    ## if tracks aren't established yet just add each detection as a track 
    ## go over list of tracks
    for t in self.tracks:
      for d in new_detections:
        ##compare to the very last of the elements we are tracking
        if d.compare(t[-1],self.threshold):
          t.append(d.copy())
    ## go through the detections and add the ones with no successful comparisons to the tracks as starting points

    for d in new_detections:
      if d.count == 0:
        self.tracks.append([d])



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
  def __repr__(self):
    return f"{self.center[0]} {self.center[1]} count {self.count}"

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
def test_video():
  threshold = 10
  videos = glob.glob("mine*/*mp4")
  vid ="mine-4_rockfall_clips/Camera 1 - 192.168.0.105 (FLIR A400) - 14-20210528-225029.mp4"
  simple_process = SimplestPass(vid,threshold)
  
  simple_process.process()

def test_track():
  ## make an image use kpc on thing and then try to pass the values to a tracker
  threshold = 10
  tracker= TrackSetCentroids(threshold)
  kpc = KPCalc()
  images = glob.glob("images/*png")
  im = cv2.imread(images[int(len(images)/2)])
  #cv2.imshow("bin image",im)
  kpc.calcBlobs(im)
  detections = []
  for kp in kpc.kp:
    centroid_object = CentroidObject(np.array([kp.pt[0],kp.pt[1]]))
    #print(centroid_object)
    detections.append(centroid_object)

  #print(detections)
  tracker.compute(detections)
  #print(tracker.tracks)

def timer(f):
  start = time.time()
  f()
  print("seconds passed",time.time() - start)


timer(test_video)
