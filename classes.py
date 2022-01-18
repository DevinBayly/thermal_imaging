from tqdm import tqdm
import base64
import numpy as np
import pickle
import cv2
import glob
import pandas as pd
import time
import json
import shutil
import multiprocessing as mp
import os

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
        ##print(self.kp)
        
        #cv2.imshow("blobs",blobs)
        return self.kp



class GenericVideo:
    def __init__ (self,vid,frames_to_process =0,id_val = 0):
      self.vid = vid
      self.id_val = id_val
      self.cap = None
      self.frame = None
      overly_verbose_name = f"{'_'.join(self.vid.split('.')[:-1])}_{time.time()}"
      ## stripping out the non essential characters
      self.output_name = ''.join([c if c.isalnum() else '_' for c in overly_verbose_name]) +".json"  
      self.frames_to_process = frames_to_process
      self.cap = cv2.VideoCapture(self.vid)
      self.totalFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
      self.frames = 0
      self.videosRecordedFPS = int(self.cap.get(cv2.CAP_PROP_FPS))
      ##decide on limit to the processing
      if self.frames_to_process > 0:
        ## this means we are breaking the video up to process faster
        self.starting_frame = self.frames_to_process*self.id_val
      else:
        ## means we are starting from the beginning
        self.starting_frame = 0
        self.frames_to_process= self.totalFrames
      self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.starting_frame)
    def process(self):
      empty = np.zeros((200,200))
      #cv2.imshow("processing",empty)
      #time.sleep(10)
      #print("proceeding now")
      ## get the total number of frames to make estimate
        
      print(self.frames_to_process)
      for i in tqdm(range(self.frames_to_process)):
          tstart= time.time()
          ret,frame = self.cap.read()
          self.frames+=1
          if not ret:
              break
          self.frame = frame
          self.tstamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)
          self.doSteps()
          ##cv2.imshow("frame",self.frame)
          k = cv2.waitKey(30) &0xff
          if k == 27:
              break
          ##print("seconds passed",time.time()-tstart)
      self.cap.release()
      cv2.destroyAllWindows()
      ## handle exporting
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
      return
      with open(self.output_name,"wb") as phile:
        ##print(self.out_binary)
        phile.write(self.out_binary)


class SimplestPass(GenericVideo):
    def __init__(self,vid,centroid_threshold,time_limit,frames_to_process =0,id_val = 0):
      super().__init__(vid,frames_to_process ,id_val )
      print("simplest vid ",vid)
      self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
      self.kpc = KPCalc()
      self.tracker = TrackerCentroids(centroid_threshold,time_limit)
      self.frame_count = 0
    def doBlobAnalysis(self):
      #calculate the newest detection points
      self.kpc.calcBlobs(self.fgmask)

      ret,thresh = cv2.threshold(self.fgmask,125,255,0)
      contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      ##
      cv2.drawContours(self.frame,contours,-1,(0,255,0),1)
      bounding = []
      ## what is the matching that we can be doing between the detected centroids?
      ## have to test for "contains" like whether the centroid fits in the bounds of the box
      ## alternative is to calculate the centroid from the contour and then we would know what the correspondence is. 
      ## set the area limit to be 

      for i,c in enumerate(contours):
          contour_poly = cv2.approxPolyDP(c, 3, True)
          bounds =  cv2.boundingRect(contour_poly)
          ## calculate the area
          if bounds[2]> 2 and bounds[3] > 2:
              cv2.rectangle(self.frame,(bounds[0],bounds[1]),(bounds[0]+bounds[2],bounds[1]+bounds[3]),(255,0,0),1)
      blank = np.zeros((1,1))
      empty =np.zeros(self.frame.shape).astype("uint8")
      centroids = cv2.drawKeypoints(empty, self.kpc.kp, blank, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
      self.frame += centroids
      #cv2.imshow("processing",self.frame)
      if len(self.kpc.kp) > 0:
        pass
        #cv2.imwrite(f"images/{self.frame_count}.png",self.fgmask)
      # make blobs into list of detections
    def doBGSeg(self):
      #print(self.frame_count)
      self.frame_count+=1 
      self.fgmask = self.fgbg.apply(self.frame)
      ##cv2.imshow("blob image",self.fgmask)
    def doTrack(self):
      #pass in the new detections which are on the self.kpc this stands for keypoints calc I think
      #make them into the correct tracked objects
      detections = []
      for kp in self.kpc.kp:
        centroid_object = CentroidObject(np.array([kp.pt[0],kp.pt[1]]),self.tstamp)
        ##print(centroid_object)
        detections.append(centroid_object)
      self.tracker.compute(detections)
      ## perform a trace of the existing paths so far
      trackim = np.zeros_like(self.frame).astype("uint8")
      for t in self.tracker.tracks:
        linepts = []
        for d in t.detections:
          linepts.append(d.center)
        ## draw the line of the image now
        ###print(linepts)
        ## maybe associate track with color also
        ###print(t.color)
        ###print(linepts)
        cv2.polylines(trackim,[np.array(linepts,np.int32)],False,t.color)
      #cv2.imshow("tracks",trackim)

    def export(self):
        ## make a dataframe out of the results from the tracker
        tracker_res = self.tracker.export()
        df = pd.DataFrame(tracker_res)
        ## save the file to the local disk
        df.to_csv("data.csv",index_label="index")

    def getAllData(self):
      ## 
      ##output = []
      ##for track in self.tracker.tracks:
      ##  trackList = []
      ##  for d in track:
      ##    trackList.append({"x":d.center[0],"y":d.center[1]})
      ##  output.append(trackList)
      ##self.serialized = json.dumps(output)
      ##self.out_binary = self.serialized.encode("utf-8")
      pass

  

## TODO write something that takes certain tracks out of the listing when their time is up

## this will only perform background segmentation and 
class OnlyDetect(SimplestPass):
  def __init__(self,vid,centroid_threshold,time_limit,frames_to_process=0,id_val=0):
    super().__init__(vid,centroid_threshold,time_limit,frames_to_process ,id_val )
    self.tstamp_logger = []
    self.export_timer = time.time()
    self.centroid_threshold = centroid_threshold
  ## overload the doSteps method from the parent()
  def doSteps(self):
    ##leave out the tracking
    self.doBGSeg()
    self.doBlobAnalysis()
    if time.time() - self.export_timer > 5*60:
      print("exporting checkpoint")
      self.export_timer = time.time()
      self.export() 
  def doBlobAnalysis(self):
    self.kpc.calcBlobs(self.fgmask)
    ## now check how many detections were made and export number of ticks
    num_detections = len(self.kpc.kp)
    blank = np.zeros((1,1))
    empty =np.zeros(self.frame.shape).astype("uint8")
    self.centroids = cv2.drawKeypoints(empty, self.kpc.kp, blank, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    if  num_detections> 0 :
      self.tstamp_logger.append({"tstamp_ms":self.tstamp,"tstamp": conv_ms_tstamp_string(self.tstamp),"number":num_detections,"detections":[{"x":d.pt[0],"y":d.pt[1]} for d in self.kpc.kp]})
  @staticmethod
  def make_only_for_parallel(vid,centroid_threshold,time_limit,frames_to_process,id_val):
    print("values are",vid,centroid_threshold,time_limit,frames_to_process,id_val)
    process_video_copy = f"/tmp/thermal_video{id_val}.mp4"
    shutil.copy("/tmp/thermal_video.mp4",process_video_copy)
    only = OnlyDetect(process_video_copy,centroid_threshold,time_limit,frames_to_process,id_val)
    
    only.process()
  def export(self):
    ## convert the tstamp_logger list into a json list that can be uploaded
    fname = "thermal_logger"+f"{self.id_val:02d}"+self.output_name
    with open(fname,"w") as phile:
      phile.write(json.dumps(self.tstamp_logger))
    ## export a snapshot to use in background of logger viewing
    png_name = "thermal_logger_img"+f"{self.id_val:02d}.png"
    try:
      cv2.imwrite(png_name,self.frame)

    except Exception as e:
      print(e)


def test_parallel_video(passclass,ifolder,ofolder):
  vids =process_folders(ifolder,ofolder)
  starting_directory= os.getcwd()
  threshold = 20
  time_limit = 4000 # in milliseconds
  for vid in vids:
    #vid ="mine-4_rockfall_clips/Camera 2 - 192.168.0.121 (FLIRFC-632-ID-22947C)-20210526-234803.mp4"
    shutil.copy(vid,"/tmp/thermal_video.mp4")
    os.chdir("/tmp")
    num_cpus = 3
    _tempcap = cv2.VideoCapture("thermal_video.mp4")
    total_frames = _tempcap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("total frames" ,total_frames)
    frames_per_cpu= int(total_frames/num_cpus) + 1
    ## leverage multiprocessing now
    processes = []
    for id_val in range(num_cpus):
      ## make a collection of only detectors and start them all up
      process = mp.Process(target=OnlyDetect.make_only_for_parallel,args = ("thermal_video.mp4",threshold,time_limit,frames_per_cpu,id_val,))
      processes.append(process)
    print("now starting the processes")  
    #start each processor
    for p in processes:
      p.start()
    print("and now joining")
    #await their endings one by one
    for p in processes:
      p.join()
    
    ## should try to combine the videos now
    os.chdir(starting_directory)
    os.chdir(ofolder)
    combine("/tmp",vid.split("/")[-1])
    os.chdir(starting_directory)

  
# goal have a folder that you can export to, and the contents of this are compared to the inputs folder so that as things finish over time you know that you aren't re-runnign things
def process_folders(in_folder,out_folder):
    ## get all the mp4s in the in folder and the jsons in the out folder
    ## look for the names of the mp4s in the jsons
  ## look for a file in the out folder called finished which is just a list of the file names 
  mp4s = glob.glob(in_folder + "/*mp4",recursive = True)
  jsons = glob.glob(out_folder + "/*json",recursive = True)
  videos_to_process =[]
  ## iterate over the jsons and if they are in the mp4s then remove that entity from the mp4s.
  for j in jsons:
    #here's the part of the json name that uses the mp4 title
    start = j.find("Camera")
    end = j.find(".json")
    jvideo_name = j[start:end]
    for mp4 in mp4s:
      if jvideo_name == mp4.split("/")[-1]:
        print("found completed",mp4,j)
        mp4s.remove(mp4)
        break
  # send this to a function that is looking for videos to process
  return mp4s


