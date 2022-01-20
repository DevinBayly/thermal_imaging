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
  """
  A class representing the keypoint calculator. This is using the SimpleBlobDetector from OpenCV.
  The parameters used here are from the default listed in https://learnopencv.com/blob-detection-using-opencv-python-c/

  Attributes:

  kp: list
     this is a list of the keypoints that we get back when we run the detector.detect(im) line
  detector: SimpleBlobDetector
     this is a opencv blob detector built from the parameters listed
  params: SimpleBlobDetector_Params
     the parameters that control whether a blob in the image is considered a blob to get the keypoint of.

  Methods:

  init(self)
    this just generates the blob detector from the parameters and sets the instance attribute variables
  calcBlobs(self,im)
    the image is a black and white frame from the video that we get after the background segmentation has completed
    when the self.detector is run we get back a list of keypoint objects. They are defined by https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html
    and we can access their pt property to get back x,y coordinates in screen space for what the centroid of the blob was.
  """

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

  def calcBlobs(self, im):
      self.kp = self.detector.detect(im)
      return self.kp


class GenericVideo:
  """
  This class represents the base of a video to be read and processed. Inheriting classes will specialize functions related to processing the frames that are within the video. This class has been configured to allow for parallel processing where the user specifies that only certain frames are to be processed, and then the instance gets an id to uniquely identify the output files. This allows for the outputs from different cpu cores to be aggregated back together later (see the combine function from this file). 
  
  Attributes:
  
  vid: string
    this is the file name of the video that we will process
  id_val : int 
    this is the id associated with the processing job, uniquely identifies the class that is processing a part of a much longer video
  cap : opencv capture 
    this is the object that holds a represnetation of the video that we can read frames from using Open Cv
  frame : opencv frame/ numpy array
    this is a single frame of the video, many opencv methods can work on this and the representation works for certain numpy operations as well
  output_name : string 
    this is the name of the file that we will eventually write to at the end of the processing
  frames_to_process: int
    when a larger video is broken into sections that are handled by multiple generic video processes we will establish a number of frames that the videos need to go through. This is that number. It almost always is calculated as (total video frames)/(number of processing cores)
  frames: int

  """
  def __init__ (self, vid, frames_to_process =0,id_val = 0):
      self.vid = vid
      self.id_val = id_val
      self.cap = None
      self.frame = None
      print(vid)
      overly_verbose_name = f"{'_'.join(self.vid.split('.')[:-1])}_{time.time()}"
      # stripping out the non essential characters
      self.output_name = ''.join([c if c.isalnum() else '_' for c in overly_verbose_name]) + ".json"
      self.frames_to_process = frames_to_process
      self.cap = cv2.VideoCapture(self.vid)
      self.totalFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
      self.frames = 0
      self.videosRecordedFPS = int(self.cap.get(cv2.CAP_PROP_FPS))
      # decide on limit to the processing
      if self.frames_to_process > 0:
          # this means we are breaking the video up to process faster
          self.starting_frame = self.frames_to_process*self.id_val
      else:
          # means we are starting from the beginning
          self.starting_frame = 0
          self.frames_to_process = self.totalFrames
      # this skips ahead to the starting frame that is specified in the cap
      self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.starting_frame)

  def process(self):
      """
      this function starts at the beginning frame and continues from there calling the doSteps method at the appropriate time in the processing. It also handles the proper shutdown steps after opening the video file and creating a resource from it. It also handles the export step is there is a method defined to perform export at the end of the process"""
      empty = np.zeros((200, 200))

      print(self.frames_to_process)
      for i in tqdm(range(self.frames_to_process)):
          tstart = time.time()
          ret, frame = self.cap.read()
          self.frames += 1
          if not ret:
              break
          self.frame = frame
          self.tstamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)
          self.doSteps()
          # cv2.imshow("frame",self.frame)
          k = cv2.waitKey(30) & 0xff
          if k == 27:
              break
          ##print("seconds passed",time.time()-tstart)
      self.cap.release()
      cv2.destroyAllWindows()
      # handle exporting
      self.export()

  def doSteps(self):
      """These are the step methods that comprise a tracking algorithm. first background segmentation and then blobl analysis and then tracking if that process is included in the algorithm. """
      self.doBGSeg()
      self.doBlobAnalysis()
      self.doTrack()

  def doBGSeg(self):
      """this is the background segmentation step. We leave it undefined at the Generic Video level but classes that inherit can override this. """
      pass

  def doBlobAnalysis(self):
      """this is the blob analysis method. It will be implemented in the classes that inherit from the generic video class"""
      pass

  def doTrack(self):
      """This is the Tracking method. Inheriting classes will specialize this method to perform variations of the tracking if it is included."""
      pass

  def export(self):
      """This is the export method. Inheriting classes will make this handle their specific export steps."""
      return

class SimplestPass(GenericVideo):
  def __init__(self, vid, centroid_threshold,time_limit,frames_to_process =0,id_val = 0):
      super().__init__(vid, frames_to_process , id_val )
      print("simplest vid ", vid)
      self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
      self.kpc = KPCalc()
      self.tracker = None
      self.frame_count = 0

  def doBlobAnalysis(self):
      # calculate the newest detection points
      self.kpc.calcBlobs(self.fgmask)

      ret, thresh = cv2.threshold(self.fgmask, 125,255,0)
      contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      ##
      cv2.drawContours(self.frame, contours, -1,(0,255,0),1)
      bounding = []
      # what is the matching that we can be doing between the detected centroids?
      # have to test for "contains" like whether the centroid fits in the bounds of the box
      # alternative is to calculate the centroid from the contour and then we would know what the correspondence is.
      # set the area limit to be

      for i, c in enumerate(contours):
          contour_poly = cv2.approxPolyDP(c, 3, True)
          bounds = cv2.boundingRect(contour_poly)
          # calculate the area
          if bounds[2] > 2 and bounds[3] > 2:
              cv2.rectangle(self.frame, (bounds[0], bounds[1]),(bounds[0]+bounds[2],bounds[1]+bounds[3]),(255,0,0),1)
      blank = np.zeros((1, 1))
      empty = np.zeros(self.frame.shape).astype("uint8")
      centroids = cv2.drawKeypoints(empty, self.kpc.kp, blank, (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
      self.frame += centroids
      # cv2.imshow("processing",self.frame)
      if len(self.kpc.kp) > 0:
          pass
          # cv2.imwrite(f"images/{self.frame_count}.png",self.fgmask)
      # make blobs into list of detections

  def doBGSeg(self):
      # print(self.frame_count)
      self.frame_count += 1
      self.fgmask = self.fgbg.apply(self.frame)
      ##cv2.imshow("blob image",self.fgmask)

  def doTrack(self):
      # pass in the new detections which are on the self.kpc this stands for keypoints calc I think
      # make them into the correct tracked objects
      detections = []
      for kp in self.kpc.kp:
          centroid_object = CentroidObject(np.array([kp.pt[0], kp.pt[1]]), self.tstamp)
          # print(centroid_object)
          detections.append(centroid_object)
      self.tracker.compute(detections)
      # perform a trace of the existing paths so far
      trackim = np.zeros_like(self.frame).astype("uint8")
      for t in self.tracker.tracks:
          linepts = []
          for d in t.detections:
              linepts.append(d.center)
          # draw the line of the image now
          # print(linepts)
          # maybe associate track with color also
          # print(t.color)
          # print(linepts)
          cv2.polylines(trackim, [np.array(linepts, np.int32)],False,t.color)
      # cv2.imshow("tracks",trackim)

  def export(self):
        # make a dataframe out of the results from the tracker
      tracker_res = self.tracker.export()
      df = pd.DataFrame(tracker_res)
      # save the file to the local disk
      df.to_csv("data.csv", index_label="index")

  

# TODO write something that takes certain tracks out of the listing when their time is up

# this will only perform background segmentation and
class OnlyDetect(SimplestPass):
  def __init__(self, vid, centroid_threshold,time_limit,frames_to_process=0,id_val=0):
      super().__init__(vid, centroid_threshold, time_limit,frames_to_process ,id_val )
      self.tstamp_logger = []
      self.export_timer = time.time()
      self.centroid_threshold = centroid_threshold
  # overload the doSteps method from the parent()

  def doSteps(self):
      # leave out the tracking
      self.doBGSeg()
      self.doBlobAnalysis()
      if time.time() - self.export_timer > 5*60:
          print("exporting checkpoint")
          self.export_timer = time.time()
          self.export()

  def doBlobAnalysis(self):
      self.kpc.calcBlobs(self.fgmask)
      # now check how many detections were made and export number of ticks
      num_detections = len(self.kpc.kp)
      blank = np.zeros((1, 1))
      empty = np.zeros(self.frame.shape).astype("uint8")
      self.centroids = cv2.drawKeypoints(empty, self.kpc.kp, blank, (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
      if  num_detections > 0:
          self.tstamp_logger.append({"tstamp_ms": self.tstamp, "tstamp": conv_ms_tstamp_string(self.tstamp),"number":num_detections,"detections":[{"x":d.pt[0],"y":d.pt[1]} for d in self.kpc.kp]})

  @staticmethod
  def make_only_for_parallel(vid, centroid_threshold, time_limit,frames_to_process,id_val):
      print("values are", vid, centroid_threshold,time_limit,frames_to_process,id_val)
      only = OnlyDetect(vid, centroid_threshold, time_limit,frames_to_process,id_val)
      only.process()

  def export(self):
      # convert the tstamp_logger list into a json list that can be uploaded
      # self.vid is something like thermal_video_0.mp4_0
      fname =  self.vid+f"_{self.id_val:02d}.json"
      with open(fname, "w") as phile:
          phile.write(json.dumps(self.tstamp_logger))
      # export a snapshot to use in background of logger viewing
      png_name = self.vid+f"_{self.id_val:02d}.png"
      try:
          cv2.imwrite(png_name, self.frame)

      except Exception as e:
          print(e)


def test_parallel_video(passclass, ifolder, ofolder,video_number):
  print(ifolder,ofolder)
  videos= process_folders(ifolder, ofolder)
  vid = videos[int(video_number)]
  starting_directory = os.getcwd()
  threshold = 20
  time_limit = 4000  # in milliseconds
  shutil.copy(vid, f"/tmp/thermal_video_{video_number}.mp4")
  os.chdir("/tmp")
  num_cpus = 15
  _tempcap = cv2.VideoCapture(f"thermal_video_{video_number}.mp4")
  total_frames = _tempcap.get(cv2.CAP_PROP_FRAME_COUNT)
  print("total frames", total_frames)
  frames_per_cpu = int(total_frames/num_cpus) + 1
  # leverage multiprocessing now
  processes = []
  for id_val in range(num_cpus):
      # make a collection of only detectors and start them all up
      process = mp.Process(target=OnlyDetect.make_only_for_parallel, args = (f"thermal_video_{video_number}.mp4", threshold,time_limit,frames_per_cpu,id_val,))
      processes.append(process)
  print("now starting the processes")
  # start each processor
  for p in processes:
      p.start()
  print("and now joining")
  # await their endings one by one
  for p in processes:
      p.join()

  # should try to combine the videos now
  os.chdir(starting_directory)
  os.chdir(ofolder)
  combine("/tmp", vid.split("/")[-1],f"thermal_video_{video_number}.mp4")
  os.chdir(starting_directory)


def combine(pth,name,vid):
  logs = glob.glob(pth+"/" + vid + "*json")
  all_data =[]
  ## get all the data
  for log in logs:
    with open(log,"r") as phile:
      all_data.extend(json.loads(phile.read()))
  
##get all the backgroundss
  imgs = glob.glob(pth+"/" + vid + "*png")
  all_imgs =[]
  for img in imgs:
    with open(img,"rb") as phile:
      b64_text = base64.b64encode(phile.read()).decode("utf-8")
      all_imgs.append(b64_text)
  all_log = dict(data=all_data,background_images=all_imgs)
  with open(f"all_logs_{name}.json","w") as phile:
    phile.write(json.dumps(all_log)) 
  ## now remove the logs
  for el in imgs+logs:
    try:
      os.remove(el)
    except Exception as e:
      print("error, might have been someone elses log file?",e)
      
def conv_ms_tstamp_string(ms):
  t_s = int(ms/1000)%60
  t_mins = int(ms/(1000*60))%60
  t_hours = int(ms/(1000*60*60))%60
  return f"{t_hours:02d}:{t_mins:02d}:{t_s:02d}"
    
# goal have a folder that you can export to, and the contents of this are compared to the inputs folder so that as things finish over time you know that you aren't re-runnign things
def process_folders(in_folder, out_folder):
    # get all the mp4s in the in folder and the jsons in the out folder
    # look for the names of the mp4s in the jsons
  # look for a file in the out folder called finished which is just a list of the file names
  mp4s = glob.glob(in_folder + "/**/*mp4", recursive= True)
  jsons = glob.glob(out_folder + "/**/*json", recursive= True)
  # iterate over the jsons and if they are in the mp4s then remove that entity from the mp4s.
  print(len(mp4s),len(jsons))
  for j in jsons:
      # here's the part of the json name that uses the mp4 title
      start = j.find("Camera")
      end = j.find(".json")
      jvideo_name = j[start:end]
      for mp4 in mp4s:
          if jvideo_name == mp4.split("/")[-1]:
              print("found completed", mp4, j)
              mp4s.remove(mp4)
              break
  # send this to a function that is looking for videos to process
  print(mp4s)
  return mp4s
