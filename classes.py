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
      print(vid)
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
    only = OnlyDetect(vid,centroid_threshold,time_limit,frames_to_process,id_val)
    
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



class SimplestPassMOG2(SimplestPass):
  def __init__(self,vid,centroid_threshold):
    super().__init__(vid,centroid_threshold)
    self.fgbg = cv2.createBackgroundSubtractorMOG2()



class TrackerBase:
  def __init__(self):
    self.tracks =[]
  def compute(self,new_detections):
    #TB overridden
    pass
#not implemented  
class TrackerBB(TrackerBase):
  def __init__(self):
    super().__init__()
  def compute(self,new_detections):
    pass


class TrackerCentroids(TrackerBase):
  def __init__(self,threshold,time_limit):
    super().__init__()
    self.threshold = threshold
    self.time_limit = time_limit
  def export(self):
    ## create a list of values returned from the actual detection, and the thing calling export on the tracker can make the dataframe from it
    track_res = []
    for track in self.tracks:
        track_res.append(track.export())
    return track_res


  def compute(self,new_detections):
    ## if tracks aren't established yet just add each detection as a track 
    ## go over list of tracks
    #print("number of tracks",len(self.tracks))
    trackim= np.zeros
    latest_tstamp = 0 ## just initializing
    if len(new_detections) >0:
      latest_tstamp = new_detections[0].tstamp

    for t in self.tracks:
      for d in new_detections:
        ##compare to the very last of the elements we are tracking
        if d.compare(t.detections[-1],self.threshold):
          t.detections.append(d.copy())
      ## if the last detections time stamp is too old (in ms) then we should close up the track
      
      if latest_tstamp - t.detections[-1].tstamp > self.time_limit:
        t.close_track()

      ## plot the tracks
    #print("number of detections",len(new_detections))

    
    
    ## make new tracks for the new unspoken for detections

    for d in new_detections:
      if d.count == 0:
        randColor = (np.random.random((3))*255).astype("uint8").tolist()
        self.tracks.append(CentroidTrack(randColor,d,d.tstamp))
      

class Track:
  def __init__(self,color,det,timestart):
    self.color = (color[0],color[1],color[2])

    self.detections =[det]
    # this is a timestamp value
    self.timestart = timestart
  ## expectation is that a dictionary gets returned
    self.timeend = None
  def export(self):
    pass

#class BBox

class CentroidTrack(Track):
  def __init__(self,color,det,timestart):
    super().__init__(color,det,timestart)



  def close_track(self):
    self.timeend = self.detections[-1].tstamp 

  ## include these pieces of information
  ## the timestamp that the track started
  ## timestamp that it ended !! TODO think about how to work this out
  # the bounding box that it corresponds to
  # a tag related to whether it was rockfall or not
  def export(self):
    detections_as_arr = np.array([d.center for d in self.detections])
    bbox = np.min(detections_as_arr,axis =0),np.max(detections_as_arr,axis =0)
    t_start_ms=self.timestart
    start_tstamp_string = conv_ms_tstamp_string(t_start_ms)
    if self.timeend == None:
      self.close_track()
    t_end_ms=self.timeend
    end_tstamp_string = conv_ms_tstamp_string(t_end_ms)


    duration_ms = t_end_ms - t_start_ms
    duration_tstamp = conv_ms_tstamp_string(duration_ms)
    return {"onset_ms":t_start_ms,
            "onset_tstamp":start_tstamp_string,
            "end_ms":t_end_ms,
            "end_tstamp":end_tstamp_string,
            "duration_ms":duration_ms,
            "duration_tstamp":duration_tstamp,
            "bbox_min_x":bbox[0][0],
            "bbox_min_y":bbox[0][1],
            "bbox_max_x":bbox[1][0],
            "bbox_max_y":bbox[1][1]}


def conv_ms_tstamp_string(ms):
  t_s = int(ms/1000)%60
  t_mins = int(ms/(1000*60))%60
  t_hours = int(ms/(1000*60*60))%60
  return f"{t_hours:02d}:{t_mins:02d}:{t_s:02d}"

def combine(pth,name):
  logs = glob.glob(pth+"/thermal_logger*json")
  all_data =[]
  ## get all the data
  for log in logs:
    with open(log,"r") as phile:
      all_data.extend(json.loads(phile.read()))
  
##get all the backgroundss
  imgs = glob.glob(pth+"/thermal_logger_img*")
  all_imgs =[]
  for img in imgs:

    with open(img,"rb") as phile:
      b64_text = base64.b64encode(phile.read()).decode("utf-8")
      all_imgs.append(b64_text)
  all_log = dict(data=all_data,background_images=all_imgs)
  with open(f"all_logs_{name}.json","w") as phile:
    phile.write(json.dumps(all_log)) 
  ## now remove the logs
  for log in logs:
    try:
      os.remove(log)
    except Exception as e:
      print("error, might have been someone elses log file?",e)

## this object is on frame's detection, multiple of these make up a set to track
class TrackObjectBase:
  def __init__(self,center,tstamp):
    self.center = center
    self.count=0
    self.tstamp = tstamp
  def compare(self,other,threshold):
    pass
  def copy(self):
    pass

class BBObject(TrackObjectBase):
  ## upper left and bottom right
  def __init__(self,ul,br):
    super().__init__(self,center,tstamp)
    self.ul = ul
    self.br = br
  def contains(self,pt):
    pass
## TODO finish line

class CentroidObject(TrackObjectBase):
  def __init__(self,center,tstamp):
    super().__init__(center,tstamp)
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
    copy = CentroidObject(self.center,self.tstamp)
    copy.count = self.count
    return copy

# testing classes
def test_video(passClass):
  threshold = 5
  time_limit = 4000 # in milliseconds
  videos = glob.glob("./*/*mp4")
  print(videos,"these were the mp4s found")
  vid ="mine-4_rockfall_clips/Camera 2 - 192.168.0.121 (FLIRFC-632-ID-22947C)-20210526-234803.mp4"
  print("executing on ",vid)
  pass_var = passClass(vid,threshold,time_limit)
  
  pass_var.process()
  ## now finish up by exporting the data
  pickle.dump(pass_var.tracker,open("testpickle","wb"))
  return pass_var



def test_detect():
  threshold = 20
  time_limit = 4000
  vid = "/xdisk/chrisreidy/baylyd/thermal_imaging/Mine-4/052621/Camera 2 - 192.168.0.121 (FLIRFC-632-ID-22947C)-20210526-082632.mp4"
  shutil.copy(vid,"/tmp/thermal_video.mp4")
  os.chdir("/tmp")
  OnlyDetect.make_only_for_parallel("thermal_video.mp4",threshold,time_limit,0,0)
  

def test_parallel_video(passclass,ifolder,ofolder):
  vids =process_folders(ifolder,ofolder)
  starting_directory= os.getcwd()
  threshold = 20
  time_limit = 4000 # in milliseconds
  for vid in vids:
    #vid ="mine-4_rockfall_clips/Camera 2 - 192.168.0.121 (FLIRFC-632-ID-22947C)-20210526-234803.mp4"
    shutil.copy(vid,"/tmp/thermal_video.mp4")
    os.chdir("/tmp")
    num_cpus = 16
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

  
def test_track():
  ## make an image use kpc on thing and then try to pass the values to a tracker
  threshold = 10
  tracker= TrackerCentroids(threshold)
  kpc = KPCalc()
  images = glob.glob("images/*png")
  im = cv2.imread(images[int(len(images)/2)])
  #cv2.imshow("bin image",im)
  kpc.calcBlobs(im)
  detections = []
  for kp in kpc.kp:
    centroid_object = CentroidObject(np.array([kp.pt[0],kp.pt[1]]))
    ##print(centroid_object)
    detections.append(centroid_object)

  ##print(detections)
  tracker.compute(detections)
  ##print(tracker.tracks)



def timer(msg,f,passClass):
  print(msg)
  start = time.time()
  f(passClass)
  print("seconds passed",time.time() - start)


#timer("outer timer",test_video,SimplestPass)

# goal have a folder that you can export to, and the contents of this are compared to the inputs folder so that as things finish over time you know that you aren't re-runnign things
def process_folders(in_folder,out_folder):
    ## get all the mp4s in the in folder and the jsons in the out folder
    ## look for the names of the mp4s in the jsons
  ## look for a file in the out folder called finished which is just a list of the file names 
  mp4s = glob.glob(in_folder + "/**/*mp4",recursive = True)
  jsons = glob.glob(out_folder + "/**/*json",recursive = True)
  print(mp4s)
  print(jsons)
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


#  print(mp4s,jsons)

