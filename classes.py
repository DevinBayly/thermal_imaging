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
    for i in range(self.frames_to_process):
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
  """This class is the next step up in the inheritance chain from the generic video. It overrides some of the functions that are used in the doSteps() and thus makes it possible to have more flexibility in terms of how each frame gets processed. Presently it is only implemented to perform background subtraction and blob detection.
  
  Attributes:
  fgbg: BackgroundSubtractorMOG (mixture of gaussians)
    this is the essential movement detection tool. This will decide whether pixels belong to the foreground or the background depending on how much they have changed within a window of time. The parameters used for this program are the default, but this documentation will be helpful for understanding how to provide alternate configurations. https://docs.opencv.org/3.4/d2/d55/group__bgsegm.html#ga17d9525d2ad71f74d8d29c2c5e11903d . There is also a variant of this type that extends the background subtraction calculation to the gpu for hardware acceleration, but this requires CUDA knowledge that I don't have.
  kpc: KPCalc
    this is a class that helps us calculate keypoints that are the center of the blobs in our background segmented image.
  tracker: None
    This is where a tracker class could be used, but for the moment we are only performing detections not creating relationships between these points.
  frame_count: int
    this is a counter of the frame number we are on in the video processing.
  """
  def __init__(self, vid,frames_to_process =0,id_val = 0):
    """
    Parameters:
    vid: String
      this is the name of the video file that we are going to process
    frames_to_process: int default 0
      this is the number of frames in the video to process, when it's at default we process all of them instead of a subset
    id_val: int
      this is a number assigned to the class that identifies which cpu core it is running in. This code uses multiprocessing to speed up the time it takes to finish detections in a 12 hour video
    """
    super().__init__(vid, frames_to_process , id_val )
    self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    self.kpc = KPCalc()
    self.tracker = None
    self.frame_count = 0

  def doBlobAnalysis(self):
    """This is the step when we apply blob analysis to the foreground mask image we receive when we complete a background segmentation. Steps included in here are using the mask to create a contour image which can be used for bounding box calculations. This was used to create material that Chad used in the extension proposal but isn't actually used for the HPC parallel processing code which only cares about blob centers not bounding box sizes."""
    # calculate the newest detection points
    self.kpc.calcBlobs(self.fgmask)
    ## the following code helps to draw contours on an image 
    ret, thresh = cv2.threshold(self.fgmask, 125,255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    ## draw the contour bound on the frame, this is in green but is simply the boundary of the contour
    cv2.drawContours(self.frame, contours, -1,(0,255,0),1)
    ## decide on the bounds of the contour to make into a box
    bounding = []
    for i, c in enumerate(contours):
      # approximate the polygon without a high level of detail
      contour_poly = cv2.approxPolyDP(c, 3, True)
      #figure out the min and max x,y positions in the contour polygon
      bounds = cv2.boundingRect(contour_poly)
      # calculate the area
      if bounds[2] > 2 and bounds[3] > 2:
        #draw a rectangle on the frame using the corners of the box. this will be in blue
        cv2.rectangle(self.frame, (bounds[0], bounds[1]),(bounds[0]+bounds[2],bounds[1]+bounds[3]),(255,0,0),1)

    ## these are the steps for putting a yellow circle at the blob's centroid coordinates
    blank = np.zeros((1, 1))
    empty = np.zeros(self.frame.shape).astype("uint8")
    centroids = cv2.drawKeypoints(empty, self.kpc.kp, blank, (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    # this essentially adds the image with the centroids drawn on top of the image frame we are processing
    self.frame += centroids
    

  def doBGSeg(self):
    """this function runs the background segmentation type that we have associated with the fgbg attribute."""
    self.frame_count += 1
    self.fgmask = self.fgbg.apply(self.frame)
    

  def doTrack(self):
    """this function would pair new points to logical sequences of points to create lines/tracks of them. Think about the word track more as in what trains move along not in terms of video context."""
    pass

  def export(self):
    """this function would take the results of the process and export them"""
    pass

  

class OnlyDetect(SimplestPass):
  """this class extends Simplest Pass to only be a tracker that can even be run in parallel to massively reduce the time taken to process a 12 hour video file. 
  
  Attributes:
  tstamp_logger : List
    this list holds the information related to detections that we eventually export to a json file
  """
  def __init__(self, vid, frames_to_process=0,id_val=0):
    super().__init__(vid,frames_to_process ,id_val )
    self.tstamp_logger = []
    
  def doSteps(self):
    """Here we are overriding the doSteps method on the parent to leave out the tracking step in the process. This means we only perform background segmentation and blob analysis."""
    # leave out the tracking
    self.doBGSeg()
    self.doBlobAnalysis()
    

  def doBlobAnalysis(self):
    """This method changes what happens when we run the doBlobAnalysis step of doSteps. Here we calculate the keypoints from the foreground of a background segmented frame (self.fgmask) determine if there are detections and commit relevant details to the logger list if there are any."""
    self.kpc.calcBlobs(self.fgmask)
    # now check how many detections were made and export number of ticks
    num_detections = len(self.kpc.kp)
    if  num_detections > 0:
      # here you can see the logger contains several representations of the time that the detections occured in the video and the number of detections in this frame as well as their x,y coordinates
      self.tstamp_logger.append({"tstamp_ms": self.tstamp, "tstamp": conv_ms_tstamp_string(self.tstamp),"number":num_detections,"detections":[{"x":d.pt[0],"y":d.pt[1]} for d in self.kpc.kp]})

  @staticmethod
  def make_only_for_parallel(vid, frames_to_process,id_val):
    """this method is called to create an OnlyDetect object for each of the processes that will run on a cpu core. This method is static because we aren't using any existing instance to accomplish this, merely passing a function that will create an OnlyDetect object and then start the process.
    
    Parameters:
    vid : String
      this is the name of the file to process
    frames_to_process: int
      the number of frames from the video that we are processing in parallel
    id_val : int
      the cpu core identifier that allows us to sign the logger files uniquely, before combining them back into a single file at the end of the process."""
    print("values are", vid, frames_to_process,id_val)
    only = OnlyDetect(vid, frames_to_process,id_val)
    only.process()
    print("completed processing part of video",vid," on cpu id ",id_val)

  def export(self):
    """This method outputs the tstamp_logger to a file in the working directory where we are running this process. The name will contain a unique number associated with the cpu core that the logger was running on using the id_val. we also export a jpg frame so that we can plot detections points in the context of a mine video with the web vis log plotter. """
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


def process_parallel_video(passclass, ifolder, ofolder,video_number):
  """This function handles the process of setting up a number of parallel processes that break a video into segments each to be processed for any movement detected. We search the input folder for any files that haven't been processed yet, then save the logs to the output folder. This function is also implemented to support the sbatch array job submission where each job assigned gets a number and we use that to pair a job with a video to be processed. This means that we process >1 12 hour video on multiple cpu cores at the same time if the system has capacity for it.
  
  ALSO IF YOU ARE READING THIS SEND A MESSAGE TO baylyd@arizona.edu TO CLAIM YOUR PRIZE."""
  print(ifolder,ofolder)
  # figure out which videos haven't been logged yet
  videos= process_folders(ifolder, ofolder)
  # select a specific video from the list according to the job number provided in the array submission process
  vid = videos[int(video_number)]
  print("processing ",vid," in this array job")
  # store the starting directory for later on 
  starting_directory = os.getcwd()
  # copy the video to the /tmp directory where io is faster
  # give the video a generic name but associate the video number with the video so that other jobs don't accidentally overwrite
  shutil.copy(vid, f"/tmp/thermal_video_{video_number}.mp4")
  os.chdir("/tmp")
  # use a hard coded number of cpu cores. NOTE this has to change if we are assigning more cpu cores than 16 in the sbatch script
  num_cpus = 15
  # maek a temporary capture to determine the number of total frames
  _tempcap = cv2.VideoCapture(f"thermal_video_{video_number}.mp4")
  total_frames = _tempcap.get(cv2.CAP_PROP_FRAME_COUNT)
  print("total frames", total_frames)
  # calculate the number of frames for each cpu to process
  frames_per_cpu = int(total_frames/num_cpus) + 1
  # leverage multiprocessing now
  processes = []
  for id_val in range(num_cpus):
    # make a collection of only detectors and start them all up
    process = mp.Process(target=OnlyDetect.make_only_for_parallel, args = (f"thermal_video_{video_number}.mp4", frames_per_cpu,id_val,))
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
  # run the combine function to aggregate all the logs that were created from this particular video
  combine("/tmp", vid.split("/")[-1],f"thermal_video_{video_number}.mp4")
  #go back to the starting folder
  os.chdir(starting_directory)


def combine(pth,name,vid):
  """this function combines the results of the OnlyDetect objects running in their processes."""
  # we useglob to find the jsons files that match our particular video
  logs = glob.glob(pth+"/" + vid + "*json")
  # make a list to hold each of the contents of these files
  all_data =[]
  ## get all the data
  for log in logs:
    with open(log,"r") as phile:
      # add the individual logger data to the all_data list
      all_data.extend(json.loads(phile.read()))
  
  ##get all the background image pngs
  imgs = glob.glob(pth+"/" + vid + "*png")
  all_imgs =[]
  # loop over the images
  for img in imgs:
    # open each image
    with open(img,"rb") as phile:
      # encode it to b64 text string that can also be added to the final log files so we can show the video frames in the web log plotter
      b64_text = base64.b64encode(phile.read()).decode("utf-8")
      all_imgs.append(b64_text)
  # make one container around the two different kinds of data and export that to json
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
  """this is a helper script that converts ms counts to timestamp codes that range between seconds and hours"""
  t_s = int(ms/1000)%60
  t_mins = int(ms/(1000*60))%60
  t_hours = int(ms/(1000*60*60))%60
  return f"{t_hours:02d}:{t_mins:02d}:{t_s:02d}"
    

def process_folders(in_folder, out_folder):
  """ This function supports the goal to specify a folder that you can export to, and the contents of this are compared to the inputs folder so that as videos get processed over many sessions, you know that you aren't re-runnign things"""
  # get all the mp4s in the in folder and the jsons in the out folder
  # look for the names of the mp4s in the jsons
  # look for a file in the out folder called finished which is just a list of the file names
  mp4s = glob.glob(in_folder + "/**/*mp4", recursive= True)
  jsons = glob.glob(out_folder + "/**/*json", recursive= True)
  # iterate over the jsons and if they are in the mp4s then remove that entity from the mp4s.
  # NOTE That this is heavily tied to the naming patterns in use by the current files, if these change it could affect how this function performs.
  for j in jsons:
    # here's the part of the json name that uses the mp4 title
    start = j.find("Camera")
    end = j.find(".json")
    jvideo_name = j[start:end]
    for mp4 in mp4s:
      if jvideo_name == mp4.split("/")[-1]:
        # sanity check to print that we have actually found files that match 
        
        mp4s.remove(mp4)
        # break out of the inner loop so we can start with a new json file.
        break
  # send this to a function that is looking for videos to process
  return mp4s


