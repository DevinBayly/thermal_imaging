import argparse
from classes import *

argparser = argparse.ArgumentParser()
argparser.add_argument("input_folder")
argparser.add_argument("output_folder")
argparser.add_argument("video_number")
args = argparser.parse_args()

try:
    process_parallel_video(OnlyDetect,args.input_folder,args.output_folder,args.video_number)
except Exception as e:
    print("encountered error \n",e)