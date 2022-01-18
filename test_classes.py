import argparse
from classes import *

argparser = argparse.ArgumentParser()
argparser.add_argument("input_folder")
argparser.add_argument("output_folder")
args = argparser.parse_args()

test_parallel_video(OnlyDetect,args.input_folder,args.output_folder)