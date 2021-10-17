import cv2

import glob
print(cv2.__version__)
images = glob.glob("./*.png")

params = cv2.SimpleBlobDetector_Params()


params.minThreshold = 1
params.maxThreshold = 200



detector = cv2.SimpleBlobDetector_create(params)

im = cv2.imread("mog1_binarized_276.png",cv2.IMREAD_GRAYSCALE)

detector.empty()

kp = detector.detect(im)

print(kp)
