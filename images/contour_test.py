import cv2
print(cv2.__version__)
img = cv2.imread("265.png")
im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("original.png",im)
ret,thresh = cv2.threshold(im,125,255,0)
cv2.imwrite("thresholded.png",thresh)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
##
cv2.drawContours(img,contours,-1,(0,255,0),1)

cv2.imwrite("contoured.png",img)
