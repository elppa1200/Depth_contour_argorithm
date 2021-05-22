'''
Coded by Huere
Twitter @Elppa_Huere

This code uses two digital camera(ex. webcam, picam).
You have to tuning parameter on your own condition.
'''


import numpy as np
import cv2, sys


def hullsum(hull,lenth, pos):
    a = []
    for i in range(lenth):
        a.append(hull[i][0][pos])
    rtn = sum(a)/lenth
    return(int(rtn))

global imgR, imgL

cap0 = cv2.VideoCapture(0) # cap0 is right camera
cap1 = cv2.VideoCapture(1) # cap1 is left camera

_,frm0 = cap0.read()
_,frm1 = cap1.read()

imgR= cv2.imread(frm0,0)
imgL= cv2.imread(frm1,0)

# imgR= cv2.imread('depth/img/right.jpg',cv2.IMREAD_GRAYSCALE)
# imgL= cv2.imread('depth/img/left.jpg', cv2.IMREAD_GRAYSCALE)

imgL = cv2.resize(imgL, dsize=(640, 480), interpolation=cv2.INTER_AREA)
imgR = cv2.resize(imgR, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    
stereo = cv2.StereoBM_create(numDisparities=224, blockSize=7)#tuning this para
disparity = stereo.compute(imgL,imgR)

_, thresh = cv2.threshold(disparity ,np.max(disparity)*0.5,np.max(disparity), cv2.THRESH_BINARY)
kernel = np.ones((7,7), np.uint8)
result = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
result = result.astype(np.uint8)

cntr, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntr = max(cntr, key = lambda x: cv2.contourArea(x))

epsil = 0.0005*cv2.arcLength(cntr,True)
appctr = cv2.approxPolyDP(cntr,epsil,True)
    
hull = cv2.convexHull(appctr, returnPoints=False)
defects = cv2.convexityDefects(appctr, hull)

for i in range(defects.shape[0]):
    s, e, _, _ = defects[i,0]
    start = tuple(appctr[s][0])
    end = tuple(appctr[e][0])

    cv2.line(frm0,start, end, [20,255,255], 2)

cv2.imshow('frame',frm0)

k = cv2.waitKey(5) & 0xFF
if k == 27: 
    sys.exit()
