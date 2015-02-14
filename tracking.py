#!/usr/bin/env python
import numpy as np
import cv2
import glob
import sys

imgs = glob.glob("dataset/Walking/img/*.jpg")
imgs.sort()

first_frame = cv2.imread(imgs[0])
previous =  cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(first_frame)
hsv[...,1] = 255
    
for imgname in imgs[1:]:
    print (imgname)
    current = cv2.imread(imgname)
    current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(previous, current, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame', rgb);
    cv2.waitKey(100)
    
    previous = current

cv2.destroyAllWindows()
