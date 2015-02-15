#!/usr/bin/env python
import numpy as np
import cv2
import glob
import sys

imgs = glob.glob("dataset/Walking/img/*.jpg")
imgs.sort()

first_frame = cv2.imread(imgs[0])
previous =  cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
#previous = cv2.fastNlMeansDenoising(previous)
hsv = np.zeros_like(first_frame)
hsv[...,1] = 255
    
for imgname in imgs[1:]:
    print (imgname)
    current_rgb = cv2.imread(imgname)
    current = cv2.cvtColor(current_rgb, cv2.COLOR_BGR2GRAY)
    #current = cv2.fastNlMeansDenoising(current)
    cv2.imshow('previous', previous)
    cv2.imshow('current', current)
    flow = cv2.calcOpticalFlowFarneback(previous, current, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('motion', rgb);
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 255//8, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
    #cv2.drawContours(current_rgb, contours, -1, (0,255,0), 3)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w * h > 200:
            cv2.rectangle(current_rgb, (x,y), (x + w, y + h), (255,0,0),2)

    cv2.imshow('contour', current_rgb);
    cv2.waitKey(100)
    
    previous = current

cv2.destroyAllWindows()
