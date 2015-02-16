#!/usr/bin/env python
import numpy as np
import cv2
import glob
import sys
from scipy.cluster.vq import kmeans2
from collections import defaultdict

class FrameData:
    def __init__(self, frame):
        self.frame = frame
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.track_points = None

class Cluster:
    def __init__(self, cluster):
        self.color = np.random.randint(0,255,3)      
        self.cluster = cluster

    def draw(self, img):
        rect= cv2.minAreaRect(np.array(self.cluster))
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,self.color,2)
    
class Tracking:
    def __init__(self):
        self.oldframes = []
        self.refresh_interval = 1
        self.tracks = []
        self.track_len = 10
        self.frame_index = 0
        self.clusters = []

        self.lk_params = dict( 
            winSize  = (10, 10), 
            maxLevel = 5, 
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.feature_params = dict(
            maxCorners = 3000, 
            qualityLevel = 0.25,
            minDistance = 3,
            blockSize = 3
        )

        self.cluster_criteria = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
        self.fgmask = None
        self.fgbg = cv2.BackgroundSubtractorMOG2(history=200, varThreshold=16, bShadowDetection=True)
        self.cur_frame = None
        self.prev_frame = None
        self.flow = None

    def showCluster(self):
        if (len(self.tracks) < 1):
            return
        points = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 2)
        velocity = np.float32([np.subtract(tr[-1],tr[-2]) for tr in self.tracks]).reshape(-1,2) * 10 **3
        cluster_data = np.hstack((points, velocity))
        points = points[(np.absolute(cluster_data[:,2]) > 1) & (np.absolute(cluster_data[:,3]) > 1)]
        cluster_data = cluster_data[(np.absolute(cluster_data[:,2]) > 1) & (np.absolute(cluster_data[:,3]) > 1)]
        #cluster_n = int((len(points) / 2)** 1/2)
        cluster_n = 10
        #ret, labels, centers = (cv2.kmeans(cluster_data, cluster_n, self.cluster_criteria, 10, 0))
        centers, labels = kmeans2(cluster_data, cluster_n)

        clusters = defaultdict(list)        
        for (x,y), label in zip(points.reshape(-1,2), labels.ravel()):
            clusters[label].append((x,y))

        self.clusters = []
        for key, val in clusters.iteritems():
            self.clusters.append(Cluster(val))
            
    def processFrame(self, frame):
        self.prev_frame = self.cur_frame
        self.cur_frame = FrameData(frame);
        self.oldframes.append(self.cur_frame)
        self.fgmask = self.fgbg.apply(self.cur_frame.frame, learningRate=0.01)

        if self.prev_frame and len(self.tracks) > 0:
            last_points = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            found_points, st0, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_frame.gray, 
                self.cur_frame.gray,
                last_points,
                None,
                **self.lk_params
            )            
            refound_points, st1, _ = cv2.calcOpticalFlowPyrLK(
                self.cur_frame.gray,
                self.prev_frame.gray,
                found_points,
                None, 
                **self.lk_params
            )
            potential_points = abs(last_points-refound_points).reshape(-1, 2).max(-1)
            within_range = abs(refound_points - last_points).reshape(-1,2).max(-1) < 1
            new_tracks = []
            for tr, (x,y), range_ok in zip(self.tracks, found_points.reshape(-1,2), within_range):
                if not range_ok:
                    continue
                tr.append((x,y))
                if len(tr) > self.track_len:
                    del tr[0]
                new_tracks.append(tr)
            self.tracks = new_tracks
            self.cullTracks()
            
        #self.showCluster()

    def cullTracks(self):
        self.tracks.sort()
        new_tracks = []
        prev = None
        for tr in self.tracks:
            x,y = tr[-1]
            x = int(x)
            y = int(y)
            if prev == None or prev != (x, y):
                new_tracks.append(tr)
                prev = (x,y)
        self.tracks = new_tracks    
        
    def nextFrame(self, frame):
        self.frame_index += 1
        self.processFrame(frame)
        self.updateTrackingPoints()
        self.debug()

    def updateTrackingPoints(self):
        if self.frame_index % self.refresh_interval == 0:
            points = cv2.goodFeaturesToTrack(self.cur_frame.gray, **self.feature_params)
            self.tracks += [[(x,y)] for (x,y) in points.reshape(-1,2)]

    def debug(self):
        img = self.cur_frame.frame.copy()
        cv2.polylines(img, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
        cv2.putText(img, "Tracks %d" % len(self.tracks), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        mask = self.fgmask
        k = np.ones((3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.blur(mask, (10,10))
        mask = cv2.dilate(mask,np.ones((5,5), np.uint8),iterations=1)
        cv2.imshow('fgmask1', mask)
        ret, mask = cv2.threshold(mask, 255//5,255,0)
        mask = cv2.erode(mask, np.ones((5,5), np.uint8))
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
        cv2.drawContours(img, contours, -1, (255,0,0), 3)
        [c.draw(img) for c in self.clusters]
        cv2.imshow("cur frame", img)
        cv2.imshow('fgmask', mask)
        cv2.waitKey(1)


t = Tracking()    
if False:
    imgs = glob.glob("dataset/Subway/img/*.jpg")
    imgs.sort()
    
    for imgname in imgs[0:]:
        t.nextFrame(cv2.imread(imgname))
else:
    cap = cv2.VideoCapture("/home/crzysdrs/downloads/qmul_junction.avi")
    ret = 1
    while ret != 0:
        ret, frame = cap.read()
        t.nextFrame(frame)
        

cv2.destroyAllWindows()
