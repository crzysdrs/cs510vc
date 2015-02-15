#!/usr/bin/env python
import numpy as np
import cv2
import glob
import sys

class FrameData:
    def __init__(self, frame):
        self.frame = frame
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.track_points = None

class Tracking:
    def __init__(self, frame_0):
        self.refresh_interval = 1
        self.tracks = []
        self.track_len = 10
        self.frame_index = 0

        self.lk_params = dict( 
            winSize  = (10, 10), 
            maxLevel = 5, 
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.feature_params = dict(
            maxCorners = 3000, 
            qualityLevel = 0.5,
            minDistance = 3,
            blockSize = 3
        )

        self.cur_frame = None
        self.processFrame(frame_0)
    
    def processFrame(self, frame):
        self.prev_frame = self.cur_frame
        self.cur_frame = FrameData(frame);

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
            for tr, (x,y), range_flag in zip(self.tracks, found_points.reshape(-1,2),  within_range):
                if not range_flag:
                    continue
                tr.append((x,y))
                if len(tr) > self.track_len:
                    del tr[0]
                new_tracks.append(tr)
            self.tracks = new_tracks
            self.cullTracks()

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
        img = self.cur_frame.frame
        cv2.polylines(img, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
        cv2.putText(img, "Tracks %d" % len(self.tracks), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.imshow("cur frame", img)
        cv2.waitKey(1)

imgs = glob.glob("dataset/Walking/img/*.jpg")
imgs.sort()

t = Tracking(cv2.imread(imgs[0]))
for imgname in imgs[1:]:
    t.nextFrame(cv2.imread(imgname))

cv2.destroyAllWindows()
