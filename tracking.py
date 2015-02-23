#!/usr/bin/env python
import numpy as np
import cv2
import glob
import sys
from scipy.cluster.vq import kmeans2
from sklearn.cluster import DBSCAN
from collections import defaultdict

def draw_motion_comp(vis, (x, y, w, h), angle, color):
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0))
    r = min(w/2, h/2)
    cx, cy = x+w/2, y+h/2
    angle = angle*np.pi/180
    cv2.circle(vis, (cx, cy), r, color, 3)
    cv2.line(vis, (cx, cy), (int(cx+np.cos(angle)*r), int(cy+np.sin(angle)*r)), color, 3)

def object_seg(a, b):
    dist = np.linalg.norm(a[0:1]-b[0:1])
    return dist * (a[2] - b[2])**2 + (a[3] - b[3])**2 / 5 * 2

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
        self.hotpoints = []
        self.track_len = 10
        self.frame_index = 0
        self.clusters = []
        self.history = 100
        self.motion_history = None
        self.lk_params = dict( 
            winSize  = (10, 10), 
            maxLevel = 5, 
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.feature_params = dict(
            maxCorners = 3000, 
            qualityLevel = 0.15,
            minDistance = 3,
            blockSize = 3
        )

        self.cluster_criteria = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
        self.fgmask = None
        self.fgbg = cv2.BackgroundSubtractorMOG2()
        self.cur_frame = None
        self.prev_frame = None
        self.flow = None
        self.centers = None

    def highlightMotion(self, img):
        circle_radius = 4
        motion_img = np.zeros_like(self.cur_frame.frame)
        tracks = filter(lambda x: len(x) < 10, self.tracks)
        motion_tracks = filter(lambda x: len(x) >= 10, self.tracks)
        motion_tracks = filter(lambda x: np.any(np.absolute(np.subtract(x[-1],x[0])) > 2), motion_tracks)
        if len(motion_tracks) == 0:
            return motion_img
        points = np.float32([tr[-1] for tr in motion_tracks]).reshape(-1, 2)
        velocity = np.float32([np.subtract(tr[-1],tr[-2]) for tr in motion_tracks]).reshape(-1,2) * 10 **3
        cluster_data = np.hstack((points, velocity))
        points = points[(np.absolute(cluster_data[:,2]) > 1) & (np.absolute(cluster_data[:,3]) > 1)]
        cluster_data = cluster_data[(np.absolute(cluster_data[:,2]) > 1) & (np.absolute(cluster_data[:,3]) > 1)]
        mag, ang = cv2.cartToPolar(velocity[...,0], velocity[...,1])
        hsv = np.uint8(np.zeros((mag.shape[0], 3)))
        hsv[...,0] = (ang*180/np.pi/2).reshape(ang.shape[0])
        hsv[...,1] = 255
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX).reshape(ang.shape[0])
        rgb = cv2.cvtColor(hsv.reshape(ang.shape[0], -1, 3),cv2.COLOR_HSV2BGR)
        rgb = rgb.reshape(ang.shape[0], 3)
        culledtracks = []
        for (t, c) in zip(motion_tracks, rgb):
            p = map(int, t[-1])
            try:
                target = motion_img[p[1], p[0]]            
                if np.all(target == 0):
                    cv2.circle(motion_img, (p[0], p[1]), circle_radius, map(int, c), -1)
                    cv2.circle(img, (p[0], p[1]), circle_radius, map(int, c), -1)
                    culledtracks.append(t)
            except IndexError:
                pass

        self.tracks = tracks + culledtracks
        return motion_img

    def motionTracking(self, motion_mask):
        MHI_DURATION = 4
        MAX_TIME_DELTA = 20
        MIN_TIME_DELTA = 2
        h, w = self.cur_frame.frame.shape[:2]
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[:,:,1] = 255
        if self.motion_history == None:
            self.motion_history = np.zeros((h, w), np.float32)
        motion_history = self.motion_history
        timestamp = self.frame_index
        cv2.updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION)
        mg_mask, mg_orient = cv2.calcMotionGradient(motion_history, MAX_TIME_DELTA, MIN_TIME_DELTA, apertureSize=7 )
        seg_mask, seg_bounds = cv2.segmentMotion(motion_history, timestamp, MAX_TIME_DELTA)
        
        visual_name = 'motion_hist'
        if visual_name == 'input':
            vis = self.cur_frame.frame.copy()
        elif visual_name == 'frame_diff':
            pass
            #vis = frame_diff.copy()
        elif visual_name == 'motion_hist':
            vis = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        elif visual_name == 'grad_orient':
            hsv[:,:,0] = mg_orient/2
            hsv[:,:,2] = mg_mask*255
            vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        for i, rect in enumerate([(0, 0, w, h)] + list(seg_bounds)):
            x, y, rw, rh = rect
            area = rw*rh
            if area < 400:
                continue
            silh_roi   = motion_mask   [y:y+rh,x:x+rw]
            orient_roi = mg_orient     [y:y+rh,x:x+rw]
            mask_roi   = mg_mask       [y:y+rh,x:x+rw]
            mhi_roi    = motion_history[y:y+rh,x:x+rw]
            if cv2.norm(silh_roi, cv2.NORM_L1) < area*0.05:
                continue
            angle = cv2.calcGlobalOrientation(orient_roi, mask_roi, mhi_roi, timestamp, MHI_DURATION)
            color = ((255, 0, 0), (0, 0, 255))[i == 0]
            draw_motion_comp(vis, rect, angle, color)
        cv2.imshow("vis", vis)

    def processFrame(self, frame):
        self.prev_frame = self.cur_frame
        self.cur_frame = FrameData(frame);
        self.oldframes.append(self.cur_frame)
        if len(self.oldframes) > self.history:
            del self.oldframes[0]

        self.fgmask = self.fgbg.apply(self.cur_frame.frame, learningRate=0.001)

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
                last = tr[-1]
                if not range_ok:
                    continue
                tr.append((x,y))
                if len(tr) > self.track_len:
                    del tr[0]
                new_tracks.append(tr)
            self.tracks = new_tracks
        
    def nextFrame(self, frame):
        self.frame_index += 1
        self.processFrame(frame)
        self.updateTrackingPoints()
        self.debug()

    def updateTrackingPoints(self):
        if self.frame_index % self.refresh_interval == 0:
            motion_tracks = filter(lambda x: len(x) >= self.refresh_interval, self.tracks)
            motion_tracks = filter(lambda x: np.any(np.absolute(np.subtract(x[-1],x[0])) > 0.25), motion_tracks)
            self.tracks = motion_tracks;
            points = cv2.goodFeaturesToTrack(self.cur_frame.gray, **self.feature_params)
            self.tracks += [[(x,y)] for (x,y) in points.reshape(-1,2)]

    def debug(self):
        iters = 3
        img = self.cur_frame.frame.copy()
        cv2.putText(img, "Hot: %d" % len(self.hotpoints), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.putText(img, "Frame: %d" % self.frame_index, (0,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.putText(img, "Tracks: %d" % len(self.tracks), (0,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        mask = self.fgmask.copy()
        motion = self.highlightMotion(img)
        self.motionTracking(cv2.cvtColor(motion, cv2.COLOR_BGR2GRAY))
        cv2.polylines(img, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
        cv2.imshow("Result", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    t = Tracking()        
    cap = cv2.VideoCapture(sys.argv[1])
    ret = 1
    samples = 0
    while ret != 0:
        ret, frame = cap.read()
        if ret == 0:
            continue
        t.nextFrame(frame)
    cv2.destroyAllWindows()
