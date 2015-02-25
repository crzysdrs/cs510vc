#!/usr/bin/env python
import numpy as np
import cv2
from kalman2d import Kalman2D
import glob
import sys
from pygame import Rect as PyRect

def pyrect2np(rect):    
    return np.matrix([rect.x, rect.y, 1], np.float32)

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

class TrackManager:
    def __init__(self):
        self.tracked = []

    def draw(self, img):
        [x.draw(img) for x in self.tracked]

    def addRect(self, area):
        #pyRect (left, top, width, height)
        new_area = PyRect(area[0], area[1], area[2], area[3])
        
        possible = filter(lambda x: x.rect.colliderect(new_area), self.tracked)
        if len(possible) == 0:
            self.tracked.append(Tracked(new_area, []))
        elif len(possible) == 1:
            p = possible[0]
            p.updateRect(new_area)
        else:
            #hits multiple existing tracked items, ignore
            pass
    
    def updateFrame(self, img, img_hsv):
        [x.updateFrame(img_hsv) for x in self.tracked]

class Tracked:
    def __init__(self, rect, points):
        self.color = np.random.randint(0,255,3)
        self.rect = rect
        self.points = points
        self.roi_hist = None
        self.kalman = Kalman2D(x=self.rect.x, y=self.rect.y)
        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        
        self.camshift = None
        self.updated = False

    def updateHist(self, img_hsv):
        r = self.rect
        hsv_roi = img_hsv[r.top:r.bottom, r.left:r.right]
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
        self.roi_hist = roi_hist

    def draw(self, img):
        cv2.rectangle(img, self.rect.topleft, self.rect.bottomright,
                      self.color)

        #if self.camshift != None:
        #    box = cv2.cv.BoxPoints(self.camshift)
        #    box = np.int0(box)
        #    cv2.drawContours(img,[box],0,(0,0,255),2)

    def updateFrame(self, img_hsv):
        if self.roi_hist == None:
            self.updateHist(img_hsv)

        if not self.updated:
            self.kalman.update_none()
            predict = self.kalman.getPredicted()
            self.rect.x = predict[0]
            self.rect.y = predict[1]
        self.updated = False
        #dst = cv2.calcBackProject([img_hsv],[0],self.roi_hist,[0,180],1)
        #if r.width <= 0 or r.height <=0:
        #    return
        #ret, tw = cv2.CamShift(dst, (r.left, r.top, r.width, r.height), self.term_crit)
        #self.rect = PyRect(tw[0], tw[1], tw[2], tw[3])        
        self.camshift = ret

    def updateRect(self, rect):
        if not self.updated:
            self.rect = rect
            self.kalman.update(self.rect.x, self.rect.y)
            self.updated = True
    def mergeRect(self, rect):
        pass
        #elf.rect = self.rect.union(rect)

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
        self.trackmanager = TrackManager()

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
        MHI_DURATION = 5
        MAX_TIME_DELTA = 10
        MIN_TIME_DELTA = 3
        h, w = self.cur_frame.frame.shape[:2]
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[:,:,1] = 255
        if not self.prev_frame:
            return
        if self.motion_history == None:
            self.motion_history = np.zeros((h, w), np.float32)
        motion_history = self.motion_history
        timestamp = self.frame_index
        frame_diff = cv2.absdiff(self.cur_frame.frame, self.prev_frame.frame)
        #frame_diff = cv2.morphologyEx(frame_diff,cv2.MORPH_OPEN,np.ones((5,5)), iterations = 1)
        cv2.imshow("frame_diff", frame_diff)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        gray_diff = cv2.bitwise_or(gray_diff, motion_mask)
        cv2.imshow("gray_diff", gray_diff)
        ret, motion_mask = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2,2),np.uint8)
        mymask = self.fgmask.copy()
        mymask = cv2.morphologyEx(mymask,cv2.MORPH_OPEN,np.ones((3,3)), iterations = 1)
        mymask = cv2.dilate(mymask, np.ones((3,3)))
        cv2.imshow("fgmask_unmod", self.fgmask)
        ret, mymask = cv2.threshold(mymask, 20, 255, cv2.THRESH_BINARY)
        if self.frame_index > 50:
            gray_diff_masked = cv2.addWeighted(gray_diff, 1, mymask, 0.25, 0.0)
        else:
            gray_diff_masked = gray_diff

        cv2.imshow("masked", gray_diff_masked)
        ret, motion_mask = cv2.threshold(gray_diff_masked, 20, 1, cv2.THRESH_BINARY)
        cv2.updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION)
        mg_mask, mg_orient = cv2.calcMotionGradient(motion_history, MAX_TIME_DELTA, MIN_TIME_DELTA, apertureSize=7 )
        seg_mask, seg_bounds = cv2.segmentMotion(motion_history, timestamp, MAX_TIME_DELTA)
        
        visual_name = 'motion_hist'
        if visual_name == 'input':
            vis = self.cur_frame.frame.copy()
        elif visual_name == 'frame_diff':
            vis = frame_diff.copy()
        elif visual_name == 'motion_hist':
            vis = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        elif visual_name == 'grad_orient':
            hsv[:,:,0] = mg_orient/2
            hsv[:,:,2] = mg_mask*255
            vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        for i, rect in enumerate([(0, 0, w, h)] + list(seg_bounds)):
            if i == 0:
                continue
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
            self.trackmanager.addRect(rect)
            draw_motion_comp(vis, rect, angle, color)
        cv2.imshow("vis", vis)

    def processFrame(self, frame):
        self.prev_frame = self.cur_frame
        self.cur_frame = FrameData(frame);
        self.oldframes.append(self.cur_frame)
        if len(self.oldframes) > self.history:
            del self.oldframes[0]
        self.fgmask = self.fgbg.apply(self.cur_frame.frame, learningRate=0.003)
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
        self.trackmanager.updateFrame(self.cur_frame.frame,
                                      cv2.cvtColor(self.cur_frame.frame, cv2.COLOR_BGR2HSV))
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
        cv2.putText(img, "Hot: %d" % len(self.hotpoints),
                    (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.putText(img, "Frame: %d" % self.frame_index,
                    (0,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.putText(img, "Tracks: %d" % len(self.tracks),
                    (0,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        mask = self.fgmask.copy()
        motion = self.highlightMotion(img)
        self.motionTracking(cv2.cvtColor(motion, cv2.COLOR_BGR2GRAY))
        cv2.polylines(img, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
        self.trackmanager.draw(img)
        cv2.imshow("Result", img)        
        cv2.waitKey(1)


if __name__ == '__main__':
    while(1):
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
