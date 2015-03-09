#!/usr/bin/env python
from __future__ import division
import numpy as np
import cv2
from kalman2d import Kalman2D
import glob
import sys
from pygame import Rect as PyRect

def fold(f, l, a):
    """
    f: the function to apply
    l: the list to fold
    a: the accumulator, who is also the 'zero' on the first call
    """ 
    return a if(len(l) == 0) else fold(f, l[1:], f(a, l[0]))

def pyrect2np(rect):    
    return np.matrix([rect.x, rect.y, 1], np.float32)

def draw_motion_comp(vis, (x, y, w, h), angle, color):
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0))
    r = min(w//2, h//2)
    cx, cy = x+w//2, y+h//2
    angle = angle*np.pi//180
    cv2.circle(vis, (cx, cy), r, color, 3)
    cv2.line(vis, (cx, cy), (int(cx+np.cos(angle)*r), int(cy+np.sin(angle)*r)), color, 3)

def object_seg(a, b):
    dist = np.linalg.norm(a[0:1]-b[0:1])
    return dist * (a[2] - b[2])**2 + (a[3] - b[3])**2 / 5 * 2

def dist(a, b):
    return np.linalg.norm(np.array(a)-np.array(b))

def partition(pred, items):
    bools = map(pred, items)
    l_a = []
    l_b = []
    for (b, item) in zip(bools, items):
        if b:
            l_a.append(item)
        else:
            l_b.append(item)
    return (l_a, l_b)


split_tracks = lambda x: len(x.history()) >= 5

class AllTracks:
    MOVING_SPEED = 0.5
    def __init__(self):
        self.point_tracks = []
        self.feature_params = dict(
            maxCorners = 5000, 
            qualityLevel = 0.10,
            minDistance = 3,
            blockSize = 3
        )

    def getTracks(self):
        return map(lambda x : x.history(), self.point_tracks)

    def getPointTracks(self):
        return self.point_tracks
    
    def getMovingTracks(self, speed=None):
        if not speed:
            speed = AllTracks.MOVING_SPEED
        (short_tracks, long_tracks) = partition(split_tracks, self.point_tracks)
        motion_tracks = filter(lambda x: dist(x.current(), x.last()) > speed, long_tracks)
        return motion_tracks

    def count(self):
        return len(self.point_tracks)
    
    def refresh(self, cur_frame):
        (short_tracks, long_tracks) = partition(split_tracks, self.point_tracks)
        motion_tracks = filter(lambda x: dist(x.current(), x.last()) > AllTracks.MOVING_SPEED, long_tracks)
        points = cv2.goodFeaturesToTrack(cur_frame.gray, **self.feature_params)
        self.point_tracks = short_tracks + motion_tracks + [PointTrack(p) for p in points.reshape(-1,2)]
        
    def update(self, prev_frame, cur_frame):
        if prev_frame == None:
            return
        lk_params = dict( 
            winSize  = (10, 10), 
            maxLevel = 5, 
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        last_points = np.array(map(lambda t : t.current(), self.point_tracks)).reshape(-1,2)
        found_points, st0, _ = cv2.calcOpticalFlowPyrLK(
            prev_frame.gray, 
            cur_frame.gray,
            last_points,
            None,
            **lk_params
        )            
        refound_points, st1, _ = cv2.calcOpticalFlowPyrLK(
            cur_frame.gray,
            prev_frame.gray,
            found_points,
            None, 
            **lk_params
        )
        potential_points = abs(last_points-refound_points).reshape(-1, 2).max(-1)
        within_range = abs(refound_points - last_points).reshape(-1,2).max(-1) < 1
        updated_tracks = []
        for pt, (x,y), range_ok in zip(self.point_tracks, found_points.reshape(-1,2), within_range):
            if range_ok:
                pt.addHistory((x,y))
                updated_tracks.append(pt)
            else:
                pt.missing()
        
        self.point_tracks = updated_tracks

class PointTrack:
    next_id = 0
    max_len = 10
    def __init__(self, point):
        self.id = PointTrack.next_id
        PointTrack.next_id += 1
        self.track = [point]
        self.death_notify = []

    def addDeathObserver(self, observer):
        self.death_notify.append(observer)

    def addHistory(self, new_point):
        self.track.append(new_point)
        if len(self.track) > PointTrack.max_len:
            del self.track[0]
        
    def current(self):
        return self.track[-1]

    def last(self):
        return self.track[0]

    def missing(self):
        for n in self.death_notify:
            n(self)

    def history(self):
        return self.track


class FrameData:
    def __init__(self, frame):
        self.frame = frame
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.track_points = None

class ObjectManager:
    def __init__(self, tracker):
        self.objects = []
        self.tracker = tracker
        self.next_id = 0

    def draw(self, img):
        [o.draw(img) for o in self.objects]

    def addRect(self, area):
        #pyRect (left, top, width, height)
        new_area = PyRect(area[0], area[1], area[2], area[3])
        tracks = map(lambda x : x.current(), self.tracker.all_tracks.getMovingTracks())
        matching_tracks = filter (lambda t : new_area.collidepoint(t), tracks)            
        possible = filter(lambda x: x.rect.colliderect(new_area), self.objects)
        if len(matching_tracks) == 0:
            # no feature points means we would have a hard time tracking
            pass
        elif len(possible) == 0:
            self.objects.append(Object(self.next_id, self, new_area))
            self.next_id += 1
        elif len(possible) == 1:
            p = possible[0]
            p.addRect(new_area)
        else:
            #hits multiple existing tracked items, ignore
            pass
    
    def updateFrame(self, img):
        h, w = img.shape[:2]
        img_rect = PyRect(0,0,w,h)
        self.objects = filter(lambda x : not x.killMe(img_rect), self.objects)
        [o.updateFrame() for o in self.objects]

class Object:
    tooYoung = 10
    def __init__(self, my_id, manager, rect):
        self.id = my_id
        self.manager = manager
        self.color = np.random.randint(0,255,3)
        self.rect = rect
        self.age = 0
        self.k_center = Kalman2D(x=self.rect.centerx, y=self.rect.centery)
        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )       
        self.rects = []
        self.point_tracks = []
        self.updates = 0
        self.updatePoints()
        self.notify("Created %s" % self.rect)
        self.kalman_used = False
        self.history = []

    def killMe(self, img_rect):
        if not self.rect.colliderect(img_rect):
            self.notify("No longer in frame")
            return True
        elif self.age >= Object.tooYoung and self.updates < Object.tooYoung / 2:
            self.notify("Aborted, Age %d Updates %d" % (self.age, self.updates))
            return True
        else:
            return False

    def notify(self, msg):
        print "(Frame %04d) ID %03d : %s" % (self.manager.tracker.frame_index, self.id, msg)

    def draw(self, img):
        if self.age <= Object.tooYoung:
            return
        if self.kalman_used:
            text_color = (0,0,255)
        else:
            text_color = (255,255,255)

        for p in self.point_tracks:
            cv2.circle(img, p.current(), 3, self.color, -1)

        cv2.rectangle(img, self.rect.topleft, self.rect.bottomright,
                      self.color) 
        cv2.putText(img, "%d" % self.id,
                    self.rect.bottomleft, cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    def removePointTrack(self, pt):
        self.point_tracks = filter(lambda p: p != pt, self.point_tracks)

    def updatePoints(self):
        tracks = self.manager.tracker.all_tracks.getMovingTracks()
        self.point_tracks = filter(lambda t : self.rect.collidepoint(t.current()), tracks)
        map(lambda pt : pt.addDeathObserver(self.removePointTrack), self.point_tracks)
        self.notify("Watching %d points" % (len(self.point_tracks)))

    def determineRect(self):
        find = map(lambda p: p.current(), self.point_tracks)
        updated = False
        perfect = None
        if len(find) > 0:
            search = np.array(find, dtype=np.int32).reshape(-1, 1, 2)
            x,y,w,h = cv2.boundingRect(search)
            perfect = PyRect(x,y,w,h)

        for (index, new_rect) in enumerate(self.rects):
            old_rect = self.rect
            proposed_rect = self.mergeRect(old_rect, new_rect)

            all_rects = [("new", new_rect), ("proposed", proposed_rect)]
            matched = [len(filter(lambda p : p == True, [r[1].collidepoint(x) for x in find])) for r in all_rects]
            best_fit = max(zip(all_rects, matched), key=lambda m : m[1])

            percent = 0
            if len(find) > 0:
                percent = best_fit[1] / len(find)

            best_fit = best_fit[0]
            if best_fit[1] != old_rect and percent > 0.70:
                self.rect = best_fit[1]
                updated = True

        self.rects = []

        if updated:
            self.updates += 1

        return updated

    def updateFrame(self):
        self.age += 1
        updated = self.determineRect()
        if not updated:
            self.k_center.update_none()
            predict = self.k_center.getPredicted()
            self.rect.center = tuple(predict)
        else:
            self.k_center.update(self.rect.center[0], self.rect.center[1])
        self.updatePoints()
        self.kalman_used = not updated
        self.cor = None
        self.history.append(self.rect.copy())
        if len(self.history) > 10:
            del self.history[0]
        
       
    def mergeRect(self, old_rect, new_rect):
        merged_rect = PyRect(0,0,0,0)
        merged_rect.size = (np.array(old_rect.size) - np.array(new_rect.size)) / 2 + new_rect.size
        merged_rect.center =(np.array(old_rect.center) - np.array(new_rect.center)) / 2 + np.array(new_rect.center)
        return merged_rect

    def addRect(self, new_rect):
        self.rects.append(new_rect);
                        
class Tracking:
    LEARNING_RATE = 0.003
    
    def __init__(self):
        self.oldframes = []
        self.refresh_interval = 1
        self.frame_index = 0
        self.history = 100
        self.motion_history = None
        self.objmanager = ObjectManager(self)
        self.all_tracks = AllTracks()
        self.fgmask = None
        self.fgbg = cv2.BackgroundSubtractorMOG2()
        self.cur_frame = None
        self.prev_frame = None
        self.flow = None
        self.centers = None

    def highlightMotion(self, img):
        circle_radius = 3
        motion_img = np.zeros_like(self.cur_frame.frame)
        #motion_tracks = self.all_tracks.getMovingTracks()
        motion_tracks = filter(lambda x: len(x) >= 10, self.all_tracks.getTracks())
        motion_tracks = filter(lambda x: dist(x[-1], x[0]) > 1, motion_tracks)
        
        for t in motion_tracks:
            p = map(int, t[-1])
            cv2.circle(motion_img, (p[0], p[1]), circle_radius, (255,255,255), -1)

        return motion_img

    def optFlowMotionMask(self, tracked_points):
        frame_diff = cv2.absdiff(self.cur_frame.frame, self.prev_frame.frame) 
        frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        #ret, frame_diff = cv2.threshold(frame_diff, 1, 255, cv2.THRESH_BINARY)

        flow = cv2.calcOpticalFlowFarneback(self.prev_frame.gray, self.cur_frame.gray,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        optical_flow = np.zeros((self.h,self.w,1), np.uint8)
        optical_flow[:,:,0] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        #optical_flow[:,:,0] = np.clip(mag * 255, 0, 255)
        ret, optical_flow = cv2.threshold(optical_flow, 30, 255, cv2.THRESH_BINARY)

        mask = self.fgmask.copy()
        #remove shadow
        ret, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        #remove noise
        #mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3)), iterations = 1)
        #include tracked points
        mask = cv2.bitwise_or(mask, tracked_points)
        #include silhouettes
        #mask = cv2.bitwise_and(mask, frame_diff)
        #mask all nonmoving parts of the image
        mask = cv2.bitwise_and(mask, optical_flow)
        #mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((4,4)), iterations = 1)
        #ret, mask = cv2.threshold(cv2.distanceTransform(mask, cv2.cv.CV_DIST_L2, 3).astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
        return mask

    def motionTracking(self, tracked_points):
        MHI_DURATION = 2
        MAX_TIME_DELTA = 10
        MIN_TIME_DELTA = 2
        hsv = np.zeros((self.h, self.w, 3), np.uint8)
        hsv[:,:,1] = 255
        if not self.prev_frame:
            return
        if self.motion_history == None:
            self.motion_history = np.zeros((self.h, self.w), np.float32)
        motion_history = self.motion_history

        optflow_mask = self.optFlowMotionMask(tracked_points)
        timestamp = self.frame_index
        chosen_mask = optflow_mask
        cv2.updateMotionHistory(chosen_mask, motion_history, timestamp, MHI_DURATION)
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

        for i, rect in enumerate([(0, 0, self.w, self.h)] + list(seg_bounds)):
            if i == 0:
                continue
            x, y, rw, rh = rect
            area = rw*rh
            if area < 400:
                continue
            silh_roi   = chosen_mask   [y:y+rh,x:x+rw]
            orient_roi = mg_orient     [y:y+rh,x:x+rw]
            mask_roi   = mg_mask       [y:y+rh,x:x+rw]
            mhi_roi    = motion_history[y:y+rh,x:x+rw]
            if cv2.norm(silh_roi, cv2.NORM_L1) < area*0.05:
                continue
            angle = cv2.calcGlobalOrientation(orient_roi, mask_roi, mhi_roi, timestamp, MHI_DURATION)
            color = ((255, 0, 0), (0, 0, 255))[i == 0]
            self.objmanager.addRect(rect)
            draw_motion_comp(vis, rect, angle, color)
        cv2.imshow("vis", vis)

    def processFrame(self, frame):
        self.prev_frame = self.cur_frame
        self.cur_frame = FrameData(frame);
        self.oldframes.append(self.cur_frame)
        if len(self.oldframes) > self.history:
            del self.oldframes[0]
        self.fgmask = self.fgbg.apply(self.cur_frame.frame, learningRate=Tracking.LEARNING_RATE)
        self.all_tracks.update(self.prev_frame, self.cur_frame)
        
    def seedBackground(self, img):
        self.fgmask = self.fgbg.apply(img, learningRate=Tracking.LEARNING_RATE)

    def nextFrame(self, frame):
        self.frame_index += 1
        self.processFrame(frame)
        self.updateTrackingPoints()
        self.h, self.w = self.cur_frame.frame.shape[:2]
        self.objmanager.updateFrame(self.cur_frame.frame)
        self.debug()

    def updateTrackingPoints(self):
        if self.frame_index % self.refresh_interval == 0:
            self.all_tracks.refresh(self.cur_frame)

    def debug(self):
        iters = 3
        img = self.cur_frame.frame.copy()
        cv2.putText(img, "Frame: %d" % self.frame_index,
                    (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.putText(img, "Tracks: %d" % self.all_tracks.count(),
                    (0,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        mask = self.fgmask.copy()
        motion = self.highlightMotion(img.copy())
        self.motionTracking(cv2.cvtColor(motion, cv2.COLOR_BGR2GRAY))
        #cv2.polylines(img, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
        self.objmanager.draw(img)
        cv2.imshow("Result", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    t = Tracking()
    ret = 1
    samples = 0
    cap = cv2.VideoCapture(sys.argv[1])
    while samples < 100 and ret != 0:
        ret, frame = cap.read()
        if ret == 0:
            continue
        t.seedBackground(frame)
        samples += 1

    cap = cv2.VideoCapture(sys.argv[1])
    while ret != 0:
        ret, frame = cap.read()
        if ret == 0:
            continue
        t.nextFrame(frame)
    cv2.destroyAllWindows()
