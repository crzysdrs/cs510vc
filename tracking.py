#!/usr/bin/env python
from __future__ import division
import numpy as np
import cv2
from kalman2d import Kalman2D
import glob
import sys
import math
from pygame import Rect as PyRect
import argparse
import itertools

def draw_motion_comp(vis, (x, y, w, h), angle, color):
    """
    Draw the motion templating information into the specified image.
    """
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0))
    r = min(w//2, h//2)
    cx, cy = x+w//2, y+h//2
    angle = angle*np.pi//180
    cv2.circle(vis, (cx, cy), r, color, 3)
    cv2.line(vis, (cx, cy), (int(cx+np.cos(angle)*r), int(cy+np.sin(angle)*r)), color, 3)

def partition(pred, items):
    """
    Partition the list items into two lists based on the predicate function.
    """
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
    """
    AllTracks is a container class for the point tracking.
    By grouping all this work here, we can more easily interact with opencv
    via numpy to compute all information for points one time per frame.
    """
    MOVING_SPEED = 2
    def __init__(self):
        self.point_tracks = []
        self.feature_params = dict(
            maxCorners = 5000, 
            qualityLevel = 0.10,
            minDistance = 3,
            blockSize = 3
        )
        self.refreshes = 0

    def getTracks(self):
        return map(lambda x : x.history(), self.point_tracks)

    def getPointTracks(self):
        return self.point_tracks
    
    def getMovingTracks(self, speed=None):
        if not speed:
            speed = AllTracks.MOVING_SPEED
        (long_tracks, short_tracks) = partition(split_tracks, self.point_tracks)
        motion_tracks = filter(lambda x: x.dist > speed, long_tracks)
        return motion_tracks

    def count(self):
        return len(self.point_tracks)
    
    def reduceDensity(self):                
        intified = map(lambda x : (tuple(map(int, x.current())), len(x.history()), x), self.point_tracks)
        intified = sorted(intified, key=lambda x: x[0])
        intified = itertools.groupby(intified, lambda x: x[0])
            
        low_density = []
        for pos, group in intified:
            last_entry = None
            for g in group:
                if last_entry:
                    last_entry.missing()
                last_entry = g[2]
            low_density.append(last_entry)

        self.point_tracks = low_density

    def refresh(self, cur_frame):
        (long_tracks, short_tracks) = partition(split_tracks, self.point_tracks)
        motion_tracks = filter(lambda x: x.dist > AllTracks.MOVING_SPEED, long_tracks)
        points = cv2.goodFeaturesToTrack(cur_frame.gray, **self.feature_params)
        self.point_tracks = short_tracks + motion_tracks + [PointTrack(p) for p in points.reshape(-1,2)]

        self.refreshes += 1
        if self.refreshes > 10:
            self.reduceDensity()
            self.refreshes = 0

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

        curr = np.array(map(lambda x : x.current(), self.point_tracks))
        last = np.array(map(lambda x : x.last(), self.point_tracks))
        dist = np.linalg.norm(curr-last, axis=1)
        count = (np.array(map(lambda x: len(x.history()), self.point_tracks)))
        vel = (curr - last)
        vel[::,0] = vel[::,0] / count
        vel[::,1] = vel[::,1] / count
        for v, d, pt in zip(vel, dist, self.point_tracks):
            pt.dist = d
            pt.vel = v

class PointTrack:
    """
    A point track is simply a history for a given point as it moves through frames.
    This allows us to compute velocity, or movement for a given point, as well
    as use it to track objects.
    """
    next_id = 0
    max_len = 10
    def __init__(self, point):
        self.id = PointTrack.next_id
        PointTrack.next_id += 1
        self.track = [point]
        self.death_notify = []
        self.dist = 0
        self.vel = 0

    def addDeathObserver(self, observer):
        self.death_notify = filter(lambda x : observer != x, self.death_notify)
        self.death_notify.append(observer)

    def addHistory(self, new_point):
        self.track.append(new_point)
        if len(self.track) > PointTrack.max_len:
            del self.track[0]
        
    def current(self):
        return self.track[-1]

    def dist(self):
        return self.dist

    def last(self):
        return self.track[0]

    def missing(self): 
        for n in self.death_notify:
            n(self)

    def velocity(self):
        return self.vel

    def history(self):
        return self.track


class FrameData:
    """
    Wrapper for frames, to keep track of gray scale image and not recompute it unneccesarily.
    """
    def __init__(self, frame):
        self.frame = frame
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.track_points = None

class ObjectManager:
    """
    Keeps track of all the objects in the scene.
    """
    def __init__(self, tracker):
        self.objects = []
        self.tracker = tracker
        self.next_id = 0

    def draw(self, img):
        [o.draw(img) for o in self.objects]

    def addRect(self, area, angle):
        #pyRect (left, top, width, height)
        new_area = PyRect(area[0], area[1], area[2], area[3])
        center = new_area.center
        shrink = new_area.copy()
        shrink.size = tuple(np.array(new_area.size) * 0.90)
        shrink.center = center
        tracks = map(lambda x : x.current(), self.tracker.all_tracks.getMovingTracks())
        matching_tracks = filter (lambda t : new_area.collidepoint(t), tracks)            
        possible = filter(lambda x: x.rect.colliderect(shrink), self.objects)
        if len(matching_tracks) <= 1:
            return # too few tracks
        elif len(possible) == 0:
            self.objects.append(Object(self.next_id, self, new_area))
            self.next_id += 1
        elif len(possible) == 1:
            p = possible[0]
            p.addRect(new_area, angle)
        else:
            #hits multiple existing tracked items, ignore
            pass
    
    def updateFrame(self, img):
        h, w = img.shape[:2]
        img_rect = PyRect(0,0,w,h)
        self.objects = filter(lambda x : not x.killMe(img_rect), self.objects)
        [o.updateFrame() for o in self.objects]

class Object:
    """
    Represents a moving object in the scene.
    """
    tooYoung = 10
    def __init__(self, my_id, manager, rect):
        self.manager = manager
        self.debug = self.manager.tracker.args.debug
        self.rect = rect
        self.id = my_id
        self.color = np.random.randint(0,255,3)
        self.notify("Created %s" % self.rect)
        self.age = 0
        self.k_center = Kalman2D(x=self.rect.centerx, y=self.rect.centery)
        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )       
        self.rects = []
        self.point_tracks = []
        self.updates = 0
        self.updatePoints()
        self.history = [rect]
        self.method = ""

    def killMe(self, img_rect):
        """
        Notify the objectmanager that the object is no longer worth tracking.
        """
        if not self.rect.colliderect(img_rect):
            self.notify("No longer in frame")
            return True
        elif self.age == Object.tooYoung and (self.updates < Object.tooYoung / 2 or len(self.point_tracks) <= 1):
            self.notify("Aborted, Age %d Updates %d" % (self.age, self.updates))
            return True
        else:
            return False

    def notify(self, msg):
        """ Debug Routine for keeping track of info related to this object"""
        if self.debug:
            print "(Frame %04d) ID %03d : %s" % (self.manager.tracker.frame_index, self.id, msg)

    def draw(self, img):
        if self.method == "kalman":
            text_color = (0,0,255)
        elif self.method == "velocity":
            text_color = (0,255,0)
        elif self.method == "motion":
            text_color = (255,255,255)
        else:
            text_color = (0,0,0)

        draw = True
        if self.age <= Object.tooYoung:
            draw = self.manager.tracker.frame_index % 2 == 0 and self.debug

        if draw:
            if self.debug:
                for p in self.point_tracks:
                    cv2.circle(img, p.current(), 3, self.color, -1)

            cv2.rectangle(img, self.rect.topleft, self.rect.bottomright,
                          self.color) 

            cv2.putText(img, "%d" % self.id,
                    self.rect.bottomleft, cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    def removePointTrack(self, pt):
        self.point_tracks = filter(lambda p: p != pt, self.point_tracks)

    def updatePoints(self):
        """ Find all the trackable points in the given object """
        tracks = self.manager.tracker.all_tracks.getMovingTracks()
        self.point_tracks = filter(lambda t : self.rect.collidepoint(t.current()), tracks)
        map(lambda pt : pt.addDeathObserver(self.removePointTrack), self.point_tracks)
        self.notify("Watching %d points" % (len(self.point_tracks)))

    def determineRect(self):
        """ Determine which motion rect in the scene correlates to this object."""
        dist = np.array(self.history[-1].center) - np.array(self.history[0].center)
        angle = math.atan2(dist[1], dist[0]) * 180 / np.pi

        if len(self.rects) > 1:
            self.rects = filter(lambda x : abs((x[1] - angle) % 360) < 120, self.rects)

        self.rects = map(lambda x : x[0], self.rects)

        if len(self.rects) == 0:
            return
        find = map(lambda p: p.current(), self.point_tracks)
        updated = False
        perfect = None
        
        new_rect = self.rects[0].unionall(self.rects[1:])
        old_rect = self.rect
        proposed_rect = self.mergeRect(old_rect, new_rect)
        if proposed_rect.size[0] * proposed_rect.size[1] > Tracking.MAX_SIZE:
            self.rects = []
            return False

        all_rects = [("proposed", proposed_rect), ("new_rect", new_rect)]

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
            self.rect.center = self.k_center.getPredicted()
            self.method = "kalman"
        else:
            self.k_center.update(self.rect.center[0], self.rect.center[1])
            self.rect.center = self.k_center.getCorrected()
            self.method = "motion"

        self.updatePoints()
        
        self.cor = None
        self.history.append(self.rect.copy())
        if len(self.history) > 10:
            del self.history[0]
        
       
    def mergeRect(self, old_rect, new_rect):
        merged_rect = PyRect(0,0,0,0)
        #favors retaining the old size
        merged_rect.size = (np.array(new_rect.size) - np.array(old_rect.size)) / 10 + old_rect.size
        #favors moving the center to match the new center
        merged_rect.center = (np.array(new_rect.center) - np.array(old_rect.center)) / 1.5 + np.array(old_rect.center)
        return merged_rect

    def addRect(self, new_rect, angle):
        self.rects.append((new_rect, angle))
                        
class Tracking:
    """
    The class that handles all the frame reading and visualization of the debug info"""
    LEARNING_RATE = 0.003
    MAX_SIZE = 80 * 80
    MASK_SHADOW = False

    def __init__(self, args, cap):
        Tracking.MAX_SIZE = args.max_size
        Tracking.LEARNING_RATE = args.learning_rate
        Tracking.SHADOW = args.mask_shadow
        self.args = args
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
        self.total_frames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

        frame_size = (cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH),
                      cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        frame_size = tuple(map(int, frame_size))
        if args.output:
            self.video = cv2.VideoWriter()
            self.video.open(filename=self.args.output,
                            fourcc=int(cap.get(cv2.cv.CV_CAP_PROP_FOURCC)),
                            fps=int(cap.get(cv2.cv.CV_CAP_PROP_FPS)),
                            frameSize=frame_size,
                            isColor=True
                        )
        else:
            self.video = None

    def highlightMotion(self, img):
        circle_radius = 3
        motion_img = np.zeros_like(self.cur_frame.frame)
        motion_tracks = filter(lambda x: len(x.history()) >= 5 and x.dist > 2,
                               self.all_tracks.getPointTracks())
        motion_tracks = map(lambda x: x.current(), motion_tracks)

        for t in motion_tracks:
            p = map(int, t)
            cv2.circle(motion_img, (p[0], p[1]), circle_radius, (255,255,255), -1)
            if self.args.debug:
                cv2.circle(img, (p[0], p[1]), circle_radius, (255,255,255), -1)

        return motion_img

    def optFlowMotionMask(self, tracked_points):
        """ Build a motion mask """
        flow = cv2.calcOpticalFlowFarneback(self.prev_frame.gray, self.cur_frame.gray,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        optical_flow = np.zeros((self.h,self.w,1), np.uint8)
        optical_flow[:,:,0] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        ret, optical_flow = cv2.threshold(optical_flow, 30, 255, cv2.THRESH_BINARY)

        mask = self.fgmask.copy()
        #remove shadow
        if Tracking.MASK_SHADOW:
            ret, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        #include tracked points
        mask = cv2.bitwise_or(mask, tracked_points)
        #mask all nonmoving parts of the image
        mask = cv2.bitwise_and(mask, optical_flow)
        return mask

    def motionTracking(self, tracked_points):
        """ Compute the motion template of the image and potentially create objects. """
        MHI_DURATION = 3
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
        
        vis = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        for i, rect in enumerate([(0, 0, self.w, self.h)] + list(seg_bounds)):
            if i == 0:
                continue
            x, y, rw, rh = rect
            area = rw*rh
            if area < 400:
                continue
            elif area > Tracking.MAX_SIZE:
                continue
            silh_roi   = chosen_mask   [y:y+rh,x:x+rw]
            orient_roi = mg_orient     [y:y+rh,x:x+rw]
            mask_roi   = mg_mask       [y:y+rh,x:x+rw]
            mhi_roi    = motion_history[y:y+rh,x:x+rw]
            if cv2.norm(silh_roi, cv2.NORM_L1) < area*0.05:
                continue
            angle = cv2.calcGlobalOrientation(orient_roi, mask_roi, mhi_roi, timestamp, MHI_DURATION)
            color = ((255, 0, 0), (0, 0, 255))[i == 0]
            self.objmanager.addRect(rect, angle)
            draw_motion_comp(vis, rect, angle, color)
        
        if self.args.debug:
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
        print "***** Input %s Output %s Frame %04d/%04d %02d%%" % (self.args.video,
                                                                   self.args.output,
                                                                   self.frame_index,
                                                                   self.total_frames,
                                                                   self.frame_index / self.total_frames * 100)
        self.processFrame(frame)
        self.updateTrackingPoints()
        self.h, self.w = self.cur_frame.frame.shape[:2]
        img = self.cur_frame.frame.copy()
        motion = self.highlightMotion(img)
        self.motionTracking(cv2.cvtColor(motion, cv2.COLOR_BGR2GRAY))
        self.objmanager.updateFrame(self.cur_frame.frame)
        self.objmanager.draw(img)
        if self.args.debug:
            self.debug(img)
            cv2.imshow("Result", img)
        elif not self.video:
            cv2.imshow("Result", img)

        if self.video:
            self.video.write(img)

    def done(self):
        if self.video:
            self.video.release()

    def updateTrackingPoints(self):
        if self.frame_index % self.refresh_interval == 0:
            self.all_tracks.refresh(self.cur_frame)

    def debug(self, img):
        """ Put any useful debug stuff here """
        cv2.putText(img, "Frame: %d" % self.frame_index,
                    (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.putText(img, "Tracks: %d" % self.all_tracks.count(),
                    (0,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.waitKey(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Motion Tracking')
    parser.add_argument('video', help='video to process')
    parser.add_argument('--max_size', default=80*80, type=int, help='maximum size of possible motion')
    parser.add_argument('--learning_rate', default=0.003, type=float, help='background learning rate')
    parser.add_argument('--mask_shadow', default=False, action='store_true', help='use the shadow mask')
    parser.add_argument('--debug', default=False, action='store_true', help='enable debugging')
    parser.add_argument('--output', help='output to video file')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    t = Tracking(args, cap)
    ret = 1
    samples = 0
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

    t.done()
    cv2.destroyAllWindows()
