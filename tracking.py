#!/usr/bin/env python
import numpy as np
import cv2
from kalman2d import Kalman2D
import glob
import sys
from pygame import Rect as PyRect


lk_params = dict( 
    winSize  = (10, 10), 
    maxLevel = 5, 
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)


def followPoints(prev_frame, cur_frame, last_points):
    if len(last_points) == 0:
        return []

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
    new_tracks = []
    corresponding = []
    for old_p, (x,y), range_ok in zip(last_points, found_points.reshape(-1,2), within_range):
        corresponding.append((range_ok, old_p, (x,y)))
    return corresponding

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
    def __init__(self, tracker):
        self.tracked = []
        self.tracker = tracker

    def draw(self, img):
        [x.draw(img) for x in self.tracked]

    def addRect(self, area):
        #pyRect (left, top, width, height)
        new_area = PyRect(area[0], area[1], area[2], area[3])
        tracks = self.tracker.tracks
        matching_tracks = filter (lambda tr : new_area.collidepoint(tr[0]), tracks)
               
        possible = filter(lambda x: x.rect.colliderect(new_area), self.tracked)
        if len(matching_tracks) == 0:
            # no feature points means we would have a hard time tracking
            pass
        elif len(possible) == 0:
            self.tracked.append(Tracked(self, new_area, [tr[-1] for tr in matching_tracks]))
        elif len(possible) == 1:
            p = possible[0]
            #p.updateRect(new_area)
        else:
            #hits multiple existing tracked items, ignore
            pass
    
    def updateFrame(self, img, img_hsv):
        [x.updateFrame() for x in self.tracked]

class Tracked:
    def __init__(self, manager, rect, points):
        self.manager = manager
        self.color = np.random.randint(0,255,3)
        self.rect = rect
        self.age = 0
        self.points = points
        self.center = self.updateCenter()
        self.k_center = Kalman2D(x=self.rect.centerx, y=self.rect.centery)
        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )       
        self.updated = False

    def updateCenter(self):
        if len(self.points) > 1:
            p = fold(lambda p, q : np.add(p, q), self.points, (0,0))
            p = p / len(self.points)
            new_center = p
        elif len(self.points) == 1:
            new_center = self.points[0]
        else:            
            new_center = self.center
        return new_center

    def draw(self, img):
        if len(self.points) > 0:
            cv2.rectangle(img, self.rect.topleft, self.rect.bottomright,
                          self.color)

    def updateFrame(self):
        self.age += 1
        match = followPoints(self.manager.tracker.prev_frame,
                             self.manager.tracker.cur_frame,
                             np.float32(self.points).reshape(-1, 2))

        #keep found points
        self.points = [p[2] for p in filter(lambda p : p[0], match)]
        old_center = self.center
        old_rect = self.rect.copy()
        self.center = self.updateCenter()
        if len(self.points) == 0:
            self.k_center.update_none()
            predict = self.k_center.getPredicted()
            self.rect.center = tuple(predict)
        else:
            #self.rect.center = np.add(np.subtract(old_center, old_rect.center), self.center)
            self.rect.center = self.center
            self.k_center.update(self.rect.center[0], self.rect.center[1])

    def updateRect(self, rect):
        self.rect = rect
        self.k_center.update(self.rect.centerx, self.rect.centery)
        self.updated = True
        

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
        self.trackmanager = TrackManager(self)

        
        self.feature_params = dict(
            maxCorners = 5000, 
            qualityLevel = 0.10,
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
        circle_radius = 3
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
                    cv2.circle(motion_img, (p[0], p[1]), circle_radius, (255,255,255), -1)
                    cv2.circle(img, (p[0], p[1]), circle_radius, map(int, c), -1)
                    culledtracks.append(t)
            except IndexError:
                pass

        self.tracks = tracks + culledtracks
        return motion_img

    def optFlowMotionMask(self, tracked_points):
        frame_diff = cv2.absdiff(self.cur_frame.frame, self.prev_frame.frame) 
        frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        #ret, frame_diff = cv2.threshold(frame_diff, 1, 255, cv2.THRESH_BINARY)
        cv2.imshow("frame_diff", frame_diff)

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
        cv2.imshow("optflow_mask", optflow_mask)
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
                **lk_params
            )            
            refound_points, st1, _ = cv2.calcOpticalFlowPyrLK(
                self.cur_frame.gray,
                self.prev_frame.gray,
                found_points,
                None, 
                **lk_params
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
        
    def seedBackground(self, img):
        self.fgmask = self.fgbg.apply(img, learningRate=0.003)

    def nextFrame(self, frame):
        self.frame_index += 1
        self.processFrame(frame)
        self.updateTrackingPoints()
        self.h, self.w = self.cur_frame.frame.shape[:2]
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
