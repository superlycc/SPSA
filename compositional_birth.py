"""On-demand, causal multi-owner birth witnesses around a frozen Hybrid-SORT."""

import inspect
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from zef_run_tracker import build_tracker


@dataclass
class BirthConfig:
    mode: str = "transport"  # disabled, score, area, transport
    covered_threshold: float = .75
    score_threshold: float = .7
    area_threshold: float = .3
    radius: float = 6.0
    owner_fraction: float = .15
    entropy_threshold: float = .45
    minimum_points: int = 8
    minimum_history: int = 5


def composition_statistics(points,labels,foreground,radius):
    labels = np.asarray(labels)
    if len(points) == 0 or len(foreground) == 0:
        return dict(coverage=0.0,shares=[],entropy=0.0)
    distance,indices = cKDTree(points).query(foreground,k=1)
    covered = distance <= radius
    unique = np.unique(labels)
    if not covered.any():
        return dict(coverage=0.0,shares=[0.0]*len(unique),entropy=0.0)
    assigned = labels[indices[covered]]
    # Only sources explaining this candidate belong in its entropy denominator.
    local = np.unique(assigned)
    shares = np.array([(assigned == owner).mean() for owner in local])
    positive = shares[shares > 0]
    entropy = float(-(positive*np.log(positive)).sum()/np.log(len(local))) if len(local) > 1 else 0.0
    return dict(coverage=float(covered.mean()),shares=shares.tolist(),entropy=entropy)


def veto_composition(statistics,config):
    return (statistics["coverage"] >= config.covered_threshold and
            sum(s >= config.owner_fraction for s in statistics["shares"]) >= 2 and
            statistics["entropy"] >= config.entropy_threshold)


def rectangle_coverage(candidate,boxes):
    """Exact rectangle-union area inside a candidate; never an instance mask."""
    x1,y1,x2,y2 = candidate[:4]
    clipped = [[max(x1,b[0]),max(y1,b[1]),min(x2,b[2]),min(y2,b[3])] for b in boxes]
    clipped = [b for b in clipped if b[2] > b[0] and b[3] > b[1]]
    edges = sorted({v for b in clipped for v in (b[0],b[2])})
    total = 0.
    for left,right in zip(edges,edges[1:]):
        intervals = sorted((b[1],b[3]) for b in clipped if b[0] < right and b[2] > left)
        end = -np.inf
        length = 0.
        for top,bottom in intervals:
            length += max(0,bottom-max(top,end))
            end = max(end,bottom)
        total += (right-left)*length
    return float(total/max((x2-x1)*(y2-y1),1e-12))


class CompositionalBirth:
    def __init__(self,config=None):
        self.config = config or BirthConfig()
        self.host,self.host_parameters = build_tracker("HybridSORT")
        self.host.iou_threshold = .2
        self.module = inspect.getmodule(type(self.host))
        self.frame = 0
        self.events = []
        self.counters = dict(birth_requests=0,vetoed=0,witnesses_available=0,
                             image_pairs_decoded=0,transport_points=0)

    def owner_tracks(self):
        owners = []
        for track in self.host.trackers:
            if (track.id in self.previous_boxes and track.time_since_update == 0 and
                    track.hits >= self.config.minimum_history and
                    track.last_observation[4] > .5):
                owners.append((track.id,self.previous_boxes[track.id],
                               track.last_observation[:4].copy()))
        return owners

    @staticmethod
    def gray(path):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode observed image {path}")
        return cv2.cvtColor(cv2.resize(image,None,fx=.5,fy=.5),cv2.COLOR_BGR2GRAY)

    @staticmethod
    def foreground(gray,box):
        h,w = gray.shape
        x1,y1,x2,y2 = np.asarray(box[:4]*.5,dtype=int)
        x1,x2 = np.clip([x1,x2],0,w)
        y1,y2 = np.clip([y1,y2],0,h)
        roi = gray[y1:y2,x1:x2]
        if roi.size == 0:
            return np.empty((0,2),dtype=np.float32)
        ys,xs = np.where(roi < np.median(roi)-6)
        return np.column_stack((xs+x1,ys+y1)).astype(np.float32)

    def witness(self,candidate,owners):
        if self.images is None:
            self.images = (self.gray(self.previous_image),self.gray(self.current_image))
            self.counters["image_pairs_decoded"] += 1
        previous,current = self.images
        all_boxes = np.array([b*.5 for _,b,_ in owners])
        points,labels = [],[]
        for index,(identity,old,_) in enumerate(owners):
            sampled = self.foreground(previous,old)
            if len(sampled) < self.config.minimum_points:
                continue
            if len(sampled) > 160:
                sampled = sampled[np.linspace(0,len(sampled)-1,160,dtype=int)]
            centers = (all_boxes[:,:2]+all_boxes[:,2:])/2
            sizes = np.maximum(all_boxes[:,2:]-all_boxes[:,:2],1)
            distance = np.sum(((sampled[:,None,:]-centers[None,:,:])/sizes[None,:,:])**2,axis=2)
            inside = ((sampled[:,None,:] >= all_boxes[None,:,:2]).all(axis=2)&
                      (sampled[:,None,:] <= all_boxes[None,:,2:]).all(axis=2))
            distance[~inside] = np.inf
            sampled = sampled[np.argmin(distance,axis=1) == index]
            if len(sampled) < self.config.minimum_points:
                continue
            source = sampled.reshape(-1,1,2)
            target,valid_a,_ = cv2.calcOpticalFlowPyrLK(
                previous,current,source,None,winSize=(21,21),maxLevel=3)
            if target is None:
                continue
            returned,valid_b,_ = cv2.calcOpticalFlowPyrLK(
                current,previous,target,None,winSize=(21,21),maxLevel=3)
            if returned is None:
                continue
            valid = ((valid_a.ravel() == 1)&(valid_b.ravel() == 1)&
                     (np.linalg.norm((returned-source).reshape(-1,2),axis=1) < 1.5))
            transported = target.reshape(-1,2)[valid]
            if len(transported) < self.config.minimum_points:
                continue
            points.extend(transported)
            labels.extend([identity]*len(transported))
        if len(set(labels)) < 2:
            return dict(coverage=0.0,shares=[],entropy=0.0,valid_owners=len(set(labels)))
        self.counters["witnesses_available"] += 1
        self.counters["transport_points"] += len(points)
        foreground = self.foreground(current,candidate)
        if len(foreground) < 12:
            return dict(coverage=0.0,shares=[],entropy=0.0,valid_owners=len(set(labels)))
        statistics = composition_statistics(np.asarray(points),np.asarray(labels),foreground,self.config.radius)
        statistics["valid_owners"] = len(set(labels))
        return statistics

    def certify(self,candidate):
        if self.config.mode == "disabled" or self.frame <= self.host.min_hits:
            return False,{}
        owners = self.owner_tracks()
        if self.config.mode == "score":
            return candidate[4] < self.config.score_threshold,dict(score=float(candidate[4]))
        if self.config.mode == "area":
            coverage = rectangle_coverage(candidate,[b for _,_,b in owners])
            return coverage > self.config.area_threshold,dict(rectangle_coverage=coverage)
        if len(owners) < 2:
            return False,dict(valid_owners=len(owners))
        statistics = self.witness(candidate,owners)
        return veto_composition(statistics,self.config),statistics

    def update(self,detections,info,current_image,previous_image):
        self.frame += 1
        self.current_image,self.previous_image = current_image,previous_image
        self.images = None
        self.previous_boxes = {
            t.id:t.last_observation[:4].copy() for t in self.host.trackers
            if t.time_since_update == 0 and t.last_observation[4] > .5}
        original_init = self.module.KalmanBoxTracker.__init__
        rejected = set()

        def initialize(track,bbox,*args,**kwargs):
            veto,statistics = self.certify(bbox)
            original_init(track,bbox,*args,**kwargs)
            self.counters["birth_requests"] += 1
            if veto:
                rejected.add(track.id)
                self.counters["vetoed"] += 1
            self.events.append(dict(frame=self.frame,track_id=track.id+1,veto=veto,
                                    detection=np.asarray(bbox).tolist(),statistics=statistics))

        self.module.KalmanBoxTracker.__init__ = initialize
        try:
            result = self.host.update(detections.copy(),info,info)
        finally:
            self.module.KalmanBoxTracker.__init__ = original_init
        # Tentative construction supplies an ID but rejected states never persist
        # or appear in output; existing host associations and histories stay intact.
        self.host.trackers = [t for t in self.host.trackers if t.id not in rejected]
        if len(result):
            result = result[~np.isin(result[:,4],[i+1 for i in rejected])]
        return result
