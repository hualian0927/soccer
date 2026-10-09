"""Conservative jersey-color constraints around the existing BoT-SORT engine."""

from collections import Counter, defaultdict

import cv2
import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT


def shirt_class(image, box):
    x1,y1,x2,y2 = box
    w,h = x2-x1,y2-y1
    crop = image[max(0,int(y1+.18*h)):max(0,int(y1+.5*h)),
                 max(0,int(x1+.25*w)):max(0,int(x1+.75*w))]
    if crop.size < 36:
        return 0
    hsv = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    hue,sat,val = cv2.split(hsv)
    scores = {1:np.mean((hue>=90)&(hue<=135)&(sat>75)&(val>35)),
              2:np.mean((sat<65)&(val>160)),
              3:np.mean(((hue<12)|(hue>170))&(sat>100)&(val>60)),
              4:np.mean((hue>=20)&(hue<45)&(sat>100)&(val>130))}
    label = max(scores,key=scores.get)
    return label if scores[label] > .48 else 0


class JerseyBOTSORT(BOTSORT):
    def __init__(self,*args,**kwargs):
        self.jersey_votes = defaultdict(Counter)
        super().__init__(*args,**kwargs)

    def reset(self):
        super().reset()
        self.jersey_votes.clear()

    def get_dists(self,tracks,detections):
        distances = super().get_dists(tracks,detections)
        for i,track in enumerate(tracks):
            if int(track.cls):
                self.jersey_votes[track.track_id][int(track.cls)] += 1
            votes = self.jersey_votes[track.track_id]
            dominant = votes.most_common(1)[0][0] if votes else 0
            for j,det in enumerate(detections):
                if dominant and int(det.cls) and dominant != int(det.cls):
                    distances[i,j] = 1.0
        return distances
