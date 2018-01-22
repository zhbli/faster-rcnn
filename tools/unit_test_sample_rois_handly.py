import numpy as np
"""vars:
r: IoU ratio we want.
roi: [s1, t1, s2, t2]
s1a~t2b: roi boundary
s1a <= s1 <= s1b
s2a <= s2 <= s2b
t1a <= t1 <= t1b
t2 = t2a or t2b
"""
def get_IoU(BBGT, bb):
    # compute overlaps
    # intersection
    BBGT = BBGT.reshape(-1, 4)
    ixmin = np.maximum(BBGT[:, 0], bb[0])
    iymin = np.maximum(BBGT[:, 1], bb[1])
    ixmax = np.minimum(BBGT[:, 2], bb[2])
    iymax = np.minimum(BBGT[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

    overlaps = inters / uni
    ovmax = np.max(overlaps)
    return ovmax

gt = np.array([50, 50, 100, 100])
x1, y1, x2, y2  = gt[0], gt[1], gt[2], gt[3]
r = 0.3
area_gt = (y2-y1) * (x2-x1)
im_width, im_height = 200, 200

"""STEP 1: s1a"""
s1a = x2 - (x2-x1)/r
"""STEP 2: s1b"""
s1b = x2 - r*(x2-x1)
# Get s1
s1 = np.random.rand()*(s1b-s1a) + s1a
s1 = max(0, s1)

"""STEP 3: s2a"""
s2a = r*(x2-min(x1,s1)) + max(x1,s1)
"""STEP 4: s2b"""
s2b = min(x1,s1) + (x2-max(x1,s1))/r
#Get s2
s2 = np.random.rand()*(s2b-s2a) + s2a
s2 = min(s2, im_width)

intersection_width = min(x2,s2) - max(x1,s1)
"""STEP 5: t1a"""
intersection = (y2-y1) * intersection_width
t1a = y2 - (intersection/r-area_gt+intersection) / (s2-s1)
"""STEP 6: t1b"""
t1b = y2 - area_gt / (intersection_width/r-s2+s1+intersection_width)
# Get t1
t1 = np.random.rand()*(t1b-t1a) + t1a
t1 = max(0, t1)

"""STEP 7: t2a"""
t2a = (max(t1,y1)*intersection_width - r*s2*t1 + r*s1*t1 + r*area_gt + r*intersection_width*max(t1,y1)) / (intersection_width - r*s2 + r*s1 + r*intersection_width)
"""STEP 8: t2b"""
intersection = (y2-max(t1,y1)) * intersection_width
t2b = (intersection/r - area_gt + intersection) / (s2 - s1) + t1
# Get t2
# t2 = np.random.rand()*(t2b-t2a) + t2a
t2 = np.random.choice([t2a, t2b], 1)
t2 = min(t2, im_height)

roi = np.array([s1, t1, s2, t2])
IoU = get_IoU(gt, roi)
print(IoU)
