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
    ixmin = np.maximum(BBGT[:, 0], bb[:,0])
    iymin = np.maximum(BBGT[:, 1], bb[:,1])
    ixmax = np.minimum(BBGT[:, 2], bb[:,2])
    iymax = np.minimum(BBGT[:, 3], bb[:,3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb[:, 2] - bb[:, 0] + 1.) * (bb[:, 3] - bb[:, 1] + 1.) +
           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps

gt = np.array([[50, 10, 60, 320], [150, 100, 660, 320], [150, 10, 660, 320]])
num_gt = gt.shape[0]
r = np.array([0.3, 0.8, 0.5, 0.6, 0.4])
num_roi = r.shape[0]
gt_choice_idx = np.random.choice(num_gt, num_roi)
gt_selected = gt[gt_choice_idx]

x1, y1, x2, y2  = gt_selected[:,0], gt_selected[:,1], gt_selected[:,2], gt_selected[:,3]

area_gt = (y2-y1) * (x2-x1)
im_width, im_height = 700, 700

"""STEP 1: s1a"""
s1a = x2 - (x2-x1)/r
s1a = np.maximum(0, s1a)
"""STEP 2: s1b"""
s1b = x2 - r*(x2-x1)
# Get s1
s1 = np.random.rand(num_roi)*(s1b-s1a) + s1a

"""STEP 3: s2a"""
s2a = r*(x2-np.minimum(x1,s1)) + np.maximum(x1,s1)
"""STEP 4: s2b"""
s2b = np.minimum(x1,s1) + (x2-np.maximum(x1,s1))/r
s2b = np.minimum(im_width, s2b)
#Get s2
s2 = np.random.rand(num_roi)*(s2b-s2a) + s2a

intersection_width = np.minimum(x2,s2) - np.maximum(x1,s1)
"""STEP 5: t1a"""
intersection = (y2-y1) * intersection_width
t1a = y2 - (intersection/r-area_gt+intersection) / (s2-s1)
t1a = np.maximum(0, t1a)
"""STEP 6: t1b"""
t1b = y2 - area_gt / (intersection_width/r-s2+s1+intersection_width)
# Get t1
t1 = np.random.rand(num_roi)*(t1b-t1a) + t1a

"""STEP 7: t2a"""
t2a = (np.maximum(t1,y1)*intersection_width - r*s2*t1 + r*s1*t1 + r*area_gt + r*intersection_width*np.maximum(t1,y1)) / (intersection_width - r*s2 + r*s1 + r*intersection_width)
"""STEP 8: t2b"""
intersection = (y2-np.maximum(t1,y1)) * intersection_width
t2b = (intersection/r - area_gt + intersection) / (s2 - s1) + t1
t2b = np.minimum(im_height, t2b)
# Get t2
t2 = np.stack((t2a, t2b))
t2_choice_idx = np.random.choice(2, num_roi)
t2 = t2[t2_choice_idx, range(0,num_roi)]

roi = np.stack((s1, t1, s2, t2),1)
IoU = get_IoU(gt_selected, roi)
print(IoU)
