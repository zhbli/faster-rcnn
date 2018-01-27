import _init_paths
from datasets.pascal_voc import pascal_voc
import cv2
import numpy as np
import pickle

def on_mouse(event, x, y, flags, param):
    global img, point1, point2, current_gts
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        point1 = (x,y)
        cv2.circle(img2, point1, 10, (0,255,0), 2)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        cv2.rectangle(img2, point1, (x,y), (255,0,0), 2)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP: # if left button is up
        point2 = (x,y)
        x1 = min(point1[0],point2[0])
        y1 = min(point1[1],point2[1])
        x2 = max(point1[0], point2[0])
        y2 = max(point1[1], point2[1])
        current_gts = np.append(current_gts, np.asarray([x1, y1, x2, y2]).reshape(1, 4), axis=0)
        for k in range(current_gts.shape[0]):
            cv2.rectangle(img2, (current_gts[k,0], current_gts[k,1]), (current_gts[k,2], current_gts[k,3]), (0, 0, 255), 2)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        current_gts = np.zeros([0, 4])
        cv2.imshow('image', img2)

def main():
    # declare vars
    global current_gts
    whole_gts = {}
    #

    # load img_set
    img_set = pascal_voc('zhbli_pottedplant_untruncated_easy_trainval', '2007')
    #

    # handle every img
    for i in range(img_set.num_images):
        ## declare vars
        global current_gts
        current_gts = np.zeros([0, 4]).astype(np.int)
        ##
        img_name = '/data/zhbli/VOCdevkit/VOC2007/JPEGImages/{:s}.jpg'.format(img_set.image_index[i])
        pottedplant_gt = img_set.roidb[i]['boxes']
        global img
        img = cv2.imread(img_name)
        assert img is not None, "fail to load img"
        for j in range(len(pottedplant_gt)):
            cv2.rectangle(img, (pottedplant_gt[j,0], pottedplant_gt[j,1]), (pottedplant_gt[j,2], pottedplant_gt[j,3]), (255, 0, 0), 1)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', on_mouse)
        cv2.imshow('image', img)
        cv2.waitKey(-1)
        ## save new gts
        whole_gts[img_set.image_index[i]] = current_gts
        ##
    #

    # serialize gts
    f = open('tools/leaf_gts.pkl', 'wb')
    pickle.dump(whole_gts, f)
    f.close()
    #



if __name__ == '__main__':
    main()