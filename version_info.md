# v0.0
Original pytorch-faster-rcnn.

# v1.0
Just train 2 class: pottedplant / background.
Just use untruncated + easy pottedplant ground_truth to train net.

Usage: Run `trainval_net.py --weight /home/zhbli/Project/fast-rcnn/data/imagenet_weights/vgg16.pth --imdb voc_2007_zhbli_pottedplant_untruncated_easy_trainval --imdbval voc_2007_test --iters 2600 --cfg experiments/cfgs/vgg16.yml --net vgg16 --set ANCHOR_SCALES [8,16,32] ANCHOR_RATIOS [0.5,1,2] TRAIN.STEPSIZE [1800]`

Change list:
- [Modify_function] _load_pascal_annotation
- [Modify_file] factory.py
- [Modify_parameters] iters=2600, TRAIN.STEPSIZE=1800, imdb=voc_2007_zhbli_pottedplant_untruncated_easy_trainval

Result:  
(not good, may because iter_num is too small. Or say, 2600 iters is not enough to train the RPN net)  
(The next reason to explain the low ap is, because we just use 300+ images to train net,  the train process did not utilize the negative examples in other images)  
AP for pottedplant = 0.3283  
Mean AP = 0.3283  

# v2.0
Sample rois manually.

Existing bug:  
- Many gt_boxes are [0, 0, 0, 0]

Usage: Run `trainval_net.py --weight /home/zhbli/Project/fast-rcnn/data/imagenet_weights/vgg16.pth --imdb voc_2007_zhbli_pottedplant_untruncated_easy_trainval --imdbval voc_2007_test --iters 5200 --cfg experiments/cfgs/vgg16.yml --net vgg16 --set ANCHOR_SCALES [8,16,32] ANCHOR_RATIOS [0.5,1,2] TRAIN.STEPSIZE [3600]`

Change list:
- [Modify_function] _sample_rois
- [Modify_parameters] iters=5200, TRAIN.STEPSIZE=3600

Result:
The error is stable below 0.1.
AP for pottedplant = 0.3003
Mean AP = 0.3003

# v2.1
Fix bugs in v2.0.

Usage: Run `trainval_net.py --weight /home/zhbli/Project/fast-rcnn/data/imagenet_weights/vgg16.pth --imdb voc_2007_zhbli_pottedplant_untruncated_easy_trainval --imdbval voc_2007_test --iters 5200 --cfg experiments/cfgs/vgg16.yml --net vgg16 --set ANCHOR_SCALES [8,16,32] ANCHOR_RATIOS [0.5,1,2] TRAIN.STEPSIZE [3600]`  

Change list:  
- [Modify_file] pascal_voc.py  

Result:
The error is stable below 0.1.
AP for pottedplant = 0.2935
Mean AP = 0.2935
(The bug in v2.0 did not reduce ap)

# v3.0
Annotate bbox.

Usage: Run `annotate_bbox.py`

Change list:
- [Add_file] annotate_bbox.py

# v3.1
Load leaf gts to roidb.

Usage: None

Change list:
- [Modify_file] pascal_voc.py

# v3.2
Train net with 3 classes: background, pottedplant and leaf.

Usage: Run `trainval_net.py`

Change list:   
- [Modify_function] _sample_rois.py

Result:
(Sample 64 pottedplant rois and 64 leaf rois)  
The error is stable below 0.3.
AP for pottedplant = 0.2837
AP for leaf = 0.0000

(Sample 32 pottedplant rois and 32 leaf rois)
The error is stable below 0.1.
AP for pottedplant = 0.2995
AP for leaf = 0.0000