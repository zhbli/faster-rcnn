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
AP for pottedplant = 0.3283
Mean AP = 0.3283