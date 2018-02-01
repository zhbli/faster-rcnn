# v0.0
Based on github/zhbli/faster-rcnn/master/v3.0

# v1.0
(Useless) Train net using hard negatives.
Do test after several training iters, find one error detect box in one img, and sample 32 negative RoIs around it.

Usage: Run `trainval_net.py`

Change list:  
- [Modify_file] factory.py  
- [Add_function] test_during_train
- [Modify_function] train_model

Result:
(No use)
AP for aeroplane = 0.6970
AP for bicycle = 0.7984
AP for bird = 0.6920
AP for boat = 0.5537
AP for bottle = 0.5429
AP for bus = 0.7911
AP for car = 0.8116
AP for cat = 0.8195
AP for chair = 0.5189
AP for cow = 0.7600
AP for diningtable = 0.6819
AP for dog = 0.8109
AP for horse = 0.8328
AP for motorbike = 0.7373
AP for person = 0.7789
AP for pottedplant = 0.4213
AP for sheep = 0.7039
AP for sofa = 0.6631
AP for train = 0.7520
AP for tvmonitor = 0.7352
Mean AP = 0.7051
(Origin: 0.7106)
