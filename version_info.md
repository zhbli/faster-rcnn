# v0.0
Original pytorch-faster-rcnn.

# v1.0
Train net with multi_roi_pooling.
For a roi, do RoI pooling on 3 feature maps with different scales.
Then, concat the 3 pooled features as the feature of this RoI.

Usage: Run `trainval_net.py`

Change list:
- [Modify_function] vgg16.py -> _image_to_head
- [Modify_function] network.py -> _crop_pool_layer