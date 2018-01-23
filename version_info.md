# v0.0
Original pytorch-faster-rcnn.

# v0.1
Unit test for sampling rois handly.

Usage: Run `unit_test_sample_rois_handly.py`

Change list:
- [Add_flie] unit_test_sample_rois_handly.py

# v0.2
Given any number of ground_truths, generate any number of rois with specified IoUs.

Usage: Run `unit_test_sample_rois_handly.py`

Change list:
- [Modify_file] unit_test_sample_rois_handly.py

# v0.3
Test `unit_test_sample_rois_handly.py` with actual input of function `_sample_rois`

Usage: Run `unit_test_sample_rois_handly.py`

Change list:
- [Modify_file] unit_test_sample_rois_handly.py
- [Add_file] _sample_rois_input.txt

# v1.0
Train net using manually sampled rois.

Usage: Run `trainval_net.py`

Change list:
- [Modify_function] _sample_rois

Result: 
Mean AP = 0.6084, 70000 iters; mAP = 0.63, 120000 iters. The result is net good.
If set fg_num = 64, We can get mAP = 0.6974. It means the code I have changed has no bug.

