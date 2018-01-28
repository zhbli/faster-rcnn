# v0.0
Original pytorch-faster-rcnn.

# v1.0  
Get all missed ground truths.

Usage: Run `reval.py`

Change list:  
- [Modify_function] voc_eval
- [Add_files] missed_gt_*.pkl: missed ground truth files

# v2.0
Check IoU between 300 det_results and every missed_gt in one img.  

Usage: Run `analyze_missed_gts.py`

Change list:  
- [Add_file] analyze_missed_gts.py

Result:  
147 missed_gts in pottedplant.  
Compared with 300 final detect boxes, 78 gts' IoU > 0.5(most scores of them are not very small), and 69 gts' IoU < 0.5.  
  
# v2.1
For missed_gts whose IoU < 0.5 with final det_result, check their IoU with 300 high_scored_RoIs from PRN.  

Change list:
- [Modify_file] analyze_missed_gts.py

Result:  
cfg.TEST.RPN_PRE_NMS_TOP_N = 6000, cfg.TEST.RPN_POST_NMS_TOP_N = 300: IoU > 0.5: 85, IoU < 0.5: 62
cfg.TEST.RPN_PRE_NMS_TOP_N = 6000, cfg.TEST.RPN_POST_NMS_TOP_N = 3000: IoU > 0.5: 92, IoU < 0.5: 55
cfg.TEST.RPN_PRE_NMS_TOP_N = Inf, cfg.TEST.RPN_POST_NMS_TOP_N = 1000: IoU > 0.5: 116, IoU < 0.5: 31

# v2.2
Check if whole rpn results can cover all missed_gts. 

Change list:
- [Modify_file] analyze_missed_gts.py
- [Modify_function] proposal_layer 

Result:
IoU > 0.5: 129, IoU < 0.5: 18