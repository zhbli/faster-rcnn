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
  
