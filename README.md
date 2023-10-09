# ROS MMSeg
Need to install [mmseg](https://mmsegmentation.readthedocs.io/en/latest/get_started.html)


Change config_file and checkpoint_file in scripts/semantic_predictor_segformer.py
## Input 
CompressedImage topic: /zed_node/left/image_rect_color/compressed
## Output
Image topic: /segmentated
