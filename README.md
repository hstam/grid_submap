# Grid Submap
Using the Grid Submap node, you can create grid submaps and save them on .txt files.

# Grid Submap Detection
Using the Grid Submap Detection, you can detect gridmaps inside other gridmaps, based on txt files that are used as "training".

A few guidelines on how to use the config file are needed:
1. Use the `layers` parameter to set the layers of the gridmap that should be used for detection.
2. Use the `gridmap_topic` parameter to set the input gridmap topic
3. Use the `specific_files` to filter the files in the data folder of the package that will be used as "training". For example, you can use something like specific_files: ["recording1", "recording16"]
4. Use the `combined_layers` parameter to generate 3-dimensional detection. This parameter should always contain 3 elements.
5. Use the `feature_matching_method` to change the detection method. "TM" (Template Matching) is the default, but you can also use "ORB", "SIFT", "FLANN" and "SURF". "TM" is strongly recommended.
6. Use the `combined_only` parameter to get combination-based detection results of the layers of interest (no results based on individual layers)
7. Use the `write_stats_file` parameter to log detection results after visualizing them.
8. Use the `magic_thresholds` parameter to change the threshold of the "accepted" detections for any of the Template Matching methods. Keep in mind that the first four number are the lower bound and the last two the upper bound, since their corresponding methods work that way. The corresponding methods based on the `magic_thresholds`'s index are:
'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

The `data` folder includes many submaps and results that you may or may not find useful. :koala: