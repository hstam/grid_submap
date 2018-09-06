#!/usr/bin/env python
import tf
import cv2
import time
import rospy
import rospkg
import collections
import numpy as np
import math as mt
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from grid_map_msgs.msg import GridMap

magic_thresholds = []
combined_layers = []
stats_file = False
multiple = False
patterns = []
ready = False
layers = []
method = 0
path = ""
specific_files = []

class Pattern:
    def __init__(self, filename, layer, specific="", combined=False, data=[]):
        self.filename = filename
        self.id = filename.split("_")[-2] if len(filename) > 0 else "0"
        self.layer = layer
        self.specific = specific
        self.combined = combined
        self.data = data

def gridCellToPixel(v):
    if v != v: #inf, nan etc
        # I hate this handling of infs and nans...
        return 0
    else:
        return int(255 * v)

def gridMapLayerToImage(gm, layer):
    img = []
    index = gm.layers.index(layer)
    img = [gridCellToPixel(i) for i in gm.data[index].data]
    img = np.reshape(img, (gm.data[index].layout.dim[1].size, gm.data[index].layout.dim[0].size))
    return img

def gridMapLayersToImage(gm, layers):
    if len(layers) == 3:
        r = gridMapLayerToImage(gm, layers[0])
        g = gridMapLayerToImage(gm, layers[1])
        b = gridMapLayerToImage(gm, layers[2])
        img = np.zeros((len(r[0]), len(r), 3), 'uint8')
        for i in range(len(img)):
            for j in range(len(img[0])):
                img[i,j,0] = gridCellToPixel(r[j,i])
                img[i,j,1] = gridCellToPixel(g[j,i])
                img[i,j,2] = gridCellToPixel(b[j,i])
        return img

# ORB, SIFT, FLANN and SURF based on
# https://github.com/yorgosk/grid_map_to_image/blob/master/scripts/matching_test.py
def ORB(img1, img2):
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    clusters = np.array([des1])
    bf.add(clusters)

    # Train: Does nothing for BruteForceMatcher though.
    bf.train()

    # Match descriptors.
    matches = bf.match(des2)

    return kp1, kp2, matches

def SIFT(img1, img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()

    matches = []
    if (len(kp1) > 0) and (len(kp2) > 0):
        matches = bf.knnMatch(des1, des2, k=2)

    return kp1, kp2, matches

def FLANN(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = []

    if (len(kp1) >= 2) and (len(kp2) >= 2):
        matches = flann.knnMatch(des1, des2, k=2)

    return kp1, kp2, matches

def SURF(img1, img2):

    #img2 = cv2.GaussianBlur(img2,(3,3),3)
    
    plt.subplot(121)
    plt.imshow(img1)
    plt.title('QueryImage')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(img2)
    plt.title('TrainImage')
    plt.xticks([])
    plt.yticks([])

    plt.show()

    # Initiate SURF detector
    surf = cv2.xfeatures2d.SURF_create(500, 10, 50, True, True)

    # find the keypoints and descriptors with SURF
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)

    # Create BFMatcher and add cluster of training images. One for now.
    bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False) # crossCheck not supported by BFMatcher
    clusters = np.array([des1])
    bf.add(clusters)                                                                #add is used to add descriptor of multiple test images

    # Train: Does nothing for BruteForceMatcher though.
    bf.train()

    matches = bf.match(des2)                                                                                                                             

    return kp1, kp2, matches

def transformationsHomography(img1, img2, kp1, kp2, matches):
    MIN_MATCH_COUNT = 5

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m in matches:
        if m.distance < 0.7:
            good.append(m)

    print len(good)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
        if M != None:
            dst = cv2.perspectiveTransform(pts,M)

            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                               singlePointColor = None,
                               matchesMask = matchesMask, # draw only inliers
                               flags = 2)

            img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
            plt.imshow(img3, 'gray'),plt.show()
        else:
            print "Not enough points found for homography"

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)

def getMask(w,h,up,left):
    mask = np.zeros((w,h),dtype = np.int)

    if up:
        if left: 
            x0, y0 = 0, (h-2)
            x1, y1 = (w-2), 0 
            x, y = np.round(np.linspace(x0,x1,w-1)), np.round(np.linspace(y0,y1,w-1))
            x, y = x.astype(np.int), y.astype(np.int)

            for i in range(w-1):                
                    mask[i][0:y[i]+1] = np.ones(y[i]+1,dtype = np.int)
        else:
            x0, y0 = 0, 1
            x1, y1 = (w-2), (h-1) 
            x, y = np.round(np.linspace(x0,x1,w-1)), np.round(np.linspace(y0,y1,w-1))
            x, y = x.astype(np.int), y.astype(np.int)

            for i in range(w-1):
                    mask[i][y[i]:h] = np.ones(h-y[i],dtype = np.int)
    else:
        if left: 
            x0, y0 = 1, 0
            x1, y1 = (w-1), (h-2) 
            x, y = np.round(np.linspace(x0,x1,w-1)), np.round(np.linspace(y0,y1,w-1))
            x, y = x.astype(np.int), y.astype(np.int)
            
            for i in range(1,w):
                mask[i][0:y[i-1]+1] = np.ones(y[i-1]+1,dtype = np.int)
        else:
            x0, y0 = 1, (h-1)
            x1, y1 = (w-1), 1
            x, y = np.round(np.linspace(x0,x1,w-1)), np.round(np.linspace(y0,y1,w-1))
            x, y = x.astype(np.int), y.astype(np.int)
            
            for i in range(1,w):
                mask[i][y[i-1]:h] = np.ones(h-y[i-1],dtype = np.int)

    return mask


def templateMatching(img, template, layer_name):
    global stats_file, path, magic_thresholds, multiple, specific_files
    img2 = img.copy()
    plt.imshow(img, cmap = 'gray')
    (w,h) = template.shape[:2]
    (cx,cy) = (h/2,w/2)

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    plt.subplot(121)
    plt.imshow(template)
    plt.title('TemplateImage')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(img)
    plt.title('Image')
    plt.xticks([])
    plt.yticks([])
    plt.show()

#**************************************************

    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:

        img = img2.copy()
        method = eval(meth)
        threshold = magic_thresholds[methods.index(meth)]

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     
        # loop over the rotation-angles of the template image
        for theta in np.linspace(-30.0, 30.0, 30):

            #rotate properly the image
            M = cv2.getRotationMatrix2D((cx,cy),theta,1)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])  
            new_h = int(h*cos+w*sin)
            new_w = int(h*sin+w*cos)

            M[0, 2] += new_h/2-cx               
            M[1, 2] += new_w/2-cy

            rotated = cv2.warpAffine(template,M,(new_h,new_w))

            #create the mask required for the rotated template image
            mask = np.ones((new_w,new_h), dtype = np.int)

            s1 = h*sin
            s2 = h*cos
            if s1 == max(s1,s2):
                s1 = int(mt.floor(s1))
                s2 = int(mt.ceil(s2))
            else:
                s1 = int(mt.ceil(s1))
                s2 = int(mt.floor(s2))

            s3 = w*cos
            s4 = w*sin
            if s3 == max(s3,s4):
                s3 = int(mt.floor(s3))
                s4 = int(mt.ceil(s4))
            else:
                s3 = int(mt.ceil(s3))
                s4 = int(mt.floor(s4))

            if theta > 0:
                mask[0:s1,0:s2] = getMask(s1,s2,False,False)
                mask[0:s3,new_h-s4:new_h] = getMask(s3,s4,False,True)
                mask[new_w-s3:new_w,0:s4] = getMask(s3,s4,True,False)
                mask[new_w-s1:new_w,new_h-s2:new_h] = getMask(s1,s2,True,True)
            else:
                mask[0:s3,0:s4] = getMask(s3,s4,False,False)
                mask[0:s1,new_h-s2:new_h] = getMask(s1,s2,False,True)
                mask[new_w-s1:new_w,0:s2] = getMask(s1,s2,True,False)
                mask[new_w-s3:new_w,new_h-s4:new_h] = getMask(s3,s4,True,True)

            mask = np.ascontiguousarray(mask, dtype=np.uint8)

            plt.subplot(121)

            if method in [cv2.TM_SQDIFF, cv2.TM_CCORR_NORMED]:
                plt.imshow(rotated)
                res = cv2.matchTemplate(img, rotated, method, mask)
            else:
                inv_mask = 1-mask                                               #get the inverse binary mask
                inpainted = cv2.inpaint(rotated,inv_mask,3,cv2.INPAINT_TELEA)
                plt.imshow(inpainted)
                res = cv2.matchTemplate(img, inpainted, method)


            plt.title('TemplateImage')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(122)
            plt.imshow(img)
            plt.title('Image')
            plt.xticks([])
            plt.yticks([])
            plt.show()

#*****************************************************

            # Apply template Matching
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            print "Method = " + meth
            print "Threshold = " + str(threshold)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
                print "Value = (<)" + str(min_val)
            else:
                top_left = max_loc
                print "Value = (>) " + str(max_val)
            bottom_right = (top_left[0] + h, top_left[1] + w)

            if stats_file:
                nothing = True

                if ((meth == methods[4] and min_val < magic_thresholds[4]) 
                    or (meth == methods[5] and min_val < magic_thresholds[5])
                    or (meth == methods[0] and max_val > magic_thresholds[0])
                    or (meth == methods[1] and max_val > magic_thresholds[1])
                    or (meth == methods[2] and max_val > magic_thresholds[2])
                    or (meth == methods[3] and max_val > magic_thresholds[3])):

                    nothing = False
                    if multiple:
                        loc = np.where( res >= threshold)
                        if len(np.shape(img)) < 3:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                            for pt in zip(*loc[::-1]):
                                cv2.rectangle(img, pt, (pt[0] + h, pt[1] + w), (0,0,255), 2)
                            res = np.transpose(res)
                            img = np.transpose(img, (1, 0, 2))
                        else:            
                            for pt in zip(*loc[::-1]):
                                cv2.rectangle(img, pt, (pt[0] + h, pt[1] + w), (0,0,255), 2)
                    else:
                        if len(np.shape(img)) < 3:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                            cv2.rectangle(img, top_left, bottom_right, (255, 255, 0), 2) #yellow
                            res = np.transpose(res)
                            img = np.transpose(img, (1, 0, 2))
                        else:
                            cv2.rectangle(img, top_left, bottom_right, (255, 255, 0), 2)

                if nothing and len(np.shape(img)) < 3:
                    res = np.transpose(res)
                    img = np.transpose(img)

            

        plt.subplot(121)
        plt.imshow(res, cmap = 'gray')
        plt.title('Matching Result')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(122)
        plt.imshow(img, cmap = 'gray')
        plt.title('Detected Point')
        plt.xticks([])
        plt.yticks([])
        plt.suptitle(meth)

        plt.show()

        an = str(raw_input("Was the highlighted (if present) part of the image correct? (y/n)\n"))
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if not specific_files:
            cv2.imwrite(path+"/bag0/log_" + layer_name + "_" + meth + "_thres-" + str(threshold) + "_all_" + an + "_" + timestr + ".png", img)
        else:
            cv2.imwrite(path+"/bag0/log_" + layer_name + "_" + meth + "_thres-" + str(threshold) + "_part_" + an + "_" + timestr + ".png", img)     

    for meth in methods:
        i = 2*methods.index(meth)+1
        img = img2.copy()
        method = eval(meth)
        threshold = magic_thresholds[methods.index(meth)]

        # Apply template Matching
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print "Mean value: "
        print np.mean(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        print "Method = " + meth
        print "Threshold = " + str(threshold)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            #print "Value = (<)" + str(min_val)
        else:
            top_left = max_loc
            #print "Value = (>) " + str(max_val)
        bottom_right = (top_left[0] + h, top_left[1] + w)

        if stats_file:
            #print "Check the images, decide if the selected area corresponds to the trained image, close the window and answer with 'y' or 'n'"
            nothing = True

            if ((meth == methods[4] and min_val < magic_thresholds[4]) 
                or (meth == methods[5] and min_val < magic_thresholds[5])
                or (meth == methods[0] and max_val > magic_thresholds[0])
                or (meth == methods[1] and max_val > magic_thresholds[1])
                or (meth == methods[2] and max_val > magic_thresholds[2])
                or (meth == methods[3] and max_val > magic_thresholds[3])):

                nothing = False
                if multiple:
                    loc = np.where( res >= threshold)
                    if len(np.shape(img)) < 3:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        for pt in zip(*loc[::-1]):
                            cv2.rectangle(img, pt, (pt[0] + h, pt[1] + w), (0,0,255), 2)
                        res = np.transpose(res)
                        img = np.transpose(img, (1, 0, 2))
                    else:
                        for pt in zip(*loc[::-1]):
                            cv2.rectangle(img, pt, (pt[0] + h, pt[1] + w), (0,0,255), 2)
                else:
                    if len(np.shape(img)) < 3:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        cv2.rectangle(img, top_left, bottom_right, (255, 255, 0), 2)
                        res = np.transpose(res)
                        img = np.transpose(img, (1, 0, 2))
                    else:
                        cv2.rectangle(img, top_left, bottom_right, (255, 255, 0), 2)

            if nothing and len(np.shape(img)) < 3:
                res = np.transpose(res)
                img = np.transpose(img)


            plt.subplot(6,2,i)
            plt.imshow(res, cmap = 'gray')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(6,2,i+1)
            plt.imshow(img, cmap = 'gray')
            plt.xticks([])
            plt.yticks([])
        
    plt.show()

    if stats_file:
        print "\033[1;33mDone with all methods! Letting the next grid map through for inspection...\033[0m"

    return top_left, bottom_right

def midPoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def gridMapCallback(msg):
    global ready, combined_layers, patterns, layers, combined_only, specific_files
    
    if ready:
        ready = False
        if len(combined_layers) == 3:
            comb_img = gridMapLayersToImage(msg, combined_layers)
            for p in patterns:
                if p.combined:
                    matches = []
                    kp1 = []
                    kp2 = []
                    if method == 0:
                        kp1, kp2, matches = ORB(comb_img, p.data)
                    elif method == 1:
                        kp1, kp2, matches = SIFT(comb_img, p.data.astype(np.uint8))
                    elif method == 2:
                        kp1, kp2, matches = FLANN(comb_img, p.data)
                    elif method == 3:
                        kp1, kp2, matches = SURF(comb_img, p.data)
                    elif method == 4:
                        top_left, bottom_right = templateMatching(comb_img.astype(np.uint8), p.data.astype(np.uint8), "combined")
                        center_point = midPoint(top_left, bottom_right)
                        # TODO get the transformation from the gridmap and publish a tf
                    if method != 4:
                        transformationsHomography(comb_img, p.data, kp1, kp2, matches)
                    # TODO transformations based on returned values
        if not combined_only:                                                                                           
            layer_imgs = []
            for l in layers:
                layer_imgs.append(Pattern("", l, data = gridMapLayerToImage(msg, l)))
            for img in layer_imgs:
                for p in patterns:
                    if img.layer == p.layer:
                        matches = []
                        kp1 = []
                        kp2 = []
                        if method == 0:
                            kp1, kp2, matches = ORB(img.data.astype(np.uint8), p.data.astype(np.uint8))
                        elif method == 1:
                            kp1, kp2, matches = SIFT(img.data.astype(np.uint8), p.data.astype(np.uint8))
                        elif method == 2:
                            kp1, kp2, matches = FLANN(img.data.astype(np.uint8), p.data.astype(np.uint8))
                        elif method == 3:
                            kp1, kp2, matches = SURF(img.data.astype(np.uint8), p.data.astype(np.uint8))
                        elif method == 4:
                            top_left, bottom_right = templateMatching(img.data.astype(np.uint8), p.data.astype(np.uint8), p.layer)
                            center_point = midPoint(top_left, bottom_right)
                            # TODO get the transformation from the gridmap and publish a tf

                        if method != 4:
                            transformationsHomography(img.data.astype(np.uint8), p.data, kp1, kp2, matches)
                        # TODO transformations based on returned values
        ready = True

def initPatterns(path, layers, combined_layers):
    global specific_files
    files = [f for f in listdir(path) if isfile(join(path, f))]
    patterns_oi = []

    for f in files:
        layers_ok = False
        specific_ok = len(specific_files) <= 0
        curr_l = ""
        curr_s = ""
        for l in layers:
            if l in f and "txt" in f[-3:]:
                layers_ok = True
                curr_l = l
                break
        for sf in specific_files:
            if sf in f and "txt" in f[-3:]:
                specific_ok = True
                curr_s = sf
                break
        if layers_ok and specific_ok:
            patterns_oi.append(Pattern(f, curr_l, specific = curr_s))

    del files[:]

    for p in range(len(patterns_oi)):
        v = []
        with open(path+patterns_oi[p].filename) as f:
            w = int(f.readline())
            h = int(f.readline())
            for line in f:
                values = line.split("#")
                values = [gridCellToPixel(float(val)) for val in values]
                v.append(values)

            patterns_oi[p].data = np.reshape(v, (h,w))

    if len(combined_layers) == 3:
        for p in patterns_oi:
            if p.layer == combined_layers[0]:
                stop = False
                for p_ in patterns_oi:
                    if stop:
                        break
                    if p_.layer == combined_layers[1] and p_.id == p.id:
                        for p__ in patterns_oi:
                            if p__.layer == combined_layers[2] and p__.id == p.id:
                                stop = True
                                d = np.zeros((len(p__.data[0]), len(p__.data), 3), 'uint8')
                                for i in range(len(d)):
                                    for j in range(len(d[0])):
                                        d[i,j,0] = gridCellToPixel(p.data[j,i])
                                        d[i,j,1] = gridCellToPixel(p_.data[j,i])
                                        d[i,j,2] = gridCellToPixel(p__.data[j,i])
                                patterns_oi.append(Pattern(p.filename+"+"+p_.filename+"+"+p__.filename, p.layer+"+"+p_.layer+"+"+p__.layer, specific = p.specific, combined = True, data = d))
                                break
    return patterns_oi

def init():
    global ready, patterns, layers, combined_layers, method, combined_only, stats_file, path,magic_thresholds, multiple, specific_files
    rospy.init_node("grid_submap_detection")

    rospack = rospkg.RosPack()
    path = rospack.get_path("grid_submap") + "/data/"

    layers = rospy.get_param("~layers", ["traversability_step", "traversability_slope", "traversability_roughness", "traversability", "elevation"])
    gridmap_topic = rospy.get_param("~gridmap_topic", "/traversability_estimation/traversability_map")
    specific_files = rospy.get_param("~specific_files", [])
    combined_layers = rospy.get_param("~combined_layers", ["traversability_slope", "traversability_step", "traversability_roughness"])
    feature_matching_method = rospy.get_param("~feature_matching_method", "TM") # Alternatives "ORB", SIFT", "FLANN", "SURF", "TM"
    combined_only = rospy.get_param("~combined_only", True)
    stats_file = rospy.get_param("~write_stats_file", False)
    multiple = rospy.get_param("~multiple", False)

    #multiple = True
    stats_file = True

    #'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    magic_thresholds = rospy.get_param("~magic_thresholds", [250000, 0.7, 700000, 0.7, 30000, 0.3])

    if feature_matching_method == "ORB":
        method = 0
    elif feature_matching_method == "SIFT":
        method = 1
    elif feature_matching_method == "FLANN":
        method = 2
    elif feature_matching_method == "SURF":
        method = 3
    elif feature_matching_method == "TM": # Template Matching
        method = 4

    # Methods 1 and 2 are not providing the correct format for matches(?)
    patterns = initPatterns(path, layers, combined_layers)
    rospy.Subscriber(gridmap_topic, GridMap, gridMapCallback)

    ready = True

    rospy.spin()

if __name__ == '__main__':
    init()