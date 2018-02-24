#!/usr/bin/env python
import tf
import cv2
import rospy
import rospkg
import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from grid_map_msgs.msg import GridMap

combined_layers = []
patterns = []
ready = False
layers = []
method = 0

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
        for i in range(len(img[0])):
            for j in range(len(img)):
                img[i,j,0] = gridCellToPixel(r[i,j])
                img[i,j,1] = gridCellToPixel(g[i,j])
                img[i,j,2] = gridCellToPixel(b[i,j])
        return img

# ORB, SIFT, FLANN and SURF
# based on https://github.com/yorgosk/grid_map_to_image/blob/master/scripts/matching_test.py
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
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    plt.imshow(img3), plt.show()

def SIFT(img1, img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m_n in matches:
        if len(m_n) == 2:
            (m, n) = m_n
            if m.distance < 0.75 * n.distance:
                good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    plt.imshow(img3), plt.show()

def FLANN(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    if (len(kp1) >= 2) and (len(kp2) >= 2):
        matches = flann.knnMatch(des1, des2, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in xrange(len(matches))]

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i]=[1, 0]

        draw_params = dict(matchColor = (0, 255, 0),
                           singlePointColor = (255, 0, 0),
                           matchesMask = matchesMask,
                           flags = 0)

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    plt.imshow(img3,), plt.show()

def SURF(img1, img2):
    # Initiate SURF detector
    surf = cv2.xfeatures2d.SURF_create(400)

    # find the keypoints and descriptors with SURF
    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(img2,None)

    # # BFMatcher with default params
    # bf = cv2.BFMatcher()

    # Create BFMatcher and add cluster of training images. One for now.
    bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False) # crossCheck not supported by BFMatcher
    clusters = np.array([des1])
    bf.add(clusters)

    # Train: Does nothing for BruteForceMatcher though.
    bf.train()

    # Match descriptors.
    # matches = bf.knnMatch(des1,des2, k=2)
    matches = bf.match(des2)
    matches = sorted(matches, key = lambda x:x.distance)

    # Apply ratio test
    good = []
    for m in matches:
        if m.distance < 0.75:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    plt.imshow(img3),plt.show()

def gridMapCallback(msg):
    global ready, combined_layers, patterns, layers
    if ready:
        ready = False
        if len(combined_layers) == 3:
            comb_img = gridMapLayersToImage(msg, combined_layers)
            s =  np.shape(comb_img)
            comb_img.data = np.reshape(comb_img.data, s)
            for p in patterns:
                if p.combined:
                    if method == 0:
                        ORB(np.array(comb_img.data), p.data.astype(np.uint8))
                    elif method == 1:
                        SIFT(np.array(comb_img.data), p.data.astype(np.uint8))
                    elif method == 2:
                        FLANN(np.array(comb_img.data), p.data.astype(np.uint8))
                    elif method == 3:
                        SURF(np.array(comb_img.data), p.data.astype(np.uint8))
                    # Above calls are all broken! Damn noooice!
                    # TODO transformations based on returned values (TODO)
        layer_imgs = []
        for l in layers:
            layer_imgs.append(Pattern("", l, data = gridMapLayerToImage(msg, l)))
        return 0
        for img in layer_imgs:
            for p in patterns:
                if img.layer == p.layer:
                    if method == 0:
                        ORB(img.data.astype(np.uint8), p.data.astype(np.uint8))
                    elif method == 1:
                        SIFT(img.data.astype(np.uint8), p.data.astype(np.uint8))
                    elif method == 2:
                        FLANN(img.data.astype(np.uint8), p.data.astype(np.uint8))
                    elif method == 3:
                        SURF(img.data.astype(np.uint8), p.data.astype(np.uint8))
                    # TODO transformations based on returned values (TODO)
        ready = True

def initPatterns(path, layers, specific_files, combined_layers):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    patterns_oi = []
    for f in files:
        layers_ok = False
        specific_ok = len(specific_files) <= 0
        curr_l = ""
        curr_s = ""
        for l in layers:
            if l in f:
                layers_ok = True
                curr_l = l
                break
        for sf in specific_files:
            if sf in f:
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
            patterns_oi[p].data = np.reshape(v, (w,h))

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
                                for i in range(len(d[0])):
                                    for j in range(len(p__.data)):
                                        d[i,j,0] = gridCellToPixel(p.data[i,j])
                                        d[i,j,1] = gridCellToPixel(p_.data[i,j])
                                        d[i,j,2] = gridCellToPixel(p__.data[i,j])
                                patterns_oi.append(Pattern(p.filename+"+"+p_.filename+"+"+p__.filename, p.layer+"+"+p_.layer+"+"+p__.layer, specific = p.specific, combined = True, data = d))
                                break
    return patterns_oi

def init():
    global ready, patterns, layers, combined_layers, method
    rospy.init_node("grid_submap_detection")

    rospack = rospkg.RosPack()
    path = rospack.get_path("grid_submap") + "/data/"

    layers = rospy.get_param("~layers", ["traversability_step", "traversability_slope", "traversability_roughness", "traversability", "elevation"])
    gridmap_topic = rospy.get_param("~gridmap_topic", "/traversability_estimation/traversability_map")
    specific_files = rospy.get_param("~specific_files", [])
    combined_layers = rospy.get_param("~combined_layers", [])
    feature_matching_method = rospy.get_param("~feature_matching_method", "ORB") # Alternatives "SIFT", "FLANN", "SURF"
    to_file = rospy.get_param("~to_file", False)
    if feature_matching_method == "SIFT":
        method = 1
    elif feature_matching_method == "FLANN":
        method = 2
    elif feature_matching_method == "SURF":
        method = 3

    combined_layers = ["traversability_slope", "traversability_step", "traversability_roughness"]

    patterns = initPatterns(path, layers, specific_files, combined_layers)
    rospy.Subscriber(gridmap_topic, GridMap, gridMapCallback)

    ready = True

    rospy.spin()

if __name__ == '__main__':
    init()