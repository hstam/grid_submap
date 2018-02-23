#!/usr/bin/env python
import tf
import rospy
import rospkg
import numpy as np
from os import listdir
from os.path import isfile, join
from grid_map_msgs.msg import GridMap

combined_layers = []
patterns = []
ready = False
layers = []

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

def gridMapCallback(msg):
    global ready, combined_layers, patterns
    if ready:
        ready = False
        if len(combined_layers) == 3:
            comb_img = gridMapLayersToImage(msg, combined_layers)
            for p in patterns:
                if p.combined:
                    continue
                    # TODO result of rgb here
        layer_imgs = []
        for l in layers:
            layer_imgs.append(Pattern("", l, data = gridMapLayerToImage(msg, l)))

        for img in layer_imgs:
            for p in patterns:
                if img.layer == p.layer:
                    continue
                    # TODO rest of results here
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
                            if p_.layer == combined_layers[2] and p__.id == p.id:
                                stop = True
                                d = np.zeros((len(p__.data[0]), len(p__.data), 3), 'uint8')
                                d[..., 0] = [gridCellToPixel(dato) for dato in p.data]
                                d[..., 1] = [gridCellToPixel(dato) for dato in p_.data]
                                d[..., 2] = [gridCellToPixel(dato) for dato in p__.data]
                                patterns_oi.append(Pattern(p.filename+"+"+p_.filename+"+"+p__.filename, p.layer+"+"+p_.layer+"+"+p__.layer, specific = p.specific, combined = True, data = d))
                                break
    return patterns_oi

def init():
    global ready, patterns, layers, combined_layers
    rospy.init_node("grid_submap_detection")

    rospack = rospkg.RosPack()
    path = rospack.get_path("grid_submap") + "/data/"

    layers = rospy.get_param("~layers", ["traversability_step", "traversability_slope", "traversability_roughness", "traversability", "elevation"])
    gridmap_topic = rospy.get_param("~gridmap_topic", "/traversability_estimation/traversability_map")
    specific_files = rospy.get_param("~specific_files", [])
    combined_layers = rospy.get_param("~combined_layers", [])
    to_file = rospy.get_param("~to_file", False)

    rospy.Subscriber(gridmap_topic, GridMap, gridMapCallback)

    patterns = initPatterns(path, layers, specific_files, combined_layers)
    ready = True

    rospy.spin()

if __name__ == '__main__':
    init()