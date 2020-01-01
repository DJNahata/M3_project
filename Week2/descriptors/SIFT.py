import cv2
import numpy as np


class DenseSIFT():
    
    def __init__(self):
        self.SIFT = cv2.xfeatures2d.SIFT_create()

    def compute(self, img_paths, step_size, descriptor_sizes):
        descriptors = []
        for img_path in img_paths:
            img = cv2.imread(img_path, 0)
            kps = []
            for x in range(0, img.shape[1], step_size):
                for y in range(0, img.shape[0], step_size):
                    for descriptor_size in descriptor_sizes:
                        kp = cv2.KeyPoint(x, y, descriptor_size)
                        kps.append(kp)
            _, des = self.SIFT.compute(img, kps)
            descriptors.append(des)
        return descriptors


class DenseSIFTPyramid():
        
    def __init__(self):
        self.SIFT = cv2.xfeatures2d.SIFT_create()

    def compute(self, img_paths, step_size, descriptor_sizes, n_partitions=1):
        descriptors = []
        for img_path in img_paths:
            img = cv2.imread(img_path, 0)
            a = img.shape[1]
            b = img.shape[0]
            partitions = [[int(i/n_partitions*a),int((i+1)/n_partitions*a),int(j/n_partitions*b),int((j+1)/n_partitions*b)] for i in range(n_partitions) for j in range(n_partitions)]
            all_des = []
            for xmin, xmax, ymin, ymax in partitions:
                kps = []
                for x in range(xmin, xmax, step_size):
                    for y in range(ymin, ymax, step_size):
                        for descriptor_size in descriptor_sizes:
                            kp = cv2.KeyPoint(x, y, descriptor_size)
                            kps.append(kp)
                _, des = self.SIFT.compute(img, kps)
                all_des.append(des)
            descriptors.append(all_des)
        return descriptors
