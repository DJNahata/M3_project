import cv2


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
