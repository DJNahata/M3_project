import numpy as np
from copy import deepcopy


class Normalizer():

    def __init__(self):
        pass

    def normalize(self, x, norm, alpha=0.5):
        x_norm = deepcopy(x)
        for i,u in enumerate(x):
            x_norm[i] = self.__normalize(u, norm, alpha)
        return x_norm        

    def __normalize(self, u, norm, alpha):
        if norm == "l1":
            norm = np.linalg.norm(u, ord=1)
            u_norm = u / norm
        elif norm == "l2":
            norm = np.linalg.norm(u, ord=2)
            u_norm = u / norm
        elif norm == "power":
            u = np.sign(u) * np.abs(u) ** alpha
            norm = np.linalg.norm(u, ord=2)
            u_norm = u / norm
        return u_norm
