import numpy as np

from sklearn.cluster import MiniBatchKMeans


class VisualWords():

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, descriptors):
        self.codebook = MiniBatchKMeans(n_clusters=self.n_clusters,
            verbose=False,
            batch_size=self.n_clusters*20,
            compute_labels=False,
            reassignment_ratio=10**-4,
            random_state=42)
        descriptors = np.vstack(descriptors)
        self.codebook.fit(descriptors)

    def get_visual_words(self, descriptors):
        visual_words = np.empty((len(descriptors), self.n_clusters), dtype=np.float32)
        for i, descriptor in enumerate(descriptors):
            words = self.codebook.predict(descriptor)
            visual_words[i,:] = np.bincount(words, minlength=self.n_clusters)
        return visual_words
