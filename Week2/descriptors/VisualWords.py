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

class VisualWordsPyramid():

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, descriptors):
        self.codebook = MiniBatchKMeans(n_clusters=self.n_clusters,
            verbose=False,
            batch_size=self.n_clusters*20,
            compute_labels=False,
            reassignment_ratio=10**-4,
            random_state=42)
        full_descriptors = []
        for des in descriptors:
            to_append = []
            for subdes in des:
                to_append.extend(subdes)
            full_descriptors.append(to_append)
        full_descriptors = np.vstack(full_descriptors)
        self.codebook.fit(full_descriptors)

    # def get_visual_words(self, descriptors):
    #     visual_words = np.empty((len(descriptors), self.n_clusters*len(descriptors[0])), dtype=np.float32)
    #     for i, descriptor in enumerate(descriptors):
    #         for j, subdescriptor in enumerate(descriptor):
    #             words = self.codebook.predict(subdescriptor)
    #             visual_words[i,j*self.n_clusters:(j+1)*self.n_clusters] = np.bincount(words, minlength=self.n_clusters)
    #     return visual_words

    def get_visual_words(self, descriptors):
        visual_words = np.empty((len(descriptors), self.n_clusters*(len(descriptors[0])+1)), dtype=np.float32)
        for i, descriptor in enumerate(descriptors):
            full_descriptor = []
            for subdes in descriptor:
                full_descriptor.extend(subdes)
            words = self.codebook.predict(full_descriptor)
            visual_words[i,:self.n_clusters] = np.bincount(words, minlength=self.n_clusters)

            for j, subdescriptor in enumerate(descriptor):
                words = self.codebook.predict(subdescriptor)
                visual_words[i,(j+1)*self.n_clusters:(j+2)*self.n_clusters] = np.bincount(words, minlength=self.n_clusters)
        return visual_words
