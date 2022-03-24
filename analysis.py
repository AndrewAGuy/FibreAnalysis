from skimage import io
from skimage.color import rgb2gray
from skimage.morphology import skeletonize, medial_axis
from skimage.feature import hessian_matrix
from skimage import filters
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import math

filter_method = filters.frangi
filter_kwargs = {
    'black_ridges': False
}
skeleton_method = skeletonize
threshold_method = filters.threshold_otsu
threshold_kwargs = { }
threshold_max_responder = True
normalize_first = False

class FilteredImage:
    def __init__(self, image, sigma):
        self.image = filter_method(image, sigmas = [sigma], **filter_kwargs)
        self.sigma = sigma
        self.hessian = hessian_matrix(image, sigma = sigma, order = 'xy')

    def get_orientation(self, index):
        hxx = self.hessian[0][index]
        hxy = self.hessian[1][index]
        hyy = self.hessian[2][index]
        H = np.array([[hxx, hxy], [hxy, hyy]])
        l, V = la.eig(H)
        v = V[:,np.argmin(l)]
        # Image orientation reverses y
        angle = math.atan2(-v[1], v[0])
        return angle if angle >= 0 else angle + math.pi

class TubenessImage:
    def __init__(self, image, sigmas):
        self.image = image / np.amax(image) if normalize_first else image
        self.images = { s: FilteredImage(image, s) for s in sigmas }
        self.set_max_scales()
        
    def set_max_scales(self):
        self.max = np.zeros(self.image.shape)
        self.max_sigma = np.zeros(self.image.shape)
        for s, filtered in self.images.items():
            self.max = np.maximum(self.max, filtered.image)
            idx = self.max == filtered.image
            self.max_sigma[idx] = filtered.sigma

    def set_skeleton(self, threshold):
        normalized = self.max / np.amax(self.max)
        self.skeleton = skeleton_method(normalized >= threshold)

    def set_skeleton_auto(self):
        if threshold_max_responder:
            max_responder = self.images[max(self.images,
                                         key = lambda k: np.sum(
                                             self.images[k].image))]
            print("max responder: " + str(max_responder.sigma))
            normalized = max_responder.image
            normalized = normalized / np.amax(normalized)
        else:
            normalized = self.max / np.amax(self.max)
        threshold = threshold_method(normalized, **threshold_kwargs)
        self.skeleton = skeleton_method(normalized >= threshold)
        print("threshold = " + str(threshold))

    def get_diameter_count(self):
        masked = self.skeleton * self.max_sigma
        sigmas = np.fromiter(self.images.keys(), dtype=float)
        count = [np.sum(masked == s) for s in sigmas]
        return sigmas, count

    def get_diameters(self):
        masked = self.skeleton * self.max_sigma
        idx = np.nonzero(masked)
        return [masked[tup] for tup in zip(idx[0],idx[1])]
    
    def get_orientations(self):
        masked = self.skeleton * self.max_sigma
        idx = np.nonzero(masked)
        return [self.images[masked[tup]].get_orientation(tup)
                for tup in zip(idx[0],idx[1])]
        
def load(path):
    image = io.imread(path)
    if len(image.shape) == 3:
        image = rgb2gray(image)
    return image

def plot_hist(image, sigmas, threshold = None, outpath = None):
    ti = TubenessImage(image, sigmas)
    if threshold is None:
        ti.set_skeleton_auto()
    else:
        ti.set_skeleton(threshold)
    sigmas, count = ti.get_diameter_count()
    ori = ti.get_orientations()
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    fig.show()
    fig, ax = plt.subplots()
    ax.imshow(ti.max, cmap="gray")
    fig.show()
    fig, ax = plt.subplots()
    ax.imshow(ti.skeleton, cmap="gray")
    fig.show()
    fig, ax = plt.subplots()
    ax.plot(sigmas,count)
    fig.show()
    fig, ax = plt.subplots()
    ax.hist(ori,bins=25)
    fig.show()
    if outpath is not None:
        np.savetxt(outpath + "_scales.txt", np.array([sigmas, count]))
        np.savetxt(outpath + "_angles.txt", np.array(ori))
    
