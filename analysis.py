from skimage.io import imread
from skimage.color import rgb2gray
from skimage.morphology import skeletonize, opening, disk
from skimage.feature import hessian_matrix
from skimage.exposure import equalize_hist
from skimage.filters import frangi, threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la

##skeleton_method = skeletonize
##threshold_method = threshold_otsu
##threshold_kwargs = { }
##threshold_max_responder = True

class FilteredImage:
    @staticmethod
    def ensure_filter(method, kwargs):
        if method is None:
            method = frangi
        if kwargs is None:
            kwargs = {
                'black_ridges': False
            }
        return method, kwargs
    
    def __init__(self, image, sigma,
                 method = None, kwargs = None):
        method, kwargs = FilteredImage.ensure_filter(method, kwargs)
        self.image = method(image, sigmas = [sigma], **kwargs)
        self.sigma = sigma
        self.hessian = hessian_matrix(image, sigma = sigma, order = 'rc')

    def get_orientation(self, index):
        hrr = self.hessian[0][index]
        hrc = self.hessian[1][index]
        hcc = self.hessian[2][index]
        H = np.array([[hrr, hrc],
                      [hrc, hcc]])
        l, V = la.eigh(H)
        v = V[:, np.argmin(np.abs(l))]
        # This now gives angle from -y axis, anticlockwise
        angle = np.arctan2(v[1], v[0])
        return angle if angle >= 0 else angle + np.pi

class TubenessImage:
    def __init__(self, image, sigmas,
                 equalize = False, normalize = True,
                 filter_method = None, filter_kwargs = None):
        filter_method, filter_kwargs = FilteredImage.ensure_filter(
            filter_method, filter_kwargs)
        if equalize:
            image = equalize_hist(image)
        self.image = image / np.amax(image) if normalize else image
        self.images = { s: FilteredImage(self.image, s,
                                         filter_method, filter_kwargs)
                        for s in sigmas }
        self.set_max_scales()
        self.mask = np.ones(self.image.shape, dtype = bool)
        self.set_scores()
        
    def set_max_scales(self):
        self.max = np.zeros(self.image.shape)
        self.max_sigma = np.zeros(self.image.shape)
        for s, f in self.images.items():
            self.max = np.maximum(self.max, f.image)
            idx = self.max == f.image
            self.max_sigma[idx] = s

    def set_scores(self):
        # Set scores of each filtered image
        # Total response to the filter, pixels dominated by this filter
        # and total vesselness score contributed by this scale
        for s, f in self.images.items():
            f.response = np.sum(f.image * self.mask)
            idx = (self.max_sigma == s) * self.mask
            f.dominance = np.sum(idx)
            f.contribution = np.sum(idx * self.max)

    def get_scores(self):
        scores = {
            'response': [ f.response for f in self.images.values() ],
            'dominance': [ f.dominance for f in self.images.values() ],
            'contribution': [ f.contribution for f in self.images.values() ],
            'intensity': [ f.contribution / f.dominance
                           for f in self.images.values() ],
            }
        return list(self.images.keys()), scores

    @staticmethod
    def get_threshold(image, method = None, kwargs = None):
        if method is None:
            method = threshold_otsu
        if kwargs is None:
            kwargs = { }
        return method(image, **kwargs)

    def get_filtered(self, idx):
        filtered = np.zeros(self.image.shape)
        for s in idx:
            filtered = np.maximum(filtered, self.images[s].image)
        return filtered

    def set_skeleton(self, intensity_fraction = 0.9, opening_radius = 2,
                     t_method = None, t_kwargs = None, s_method = None):
        intensity = np.array([ f.contribution / f.dominance
                               for f in self.images.values() ])
        idx = np.nonzero(intensity >= intensity_fraction * np.amax(intensity))
        t_image = self.get_filtered(idx[0])
        threshold = TubenessImage.get_threshold(t_image, t_method, t_kwargs)
        self.mask = self.max > threshold
        if opening_radius != 0:
            self.mask = opening(self.mask, disk(opening_radius))
        if s_method is None:
            s_method = skeletonize
        self.mask = s_method(self.mask)

##    def set_skeleton(self, threshold):
##        normalized = self.max / np.amax(self.max)
##        self.skeleton = skeleton_method(normalized >= threshold)
##
##    def set_skeleton_auto(self):
##        if threshold_max_responder:
##            self.max_responder = self.images[max(self.images,
##                                         key = lambda k: np.sum(
##                                             self.images[k].image))]
##            print("max responder: " + str(self.max_responder.sigma))
##            normalized = self.max_responder.image
##            #normalized = normalized / np.amax(normalized)
##        else:
##            normalized = self.max #/ np.amax(self.max)
##        self.threshold = threshold_method(normalized, **threshold_kwargs)
##        #normalized = self.max / np.amax(self.max)
##        self.skeleton = skeleton_method(self.max >= self.threshold)
##        print("threshold = " + str(self.threshold))
##        #self.skeleton = dilation(self.skeleton, disk(2))

    def get_diameter_count(self):
        masked = self.mask * self.max_sigma
        sigmas = np.fromiter(self.images.keys(), dtype=float)
        count = [np.sum(masked == s) for s in sigmas]
        return sigmas, count

    def get_diameters(self):
        masked = self.mask * self.max_sigma
        idx = np.nonzero(masked)
        return [ masked[tup] for tup in zip(idx[0], idx[1]) ]
    
    def get_orientations(self):
        masked = self.mask * self.max_sigma
        idx = np.nonzero(masked)
        return [ self.images[masked[tup]].get_orientation(tup)
                 for tup in zip(idx[0], idx[1]) ]
        
def load(path):
    image = imread(path)
    if len(image.shape) == 3:
        image = rgb2gray(image)
    return image

def plot_scores(ti):
    fig, ax = plt.subplots(4,1)
    ti.set_scores()
    s, d = ti.get_scores()
    ax[0].plot(s, d['response'])
    ax[1].plot(s, d['dominance'])
    ax[2].plot(s, d['contribution'])
    ax[3].plot(s, d['intensity'])
##    ax[0,1].plot(s, d['masked_response'])
##    ax[1,1].plot(s, d['masked_dominance'])
##    ax[2,1].plot(s, d['masked_contribution'])
##    ax[3,1].plot(s, np.array(d['masked_contribution'])
##                 /np.array(d['masked_dominance']))
    fig.show()

def plot_hist(image, sigmas, threshold = None, outpath = None):
    ti = TubenessImage(image, sigmas)
    plot_scores(ti)
    ti.set_skeleton()
    plot_scores(ti)
##    if threshold is None:
##        ti.set_skeleton_auto()
##    else:
##        ti.set_skeleton(threshold)
    sigmas, count = ti.get_diameter_count()
    ori = ti.get_orientations()
    fig, ax = plt.subplots(3,2)
    ax[0,0].imshow(ti.image, cmap="gray")
    ax[1,0].imshow(ti.max, cmap="gray")
    ax[2,0].imshow(ti.mask, cmap="gray")
    ax[0,1].plot(sigmas,count)
    ax[1,1].hist(ori,bins=25)
    ax[2,1].plot(list(ti.images.keys()),
                 list(map(lambda k: np.sum(ti.max_sigma==k), ti.images.keys()))
                 )
    #ax[2,1].imshow(filter_method(image,sigmas=sigmas,**filter_kwargs),
    #               cmap="gray")
    fig.show()
    if outpath is not None:
        np.savetxt(outpath + "_scales.txt", np.array([sigmas, count]))
        np.savetxt(outpath + "_angles.txt", np.array(ori))
    
