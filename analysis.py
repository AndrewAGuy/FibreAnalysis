from skimage.io import imread
from skimage.color import rgb2gray
from skimage.morphology import skeletonize, opening, closing, disk
from skimage.feature import hessian_matrix
from skimage.exposure import equalize_hist
from skimage.filters import frangi, threshold_otsu
from skimage.measure import label
import numpy as np
from numpy import linalg as la


class FilteredImage:
    """
    An image which has been filtered using Frangi's filter for a given sigma.
    The image represents a probability of a pixel being a vessel of radius
    sigma. Also stores the Hessian matrix for this sigma, to extract the
    orientations when queried at a point.

    Attributes:
        sigma : float
        image : ndarray
        hessian : list[ndarray]
        response : float
        dominance : int
        contribution : float

    References: 
        Frangi, Alejandro F., et al. "Multiscale vessel enhancement filtering."
        International conference on medical image computing and 
        computer-assisted intervention. Springer, Berlin, Heidelberg, 1998.
    """

    @staticmethod
    def ensure_filter(method, kwargs):
        """
        Sets a method and arguments object to the defaults
        (frangi, white vessels on black background) if not specified.
        """
        if method is None:
            method = frangi
        if kwargs is None:
            kwargs = {
                'black_ridges': False
            }
        return method, kwargs

    def __init__(self, image, sigma,
                 method=None, kwargs=None):
        """
        Parameters:
            image : ndarray
                Grayscale image to filter.
            sigma : float
                The radius of the fibres the filter will respond to.
            method : callable
                The function to use to look to fibres. Defaults to `frangi`.
                Must take an image as its first argument and a list of scales
                `sigmas`.
            kwargs : dict
                The optional keyword arguments for `method`. Defaults to 
                identifying white vessels.
        """
        method, kwargs = FilteredImage.ensure_filter(method, kwargs)
        self.image = method(image, sigmas=[sigma], **kwargs)
        self.sigma = sigma
        self.hessian = hessian_matrix(image, sigma=sigma, order='rc')

    def get_orientation(self, index):
        """
        For a pixel specified by a (row, col) tuple, returns the orientation in
        radians from the -y axis, anticlockwise. Because we cannot distinguish
        between directions, clamps to range [0, pi].

        Parameters        
            index : (int, int)
                The point to query as a 2-tuple
        Returns        
            orientation : float in range [0, pi]
        """
        hrr = self.hessian[0][index]
        hrc = self.hessian[1][index]
        hcc = self.hessian[2][index]
        H = np.array([[hrr, hrc],
                      [hrc, hcc]])
        l, V = la.eigh(H)
        v = V[:, np.argmin(np.abs(l))]
        angle = np.arctan2(v[1], v[0])
        return angle if angle >= 0 else angle + np.pi


class TubenessImage:
    """
    An image, filtered at multiple length scales for extracting statistics.
    Approach is intended to reject spurious fibres where possible, so is very
    conservative when segmenting foreground. Statistics should be interpreted
    as the probability of a randomly selected point on a fibre in the top layer
    having the given characteristic.
    """
    
    def __init__(self, image, sigmas,
                 equalize=False, normalize=True,
                 filter_method=None, filter_kwargs=None):
        """
        
        """
        filter_method, filter_kwargs = FilteredImage.ensure_filter(
            filter_method, filter_kwargs)
        if equalize:
            image = equalize_hist(image)
        self.image = image / np.amax(image) if normalize else image
        self.images = {s: FilteredImage(self.image, s,
                                        filter_method, filter_kwargs)
                       for s in sigmas}
        self.set_max_scales()
        self.mask = np.ones(self.image.shape, dtype=bool)
        self.set_scores()

    def set_max_scales(self):
        """
        Generates the maximum response image (the traditional Frangi filter)
        as well as recording the length scale which produced the value at each
        pixel.
        """
        self.max = np.zeros(self.image.shape)
        self.max_sigma = np.zeros(self.image.shape)
        for s, f in self.images.items():
            self.max = np.maximum(self.max, f.image)
            idx = self.max == f.image
            self.max_sigma[idx] = s

    def set_scores(self):
        """
        Computes the scores returned by `get_scores`.
        """
        for s, f in self.images.items():
            f.response = np.sum(f.image * self.mask)
            idx = (self.max_sigma == s) * self.mask
            f.dominance = np.sum(idx)
            f.contribution = np.sum(idx * self.max)

    def get_scores(self):
        """
        Gets the scores associated with each length scale.
        Scores returned:
            - Response: sum of filter response at that scale
            - Dominance: number of pixels which responded most intensely at
              the given scale
            - Contribution: sum of filter response at dominant pixels
            - Intensity: average filter response at dominant pixels,
              equal to Contribution/Dominance

        Returns
            scales : list
                List of scales the filter was run at.
            scores : dict
                Dictionary with 4 entries, each of which is a
                list the same length as scales.
        """
        scores = {
            'response': [f.response for f in self.images.values()],
            'dominance': [f.dominance for f in self.images.values()],
            'contribution': [f.contribution for f in self.images.values()],
            'intensity': [f.contribution / f.dominance
                          for f in self.images.values()],
        }
        return list(self.images.keys()), scores

    @staticmethod
    def get_threshold(image, method=None, kwargs=None):
        """
        Ensures that the thresholding method is valid, defaulting to Otsu's
        method.
        """
        if method is None:
            method = threshold_otsu
        if kwargs is None:
            kwargs = {}
        return method(image, **kwargs)

    def get_filtered(self, sigmas):
        """
        For the specified filter length scales, take the pointwise maximum.
        """
        filtered = np.zeros(self.image.shape)
        for s in sigmas:
            filtered = np.maximum(filtered, self.images[s].image)
        return filtered

    def get_foreground(self, intensity_fraction=0.9,
                       t_method=None, t_kwargs=None):
        """
        Thresholds the output of Frangi's filter by taking the pointwise
        maximum of filter images at length scales which are nearly as intense
        as the length scale which contributed most intensely.
        """
        intensity = np.array([f.contribution / f.dominance
                              for f in self.images.values()])
        sigmas = np.array(list(self.images.keys()))
        idx = np.nonzero(intensity >= intensity_fraction * np.amax(intensity))
        t_image = self.get_filtered(sigmas[idx[0]])
        threshold = TubenessImage.get_threshold(t_image, t_method, t_kwargs)
        return self.max > threshold

    def set_skeleton(self, intensity_fraction=0.9, opening_radius=2,
                     t_method=None, t_kwargs=None, s_method=None):
        """
        Takes the foreground of the image, removes spurious blobs and branches
        by morphological opening, then skeletonizes the result.

        Parameters:
            intensity_fraction: float 
                See `get_foreground`
            opening_radius: int
                The disk diameter used for opening
            t_method: callable (ndarray -> float)
                See `get_foreground`
            t_kwargs: dict
                See `get_foreground`
            s_method: callable (ndarray -> ndarray)
                Skeletonization method, defaults to `skeletonize`
        """
        self.mask = self.get_foreground(intensity_fraction, t_method, t_kwargs)
        if opening_radius != 0:
            self.mask = opening(self.mask, disk(opening_radius))
        if s_method is None:
            s_method = skeletonize
        self.mask = s_method(self.mask)

    def get_diameter_count(self):
        masked = self.mask * self.max_sigma
        sigmas = np.fromiter(self.images.keys(), dtype=float)
        count = [np.sum(masked == s) for s in sigmas]
        return sigmas, count

    def get_diameters(self):
        masked = self.mask * self.max_sigma
        idx = np.nonzero(masked)
        return [masked[tup] for tup in zip(idx[0], idx[1])]

    def get_orientations(self):
        masked = self.mask * self.max_sigma
        idx = np.nonzero(masked)
        return [self.images[masked[tup]].get_orientation(tup)
                for tup in zip(idx[0], idx[1])]


def load(path):
    """
    Opens an image and converts it to grayscale.

    Parameters:
        path : str
            The file to open.
    Returns:
        image : ndarray
            The image as a 2D grayscale image.
    """
    image = imread(path)
    if len(image.shape) == 3:
        image = rgb2gray(image)
    return image


class PoreImage:
    def __init__(self, tubeImage):
        self.tube = tubeImage

    def segment(self, mode='i', threshold=None, close=0):
        if mode == 'i':
            if threshold is None:
                threshold = 0.9
            intensity = np.array([f.contribution / f.dominance
                                  for f in self.tube.images.values()])
            sigmas = np.array(list(self.tube.images.keys()))
            idx = np.nonzero(intensity >= threshold * np.amax(intensity))
            t_image = self.tube.get_filtered(sigmas[idx[0]])
            threshold = threshold_otsu(t_image)
            self.foreground = self.tube.max > threshold
        elif mode == 'v':
            if threshold is None:
                threshold = threshold_otsu(self.tube.max)
            self.foreground = self.tube.max > threshold
        elif mode == 'f':
            if threshold is None:
                threshold = threshold_otsu(self.tube.image)
            self.foreground = self.tube.image > threshold
        elif mode == 'h':
            # threshold_otsu(self.tube.image)
            self.foreground = self.tube.image > threshold
            # if threshold is None:
            #    threshold = 0.9
            intensity = np.array([f.contribution / f.dominance
                                  for f in self.tube.images.values()])
            sigmas = np.array(list(self.tube.images.keys()))
            idx = np.nonzero(intensity >= threshold * np.amax(intensity))
            t_image = self.tube.get_filtered(sigmas[idx[0]])
            threshold = threshold_otsu(t_image)
            self.foreground |= self.tube.max > threshold

        if close != 0:
            self.foreground = closing(self.foreground, disk(close))

    def set_foreground(self, i_factor=0.9, c_radius=2, o_radius=2):
        tf = self.tube.get_foreground(i_factor)
        if c_radius != 0:
            tf = closing(tf, disk(c_radius))
        fg = self.tube.image > threshold_otsu(self.tube.image)
        if o_radius != 0:
            fg = opening(fg, disk(o_radius))
        self.foreground = tf | fg

    def analyze(self):
        self.labelled, self.max_label = label(
            ~self.foreground, return_num=True, connectivity=1)
        self.areas = [np.sum(self.labelled == k)
                      for k in range(1, self.max_label + 1)]
