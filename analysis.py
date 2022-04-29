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
    The image represents a probability of a pixel being a fibre of radius
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
        (frangi, white fibres on black background) if not specified.
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
                identifying white fibres.
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


def morph(image, radius):
    """
    For a foreground image, modify using opening/closing.

    Parameters:
        image : ndarray
        radius: int
            If positive, close the image. If negative, open.
    """
    if radius > 0:
        return closing(image, disk(radius))
    elif radius < 0:
        return opening(image, disk(-radius))
    else:
        return image


class FibreImage:
    """
    An image, filtered at multiple length scales for extracting statistics.
    Approach is intended to reject spurious fibres where possible, so is very
    conservative when segmenting foreground. Statistics should be interpreted
    as the probability of a randomly selected point on a fibre in the top layer
    having the given characteristic.

    Attributes:
        image : ndarray
            Possibly processed (equalized/normalized)
        images : dict: float -> `FilteredImage`
            Maps filter length scale to filtered image
        mask : ndarray
            Binary mask used for sampling statistics
        max : ndarray
            The maximum response image over all filter scales
        max_sigma : ndarray
            The length scale which responded most intensely
    """

    def __init__(self, image, sigmas,
                 equalize=False, normalize=True,
                 filter_method=None, filter_kwargs=None):
        """
        Preprocesses the image, creates filtered images for specified scales.
        Sets the maximum response and associated length scales.
        Sets mask to all-ones and computes scores for this mask.
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
            'response': np.array([f.response for f in self.images.values()]),
            'dominance': np.array([f.dominance for f in self.images.values()]),
            'contribution': np.array([f.contribution
                                      for f in self.images.values()]),
            'intensity': np.array([f.contribution / f.dominance
                                   for f in self.images.values()]),
        }
        return np.fromiter(self.images.keys()), scores

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
        sigmas = np.fromiter(self.images.keys(), dtype=float)
        idx = np.nonzero(intensity >= intensity_fraction * np.amax(intensity))
        t_image = self.get_filtered(sigmas[idx[0]])
        threshold = FibreImage.get_threshold(t_image, t_method, t_kwargs)
        return self.max > threshold

    def set_skeleton(self, intensity_fraction=0.9, opening_radius=2,
                     t_method=None, t_kwargs=None, s_method=None):
        """
        Takes the foreground of the image, removes spurious blobs and branches
        by morphological opening, then skeletonizes the result.

        Parameters:
            intensity_fraction : float 
                See `get_foreground`
            opening_radius : int
                The disk diameter used for opening (if negative, closes).
            t_method : callable (ndarray -> float)
                See `get_foreground`
            t_kwargs : dict
                See `get_foreground`
            s_method : callable (ndarray -> ndarray)
                Skeletonization method, defaults to `skeletonize`
        """
        self.mask = self.get_foreground(intensity_fraction, t_method, t_kwargs)
        self.mask = morph(self.mask, -opening_radius)
        if s_method is None:
            s_method = skeletonize
        self.mask = s_method(self.mask)

    def get_diameters(self):
        """
        """
        masked = self.mask * self.max_sigma
        idx = np.nonzero(masked)
        return np.array([masked[tup] for tup in zip(idx[0], idx[1])])

    def get_orientations(self):
        """
        """
        masked = self.mask * self.max_sigma
        idx = np.nonzero(masked)
        return np.array([self.images[masked[tup]].get_orientation(tup)
                         for tup in zip(idx[0], idx[1])])


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
    """
    For images with intense fibres in a top layer and less intense in lower 
    layers, estimate the pore statistics in the top layer by identifying 
    top layer fibres.

    Attributes:
        fibres : FibreImage
    """

    def __init__(self, fibreImage):
        """
        """
        self.fibres = fibreImage

    def set_foreground(self, i_factor=0.9, v_radius=2, f_radius=2, fg_std=1):
        """
        Generates the image foreground. Starting from the same method as
        `FibreImage.get_foreground`, opens or closes this based on parameter
        `v_radius`. Treating these prominent fibres as the 'top layer', the
        mean and standard deviation of their pixels is used to generate a
        threshold as `mean - fg_std * std`. This foreground is then 
        opened/closed depending on `f_radius`, and merged with the fibres.

        Parameters:
            i_factor : float
                See `FibreImage.get_foreground`.
            v_radius : float
                See `morph`, applied to fibre foreground.
            f_radius : float
                See `morph`, applied to image foreground.
            fg_std : float
                Generates image threshold from `mean(F) - fg_std * std(F)`,
                where F are fibre pixels.
        """
        tf = self.fibres.get_foreground(i_factor)
        tf = morph(tf, v_radius)

        vfg = self.fibres.image[np.nonzero(tf)]
        fg_thresold = np.mean(vfg) - fg_std * np.std(vfg)
        fg = self.fibres.image > fg_thresold
        fg = morph(fg, f_radius)

        self.foreground = tf | fg

    def analyze(self):
        self.labelled, self.max_label = label(
            ~self.foreground, return_num=True, connectivity=1)
        self.areas = np.array([np.sum(self.labelled == k)
                               for k in range(1, self.max_label + 1)])
