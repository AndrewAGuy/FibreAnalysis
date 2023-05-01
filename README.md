# FibreAnalysis

Image processing intended for use in characterising densified collagen fibres in SEM images.
Developed for and published in: 
- A. W. Justin et al., ‘Densified collagen tubular grafts for human tissue replacement and disease modelling applications’, Biomaterials Advances, vol. 145, p. 213245, Feb. 2023, [doi: 10.1016/j.bioadv.2022.213245](https://doi.org/10.1016/j.bioadv.2022.213245).


### Installation
```sh
pip install git+https://github.com/AndrewAGuy/FibreAnalysis.git
```

### Basic usage
```py
from FibreAnalysis import load, FibreImage, PoreImage

path = '...'
sigmas = [...]

I = load(path)
F = FibreImage(I, sigmas)
P = PoreImage(F)

# Skeleton statistics in pixels and radians
F.set_skeleton()
radii = F.get_radii()
angles = F.get_orientations()

# Pore statistics
properties = P.get_props()
porosity = P.porosity()

# Debug info, skeleton
F.set_scores()
scores = F.get_scores()         # Object with attributes:
                                # 'scales', 'response', 'dominance',
                                # 'contribution', 'intensity',
                                # which are arrays of equal length
fibres = F.get_foreground()     # 2D binary image, size of I
skeleton = F.mask               # 2D binary image, size of I
# More debug info is available by inspecting:
#   F.images        dict: float -> FilteredImage
#   F.max           2D float image, size of I
#   F.max_sigma     2D float image, size of I

# Debug info, pores
pores = ~P.foreground           # 2D binary image, size of I
```

## Why?
Traditional ImageJ plugins (Frangi, DiameterJ, Tubeness) as well as other options for analyzing tube-like data (AngioTool) weren't working for our use-case due to the following:
- Fibres were falling below the minimum diameter for some plugins, and the foreground contains merged fibre bundles.
- Many fibres were merging together and giving responses at higher diameters.
- Edge highlights were giving a response at lower diameters.

This was leading to poor skeletonization and obviously incorrect results for diameter and connectivity.

### Why not ML?
Lack of data and motivation to label it, set up training and evaluation etc.
I'm sure it's a far better approach, and you could easily replace the foreground method with ML if you put the effort in.

## How this approach works
We accept that extracting fibre connectivity data from these images is infeasible (for fibres which have sections in contact with each other, how should this even be treated?).
We therefore go looking for fibres which can be clearly extracted from the image, take their centrelines and take summary statistics from these, i.e., we are looking at the fraction of points identified in an image as a fibre centreline that have a given diameter and orientation.

### The filter
The filter used is Frangi's vesselness filter, a classical filter that looks at the eigenvalues of the Hessian matrix of the image, smoothed at a given length scale.
The length scale is the fibre radius, as the second derivative of the Gaussian has its zero-crossings at this point.

Orientations of the tube-like structures can be extracted at a point by considering the eigenvector associated with the near-zero eigenvalue of the smoothed Hessian, as this is the direction in which the image intensity does not change.

### Extracting the skeleton
Simply segmenting the image does not work, as this is noisy and leads to skeletons which oversample the fibre edges and centres of bundles.
Closing the image does not help either, as this still places points in the bundles.
Segmenting the output of Frangi's filter does not fare much better, as spurious fibres are detected in the noise, fibre edges and bundles.

Our solution was developed by inspecting a number of properties of the filter outputs at each scale:
- Response: the sum over the output image at that scale (interpret as the average probability of a pixel in the image being a fibre at that scale)
- Dominance: the number of pixels in the MIP over all scales that were taken from this scale
- Contribution: the sum over those pixels (the total "fibre probability" contributed at this scale)
- Intensity = Contribution / Dominance: the average contribution per pixel

We can see that even though there is a total high response over the larger length scales (bundles) and many pixels in the output of the multi-scale Frangi filter came from this, the intensity of those is very low.
We therefore construct a new image from the filter outputs for length scales with an intensity above a certain fraction of the peak (default = 0.9), and determine the filter threshold from this using Otsu's method.
We then take the foreground, and remove objects below a certain size by opening the image with a given radius (default = 2), which suppresses any spurious fibres detected in the edges of larger fibres.
Finally, this foreground is skeletonized.

### Pores

Pores are segmented using a similar approach, in which we find the most dominant fibres and threshold them.
We then use the distribution of intensities in these fibres to set a foreground threshold for the original image, ensuring that regions of high constant intensity (a solid region) are not picked up as a pore.
This foreground is then closed to reduce the impact of noise, then labelled and region statistics extracted.

Note that we have a trade-off, and it may be worth running multiple parameter values (and knowing reasonable length scales beforehand):
- Closing removes small pores, and there is a minimum pore size after this is complete (the stencil used in the erosion applied to a single pixel).
We therefore remove pores below a certain size.
- In regions of noise or low signal, pores can merge.
Countering this (lower threshold, higher closing radius) eliminates the genuine pores at smaller scales.