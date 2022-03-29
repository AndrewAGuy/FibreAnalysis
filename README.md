# FibreAnalysis

Image processing intended for use in determining orientation and diameter of densified collagen fibres in SEM images.

## Problem
Traditional ImageJ plugins and (Frangi, DiameterJ, Tubeness) as well as other options for analyzing tube-like data (AngioTool) weren't working for our use-case due to the following:
- Fibres were falling below the minimum diameter for some plugins, and the foreground contains merged fibre bundles.
- Many fibres were merging together and giving responses at higher diameters.
- Edge highlights were giving a response at lower diameters.

This was leading to poor skeletonization and obviously incorrect results for diameter and connectivity.

## Solution
We accept that extracting fibre connectivity data from these images is infeasible (for fibres which have sections in contact with each other, how should this even be treated?).
We therefore go looking for fibres which can be clearly extracted from the image, take their centrelines and take summary statistics from these --- i.e., we are looking at the fraction of points identified in an image as a fibre centreline that have a given diameter and orientation.

### The filter
The filter used is Frangi's vesselness filter, a classical filter that looks at the eigenvalues of the Hessian matrix of the image, smoothed at a given length scale.
The length scale is the vessel radius, as the second derivative of the Gaussian has its zero-crossings at this point.

Orientations of the tube-like structures can be extracted at a point by considering the eigenvector associated with the near-zero eigenvalue of the smoothed Hessian, as this is the direction in which the image intensity does not change.

### Extracting the skeleton
Simply segmenting the image does not work, as this is noisy and leads to skeletons which oversample the fibre edges and centres of bundles.
Closing the image does not help either, as this still places points in the bundles.
Segmenting the output of Frangi's filter does not fare much better, as spurious vessels are detected in the noise, vessel edges and bundles.

Our solution was developed by inspecting a number of properties of the filter outputs at each scale:
- Response: the sum over the output image at that scale (interpret as the average probability of a pixel in the image being a vessel at that scale)
- Dominance: the number of pixels in the MIP over all scales that were taken from this scale
- Contribution: the sum over those pixels (the total "vessel probability" contributed at this scale)
- Intensity = Contribution / Dominance: the average contribution per pixel

We can see that even though there is a total high response over the larger length scales (bundles) and many pixels in the output of the multi-scale Frangi filter came from this, the intensity of those is very low.
We therefore construct a new image from the filter outputs for length scales with an intensity above a certain fraction of the peak (default = 0.9), and determine the filter threshold from this using Otsu's method.
We then take the foreground, and remove objects below a certain size by opening the image with a given radius (default = 2), which suppresses any spurious fibres detected in the edges of larger fibres.
Finally, this foreground is skeletonized.