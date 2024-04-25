import math
import numpy as np
import cv2 as cv
import urllib.request
import IPython
import base64
import html
import utils
from utils import *

class Fingerprint:

    def __init__(self, enrolfile):
        self.mask = None
        self.enhanced = None  # Start from fingerprint_process()
        self.cn = None  # Start from detect_minutiae_pos_and_dir()
        self.valid_minutiae = None  # Start from detect_minutiae_pos_and_dir()
        self.local_structures = None  # Start from create_local_structures()
        self.fingerprint = cv.imread(enrolfile, cv.IMREAD_GRAYSCALE)

    def fingerprint_process(self):
        # *** fingerprint segmentation ***
        # Calculate the local gradient (using Sobel filters)
        gx, gy = cv.Sobel(self.fingerprint, cv.CV_32F, 1, 0), cv.Sobel(self.fingerprint, cv.CV_32F, 0, 1)
        
        # Calculate the magnitude of the gradient for each pixel
        gx2, gy2 = gx**2, gy**2
        gm = np.sqrt(gx2 + gy2)
        
        # Integral over a square window
        sum_gm = cv.boxFilter(gm, -1, (25, 25), normalize = False)
        
        # Use a simple threshold for segmenting the fingerprint pattern ***
        thr = sum_gm.max() * 0.2
        self.mask = cv.threshold(sum_gm, thr, 255, cv.THRESH_BINARY)[1].astype(np.uint8)

        # *** Estimate local ridge orientation ***
        W = (23, 23)
        gxx = cv.boxFilter(gx2, -1, W, normalize = False)
        gyy = cv.boxFilter(gy2, -1, W, normalize = False)
        gxy = cv.boxFilter(gx * gy, -1, W, normalize = False)
        gxx_gyy = gxx - gyy
        gxy2 = 2 * gxy

        orientations = (cv.phase(gxx_gyy, -gxy2) + np.pi) / 2 # '-' to adjust for y axis direction

        # *** Estimate local ridge frequency ***
        region = self.fingerprint[10:90,80:130]
        
        # before computing the x-signature, the region is smoothed to reduce noise
        smoothed = cv.blur(region, (5,5), -1)
        xs = np.sum(smoothed, 1) # the x-signature of the region
        
        # Find the indices of the x-signature local maxima
        local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]
        
        # Calculate all the distances between consecutive peaks
        distances = local_maxima[1:] - local_maxima[:-1]

        # Estimate the ridge line period as the average of the above distances
        ridge_period = np.average(distances)

        # *** Enhance fingerprint ***
        # Create the filter bank
        or_count = 8
        gabor_bank = [gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi/or_count)]
        
        # Filter the whole image with each filter
        # Note that the negative image is actually used, to have white ridges on a black background as a result
        nf = 255-self.fingerprint
        all_filtered = np.array([cv.filter2D(nf, cv.CV_32F, f) for f in gabor_bank])
        y_coords, x_coords = np.indices(self.fingerprint.shape)
        # For each pixel, find the index of the closest orientation in the gabor bank
        orientation_idx = np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32) % or_count
        # Take the corresponding convolution result for each pixel, to assemble the final result
        filtered = all_filtered[orientation_idx, y_coords, x_coords]
        # Convert to gray scale and apply the mask
        self.enhanced = self.mask & np.clip(filtered, 0, 255).astype(np.uint8)

    def detect_minutiae_pos_and_dir(self):

        # Binarization
        _, ridge_lines = cv.threshold(self.enhanced, 32, 255, cv.THRESH_BINARY)
        
        # Thinning
        skeleton = cv.ximgproc.thinning(ridge_lines, thinningType = cv.ximgproc.THINNING_GUOHALL)
        
        # Create a filter that converts any 8-neighborhood into the corresponding byte value [0,255]
        cn_filter = np.array([[  1,  2,  4],
                              [128,  0,  8],
                              [ 64, 32, 16]
                             ])
        
        # Create a lookup table that maps each byte value to the corresponding crossing number
        all_8_neighborhoods = [np.array([int(d) for d in f'{x:08b}'])[::-1] for x in range(256)]
        cn_lut = np.array([compute_crossing_number(x) for x in all_8_neighborhoods]).astype(np.uint8)
        
        # Skeleton: from 0/255 to 0/1 values
        skeleton01 = np.where(skeleton!=0, 1, 0).astype(np.uint8)
        
        # Apply the filter to encode the 8-neighborhood of each pixel into a byte [0,255] 
        neighborhood_values = cv.filter2D(skeleton01, -1, cn_filter, borderType = cv.BORDER_CONSTANT)
        
        # Apply the lookup table to obtain the crossing number of each pixel from the byte value of its neighborhood
        cn = cv.LUT(neighborhood_values, cn_lut)
        
        # Keep only crossing numbers on the skeleton
        cn[skeleton==0] = 0
        
        # crossing number == 1 --> Termination, crossing number == 3 --> Bifurcation
        minutiae = [(x,y,cn[y,x]==1) for y, x in zip(*np.where(np.isin(cn, [1,3])))]
        
        # A 1-pixel background border is added to the mask before computing the distance transform
        mask_distance = cv.distanceTransform(cv.copyMakeBorder(self.mask, 1, 1, 1, 1, cv.BORDER_CONSTANT), cv.DIST_C, 3)[1:-1,1:-1]
        filtered_minutiae = list(filter(lambda m: mask_distance[m[1], m[0]]>10, minutiae))

        # *** Estimate_minutiae_directions ***
        r2 = 2**0.5 # sqrt(2)

        # The eight possible (x, y) offsets with each corresponding Euclidean distance
        xy_steps = [(-1,-1,r2),( 0,-1,1),( 1,-1,r2),( 1, 0,1),( 1, 1,r2),( 0, 1,1),(-1, 1,r2),(-1, 0,1)]

        # LUT: for each 8-neighborhood and each previous direction [0,8], 
        #      where 8 means "none", provides the list of possible directions
        nd_lut = [[compute_next_ridge_following_directions(pd, x) for pd in range(9)] for x in all_8_neighborhoods]
        
        self.valid_minutiae = []
        for x, y, term in filtered_minutiae:
            d = None
            if term: # termination: simply follow and compute the direction        
                d = follow_ridge_and_compute_angle(cn, nd_lut, xy_steps, neighborhood_values, x, y)
            else: # bifurcation: follow each of the three branches
                dirs = nd_lut[neighborhood_values[y,x]][8] # 8 means: no previous direction
                if len(dirs)==3: # only if there are exactly three branches
                    angles = [follow_ridge_and_compute_angle(cn, nd_lut, xy_steps, neighborhood_values, x+xy_steps[d][0], y+xy_steps[d][1], d) for d in dirs]
                    if all(a is not None for a in angles):
                        a1, a2 = min(((angles[i], angles[(i+1)%3]) for i in range(3)), key=lambda t: angle_abs_difference(t[0], t[1]))
                        d = angle_mean(a1, a2)                
            if d is not None:
                self.valid_minutiae.append( (x, y, term, d) )
        # print(self.valid_minutiae)

    def create_local_structures(self):

        # Compute the cell coordinates of a generic local structure
        mcc_radius = 70
        mcc_size = 16

        g = 2 * mcc_radius / mcc_size
        x = np.arange(mcc_size)*g - (mcc_size/2)*g + g/2
        y = x[..., np.newaxis]
        iy, ix = np.nonzero(x**2 + y**2 <= mcc_radius**2)
        ref_cell_coords = np.column_stack((x[ix], x[iy]))

        mcc_sigma_s = 7.0
        mcc_tau_psi = 400.0
        mcc_mu_psi = 1e-2

        # n: number of minutiae
        # c: number of cells in a local structure

        xyd = np.array([(x,y,d) for x,y,_,d in self.valid_minutiae]) # matrix with all minutiae coordinates and directions (n x 3)

        # rot: n x 2 x 2 (rotation matrix for each minutia)
        d_cos, d_sin = np.cos(xyd[:,2]).reshape((-1,1,1)), np.sin(xyd[:,2]).reshape((-1,1,1))
        rot = np.block([[d_cos, d_sin], [-d_sin, d_cos]])

        # rot@ref_cell_coords.T : n x 2 x c
        # xy : n x 2
        xy = xyd[:,:2]
        # cell_coords: n x c x 2 (cell coordinates for each local structure)
        cell_coords = np.transpose(rot@ref_cell_coords.T + xy[:,:,np.newaxis],[0,2,1])

        # cell_coords[:,:,np.newaxis,:]      :  n x c  x 1 x 2
        # xy                                 : (1 x 1) x n x 2
        # cell_coords[:,:,np.newaxis,:] - xy :  n x c  x n x 2
        # dists: n x c x n (for each cell of each local structure, the distance from all minutiae)
        dists = np.sum((cell_coords[:,:,np.newaxis,:] - xy)**2, -1)

        # cs : n x c x n (the spatial contribution of each minutia to each cell of each local structure)
        cs = np.exp(-0.5 * dists / (mcc_sigma_s**2)) / (math.tau**0.5 * mcc_sigma_s)
        diag_indices = np.arange(cs.shape[0])
        cs[diag_indices,:,diag_indices] = 0 # remove the contribution of each minutia to its own cells

        # local_structures : n x c (cell values for each local structure)
        self.local_structures = 1. / (1. + np.exp(-mcc_tau_psi * (np.sum(cs, -1) - mcc_mu_psi)))
        # print(self.local_structures)


