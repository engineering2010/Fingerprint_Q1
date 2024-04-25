import math
import numpy as np
import cv2 as cv
import os

def fingerprint_comparison(m1, ls1, ofn):
        
        # f1, m1, ls1 = self.fingerprint, self.valid_minutiae, self.local_structures
        
        basedPath = 'D:\Fingerprint\sample'  # Need changed ******

        realPath = os.path.join(basedPath, ofn)

        #ofn = 'samples/sample_1_2' # Fingerprint of a different finger
        (m2, ls2) = np.load(realPath, allow_pickle=True).values()
        
        # Compute all pairwise normalized Euclidean distances between local structures in v1 and v2
        # ls1                       : n1 x  c
        # ls1[:,np.newaxis,:]       : n1 x  1 x c
        # ls2                       : (1 x) n2 x c
        # ls1[:,np.newaxis,:] - ls2 : n1 x  n2 x c 
        # dists                     : n1 x  n2
        dists = np.linalg.norm(ls1[:,np.newaxis,:] - ls2, axis = -1)
        dists /= np.linalg.norm(ls1, axis = 1)[:,np.newaxis] + np.linalg.norm(ls2, axis = 1) # Normalize as in eq. (17) of MCC paper

        # Select the num_p pairs with the smallest distances (LSS technique)
        num_p = 5 # For simplicity: a fixed number of pairs
        pairs = np.unravel_index(np.argpartition(dists, num_p, None)[:num_p], dists.shape)
        score = 1 - np.mean(dists[pairs[0], pairs[1]]) # See eq. (23) in MCC paper

        return score