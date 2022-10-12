#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from skimage.feature import hog
from skimage import feature
import cv2
from skimage.feature import canny
from scipy import ndimage
import imageio

def localBinaryPattern(img, num_points, radius):
	eps = 1e-7
	lbp = feature.local_binary_pattern(img, num_points, radius, method = 'uniform')
	
	(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3),
			range=(0, num_points + 2))
	
	hist = hist.astype("float")
	hist /= (hist.sum() + eps)
	
	return hist
 

def largest_contour_rect(saliency):
    contours, hierarchy = cv2.findContours(saliency * 1,
    cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key = cv2.contourArea)
    return cv2.boundingRect(contours[-1])

def backproject(source, target, levels = 2, scale = 1):
    hsv = cv2.cvtColor(source,  cv2.COLOR_BGR2HSV)
    hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    # calculating object histogram
    roihist = cv2.calcHist([hsv],[0, 1], None, \
        [levels, levels], [0, 180, 0, 256] )

    # normalize histogram and apply backprojection
    cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256], scale)
    return dst

def saliency(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    backproj = np.uint8(backproject(img, img, levels = 2))
    cv2.normalize(backproj,backproj,0,255,cv2.NORM_MINMAX)

    saliencies = [backproj, backproj, backproj]
    saliency = cv2.merge(saliencies)
    cv2.pyrMeanShiftFiltering(saliency, 20, 200, saliency, 1)
    saliency = cv2.cvtColor(saliency, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(saliency, saliency)
    (T, saliency) = cv2.threshold(saliency, 180, 255, cv2.THRESH_BINARY)
    return saliency

def getSubImage(rect, src):
    # Get center, size, and angle from rect
    center, size, theta = rect
    # Convert to int 
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # Get rotation matrix for rectangle
    M = cv2.getRotationMatrix2D( center, theta, 1)
    # Perform rotation on src image
    dst = cv2.warpAffine(src, M, src.shape[:2])
    out = cv2.getRectSubPix(dst, size, center)
    return out

def largest_N_contours(thresh, N = 3):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    candidates = []
    
    
    i = 0
    if contours:
        while len(candidates) < N and i < len(contours):
            cand = contours[i]
            epsilon = 0.1*cv2.arcLength(cand,True)
            approx = cv2.approxPolyDP(cand,epsilon,True)
            x,y,w,h = cv2.boundingRect(approx)
            roi_area = w * h
            img_area = thresh.shape[0] * thresh.shape[1]
            
            roi = thresh[y:y+h, x:x+w]
            n_black_pixel = np.sum(roi==0)

            if n_black_pixel / (roi.shape[0] * roi.shape[1]) > 0.45:
                candidates.append(cand)
            
            if len(candidates) > N:
                break
            i += 1
    return candidates

def edge_based_segmentated(img):
    edges = canny(img / 255.)
    filled = ndimage.binary_fill_holes(edges)
    return np.uint8(filled)
    
        
def refine_saliency_with_grabcut(img, saliency):
    rect = largest_contour_rect(saliency)
    bgdmodel = np.zeros((1, 65),np.float64)
    fgdmodel = np.zeros((1, 65),np.float64)
    saliency[np.where(saliency > 0)] = cv2.GC_FGD
    mask = saliency
    cv2.grabCut(img, mask, rect, bgdmodel, fgdmodel, \
                1, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    return mask


def extract_lbp(imgs, N = 24, R = 16):
    hist = []
    for img in imgs:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = localBinaryPattern(img_gray, N, R)
        hist.append(lbp)
        
    return np.array(hist)


def hist3D(imgs):
    hist = np.zeros((len(imgs),8000))
    for n in range(len(imgs)):
        hist[n] = cv2.calcHist([imgs[n]], [0,1,2], None, [20,20,20], [0,256,0,256,0,256]).flatten()
    return hist


def calcHist(imgs):
    hist = []
    for img in imgs:
        h = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist.append(h.flatten())
    
    return np.array(hist)


def computeHogArray(X, visualize = 'True', PPC = 16, CPB = 1):
    hog_imgs = np.empty((len(X),64, 64))
    fd = np.empty((len(X),128))
    for idx, fname in enumerate(X):
        fd[idx], hog_imgs[idx] = hog(X[idx], orientations=8, pixels_per_cell = (PPC,PPC),
                          cells_per_block=(CPB,CPB), visualize=visualize, 
                          feature_vector = True, block_norm='L2')
    
    return fd

def rle_decode(mask_rle, mask_value=255, shape=(768,768)):
    ## this function convert RLE encoding into image_mask
    if type(mask_rle) == float: return None  ## My way of reading 
    
    if type(mask_rle) == str:
        mask_rle = [mask_rle]
    
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)  ## initialzing images to all 0
    for mask in mask_rle:
        s = mask.split()
        starts, lengths = np.asarray(s[0::2], dtype=int)-1, np.asarray(s[1::2], dtype=int)
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            img[lo:hi] = mask_value
    return img.reshape(shape).T

def rle_to_vertices(mask_rle, return_img=False, shape=(768,768)):
    ## This function finding out the center, length, width, angle of the RLE and return these parameters plus the image with box countour
    if type(mask_rle) == float: return None
    mask_img = rle_decode(mask_rle, shape=shape) # Generate masked images
    ret, mask_img = cv2.threshold(mask_img, 127, 255, 0) 
    contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        ## Storing center information so to assign ship to proper grid
        center_x = int(rect[0][0])
        center_y = int(rect[0][1])
        len_x = int(rect[1][0])
        len_y = int(rect[1][1])
        angle = int(rect[2])
        boxes.append([center_x, center_y, len_x, len_y, angle]) 
    if return_img:
        for center_x, center_y, len_x, len_y, angle in boxes:
            box = cv2.boxPoints(((center_x, center_y), (len_x, len_y),angle))
            box = np.int0(box)
            cv2.drawContours(mask_img,[box],0,200,1)
        return boxes, mask_img
    else:
        return boxes
