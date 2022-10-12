#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import hog
import imageio
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
import pickle
import os
from sub_utility import hist3D, computeHogArray, rle_decode, rle_to_vertices, saliency, largest_N_contours, calcHist, extract_lbp, localBinaryPattern, getSubImage
import mahotas as mt
from sklearn.ensemble import AdaBoostClassifier


def returnCandidates(img, i):
    img = cv2.GaussianBlur(img,(5,5),0)
    
    b = saliency(img)
    
    contours, hier = cv2.findContours(np.uint8(b), cv2.RETR_LIST,  cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:3]
    
    candidates = []
    for c in cnts:
        rect = cv2.minAreaRect(c)
        center_x = int(rect[0][0])
        center_y = int(rect[0][1])
        len_x = int(rect[1][0])
        len_y = int(rect[1][1])
        angle = int(rect[2])
        
        if len_x < 400 and len_y < 400:
            candidates.append([center_x, center_y, len_x, len_y, angle])
            x1 = int(center_x - len_x/2)
            x2 = int(center_x + len_x/2)
            y1 = int(center_y - len_y/2)
            y2 = int(center_y + len_y/2)
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 1)

    imageio.imwrite(f'test{i}.jpg', img)
    return candidates

def shipBouningBox(img, center, side_len):
    box = cv2.boxPoints((center, (side_len, side_len), 0))
    box = np.int0(box)
    
    cv2.drawContours(img, box, 0, 200, 1)
    
    return img

def IoU(img,boxA,boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
    center_xA, center_yA, len_xA, len_yA, angleA = boxA
    center_xB, center_yB, len_xB, len_yB, angleB = boxB
    
    
    # we find the upper left coordinate of box A
    ul_XA = max(center_xA - len_xA/2,0)
    ul_YA = max(center_yA - len_yA/2,0)
    
    # we find the bottom right coordinate of box A
    br_XA = min(center_xA + len_xA/2,img.shape[1])
    br_YA = min(center_yA + len_yA/2,img.shape[0])
    
        
    # we find the upper left coordinate of box B
    ul_XB = max(center_xB - len_xB/2,0)
    ul_YB = max(center_yB - len_yB/2,0)
    
    # we find the bottom right coordinate of box B
    br_XB = min(center_xB + len_xB/2,img.shape[1])
    br_YB = min(center_yB + len_yB/2,img.shape[0])
    
    dx = min(br_XA,br_XB) - max(ul_XA,ul_XB)
    dy = min(br_YA,br_YB) - max(ul_YA,ul_YB)
    
    if dx >= 0 and dy >= 0:
        inter_area = dx*dy
        union = (len_xA*len_yA) + (len_xB*len_yB) - inter_area
        print(inter_area/union)
     
    # return the intersection over union value

def matchPredictions(clf, img, rle, candidates):
    # get all the candidates in an image
    # get the preedictions for all the images
    # for every prediction that's a ship
    # check the prediction with every rle vertices
    # and compute the iou
    
    
    for box in candidates:
        cx,cy,w,h,a = boxdddafa
        side = int(max(w,h) / 2)
        roi_left = max(cx-side,0)
        roi_right = min(cx+side,img.shape[1])
        roi_top = max(cy-side,0)
        roi_bottom = min(cy+side,img.shape[0])
        
        roi = img[roi_top:roi_bottom,roi_left:roi_right]
        roi = cv2.resize(roi, (roi_size,roi_size))
        fd, hogg = hog(roi, orientations=8, pixels_per_cell = (8,8),
                          cells_per_block=(1,1), visualize='True', 
                          feature_vector = True, block_norm='L2')

        fd = fd.reshape(1,512)
        prd = clf.predict(fd)
        vertices = rle_to_vertices(rle)
        if prd[0] == 1:
            if vertices:
                for vertix in vertices:
                    plt.imshow(roi, cmap = 'gray')
                    plt.show()
                    t = IoU(img,vertix, box)
                    box = cv2.boxPoints(((cx,cy), (w, h), a))
                    ctr = np.array(box).reshape((-1,1,2)).astype(np.int32)
                    cv2.drawContours(img,ctr,0,(200,200,200),1)
                    print(t)

        
 
def findShips(clf, images_df):
    i = 0
    for path,rle in zip(images_df['path'],images_df['EncodedPixels']):
        img = imageio.imread(path, 'jpg')
        cands = returnCandidates(img, i)
        matchPredictions(clf, img, rle, cands)
        i = i + 1
        
def extract_rotated_roi(X, roi_size = 64):

    labels = []
    i = 0
    ship_train = []
    for path, pixels in zip(X['path'],X['EncodedPixels']):
        if type(pixels) == str:
            vertices = rle_to_vertices(pixels)
            x,y,w,h,a = vertices[0]
            if len(vertices) == 0: continue
            try:
                img = imageio.imread(path, 'jpg', pilmode = 'L')
            except FileNotFoundError:
                print(f"Skipping {path}")
                continue
            
            rect = ((x,y),(w,h),(a))
            roi = getSubImage(rect, img)
            if roi is None: continue
        
            if roi.shape[0]/roi.shape[1] < 1.0:
                roi = roi.T
            roi = cv2.resize(roi, (roi_size,roi_size))
            ship_train.append(roi)
            labels.append(1) # no ship
            i += 1
        else:
            img = imageio.imread(path, 'jpg', pilmode = 'L')
            roi = img[0:roi_size,0:roi_size]
            ship_train.append(roi)
            labels.append(0) # no ship
            i += 1
        
    return ship_train, np.array(labels) 
    


def extract_roi(X, roi_size = 64):
    labels = []
    i = 0
    
    ship_train = []
    
    for path, pixels in zip(X['path'],X['EncodedPixels']):
        if type(pixels) == str:
            vertices = rle_to_vertices(pixels)
            x,y,w,h,a = vertices[0]
            if len(vertices) == 0: continue
            try:
                img = imageio.imread(path, 'jpg')
            except FileNotFoundError:
                print(f"Skipping {path}")
                continue
            side = int(max(w,h) / 2) + 32
    
            roi_left = max(x-side,0)
            roi_right = min(x+side,img.shape[1])
            roi_top = max(y-side,0)
            roi_bottom = min(y+side,img.shape[0])
            roi = img[roi_top:roi_bottom,roi_left:roi_right,:]

            if roi.shape[0] == 0 or roi.shape[1] == 0:
                print(f"{path},{x},{y},{w},{h},{a}")
                continue
            
            roi = cv2.resize(roi, (roi_size,roi_size))
            ship_train.append(roi)
            labels.append(1) # no ship
            i += 1
        else:
            img = imageio.imread(path, 'jpg')
            roi = img[0:roi_size,0:roi_size,:]
            ship_train.append(roi)
            labels.append(0) # no ship
            i += 1
            
    return np.array(ship_train), np.array(labels)

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            
    
def measure_score(clf, df, pca = None, scaler = None, PPC = 16, CPB = 4, N = 24, R = 16):
    i = 0
    NumImgs = 0
    
    desc = LocalBinaryPatterns(N, R)
    ship_count = 0
    the_score = 0
    for path, rle in zip(df['path'],df['EncodedPixels']):
        i += 1
        if type(rle) == str:
            is_ship = 1
            ship_location = rle_to_vertices(rle)
        else:
            is_ship = 0
                
            
        img = imageio.imread(path,'jpg')
        saliency_map = saliency(cv2.GaussianBlur(img,(5,5),0))
        
        saliency_map = cv2.copyMakeBorder(saliency_map,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))
        img = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))
        candis = largest_N_contours(saliency_map)
        for cand in candis:
            NumImgs+=1
            epsilon = 0.1*cv2.arcLength(cand,True)
            approx = cv2.approxPolyDP(cand,epsilon,True)
            x,y,w,h = cv2.boundingRect(approx)
            roi = img[y:y+h, x:x+w]
            roi_rs = cv2.resize(roi, (64, 64))
            roi_hog, roi_hog_viz = hog(roi_rs, orientations=8, pixels_per_cell = (PPC,PPC),
                          cells_per_block=(CPB,CPB), visualize='True', 
                          feature_vector = True, block_norm='L2')
            roi_hog = roi_hog[np.newaxis,:]
            
            
            
            hist = hist3D(np.expand_dims(roi_rs,axis=0))
            roi_rs = cv2.GaussianBlur(img,(9,9),0)
            roi_lbp = localBinaryPattern(cv2.cvtColor(roi_rs, cv2.COLOR_BGR2GRAY), N, R)[np.newaxis,:]
            
            fd = np.hstack((roi_hog, roi_lbp, hist))
            if pca is not None:
                fd = pca.transform(fd)
            if scaler:
                fd = scaler.fit_transform(fd)
            prd = clf.predict(fd)
            saliency_map = cv2.rectangle(saliency_map, (x,y), (x+w,y+h), (0, 255, 0), 2)
            
            if prd[0] == 1:
                ship_count += 1
                img = cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0), 2)
            if prd[0] == 0:
                pass
                img = cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
            if is_ship == prd[0]:
                the_score += 1        
    print("SCORE: ",the_score/NumImgs)
    return ship_count
    
def count_ships(X):
    ship_count = 0
    for pixels in X['EncodedPixels']:
        if type(pixels) == str:
            ship_count += 1
            
    return ship_count
    

if __name__ == '__main__':
    from skimage.feature import local_binary_pattern
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    kfold = KFold(10)
    
    new_size = (256, 256)
    dataset = pd.read_pickle('ships_unbalanced_S90_NS10.plk')
    
    for train_idx, test_idx in kfold.split(dataset):
        X_train = dataset.iloc[train_idx]
        X_test = dataset.iloc[test_idx]
        ship_size_list = []
        i = 0
        
        print("Extracting ROI...")
        ship_train, labels_train = extract_roi(X_train)
        
        print("Extracting features...")
        hog_train = computeHogArray(ship_train, PPC = 16, CPB = 4)
        lbp_train = extract_lbp(ship_train, N = 48, R = 16)
        hist_train = hist3D(ship_train)
        training_features = np.hstack((hog_train, lbp_train, hist_train))
        print(training_features.shape)        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=200)
        training_features = pca.fit_transform(training_features)
        
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = 200)
        clf.fit(training_features, labels_train)
        the_s = measure_score(clf,X_test, pca = pca, PPC = 16, CPB = 4, N = 48, R = 16)
    
