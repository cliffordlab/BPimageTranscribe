#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for helper functions required to transcribe the images
1. adjust_gamma(): Performs gamma correction on input image
2. imgs_to_array(): Converts each image in folder into a numpy array & concatenates them
3. get_lcd(): Takes input image filename & extracts binary thresholded single monitor BP LCD frames from it  
@author: skulk26
"""
import numpy as np
import cv2

def adjust_gamma(image, gamma=1.0):
        """
        Credit: https://stackoverflow.com/questions/33322488/how-to-change-image-illumination-in-opencv-python
        Parameters:
            image: A grayscale image (NxM int array in [0, 255]
            gamma: A positive float. If gamma<1 the image is darken / if gamma>1 the image is enlighten / if gamma=1 nothing happens.
        Returns: the enlighten/darken version of image
        """
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
        return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

def imgs_to_array(filenames, src_folder):
    """
    Function resizes input single monitor images into chosen image size((180 X 80) in paper, and converts into array 
    by reshaping into a (180 x 80 x 1) size array for every image.
    """
    X = []
    for fname in filenames:
        ID =  src_folder + "%s" % (fname)
        img = Image.open(ID)            
        img=img.resize((180,80))
        img = np.array(img)
        img = img.reshape((img.shape[0],img.shape[1],1))
        print(img.shape[0],img.shape[1])
        X.append(img)
    X = np.asarray(X)
    return X

def get_lcd(fname):
    image=cv2.imread(fname)
    #Preprocess image under test
    if image is not None:
        #Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Apply bilateral filter for smoothing test image(noise reduction), while preserving edges    
        blurred = cv2.bilateralFilter(gray, 11, 11, 11)
        #Apply gamma correction to adjust image illumination
        gamma = adjust_gamma(blurred, gamma=0.7)
        #Apply adaptive thresholding with a mean neighborhood threshold calculation as image may have varying illuniation
        adt_thresh=cv2.adaptiveThreshold(gamma,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        #Erode thresholded image using a 3x3 rectangular kernel
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        eroded=cv2.erode(adt_thresh, kernel)
        #Invert image to find contours in image
        inverse=cv2.bitwise_not(eroded)
        _,contours, _ = cv2.findContours(inverse.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)   
        bp_cnt = []
        d=image.copy()  
        #Keep only those contours that respect chosen aspect_ration and size
        for contour in contours:
            x,y,w,h=cv2.boundingRect(contour)
            cv2.rectangle(d,(x,y),(x+w,y+h),(255,0,0),2)
            aspect_ratio=w/h
            size=w*h
            if 2.7 <= aspect_ratio <=4.0 and 7000<= size <25000:# and int(aspect_ratio)!=int(prevh):
                hr_cnt.append(contour)
                prevh=aspect_ratio
        
        
            elif 0.95<= aspect_ratio<=1.5 and 20000<= size <80000:# and int(aspect_ratio)!=int(prevb):
                bp_cnt.append(contour)
                coord=int(str(x)[:2])
                prevb=aspect_ratio
            
        if bp_cnt!=[]:
            print(fname)
            cnt2=max(bp_cnt, key=cv2.contourArea)
            x,y,w,h=cv2.boundingRect(cnt2)
            print("BP",x,y,w,h,w*h,w/h, cv2.contourArea(cnt2))
            upper_left = (int(w / 8), int(h / 16))
            bottom_right = (int(w * 7 / 8), int(h * 15 / 16))
            frame=inverse[y:y+h, x:x+w]
            mask=cv2.rectangle(frame.copy(), upper_left, bottom_right, (0, 0, 0),-1)
            final_img=frame-mask    
            w, h=final_img.shape

            return final_img
