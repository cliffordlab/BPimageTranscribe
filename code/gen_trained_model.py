#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File to preprocess input images into training dataset and generate a CNN model trained on images defined in the study. 
Input: 
    1. Dataset/data/: Folder consisting of input RGB BP meter images
    2. Dataset/labels.csv: File storing LCD numerical and quality data for each image (Format: filename | SBP | DBP | quality)
Intermediary Output:
    1. Dataset/training_data/frames_BP/ : Folder to save preprocessed binary thresholded single monitor frames as training data 
    2. Dataset/training_data/frame_labels.csv: File(.csv) saving LCD numerical and quality data for each single monitor frame

Output: CNN model saved as Dataset/training_data/best_model.h5

@author: skulk26
"""

import numpy as np
import cv2
import os
import shutil
import pandas as pd
import glob
from helper_functions import adjust_gamma, get_lcd, imgs_to_array
from CNN_Model import Model_Multi
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
from sklearn.model_selection import train_test_split
session_config = tf.compat.v1.ConfigProto()
session_config.gpu_options.visible_device_list = "0"
session_config.gpu_options.allow_growth = True
set_session(tf.compat.v1.Session(config=session_config))


#Input files (input images in Dataset/data/ and labels.csv)
root = '../Dataset/'
src_folder = 'Dataset/data/'
df = pd.read_csv(root + 'labels.csv', index_col =0)

dest_folder = 'Dataset/training_data/'
if not os.path.exists(bpf):
    os.path.makedirs(dest_folder)
bpf= dest_folder+'frames_BP/'    
if os.path.exists(bpf):
    shutil.rmtree(bpf)
    os.makedirs(bpf)
else:
    os.makedirs(bpf)    


dict_lst = []
for fname in glob.glob(src_folder +'*.jpg'):
    preprocessed_img = get_lcd(fname)

    #Save Systolic and diastolic contours with _SP.jpg and _DP.jpg suffix respectively            
    tag=os.path.basename(fname).split('.')[0]        #Filename Tag for LCD frames
    cv2.imwrite(bpf + tag+'_SP.jpg', final_img[0:int(h/2),0:w])
    cv2.imwrite(bpf + tag+'_DP.jpg', final_img[int(h/2):h,0:w])

    img_info = df.loc[df['filename'].contains(tag), ['SBP', 'DBP', 'quality']].values()
    #Adding image data into labels file
    dict_lst.append({'filename': bpf + tag+'_SP.jpg', 'true_value' : img_info[0], 'quality': img_info[2]})
    dict_lst.append({'filename': bpf + tag+'_DP.jpg', 'true_value' : img_info[1], 'quality': img_info[2]})


frame_labels = pd.DataFrame(dict_lst, True)
frame_labels.to_csv(dest_folder+"frame_labels.csv")   
print("Single monitor frames and label files created!")


# Convert input single monitor frames into numpy array used for training CNN model
df = frame_labels[['filename', 'quality']]
df['d1']=np.where(df['true']<100, '10', df['true'].floordiv(100)).astype(int)
df['d2']=np.where(df['true']<100, df['true'].floordiv(10), (df['true'].mod(100)).floordiv(10)).astype(int)
df['d3']=df['true'].mod(10).astype(int)

#Generate training and validation dataset in ratio 3:1 for the proposed CNN model 
X =  df['filename']
y = df[['d1', 'd2', 'd3', 'quality']]            
ids_train, ids_val, info_train, info_val = train_test_split(X, y, test_size=0.25, random_state=1)        

#Reset index
ids_train, ids_val = ids_train.reset_index(drop=True), ids_val.reset_index(drop=True)
info_train,  info_val = info_train.reset_index(drop=True), info_val.reset_index(drop=True) 

#Convert images into numpy array(X) and labels into list of values(y) for training CNN model
X_train = imgs_to_array(ids_train, dest_folder + 'frames_BP/')
y_train = info_train[['d1', 'd2', 'd3']]

#Convert images into numpy array(X) and labels into list of values(y) for validating the CNN model
X_val = convert_to_arrays(ids_val, dest_folder + 'frames_BP/')
y_val = info_val[['d1', 'd2', 'd3']]
y_val_vect =  [y_val["d1"], y_val["d2"], y_val["d3"]]

#Train CNN model using training & validation dataset

model = Model_Multi(X_train, y_train, X_test, y_test, root +'best_model.h5')
model.train_predict(dest_folder)
print('Trained CNN model saved at {}'.format(root +'best_model.h5'))
