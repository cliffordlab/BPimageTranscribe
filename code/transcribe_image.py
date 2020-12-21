#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Main script performing serial operations to transcribe a test image into numerical values as under:

@author: skulk26 
"""
import os
from keras.models import load_model
import numpy as np
import pandas as pd
from helper_functions import imgs_to_array, get_lcd
import glob
"""
Directory structure:
    Test_case/: 
        2. test_data/ :  
            2.1 Sample images under test
            2.2 'labels.csv' to save filename of each test image
        3. results/: 
            3.1 frames_BP/ : Folder to save test preprocessed binary single monitor lcd_data 
            3.2 labels_BP/: File(.csv) saving filename of images in frames_BP/ folder
            3.3 test_data_predictions.csv: Transcription results for every test image (Format: filename | predicted_SBP| predicted_DBP)
"""
root = '../Test_case/'
data = root + 'test_data/'

results = root + 'results/'
if not os.path.exists(results):
    os.mkdir(results)
lcd_data = root + 'frames_BP/'
if not os.path.exists(lcd_data):
    os.mkdir(lcd_data)

#Load CNN model trained on data pre-defined in the paper
model=load_model('../Dataset/best_model.h5')

# Preprocess all images in test_data folder into binarized single monitor lcd_data with a label file
frame_labels = pd.DataFrame(columns = ['filename'])
for fname in glob.glob(data +'*.jpg'):
    preprocessed_img = get_lcd(fname)

    #Save Systolic and diastolic contours with _SP.jpg and _DP.jpg suffix respectively            
    tag=os.path.basename(fname).split('.')[0]        #Filename Tag for LCD lcd_data
    cv2.imwrite(lcd_data + tag+'_SP.jpg', final_img[0:int(h/2),0:w])
    cv2.imwrite(lcd_data + tag+'_DP.jpg', final_img[int(h/2):h,0:w])

    #Adding image data into labels file
    frame_labels.filename.append(lcd_data + tag+'_SP.jpg')
    frame_labels.filename.append(lcd_data + tag+'_DP.jpg')

frame_labels.to_csv(lcd_data+"labels_BP.csv")   
print("Single monitor lcd_data and label files created!")


#Transcription of all images in test folder
X_test = imgs_to_array(lcd_data.filename[:], lcd_data)
y_pred = model.predict(X_test)

# Create test_data_predictions.csv file to save Systolic and Diastolic BP values for every test image
df.columns = pd.DataFrame(columns=['filename', 'predicted_SBP', 'predicted_DBP'])
for i in range(X_test.shape[0]):
    pred_list_i = [np.argmax(pred[i]) for pred in y_pred]
    predicted_num = 100* pred_list_i[0] + 10 * pred_list_i[1] + 1* pred_list_i[2]
    if predicted_num >= 1000:
        predicted_num = predicted_num-1000

    fname = lcd_data.filename[i]         
    if fname.contains('_SP'):
        df.loc[df['filename']==fname.strip('_SP.jpg'), 'predicted_SBP'] = predicted_num
    elif fname.contains('_DP'):
        df.loc[df['filename']==fname.strip('_DP.jpg'), 'predicted_DBP'] = predicted_num        
df.csv(results + 'test_data_predictions.csv')
