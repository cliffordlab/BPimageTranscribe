#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script for text detection using Tesseract OCR
@author: nasimkatebi
"""

import argparse
import os
import numpy as np
import cv2
import pytesseract
import pandas as pd
from PIL import Image, ImageDraw



def main(args):
    """Main function for transcribing BP images using Tesseract OCR.
    
    Args:
        - data_dir
        - saving_dir
        - data_label
        - draw_boxes
        - Image_with_box_dir

    Returns:
        - TesseractOCRresults.csv
        - Images with detected boxes (if draw_boxes==True )

    """
    # creating result folders
    if not os.path.exists(args.saving_dir):
        os.mkdir(args.saving_dir)
    if args.draw_boxes:

        if not os.path.exists(args.Image_with_box_dir):
            os.mkdir(args.Image_with_box_dir)



    #Loading filenames
    bp_data=pd.read_csv(args.data_label)
    # creating result dataframe
    df = pd.DataFrame(columns = ['filename','text'])
    df['filename']=bp_data['filename']



    # Loading the images to transcribe
    for i in bp_data.filename.index:
        image_filename= args.data_dir+ bp_data.filename[i]
        img = cv2.imread(image_filename)

        # pytesseract config can be changed based on your application 
        # --psm: (Page segmentation modes) specifies the charactristics of the image
        # tessedit_char_whitelist=0123456789 limits the results to just numbers

        result = pytesseract.image_to_string(img,lang='eng',config=' --psm 12 -c tessedit_char_whitelist=0123456789')
        result=result.replace('\n\n',',')
        print((result))
        df['text'].iloc[i]=result


        # saving images with boxes if draw_boxes=True
        if args.draw_boxes:
            h, w, c = img.shape
            boxes = pytesseract.image_to_boxes(img,lang='eng',config=' --psm 12 -c tessedit_char_whitelist=0123456789')
            for b in boxes.splitlines():
                b = b.split(' ')
                img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 255),2)
            name=bp_data.filename[i]
            cv2.imwrite(args.Image_with_box_dir+name[:-4]+'bound.jpg',img)


    
    df.to_csv(args.saving_dir+'TesseractOCRresults.csv')




if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--data_dir',
    default='../Test_case/test_data/',
    type=str,
    help='The directory of the images you want to transcribe.')
    parser.add_argument(
    '--saving_dir',
    default='result/',
    type=str,
    help='The directory to save the results of the Tesseract OCR.')
    parser.add_argument(
    '--data_label',
    default='../Test_case/test_data/labels.csv',
    type=str,
    help='The csv file of the data. It should include "filename" column.')
    parser.add_argument(
    '--draw_boxes',
    default=False,
    help='If True, the code will save the images with the detected boxes in "Image_with_Box_directory"')
    parser.add_argument(
    '--Image_with_box_dir',
    default='result/images/',
    help='The directory for saving the images with detected boxes.')

    args = parser.parse_args() 
  
    # Calls main function  
    main(args)