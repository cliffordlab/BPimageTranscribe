#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script for text detection using Google vision API 

@author: nasimkatebi
"""


import argparse
import io
import pandas as pd
from google.cloud import vision
import os
from PIL import  ImageDraw,Image

def draw_boxes(image, bounds, color,width=5):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        draw.line([
            bound.vertices[0].x, bound.vertices[0].y,
            bound.vertices[1].x, bound.vertices[1].y,
            bound.vertices[2].x, bound.vertices[2].y,
            bound.vertices[3].x, bound.vertices[3].y,
            bound.vertices[0].x, bound.vertices[0].y],fill=color, width=width)
    return image

def main(args):
    """Main function for transcribing BP images using Google vison API.
    
    Args:
        - data_dir
        - saving_dir
        - data_label
        - draw_boxes
        - Image_with_box_dir

    Returns:
        - Google_API_results.csv
        - Images with detected boxes (if draw_boxes==True )

    """
    # creating result folders
    if not os.path.exists(args.saving_dir):
        os.mkdir(args.saving_dir)
    if args.draw_boxes:

        if not os.path.exists(args.Image_with_Box_directory):
            os.mkdir(args.Image_with_Box_directory)



    #Loading filenames
    bp_data=pd.read_csv(args.data_label)
    # creating result dataframe
    df = pd.DataFrame(columns = ['filename','text'])
    df['filename']=bp_data['filename']

    # Instantiates a client
    client = vision.ImageAnnotatorClient()


    for i in bp_data.filename.index:
        image_filename= args.data_dir+ bp_data.filename[i]
        # Loading the image into memory
        with io.open(image_filename, 'rb') as image_file:
            content = image_file.read()
            content_image = vision.Image(content=content)
            response = client.text_detection(image=content_image)
            result=[]
            bounds=[]
            for text in response.text_annotations:
                print(text)
                result.append(text.description+',')
                if args.draw_boxes:
                    bounds.append(text.bounding_poly) # vertices of the detected boxes
     


        df['text'].iloc[i]=result

        if args.draw_boxes:
            image=Image.open(image_filename)
            image  = image.transpose(Image.ROTATE_270)
            image_withBox=draw_boxes(image, bounds, 'yellow')
            name=bp_data.filename[i]
            image_withBox.save( args.Image_with_box_dir+name[:-4]+'bound.jpg')       
    df.to_csv(args.saving_dir+'GoogleAPIresults.csv')




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
    help='The directory to save the results of the Google vision API text detection.')
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