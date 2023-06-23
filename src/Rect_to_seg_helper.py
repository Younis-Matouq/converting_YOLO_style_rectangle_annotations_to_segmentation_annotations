#Imports:
import os
import json
import numpy as np
from pathlib import Path
import cv2



###########################################################
def image_reader(img_path):
    '''This function takes a path of an image and then reads it and return an RGB image array.'''
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    img_dim=list(image.shape[:2])
    return image,img_dim

def get_segmentations_from_mask(masks,smooth):
    '''This function takes masks of segmentations and return a list of the polygons coordinates.'''

    segmentations=[]
    for obj in range(len(masks)):
        #check if the object is not empty:
        if masks[obj,:,:,:].int().sum()==0:
            continue
        
        gray=masks[obj,:,:,:].int().data.detach().cpu().numpy().squeeze().astype(np.uint8) #convert mask to gray image

        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#find the contours of the object

        
        contours=np.vstack(contours)
        # approximating polygons from contours
        epsilon = 0.001*cv2.arcLength(contours,True)
        approx = cv2.approxPolyDP(contours,epsilon,True)

        if smooth:
            # Compute the convex hull
            hull = cv2.convexHull(approx)
            hull_points = hull.squeeze()

            segmentations.append(hull_points)
        else:
            segmentations.append(approx)
    
    return segmentations


def get_points_from_json(json_file):
    '''This fuunction takes a json file containing annotations and returns a dict with the name of the unique classes as keys and the values are the bboxes coordinates.'''
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    shapes = data['shapes']

    #check if the shapes is empty (no annotations).
    if not len(shapes):
        return None
    points_dict = {}

    for shape in shapes:
        label = shape['label']
        points = shape['points']
        if label not in points_dict:
            points_dict[label] = []
        
        points_dict[label].append(sum(points,[]))

    return points_dict

def write_json(save_path,file_name,data_example):
    '''This function writes a Json file.
        Parameters:
              save_path: location where the file will be saved.
              file_name: a string representing the name of the file.
              data_example: a dictionary representing Labelme accepted format.'''
    completeName = os.path.join(save_path, file_name+".json") 
    with open(completeName, "w") as outfile:
        outfile.write(json.dumps(data_example,indent=2)) 
            
            
def json_file_data_format(img_path,img_height=780,img_width=1920):
    """This function returns a dictionary in Labelme standard format."""
    image_name= Path(img_path).stem

    return {"version": "5.1.1", "flags": {},  "shapes":[],
              "imagePath": image_name+'.png',
              "imageData": None,
              "imageHeight": img_height,
              "imageWidth": img_width }, image_name


def text_writer(file_name,names):
    '''This function takes a list and writes it into a text file.'''
    with open(f"{file_name}.txt", "w") as file:
        file.writelines(names)