
import os
import glob
import json
import numpy as np
from PIL import Image
from .image_augmentation import is_file_empty



def get_img_dimensions(img_path):
    img= Image.open(img_path)
    height, width= img.size[1], img.size[0]
    return height, width

def write_json(save_path,file_name,data_example):
        completeName = os.path.join(save_path, file_name+".json") 
        with open(completeName, "w") as outfile:
            outfile.write(json.dumps(data_example,indent=2))    
            
            
helper= {'dimensions_of_image': get_img_dimensions,
         'json_writer': write_json}

def convert_txt_to_labelme(class_dict, results_path, yolo_text_files,image_width= 1920,image_height=780
                          , img_dimensions= False, img_extension= '.png', images_path=[None]):
    
    '''This function converts a text file that has yolo like annotation     
    to json file that is acceptable by labelme app, function parameters description:
    
    class_dict: accepts a dictionary that has keys similar to the classes ID and values similar to the
    the name of the classes. example---------> class_dict={'0': 'pothole', '1': 'manhole', '2': 'patch'}.
    
    results_path: the results where the json files will be saved. 
    
    yolo_text_files: a list that has all of the text files pathes. 
    
    image_width: the width of the images.
    
    image_height: the height of the images. 
    
    image_dimensions: the defult for this variable is False, if True it will check the dimensions of you images and 
    overwrite them in image_width and image_height.
    
    img_extension: the extension of the images, eg: png, jpg,ect....
    
    images_path: the defult of this variable is a list of None, you can give it the pathes of your images incase
    the images are not in the same directory as the text files.
    '''
    
    
    
    
    
    
    
    if len(images_path) != len(yolo_text_files) and images_path != [None]:
        raise ValueError(f'''The number of text files does not match the
                         number of images, images:{len(images_path)}, text:{len(yolo_text_files)}''')
    
    if len(yolo_text_files) != len(images_path) and images_path== [None]:
         images_path= images_path * len(yolo_text_files)
            
    for file, img in zip(yolo_text_files, images_path):

        #check if the file does not contain annotations:
        if is_file_empty(file):
            continue

        file_name= file.rsplit('\\',1)[1].split('.')[0] # Get_file_name
        image_name= file_name + img_extension 
        
        if images_path[0] == None:
            image_path= file.rsplit('.')[0] + img_extension
            if os.path.exists(image_path):
                pass
            else:
                raise ValueError(f"This path does not exist for this image: {image_path}")
                
        else:
            image_path = img
    
    
        if img_dimensions:
            image_height,image_width= helper['dimensions_of_image'](image_path)
        
    
    
    
        
        data_example= {"version": "5.1.1", "flags": {},  "shapes":[],
                      "imagePath": image_name,
                      "imageData": None,
                      "imageHeight": image_height,
                      "imageWidth": image_width }
        # reading_txt_file_ that has yolo annotation
        yolo_data=np.loadtxt(file)
        if yolo_data.ndim == 1:
            yolo_data= yolo_data.reshape(1,-1)
        else: 
            pass 


        for obj in yolo_data:
            x_center, y_center, delta_x, delta_y = obj[1],obj[2],obj[3],obj[4]
            x_1= np.abs(image_width *(x_center- (delta_x/2))) # width 
            y_1= np.abs(image_height *(y_center- (delta_y/2))) # height
            x_2= np.abs(x_1 + delta_x*image_width)
            y_2= np.abs(y_1 + delta_y*image_height)
            data_example['shapes'].append({'label': class_dict[str(np.int32(obj[0]))],
            'points':[[ x_1, y_1],[ x_2, y_2]], 'group_id': None, 
            'shape_type':"rectangle",
            "flags": {}})

        
        helper['json_writer'](results_path,file_name,data_example)

