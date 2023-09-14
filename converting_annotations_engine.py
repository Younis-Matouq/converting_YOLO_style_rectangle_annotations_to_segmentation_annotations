#Imports:
import os
import glob
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import torch
import yaml

from src.Rect_to_seg_helper import *
from src.SAM_helper import *
#########################################################################################################################
def run(img_directory,annotation_directory,saving_path,sam_checkpoint_path,smooth):
    '''This code converts YOLO-style rectangle annotations to segmentation annotations.
       It takes the bounding box coordinates provided in the YOLO format and generates corresponding segmentation masks.
       The output of the script will be json files saved in the spacified path.
         Parameters:
             img_directory: The directory of the annotated images.
             annotation_directory:The directory of the annotations crosponding to the images.
             saving_path: The path where the results will be saved in.
             sam_checkpoint_path: The path to SAM checkpoint.
             '''
    torch.cuda.empty_cache()
    print('The process of converting rectangle annotations to segmentations has begun.')

    

    

    #load the model
    print('Loading The Model...........')
    predictor=model_loader_predictor(sam_checkpoint=sam_checkpoint_path)
    print('The Model is loaded')

    all_annotations=glob.glob(os.path.join(annotation_directory,'*.json'))
    print(f'Total number of JSON files: {len(all_annotations)}.')

    missing_images=[]
    files_with_exceptions=[]
    print('Code execution will start now.')
    for num, annotation_file in tqdm(enumerate(all_annotations), total=len(all_annotations), desc="Processing"):
        try:
    #for i,annotation_file in enumerate(all_annotations):
            
            #get image path:
            img_file= os.path.join(img_directory,Path(annotation_file).stem+'.png')
            #check if the image exists:
            if not os.path.exists(img_file):

                warnings.warn(f"This file does not  exist: {img_file}.", UserWarning)
                missing_images.append(img_file)
                continue
            #read image:
            image,img_dim=image_reader(img_file)

            #predict the image:
            sam_predictor(image,predictor)

            #read the json file and get the annotated objects:
            classes_dict=get_points_from_json(annotation_file)
            if not classes_dict:#check if the file is empty
                continue

            #initiate a json object to store the segmentation annotations:
            data_example, image_name= json_file_data_format(img_path=img_file,img_height=img_dim[0],img_width=img_dim[1])

            for key in classes_dict.keys():
                
                bboxes=np.vstack(classes_dict[key])
                #generate the masks of interest:
                masks=generate_masks_from_bboxes(bboxes,predictor,image)
                #get a list of segmentations:
                polygons=get_segmentations_from_mask(masks,smooth)
                
                #append to the annotations to the json file:
                for obj in polygons:

                    obj=np.vstack(obj).squeeze()
                    #elemenat files that have one dim
                    if np.ndim(obj)==1:
                        continue

                    #fill the annotation data
                    data_example['shapes'].append({'label':key,
                    'points':obj.tolist(), 'group_id': None, 
                    'shape_type':"polygon",
                    "flags": {}})


            #write the annotations into json file:
            write_json(saving_path,image_name,data_example)
            print(f'This is file number: {num} | Progress: {(num/len(all_annotations))*100}%')
        except Exception as e:
            print(e)
            warnings.warn(f"This file has an Exception check it: {annotation_file}.", UserWarning)
            files_with_exceptions.append(annotation_file)
            continue
    
    os.mkdir(os.path.join(saving_path,'missing_and_exception_files'))
    f_missing_name=os.path.join(saving_path,'missing_and_exception_files','missing_images')
    f_with_exceptions_name=os.path.join(saving_path,'missing_and_exception_files','files_with_exceptions')

    text_writer(f_missing_name,missing_images)
    text_writer(f_with_exceptions_name,files_with_exceptions)

    print('The process has been successfully completed!! Thanks for using CSSRI tools....')

def parse_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="CSSRI tool for converting YOLO-style rectangle annotations to segmentation annotations.")
    # parser.add_argument("img_directory", type=str, help="path to the source directory where you have the images.")
    # parser.add_argument("annotation_directory", type=str, help="path to the source directory where you have the annotations.")
    # parser.add_argument("saving_path", type=str, help="Path where the json files will be saved in.")
    # parser.add_argument("sam_checkpoint_path", type=str, help="Path to sam checkpoint.")
    # parser.add_argument("--smooth", action='store_true', help="Optional argument to apply smoothening to the polygons.")

    parser = argparse.ArgumentParser(description="CSSRI tool for converting YOLO-style rectangle annotations to segmentation annotations.")
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    config = parse_config(args.config)

    #run(args.img_directory, args.annotation_directory, args.saving_path, args.sam_checkpoint_path,args.smooth)
    run(config['img_directory'], config['annotation_directory'], config['saving_path'], config['sam_checkpoint_path'], config.get('smooth', False))