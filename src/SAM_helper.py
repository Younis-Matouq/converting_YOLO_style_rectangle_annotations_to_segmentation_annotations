#Imports:
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
from segment_anything import sam_model_registry, SamPredictor


####################################################################
def model_loader_predictor(model_type = "vit_h",sam_checkpoint =r"C:\Users\matou\Downloads\sam_vit_h_4b8939.pth",device='cuda:0'):
    '''Parameters:
            sam_checkpoint: path to SAM weights.
        return SAM model
            '''
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    return predictor

def sam_predictor(image,predictor):
    '''This function takes an image and a sam`s model and return a segemented image.
            Parameters:
                image: RGP cv2 image.
                predictor: SAM predictor model (SamPredictor).
                '''
    torch.cuda.empty_cache()
    predictor.set_image(image)
    torch.cuda.empty_cache()
        
def generate_masks_from_bboxes(bboxs_coords,predictor,image):
    '''This function generates masks from bboxes coordinates using SAM's predictor. The function takes the following parameters.
        Parameters:
            bboxs_coords: np array containing the bboxes coordinates.
            predictor: SAM's predictor object.
            image: image: RGP cv2 image.
    '''
    input_boxes = torch.from_numpy(bboxs_coords).to(device=predictor.device)

    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
                                        )
    return masks