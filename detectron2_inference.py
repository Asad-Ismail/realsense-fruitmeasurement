import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import BoxMode



def find_orientation(cucumber):
    """horizontal or vertical cucumbers true if vertical else false"""
    xs=cucumber[:,0]
    ys=cucumber[:,1]
    xmin=np.min(xs)
    xmax=np.max(xs)
    ymin=np.min(ys)
    ymax=np.max(ys)
    return ymax-ymin> xmax-xmin


def left_right_bound(img):
    cuc=cv2.findNonZero(img).squeeze()
    line_points=[]
    r_data={}
    vertical=find_orientation(cuc)
    if vertical:
        xs=cuc[:,0]
        ys=cuc[:,1]
    else:
        xs=cuc[:,1]
        ys=cuc[:,0]
    # loop through all the unique ys
    r_data["orient"]=vertical
    u_ys=sorted(set(ys))
    for y in u_ys:
        tmp=[]
        m_p=np.argwhere(ys==y)
        x_p=xs[m_p]
        if vertical:
            tmp.append((np.min(x_p),y))
            tmp.append((np.max(x_p),y))
        else:
            tmp.append((y,np.min(x_p)))
            tmp.append((y,np.max(x_p)))
        line_points.append(tmp)
    r_data["data"]=line_points
    return r_data



def draw_bounds(img,data,color=(255,0,0)):
    for point in data:
        cv2.circle(img,point,1,color)



class detectroninference:
    def __init__(self,model_path,num_cls=1,name_classes=["cuc"]):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml"))
        self.cfg.DATASETS.TRAIN = ("cuc_train",)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 10
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml")  # Let training initialize from model zoo
        #self.cfg["MODEL"]["ANCHOR_GENERATOR"]["ASPECT_RATIOS"][0]=[0.03,0.3,1.0,6.0]
        self.cfg["MODEL"]["ANCHOR_GENERATOR"]["ASPECT_RATIOS"][0]=[0.03,1.0,6.0]
        self.cfg.SOLVER.IMS_PER_BATCH = 1
        self.cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        self.cfg.SOLVER.MAX_ITER = 50000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_cls 
        self.cfg.OUTPUT_DIR="output_resnet101" # only has one class (cucmber). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
        self.cfg.MODEL.WEIGHTS = model_path  # path to the model we just trained
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.20   # set a custom testing threshold
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES=1
        self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES=1
        #
        self.cfg.MODEL.RETINANET.NUM_CLASSES=1
        self.predictor = DefaultPredictor(self.cfg)
        self.cuc_metadata = MetadataCatalog.get("cuc").set(thing_classes=name_classes)
        #self.cuc_metadata = MetadataCatalog.get("cuc").set(thing_classes=["head","body","tail"])
        
        #self.cuc_metadata = MetadataCatalog.get("cuc").set(thing_classes=["cuc"])

    
    def apply_mask(self,mask,img):
        all_masks=np.zeros(mask.shape,dtype=np.uint8)
        all_patches=np.zeros((*mask.shape,3),dtype=np.uint8)
        """Apply the given mask to the image."""
        for i in range(all_masks.shape[0]):
                all_masks[i][:, :] = np.where(mask[i] == True,255,0)
                for j in range(3):
                    all_patches[i][:, :,j] = np.where(mask[i] == True,img[:,:,j],0)
        return all_masks,all_patches


    def pred(self,img):
        orig_img=img.copy()
        height,width=img.shape[:2]
        outputs = self.predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(img[:, :, ::-1],
                        metadata=self.cuc_metadata, 
                        scale=1, 
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        masks = np.asarray(outputs["instances"].to("cpu").pred_masks)
        masks,patches=self.apply_mask(masks,orig_img)
        classes=outputs["instances"].pred_classes.to("cpu").numpy()
        boxes=(outputs["instances"].pred_boxes.to("cpu").tensor.numpy())
        #print(c)
        return out.get_image()[:, :, ::-1],masks,patches,boxes,classes,outputs["instances"].scores.to("cpu").numpy()


if __name__=="__main__":
    #cuc=detectroninference("/home/asad/dev/detectron2/output_resnet101_config_up1/model_0017999.pth")
    cuc=detectroninference("/media/asad/ADAS_CV/cuc/output_axels2/model_final.pth")
    #cuc_cls=cuc_class()
    #img_path="/home/asad/annotated_700_cucumber/just_images"
    #img_path="/media/asad/ADAS_CV/cuc/test_images"
    #img_path="/media/asad/8800F79D00F79104/test_cuc_kyle"
    img_path="/media/asad/adas_cv_2/cucumber_kyle_selected"
    for filename in os.listdir(img_path):
        f_p=os.path.join(img_path,filename)
        img=cv2.imread(f_p)
        detected_cucumber,all_masks,all_patches,*_=cuc.pred(img)
        for ind,mask in enumerate(all_masks):
            #Mask
            resize_shape=img.shape[:-1]
            resize_shape=tuple((int(x*1) for x in resize_shape))
            # reverse x and y for size tuple
            resize_shape= resize_shape[::-1] 
            img=cv2.resize(img.astype(np.uint8),resize_shape)
            mask=cv2.resize(mask.astype(np.uint8),resize_shape)
            cuc_patch=cv2.resize(all_patches[ind].astype(np.uint8),resize_shape)
            
            #cv2.imwrite("image.png",img)
            #cv2.imwrite(f"/media/asad/ADAS_CV/scalecam/patches/{file_index}_{ind}_m"+".jpg",mask)
            #cv2.imwrite(f"/media/asad/ADAS_CV/scalecam/patches/{file_index}_{ind}_p"+".png",cuc_patch)
            #cuc_cat,cuc_prob=cuc_cls(cuc_patch)
            imgray=mask
            data=left_right_bound(imgray)
            lower_bound=[point[0] for point in data["data"]]
            upper_bound=[point[1] for point in data["data"]]
            backbone=[((point[0][0]+point[1][0])//2,point[0][1]) for point in data["data"]]

            x_mid=[x for x,y in backbone]
            y_mid=[y for x,y in backbone]

            #Straight line of backbone
            x_pred=np.poly1d(np.polyfit(y_mid, x_mid, 1))(y_mid)
            st_backbone_pred=[(int(x),y) for x,y in zip(x_pred,y_mid)]

            err_x=[np.abs(x1-x2) for x1,x2 in zip(x_pred,x_mid)]

            residual=sum(err_x)/len(err_x)
            residual_median=np.median(err_x)

            #coeff_u=np.polyfit(y_mid, x_mid, 1,full=True)
            #residual=(coeff_u[1][0]/len(x_mid)+coeff_u[1][0]/len(y_mid))/10
            #print("Mean Residuals", residual)
            #Put text stuff
            # fontScale
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (0, 255, 255)
            thickness = 1
            # Using cv2.putText() method
            cv2.putText(img, str(f"{residual:.1f},{residual_median:.1f}"), st_backbone_pred[0], font, fontScale, color, thickness, cv2.LINE_AA)

            bound_img=np.zeros_like(img)
            draw_bounds(img,lower_bound)
            draw_bounds(img,upper_bound,(0,0,255))
            draw_bounds(img,backbone,(0,255,0))
            draw_bounds(img,st_backbone_pred,(0,0,0))
        #cv2.imshow("Detected",detected_cucumber)
        #cv2.imshow("Bound",img)
        #cv2.imwrite("/home/asad/2.png",detected_cucumber)
        #cv2.waitKey(0)
        cv2.imwrite("/media/asad/adas_cv_2/cucumber_kyle_results/"+filename,img)
        #cv2.imwrite("/home/asad/test_backbone.png",img)