from detectron2.data import DatasetMapper
import copy
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog
import numpy as np
import time
import torch


class ExtendedAugInput(T.AugInput):
    def __init__(self, image, *, 
                 boxes = None, 
                 sem_seg = None, 
                 dataset_dict = None):
        super().__init__(image, sem_seg=sem_seg, boxes=boxes)
        self.dataset_dict = dataset_dict
        self.is_train = True

class ExtendedDatasetMapper(DatasetMapper):
    def _convert_annotations(self, 
                             dataset_dict, 
                             boxes, 
                             image_shape):
        annos = []
        # get annotations info from dataset_dict
        for i, annotation in enumerate(dataset_dict.pop("annotations")):
            bbox = boxes[i]
            annotation["bbox"] = bbox
            annotation["bbox_mode"] = BoxMode.XYXY_ABS
            annos.append(annotation)

        instances = utils.annotations_to_instances(annos, 
                                                   image_shape,
                                                   mask_format=self.instance_mask_format)
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        # Step 1: Read images and annotations from dataset_dict
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
                ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.array([
            BoxMode.convert(
                obj["bbox"], 
                obj["bbox_mode"], BoxMode.XYXY_ABS) 
            for obj in dataset_dict['annotations']])
        
        # Step 2: Perform transformations/augmentations
        aug_input = ExtendedAugInput(image, 
                       sem_seg = sem_seg_gt, 
                       dataset_dict = dataset_dict,
                       boxes = boxes)

        transforms    = self.augmentations(aug_input)
        image         = aug_input.image
        sem_seg_gt    = aug_input.sem_seg
        dataset_dict  = aug_input.dataset_dict
        boxes         = aug_input.boxes

        image_shape = image.shape[:2]  
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))

        
        # Step 3: Convert annotations to instances
        if "annotations" in dataset_dict:
            self._convert_annotations(dataset_dict, 
                                      boxes, 
                                      image_shape)

        return dataset_dict
        
class DataDictSampler():
    def __init__(self, name_ds):
        ds = DatasetCatalog.get(name_ds)
        self.ds = ds

    def get_items(self, n=3):
        indices = np.random.randint(
            low   = 0, 
            high  = len(self.ds)-1,
            size  = n)
        return [copy.deepcopy(self.ds[_]) for _ in indices]        

class MixUpAug(T.Augmentation):
    def __init__(self, cfg, src_weight = 0.5, dst_weight = 0.5):
        self.cfg = cfg
        self.sampler = DataDictSampler(cfg.DATASETS.TRAIN[0])
        self.src_weight = src_weight
        self.dst_weight = dst_weight

    def get_transform(self, image, dataset_dict):
        cfg = self.cfg
        # Step 1: get one more random input
        ds_dict = self.sampler.get_items(n=1)[0]
        mu_image = utils.read_image(ds_dict["file_name"], 
                                    format=cfg.INPUT.FORMAT)
        utils.check_image_size(ds_dict, mu_image)
        # Step 2: append annotations and get mix-up boxes
        annotations = ds_dict["annotations"]
        dataset_dict["annotations"] += annotations
        mu_boxes = np.array([
            BoxMode.convert(
                obj["bbox"], obj["bbox_mode"], 
                BoxMode.XYXY_ABS) 
            for obj in annotations
            ])

        # Step 3: return an object of MixUpTransform
        return MixUpTransform(image       = image,
                              mu_image    = mu_image,
                              mu_boxes    = mu_boxes,
                              src_weight  = self.src_weight,
                              dst_weight  = self.dst_weight)


    def __repr__(self):
        
        return f"MixUp(src {self.src_weight}, dst {self.dst_weight})"

class MixUpTransform(T.Transform):
    def __init__(self, image, mu_image, mu_boxes, 
               src_weight = 0.5, 
               dst_weight = 0.5):
        # Step 1: resize the mu_image and mu_boxes
        image_size = image.shape[:2]
        rse = T.ResizeShortestEdge([min(image_size)], 
                              min(image_size), 
                              "choice")
        aug_i = T.AugInput(image=mu_image, boxes = mu_boxes)
        rse(aug_i)
        mu_image, mu_boxes = aug_i.image, aug_i.boxes

        # Step 2: pad mu_image
        img = np.zeros_like(image).astype('float32')
        img[:mu_image.shape[0], :mu_image.shape[1], :]=mu_image

        # Step 3: save values
        self.mu_image = img
        self.mu_boxes = mu_boxes
        self.src_weight = src_weight
        self.dst_weight = dst_weight


  
    def apply_image(self, image):
        bl_tfm = T.BlendTransform(src_image   = self.mu_image, 
                                  src_weight  = self.src_weight,
                                  dst_weight  = self.dst_weight)
        return bl_tfm.apply_image(image)
    
    def apply_coords(self, coords):
        return coords

    def apply_box(self, boxes):
        # combine boxes
        boxes = np.vstack([boxes, self.mu_boxes])
        return boxes 

class MosaicAug(T.Augmentation):
    def __init__(self, cfg):
        self.cfg = cfg
        self.sampler = DataDictSampler(cfg.DATASETS.TRAIN[0])

    def get_transform(self, image, dataset_dict):
        cfg = self.cfg
        # get three more random images
        mo_items = self.sampler.get_items()
        # images
        mo_images = []
        mo_boxes = []
        # create ShortestEdge resize to resize images
        for ds_dict in mo_items:
            mo_image = utils.read_image(ds_dict["file_name"], 
                                  format=cfg.INPUT.FORMAT)
            utils.check_image_size(ds_dict, mo_image)
            mo_images.append(mo_image)
            annotations = ds_dict["annotations"]
            # mo_boxes
            mo_boxes.append(np.array([
                BoxMode.convert(
                    obj["bbox"], 
                    obj["bbox_mode"], 
                    BoxMode.XYXY_ABS) 
                for obj in annotations]))
            # add annotations
            dataset_dict["annotations"] += annotations

        mt = MosaicTransform(mo_images, mo_boxes)
        return mt

    def __repr__(self):
        return "MosaicAug(4)"

class MosaicTransform(T.Transform):
    def __init__(self, mo_images, mo_boxes):
        self.mo_images = mo_images
        self.mo_boxes = mo_boxes
    
    def get_loc_info(self, image):
        images = [image] + self.mo_images
        heights = [i.shape[0] for i in images]
        widths = [i.shape[1] for i in images]
        ch = max(heights[0], heights[1])
        cw = max(widths[0], widths[2])
        h = (max(heights[0], heights[1]) + max(heights[2], heights[3]))
        w = (max(widths[0], widths[2]) + max(widths[1], widths[3]))
        # pad or start coordinates
        y0, x0 = ch-heights[0], cw - widths[0]
        y1, x1 = ch-heights[1], cw
        y2, x2 = ch, cw - widths[2]
        y3, x3 = ch, cw
        x_pads = [x0, x1, x2, x3]
        y_pads = [y0, y1, y2, y3]
        return (h, w, ch, cw, widths, heights, x_pads, y_pads)
  
    def apply_image(self, image):
        # get the loc info
        self.loc_info = self.get_loc_info(image)
        h, w, ch, cw, widths, heights, x_pads, y_pads = self.loc_info
        # output
        output = np.zeros((h, w, 3)).astype('float32')
        images = [image] + self.mo_images
        for i, img in enumerate(images):
            output[y_pads[i]: y_pads[i] + heights[i],
                   x_pads[i]: x_pads[i] + widths[i],
                   :] = img

        return output

    def apply_coords(self, coords):
        return coords

    def apply_box(self, boxes):
        # combine boxes
        boxes = [boxes] + self.mo_boxes
        # now update location values
        _, _, _, _, _, _, x_pads, y_pads = self.loc_info
        for i, bbox in enumerate(boxes):
            bbox += np.array([x_pads[i], y_pads[i], x_pads[i], y_pads[i]])
        # flatten it
        boxes = np.vstack(boxes)
        return boxes
        
        







