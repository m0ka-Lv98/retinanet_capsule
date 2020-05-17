import os
import json
import copy
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class MedicalBboxDataset(Dataset):
    
    def __init__(self, annotation, data_path, transform=None):
        if isinstance(annotation, dict):
            self.coco = COCO()
            self.coco.dataset = annotation
            self.coco.createIndex()
        else:
            self.coco = COCO(annotation)
        self.data_path = data_path
        self.imgids = self.coco.getImgIds()
        self.set_transform(transform)
        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        """
        :return: Number of images
        :rtype: int
        """
        return len(self.imgids)
    
    def __getitem__(self, i):
        
        if isinstance(i, slice):
            imgids = self.imgids[i]
            return self.split_by_imgids(imgids)
        else:
            return self.transform({
                'img': self.load_image(i),
                **self.load_annotations(i)
            })
    
    def load_image(self, i):
        '''
        Args:
            i (int): Index of image
        
        Returns:
            numpy.ndarray: Selected image
        '''
        imgid = self.imgids[i]
        img_info = self.coco.loadImgs(imgid)[0]
        img_path = os.path.join(self.data_path, img_info['file_name'])
        img = Image.open(img_path)
        return np.array(img)
    
    def load_annotations(self, i):
        '''
        Args:
            i (int): Index of image
        
        Returns:
            dict: Annotation of the selected image
        '''
        imgid = self.imgids[i]
        annids = self.coco.getAnnIds(imgIds=imgid)
        anno_info = self.coco.loadAnns(annids)
        annotations     = np.zeros((0, 5))
        bboxes, labels = [], []
        
        for anno in anno_info:
            bboxes.append(anno['bbox'])
            label = self.coco.getCatIds().index(anno['category_id'])
            labels.append(label)

            if anno['bbox'][2] < 1 or anno['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = anno['bbox']
            annotation[0, 4]  = self.coco_label_to_label(anno['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
        
        bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
        bboxes[:, 2:] += bboxes[:, :2]  # xywh -> xyxy
        labels = np.array(labels, dtype=np.int)
        
        return {
            'annot' : annotations,
            'bboxes': bboxes,
            'labels': labels
        }
    
    def set_transform(self, transform):
        '''
        Args:
            transform (function): Function to transform
        '''
        self.transform = transform if transform else lambda x: x
    
    def split(self, split, split_path):
        if not isinstance(split, (tuple, list, set)):
            split = split,
        split_data = json.load(open(split_path))

        imgids = []
        for s in split:
            imgids += split_data['image_id'][s]
        
        return self.split_by_imgids(imgids)
        
    def split_by_imgids(self, imgids):
        coco_format = {
            'info': self.coco.dataset['info'],
            'categories': self.coco.dataset['categories'],
            'images': self.coco.loadImgs(imgids),
            'annotations': self.coco.loadAnns(self.coco.getAnnIds(imgIds=imgids))
        }
        return MedicalBboxDataset(coco_format, self.data_path, self.transform)

    def integrate_classes(self, new_cats, idmap):
        annotations = copy.deepcopy(self.coco.dataset['annotations'])
        for anno in annotations:
            anno['category_id'] = idmap[anno['category_id']]

        coco_format = {
            'info': self.coco.dataset['info'],
            'categories': new_cats,
            'images': self.coco.dataset['images'],
            'annotations': annotations
        }
        return MedicalBboxDataset(coco_format, self.data_path, self.transform)

    def with_annotation_imgids(self):
        imgids = []
        for catid in self.coco.getCatIds():
            imgids += self.coco.getImgIds(catIds=catid)
        return imgids
    
    def with_annotation(self):
        imgids = self.with_annotation_imgids()
        return self.split_by_imgids(imgids)
    
    def without_annotation(self):
        imgids = list(set(self.imgids) - set(self.with_annotation_imgids()))
        return self.split_by_imgids(imgids)
    
    def get_coco(self):
        return self.coco

    def get_category_names(self):
        catids = self.coco.getCatIds()
        categories = self.coco.loadCats(catids)
        return [cat['name'] for cat in categories]
    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]


    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.imgids[image_index])[0]
        return float(image['width']) / float(image['height'])