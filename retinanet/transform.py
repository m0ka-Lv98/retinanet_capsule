import copy
import numpy as np
import cv2

from imgaug import BoundingBoxesOnImage
from imgaug import augmenters as iaa
from imgaug import parameters as iap


class ToFixedSize:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, data):
        data = copy.copy(data)
        
        raw_h, raw_w = data['img'].shape[:2]
        mag = min(self.size[0]/raw_h, self.size[1]/raw_w)
        h = round(raw_h * mag)
        w = round(raw_w * mag)
        
        image = data['img']
        data['img'] = np.zeros([*self.size, data['img'].shape[2]])
        data['img'][:h, :w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
        
        data['bboxes'][:, 0::2] *= w / raw_w
        data['bboxes'][:, 1::2] *= h / raw_h
        
        
        return data


class Augmentation:
    def __init__(self, settings):
        seq = []
        
        def active(t):
            return t in settings and settings[t] != False
        
        # 回転, 左右反転, 上下反転で8パターン
        if active('flip'):
            seq += [
                iaa.Affine(rotate=iap.Binomial(0.5)*90),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5)
            ]
        
        if active('gamma_per_channel'):
            low, high = settings['gamma_per_channel']
            seq.append(iaa.GammaContrast([low, high], per_channel=True))
            
        if active('gamma'):
            low, high = settings['gamma']
            seq.append(iaa.GammaContrast([low, high]))
        
        if active('gaussnoise'):
            intensity = settings['gaussnoise']
            seq.append(
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, intensity), per_channel=1)
            )
        
        self.seq = iaa.Sequential(seq)
    
    def __call__(self, data):
        data = copy.copy(data)

        seq = self.seq.to_deterministic()

        image = data['img']
        bboxes = BoundingBoxesOnImage.from_xyxy_array(data['bboxes'], shape=image.shape)
        
        image, bboxes = seq(image=image, bounding_boxes=bboxes)

        data['img'] = image
        data['bboxes'] = BoundingBoxesOnImage.to_xyxy_array(bboxes)

        return data


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, data):
        data = copy.copy(data)
        data['img'] = (data['img'] - self.mean) / self.std
        return data

class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, data):
        data = copy.copy(data)
        if data['img'].
        data['img'] = data['img'].transpose(1,2,0)
        data['img'] = (data['img'] * self.std) + self.mean
        data['img'] = data['img'].astype(np.uint8)
        return data


class HWCToCHW:
    def __call__(self, data):
        data = copy.copy(data)
        data['img'] = data['img'].transpose(2, 0, 1)
        return data
