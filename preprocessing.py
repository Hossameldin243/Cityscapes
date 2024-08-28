import torch

# Equalize Image Histogram

def equalize_image_histogram(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(img)

class equalizeHistogram:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = equalize_image_histogram(sample[key])
            
        return sample
    
# Adjust Gamma

def adjust_gamma(img, gamma=1.0):
    img = img.astype(np.float32) / 255.0
    img = np.power(img, gamma) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

class adjustGamma:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = adjust_gamma(sample[key], gamma=1.2)
            
        return sample
    
# Contrast Stretch

def contrast_stretch(img, lower_percentile=2, upper_percentile=98):
    img_array = img.astype(np.float32)
    p2, p98 = np.percentile(img_array, (lower_percentile, upper_percentile))
    stretched_img = np.clip((img_array - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
    return stretched_img

class contrastStretch:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = contrast_stretch(sample[key], lower_percentile=2, upper_percentile=98)
            
        return sample
    
# Remap

class Remap:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = sample[key].astype(np.uint8)
            sample[key][sample[key] == 0] = 20
            sample[key][sample[key] == 255] = 0
            sample[key][sample[key] == 19] = 0
            sample[key][sample[key] == 20] = 19
            
        return sample
    
# Grayscale

class toGrayscale:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            if len(sample[key].shape) == 3:
                sample[key] = cv2.cvtColor(sample[key], cv2.COLOR_BGR2GRAY)
            
        return sample
    
# Resize

class Resize:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = cv2.resize(sample[key], (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

        return sample
    
# Normalise

class Normalise:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            
            sample[key] = sample[key].astype(np.float32) / 255.0
            sample[key] = np.expand_dims(sample[key], axis=0)
            
        return sample
    
# Tensor Conversion

class toTensor:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            if key == "image_path":
                sample[key] = torch.tensor(sample[key], dtype=torch.float32)
            
            if key == "annotation_path":   
                sample[key] = torch.tensor(sample[key], dtype=torch.long)             
        return sample
    
# Random Flip

import numpy as np
import cv2
import random

class RandomFlip:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        # Decide randomly whether to flip horizontally, vertically, or not at all
        flip_code = random.choice([None, 1])  # None means no flip, 0 for vertical, 1 for horizontal

        if flip_code is not None:
            for key in self.keys:
                if key in sample and sample[key] is not None:
                    sample[key] = cv2.flip(sample[key], flip_code)
                    
        return sample
    
# Random Rotate

import numpy as np
import cv2
import random

class RandomRotate:
    def __init__(self, keys, max_angle=15):
        self.keys = keys
        self.max_angle = max_angle

    def __call__(self, sample):
        # Decide randomly on an angle between -max_angle and max_angle
        if random.random() < 1:
            angle = random.uniform(-self.max_angle, self.max_angle)

            if angle != 0:
                for key in self.keys:
                    if key in sample and sample[key] is not None:
                        img = sample[key]
                        # Get image dimensions
                        (h, w) = img.shape[:2]
                        # Compute the center of the image
                        center = (w // 2, h // 2)
                        # Generate the rotation matrix
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        # Perform the rotation
                        rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                        # Update sample with rotated image
                        sample[key] = rotated_img

        return sample
    
