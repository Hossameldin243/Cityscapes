import cv2

class Load:
    def __init__(self, keys, images_dir, gtfine_dir):
        self.keys = keys
        self.images_dir = images_dir
        self.gtfine_dir = gtfine_dir
    
    def __call__(self, sample):
        for key in self.keys:

            sample[key] = cv2.imread(sample[key])
            
        return sample