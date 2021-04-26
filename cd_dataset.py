import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from albumentations.augmentations.transforms import Resize, Flip

class Cats_and_Dogs(Dataset):
    def __init__(self, dataframe):
        super().__init__()
        self.image_ids = dataframe['idx']
        self.df = dataframe
        self.transform = A.Compose([
                  Flip(0.8),
                  Resize(250, 250),
                  A.RandomBrightnessContrast(0.5),
                  ToTensor()],
                  bbox_params = A.BboxParams(format='pascal_voc'))
        
    def __getitem__(self, index: int):
        record = self.df[self.df['idx'] == index]
        image = cv2.imread(record.img_name.item(), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        sample = {}
        box = []
        label = []

        box.append([int(record.xmin),
                    int(record.ymin),
                    int(record.xmax),
                    int(record.ymax),
                    int(record.target)])
        label.append([int(record.target)])

        sample['image'] = image
        sample['bboxes'] = box
        
        sample = self.transform(image = sample['image'], bboxes = sample['bboxes'])
        target = list(x / 250 for x in sample['bboxes'][0][:4])
        target.append(sample['bboxes'][0][-1])
        target = torch.as_tensor(target, dtype = torch.float32)
        image = sample['image'] / 255.0

        return image, target

    def __len__(self):
        return self.image_ids.shape[0]