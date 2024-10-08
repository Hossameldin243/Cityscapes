import os as os
from tqdm import tqdm
from CSVWriter import save_csv
from Load import Load
import torch
from preprocessing import *
from pytorch_toolbelt.losses import JaccardLoss, DiceLoss
from torchvision.transforms import Compose
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import PixelAccuracy
from EarlyStopping import EarlyStopping
from DataLoading import dataset
from torch.utils.data import DataLoader
import pandas as pd
import csv

modelPath = './deeplab_model_67.pth'

root_dir = '/home/hossam/Cityscapes/archive/kaggle/input/cityscapes/Cityspaces/'

images_dir = os.path.join(root_dir, 'images/val')
gtfine_dir = os.path.join(root_dir, 'gtFine/val')

cities = ['frankfurt','lindau','munster']

data = []

for city in cities:
    city_images_dir = os.path.join(images_dir, city)
    city_gtfine_dir = os.path.join(gtfine_dir, city)
    
    for img_file in os.listdir(city_images_dir):
        img_path = os.path.join(city_images_dir, img_file)
        base_name = img_file.replace('_leftImg8bit.png', '')
        label_train_ids_file = f'{base_name}_gtFine_labelTrainIds.png'
        label_train_ids_path = os.path.join(city_gtfine_dir, label_train_ids_file)
        
        if os.path.exists(label_train_ids_path):
            data.append([img_path, label_train_ids_path])
        else:
            print(f'{label_train_ids_path} not found')

save_csv('test_data.csv', data)

test_csv = 'test_data.csv'
test_df = pd.read_csv(test_csv)

Test_data_transform = Compose([
    Load(keys=['image_path', 'annotation_path'], images_dir=images_dir, gtfine_dir=gtfine_dir),
    Remap(keys=['annotation_path']),
    adjustGamma(keys=['image_path']),
    contrastStretch(keys=['image_path']),
    toGrayscale(keys=['annotation_path']),
    Normalise(keys=['image_path']),
    toTensor(keys=['image_path', 'annotation_path'])
])

test_dataset = dataset(
    input_dataframe=test_df,
    root_dir="",
    KeysOfInterest=['image_path', 'annotation_path'],
    data_transform=Test_data_transform
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=2,
    shuffle=False
)

# Load Model

model = smp.DeepLabV3Plus(
    encoder_name="mobilenet_v2", 
    encoder_weights="imagenet",  
    in_channels=3,
    classes=20 
)

device = torch.device("cpu")
criterion = DiceLoss(mode='multiclass', from_logits=False, ignore_index=0)

early_stopping = EarlyStopping(patience=5, min_delta=0.001)    
    
model = model.to(device)    
    
model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))

# Test Model

def test(loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    total_batches = len(loader)
    
    total_pixel_correct = 0
    total_pixel_count = 0
    total_iou_sum = 0.0
    num_classes = 20 

    jaccard_loss_fn = smp.losses.JaccardLoss(mode="multiclass", smooth=1e-7, from_logits = False, classes=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    
    with tqdm(total=total_batches, desc='Testing', unit='batch') as pbar:
        with torch.no_grad():
            for batch in loader:
                images = batch['image_path'].to(device)
                annotations = batch['annotation_path'].to(device)

                if images.dim() == 5:
                    images = images.squeeze(1)

                if images.size(-1) == 3: 
                    images = images.permute(0, 3, 1, 2) 
                
                outputs = model(images)
                outputs = torch.softmax(outputs, dim=1)
                
                loss = criterion(outputs, annotations)
                running_loss += loss.item()

                jaccard_loss = jaccard_loss_fn(outputs, annotations)
                total_iou_sum += 1 - jaccard_loss.item()
                
                pbar.update(1)
    
    avg_loss = running_loss / total_batches    
    avg_score = 1-avg_loss
    print(f"Average Dice Score: {avg_score:.4f}")

    mean_iou_score = total_iou_sum / total_batches
    print(f"Mean IoU: {mean_iou_score:.4f}")

test(test_loader, model, criterion, device)