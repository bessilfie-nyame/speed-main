## Load Dataset

from video_dataset import  VideoFrameDataset, ImglistToTensor
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import pandas as pd
import json

import warnings
warnings.filterwarnings("ignore")

train_videos_root = os.path.join(os.getcwd(), 'data/train/dataset')
train_annotation_file = os.path.join(train_videos_root, 'annotations.txt')

validation_videos_root = os.path.join(os.getcwd(), 'data/validation/dataset')
validation_annotation_file = os.path.join(validation_videos_root, 'annotations.txt')

videos_root = os.path.join(os.getcwd(), 'data')
annotation_file = os.path.join(videos_root, 'annotations.txt')

preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(256),  # image batch, resize smaller edge to 256
        transforms.CenterCrop(224),  # image batch, center crop to square 224x224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

test_preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(256),  # image batch, resize smaller edge to 256
        transforms.CenterCrop(224),  # image batch, center crop to square 224x224
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Train
train_dataset = VideoFrameDataset( 
    root_path=train_videos_root,
    annotationfile_path=train_annotation_file,
    num_segments=5,
    frames_per_segment=1,
    imagefile_template='img_{:05d}.jpg',
    transform=preprocess,
    test_mode=False
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)  

# Val
val_dataset = VideoFrameDataset( 
    root_path=validation_videos_root,
    annotationfile_path=validation_annotation_file,
    num_segments=5,
    frames_per_segment=1,
    imagefile_template='img_{:05d}.jpg',
    transform=test_preprocess,
    test_mode=True
)

val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)   

# Test
# test_dataset = VideoFrameDataset( 
#     root_path=videos_root,
#     annotationfile_path=annotation_file,
#     num_segments=10,
#     frames_per_segment=1,
#     imagefile_template='img_{:05d}.jpg',
#     transform=test_preprocess,
#     test_mode=True
# )

# test_dataloader = torch.utils.data.DataLoader(
#     dataset=test_dataset,
#     batch_size=1,
#     shuffle=True,
#     num_workers=4,
#     pin_memory=True
# )   

loaders = {
    "train": train_dataloader,
    "val": val_dataloader
}