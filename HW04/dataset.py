import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Optional, List, Tuple
import json

class FERDataset(Dataset):
    
    def __init__(self, 
                 root_dir: str,
                 mode: str = 'train',
                 transform: Optional[transforms.Compose] = None):

        self.root_dir = os.path.join(root_dir, 'Images', mode)
        self.mode = mode
        self.transform = transform
        
        with open(os.path.join(root_dir, 'index_mapping'), 'r') as f:
            self.emotion_map = json.load(f)
            
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        if self.mode == 'train':
            for emotion, idx in self.emotion_map.items():
                emotion_dir = os.path.join(self.root_dir, emotion)
                if not os.path.exists(emotion_dir):
                    continue
                    
                for img_name in os.listdir(emotion_dir):
                    img_path = os.path.join(emotion_dir, img_name)
                    samples.append((img_path, idx))
                print(f"Loaded {len(samples)} images for {emotion}")
        else:
            submission_path = os.path.join(os.path.dirname(self.root_dir), 'sample_submission.csv')
            with open(submission_path, 'r') as f:
                next(f)  # Skip header
                for line in f:
                    filename, _ = line.strip().split(',')
                    img_path = os.path.join(self.root_dir, filename)
                    samples.append((img_path, 0))  
                    
        print(f"Total {self.mode} samples: {len(samples)}")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(mode: str = 'train') -> transforms.Compose:
    """
    Implementa exactamente las transformaciones mencionadas en el paper
    """
    if mode == 'train':
        return transforms.Compose([
            # 1. Rescaling ±20%
            transforms.RandomResizedCrop(
                size=48,
                scale=(0.8, 1.2),
                ratio=(1.0, 1.0)  
            ),
            
            # 2. Random shifting ±20%
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.2, 0.2),
                )
            ], p=0.5),
            
            # 3. Random rotation ±10°
            transforms.RandomApply([
                transforms.RandomRotation(10)
            ], p=0.5),
            
            # 4. Random horizontal flip
            transforms.RandomHorizontalFlip(),
            
            # 5. Ten crop 
            transforms.TenCrop(40),
            
            # 6. To tensor y normalization
            transforms.Lambda(lambda crops: torch.stack([
                transforms.ToTensor()(crop) for crop in crops
            ])),
            
            # 7. Random erasing (50% probability)
            transforms.Lambda(lambda crops: torch.stack([
                transforms.RandomErasing(p=0.5)(crop) for crop in crops
            ])),
            
            # 8. Final nomralization
            transforms.Lambda(lambda crops: crops / 255.0)
        ])
    else:
        return transforms.Compose([
            transforms.TenCrop(40),
            transforms.Lambda(lambda crops: torch.stack([
                transforms.ToTensor()(crop) for crop in crops
            ])),
            transforms.Lambda(lambda crops: crops / 255.0)
        ])

def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    
    train_dataset = FERDataset(
        root_dir=data_dir,
        mode='train',
        transform=get_transforms('train')
    )
    
    test_dataset = FERDataset(
        root_dir=data_dir,
        mode='test',
        transform=get_transforms('test')
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def visualize_augmentations(dataset, idx=0):
    """Visualiza las aumentaciones aplicadas a una imagen"""
    import matplotlib.pyplot as plt
    
    image, label = dataset[idx]
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:  # (n_crops, c, h, w)
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            for i, ax in enumerate(axes.flat):
                ax.imshow(image[i][0], cmap='gray')
                ax.axis('off')
                ax.set_title(f'Crop {i+1}')
            plt.suptitle(f'Label: {label}')
            plt.show()