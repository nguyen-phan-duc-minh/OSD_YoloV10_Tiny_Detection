import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Dict, Tuple, Optional

class DroneDataset(Dataset):    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        frame_sampling_rate: int = 5,
        max_frames_per_video: int = 1000,
        augmentation: bool = True,
        image_size: Tuple[int, int] = (640, 640)
    ):
        self.data_dir = data_dir
        self.split = split
        self.frame_sampling_rate = frame_sampling_rate
        self.max_frames_per_video = max_frames_per_video
        self.image_size = image_size
        
        # Setup data paths
        if split == "train":
            self.samples_dir = os.path.join(data_dir, "train", "samples")
            self.annotations_path = os.path.join(data_dir, "train", "annotations", "annotations.json")
        else:
            self.samples_dir = os.path.join(data_dir, "public_test", "samples")
            self.annotations_path = None
        
        # Load annotations if available
        self.annotations = {}
        if self.annotations_path and os.path.exists(self.annotations_path):
            with open(self.annotations_path, 'r') as f:
                annotations_list = json.load(f)
            for ann in annotations_list:
                self.annotations[ann['video_id']] = ann['annotations']
        
        # Get video IDs
        self.video_ids = [d for d in os.listdir(self.samples_dir) 
                         if os.path.isdir(os.path.join(self.samples_dir, d))]
        
        # Setup augmentation
        self.augmentation = self._setup_augmentation() if augmentation else None
        self.basic_transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Extract and cache frame data
        self.frame_data = self._extract_frame_data()
        
    def _setup_augmentation(self) -> A.Compose:
        """Setup albumentations augmentation pipeline."""
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    def _extract_frame_data(self) -> List[Dict]:
        """Extract frame data from videos with annotations."""
        frame_data = []
        
        for video_id in self.video_ids:
            video_path = os.path.join(self.samples_dir, video_id, "drone_video.mp4")
            if not os.path.exists(video_path):
                continue
                
            # Load reference object images
            object_images = self._load_reference_images(video_id)
            
            # Extract frames with annotations
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Get annotated frames for this video
            annotated_frames = set()
            if video_id in self.annotations:
                for ann in self.annotations[video_id]:
                    for bbox in ann['bboxes']:
                        annotated_frames.add(bbox['frame'])
            
            frame_count = 0
            extracted_frames = 0
            
            while extracted_frames < self.max_frames_per_video and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames at specified rate, or if frame has annotations
                if (frame_count % self.frame_sampling_rate == 0 or 
                    frame_count in annotated_frames):
                    
                    # Get bounding boxes for this frame
                    bboxes = []
                    if video_id in self.annotations:
                        for ann in self.annotations[video_id]:
                            for bbox in ann['bboxes']:
                                if bbox['frame'] == frame_count:
                                    bboxes.append([
                                        bbox['x1'], bbox['y1'], 
                                        bbox['x2'], bbox['y2']
                                    ])
                    
                    frame_data.append({
                        'video_id': video_id,
                        'frame_id': frame_count,
                        'frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        'object_images': object_images,
                        'bboxes': bboxes,
                        'has_object': len(bboxes) > 0
                    })
                    extracted_frames += 1
                
                frame_count += 1
            
            cap.release()
        
        return frame_data
    
    def _load_reference_images(self, video_id: str) -> List[np.ndarray]:
        """Load reference object images for a video."""
        object_images = []
        object_dir = os.path.join(self.samples_dir, video_id, "object_images")
        
        for img_name in sorted(os.listdir(object_dir)):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(object_dir, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                object_images.append(img)
        
        return object_images
    
    def __len__(self) -> int:
        return len(self.frame_data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a training sample."""
        data = self.frame_data[idx]
        
        # Convert frame to PIL for transformations
        frame = data['frame']
        bboxes = data['bboxes'].copy()
        
        # Apply augmentation if training
        if self.augmentation and len(bboxes) > 0:
            labels = [1] * len(bboxes)  # All objects are class 1
            augmented = self.augmentation(image=frame, bboxes=bboxes, labels=labels)
            frame = augmented['image']
            bboxes = augmented['bboxes']
        else:
            transformed = self.basic_transform(image=frame)
            frame = transformed['image']
        
        # Process reference object images
        object_features = []
        for obj_img in data['object_images']:
            obj_transformed = self.basic_transform(image=obj_img)
            object_features.append(obj_transformed['image'])
        
        # Convert bboxes to tensor format
        if len(bboxes) > 0:
            bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
            labels_tensor = torch.ones(len(bboxes), dtype=torch.long)  # All objects are class 1
        else:
            bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.long)
        
        return {
            'image': frame,
            'object_images': torch.stack(object_features) if object_features else torch.empty(0, 3, *self.image_size),
            'bboxes': bboxes_tensor,
            'labels': labels_tensor,
            'video_id': data['video_id'],
            'frame_id': data['frame_id'],
            'has_object': data['has_object']
        }

def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for variable-length bounding boxes."""
    images = torch.stack([item['image'] for item in batch])
    object_images = [item['object_images'] for item in batch]
    
    # Pad object images to same length
    max_ref_imgs = max(len(obj_imgs) for obj_imgs in object_images)
    if max_ref_imgs > 0:
        padded_object_images = []
        for obj_imgs in object_images:
            if len(obj_imgs) < max_ref_imgs:
                padding = torch.zeros(max_ref_imgs - len(obj_imgs), *obj_imgs.shape[1:])
                obj_imgs = torch.cat([obj_imgs, padding], dim=0)
            padded_object_images.append(obj_imgs)
        object_images = torch.stack(padded_object_images)
    else:
        object_images = torch.empty(len(batch), 0, 3, batch[0]['image'].shape[1], batch[0]['image'].shape[2])
    
    # Keep bboxes as list since they have variable lengths
    bboxes = [item['bboxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    video_ids = [item['video_id'] for item in batch]
    frame_ids = [item['frame_id'] for item in batch]
    has_objects = torch.tensor([item['has_object'] for item in batch], dtype=torch.bool)
    
    return {
        'images': images,
        'object_images': object_images,
        'bboxes': bboxes,
        'labels': labels,
        'video_ids': video_ids,
        'frame_ids': frame_ids,
        'has_objects': has_objects
    }

def create_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    **kwargs
) -> torch.utils.data.DataLoader:
    dataset = DroneDataset(data_dir, split=split, **kwargs)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train")
    )