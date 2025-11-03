import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
from .light_osd_yolov10 import LightOSDYOLOv10

class UltraLightReferenceEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Ultra-lightweight custom backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 7, 2, 3, bias=False),  # -> 16 x 112 x 112
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),  # -> 16 x 56 x 56
            
            # Block 1
            nn.Conv2d(16, 32, 3, 2, 1, bias=False),  # -> 32 x 28 x 28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),  # -> 64 x 14 x 14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1))  # -> 64 x 1 x 1
        )
        
        # Simple projection
        self.projection = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim)
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = images.shape
        images = images.view(B * N, C, H, W)
        features = self.backbone(images)  # [B*N, 64, 1, 1]
        features = features.view(B * N, 64)  # [B*N, 64]
        embeddings = self.projection(features)  # [B*N, embedding_dim]
        embeddings = embeddings.view(B, N, self.embedding_dim)
        embeddings = embeddings.mean(dim=1)  # [B, embedding_dim]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

class UltraLightDetectionPostProcessor(nn.Module):
    def __init__(self, 
                 confidence_threshold: float = 0.25,
                 nms_threshold: float = 0.45,
                 max_detections: int = 100):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.register_buffer('stride', torch.tensor([4., 8., 16., 32.]))
        
    def forward(self, predictions, img_size):
        batch_results = []
        B = len(predictions[0]['cls'])
        
        for b in range(B):
            # For demo, return empty detections
            batch_results.append({
                'boxes': torch.empty(0, 4),
                'scores': torch.empty(0),
                'labels': torch.empty(0, dtype=torch.long)
            })
        
        return batch_results

class UltraLightDroneDetector(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        embedding_dim: int = 256,
        max_detections: int = 100,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.max_detections = max_detections
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.osd_yolo = LightOSDYOLOv10(num_classes=num_classes)
        self.reference_encoder = UltraLightReferenceEncoder(embedding_dim=embedding_dim)
        self.post_processor = UltraLightDetectionPostProcessor(
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            max_detections=max_detections
        )
        self.detection_feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(2 * 2 * 64, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        self.similarity_threshold = nn.Parameter(torch.tensor(0.5))
        
    def extract_detection_features(self, feature_maps, detections):
        batch_features = []
        p2_features = feature_maps[0]  # Use P2 features
        
        for i, detection in enumerate(detections):
            boxes = detection['boxes']
            
            if len(boxes) == 0:
                batch_features.append(torch.empty(0, self.embedding_dim, device=p2_features.device))
                continue
            
            batch_size, channels, height, width = p2_features.shape
            global_feat = F.adaptive_avg_pool2d(p2_features[i:i+1], (2, 2))
            num_detections = len(boxes)
            repeated_feat = global_feat.repeat(num_detections, 1, 1, 1)
            projected = self.detection_feature_extractor(repeated_feat)
            projected = F.normalize(projected, p=2, dim=1)
            batch_features.append(projected)
        
        return batch_features
    
    def compute_similarity(self, detection_features, reference_embeddings):
        """Compute similarity scores"""
        similarities = []
        
        for det_feats, ref_emb in zip(detection_features, reference_embeddings):
            if len(det_feats) == 0:
                similarities.append(torch.empty(0, device=det_feats.device))
                continue
            
            # Cosine similarity
            sim = torch.mm(det_feats, ref_emb.unsqueeze(1)).squeeze(1)
            similarities.append(sim)
        
        return similarities
    
    def forward(self, images, reference_images, targets=None):
        """Forward pass"""
        device = images.device
        batch_size = images.shape[0]
        
        # Encode reference images
        reference_embeddings = self.reference_encoder(reference_images)
        
        # OSD-YOLOv10 forward pass
        osd_output = self.osd_yolo(images)
        predictions = osd_output['predictions']
        feature_maps = osd_output['features']
        
        # Post-process detections
        detections = self.post_processor(predictions, images.shape[-2:])
        
        # Extract features and compute similarities
        detection_features = self.extract_detection_features(feature_maps, detections)
        similarities = self.compute_similarity(detection_features, reference_embeddings)
        
        # Filter by similarity
        filtered_detections = []
        for i, (detection, similarity) in enumerate(zip(detections, similarities)):
            if len(detection['boxes']) == 0:
                filtered_detections.append({
                    'boxes': torch.empty(0, 4, device=device),
                    'scores': torch.empty(0, device=device),
                    'labels': torch.empty(0, dtype=torch.long, device=device),
                    'similarities': torch.empty(0, device=device)
                })
                continue
            
            # Filter by similarity threshold
            valid_mask = similarity > self.similarity_threshold
            
            filtered_detections.append({
                'boxes': detection['boxes'][valid_mask],
                'scores': detection['scores'][valid_mask],
                'labels': detection['labels'][valid_mask],
                'similarities': similarity[valid_mask]
            })
        
        result = {
            'predictions': filtered_detections,
            'reference_embeddings': reference_embeddings,
            'raw_detections': detections,
            'feature_maps': feature_maps
        }
        
        # Training losses (simplified)
        if targets is not None:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            for pred in osd_output['predictions']:
                cls_loss = F.mse_loss(pred['cls'], torch.zeros_like(pred['cls']))
                total_loss = total_loss + cls_loss
            
            result['losses'] = {'total_loss': total_loss}
        
        return result
    
    def count_parameters(self):
        """Count model parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_ultra_light_model(config: Dict) -> UltraLightDroneDetector:
    model_config = config.get('model', {})
    
    return UltraLightDroneDetector(
        num_classes=model_config.get('num_classes', 1),
        embedding_dim=model_config.get('embedding_dim', 256),
        max_detections=model_config.get('max_detections', 100),
        confidence_threshold=model_config.get('confidence_threshold', 0.25),
        nms_threshold=model_config.get('nms_threshold', 0.45)
    )