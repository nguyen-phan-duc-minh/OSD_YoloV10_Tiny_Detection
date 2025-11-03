# Training Configuration - Python version
config = {
    "data": {
        "train_dir": "../data/train",
        "test_dir": "../data/public_test",
        "batch_size": 8,  # Production batch size
        "num_workers": 4,
        "augmentation": True,
        "frame_sampling_rate": 5,  # Sample every 5 frames
        "max_frames_per_video": 100  # Use many frames per video
    },
    
    "model": {
        "model_type": "ultra_light_osd_yolov10",  # Ultra-lightweight version for Jetson NX
        "num_classes": 1,
        "embedding_dim": 256,  # Reduced for efficiency
        "max_detections": 100,
        "confidence_threshold": 0.25,
        "nms_threshold": 0.45,
        "reference_backbone": "ultra_light"  # Custom ultra-light backbone (no external dependencies)
    },
    
    "training": {
        "epochs": 150,  # Increased for better convergence
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "save_interval": 10,
        "early_stopping_patience": 20,
        "batch_size": 8  # Production batch size
    },
    
    "optimization": {
        "mixed_precision": True,
        "gradient_clipping": 1.0,
        "max_parameters": 50000000  # 50M parameters limit
    },
    
    "evaluation": {
        "st_iou_threshold": 0.5,
        "tracking_overlap_threshold": 0.3
    },
    
    "tensorrt": {
        "precision": "fp16",
        "max_workspace_size": 1073741824,  # 1GB
        "optimize_for_jetson": True
    },
    
    "logging": {
        "wandb_project": "drone_detection",
        "log_interval": 100,
        "save_best_only": True
    }
}

# Export config for easy import
__all__ = ['config']