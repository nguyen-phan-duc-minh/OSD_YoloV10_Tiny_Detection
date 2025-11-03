import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import json

def bbox_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def compute_st_iou(
    pred_sequence: List[Dict], 
    gt_sequence: List[Dict],
    iou_threshold: float = 0.5
) -> float:
    if not pred_sequence and not gt_sequence:
        return 1.0  # Perfect match if both empty
    
    if not pred_sequence or not gt_sequence:
        return 0.0  # No match if one is empty
    
    # Convert to dictionaries for fast lookup
    pred_frames = {item['frame']: item['bbox'] for item in pred_sequence}
    gt_frames = {item['frame']: item['bbox'] for item in gt_sequence}
    
    # Get all frames where object appears in either prediction or ground truth
    all_frames = set(pred_frames.keys()) | set(gt_frames.keys())
    union_frames = len(all_frames)
    
    # Get intersection frames (where object appears in both)
    intersection_frames = set(pred_frames.keys()) & set(gt_frames.keys())
    
    if not intersection_frames:
        return 0.0
    
    # Calculate IoU for each intersection frame
    total_iou = 0.0
    for frame in intersection_frames:
        pred_bbox = torch.tensor(pred_frames[frame], dtype=torch.float32)
        gt_bbox = torch.tensor(gt_frames[frame], dtype=torch.float32)
        frame_iou = bbox_iou(pred_bbox, gt_bbox)
        total_iou += frame_iou
    
    # ST-IoU is sum of IoUs divided by union of frames
    st_iou = total_iou / union_frames
    return st_iou

def evaluate_video(
    predictions: List[Dict],
    ground_truth: List[Dict],
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.5
) -> Dict:
    # Filter predictions by confidence
    filtered_preds = [
        pred for pred in predictions 
        if pred.get('confidence', 1.0) >= confidence_threshold
    ]
    
    # Group predictions and ground truth by detection sequence
    # For simplicity, assume each video has one object sequence
    pred_sequence = []
    for pred in filtered_preds:
        pred_sequence.append({
            'frame': pred['frame'],
            'bbox': pred['bbox'],
            'confidence': pred.get('confidence', 1.0)
        })
    
    gt_sequence = []
    for gt in ground_truth:
        gt_sequence.append({
            'frame': gt['frame'],
            'bbox': [gt['x1'], gt['y1'], gt['x2'], gt['y2']]
        })
    
    # Sort by frame number
    pred_sequence.sort(key=lambda x: x['frame'])
    gt_sequence.sort(key=lambda x: x['frame'])
    
    # Compute ST-IoU
    st_iou = compute_st_iou(pred_sequence, gt_sequence, iou_threshold)
    
    return {
        'st_iou': st_iou,
        'pred_frames': len(pred_sequence),
        'gt_frames': len(gt_sequence),
        'pred_sequence': pred_sequence,
        'gt_sequence': gt_sequence
    }

def evaluate_submission(
    submission_path: str,
    ground_truth_path: str,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.5
) -> Dict:
    # Load submission
    with open(submission_path, 'r') as f:
        submission = json.load(f)
    
    # Load ground truth
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Convert ground truth to dictionary for fast lookup
    gt_dict = {}
    for gt_video in ground_truth:
        video_id = gt_video['video_id']
        gt_dict[video_id] = []
        for annotation in gt_video['annotations']:
            for bbox in annotation['bboxes']:
                gt_dict[video_id].append(bbox)
    
    # Evaluate each video
    video_results = {}
    st_ious = []
    
    for pred_video in submission:
        video_id = pred_video['video_id']
        
        # Get predictions for this video
        predictions = []
        for detection in pred_video.get('detections', []):
            for bbox in detection.get('bboxes', []):
                predictions.append({
                    'frame': bbox['frame'],
                    'bbox': [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']],
                    'confidence': bbox.get('confidence', 1.0)
                })
        
        # Get ground truth for this video
        gt_annotations = gt_dict.get(video_id, [])
        
        # Evaluate this video
        video_result = evaluate_video(
            predictions, gt_annotations, confidence_threshold, iou_threshold
        )
        
        video_results[video_id] = video_result
        st_ious.append(video_result['st_iou'])
    
    # Compute overall metrics
    mean_st_iou = np.mean(st_ious) if st_ious else 0.0
    
    return {
        'mean_st_iou': mean_st_iou,
        'video_results': video_results,
        'num_videos': len(st_ious),
        'st_iou_distribution': {
            'min': np.min(st_ious) if st_ious else 0.0,
            'max': np.max(st_ious) if st_ious else 0.0,
            'std': np.std(st_ious) if st_ious else 0.0,
            'median': np.median(st_ious) if st_ious else 0.0
        }
    }

def compute_precision_recall(
    predictions: List[Dict],
    ground_truth: List[Dict],
    confidence_thresholds: List[float] = None,
    iou_threshold: float = 0.5
) -> Dict:
    if confidence_thresholds is None:
        confidence_thresholds = np.arange(0.0, 1.01, 0.05)
    
    precisions = []
    recalls = []
    
    for conf_thresh in confidence_thresholds:
        # Filter predictions by confidence
        filtered_preds = [
            pred for pred in predictions 
            if pred.get('confidence', 1.0) >= conf_thresh
        ]
        
        # Count true positives, false positives, false negatives
        tp = 0
        fp = 0
        fn = len(ground_truth)
        
        # Track which ground truth boxes have been matched
        gt_matched = [False] * len(ground_truth)
        
        for pred in filtered_preds:
            pred_bbox = torch.tensor(pred['bbox'], dtype=torch.float32)
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find best matching ground truth box
            for gt_idx, gt in enumerate(ground_truth):
                if gt_matched[gt_idx]:
                    continue
                
                gt_bbox = torch.tensor([gt['x1'], gt['y1'], gt['x2'], gt['y2']], dtype=torch.float32)
                iou = bbox_iou(pred_bbox, gt_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if prediction matches ground truth
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                gt_matched[best_gt_idx] = True
                fn -= 1
            else:
                fp += 1
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Compute Average Precision (AP)
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i-1]) * precisions[i]
    
    return {
        'precisions': precisions,
        'recalls': recalls,
        'ap': ap,
        'confidence_thresholds': confidence_thresholds
    }