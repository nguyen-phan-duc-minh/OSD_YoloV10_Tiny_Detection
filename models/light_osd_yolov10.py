import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SCDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 4  
        self.conv1 = ConvBlock(in_channels, mid_channels, 3, 2, 1)
        self.conv2 = ConvBlock(in_channels, out_channels - mid_channels, 1, 1, 0)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.pool(x2)
        return torch.cat([x1, x2], dim=1)

class LightSPCC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 3
        self.conv1x1 = ConvBlock(in_channels, mid_channels, 1, 1, 0)
        self.conv3x3 = ConvBlock(in_channels, mid_channels, 3, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels // 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels // 2, mid_channels * 2, 1),
            nn.Sigmoid()
        )
        self.output_conv = ConvBlock(mid_channels * 2, out_channels, 1, 1, 0)
    
    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        concat = torch.cat([x1, x2], dim=1)
        attention = self.global_pool(concat)
        attention = self.fc(attention)
        attended = concat * attention
        return self.output_conv(attended)

class LightDysample(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

class LightDFMA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        reduction = max(channels // 16, 4)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, reduction, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduction, 1, 1),
            nn.Sigmoid()
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduction, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        sa = self.spatial_attention(x)
        x = x * sa
        return x


class LightC2fBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1, shortcut=True):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = ConvBlock(in_channels, mid_channels, 1, 1, 0)
        self.conv2 = ConvBlock(in_channels, mid_channels, 1, 1, 0)
        self.bottlenecks = nn.ModuleList([
            ConvBlock(mid_channels, mid_channels, 3, 1, 1) for _ in range(min(num_blocks, 1))
        ])
        total_channels = mid_channels * (2 + len(self.bottlenecks))
        self.conv_out = ConvBlock(total_channels, out_channels, 1, 1, 0)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        outputs = [x1, x2]
        current = x2
        
        for bottleneck in self.bottlenecks:
            current = bottleneck(current)
            outputs.append(current)
        
        return self.conv_out(torch.cat(outputs, dim=1))


class LightOSDYOLOv10Backbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 32, 3, 2, 1)  # P1: 1/2, reduced from 64
        
        # P2: 1/4
        self.scdown1 = SCDown(32, 64)  # reduced from 128
        self.c2f1 = LightC2fBlock(64, 64, 1)
        
        # P3: 1/8  
        self.scdown2 = SCDown(64, 128)  # reduced from 256
        self.c2f2 = LightC2fBlock(128, 128, 1)
        
        # P4: 1/16
        self.scdown3 = SCDown(128, 256)  # reduced from 512
        self.c2f3 = LightC2fBlock(256, 256, 1)
        
        # P5: 1/32
        self.scdown4 = SCDown(256, 512)  # reduced from 1024
        self.c2f4 = LightC2fBlock(512, 512, 1)
        
        # SPCC for P5
        self.spcc = LightSPCC(512, 512)
        
    def forward(self, x):
        # Backbone
        p1 = self.conv1(x)
        
        p2 = self.scdown1(p1)
        p2 = self.c2f1(p2)
        
        p3 = self.scdown2(p2)
        p3 = self.c2f2(p3)
        
        p4 = self.scdown3(p3)
        p4 = self.c2f3(p4)
        
        p5 = self.scdown4(p4)
        p5 = self.c2f4(p5)
        p5 = self.spcc(p5)
        
        return p2, p3, p4, p5


class LightOSDYOLOv10Neck(nn.Module):
    """Lightweight OSD-YOLOv10 Neck with FPN + PAN"""
    def __init__(self):
        super().__init__()
        
        # Top-down FPN with correct channel dimensions based on LightDysample behavior
        # p5: 512 -> LightDysample -> 256, concat with p4: 256 -> 256+256=512
        self.upsample1 = LightDysample(512, 2)
        self.concat1 = ConvBlock(256 + 256, 256, 1, 1, 0)  # 512 -> 256
        self.c2f_fpn1 = LightC2fBlock(256, 256, 1)
        
        # p4_out: 256 -> LightDysample -> 128, concat with p3: 128 -> 128+128=256  
        self.upsample2 = LightDysample(256, 2)
        self.concat2 = ConvBlock(128 + 128, 128, 1, 1, 0)  # 256 -> 128
        self.c2f_fpn2 = LightC2fBlock(128, 128, 1)
        
        # p3_out: 128 -> LightDysample -> 64, concat with p2: 64 -> 64+64=128
        self.upsample3 = LightDysample(128, 2)
        self.concat3 = ConvBlock(64 + 64, 64, 1, 1, 0)     # 128 -> 64
        self.c2f_fpn3 = LightC2fBlock(64, 64, 1)
        
        # Bottom-up PAN
        self.downsample1 = ConvBlock(64, 64, 3, 2, 1)
        self.concat4 = ConvBlock(64 + 128, 128, 1, 1, 0)
        self.c2f_pan1 = LightC2fBlock(128, 128, 1)
        
        self.downsample2 = ConvBlock(128, 128, 3, 2, 1)
        self.concat5 = ConvBlock(128 + 256, 256, 1, 1, 0)
        self.c2f_pan2 = LightC2fBlock(256, 256, 1)
        
        self.downsample3 = ConvBlock(256, 256, 3, 2, 1)
        self.concat6 = ConvBlock(256 + 512, 256, 1, 1, 0)  # Output 256 instead of 512
        self.c2f_pan3 = LightC2fBlock(256, 256, 1)  # Input 256 instead of 512
        
        # Lightweight DFMA attention modules
        self.dfma_small = LightDFMA(64)   # For small objects (XX-Small)
        self.dfma_medium = LightDFMA(128)  # For medium objects (X-Small) 
        self.dfma_large = LightDFMA(256)   # For large objects (Small)
        self.dfma_xlarge = LightDFMA(256) # For extra large objects (Medium) - updated to 256
        
    def forward(self, features):
        p2, p3, p4, p5 = features
        
        # Top-down FPN
        up1 = self.upsample1(p5)
        cat1 = torch.cat([up1, p4], dim=1)
        cat1 = self.concat1(cat1)
        fpn_p4 = self.c2f_fpn1(cat1)
        
        up2 = self.upsample2(fpn_p4)
        cat2 = torch.cat([up2, p3], dim=1)
        cat2 = self.concat2(cat2)
        fpn_p3 = self.c2f_fpn2(cat2)
        
        up3 = self.upsample3(fpn_p3)
        cat3 = torch.cat([up3, p2], dim=1)
        cat3 = self.concat3(cat3)
        fpn_p2 = self.c2f_fpn3(cat3)
        
        # Bottom-up PAN
        down1 = self.downsample1(fpn_p2)
        cat4 = torch.cat([down1, fpn_p3], dim=1)
        cat4 = self.concat4(cat4)
        pan_p3 = self.c2f_pan1(cat4)
        
        down2 = self.downsample2(pan_p3)
        cat5 = torch.cat([down2, fpn_p4], dim=1)
        cat5 = self.concat5(cat5)
        pan_p4 = self.c2f_pan2(cat5)
        
        down3 = self.downsample3(pan_p4)
        cat6 = torch.cat([down3, p5], dim=1)
        cat6 = self.concat6(cat6)
        pan_p5 = self.c2f_pan3(cat6)
        
        # Apply DFMA attention
        out_p2 = self.dfma_small(fpn_p2)    # XX-Small objects
        out_p3 = self.dfma_medium(pan_p3)   # X-Small objects  
        out_p4 = self.dfma_large(pan_p4)    # Small objects
        out_p5 = self.dfma_xlarge(pan_p5)   # Medium objects
        
        return out_p2, out_p3, out_p4, out_p5


class LightDetectionHead(nn.Module):
    """Lightweight Detection head for each scale"""
    def __init__(self, in_channels, num_classes=1, num_anchors=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Reduced shared convolutions
        self.shared_conv = ConvBlock(in_channels, in_channels // 2, 3, 1, 1)
        
        hidden_dim = in_channels // 2
        
        # Detection heads
        self.cls_head = nn.Conv2d(hidden_dim, num_anchors * num_classes, 1, 1, 0)
        self.reg_head = nn.Conv2d(hidden_dim, num_anchors * 4, 1, 1, 0)
        self.obj_head = nn.Conv2d(hidden_dim, num_anchors, 1, 1, 0)
        
    def forward(self, x):
        shared = self.shared_conv(x)
        
        cls_output = self.cls_head(shared)
        reg_output = self.reg_head(shared)
        obj_output = self.obj_head(shared)
        
        return cls_output, reg_output, obj_output


class LightOSDYOLOv10(nn.Module):    
    def __init__(self, num_classes=1, input_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = LightOSDYOLOv10Backbone(input_channels)
        self.neck = LightOSDYOLOv10Neck()
        self.head_p2 = LightDetectionHead(64, num_classes)   # XX-Small objects
        self.head_p3 = LightDetectionHead(128, num_classes)  # X-Small objects
        self.head_p4 = LightDetectionHead(256, num_classes)  # Small objects
        self.head_p5 = LightDetectionHead(256, num_classes)  # Medium objects - updated to 256
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        p2, p3, p4, p5 = neck_features
        cls_p2, reg_p2, obj_p2 = self.head_p2(p2)
        cls_p3, reg_p3, obj_p3 = self.head_p3(p3)
        cls_p4, reg_p4, obj_p4 = self.head_p4(p4)
        cls_p5, reg_p5, obj_p5 = self.head_p5(p5)
        
        return {
            'predictions': [
                {'cls': cls_p2, 'reg': reg_p2, 'obj': obj_p2},  # XX-Small
                {'cls': cls_p3, 'reg': reg_p3, 'obj': obj_p3},  # X-Small
                {'cls': cls_p4, 'reg': reg_p4, 'obj': obj_p4},  # Small
                {'cls': cls_p5, 'reg': reg_p5, 'obj': obj_p5},  # Medium
            ],
            'features': neck_features
        }
    
    def count_parameters(self):
        """Count model parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_light_osd_yolov10(num_classes=1, pretrained=False):
    model = LightOSDYOLOv10(num_classes=num_classes)
    
    if pretrained:
        # Load pretrained weights if available
        pass
    
    return model