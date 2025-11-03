# Drone Detection Package
__version__ = "1.0.0"
__author__ = "AI Engineer"
__description__ = "Complete pipeline for Drone Object Detection Challenge optimized for NVIDIA Jetson Xavier NX"

from . import models
from . import utils

__all__ = ['models', 'utils']