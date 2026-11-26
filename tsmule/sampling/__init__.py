"""Init module for sampling (segmentation, replacement, and perturbation)."""
from . import replace
from .perturb import AbstractPerturbation, Perturbation
from .segment import (
    AbstractSegmentation,
    MatrixProfileSegmentation,
    SAXSegmentation,
    SegmentationPicker,
    WindowSegmentation,
)

__all__ = [
    'AbstractSegmentation',
    'MatrixProfileSegmentation',
    'SAXSegmentation',
    'WindowSegmentation',
    'SegmentationPicker',
    'AbstractPerturbation',
    'Perturbation',
    'replace',
]
