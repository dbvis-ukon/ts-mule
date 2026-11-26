# ............................................................................
# ...............SSSSSS.......................................................
# .TTTTTTTTTTT..SSSSSSSS...MMMMM...MMMMM..UUUU...UUUU..LLLL.......EEEEEEEEEE..
# .TTTTTTTTTTT.SSSSSSSSSS..MMMMM...MMMMM..UUUU...UUUU..LLLL.......EEEEEEEEEE..
# .TTTTTTTTTTT.SSSSSSSSSS..MMMMMM..MMMMM..UUUU...UUUU..LLLL.......EEEEEEEEEE..
# .....TTTT...SSSS...SSSSS.MMMMMM.MMMMMM..UUUU...UUUU..LLLL.......EEEE........
# .....TTTT...SSSSSS.......MMMMMM.MMMMMM..UUUU...UUUU..LLLL.......EEEEEEEEEE..
# .....TTTT....SSSSSSSSS...MMMMMM.MMMMMM..UUUU...UUUU..LLLL.......EEEEEEEEEE..
# .....TTTT....SSSSSSSSSS..MMMMMM.MMMMMM..UUUU...UUUU..LLLL.......EEEEEEEEEE..
# .....TTTT......SSSSSSSSS.MMMMMMMMMMMMM..UUUU...UUUU..LLLL.......EEEEEEEEEE..
# .....TTTT...SSSS..SSSSSS.MMM.MMMMMMMMM..UUUU...UUUU..LLLL.......EEEE........
# .....TTTT...SSSS....SSSS.MMM.MMMMM.MMM..UUUU...UUUU..LLLL.......EEEE........
# .....TTTT...SSSSSSSSSSSS.MMM.MMMMM.MMM..UUUUUUUUUUU..LLLLLLLLLL.EEEEEEEEEE..
# .....TTTT....SSSSSSSSSS..MMM.MMMMM.MMM..UUUUUUUUUUU..LLLLLLLLLL.EEEEEEEEEE..
# .....TTTT.....SSSSSSSSS..MMM.MMMMM.MMM...UUUUUUUUU...LLLLLLLLLL.EEEEEEEEEE..
# ...............SSSSSS......................UUUUU............................
# ............................................................................
"""tsmule is an explainable AI library, written in Python, for time series explanations.

>>> import tsmule
>>> print(tsmule.__version__.__version__)
"""

from . import __version__  # noqa: F401
from .sampling.perturb import Perturbation
from .sampling.segment import (
    MatrixProfileSegmentation,
    SAXSegmentation,
    SegmentationPicker,
    WindowSegmentation,
)
from .xai.evaluation import PerturbationAnalysis, PerturbationBase
from .xai.lime import Kernels, LimeBase, LimeTS
from .xai.viz import (
    visualize_perturbation_masks,
    visualize_relevance,
    visualize_segmentation_mask,
)

__all__ = [
    'Perturbation',
    'MatrixProfileSegmentation',
    'SAXSegmentation',
    'WindowSegmentation',
    'SegmentationPicker',
    'Kernels',
    'LimeBase',
    'LimeTS',
    'PerturbationBase',
    'PerturbationAnalysis',
    'visualize_segmentation_mask',
    'visualize_perturbation_masks',
    'visualize_relevance',
]
