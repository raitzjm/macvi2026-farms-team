"""
LaRS dataset class for mmsegmentation 1.x (BaseSegDataset API).
Imported via custom_imports in configs.

Classes: obstacle=0, water=1, sky=2
Palette: obstacle=(247,195,37), water=(41,167,224), sky=(90,75,164)
"""
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class LaRSDataset(BaseSegDataset):
    """LaRS maritime segmentation dataset (mmseg 1.x API).

    Classes: obstacle=0, water=1, sky=2
    Palette: obstacle=(247,195,37), water=(41,167,224), sky=(90,75,164)
    """

    METAINFO = dict(
        classes=('obstacle', 'water', 'sky'),
        palette=[[247, 195, 37], [41, 167, 224], [90, 75, 164]])

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)
