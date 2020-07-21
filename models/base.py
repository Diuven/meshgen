from pytorch_lightning.core import LightningModule
from .loss import mesh_to_pcd_distance

class BaseModule(LightningModule):
    def __init__(self, hp):
        super().__init__()

    # train data
    # loss
    # optimizer
    # logging, visualization