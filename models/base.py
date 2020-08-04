import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from pytorch_lightning.core import LightningModule
from pytorch3d.structures import Meshes
from abc import ABC, abstractmethod
from omegaconf import OmegaConf
import os

from .loss import Loss
from meshgen_utils import utils

class BaseModule(LightningModule, ABC):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        # self.get_loss = Loss(hp)
    
    @abstractmethod
    def next_mesh(self, mesh):
        pass

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.hp.train.lr)

    def log_mesh(self, mesh, tag, step=None):
        verts = mesh.verts_padded()
        faces = mesh.faces_padded()
        self.logger.experiment.add_mesh(tag, vertices=verts, faces=faces, global_step=step)

    def training_step(self, batch, index):
        cmesh = self.current_mesh
        nmesh = self.next_mesh(cmesh)
        
        self.current_mesh = Meshes(nmesh.verts_padded().detach(), nmesh.faces_padded().detach())

        loss = self.get_loss(nmesh)
        return {'loss': loss, 'log': {'loss': loss}}

    def training_epoch_end(self, outputs):
        self.log_mesh(self.current_mesh, 'output_mesh', self.current_epoch * self.hp.train.epoch_size)
        mean_loss = torch.stack([x['loss'] for x in outputs]).mean()

        if self.current_epoch % 10 == 0:
            save_path = os.path.join(self.logger.log_dir, 'objects')
            mesh, pcd = self.current_mesh, (None if self.current_epoch > 0 else self.source_pcd)
            deform = (self.deform if hasattr(self, 'deform') else None)
            utils.save_result(save_path, self.current_epoch, mesh, pcd, deform)

        return {'loss': mean_loss, 'log': {'epoch_loss': mean_loss}}

    def train_dataloader(self):
        # Return pytorch dataloader, yielding zero tensor for each batch
        dhp = self.hp.data
        kargs = (OmegaConf.to_container(dhp.kargs) if dhp.kargs is not None else dict())
        mesh, pcd = utils.initial_data(dhp.file, method=dhp.method, divide_mesh=dhp.divide, **kargs)
        self.initial_mesh = self.current_mesh = mesh.to(device='cuda')
        self.source_pcd = pcd.to(device='cuda')

        self.log_mesh(self.initial_mesh, 'initial mesh')
        self.get_loss = Loss(self.hp, self.source_pcd)
        self.get_loss.show(self.initial_mesh)
        
        # dataset device?
        dataset = torch.zeros(self.hp.train.epoch_size) # placeholder dataset
        loader = DataLoader(dataset, batch_size=1, num_workers=16)
        return loader
