import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from pytorch_lightning.core import LightningModule
from abc import ABC, abstractmethod
from pytorch3d.structures import Meshes

from .loss import mesh_to_pcd_distance
from meshgen_utils import utils

class BaseModule(LightningModule, ABC):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
    
    @abstractmethod
    def next_mesh(self, mesh):
        pass

    def get_loss(self, mesh):
        loss = mesh_to_pcd_distance(mesh, self.source_pcd)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hp.train.lr)

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

        return {'loss': mean_loss, 'log': {'epoch_loss': mean_loss}}

    def train_dataloader(self):
        # Return pytorch dataloader, yielding zero tensor for each batch
        mesh, pcd = utils.initial_data(self.hp.data.file, method=self.hp.data.method)
        self.initial_mesh = self.current_mesh = mesh.to(device='cuda')
        self.source_pcd = pcd.to(device='cuda')
        self.log_mesh(self.initial_mesh, 'initial mesh')
        
        # dataset device?
        dataset = torch.zeros(self.hp.train.epoch_size) # placeholder dataset
        loader = DataLoader(dataset, batch_size=1, num_workers=16)
        return loader

    # def test_dataloader(self):
    #     if self.initial_mesh is None:
    #         mesh, pcd = utils.initial_data(self.input_file, method=self.hp.method)
    #         self.initial_mesh = self.current_mesh = mesh
    #         self.source_pcd = pcd
        
    #     loader = DataLoader(torch.zeros(1), batch_size=1)
    #     return loader