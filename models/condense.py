import torch
from torch import nn
from pytorch3d.structures import Meshes

from .base import BaseModule

class CondenseMesh(BaseModule):
    # batch size should be 1 for now
    def __init__(self, hp):
        super().__init__(hp)
        self.hp = hp
        mhp = self.hp.model

        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.LeakyReLU(0.1),
            nn.Linear(128, 512), nn.LeakyReLU(0.1), nn.Dropout(0.1),
            nn.Linear(512, 3)
        )
        self.first = True


    def get_loss(self, mesh, pcd=None):
        verts = mesh.verts_padded()[0]
        loss = verts.norm(dim=1).mean()
        if self.first:
            print("Loss: mean distance to the origin: %2.6f" % loss.item())
            self.first = False
        return loss

    def next_mesh(self, cmesh):
        # Return updated next mesh from current mesh 
        cverts = cmesh.verts_padded()[0]
        self.deform = self(cmesh) * self.hp.model.scale
        nverts = cverts + self.deform
        nmesh = Meshes([nverts], cmesh.faces_list())
        return nmesh

    def forward(self, mesh):
        # coordinates of vertices
        verts = mesh.verts_padded()[0]

        return self.net(verts)