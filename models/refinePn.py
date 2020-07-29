import torch
from torch import nn
# from pytorch3d.ops import GraphConv
from pytorch3d.structures import Meshes
from torch_geometric import nn as gnn

from .base import BaseModule
from .pointnet import PointNet2

class MeshRefinePN(BaseModule):
    # batch size should be 1 for now
    def __init__(self, hp):
        super().__init__(hp)
        self.hp = hp
        mhp = self.hp.model

        self.pn = PointNet2(last_dim=3)

    def next_mesh(self, cmesh):
        # Return updated next mesh from current mesh 
        cverts = cmesh.verts_padded()[0]
        self.deform = self(cmesh) * self.hp.model.scale
        nverts = cverts + self.deform
        nmesh = Meshes([nverts], cmesh.faces_list())
        return nmesh

    def forward(self, mesh):
        # Return estimated coordinate differences of each vertices
        # how to add vertices?

        # coordinates of vertices
        verts = mesh.verts_padded()[0]

        return self.pn(verts)