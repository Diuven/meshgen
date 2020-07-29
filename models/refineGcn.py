import torch
from torch import nn
# from pytorch3d.ops import GraphConv
from pytorch3d.structures import Meshes
from torch_geometric import nn as gnn

from .base import BaseModule
from .pointnet import PointNet2

class MeshRefineGCN(BaseModule):
    # batch size should be 1 for now
    def __init__(self, hp):
        super().__init__(hp)
        self.hp = hp
        mhp = self.hp.model

        # How to use pointnet on source pcd and merge it to the mesh?
        if mhp.use_pointnet:
            self.feature = PointNet2(mhp.init_dim)
        else:
            self.feature = (lambda x: torch.rand((x.size(0), mhp.init_dim)).to(device=x.device))

        self.gcn1 = gnn.GCNConv(mhp.init_dim, mhp.hidden_dim1, cached=True)
        self.gcn2 = gnn.GCNConv(mhp.hidden_dim1, mhp.hidden_dim2, cached=True)
        self.gcn3 = gnn.GCNConv(mhp.hidden_dim2, 3, cached=True)
        # output (V, 3) tensor

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
        vfeat0 = self.feature(verts)
        # norms = mesh.verts_normals_padded()[0]
        edges = mesh.edges_packed().transpose(0, 1) # Change when batch size != 1

        vfeat1 = torch.relu(self.gcn1(vfeat0, edges))
        vfeat2 = torch.relu(self.gcn2(vfeat1, edges))
        vdiff = torch.tanh(self.gcn3(vfeat2, edges))
        
        return vdiff