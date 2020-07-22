import torch
from torch import nn
from pytorch3d.ops import GraphConv
from pytorch3d.structures import Meshes

from .base import BaseModule

class GCN(BaseModule):
    # batch size should be 1 for now
    def __init__(self, hp):
        super().__init__(hp)
        self.hp = hp
        # input: (V, 3) tensor, (E, 2) tensor
        self.gcn1 = GraphConv(3, hp.model.hidden_dim1)
        self.gcn2 = GraphConv(hp.model.hidden_dim1, hp.model.hidden_dim2)
        self.gcn3 = GraphConv(hp.model.hidden_dim2, hp.model.hidden_dim3)
        self.gcn4 = GraphConv(hp.model.hidden_dim3, 3)
        # output (V, 3) tensor

    def next_mesh(self, cmesh):
        # Return updated next mesh from current mesh 
        cverts = cmesh.verts_padded()[0]
        nverts = cverts + (self(cmesh) * self.hp.model.scale)
        nmesh = Meshes([nverts], cmesh.faces_list())
        return nmesh

    def forward(self, mesh):
        # Return estimated coordinate differences of each vertices
        # gets normal of each vertex, and go through gcn, sigmoid, gcn
        # how to add vertices?

        # coordinates of vertices (temporary)
        # verts = mesh.verts_padded()[0]
        norms = mesh.verts_normals_padded()[0]
        # verts = torch.rand_like(verts)
        edges = mesh.edges_packed() # Change when batch size != 1

        vfeat1 = torch.sigmoid(self.gcn1(norms, edges))
        vfeat2 = torch.sigmoid(self.gcn2(vfeat1, edges))
        vfeat3 = torch.sigmoid(self.gcn3(vfeat2, edges))
        vdiff = torch.sigmoid(self.gcn4(vfeat3, edges))
        
        return vdiff