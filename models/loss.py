from pytorch3d import loss as pt3loss
from pytorch3d.ops import sample_points_from_meshes

#https://pytorch3d.readthedocs.io/en/latest/modules/loss.html#pytorch3d.loss.point_mesh_face_distance


class Loss:
    def __init__(self, hp, pcd=None):
        # handle when coefs are undefined
        self.coef = hp.loss
        self.pcd = pcd

    def __call__(self, mesh, pcd=None):
        if pcd is None: pcd = self.pcd

        face_loss = pt3loss.point_mesh_face_distance(mesh, pcd)
        edge_loss = pt3loss.point_mesh_edge_distance(mesh, pcd)
        point_loss = pt3loss.chamfer_distance(mesh.verts_padded(), pcd)[0]

        length_loss = pt3loss.mesh_edge_loss(mesh)

        mpcd = sample_points_from_meshes(mesh, 2 * pcd.points_padded()[0].shape[0])
        sample_loss, _ = pt3loss.chamfer_distance(mpcd, pcd)

        loss = 0
        loss += self.coef.face * face_loss
        loss += self.coef.edge * edge_loss
        loss += self.coef.point * point_loss
        loss += self.coef.length * length_loss
        loss += self.coef.sample * sample_loss

        return loss
