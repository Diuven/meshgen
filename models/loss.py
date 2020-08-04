import torch
from pytorch3d import loss as pt3loss
from pytorch3d.ops import sample_points_from_meshes

#https://pytorch3d.readthedocs.io/en/latest/modules/loss.html#pytorch3d.loss.point_mesh_face_distance


class Loss:
    def __init__(self, hp, pcd=None):
        # handle when coefs are undefined
        lhp = hp.loss
        self.coef = torch.tensor([
            lhp.face if lhp.face is not None else 0,
            lhp.edge if lhp.edge is not None else 0,
            lhp.point if lhp.point is not None else 0,
            lhp.length if lhp.length is not None else 0,
            lhp.normal if lhp.normal is not None else 0,
            lhp.sample if lhp.sample is not None else 0,
        ], device='cuda')
        self.pcd = pcd

    def compute_loss(self, mesh, pcd=None):
        if pcd is None: pcd = self.pcd

        face_loss = pt3loss.point_mesh_face_distance(mesh, pcd)
        edge_loss = pt3loss.point_mesh_edge_distance(mesh, pcd)
        point_loss = pt3loss.chamfer_distance(mesh.verts_padded(), pcd)[0]

        length_loss = pt3loss.mesh_edge_loss(mesh)
        normal_loss = pt3loss.mesh_normal_consistency(mesh)

        mpcd = sample_points_from_meshes(mesh, 2 * pcd.points_padded()[0].shape[0])
        sample_loss, _ = pt3loss.chamfer_distance(mpcd, pcd)

        losses = torch.tensor((
            face_loss, edge_loss, point_loss,
            length_loss, normal_loss, sample_loss
        ), requires_grad=True).to(device='cuda')

        return losses

    def __call__(self, mesh, pcd=None):
        if pcd is None: pcd = self.pcd

        losses = self.compute_loss(mesh, pcd)
        total_loss = torch.dot(self.coef, losses)

        return total_loss

    def show(self, mesh, pcd=None):
        if pcd is None: pcd = self.pcd
        losses = self.compute_loss(mesh, pcd)
        print("ASDAS: {losses[0]: 2.6f}".format(losses=losses))
        print("Losses: \
            \n pcd to face : {losses[0]: 2.6f} * {coef[0]: 1.2f} \
            \n pcd to edge : {losses[1]: 2.6f} * {coef[1]: 1.2f} \
            \n mesh point  : {losses[2]: 2.6f} * {coef[2]: 1.2f} \
            \n edge length : {losses[3]: 2.6f} * {coef[3]: 1.2f} \
            \n mesh normal : {losses[4]: 2.6f} * {coef[4]: 1.2f} \
            \n sample point: {losses[5]: 2.6f} * {coef[5]: 1.2f} "
            .format(losses=losses, coef=self.coef)
        )