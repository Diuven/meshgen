from pytorch3d.loss import point_mesh_face_distance


def mesh_to_pcd_distance(mesh, pcd):
    """
    https://pytorch3d.readthedocs.io/en/latest/modules/loss.html#pytorch3d.loss.point_mesh_face_distance
    """
    return point_mesh_face_distance(mesh, pcd)