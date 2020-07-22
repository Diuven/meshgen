from pytorch3d.loss import point_mesh_face_distance, point_mesh_edge_distance, chamfer_distance


def mesh_to_pcd_distance(mesh, pcd):
    """
    https://pytorch3d.readthedocs.io/en/latest/modules/loss.html#pytorch3d.loss.point_mesh_face_distance
    """
    face_loss = point_mesh_face_distance(mesh, pcd)
    edge_loss = point_mesh_edge_distance(mesh, pcd)
    point_loss = chamfer_distance(mesh.verts_padded(), pcd)[0]
    return face_loss + edge_loss