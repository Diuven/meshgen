import open3d as o3d

from .fileio import *

def poisson_mesh(o3d_pcd, scale=1.1, depth=7):
    """
    Create an open3d type mesh from open3d type point cloud by poisson surface reconstruction
    """
    # Better estimate?
    if not o3d_pcd.has_normals():
        o3d_pcd.estimate_normals()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        o3d_pcd, depth=depth, scale=scale)[0]
    return mesh


def alpha_mesh(o3d_pcd, alpha=0.03):
    """
    Create an open3d type mesh from open3d type point cloud by alpha shape reconstruction
    """
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(o3d_pcd, alpha=alpha)
    return mesh


def convex_mesh(o3d_pcd):
    """
    Create an open3d type mesh from open3d type point cloud by convex hull
    """
    mesh, vertices = o3d_pcd.compute_convex_hull()
    mesh.orient_triangles()
    return mesh


# Ball pivoting?