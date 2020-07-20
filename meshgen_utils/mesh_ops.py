import open3d as o3d

from .fileio import *

def poisson_mesh(o3d_pcd, scale=1.1, depth=7):
    """
    Create an open3d type mesh from open3d type point cloud.
    """
    # Estimate?
    o3d_pcd.estimate_normals()
    return o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(o3d_pcd, depth=depth, scale=scale)[0]

