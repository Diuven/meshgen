import open3d as o3d

from .fileio import *

def poisson_mesh(o3d_pcd, depth=7):
    """
    Create an open3d type mesh from open3d type point cloud.
    """
    return o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(o3d_pcd, depth=depth)

