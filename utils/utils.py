from sampler import sample_points
from mesh_ops import poisson_mesh
from fileio import *


def initial_mesh(filename):
    """
    Reads point cloud from the given filename, and returns initialized mesh
    """

    filename = str(filename)

    o3d_pcd = load_o3d_pcd(filename)
    o3d_mesh = poisson_mesh(o3d_pcd)
    pt3_mesh = o2p_mesh(o3d_mesh)

    return pt3_mesh


def save_mesh(filename, pt3_mesh):
    """
    Saves pytorch3d type mesh into the given filename
    """
    filename = str(filename)
    o3d_mesh = p2o_mesh(pt3_mesh)
    save_o3d_mesh(o3d_mesh)
