from . import mesh_ops
from . import fileio as io
from .visualize import o3d_visualize


def initial_data(filename, depth=7):
    """
    Reads point cloud from the given filename, and returns initialized mesh and point cloud
    Returns (mesh, pcd):
        mesh: pytorch3d mesh, initialized with poisson method from the original point cloud
        pcd: pytorch3d point cloud, read from the given filename
    """

    filename = str(filename)

    o3d_pcd = io.load_o3d_pcd(filename)
    o3d_mesh = mesh_ops.poisson_mesh(o3d_pcd, depth=depth)
    pt3_mesh = io.o2p_mesh(o3d_mesh)
    pt3_pcd = io.o2p_pcd(o3d_pcd)

    return pt3_mesh, pt3_pcd


def save_mesh(filename, pt3_mesh):
    """
    Saves pytorch3d type mesh into the given filename
    """
    filename = str(filename)
    o3d_mesh = io.p2o_mesh(pt3_mesh)
    io.save_o3d_mesh(o3d_mesh)


def show_overlay(pt3_mesh, pt3_pcd):
    """
    Given pytorc3d type mesh and point cloud, visualize overlay of both.
    Point cloud as green, Mesh as Blue
    """
    o3d_mesh = io.p2o_mesh(pt3_mesh)
    o3d_mesh.paint_uniform_color([0.5, 0.5, 0.75])

    o3d_pcd = io.p2o_pcd(pt3_pcd)
    o3d_pcd.paint_uniform_color([0.5, 0.75, 0.5])

    o3d_visualize([o3d_mesh, o3d_pcd])
