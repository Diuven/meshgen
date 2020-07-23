from pathlib import Path
from pytorch3d.ops import SubdivideMeshes
from glob import glob

from . import mesh_ops
from . import fileio as io
from .visualize import o3d_visualize

data_path = Path(__file__).absolute().parent.parent / 'data'

def initial_data(filename, method='poisson', divide_mesh=0, **kargs):
    """
    Reads point cloud from the given filename, and returns initialized mesh and point cloud
    Returns (mesh, pcd):
        mesh: pytorch3d mesh, initialized with poisson method from the original point cloud
        pcd: pytorch3d point cloud, read from the given filename
    """

    filename = str(filename)
    method = method.lower()
    possible_methods = ('alpha', 'convex', 'poisson', 'ball')

    o3d_pcd = io.load_o3d_pcd(data_path / filename)

    if method not in possible_methods:
        raise ValueError('method %s should be one of %s' % (method, possible_methods))
    if method == 'alpha':
        reconstruct = mesh_ops.alpha_mesh
    elif method == 'convex':
        reconstruct = mesh_ops.convex_mesh
    elif method == 'poisson':
        reconstruct = mesh_ops.poisson_mesh
    elif method == 'ball':
        reconstruct = mesh_ops.ball_mesh
    else:
        raise RuntimeError()

    o3d_mesh = reconstruct(o3d_pcd, **kargs)

    pt3_mesh = io.o2p_mesh(o3d_mesh)
    pt3_pcd = io.o2p_pcd(o3d_pcd)

    for i in range(divide_mesh):
        divider = SubdivideMeshes()
        pt3_mesh = divider(pt3_mesh)

    return pt3_mesh, pt3_pcd


# def save_mesh(filename, pt3_mesh):
#     """
#     Saves pytorch3d type mesh into the given filename
#     """
#     filename = str(filename)
#     o3d_mesh = io.p2o_mesh(pt3_mesh)
#     io.save_o3d_mesh(o3d_mesh)


def save_result(save_path, epoch_num, pt3_mesh, pt3_pcd=None):
    """
    Saves mesh (and pcd) in save_path 
    """
    if epoch_num == -1:
        mesh_path = 'mesh_last.ply'
    else:
        mesh_path = ('mesh_%04d.ply' % epoch_num)
    mesh_path = Path(save_path) / mesh_path
    o3d_mesh = io.p2o_mesh(pt3_mesh)
    io.save_o3d_mesh(mesh_path, o3d_mesh)

    if pt3_pcd is not None:
        o3d_pcd = io.p2o_pcd(pt3_pcd)
        io.save_o3d_pcd(Path(save_path) / 'pcd.ply', o3d_pcd)
    
    print('Results saved in %s' % str(mesh_path))


def load_result(load_path, epoch_num):
    """
    Loads results (saved with function above) and show with visualizer
    """
    # Unefficient operations (pt3 -> o3d -> pt3), but who cares
    if epoch_num == -1:
        mesh_path = 'mesh_last.ply'
    else:
        mesh_path = ('mesh_%04d.ply' % epoch_num)
    mesh_path = Path(load_path) / mesh_path
    o3d_mesh = io.load_o3d_mesh(mesh_path)
    pt3_mesh = io.o2p_mesh(o3d_mesh)

    o3d_pcd = io.load_o3d_pcd(Path(load_path) / 'pcd.ply')
    pt3_pcd = io.o2p_pcd(o3d_pcd)

    print("Showing results from %s" % str(mesh_path))

    show_overlay(pt3_mesh, pt3_pcd)


def show_overlay(pt3_mesh, pt3_pcd):
    """
    Given pytorc3d type mesh and point cloud, visualize overlay of both.
    Point cloud as green, Mesh as Blue
    """
    o3d_mesh = io.p2o_mesh(pt3_mesh)
    o3d_mesh.paint_uniform_color([0.5, 0.5, 0.75])

    o3d_pcd = io.p2o_pcd(pt3_pcd)
    o3d_pcd.paint_uniform_color([0.5, 0.75, 0.5])

    o3d_visualize([o3d_mesh, o3d_pcd], overlay=True)
