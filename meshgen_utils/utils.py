from pathlib import Path
from pytorch3d.ops import SubdivideMeshes
import open3d as o3d
import numpy as np
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


def save_result(save_path, epoch_num, pt3_mesh, pt3_pcd=None, deform=None):
    """
    Saves mesh (and pcd) in save_path 
    """
    save_path = Path(save_path)
    if epoch_num == -1:
        mesh_path = 'mesh_last.ply'
        deform_path = 'deform_last.ply'
    else:
        mesh_path = 'mesh_%04d.ply' % epoch_num
        deform_path = 'deform_%04d.ply' % epoch_num
    mesh_path = save_path / mesh_path
    deform_path = save_path / deform_path

    o3d_mesh = io.p2o_mesh(pt3_mesh)
    io.save_o3d_mesh(mesh_path, o3d_mesh)

    if pt3_pcd is not None:
        o3d_pcd = io.p2o_pcd(pt3_pcd)
        io.save_o3d_pcd(save_path / 'pcd.ply', o3d_pcd)
    
    if deform is not None:
        o3d_deform_pcd = o3d.geometry.PointCloud(o3d_mesh.vertices)
        o3d_deform_pcd.normals = o3d.utility.Vector3dVector(np.asarray(deform.detach().to('cpu')))
        io.save_o3d_pcd(deform_path, o3d_deform_pcd)
    
    print('Results saved in %s' % str(mesh_path))


def load_result(load_path, epoch_num):
    """
    Loads results (saved with function above) and show with visualizer
    """
    load_path = Path(load_path)
    # Unefficient operations (pt3 -> o3d -> pt3), but who cares
    if epoch_num == -1:
        mesh_path = 'mesh_last.ply'
        deform_path = 'deform_last.ply'
    else:
        mesh_path = 'mesh_%04d.ply' % epoch_num
        deform_path = 'deform_%04d.ply' % epoch_num
    mesh_path = load_path / mesh_path
    deform_path = load_path / deform_path

    o3d_mesh = io.load_o3d_mesh(mesh_path)
    pt3_mesh = io.o2p_mesh(o3d_mesh)

    o3d_pcd = io.load_o3d_pcd(load_path / 'pcd.ply')
    pt3_pcd = io.o2p_pcd(o3d_pcd)

    if deform_path.exists() and deform_path.is_file():
        o3d_deform_pcd = io.load_o3d_pcd(deform_path)

        SCALE = 1000
        scaled_deform = SCALE * np.asarray(o3d_deform_pcd.normals)
        deform_mean = np.mean(scaled_deform ** 2)
        print("Squared mean of scaled (x%d) deform vectors: %f" % (SCALE, deform_mean))

        o3d_deform_pcd.normals = o3d.utility.Vector3dVector(scaled_deform)
        pt3_deform_pcd = io.o2p_pcd(o3d_deform_pcd)
    else:
        pt3_deform_pcd = None

    print("Showing results from %s" % str(mesh_path))

    show_overlay(pt3_mesh, pt3_pcd, pt3_deform_pcd)


def show_overlay(pt3_mesh, pt3_pcd, pt3_deform_pcd=None):
    """
    Given pytorc3d type mesh and point cloud, visualize overlay of both.
    Point cloud as green, Mesh as Blue
    """
    o3d_mesh = io.p2o_mesh(pt3_mesh)
    o3d_mesh.paint_uniform_color([0.5, 0.5, 0.75])

    o3d_pcd = io.p2o_pcd(pt3_pcd)
    o3d_pcd.paint_uniform_color([0.5, 0.75, 0.5])

    geometries = [o3d_mesh, o3d_pcd]

    if pt3_deform_pcd is not None:
        o3d_deform_pcd = io.p2o_pcd(pt3_deform_pcd)
        o3d_deform_pcd.paint_uniform_color([0.75, 0.5, 0.5])
        geometries += [o3d_deform_pcd]

    o3d_visualize(geometries, overlay=True)
