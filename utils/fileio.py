import torch
import pytorch3d
import open3d as o3d
import numpy
from pathlib import Path


def o2p_mesh(o3d_mesh, device='cuda:0', dtype=torch.float64):
    """
    Convert open3d type mesh to pytorch3d type mesh
    """
    verts = numpy.asarray(o3d_mesh.vertices)
    verts = torch.Tensor(verts).to(device=device, dtype=dtype)
    faces = numpy.asarray(o3d_mesh.triangles)
    faces = torch.Tensor(faces).to(device=device)

    pt3_mesh = pytorch3d.structures.Meshes([verts], [faces]).to(device=device)

    return pt3_mesh


def p2o_mesh(pt3_mesh):
    """
    Convert pytorch3d type mesh to open3d type mesh
    """
    # batch num?
    verts = o3d.utility.Vector3dVector(pt3_mesh.verts_list()[0])
    faces = o3d.utility.Vector3iVector(pt3_mesh.faces_list()[0])
    o3d_mesh = o3d.geometry.TriangleMesh(verts, faces)

    return o3d_mesh


def p2o_pcd(pt3_pcd):
    """
    Convert pytorch3d type point cloud to open3d type point cloud
    """
    # batch num?
    verts = o3d.utility.Vector3dVector(pt3_pcd.points_list()[0])
    o3d_pcd = o3d.geometry.PointCloud(verts)

    return o3d_pcd


def load_o3d_mesh(filename):
    """
    Reads a mesh from the given filename.
    Returns open3d type mesh
    """
    filename = str(filename)

    if not Path(filename).is_file():
        raise ValueError("No such file %s" % filename)

    o3d_mesh = o3d.io.read_triangle_mesh(filename, print_progress=True)
    
    return o3d_mesh


def load_o3d_pcd(filename):
    """
    Reads a point cloud from the given filename.
    Returns open3d type point cloud
    """
    filename=  str(filename)

    if not Path(filename).is_file():
        raise ValueError("No such file %s" % filename)
    
    o3d_pcd = o3d.io.read_point_cloud(filename, print_progress=True)
    
    return o3d_pcd


def save_o3d_pcd(filename, o3d_pcd):
    """
    Saves point cloud to the file.
    Returns None
    """
    # Using open3d functions for writing the file
    filename = str(filename)

    if not Path(filename).parent.is_dir:
        print("No such directory %s. Creating..." % str(Path(filename).parent))
        Path(filename).parent.mkdir(parents=True)

    o3d.io.write_point_cloud(filename, o3d_pcd)


def save_o3d_mesh(filename, o3d_mesh):
    """
    Saves given mesh to the file.
    Returns None
    """
    filename = str(filename)

    if not Path(filename).parent.is_dir:
        print("No such directory %s. Creating..." % str(Path(filename).parent))
        Path(filename).parent.mkdir(parents=True)
    
    o3d.io.write_triangle_mesh(filename, o3d_mesh)