import torch
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
import open3d as o3d
import numpy
from pathlib import Path
from argparse import ArgumentParser

from .fileio import *


data_path = Path(__file__).absolute().parent.parent / 'data'
sample_points = sample_points_from_meshes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('mesh_file', type=str, help='file containing a mesh to sample')
    parser.add_argument('--output', type=str, default='output.ply', help='filename to save the output point cloud (ply format)')
    parser.add_argument('--points', type=int, help="number of points to sample from the mesh")
    parser.add_argument('--ratio', type=float, help='ratio between the number of points in mesh and number of points to sample')
    args = parser.parse_args()

    o3d_mesh = load_o3d_mesh(args.mesh_file)
    pt3_mesh = o2p_mesh(o3d_mesh)
    num_v = len(pt3_mesh.verts_list()[0])
    print("Loaded mesh with %d vertices and %d faces" % (num_v, len(mesh.faces_list()[0])))

    sam_v = int((args.points if args.points else num_v * (args.ratio if args.ratio else 1.5)))
    points = sample_points(mesh, num_samples=sam_v)
    pt3_pcd = pytorch3d.structures.Pointclouds(points)
    print("Sampled %d points from given mesh" % len(pt3_pcd.points_list()[0]))

    save_pcd(data_path / args.output, pt3_pcd)
    print("Done!")

