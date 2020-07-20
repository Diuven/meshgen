import torch
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
import open3d as o3d
import numpy
from pathlib import Path
from argparse import ArgumentParser

data_path = Path(__file__).absolute().parent.parent / 'data'

def load_mesh(filename, device='cuda:0', dtype=torch.float64):
    """
    Reads a mesh from the given filename.
    Returns pytorch3d TriangularMesh
        verts: Tensor of shape (V, 3), and type dtype(default to float64)
        faces: LongTensor of shape (F, 3)
    """
    # Using open3d functions for reading file, for broader format support
    filename = str(filename)

    if not Path(filename).is_file():
        raise ValueError("No such file %s" % filename)

    o3d_mesh = o3d.io.read_triangle_mesh(filename, print_progress=True)
    
    verts = numpy.asarray(o3d_mesh.vertices)
    verts = torch.Tensor(verts)
    faces = numpy.asarray(o3d_mesh.triangles)
    faces = torch.Tensor(faces)

    pt3_mesh = pytorch3d.structures.Meshes([verts], [faces])

    return pt3_mesh


def save_pcd(filename, pt3_pcd):
    """
    Saves point cloud to the file.
    pcd: pytorch3d pointcloud
    Returns None
    """
    # Using open3d functions for writing the file
    filename = str(filename)

    verts = o3d.utility.Vector3dVector(pt3_pcd.points_list()[0])
    o3d_pcd = o3d.geometry.PointCloud(verts)

    o3d.io.write_point_cloud(filename, o3d_pcd)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('mesh_file', type=str, help='file containing a mesh to sample')
    parser.add_argument('--output', type=str, default='output.ply', help='filename to save the output point cloud (ply format)')
    parser.add_argument('--points', type=int, help="number of points to sample from the mesh")
    parser.add_argument('--ratio', type=float, help='ratio between the number of points in mesh and number of points to sample')
    args = parser.parse_args()

    mesh = load_mesh(args.mesh_file)
    num_v = len(mesh.verts_list()[0])
    print("Loaded mesh with %d vertices and %d faces" % (num_v, len(mesh.faces_list()[0])))

    sam_v = int((args.points if args.points else num_v * (args.ratio if args.ratio else 1.5)))
    points = sample_points_from_meshes(mesh, num_samples=sam_v)
    pt3_pcd = pytorch3d.structures.Pointclouds(points)
    print("Sampled %d points from given mesh" % len(pt3_pcd.points_list()[0]))

    save_pcd(data_path / args.output, pt3_pcd)
    print("Done!")

