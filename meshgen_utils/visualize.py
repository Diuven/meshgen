import open3d as o3d

def o3d_visualize(geometries):
    """
    Visualize objects in GUI
    parameters:
        single object, or list of objects
    """
    if isinstance(geometries, list):
        for geo in geometries:
            if isinstance(geo, o3d.geometry.TriangleMesh):
                geo.compute_vertex_normals()
    elif isinstance(geometries, o3d.geometry.TriangleMesh):
        geometries = geometries.compute_vertex_normals()
        geometries = [geometries]
    elif isinstance(geometries, o3d.geometry.PointCloud):
        geometries = [geometries]
    else:
        raise TypeError("Object not visualizable: %s" % geometries)

    o3d.visualization.draw_geometries(geometries)

