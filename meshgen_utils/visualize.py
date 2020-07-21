import open3d as o3d


def o3d_visualize(geometries, overlay=False):
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

    keybindings = get_keybindings(geometries, overlay)
    # Following can't initialize settings
    # o3d.visualization.draw_geometries_with_key_callbacks(geometries, keybindings)
    run_visualizer(geometries, overlay, keybindings)


def get_keybindings(geometries, overlay):
    helpstring = "Visualizer hotkeys: "
    keybindings = {}

    # R: rotation
    def register_rotation(vis):
        def rotate_view(vis):
            ctr = vis.get_view_control()
            ctr.rotate(5.0, 0.0)
            return False
        if register_rotation.on:
            vis.register_animation_callback(lambda vis: False)
        else:
            vis.register_animation_callback(rotate_view)
        register_rotation.on = not register_rotation.on
        return False
    register_rotation.on = False

    helpstring += '\nR: toggle rotation'
    keybindings[ord('r')] = keybindings[ord('R')] = register_rotation

    # V: verbose (show point normal, mesh wire)
    def show_verbose(vis):
        ropt = vis.get_render_option()
        ropt.point_show_normal = not ropt.point_show_normal
        ropt.mesh_show_wireframe = not ropt.mesh_show_wireframe
        return False
    
    helpstring += '\nV: toggle verbose display (point normal vector, mesh wireframe)'
    keybindings[ord('v')] = keybindings[ord('V')] = show_verbose

    # B: mesh back face
    def show_back_face(vis):
        ropt = vis.get_render_option()
        ropt.mesh_show_back_face = not ropt.mesh_show_back_face
        return False
    
    helpstring += '\nB: toggle mesh back face view'
    keybindings[ord('b')] = keybindings[ord('B')] = show_back_face

    if overlay:
        # if this is for mesh / pcd overlay mode (probably from utils.show_overlay)
        # urghh.... duplicate code....

        # M: mesh view
        def show_mesh(vis):
            if not show_mesh.on:
                vis.add_geometry(show_mesh.geo, reset_bounding_box=False)
            else:
                vis.remove_geometry(show_mesh.geo, reset_bounding_box=False)
            show_mesh.on = not show_mesh.on
            return False
        show_mesh.on = True
        show_mesh.geo = geometries[0]
        
        helpstring += '\nM: toggle mesh visibility'
        keybindings[ord('m')] = keybindings[ord('M')] = show_mesh

        # P: point cloud view
        def show_pcd(vis):
            if not show_pcd.on:
                vis.add_geometry(show_pcd.geo, reset_bounding_box=False)
            else:
                vis.remove_geometry(show_pcd.geo, reset_bounding_box=False)
            show_pcd.on = not show_pcd.on
            return False
        show_pcd.on = True
        show_pcd.geo = geometries[1]
        
        helpstring += '\nP: toggle point cloud visibility'
        keybindings[ord('p')] = keybindings[ord('P')] = show_pcd      

    helpstring += '\nEnjoy!'
    print(helpstring, flush=True)

    return keybindings


def run_visualizer(geometries, overlay, keybindings):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    
    vis.get_render_option().mesh_show_back_face = True

    for geo in geometries:
        vis.add_geometry(geo)

    for key, value in keybindings.items():
        vis.register_key_callback(key, value)
    
    vis.run()
    vis.destroy_window()
