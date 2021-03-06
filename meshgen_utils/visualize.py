import open3d as o3d


def o3d_visualize(geometries, overlay=False):
    """
    Visualize objects in GUI
    parameters:
        single object, or list of objects
    """

    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.destroy_window()
    except RuntimeError:
        print("Visualization not possible in this enviornment.")
        return

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
            ctr.rotate(3.0, 0.0)
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

    # N: point normal vector
    def show_normal(vis):
        ropt = vis.get_render_option()
        ropt.point_show_normal = not ropt.point_show_normal
        return False
    
    helpstring += '\nN: toggle point normal vector view'
    keybindings[ord('n')] = keybindings[ord('N')] = show_normal

    # W: mesh wireframe
    def show_wireframe(vis):
        ropt = vis.get_render_option()
        ropt.mesh_show_wireframe = not ropt.mesh_show_wireframe
        return False
    
    helpstring += '\nW: toggle mesh wireframe view'
    keybindings[ord('w')] = keybindings[ord('W')] = show_wireframe

    # B: mesh back face
    def show_back_face(vis):
        ropt = vis.get_render_option()
        ropt.mesh_show_back_face = not ropt.mesh_show_back_face
        return False
    
    helpstring += '\nB: toggle mesh back face view'
    keybindings[ord('b')] = keybindings[ord('B')] = show_back_face

    # C: point cloud color option
    def switch_point_color(vis):
        cand, idx = switch_point_color.candidates, switch_point_color.index
        switch_point_color.index = idx = (idx + 1) % len(cand)
        ropt = vis.get_render_option()
        ropt.point_color_option = cand[idx]
        # return True for update geometry
        return True
    pco = o3d.visualization.PointColorOption
    switch_point_color.candidates, switch_point_color.index = (pco.Color, pco.Normal, pco.XCoordinate, pco.YCoordinate, pco.ZCoordinate), 0

    helpstring += '\nC: switch point color options'
    keybindings[ord('c')] = keybindings[ord('C')] = switch_point_color

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

        if len(geometries) >= 3:
            # D: scaled deform vector view
            def show_deform(vis):
                if not show_deform.on:
                    vis.add_geometry(show_deform.geo, reset_bounding_box=False)
                else:
                    vis.remove_geometry(show_deform.geo, reset_bounding_box=False)
                show_deform.on = not show_deform.on
                return False
            show_deform.on = False
            show_deform.geo = (geometries[2] if len(geometries) == 3 else None)

            helpstring += '\nD: toggle mesh deform vector visibility'
        else:
            # blank function for blocking default action
            def show_deform(vis):
                return False
        keybindings[ord('d')] = keybindings[ord('D')] = show_deform

    helpstring += '\nEnjoy!'
    print(helpstring, flush=True)

    return keybindings


def run_visualizer(geometries, overlay, keybindings):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    
    ropt = vis.get_render_option()
    ropt.mesh_show_back_face = True
    ropt.point_color_option = o3d.visualization.PointColorOption.Color

    for idx, geo in enumerate(geometries):
        if overlay and idx == 2:
            continue # don't add deform vector at first
        vis.add_geometry(geo)

    for key, value in keybindings.items():
        vis.register_key_callback(key, value)
    
    vis.run()
    vis.destroy_window()
